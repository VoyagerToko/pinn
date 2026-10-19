"""Background threads. The window never waits on the network or the GPU.

ParticleEngine  advects the pigment with the network's predicted velocity (midpoint rule, two queries per step)
                and rasterises it for the 2D bath; for the 3D tank it hands over positions.
FieldWorker     fetches the latest requested field (latest request wins).
AuxWorker       curves, sensor points and live centrelines for the rack; curves are cached on disk.
Warmer          compiles every scene once at start-up so switching scenes during a talk never stalls.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from viewer import scenes as S
from viewer import theme
from viewer.client import ROOT, Client

CACHE = ROOT / "runs" / "_viewer_cache"


def scene_colors(spec: S.SceneSpec) -> np.ndarray:
    """Pigment table for a scene: 0 its own pigment, 1 its partner, 2.. the drop pigments in drop order.

    In the 2D baths the partner is ink. In the 3D tank, where points are seen through each other, the partner is
    the scene's pigment let down with paper, so the tank reads as one pigment in two strengths rather than black.
    """
    own = theme.hex_rgb(spec.pigment)
    partner = theme.hex_rgb(theme.INK) if spec.dims == 2 else own * 0.45 + theme.hex_rgb(theme.PAPER) * 0.55
    rest = [theme.hex_rgb(theme.PIGMENTS[n]) for n in theme.DROP_ORDER]
    return np.stack([own, partner] + rest).astype(np.float32)


class PigmentRaster:
    """Pigment as an ARGB dye texture.

    Each step splats the particles' recent trails (bincount), adds them into a texture that keeps a decaying memory of
    where pigment has been, softens it at two scales (a fine line and a wider wash, so thinly seeded water still reads
    as dye rather than dots) and composites the pigments like ink on size.
    """

    K = 10
    DECAY = 0.90   # per step: the texture remembers the last ~10 steps, so trails lie down as continuous bands
    GAIN = 2.6
    WASH = 1.1     # weight of the wide blur

    def __init__(self):
        self.W = self.H = 0
        self.acc: Optional[np.ndarray] = None

    def reset(self):
        self.acc = None

    def render(self, hist: np.ndarray, active: np.ndarray, pig: np.ndarray, domain, colors: np.ndarray, periodic: bool) -> Optional[np.ndarray]:
        W, H = self.W, self.H
        if W < 8 or H < 8:
            return None
        x0, x1, y0, y1 = domain[:4]
        sx, sy = W / (x1 - x0), H / (y1 - y0)
        P = colors.shape[0]
        K = hist.shape[0]
        span = np.array([x1 - x0, y1 - y0], np.float32)
        a, b = hist[:-1], hist[1:]
        ok = (np.abs(a - b) < 0.25 * span).all(-1) & active[None, :]
        s = np.array([0.0, 0.34, 0.67], np.float32)[:, None, None, None]
        pts = a[None] * (1 - s) + b[None] * s  # (3, K-1, N, 2)
        age = np.arange(K - 1, dtype=np.float32)
        w = (0.12 + 0.88 * (1 - age / (K - 1)) ** 1.6)[None, :, None] * np.ones((3, 1, 1), np.float32)
        w = np.broadcast_to(w[..., None], (3, K - 1, hist.shape[1], 1))[..., 0]
        okb = np.broadcast_to(ok[None], (3, K - 1, hist.shape[1]))
        pg = np.broadcast_to(pig[None, None, :], (3, K - 1, hist.shape[1]))
        px = ((pts[..., 0] - x0) * sx).astype(np.int32)
        py = ((y1 - pts[..., 1]) * sy).astype(np.int32)
        inside = okb & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        flat = (pg[inside].astype(np.int64) * (W * H) + py[inside] * W + px[inside])
        dens = np.bincount(flat, weights=w[inside].astype(np.float64) * 0.55, minlength=P * W * H)
        # heads: a stronger dab
        hx = ((hist[0, :, 0] - x0) * sx).astype(np.int32)
        hy = ((y1 - hist[0, :, 1]) * sy).astype(np.int32)
        hin = active & (hx >= 0) & (hx < W) & (hy >= 0) & (hy < H)
        dens += np.bincount(pig[hin].astype(np.int64) * (W * H) + hy[hin] * W + hx[hin], weights=np.full(int(hin.sum()), 1.1), minlength=P * W * H)
        d = dens.reshape(P, H, W).astype(np.float32)
        if self.acc is None or self.acc.shape != d.shape:
            self.acc = d
        else:
            self.acc *= self.DECAY
            self.acc += (1.0 - self.DECAY) * d
            self.acc[self.acc < 1e-4] = 0.0
        d = self.acc
        used = [p for p in range(P) if d[p].any()]
        if not used:
            return np.zeros((H, W, 4), np.uint8)
        from scipy.ndimage import gaussian_filter

        mode = "wrap" if periodic else "constant"
        d = np.stack([gaussian_filter(d[p], 0.7, mode=mode) + self.WASH * gaussian_filter(d[p], 2.0, mode=mode) for p in used])
        alpha = 1.0 - np.exp(-self.GAIN * d)                 # (U, H, W)
        A = 1.0 - np.prod(1.0 - alpha, axis=0)               # coverage
        cs = colors[used]                                     # (U, 3)
        num = np.einsum("uhw,uc->hwc", alpha, cs)
        den = np.maximum(alpha.sum(0), 1e-6)[..., None]
        rgb = num / den * A[..., None]                        # premultiplied
        out = np.empty((H, W, 4), np.uint8)
        out[..., 0] = np.clip(rgb[..., 2], 0, 255)
        out[..., 1] = np.clip(rgb[..., 1], 0, 255)
        out[..., 2] = np.clip(rgb[..., 0], 0, 255)
        out[..., 3] = np.clip(A * 255, 0, 255)
        return out


class ParticleEngine(QThread):
    stepped = Signal(object)
    status = Signal(str, str)  # sid, message

    def __init__(self, base: str):
        super().__init__()
        self.base = base
        self.cmds: "queue.Queue" = queue.Queue()
        self.raster = PigmentRaster()
        self._stop = False
        self.rng = np.random.default_rng(7)

    # ---- commands from the window (thread-safe) ----
    def set_scene(self, spec: S.SceneSpec, domain: List[float], t: float, Re: float, trange: Optional[List[float]] = None):
        self.cmds.put(("scene", spec, domain, t, Re, trange))

    def set_idle(self):
        """Stop advecting: the current scene has no trained network."""
        self.cmds.put(("idle",))

    def set_time(self, t: float):
        self.cmds.put(("time", t))

    def set_Re(self, Re: float):
        self.cmds.put(("Re", Re))

    def set_playing(self, on: bool):
        self.cmds.put(("play", on))

    def drop(self, pts: np.ndarray, colour_index: int):
        self.cmds.put(("drop", np.asarray(pts, np.float32), colour_index))

    def reset(self):
        self.cmds.put(("reset",))

    def set_raster_size(self, W: int, H: int):
        self.cmds.put(("size", W, H))

    def stop(self):
        self._stop = True

    # ---- the loop ----
    def run(self):
        client = Client(self.base)
        spec = None
        domain = None
        t = Re = 0.0
        trange = None
        playing = False
        pos = pig = active = hist = None
        ring = 0
        last = time.perf_counter()
        compiled = set()
        ema_hz, ema_ms = 0.0, 0.0
        colors = None
        while not self._stop:
            try:
                while True:
                    c = self.cmds.get_nowait()
                    k = c[0]
                    if k == "scene":
                        spec, domain, t, Re, trange = c[1], list(c[2]), float(c[3]), float(c[4]), c[5]
                        pos, pig, active = S.initial_dye(spec, domain, self.rng)
                        self.raster.reset()
                        K = PigmentRaster.K if spec.dims == 2 else 6
                        hist = np.repeat(pos[None], K, axis=0)
                        colors = scene_colors(spec)
                        ring = 0
                        playing = False
                    elif k == "idle":
                        spec = None
                    elif k == "time":
                        t = float(c[1])
                    elif k == "Re":
                        Re = float(c[1])
                    elif k == "play":
                        playing = bool(c[1])
                    elif k == "reset" and spec is not None:
                        pos, pig, active = S.initial_dye(spec, domain, self.rng)
                        self.raster.reset()
                        hist = np.repeat(pos[None], hist.shape[0], axis=0)
                        ring = 0
                    elif k == "drop" and spec is not None:
                        pts, ci = c[1], int(c[2])
                        m = min(len(pts), S.CAPACITY // 3)
                        idx = (ring + np.arange(m)) % S.CAPACITY
                        ring = int((ring + m) % S.CAPACITY)
                        pos[idx] = pts[:m]
                        pig[idx] = ci
                        active[idx] = True
                        hist[:, idx] = pts[:m][None]
                    elif k == "size":
                        self.raster.W, self.raster.H = int(c[1]), int(c[2])
                        self.raster.reset()
            except queue.Empty:
                pass
            if spec is None:
                time.sleep(0.03)
                continue
            now = time.perf_counter()
            wall = min(now - last, 0.08)
            last = now
            lo = np.asarray(domain[0::2], np.float32)
            hi = np.asarray(domain[1::2], np.float32)
            t_next = t
            dt = spec.dye_rate * wall
            if playing and spec.play_rate > 0:
                t_next = t + spec.play_rate * wall
                if spec.coupled:
                    dt = t_next - t
            try:
                if spec.sid not in compiled:
                    self.status.emit(spec.sid, "compiling")
                t0 = time.perf_counter()
                v1, c1 = client.velocity(spec.sid, pos, t, Re)
                v1 = np.nan_to_num(v1)
                mid = pos + 0.5 * dt * v1
                if spec.periodic:
                    mid = lo + np.mod(mid - lo, hi - lo)
                v2, c2 = client.velocity(spec.sid, mid, t + (0.5 * dt if (playing and spec.coupled) else 0.0), Re)
                v2 = np.nan_to_num(v2)
                step_ms = (time.perf_counter() - t0) * 1e3
                if spec.sid not in compiled:
                    compiled.add(spec.sid)
                    self.status.emit(spec.sid, "live")
            except Exception as e:  # server gone or restarting: keep the window alive
                self.status.emit(spec.sid if spec else "", f"error: {type(e).__name__}")
                time.sleep(1.0)
                last = time.perf_counter()
                continue
            new = pos + dt * v2
            # a particle that left the fluid (the network's extrapolation beyond a wall) retires and stops moving
            bad = ~np.isfinite(new).all(-1) | (np.abs(new - 0.5 * (lo + hi)) > 2 * (hi - lo)).any(-1)
            active &= ~bad
            new = np.where(active[:, None], new, np.clip(np.nan_to_num(pos), lo, hi))
            if spec.periodic:
                new = lo + np.mod(new - lo, hi - lo)
            elif spec.emitter is not None:
                out = ((new < lo) | (new > hi)).any(-1)
                active &= ~out
            else:
                new = np.clip(new, lo + 1e-4, hi - 1e-4)
            pos = new.astype(np.float32)
            hist = np.roll(hist, 1, axis=0)
            hist[0] = pos
            if spec.emitter is not None:
                ep, eg = S.rake(spec, domain, 0, self.rng)
                m = len(ep)
                idx = (ring + np.arange(m)) % S.CAPACITY
                ring = int((ring + m) % S.CAPACITY)
                pos[idx], pig[idx], active[idx] = ep, eg, True
                hist[:, idx] = ep[None]
            if playing and spec.play_rate > 0:
                t = t_next
                if trange is not None and t > trange[1]:
                    t = trange[0]
                    pos, pig, active = S.initial_dye(spec, domain, self.rng)
                    self.raster.reset()
                    hist = np.repeat(pos[None], hist.shape[0], axis=0)
            ema_ms = 0.85 * ema_ms + 0.15 * (c1 + c2) * 1e3 / 2 if ema_ms else (c1 + c2) * 1e3 / 2
            hz = 1.0 / max(wall, 1e-3)
            ema_hz = 0.9 * ema_hz + 0.1 * hz if ema_hz else hz
            payload = {"sid": spec.sid, "t": t, "Re": Re, "vel_ms": ema_ms, "hz": ema_hz, "step_ms": step_ms, "playing": playing}
            if spec.dims == 2:
                payload["image"] = self.raster.render(hist, active, pig, domain, colors, spec.periodic)
            else:
                payload.update(hist=hist.copy(), active=active.copy(), pig=pig.copy(), colors=colors)
            self.stepped.emit(payload)
            spare = 1 / 45 - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
        client.close()


class FieldWorker(QThread):
    ready = Signal(object)
    status = Signal(str, str)

    def __init__(self, base: str):
        super().__init__()
        self.base = base
        self._want = None
        self._ev = threading.Event()
        self._stop = False
        self._seen = set()

    def request(self, sid: str, t: float, Re: float, n: int):
        self._want = (sid, float(t), float(Re), int(n))
        self._ev.set()

    def stop(self):
        self._stop = True
        self._ev.set()

    def run(self):
        client = Client(self.base)
        done = None
        while not self._stop:
            self._ev.wait(0.5)
            self._ev.clear()
            want = self._want
            if want is None or want == done:
                continue
            sid, t, Re, n = want
            try:
                if (sid, n) not in self._seen:
                    self.status.emit(sid, "compiling field")
                t0 = time.perf_counter()
                hdr, f, comp = client.field(sid, t, Re, n)
                ms = (time.perf_counter() - t0) * 1e3
                self._seen.add((sid, n))
                done = want
                self.ready.emit({"sid": sid, "t": t, "Re": Re, "n": n, "hdr": hdr, "fields": f, "ms": ms, "compute_ms": comp * 1e3})
            except Exception as e:
                self.status.emit(sid, f"field error: {type(e).__name__}")
                time.sleep(1.0)
        client.close()


class AuxWorker(QThread):
    """Rack data: cached curves, sensor points, and centrelines computed live through /velocity."""

    curve = Signal(str, object)
    points = Signal(str, object)
    centreline = Signal(str, float, object)

    def __init__(self, base: str, runs_by_sid: Dict[str, str]):
        super().__init__()
        self.base = base
        self.runs = runs_by_sid
        self.q: "queue.Queue" = queue.Queue()
        self._stop = False
        self._latest_line = None

    def want_curve(self, sid):
        self.q.put(("curve", sid))

    def want_points(self, sid):
        self.q.put(("points", sid))

    def want_centreline(self, sid, Re):
        self._latest_line = (sid, float(Re))
        self.q.put(("line",))

    def stop(self):
        self._stop = True
        self.q.put(("stop",))

    def run(self):
        client = Client(self.base)
        CACHE.mkdir(parents=True, exist_ok=True)
        ys = np.linspace(0.0, 1.0, 201, dtype=np.float32)
        while not self._stop:
            try:
                job = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if job[0] == "curve":
                    sid = job[1]
                    path = CACHE / f"{sid}__{self.runs.get(sid, 'run')}__curve.json"
                    if path.exists():
                        data = json.loads(path.read_text())
                    else:
                        data = client.curve(sid)
                        path.write_text(json.dumps(data))
                    self.curve.emit(sid, data)
                elif job[0] == "points":
                    self.points.emit(job[1], client.points(job[1]))
                elif job[0] == "line" and self._latest_line is not None:
                    sid, Re = self._latest_line
                    self._latest_line = None
                    pts = np.concatenate([np.stack([np.full_like(ys, 0.5), ys], -1), np.stack([ys, np.full_like(ys, 0.5)], -1)])
                    v, _ = client.velocity(sid, pts, 0.0, Re)
                    self.centreline.emit(sid, Re, {"s": ys, "u": v[:201, 0], "v": v[201:, 1]})
            except Exception:
                time.sleep(0.5)
        client.close()


class Warmer(QThread):
    """Compile every scene once (field at its grid size, velocity at the particle capacity)."""

    progress = Signal(str, str)

    def __init__(self, base: str, models: Dict[str, Dict], first: str):
        super().__init__()
        self.base, self.models, self.first = base, models, first
        self.runs = {sid: (f"{m.get('run', 'run')}-{m['run_stamp']}" if m.get("run_stamp") else m.get("run", "run")) for sid, m in models.items()}

    def run(self):
        client = Client(self.base)
        order = [self.first] + [s.sid for s in S.SCENES if s.sid != self.first]
        for sid in order:
            if sid not in self.models:
                continue
            spec = S.BY_ID[sid]
            ctl = self.models[sid].get("controls", {})
            t = float(ctl.get("t", [0.0])[0])
            Re = float(ctl["Re"][-1]) if "Re" in ctl else 1000.0
            self.progress.emit(sid, "warming")
            try:
                client.field(sid, t, Re, spec.field_n)
                if not self.models[sid].get("trained", True):
                    self.progress.emit(sid, "ready")
                    continue
                client.velocity(sid, np.zeros((S.CAPACITY, spec.dims), np.float32) + 0.5, t, Re)
                if spec.chart in ("centreline", "centreline_live"):
                    client.velocity(sid, np.full((402, 2), 0.5, np.float32), 0.0, Re)
                self.progress.emit(sid, "ready")
            except Exception as e:
                self.progress.emit(sid, f"error: {type(e).__name__}")
        # then the rack's curves, the slowest first-time work (the forward cylinder takes ~2 min), into the disk cache
        for sid in order:
            spec = S.BY_ID.get(sid)
            if sid not in self.models or spec is None or spec.chart in ("none", "centreline_live"):
                continue
            path = CACHE / f"{sid}__{self.runs.get(sid, 'run')}__curve.json"
            if path.exists():
                continue
            try:
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(client.curve(sid)))
            except Exception:
                pass  # the rack fetches it again on demand; a missing curve is not a broken scene
        client.close()
