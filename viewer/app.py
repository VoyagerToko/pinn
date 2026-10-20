"""The window. Run: python -m viewer   (options: --server URL, --scene ID, --shot PNG for a screenshot and exit)."""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from viewer import render, theme
from viewer import scenes as S
from viewer.bath import Bath2D
from viewer.client import Client, find_server, start_wsl_server, wsl_path, ROOT
from viewer.controls import PlayButton, ReBar, SceneRail, TabStrip, TimeBar, VerdictStamp, ViewChips
from viewer.panel import BathPanel, Body
from viewer.rack import Rack
from viewer.workers import AuxWorker, FieldWorker, ParticleEngine, Warmer

CAPTIONS = {
    "tgv2d": "The error stays below the gate across the whole time window. The exact solution is known in closed form, so this scene checks the physics itself.",
    "cavity_ok": "The network's velocity along the vertical centreline against Ghia's 1982 data. Against a converged finite-difference solution of the same problem the field differs by 0.27 %.",
    "cavity_hard": "Open the divergence view. With the lid speed and the walls imposed exactly, the two top corners cannot conserve mass, and the whole vortex comes out too weak.",
    "inverse": "The network saw only velocity at scattered points. The pressure field and both coefficients of the equations were never measured; it recovered them.",
    "cylinder_fwd": "Every bar should reach the line: the mass that enters must leave. At the reduced training budget a quarter of it is lost downstream, and the wake never sheds.",
    "tgv3d": "The reference peaks near t = 9, where the flow turns turbulent. The network's dissipation is largest at the start and never peaks: it keeps the flow smooth.",
    "deeponet": "Drag the lid speed. Every position is one forward pass of one network. The hatched band was never sampled in training; Re 400 sits inside it.",
    "stenosis": "No scan measures pressure. The network infers it from the voxel velocities and the equations of motion. The dashed line is the simplified Bernoulli rule used in clinics today, applied to the same scan.",
}
CAPTIONS["stenosis_fast"] = CAPTIONS["stenosis"]
CAPTIONS["coa"] = ("Real scanner data, real catheter. The bars are every phantom case: the catheter's peak pressure drop next to "
                   "the conventional 4D-flow estimate, the clinical Bernoulli rule, and the two networks, all from the same scan.")
CAPTION_UNTRAINED = {
    "coa": "The networks are not trained yet (they train on the desktop and arrive here by themselves). The tank shows the scan's "
           "measured speed in the real phantom; the bars compare the catheter with the conventional estimates.",
    "stenosis": "The network is not trained yet. The bath shows the scan it will learn from; the sheet shows the CFD truth it will be scored against. Train it with: python scripts/train.py --benchmark stenosis_mri, then restart the server.",
}
TAB_NOTE = {
    "real": "A synthetic study, not a clinical tool: a simulated artery, a simulated scan of it, and a network that infers the blood pressure the scan cannot measure.",
}
SHEET = {
    "tgv2d": ("Reference: exact solution", "same instant, same colour scale"),
    "cavity_ok": ("Reference: finite differences", "converged solution of the same lid, same colour scale"),
    "cavity_hard": ("Reference: finite differences", "converged solution of the same lid, same colour scale"),
    "inverse": ("Reference: Nektar data", ""),
    "cylinder_fwd": ("Reference", ""),
    "tgv3d": ("Reference", ""),
    "deeponet": ("Reference: finite differences", ""),
    "stenosis": ("Reference: CFD of the same vessel", "same instant, same colour scale"),
    "stenosis_fast": ("Reference: CFD of the same vessel", "same instant, same colour scale"),
    "coa": ("Reference: catheter pressure", ""),
}
EMPTY = {
    "coa": "The reference is a catheter: pressure measured upstream and downstream of the coarctation during the scan session, averaged over 30+ beats. The chart compares every method with it.",
    "cylinder_fwd": "FEATFLOW publishes drag, lift and pressure-probe series for this benchmark, not a field. The chart below carries the evidence.",
    "tgv3d": "The reference for the 3D vortex is a DNS dissipation curve (DeBonis 2013), not a field. It is drawn below.",
}


class Connector(QThread):
    found = Signal(str, object)
    failed = Signal()

    def __init__(self, url: Optional[str]):
        super().__init__()
        self.url = url

    def run(self):
        res = find_server([self.url] if self.url else None)
        if not res:
            self.failed.emit()
            return
        url, health = res
        try:
            c = Client(url)
            models = c.models()
            c.close()
        except Exception:
            self.failed.emit()
            return
        self.found.emit(url, {"health": health, "models": models})


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("pinnflow: the trained flows, live")
        self.resize(*args.size)
        self.setMinimumSize(1240, 760)
        ground = QWidget(objectName="ground")
        self.setCentralWidget(ground)
        outer = QVBoxLayout(ground)
        outer.setContentsMargins(28, 14, 28, 14)
        outer.setSpacing(12)

        self.tabs = TabStrip()
        outer.addWidget(self.tabs)
        self.rail = SceneRail()
        outer.addWidget(self.rail)
        self.tab = "bench"

        self.stamp = VerdictStamp()
        self.stage = QStackedWidget()
        self.bath = Bath2D()
        self.stage.addWidget(self.bath)
        self.tank = None
        self.panel = BathPanel(self.stamp, self.stage)
        self.title, self.summary, self.mode_line = self.panel.head.title, self.panel.head.summary, self.panel.mode_line
        self.rack = Rack()
        self.body = Body(self.panel, self.rack)
        outer.addWidget(self.body, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(20)
        self.play = PlayButton()
        self.timebar = TimeBar()
        self.rebar = ReBar()
        self.controls = QStackedWidget()
        self.controls.addWidget(self.timebar)
        self.controls.addWidget(self.rebar)
        self.controls.setFixedHeight(58)
        self.chips = ViewChips()
        self.live = QLabel("")
        self.live.setFont(theme.font("text", theme.SMALL, 400, tabular=True))
        self.live.setStyleSheet(f"color: {theme.HAIR};")
        self.live.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.hint = QLabel("click drops pigment, drag combs it, space plays, 1 to 7 pick a scene")
        self.hint.setFont(theme.font("text", theme.SMALL, 400))
        self.hint.setStyleSheet(f"color: {theme.INK_SOFT};")
        self.hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bottom.addWidget(self.play, 0, Qt.AlignVCenter)
        bottom.addWidget(self.controls, 3)
        bottom.addWidget(self.chips, 0, Qt.AlignVCenter)
        tail = QVBoxLayout()
        tail.setSpacing(0)
        tail.addWidget(self.live)
        tail.addWidget(self.hint)
        bottom.addLayout(tail, 2)
        outer.addLayout(bottom)

        # state
        self.url = None
        self.device = ""
        self.models: Dict[str, Dict] = {}
        self.spec: Optional[S.SceneSpec] = None
        self.view = "pigment"
        self.t = 0.0
        self.Re = 1000.0
        self.playing = False
        self.last_field: Dict[str, Dict] = {}
        self.scales: Dict = {}
        self.curves: Dict[str, Dict] = {}
        self.points: Dict[str, np.ndarray] = {}
        self.lines: Dict = {}
        self.geom: Dict[str, Dict] = {}
        self.field_ms = 0.0
        self.drop_i = 0
        self._cursor_at = 0.0
        self.engine = self.fieldw = self.aux = self.warmer = None
        self._retry = 0

        self.rail.chosen.connect(self.select_scene)
        self.tabs.chosen.connect(self.set_tab)
        self.tabs.set_active("bench", animate=False)
        self.chips.chosen.connect(self.set_view)
        self.play.toggled.connect(self.set_playing)
        self.timebar.scrubbed.connect(self.scrub)
        self.rebar.changed.connect(self.set_Re)
        self.bath.dropped.connect(self.on_drop)
        # Left and Right are not window shortcuts: a focused rail, chip row or bar uses them first (keyPressEvent below)
        for key, fn in (("Space", self.toggle_play), ("R", self.reset_dye), ("V", self.next_view), ("F", self.toggle_full), ("Escape", self.leave_full),
                        ("S", self.start_server), ("D", self.drop_random), ("T", self.toggle_tab)):
            QShortcut(QKeySequence(key), self, activated=fn)
        for i in range(7):  # 1 to 7 pick a scene within the current tab
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda i=i: self.select_index(i))

        self.bath.state = "connecting"
        self.bath.state_text = "Looking for the network server"
        self.connect_server()

    # ------------------------------------------------------------------ server
    def connect_server(self):
        self.connector = Connector(self.args.server)
        self.connector.found.connect(self.on_connected)
        self.connector.failed.connect(self.on_offline)
        self.connector.start()

    def on_offline(self):
        self.bath.state = "offline"
        cmd = f"wsl -d Ubuntu -- bash {wsl_path(ROOT / 'scripts' / 'serve_local_wsl.sh')}"
        self.bath.state_text = "No network server is answering.\nPress S to start it in WSL, or run:\n" + cmd
        self.title.setText("Waiting for the server")
        self.bath.update()
        if self._retry:
            self._retry -= 1
            QTimer.singleShot(3000, self.connect_server)

    def start_server(self):
        if self.url:
            return
        if start_wsl_server():
            self.bath.state = "connecting"
            self.bath.state_text = "Starting the network server in WSL\n(the first start loads seven models, about 30 s)"
            self.bath.update()
            self._retry = 40
            QTimer.singleShot(4000, self.connect_server)

    def on_connected(self, url: str, info: Dict):
        self.url = url
        self.device = info["health"].get("device", "")
        self.models = {m["id"]: m for m in info["models"]}
        self.rail.available = set(self.models)
        for sid, m in self.models.items():
            _apply_server_verdict(S.BY_ID.get(sid), m)
        self.engine = ParticleEngine(url)
        self.engine.stepped.connect(self.on_step)
        self.engine.status.connect(self.on_status)
        self.bath.raster_size.connect(self.engine.set_raster_size)
        self.engine.start()
        r = self.bath.domain_rect()
        self.engine.set_raster_size(max(8, int(r.width() / 2)), max(8, int(r.height() / 2)))
        self.fieldw = FieldWorker(url)
        self.fieldw.ready.connect(self.on_field)
        self.fieldw.status.connect(self.on_status)
        self.fieldw.start()
        self.aux = AuxWorker(url, {sid: _cache_key(m) for sid, m in self.models.items()})
        self.aux.curve.connect(self.on_curve)
        self.aux.points.connect(self.on_points)
        self.aux.centreline.connect(self.on_line)
        self.aux.start()
        first = self.args.scene if self.args.scene in self.models else next(s.sid for s in S.SCENES if s.sid in self.models and s.tab == "bench")
        self.warmer = Warmer(url, self.models, first)
        self.warmer.progress.connect(self.on_warm)
        self.warmer.start()
        self.select_scene(first)
        if self.args.view:
            self.set_view(self.args.view)
        if self.args.Re:
            self.set_Re(self.args.Re)
        if self.args.play:
            self.set_playing(True)
        if self.args.shot:
            QTimer.singleShot(int(self.args.wait * 1000), self.take_shot)

    # ------------------------------------------------------------------ scenes
    def select_scene(self, sid: str):
        if sid not in self.models or self.engine is None:
            return
        spec = S.BY_ID[sid]
        m = self.models[sid]
        if spec.tab != self.tab:
            self.set_tab(spec.tab, select=False)
        self.spec = spec
        ctl = m.get("controls", {})
        domain = [float(v) for v in m["domain"]]
        self.rail.set_active(sid)
        self.title.setText(spec.name)
        self.summary.setText(m.get("summary", ""))
        source = "measured, STATUS.md" if spec.tab == "bench" else ("measured, final_eval.json" if spec.trained else "awaiting training")
        self.stamp.set(spec.verdict, spec.headline, spec.headline_label, spec.pigment, source)
        self.panel.set_live("")
        views = spec.views if spec.trained else ["mri"]
        self.view = views[0]
        self.chips.configure(views, self.view, spec.pigment, spec.evidence if spec.trained else None)
        self.set_playing(False)
        self.play.setVisible(spec.play_rate > 0 and spec.trained)
        if "t" in ctl:
            t0, t1 = ctl["t"]
            self.t = float(t1) if spec.start_at_end else float(t0)
            phys = m.get("physical") or {}
            if "period_s" in phys:  # one heartbeat, shown in seconds
                self.timebar.set_range(t0, t1, "s", spec.pigment, scale=phys["period_s"] / (t1 - t0))
            else:
                self.timebar.set_range(t0, t1, "s" if sid == "cylinder_fwd" else "", spec.pigment)
            self.timebar.set_t(self.t)
            self.controls.setCurrentWidget(self.timebar)
        else:
            self.t = 0.0
        if spec.re_mode == "discrete":
            vals = sorted(float(v) for v in ctl.get("Re", [100, 400, 1000]))
            self.Re = vals[-1]
            self.rebar.configure("discrete", values=vals, pigment=spec.pigment, Re=self.Re)
            self.controls.setCurrentWidget(self.rebar)
        elif spec.re_mode == "continuous":
            lo, hi = ctl.get("Re", [100.0, 1000.0])
            self.Re = 400.0
            self.rebar.configure("continuous", lo=lo, hi=hi, holdout=ctl.get("holdout"), pigment=spec.pigment, Re=self.Re)
            self.controls.setCurrentWidget(self.rebar)
        else:
            self.Re = 1000.0
        if spec.dims == 3:
            if self.tank is None:
                from viewer.tank import Tank3D

                self.tank = Tank3D()
                self.stage.addWidget(self.tank)
            self.stage.setCurrentWidget(self.tank)
            if sid == "coa":
                self.body.set_aspect(0.95)  # an upright aorta
                if sid in self.geom:
                    self.tank.configure_vessel(self.geom[sid])
                else:
                    self.aux.want_points(sid)
            else:
                self.tank.configure_box()
                self.body.set_aspect(1.15)  # the tank's view of the cube
        else:
            self.stage.setCurrentWidget(self.bath)
            self.bath.domain = domain[:4]
            self.bath.callouts = []
            self.body.set_aspect((domain[1] - domain[0]) / (domain[3] - domain[2]))
            self.bath.field_img = None
            self.bath.pigment_img = None
            self.bath.cylinder = None
            self.bath.sensors = None
            self.bath.legend = None
            walls = m.get("walls")
            self.bath.walls = (np.asarray(walls["x"]), np.asarray(walls["h"])) if walls else None
            ready = self.rail.ready.get(sid) == "ready"
            if not spec.trained:
                self.bath.state = "untrained"
                self.bath.state_text = ""
            else:
                self.bath.state = "live" if ready else "compiling"
                self.bath.state_text = "" if ready else "Compiling this scene's network on the GPU\n(first time only, about 25 s; the pigment moves when it is ready)"
            r = self.bath.domain_rect()
            self.engine.set_raster_size(max(8, int(r.width() * self.devicePixelRatioF() / 2)), max(8, int(r.height() * self.devicePixelRatioF() / 2)))
            self.bath.update()
        if spec.trained:
            self.engine.set_scene(spec, domain, self.t, self.Re, ctl.get("t"))
        else:
            self.engine.set_idle()  # no network to carry the pigment yet
        self.request_field()
        title, note = SHEET[sid]
        self.rack.sheet.title, self.rack.sheet.note = title, note
        self.rack.sheet.domain = domain[:4] if spec.dims == 2 else [0, 1, 0, 1]
        self.rack.sheet.image = None
        self.rack.sheet.cylinder = None
        self.rack.sheet.empty = EMPTY.get(sid, "Waiting for the first frame")
        self.rack.set_compact(sid in EMPTY)  # no reference field: a short note, the chart takes the room
        self.rack.sheet.update()
        self.rack.caption.setText(CAPTIONS.get(sid, "") if spec.trained else CAPTION_UNTRAINED.get(sid, ""))
        self.rack.chart.clear("Loading the evidence for this scene")
        self.rack.show_coefficients(False)
        if spec.chart in ("centreline_live",):
            self.aux.want_curve(sid)
            self.aux.want_centreline(sid, self.Re)
        elif spec.chart != "none":
            self.aux.want_curve(sid)
        if sid == "inverse" and sid not in self.points:
            self.aux.want_points(sid)
        self.update_mode_line()

    def set_tab(self, tab: str, select: bool = True):
        self.tab = tab
        self.tabs.set_active(tab)
        self.rail.set_scenes(S.scenes_in(tab), TAB_NOTE.get(tab, ""))
        if self.spec is not None:
            self.rail.set_active(self.spec.sid, animate=False)
        if select:
            first = next((s.sid for s in S.scenes_in(tab) if s.sid in self.models), None)
            if first:
                self.select_scene(first)

    def toggle_tab(self):
        self.set_tab("real" if self.tab == "bench" else "bench")

    def select_index(self, i: int):
        scenes = S.scenes_in(self.tab)
        if i < len(scenes):
            self.select_scene(scenes[i].sid)

    def request_field(self):
        if self.spec and self.fieldw:
            self.fieldw.request(self.spec.sid, self.t, self.Re, self.spec.field_n)

    def update_mode_line(self):
        s = self.spec
        if s is None:
            return
        k, unit = self.timebar.scale, (" " + self.timebar.unit) if self.timebar.unit else ""
        tt = f"{self.t * k:.2f}{unit}"
        if not s.trained:
            txt = f"The scan at t = {tt}; drag the time bar through the beat."
        elif s.play_rate == 0:
            txt = f"Steady flow, Re {self.Re:.0f}: each update asks the network for every particle's velocity."
        elif s.coupled:
            txt = ("Playing: flow time and pigment advance together; the trails are pathlines." if self.playing
                   else f"Paused at t = {tt}; the pigment follows this instant. Space runs time.")
        else:
            txt = f"Pigment follows the flow at t = {tt}; play or drag the time bar."
        self.mode_line.setText(txt)

    # ------------------------------------------------------------------ controls
    def _views(self):
        """The views this scene can show now: an untrained scene has only its scan."""
        return self.spec.views if self.spec.trained else ["mri"]

    def set_view(self, view: str):
        if self.spec is None or view not in self._views():
            return
        self.view = view
        self.chips.configure(self._views(), view, self.spec.pigment, self.spec.evidence if self.spec.trained else None)
        self.render_field()

    def next_view(self):
        if self.spec:
            v = self._views()
            self.set_view(v[(v.index(self.view) + 1) % len(v)] if self.view in v else v[0])

    def set_playing(self, on: bool):
        if self.spec is None or self.spec.play_rate == 0 or not self.spec.trained:
            on = False
        self.playing = on
        self.play.set_playing(on)
        if self.engine:
            self.engine.set_playing(on)
        self.update_mode_line()

    def toggle_play(self):
        self.set_playing(not self.playing)

    def scrub(self, t: float):
        self.t = t
        self.timebar.set_t(t, animate=True)
        if self.engine:
            self.engine.set_time(t)
        self.request_field()
        self.rack.chart.move_cursor(t)
        self.update_mode_line()

    def nudge(self, d: int):
        if self.spec and self.controls.currentWidget() is self.timebar and not self.playing:
            span = self.timebar.t1 - self.timebar.t0
            self.scrub(min(max(self.t + d * span / 50, self.timebar.t0), self.timebar.t1))

    def set_Re(self, Re: float):
        self.Re = float(Re)
        self.rebar.set_Re(self.Re)
        if self.engine:
            self.engine.set_Re(self.Re)
        self.request_field()
        sid = self.spec.sid if self.spec else None
        if sid and self.spec.chart == "centreline_live":
            self.aux.want_centreline(sid, self.Re)
        elif sid and sid in self.curves:
            self.draw_chart()
        self.update_mode_line()

    def reset_dye(self):
        if self.engine:
            self.engine.reset()

    def on_drop(self, pts):
        if self.engine:
            self.engine.drop(pts, 2 + self.drop_i % len(theme.DROP_ORDER))
            self.drop_i += 1

    def drop_random(self):
        if self.spec is None or self.engine is None:
            return
        m = self.models[self.spec.sid]
        d = np.asarray(m["domain"], np.float32)
        lo, hi = d[0::2], d[1::2]
        c = lo + np.random.random(len(lo)).astype(np.float32) * (hi - lo)
        pts = c + np.random.normal(0, 0.04, (360, len(lo))).astype(np.float32) * (hi - lo)
        self.on_drop(np.clip(pts, lo, hi))

    def toggle_full(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def leave_full(self):
        if self.isFullScreen():
            self.showNormal()

    # ------------------------------------------------------------------ data in
    def on_status(self, sid: str, msg: str):
        if self.spec is None or sid != self.spec.sid:
            return
        if msg == "live":
            self.bath.state, self.bath.state_text = "live", ""
        elif msg == "compiling":
            if self.rail.ready.get(sid) != "ready":
                self.bath.state = "compiling"
        elif msg.startswith("error") or msg.startswith("field error"):
            self.bath.state = "offline"
            self.bath.state_text = "The network server stopped answering.\nIt is retried every second; press S to restart it in WSL."
        self.bath.update()

    def on_warm(self, sid: str, state: str):
        self.rail.ready[sid] = state
        self.rail.update()

    def on_step(self, pl: Dict):
        if self.spec is None or pl["sid"] != self.spec.sid:
            return
        if self.spec.dims == 2:
            img = pl.get("image")
            if img is not None:
                self.bath.pigment_img = render.qimage(img)
            if self.bath.state != "offline":
                self.bath.state = "live"
                self.bath.state_text = ""
            self.bath.update()
        elif self.tank is not None:
            self.tank.update_particles(pl["hist"], pl["active"], pl["pig"], pl["colors"], pl["playing"])
        if self.playing:
            self.t = pl["t"]
            self.timebar.set_t(self.t)
            self.request_field()
            now = time.perf_counter()
            if now - self._cursor_at > 0.15:
                self._cursor_at = now
                self.rack.chart.move_cursor(self.t)
                self.update_mode_line()
                if self.view == "sensors":
                    self.update_sensors()
        dev = "GPU" if "cuda" in self.device else self.device
        self.live.setText(f"live on the {dev}   {pl['hz']:.0f} updates/s   velocity {pl['vel_ms']:.0f} ms   field {self.field_ms:.0f} ms")

    def on_field(self, res: Dict):
        if self.spec is None or res["sid"] != self.spec.sid:
            return
        self.field_ms = res["ms"]
        self.last_field[res["sid"]] = res
        self.render_field()

    def render_field(self):
        if self.spec is None:
            return
        res = self.last_field.get(self.spec.sid)
        if res is None:
            return
        f, hdr, sid, view = res["fields"], res["hdr"], self.spec.sid, self.view
        if self.spec.dims == 3:
            if self.tank is None:
                return
            if sid == "coa":
                self._render_vessel(f, view)
                return
            self.tank.update_field(f, int(hdr["shape"][0]))
            return
        base_view = "speed" if view in ("pigment", "sensors") else view
        a = render.scalar(f, base_view)
        ref = render.scalar(f, base_view, ref=True)
        if a is None:
            return
        rng = hdr.get("ranges") or {}
        if base_view in rng:  # fixed over the whole heartbeat (the server measures it on the CFD truth)
            lo, hi = rng[base_view]
        elif base_view == "error":
            lo, hi = render.scale_for([a], "error")
        else:
            key = (sid, base_view, res["Re"] if self.spec.re_mode else None)
            if key not in self.scales:
                self.scales[key] = render.scale_for([a, ref], base_view)
            lo, hi = self.scales[key]
        draw_view = view if view in ("pigment", "sensors") else base_view
        self.bath.field_img = render.qimage(render.to_rgb(a, draw_view, lo, hi))
        # over a data view the pigment drops to a faint veil, so the evidence underneath reads first
        self.bath.pigment_opacity = 1.0 if view in ("pigment", "sensors") else 0.2
        self.bath.callouts = _corner_callouts(a) if (view == "divergence" and sid in ("cavity_ok", "cavity_hard")) else []
        self.bath.cylinder = hdr.get("cylinder")
        k = (hdr.get("units") or {}).get("velocity_ms", 1.0) if base_view in ("speed", "mri", "error") and hdr.get("units") else 1.0
        self.bath.legend = None if view in ("pigment", "sensors") else (render.colour_map(base_view), lo * k, hi * k, _legend_label(base_view, sid))
        if view == "sensors":
            self.update_sensors()
        else:
            self.bath.sensors = None
        self.bath.update()
        # the pulled print: the reference at the same instant and scale
        sheet = self.rack.sheet
        ref_view = base_view if ref is not None else "speed"
        if ref is None:
            ref = render.scalar(f, "speed", ref=True)
        if ref is not None:
            rkey = (sid, ref_view, res["Re"] if self.spec.re_mode else None)
            rlo, rhi = (lo, hi) if ref_view == base_view else self.scales.setdefault(rkey, render.scale_for([ref, render.scalar(f, ref_view)], ref_view))
            sheet.image = render.qimage(render.to_rgb(ref, ref_view, rlo, rhi))
            sheet.cylinder = hdr.get("cylinder")
            if sid == "inverse":
                sheet.note = f"nearest data snapshot, t = {hdr.get('t_ref', self.t):.1f}; same colour scale"
            elif sid == "deeponet":
                sheet.note = f"Re {res['Re']:.0f}, same colour scale"
            elif view in ("pigment", "sensors"):
                sheet.note = "reference speed, on the speed view's colour scale"
            elif ref_view != base_view:
                sheet.note = f"reference {ref_view}; the {base_view} view has no reference counterpart"
            else:
                sheet.note = SHEET[sid][1]
        else:
            sheet.image = None
            sheet.empty = EMPTY.get(sid, "No reference at this setting." if sid != "deeponet" else
                                    f"No reference at Re {res['Re']:.0f}: finite-difference solutions exist at 100, 400 and 1000. The network's flow is still live on the left.")
        sheet.update()
        err = render.rel_l2(f)
        if err is not None:
            self.panel.set_live(f"live: velocity error {_fmt_err(err, self.spec.headline)}")
        else:
            self.panel.set_live("")

    def _render_vessel(self, f, view):
        """Colour the phantom's lumen voxels by the view, on scales fixed for the scene."""
        if self.tank.cloud is None:
            return
        key = {"mri": "speed_mri", "speed": "speed_net", "pressure": "p_net"}.get(view)
        vals = f.get(key) if key else None
        if vals is None:
            self.tank.update_cloud(None)
            self.panel.set_live("")
            return
        if ("coa", view) not in self.scales:
            if view == "pressure":
                m = float(np.nanpercentile(np.abs(vals), 99.5)) * 1.5 or 1.0
                self.scales[("coa", view)] = (-m, m)
            else:
                ref = f.get("speed_mri", vals)
                self.scales[("coa", view)] = (0.0, float(np.nanpercentile(ref, 99.5)) * 1.2 or 1.0)
        lo, hi = self.scales[("coa", view)]
        lut = theme.MAPS["diverging" if view == "pressure" else "sequential"]
        self.tank.update_cloud(vals, lut, lo, hi, alpha=0.55 if view != "mri" else 0.8)
        label = {"mri": "scan speed", "speed": "network speed", "pressure": "network pressure, relative to the downstream plane"}[view]
        unit = "mmHg" if view == "pressure" else "m/s"
        self.panel.set_live(f"colour: {label}, {lo:.2f} to {hi:.2f} {unit}")

    def update_sensors(self):
        pts = self.points.get("inverse")
        if pts is None:
            self.bath.sensors = None
            return
        sel = np.abs(pts[:, 0] - self.t) < 0.6
        self.bath.sensors = pts[sel, 1:3]

    def on_curve(self, sid: str, data: Dict):
        self.curves[sid] = data
        if self.spec and sid == self.spec.sid:
            self.draw_chart()

    def on_points(self, sid: str, data: Dict):
        if data and sid == "coa":
            self.geom[sid] = data
            if self.spec is not None and self.spec.sid == sid and self.tank is not None:
                self.tank.configure_vessel(data)
                self.render_field()
            return
        if data:
            self.points[sid] = np.stack([data["t"], data["x"], data["y"]], -1).astype(np.float32)
            if self.view == "sensors":
                self.update_sensors()

    def on_line(self, sid: str, Re: float, line: Dict):
        self.lines[sid] = (Re, line)
        if self.spec and sid == self.spec.sid:
            self.draw_chart()

    def draw_chart(self):
        s = self.spec
        if s is None:
            return
        data = self.curves.get(s.sid)
        if s.chart == "centreline_live":
            line = self.lines.get(s.sid)
            if line is None or data is None:
                return
            self.rack.chart.draw("centreline", data, s.pigment, Re=line[0], line=line[1])
        elif data is not None and s.chart == "coa_compare":
            self.rack.chart.draw("coa_compare", data, s.pigment, t=self.t)
        elif data is not None and s.chart == "pressure_drop":
            self.rack.chart.draw("pressure_drop", data, s.pigment, t=self.t)
        elif data is not None and s.chart == "lambdas":
            self.rack.coeffs.set(data)
            self.rack.show_coefficients(True)
        elif data is not None:
            self.rack.chart.draw(s.chart, data, s.pigment, t=self.t, Re=self.Re)

    def keyPressEvent(self, e):
        """Arrows that no focused control used: step time on a timed scene."""
        if e.key() in (Qt.Key_Left, Qt.Key_Right):
            self.nudge(1 if e.key() == Qt.Key_Right else -1)
            return
        super().keyPressEvent(e)

    # ------------------------------------------------------------------ screenshots
    def take_shot(self):
        img = self.grab().toImage()
        if self.spec is not None and self.spec.dims == 3 and self.tank is not None:
            from PySide6.QtGui import QImage, QPainter

            arr = np.ascontiguousarray(self.tank.grab_image()[..., :3])
            h, w = arr.shape[:2]
            q = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            pos = self.tank.plotter.interactor.mapTo(self, self.tank.plotter.interactor.rect().topLeft())
            from PySide6.QtCore import QRectF

            p = QPainter(img)
            ia = self.tank.plotter.interactor
            p.drawImage(QRectF(pos.x(), pos.y(), ia.width(), ia.height()), q)
            p.end()
        img.save(self.args.shot)
        print("screenshot ->", self.args.shot, flush=True)
        if self.args.exit_after_shot:
            self.close()

    def closeEvent(self, e):
        for w in (self.engine, self.fieldw, self.aux):
            if w is not None:
                w.stop()
        for w in (self.engine, self.fieldw, self.aux, self.warmer):
            if w is not None:
                w.wait(3000)
        if self.tank is not None:
            self.tank.close_plotter()
        super().closeEvent(e)


def _cache_key(m) -> str:
    """Run name plus the checkpoint's timestamp: curves cached for an older (or untrained) network are never reused."""
    stamp = m.get("run_stamp")
    return f"{m['run']}-{stamp}" if stamp else m["run"]


def _apply_server_verdict(spec, m) -> None:
    """The real-world scene's verdict comes from its evaluation (final_eval.json), read by the server."""
    if spec is None or spec.tab != "real":
        return
    spec.trained = bool(m.get("trained", True))
    fe = (m.get("metrics") or {}).get("final_eval") or {}
    if spec.trained and "dp_peak_err_pct" in fe and spec.sid == "coa":
        spec.verdict = "passed" if fe["dp_peak_err_pct"] < 20.0 else "gate missed"
        spec.headline = f"{fe['dp_peak_err_pct']:.0f} %"
        spec.headline_label = f"peak drop {fe['dp_peak_pinn_mmHg']:.1f} vs catheter {fe['dp_peak_cath_mmHg']:.1f} mmHg; gate below 20 %"
    elif spec.trained and "dp_peak_err_pct" in fe:
        spec.verdict = "passed" if fe["dp_peak_err_pct"] < 10.0 else "gate missed"
        spec.headline = f"{fe['dp_peak_err_pct']:.1f} %"
        spec.headline_label = (f"peak pressure-drop error: {fe['dp_peak_pinn_mmHg']:.2f} vs {fe['dp_peak_true_mmHg']:.2f} mmHg; "
                               f"gate below 10 %")
    elif spec.trained:
        spec.verdict, spec.headline, spec.headline_label = "trained", "-", "not evaluated yet: final_eval.json is missing"
    else:
        spec.verdict, spec.headline = "not trained", "-"
        spec.headline_label = "trains on the desktop and arrives here by itself (docs/DESKTOP_TRAINING.md)"


def _legend_label(view: str, sid: str) -> str:
    if sid in ("stenosis", "stenosis_fast"):
        return {"speed": "speed [m/s]", "mri": "scan speed [m/s], voxel and phase averages with noise", "error": "|network - CFD| [m/s], this frame",
                "pressure": "pressure [mmHg], relative to the distal end", "vorticity": "vorticity"}.get(view, view)
    if view == "error":
        return "|network - reference|, this frame"
    unit = " [m/s]" if sid == "cylinder_fwd" and view == "speed" else ""
    return {"speed": "speed", "vorticity": "vorticity", "pressure": "pressure", "divergence": "divergence, ideally 0"}.get(view, view) + unit


def _fmt_err(err: float, headline: str) -> str:
    """The live number in the measured number's own notation and precision."""
    if "%" in headline:
        num = headline.split()[0]
        d = len(num.split(".")[1]) if "." in num else 0
        return f"{100 * err:.{d}f} %"
    return theme.sci(err)


def _corner_callouts(div) -> list:
    """Each top corner's largest |divergence| beside the median over the whole cavity, from this frame's grid."""
    a = np.abs(np.asarray(div, np.float32))
    nx, ny = a.shape
    k = max(2, nx // 16)
    med = float(np.nanmedian(a))
    out = []
    for xs, x, side in ((slice(0, k), 0.0, "right"), (slice(nx - k, nx), 1.0, "left")):
        v = float(np.nanmax(a[xs, ny - k:]))
        out.append({"x": x, "y": 1.0, "side": side,
                    "text": f"corner: |div| up to {v:.2g}\nwhole cavity: median {med:.2g}"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live viewer for the trained PINN scenes")
    ap.add_argument("--server", default=None, help="server URL; default: find it in WSL, then localhost")
    ap.add_argument("--scene", default=None, help="scene to open: " + ", ".join(s.sid for s in S.SCENES))
    ap.add_argument("--view", default=None)
    ap.add_argument("--Re", type=float, default=None)
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--size", type=lambda s: tuple(int(v) for v in s.lower().split("x")), default=(1600, 960))
    ap.add_argument("--shot", default=None, help="save a screenshot of the window to this PNG")
    ap.add_argument("--wait", type=float, default=12.0, help="seconds before the screenshot")
    ap.add_argument("--exit-after-shot", action="store_true")
    args = ap.parse_args(argv)
    app = QApplication(sys.argv[:1])
    theme.load_fonts()
    app.setStyleSheet(theme.QSS)
    app.setFont(theme.font("text", 13, 400))
    w = MainWindow(args)
    w.show()
    sys.exit(app.exec())
