"""Check a running scripts/serve.py from another machine: every scene's field and curve arrive intact.

    python "claude helper/tools/check_server.py" https://<tunnel>.trycloudflare.com
    python "claude helper/tools/check_server.py" http://100.119.205.73:8000 --n 128 --out runs/_check

Prints shape, value ranges, payload size and latency per request, and writes one contact-sheet PNG
(<out>/server_check.png) showing the prediction (and reference, where served) of every scene.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np


def unpack(buf: bytes):
    """Inverse of serve.pack_fields: uint32 header length, JSON header, float32 arrays."""
    n = struct.unpack("<I", buf[:4])[0]
    header = json.loads(buf[4 : 4 + n])
    body = memoryview(buf)[4 + n :]
    size = int(np.prod(header["shape"]))
    fields = {}
    for i, k in enumerate(header["fields"]):
        dt = np.dtype(header.get("dtype", "float32"))
        nb = size * dt.itemsize
        fields[k] = np.frombuffer(body[i * nb : (i + 1) * nb], dtype=dt).astype(np.float32).reshape(header["shape"])
    return header, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="server base URL")
    ap.add_argument("--n", type=int, default=128, help="2D grid points along x")
    ap.add_argument("--n3d", type=int, default=48, help="3D grid points per axis")
    ap.add_argument("--out", default="runs/_check")
    ap.add_argument("--skip-curves", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    import httpx

    base = args.base.rstrip("/")
    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=20.0), follow_redirects=True)
    t0 = time.time()
    h = client.get(f"{base}/health").json()
    print(f"health  {time.time() - t0:5.2f}s  device={h['device']}  models={h['models']}")
    models = client.get(f"{base}/models").json()
    print(f"models  {len(models)} scenes")
    for m in models:
        print(f"   {m['id']:13s} {m['status']:13s} {m['dims']}D  run={m['run']:24s} controls={m['controls']}")

    results = []
    ok = True
    for m in models:
        sid, ctl = m["id"], m["controls"]
        q = {"n": args.n3d if m["dims"] == 3 else args.n, "dtype": args.dtype}
        if "t" in ctl:
            q["t"] = round(0.5 * (ctl["t"][0] + ctl["t"][1]), 3)
        if "Re" in ctl:
            q["Re"] = 400 if sid == "deeponet" else ctl["Re"][-1]
        t0 = time.time()
        r = client.get(f"{base}/field/{sid}", params=q)
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"FIELD {sid}: HTTP {r.status_code} {r.text[:200]}")
            ok = False
            continue
        hdr, f = unpack(r.content)
        stats = []
        for k, a in f.items():
            fin = np.isfinite(a)
            if not fin.any():
                stats.append(f"{k}=ALL-NaN")
                ok = False
                continue
            stats.append(f"{k}[{np.nanmin(a):+.3g},{np.nanmax(a):+.3g}]")
        nan_frac = {k: float(np.mean(~np.isfinite(a))) for k, a in f.items()}
        extra = f" nan%={100 * max(nan_frac.values()):.0f}" if max(nan_frac.values()) > 0 else ""
        print(f"field   {sid:13s} {dt:5.2f}s  server {float(r.headers.get('x-compute-seconds', 'nan')):5.2f}s  {len(r.content) / 1e6:5.2f} MB  shape={hdr['shape']}{extra}")
        print(f"        {'  '.join(stats)}")
        err = None
        if "u_ref" in f:
            fin = np.isfinite(f["u"]) & np.isfinite(f["u_ref"])
            pred = np.stack([f["u"][fin], f["v"][fin]])
            ref = np.stack([f["u_ref"][fin], f["v_ref"][fin]])
            err = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
            print(f"        rel L2 velocity vs served reference: {err:.3e}")
        results.append((m, hdr, f, err))

        if not args.skip_curves:
            t0 = time.time()
            c = client.get(f"{base}/curve/{sid}")
            dt = time.time() - t0
            if c.status_code != 200:
                print(f"curve   {sid:13s} HTTP {c.status_code} after {dt:.1f}s {c.text[:120]}")
                ok = False
            else:
                print(f"curve   {sid:13s} {dt:5.2f}s  keys={list(c.json())}")
        if sid == "inverse":
            p = client.get(f"{base}/points/{sid}").json()
            print(f"points  {sid:13s} n={p.get('n')}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = len(results)
        fig, ax = plt.subplots(rows, 2, figsize=(11, 2.6 * rows), squeeze=False)
        for i, (m, hdr, f, err) in enumerate(results):
            if m["dims"] == 3:
                k = hdr["shape"][2] // 2
                pred, ref, name = f["speed"][:, :, k], f["q"][:, :, k], "speed | Q (mid z-slice)"
            else:
                speed = np.hypot(f["u"], f["v"])
                pred = speed
                ref = np.hypot(f["u_ref"], f["v_ref"]) if "u_ref" in f else f.get("div", f["vort"])
                name = "speed: prediction | " + ("reference" if "u_ref" in f else ("divergence" if "div" in f else "vorticity"))
            ext = hdr["extent"][:4]
            for j, a in enumerate((pred, ref)):
                im = ax[i, j].imshow(np.asarray(a).T, origin="lower", extent=ext, aspect="equal", cmap="viridis" if j == 0 else "magma")
                fig.colorbar(im, ax=ax[i, j], shrink=0.8)
            label = f"{m['id']} ({m['status']})  t={hdr.get('t')}  Re={hdr.get('Re')}"
            if err is not None:
                label += f"  relL2={err:.2e}"
            ax[i, 0].set_title(label, fontsize=8, loc="left")
            ax[i, 1].set_title(name, fontsize=8, loc="left")
        fig.tight_layout()
        fig.savefig(out / "server_check.png", dpi=110)
        print(f"contact sheet -> {out / 'server_check.png'}")
    except Exception as e:
        print("plot skipped:", e)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
