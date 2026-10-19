"""Steady-state speed of a running scripts/serve.py: full-field frames and particle velocity queries.

    python scripts/bench_server.py http://localhost:8000

Warm-up requests (compiles) are excluded; each figure is the median of several timed requests, measured
end to end from this client (server compute + transfer + decode).
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np


def timed(fn, reps=5):
    fn()  # warm-up / compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    import httpx

    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
    c = httpx.Client(timeout=600)
    models = {m["id"]: m for m in c.get(f"{base}/models").json()}
    rng = np.random.default_rng(0)
    print(f"{'scene':13s} {'request':26s} {'median':>9s} {'rate':>9s}")
    for sid, m in models.items():
        ctl, dims = m["controls"], m["dims"]
        dom = np.asarray(m["domain"], dtype=np.float32).reshape(-1, 2)
        t_vals = np.linspace(*ctl["t"], 50) if "t" in ctl else [0.0]
        Re = 650.0 if sid == "deeponet" else 1000.0
        grids = [(32,), (48,), (64,)] if dims == 3 else [(64,), (128,), (256,)]
        for (n,) in grids:
            k = {"i": 0}

            def frame():
                k["i"] += 1
                t = float(t_vals[k["i"] % len(t_vals)])
                r = c.get(f"{base}/field/{sid}", params={"n": n, "t": t, "Re": Re, "dtype": "float16"})
                r.raise_for_status()

            dt = timed(frame)
            label = f"field {n}^{dims}"
            print(f"{sid:13s} {label:26s} {dt * 1e3:7.0f} ms {1 / dt:7.1f}/s")
        for npart in (1000, 5000, 20000):
            pts = (dom[:, 0] + rng.random((npart, dims)) * (dom[:, 1] - dom[:, 0])).astype("<f4")
            body = pts.tobytes()
            k = {"i": 0}

            def vel():
                k["i"] += 1
                t = float(t_vals[k["i"] % len(t_vals)])
                r = c.post(f"{base}/velocity/{sid}", params={"t": t, "Re": Re, "dim": dims}, content=body)
                r.raise_for_status()

            dt = timed(vel)
            label = f"velocity {npart} particles"
            print(f"{sid:13s} {label:26s} {dt * 1e3:7.0f} ms {1 / dt:7.1f}/s")


if __name__ == "__main__":
    main()
