"""Evaluate the completed windows of a cylinder run that was stopped early (CPU is fine).

    JAX_PLATFORMS=cpu python scripts/eval_cylinder_partial.py runs/cylinder_G_16x4k_stopped runs/cylinder_G_ramp_4x12k

Only windows with a latest.msgpack (i.e. finished) are used; results go to <run>/eval_partial.json and
<run>/drag_lift_partial.csv. eval.json (GPU cost numbers of complete runs) is not touched.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as E  # noqa: E402  (configures JAX on import)


def main():
    for wd in map(Path, sys.argv[1:]):
        done = sorted(w for w in wd.glob("window_*") if (w / "latest.msgpack").exists())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy(wd / "config.json", tmp / "config.json")
            for w in done:
                (tmp / w.name).mkdir()
                shutil.copy(w / "latest.msgpack", tmp / w.name / "latest.msgpack")
                if (w / "metrics.csv").exists():
                    shutil.copy(w / "metrics.csv", tmp / w.name / "metrics.csv")
            out = E.eval_cylinder(E.load_cfg(tmp), tmp, dt_eval=0.01)
            out["windows_evaluated"] = [w.name for w in done]
            shutil.copy(tmp / "drag_lift.csv", wd / "drag_lift_partial.csv")
            if (tmp / "drag_lift.png").exists():
                shutil.copy(tmp / "drag_lift.png", wd / "drag_lift_partial.png")
        (wd / "eval_partial.json").write_text(json.dumps(out, indent=2, default=str))
        print(wd.name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if not k.startswith(("ref/", "featflow/"))})


if __name__ == "__main__":
    main()
