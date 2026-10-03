"""Total GPU time of the study, from the job logs.

    python scripts/gpu_hours.py

Counts (1) every scripts/run_job.sh job: from its "[run_job] ... start" line to its last "[run_job]" line in
runs/<name>.log (training, evaluation and figures; killed jobs end at their last logged line);
(2) every "!" command of the job queue (probes, evaluations, rendering) from runs/queue_daemon.log;
(3) the runs made before the job wrapper existed, with the wall-clock recorded in STATUS.md (listed below).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00)")

# runs before 2026-09-24 (no run_job.sh); hours as recorded in STATUS.md (Adam + L-BFGS where known)
EARLIER = {"tgv2d_G (2026-09-22)": 2.0, "cavity_C (2026-09-22)": 1.6, "cavity_F (2026-09-23)": 1.5}


def t(s):
    return datetime.fromisoformat(s)


def main():
    total = 0.0
    rows = []
    for log in sorted(RUNS.glob("*.log")):
        # "[run_job]" markers can sit mid-line when a killed trainer left an unterminated line
        lines = [l[l.index("[run_job]"):] for l in log.read_text(errors="ignore").splitlines() if "[run_job]" in l]
        starts = [i for i, l in enumerate(lines) if re.match(r"\[run_job\] \S+ start ", l)]
        mtime = datetime.fromtimestamp(log.stat().st_mtime).astimezone()
        for k, i in enumerate(starts):  # a log can hold several attempts
            end = (starts[k + 1] - 1) if k + 1 < len(starts) else len(lines) - 1
            a, b = TS.search(lines[i]), TS.search(lines[end])
            if not a:
                continue
            t_end = t(b.group(1)) if (b and end > i) else None
            if t_end is None and k == len(starts) - 1:  # killed without an end marker: the last write is the end
                t_end = mtime
            if t_end is None:
                continue
            h = max(0.0, (t_end - t(a.group(1))).total_seconds() / 3600)
            rows.append((log.stem, h))
            total += h
    q = RUNS / "queue_daemon.log"
    if q.exists():
        pending = {}
        for l in q.read_text().splitlines():
            m = TS.search(l)
            if not m:
                continue
            if " start: !" in l:
                pending[l.split(" start: ", 1)[1]] = t(m.group(1))
            elif " end rc=" in l and ": !" in l:
                cmd = l.split(": ", 1)[1]
                if cmd in pending:
                    h = (t(m.group(1)) - pending.pop(cmd)).total_seconds() / 3600
                    rows.append(("queue: " + cmd[:60], h))
                    total += h
    for k, h in EARLIER.items():
        rows.append((k, h))
        total += h
    for name, h in rows:
        print(f"{h:7.2f} h  {name}")
    print(f"{total:7.1f} h  total")


if __name__ == "__main__":
    main()
