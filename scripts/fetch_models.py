"""Laptop side of the training hand-off: watch the private models repo, download every new release into runs/, and
restart the live server so the viewer shows the new networks (see scripts/publish_models.sh on the training machine).

    python scripts/fetch_models.py                 # poll every 5 minutes, forever (log: runs/fetch_models.log)
    python scripts/fetch_models.py --once          # one check

Uses the GitHub CLI with the repository owner's token (gh auth token --user <owner>), so it works whichever gh account
is active. Releases already fetched are listed in runs/_fetched_releases.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEN = ROOT / "runs" / "_fetched_releases.txt"
LOG = ROOT / "runs" / "fetch_models.log"


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gh(args, env):
    return subprocess.run(["gh", *args], capture_output=True, text=True, env=env, timeout=600)


def safe_extract(archive: Path, dest: Path) -> list:
    """Extract runs/<run>/... members only; refuse absolute paths and '..'."""
    names = []
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "runs" or not (m.isfile() or m.isdir()):
                raise ValueError(f"refusing archive member {m.name!r}")
        tar.extractall(dest)
        names = [m.name for m in tar.getmembers() if m.isfile()]
    return names


def restart_server():
    script = "/mnt/c" + str(ROOT / "scripts" / "serve_local_wsl.sh").replace("\\", "/")[2:]
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", script], capture_output=True, text=True, timeout=900)
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-2:]
    log("server restarted: " + " | ".join(tail))


def check(repo: str, owner: str) -> int:
    env = dict(os.environ)
    tok = gh(["auth", "token", "--user", owner], env)
    if tok.returncode == 0 and tok.stdout.strip():
        env["GH_TOKEN"] = tok.stdout.strip()
    r = gh(["release", "list", "-R", repo, "--limit", "200", "--json", "tagName,createdAt"], env)
    if r.returncode != 0:
        log(f"cannot list releases of {repo} yet: {(r.stderr or r.stdout).strip()[:200]}")
        return 0
    seen = set(SEEN.read_text().split()) if SEEN.exists() else set()
    new = sorted((x for x in json.loads(r.stdout or "[]") if x["tagName"] not in seen), key=lambda x: x["createdAt"])
    got = 0
    for rel in new:
        tag = rel["tagName"]
        with tempfile.TemporaryDirectory() as tmp:
            d = gh(["release", "download", tag, "-R", repo, "-D", tmp, "-p", "*.tar.gz"], env)
            if d.returncode != 0:
                log(f"download of {tag} failed: {d.stderr.strip()[:200]}")
                continue
            for a in Path(tmp).glob("*.tar.gz"):
                try:
                    names = safe_extract(a, ROOT)
                    log(f"fetched {tag}: {len(names)} files -> {sorted({n.split('/')[1] for n in names})}")
                    got += 1
                except Exception as e:
                    log(f"rejected {tag}: {e}")
        with open(SEEN, "a", encoding="utf-8") as f:
            f.write(tag + "\n")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("PINN_MODELS_REPO", "VoyagerToko/pinn-models"))
    ap.add_argument("--every", type=float, default=300.0, help="seconds between checks")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-restart", action="store_true")
    args = ap.parse_args()
    owner = args.repo.split("/")[0]
    log(f"watching {args.repo} every {args.every:.0f} s")
    while True:
        try:
            if check(args.repo, owner) and not args.no_restart:
                restart_server()
        except Exception as e:  # keep watching through network hiccups
            log(f"check failed: {type(e).__name__}: {e}")
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
