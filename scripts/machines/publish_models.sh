#!/usr/bin/env bash
# Publish finished runs from the training machine (the desktop) to the private GitHub repo that the laptop watches
# (scripts/machines/fetch_models.py). Each run becomes one release with one asset, <run>.tar.gz, holding what the live server
# needs: config.json, final_eval.json, latest.msgpack, metrics.csv and the run's other small json/txt/md/png files
# (no intermediate ckpt_* files). A results folder without a checkpoint (runs/coa_report) is published the same way.
#
#   bash scripts/machines/publish_models.sh stenosis_mri_fast coa_P2_baseline_eddy
#
# Needs the GitHub CLI logged in as the repository owner (gh auth status). The models repo is created private on first use.
set -euo pipefail
REPO="${PINN_MODELS_REPO:-VoyagerToko/pinn-models}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[ "$#" -ge 1 ] || { echo "usage: publish_models.sh <run> [<run> ...]"; exit 1; }
PY="$(command -v python3 || command -v python)"
gh repo view "$REPO" >/dev/null 2>&1 || gh repo create "$REPO" --private --add-readme --description "Trained pinnflow checkpoints, fetched by the laptop viewer (scripts/machines/fetch_models.py)" >/dev/null
TMP="$(mktemp -d)"
for run in "$@"; do
  d="runs/$run"
  [ -d "$d" ] || { echo "skip $run: no $d"; continue; }
  files=()
  for f in "$d"/config.json "$d"/latest.msgpack "$d"/final_eval.json "$d"/metrics.csv "$d"/*.json "$d"/*.txt "$d"/*.md "$d"/*.png; do
    [ -f "$f" ] && files+=("$f")
  done
  [ "${#files[@]}" -gt 0 ] || { echo "skip $run: nothing to publish in $d"; continue; }
  mapfile -t files < <(printf '%s\n' "${files[@]}" | sort -u)
  tar czf "$TMP/$run.tar.gz" "${files[@]}"
  tag="$run-$(date -u +%Y%m%d-%H%M%S)"
  note="$("$PY" - "$d" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "final_eval.json"
print(json.dumps(json.loads(p.read_text()), indent=1) if p.exists() else "no final_eval.json")
EOF
)"
  gh release create "$tag" "$TMP/$run.tar.gz" -R "$REPO" -t "$tag" -n "$note" >/dev/null
  echo "published $run as $REPO@$tag ($(du -h "$TMP/$run.tar.gz" | cut -f1))"
done
rm -rf "$TMP"
