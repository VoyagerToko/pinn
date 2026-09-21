#!/usr/bin/env bash
# Fetch every dataset and reference file the code expects under data/ (handbook STEP 3).
# Run from WSL/Linux (Windows curl is slow and the tu-dortmund server rejects its TLS stack):
#     wsl -d Ubuntu -- bash /mnt/c/Users/mehul/pinn/scripts/download_data.sh
# Idempotent: existing files are skipped.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p data/cylinder_wake data/dfg_benchmark data/tgv3d_re1600

get() {  # get URL DEST
  if [ -s "$2" ]; then echo "skip  $2"; return; fi
  curl -sSL --max-time 600 -o "$2" "$1" && echo "ok    $2 ($(wc -c < "$2") bytes)" || echo "FAIL  $1"
}

# Raissi et al. 2019 cylinder wake (Nektar, Re = 100)
get "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_wake.mat" data/cylinder_wake/cylinder_nektar_wake.mat

# FEATFLOW DFG 2D-2 / 2D-3 reference series (Cd, Cl, pressure probes)
FF="https://wwwold.mathematik.tu-dortmund.de/~featflow"
get "$FF/media/dfg_bench2_2d/draglift_q2_cn_lv3-6_dt1-4.zip"        data/dfg_benchmark/dfg2d2_draglift_q2_cn_lv3-6_dt1-4.zip
get "$FF/media/dfg_bench2_2d/pressure_q2_cn_lv3-6_dt1-4.zip"        data/dfg_benchmark/dfg2d2_pressure_q2_cn_lv3-6_dt1-4.zip
get "$FF/media/dfg_bench2new_2d/20141124_dfg_bench2_official.zip"   data/dfg_benchmark/dfg2d2_official_20141124.zip
get "$FF/media/dfg_bench3_2d/draglift_q2_cn_lv1-6_dt4.zip"          data/dfg_benchmark/dfg2d3_draglift_q2_cn_lv1-6_dt4.zip
get "$FF/media/dfg_bench3_2d/pressure_q2_cn_lv1-6_dt4.zip"          data/dfg_benchmark/dfg2d3_pressure_q2_cn_lv1-6_dt4.zip
get "$FF/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html" data/dfg_benchmark/dfg2d2_page.html
get "$FF/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html" data/dfg_benchmark/dfg2d3_page.html
for z in data/dfg_benchmark/*.zip; do
  d="${z%.zip}"; [ -d "$d" ] || { mkdir -p "$d" && unzip -qo "$z" -d "$d" && echo "unzip $d"; }
done

# 3D Taylor-Green Re = 1600 references
get "https://ntrs.nasa.gov/api/citations/20130011044/downloads/20130011044.pdf"                 data/tgv3d_re1600/DeBonis2013_NASA_TGV.pdf
get "https://how4.cenaero.be/sites/how4.cenaero.be/files/2025-01/BS1_TaylorGreenVortexRe1600.pdf"   data/tgv3d_re1600/HiOCFD4_BS1_TaylorGreenVortexRe1600.pdf
get "https://how5.cenaero.be/sites/how5.cenaero.be/files/2025-01/BS1_TaylorGreenVortexRe1600_0.pdf" data/tgv3d_re1600/HiOCFD5_WS1_TaylorGreenVortexRe1600.pdf

# reference code (shallow clones)
mkdir -p external && cd external
[ -d jaxpi ]        || git clone --depth 1 https://github.com/PredictiveIntelligenceLab/jaxpi jaxpi
[ -d jaxpi-pirate ] || git clone --depth 1 -b pirate https://github.com/PredictiveIntelligenceLab/jaxpi jaxpi-pirate
[ -d SPINN ]        || git clone --depth 1 https://github.com/stnamjef/SPINN SPINN
[ -d CausalPINNs ]  || git clone --depth 1 https://github.com/PredictiveIntelligenceLab/CausalPINNs CausalPINNs
echo done
