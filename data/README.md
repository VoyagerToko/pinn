# data/

Downloaded reference data is not committed (see `.gitignore`). Recreate everything with

```bash
bash scripts/download_data.sh        # Linux / WSL
```

| folder | contents | used by |
|---|---|---|
| `cylinder_wake/` | `cylinder_nektar_wake.mat` (Raissi et al. 2019, Re=100 wake) | Benchmark C inverse problem |
| `dfg_benchmark/` | FEATFLOW Cd/Cl and pressure-probe series for DFG 2D-2 and 2D-3 (unzipped) + benchmark pages | Benchmark C validation (`pinnflow.data.load_featflow_series`) |
| `tgv3d_re1600/` | HiOCFD4/5 case descriptions, DeBonis 2013 NASA report | Benchmark D validation |

The Schaefer & Turek 1996 paper is no longer served at the handbook URL; the reference intervals are in
`pinnflow.benchmarks.DFGCylinder.reference`.
