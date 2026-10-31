# claude helper

Material made for or by the AI assistants that work on this repository. None of it is needed to install, train,
evaluate or view the project; the project itself is everything outside this folder (README.md, section 1).

| path | what it is |
|---|---|
| `DESKTOP_TRAINING.md` | the runbook for training on the desktop PC and sending the models to the laptop |
| `design/` | the viewer's design notes: `DESIGN.md` (visual system), `PRODUCT.md` (who it is for), `impeccable/` (the design tool's working files) |
| `diagnostics/` | one-off investigation scripts behind entries in `docs/STATUS.md` (phantom momentum balance, pressure fit, pressure-network test, 4D-FlowP checks, a partial cylinder evaluation, paper overlap check, GPU-hour tally) |
| `tools/` | machine utilities: server speed and integrity checks, WSL verification, desktop clean-up, the JAX-PI cavity example run |
| `presentation/presentation.md` | the presentation brief: the split between the four presenters, shared basics, one part per presenter |
| `report/` | the RM project report (college template) and its sources: `build_report.py` fills `template.docx`, `make_figures*.py` draw `figs/` from `runs/` and `docs/STATUS.md`, `postprocess.py` puts the XML in schema order, `measure_toc.py` and `sync_toc.py` keep the contents page numbers right (export a PDF with Word, measure, rebuild) |
| `restructure/PLAN.md` | the proposed folder restructure (branch `restructure`, not started) |

The scripts here find the repository from their own location, so run them from anywhere, quoting the space:

```bash
python "claude helper/diagnostics/coa_pressure_fit.py" coa_P2_baseline
```
