# Interpretable, Uncertainty-Aware Battery SOH/RUL Prognostics on Public Datasets

Code and analysis pipeline accompanying the manuscript:

> *An Interpretable and Uncertainty-Aware Framework for Lithium-Ion Battery
> State-of-Health and Remaining-Useful-Life Prognostics on Public Datasets.*
> P. Wang, P. Zhou, S. Hong, Z. Quan.

This repository reproduces every result, figure and table in the paper from **public**
battery-cycling data, under strict **cell-level, batch-aware** evaluation, with
**split-conformal** prediction intervals.

> **Honesty note.** Raw cycling data are **not** redistributed here (they are governed by the
> original providers' licences); the scripts download/parse them from the official sources. The
> code emits the same numbers reported in the paper, including the **negative/null results**
> (active learning ≈ random; cross-batch and cross-system collapse). Nothing is hard-coded to a
> target number.

---

## Repository structure

```
scripts/                 all analysis code (flat layout; run from this directory)
  severson_download.py     download the Severson/MATR fast-charging corpus (resumable)
  severson_parse.py        parse MATLAB v7.3/HDF5 -> per-cell CSV (has --selftest)
  calce_parse.py           parse CALCE CS2 (LCO) Arbin .xlsx -> per-cycle SOH + curves
  nasa_soh.py              parse + model NASA PCoE (LCO) under leave-one-cell-out
  battery_make_dataset.py  assemble the modelling table
  severson_features.py / severson_soh_extract.py   curve + IC feature extraction
  severson_soh_model.py    SOH models (ridge / SVR / RF), cell-level split
  severson_soh_ci.py       split-conformal + tree-quantile intervals
  severson_soh_ablation.py feature-group ablation (multi-seed)
  severson_rul.py          RUL: end-to-end (out-of-sample SOH) + oracle comparison
  battery_symbolic.py      genetic-programming symbolic regression (degradation law)
  active_learning_experiment.py   uncertainty vs random acquisition (multi-seed)
  crosschem_experiment.py  direct LFP->LCO transfer (zero-shot / re-fit / few-shot / calib sweep)
  crosschem_figure.py      cross-system result figure
  supp_*.py                supplementary experiments (see mapping below)
  make_figures*.py, pubstyle.py, vecsave.py   journal-figure generation (PNG + vector PDF)
  battery_autora_loop.py   optional AutoRA closed-loop driver
  build_docx.js, renumber_cites.py   manuscript assembly + citation numbering (Node.js)
```

## Installation

Python 3.11 is recommended. Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.11
uv pip install numpy "numpy<2" pandas scikit-learn scipy h5py gplearn matplotlib requests tqdm openpyxl
# optional, only for battery_autora_loop.py:
uv pip install autora
```

(or `pip install -r requirements.txt`). The manuscript builder additionally needs Node.js with
the `docx` package (`npm i docx`).

## Data acquisition (not redistributed)

| Dataset | Chemistry | How to obtain |
|---------|-----------|---------------|
| Severson / MATR | LFP | `python scripts/severson_download.py` (matr.io direct links) |
| NASA PCoE | LCO | NASA Open Data Portal: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets (place under `scripts/data/nasa/`) |
| CALCE CS2 | LCO | https://calce.umd.edu/battery-data (CS2 cells; `scripts/data/calce/`) |

Files land under `scripts/data/`. (The Sandia **SNL** multi-chemistry set used for the
chemistry-vs-condition discussion is access-limited and is **not** part of this reproduction; see
the manuscript Limitations.)

## Reproduce

Run from `scripts/` after acquiring data:

```bash
python severson_parse.py          # HDF5 -> CSV
python severson_soh_extract.py    # per-cycle SOH features (IC, v_median, ...)
python severson_soh_model.py      # SOH: ridge/SVR/RF, cell-level (Table 3-4, Fig 3)
python severson_soh_ci.py         # conformal + tree-quantile intervals (Fig 3b)
python severson_rul.py            # end-to-end RUL + oracle (Fig 4)
python battery_symbolic.py        # symbolic degradation law (Fig 6)
python nasa_soh.py                # second-dataset LOCO (Table 5, Fig 7)
python active_learning_experiment.py   # AL vs random, multi-seed (Fig 5)
python crosschem_experiment.py    # LFP->LCO transfer + calibration sweep (Table 6)
python crosschem_figure.py        # cross-system figure (Fig 11)
# supplementary experiments:
python supp_partial_window.py     # partial voltage-window SOH (Fig 9)
python supp_drift_mmd.py          # cross-batch transfer x MMD (Fig 8, S2)
python supp_aci.py                # adaptive conformal under drift (Fig 10)
python supp_permutation_importance.py  # robust importance (S1)
python supp_mi_window.py          # MI-guided window selection (S5)
python supp_batch_traj.py         # batch degradation trajectories (S6)
python supp_rul_endtoend.py       # RUL three-setting comparison (S-table)
python supp_sg_sensitivity.py supp_al_seedsize.py supp_al_diagnosis.py  # robustness/diagnostics
# figures + manuscript:
python make_figures.py && python make_figures_v2.py && python make_figures_v3.py
node build_docx.js                # assemble the .docx
```

Every figure is also exported as a vector PDF (`vecsave.py`) at 300 dpi with a journal style
(`pubstyle.py`).

## Conventions (do not drop)

- **Cell-level splitting** everywhere; cycle-level random splits leak and inflate R².
- **Multi-seed mean ± s.d.**; a single split can flip a conclusion.
- **Calibrated, not guaranteed**: with correlated per-cell cycles we report empirical conformal
  coverage, not a per-prediction guarantee; static conformal breaks under drift, ACI restores it.

## Citation

Please cite the manuscript above. BibTeX will be added on acceptance.

## Licence

Code: MIT (see `LICENSE`). Battery datasets remain under their original providers' licences and
are not included here.
