# CIN2+ information versus architecture

This repository accompanies the Frontiers in Oncology manuscript
**“Information rather than architecture: cross-learner attribution and
clinician–model complementarity in CIN2+ risk prediction for women referred
for colposcopy”** (manuscript 1814215). It contains
the full, configuration-driven analysis graph, synthetic demonstration data,
the pre-specified analysis plans, and summary-only reference results.

No patient data are present. **SYNTHETIC — not patient data.** The committed
CSV files were independently generated from published Table 2 marginal
summaries. Their outcome coefficients are human-selected and were not estimated
from real data. Consequently, synthetic results will not equal article results.

## What is reproduced

The code defines ten primary model families (logistic regression, XGBoost,
LightGBM, CatBoost, random forest, FT-Transformer, TabPFN, a tabular
transformer, and two Tab-MFM variants), `full` and `nopf` feature sets, two
full-only Tab-MFM ablations, frozen external evaluation, and all 31 non-empty
subsets of five clinical variable groups. Downstream scripts implement DeLong
inference, shared patient-level bootstrap resampling, Holm adjustment, nested
calibration, nested thresholds, Wilson intervals, decision curves, clinician–
model complementarity, exact group Shapley values, Kendall's W, permutation
inference, age sensitivity analysis, and within-stratum calibration.

The article numbers are provided only as summary results in
`results/reference/`. Patient-level OOF probabilities and external predictions
are deliberately excluded.

## Installation

Python 3.10.11 is the recorded interpreter. For CPU use:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r environment/requirements-cpu.txt
```

On Windows, replace `.venv/bin/python` with `.venv/Scripts/python`.

For CUDA 12.4, use `environment/requirements-gpu.txt`. A GPU run can differ
from CPU at floating-point level. Do not change devices during a model family's
31-subset run. The article's main analysis was CPU-based and used eight Torch
intra-op threads, as recorded in `configs/protocol.yaml`; analysis A used CPU
for the five classical families and GPU for the five neural/foundation-model
families.

## Quick synthetic run

```bash
python scripts/run_all.py --synthetic --fast --device cpu --output outputs/smoke
```

Fast mode is a software smoke test that runs each family's real estimator class.
It shortens neural training to one Tab-MFM pretraining epoch and two supervised
epochs, and uses 50 bootstrap samples / 100 permutations. It does not transform
one model's predictions into another family's output, and its estimates are not
article results. If `TABPFN_MODEL_PATH` does not identify an already-downloaded
local checkpoint, TabPFN is recorded as `SKIPPED (no local weights)` and no
replacement prediction column is created. The full settings in
`configs/protocol.yaml` are used when `--fast` is omitted.

Run the automated smoke test with:

```bash
python -m pytest tests/test_smoke.py -q
```

Outputs include Figures 1–5 and Supplementary Figure S1, summary JSON files,
performance tables, and a run summary that records the mode, device, and
elapsed time. The output directory is ignored by Git because row-level files
must never be committed.

## Authorized local data

An authorized investigator may replace the defaults with local CSV or XLSX
files:

```bash
python scripts/run_all.py --data path/to/development.xlsx \
  --external path/to/external.xlsx --device cpu --output outputs/local-full
```

Required variables and coding are in `data/schema.csv`. Any columns whose names
look like names or identifiers are dropped immediately on loading. Outputs
must remain local and untracked. Never copy real-data outputs into the
repository or release archive.

For an exact CPU audit against a previously generated local OOF file:

```bash
python tools/verify_reproduction.py \
  --candidate-oof outputs/local-full/oof_raw_all.csv \
  --reference-oof path/to/authorized/reference_oof.csv \
  --candidate-results outputs/local-full \
  --report outputs/local-full/reproduction_report.json
```

For the five classical families, the verifier enforces maximum absolute OOF
error `1e-6`. Neural and TabPFN comparisons report probability differences and
both AUROCs; an AUROC difference above `0.001` is a failed reproduction check.
CPU and GPU floating-point results are not expected to be bitwise identical.

## TabPFN weights

TabPFN is local-only in this project. Install the pinned package, obtain the
model weights through the package author's official acquisition process, read
and accept their license, and set `TABPFN_MODEL_PATH` to the local classifier
checkpoint file. Weight files are not distributed here. Remote inference APIs
are not used. A run without that local file marks TabPFN as skipped and never
constructs a substitute prediction.

## Reproducibility and privacy

The model seed is 13 and the fold definition is
`StratifiedKFold(5, shuffle=True, random_state=42)`. Preprocessing is fit inside
training folds. Neural early stopping uses a seed-13 stratified 15% subset of
the outer training fold; outer validation records never control training.
Detailed settings are in `docs/protocol.md`.

Before release, run:

```bash
python tools/privacy_scan.py .
```

The scanner rejects spreadsheet/array/model files, files above 1 MiB, direct
identifier-like values, contact identifiers, and local absolute-path markers.
The `.gitignore` additionally blocks common private artifacts. The scan is a
release guard, not a substitute for institutional disclosure review.

## Repository guide

- `synthetic/`: deterministic synthetic generator.
- `src/cin2/`: loaders, preprocessing, model definitions, CV, and Tab-MFM.
- `scripts/`: model, group-subset, external, and one-command runners.
- `analysis/`: statistical analyses and figures.
- `docs/prespecification/`: byte-preserved source plans, English translation,
  and recorded SHA-256 hashes.
- `docs/paper_map.md`: article table/figure provenance and explicitly unmapped
  items.
- `results/reference/`: summary-only article results.
- `tools/`: privacy and local reproduction audits.

## Citation

Citation metadata are in `CITATION.cff`. The DOI is intentionally a placeholder
and should be updated after article acceptance.

## License

Code is released under the MIT License. TabPFN software and weights have their
own licensing terms and are not relicensed or redistributed by this repository.
