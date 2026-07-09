# Fairness Auditing in Public Tabular Benchmarks

Processed outputs, analysis code, and reproducibility materials for the manuscript
*Fairness Auditing in Public Tabular Benchmarks: A Multi-Dataset Study of Bias and Mitigation*
(Journal of Big Data).

## Authors

M. Samy (corresponding author), Shaimaa Lazem, M. M. Gabr, Khairia El-Said El-Nadi, L. M. Fatehy

Contact: mahmoud_samy@alexu.edu.eg

## Description

This repository archives the processed results and scripts from a 30-seed fairness
audit on three public tabular benchmarks (Adult Income, UCI Bank Marketing, ACSIncome
California sample). It includes main-grid results (3,690 runs), post-processing tables,
manuscript figures, and reviewer-driven robustness checks (age binning, Equalized Odds
calibration, CFS/DI sensitivity).

## What is included

- `data_tables/` — configuration-level aggregates, per-seed results, statistical tests, rankings
- `figures/` — manuscript figures (PDF and PNG)
- `revision_robustness/` — R2A/R2B/R2C sensitivity outputs (CSVs and figures)
- `manifests/` — configuration manifest, validation report, runtime notes
- `code/` — post-processing and revision-analysis scripts
- `requirements.txt` — pinned Python package versions
- `REPRODUCIBILITY_FACT_SHEET.md` — protocol summary

## What is not included

Raw benchmark datasets are **not** redistributed. Obtain them from:

- **Adult Income** — UCI Machine Learning Repository
- **UCI Bank Marketing** — UCI Machine Learning Repository (`bank-additional-full.csv`)
- **ACSIncome** — Folktables / U.S. Census ACS microdata (2018 California 1-Year)

Provenance, row counts, positive rates, protected settings, and the UCI Bank SHA-256
fingerprint are in `data_tables/dataset_provenance_and_fingerprints.csv`.

## Software requirements

Python 3.8.5 (CPython) and the packages listed in `requirements.txt` (aif360, fairlearn,
scikit-learn, xgboost, folktables, pandas, numpy, scipy, statsmodels, matplotlib).

## Reproducing tables and figures from processed outputs

1. Create a virtual environment and install dependencies:
   `pip install -r requirements.txt`
2. The canonical per-seed result file is `data_tables/per_seed_full_results_3690.csv`.
3. Run post-processing:
   `python code/run_phase5a.py` (paths in the scripts assume a `q1_upgrade/` layout;
   adjust paths or place this repository accordingly).
4. Manuscript figure assets can be regenerated with `python code/build_manuscript_assets.py`
   when the full project layout and processed tables are available.

Revision analyses (R2A--R2C) can be re-run from raw data with the scripts in `code/`;
the archived CSVs in `revision_robustness/` document the reported sensitivity checks.

## Archiving and DOI

This package is intended for public archiving (e.g. GitHub and Zenodo). **No DOI is
assigned yet.** After upload and DOI assignment, update the manuscript Availability
statement and this README with the actual DOI. See `DOI_RELEASE_NOTE.md`.

## License

See `LICENSE_NOTE.txt`. Authors must choose a license before public release.

## Citation

See `CITATION.cff` for metadata. Update the `doi` field after Zenodo registration.
