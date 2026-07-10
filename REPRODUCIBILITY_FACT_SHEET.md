# Reproducibility Fact Sheet

Fairness Auditing in Tabular Machine Learning: A Multi-Dataset Benchmark of Bias, Mitigation, and Robustness

## Software environment
- Python 3.8.5 (CPython).
- Pinned package versions are in `requirements.txt`:
  aif360 0.6.1, fairlearn 0.12.0, scikit-learn 1.3.2,
  xgboost 2.1.4, folktables 0.0.12, pandas 2.0.3,
  numpy 1.24.4, scipy 1.10.1, statsmodels 0.14.1,
  matplotlib 3.3.2.

## Datasets (not redistributed here)
- Adult Income and Bank Marketing: UCI Machine Learning Repository.
- ACSIncome: Folktables interface to U.S. Census ACS microdata (2018, California, 1-Year).
- Row counts, targets, positive rates, protected settings, and the UCI Bank SHA-256
  fingerprint are in `data_tables/dataset_provenance_and_fingerprints.csv`.
- UCI Bank reference file: `bank-additional-full.csv` (41,188 rows).
- ACSIncome: California pool subsampled once to 49,999 rows with fixed sample seed 42;
  experiment seeds 42--71 affect train/test splits only.

## Split protocol
- 80/20 target-stratified train/test split per seed.
- 25% of the training partition held out for post-processing calibration under Equalized Odds.
- Seeds: 42--71 (30 consecutive seeds).

## Models and mitigations
- Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, MLP.
- Mitigations on all five families: Baseline, Reweighing, Equalized Odds (AIF360).
- Exponentiated Gradient: Logistic Regression on all datasets; Random Forest on Adult and
  UCI Bank only. Not applied to Gradient Boosting, XGBoost, MLP, or ACSIncome Random Forest.
- Coverage tables: `data_tables/method_scope.csv`, `manifests/configuration_manifest.csv`.

## Run tally (verified)
- 3,690 completed main-grid runs; 0 failed, 0 skipped, 0 duplicate rows.
- Additional robustness outputs (age binning, EO calibration, CFS sensitivity, Adult
  intersectional baseline, 30-seed XGBoost--EG probe) are in `revision_robustness/`;
  they supplement but do not replace the main result files.
- The XGBoost--EG probe (180 runs on three settings; `revision_robustness/xgboost_eg_probe/`)
  is separate from the original 3,690-run main grid and is not merged into main-grid averages.

## How to reproduce analysis outputs
1. Create a Python 3.8.5 environment and `pip install -r requirements.txt`.
2. Obtain raw datasets from the public sources above (not included in this package).
3. Per-seed results: `data_tables/per_seed_full_results_3690.csv`.
4. Post-processing scripts are in `code/` (`phase5a_core.py`, `run_phase5a.py`,
   `build_manuscript_assets.py`, `sam_fair_select.py`, `run_xgboost_eg_probe.py`;
   `pipeline_core.py` is a shared dependency).

## Public archive

- GitHub: https://github.com/NullSpace20/fairness-auditing-public-tabular-benchmarks
- Zenodo (this release): https://doi.org/10.5281/zenodo.21284708
