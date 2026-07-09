# Phase 4B Runtime Fact Sheet

Generated: 2026-07-08T01:29:33.887309+00:00

## Total runtime
- Wall clock: 24.00 h (86418 s)

## required
- Sum runtime: 10.23 h (36829 s)
- Configurations: 1800

## optional_main_grid_extension
- Sum runtime: 13.70 h (49331 s)
- Configurations: 1293

## Mean runtime by dataset
- acs_income_ca_2018: 5.66 s
- adult: 30.03 s
- bank_uci: 34.89 s

## Mean runtime by model
- gradient_boosting: 12.27 s
- logistic_regression: 3.34 s
- mlp: 104.06 s
- random_forest: 44.43 s
- xgboost: 6.47 s

## Mean runtime by mitigation
- ExponentiatedGradient_DP: 49.10 s
- ExponentiatedGradient_EO: 64.36 s
- baseline: 13.81 s
- equalized_odds: 20.25 s
- reweighing: 31.44 s

## LR-EG
- Mean: 7.50 s
- Sum: 0.88 h (3151 s)

## RF-EG
- Mean: 229.02 s
- Sum: 7.63 h (27483 s)
