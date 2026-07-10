"""XGBoost Exponentiated Gradient probe (separate from the 3,690-run main grid).

Feasibility mode: 2 seeds (42-43) on three selected settings.
Full mode: 30 seeds on the same settings.

Settings:
- ACSIncome age_group
- Adult sex
- UCI Bank age_group
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exponentiated_gradient_mitigation import fit_predict_exponentiated_gradient
from fairness_utils import compute_cfs
from loaders import load_acs_income_ca_2018, load_adult_readonly, load_uci_bank_additional
from pipeline_core import (
    fairness_metrics,
    fit_baseline,
    get_acs_protected_specs,
    get_adult_protected_specs,
    get_bank_protected_specs,
    get_non_feature_columns,
    performance_metrics,
    prepare_split,
)

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "xgboost_eg_probe"

SETTINGS = [
    ("acs_income_ca_2018", "age_group", load_acs_income_ca_2018, get_acs_protected_specs, "label"),
    ("adult", "sex", load_adult_readonly, get_adult_protected_specs, "income"),
    ("bank_uci", "age_group", load_uci_bank_additional, get_bank_protected_specs, "label"),
]

CONSTRAINTS = [("DP", "ExponentiatedGradient_DP"), ("EO", "ExponentiatedGradient_EO")]

_DATA_CACHE: dict[str, tuple] = {}


def get_cached_dataset(key: str):
    if key not in _DATA_CACHE:
        loader = next(l for d, _, l, _, _ in SETTINGS if d == key)
        _DATA_CACHE[key] = loader()
    return _DATA_CACHE[key]


def get_spec(key: str, protected: str):
    for ds, prot, _, specs_fn, _ in SETTINGS:
        if ds == key and prot == protected:
            specs = specs_fn()
            return next(s for s in specs if s.name == protected)
    raise KeyError((key, protected))


def run_one(dataset: str, protected: str, constraint: str, seed: int) -> dict:
    target = next(t for d, p, _, _, t in SETTINGS if d == dataset)
    df, _ = get_cached_dataset(dataset)
    spec = get_spec(dataset, protected)
    prot_cols = get_non_feature_columns(
        "adult" if dataset == "adult" else ("bank_uci" if dataset == "bank_uci" else "acs_income")
    )
    split = prepare_split(df, target, prot_cols, spec, seed, with_val=False)
    X_train, X_test, y_train, y_test, prot_train, prot_test = split[:6]

    t0 = time.perf_counter()
    try:
        y_pred, mit_name = fit_predict_exponentiated_gradient(
            X_train,
            y_train,
            prot_train,
            X_test,
            constraint=constraint,
            seed=seed,
            model_name="xgboost",
        )
        status = "success"
        err = ""
    except Exception as exc:
        runtime = time.perf_counter() - t0
        return {
            "dataset": dataset,
            "protected_attribute": protected,
            "model": "xgboost",
            "constraint": constraint,
            "mitigation": f"EG-{constraint}",
            "seed": seed,
            "status": "error",
            "error": str(exc),
            "runtime_seconds": runtime,
        }

    runtime = time.perf_counter() - t0
    fair = fairness_metrics(y_test, y_pred, prot_test, spec)
    perf = performance_metrics(y_test, y_pred)
    cfs = compute_cfs(
        abs(fair["spd"]),
        abs(1 - fair["di"]) if fair["di"] else 0.0,
        abs(fair["eod"]),
        abs(fair["aod"]),
    )
    return {
        "dataset": dataset,
        "protected_attribute": protected,
        "model": "xgboost",
        "constraint": constraint,
        "mitigation": mit_name,
        "seed": seed,
        "accuracy": perf["accuracy"],
        "abs_spd": abs(fair["spd"]),
        "abs_eod": abs(fair["eod"]),
        "cfs": cfs,
        "runtime_seconds": runtime,
        "status": status,
        "error": err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost EG probe (not main grid)")
    parser.add_argument("--feasibility", action="store_true", help="Run seeds 42-43 only")
    parser.add_argument("--seeds", type=str, default="", help="Override seed range, e.g. 42-71")
    args = parser.parse_args()

    if args.seeds:
        lo, hi = map(int, args.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    elif args.feasibility:
        seeds = [42, 43]
    else:
        seeds = list(range(42, 72))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, protected, _, _, _ in SETTINGS:
        get_cached_dataset(dataset)
        for c_code, _ in CONSTRAINTS:
            for seed in seeds:
                rows.append(run_one(dataset, protected, c_code, seed))
                print(
                    dataset,
                    protected,
                    c_code,
                    seed,
                    rows[-1]["status"],
                    f"{rows[-1].get('runtime_seconds', 0):.1f}s",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    tag = "feasibility" if args.feasibility or len(seeds) <= 2 else "full"
    out_csv = OUT_DIR / f"xgboost_eg_probe_{tag}_per_seed.csv"
    df.to_csv(out_csv, index=False)

    ok = df[df["status"] == "success"]
    summary = (
        ok.groupby(["dataset", "protected_attribute", "constraint"])
        .agg(
            n=("seed", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_abs_spd=("abs_spd", "mean"),
            mean_abs_eod=("abs_eod", "mean"),
            mean_cfs=("cfs", "mean"),
            mean_runtime_s=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / f"xgboost_eg_probe_{tag}_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "analysis": "xgboost_eg_probe",
        "seeds": seeds,
        "n_runs": len(df),
        "n_success": int((df["status"] == "success").sum()),
        "n_error": int((df["status"] == "error").sum()),
        "not_part_of_main_grid": True,
    }
    (OUT_DIR / f"xgboost_eg_probe_{tag}_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("Summary:\n", summary.to_string(index=False))
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
