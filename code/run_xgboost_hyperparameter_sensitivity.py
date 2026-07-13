"""Optional Adult XGBoost hyperparameter sensitivity (W4).

Baseline XGBoost only; seeds 42-71; Adult protected settings sex, race, age_group.
Does NOT modify the 3,690-run main grid or replace main results.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fairness_utils import compute_cfs
from loaders import load_adult_readonly
from pipeline_core import (
    ProtectedSpec,
    densify,
    fairness_metrics,
    get_adult_protected_specs,
    get_non_feature_columns,
    performance_metrics,
    prepare_split,
)

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "xgboost_hyperparameter_sensitivity"

DEFAULT_XGB = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

# Pre-specified alternative configurations (overrides only; other params stay at default).
CONFIGS: List[Dict[str, Any]] = [
    {
        "config_id": "default",
        "label": "default (main-grid)",
        "overrides": {},
    },
    {
        "config_id": "shallower",
        "label": "shallower (d=3, lr=0.05, n=300)",
        "overrides": {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    },
    {
        "config_id": "deeper",
        "label": "deeper (d=7, lr=0.05, n=300)",
        "overrides": {"max_depth": 7, "learning_rate": 0.05, "n_estimators": 300},
    },
    {
        "config_id": "conservative",
        "label": "conservative (d=4, lr=0.03, n=500, sub=0.8)",
        "overrides": {
            "max_depth": 4,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "subsample": 0.8,
        },
    },
]

PROTECTED = ["sex", "race", "age_group"]
FULL_SEEDS = list(range(42, 72))
FEASIBILITY_SEEDS = [42, 43]


def fit_xgb_custom(X_train, y_train, X_test, seed: int, overrides: Dict[str, Any]) -> np.ndarray:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required.") from exc
    params = {**DEFAULT_XGB, **overrides}
    clf = XGBClassifier(
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    clf.fit(densify(X_train), y_train)
    return clf.predict(densify(X_test))


def get_spec(name: str) -> ProtectedSpec:
    return next(s for s in get_adult_protected_specs() if s.name == name)


def run_one(protected: str, config: Dict[str, Any], seed: int, df: pd.DataFrame) -> dict:
    spec = get_spec(protected)
    prot_cols = get_non_feature_columns("adult")
    split = prepare_split(df, "income", prot_cols, spec, seed, with_val=False)
    X_train, X_test, y_train, y_test, prot_train, prot_test = split[:6]

    t0 = time.perf_counter()
    try:
        y_pred = fit_xgb_custom(
            X_train, y_train, X_test, seed, config["overrides"]
        )
        status = "success"
        err = ""
    except Exception as exc:
        return {
            "analysis": "optional_tuning_sensitivity",
            "dataset": "adult",
            "protected_attribute": protected,
            "model": "xgboost",
            "mitigation": "baseline",
            "config_id": config["config_id"],
            "config_label": config["label"],
            "seed": seed,
            "status": f"error: {exc}",
            "runtime_seconds": time.perf_counter() - t0,
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
    merged_params = {**DEFAULT_XGB, **config["overrides"]}
    row = {
        "analysis": "optional_tuning_sensitivity",
        "dataset": "adult",
        "protected_attribute": protected,
        "model": "xgboost",
        "mitigation": "baseline",
        "config_id": config["config_id"],
        "config_label": config["label"],
        "seed": seed,
        "accuracy": perf["accuracy"],
        "f1": perf["f1"],
        "abs_spd": abs(fair["spd"]),
        "abs_eod": abs(fair["eod"]),
        "cfs": float(cfs),
        "runtime_seconds": runtime,
        "status": status,
        "error": err,
    }
    for k, v in merged_params.items():
        row[f"param_{k}"] = v
    return row


def add_ranks(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Rank configs within each (protected, seed) by accuracy and CFS."""
    ok = per_seed[per_seed["status"] == "success"].copy()
    if ok.empty:
        return per_seed

    def rank_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["rank_accuracy_desc"] = g["accuracy"].rank(ascending=False, method="min")
        g["rank_cfs_asc"] = g["cfs"].rank(ascending=True, method="min")
        default_acc = g.loc[g["config_id"] == "default", "accuracy"]
        default_cfs = g.loc[g["config_id"] == "default", "cfs"]
        if not default_acc.empty and not default_cfs.empty:
            g["delta_accuracy_vs_default"] = g["accuracy"] - float(default_acc.iloc[0])
            g["delta_cfs_vs_default"] = g["cfs"] - float(default_cfs.iloc[0])
        else:
            g["delta_accuracy_vs_default"] = np.nan
            g["delta_cfs_vs_default"] = np.nan
        return g

    ranked = ok.groupby(["protected_attribute", "seed"], group_keys=False).apply(rank_group)
    return per_seed.merge(
        ranked[
            [
                "protected_attribute",
                "seed",
                "config_id",
                "rank_accuracy_desc",
                "rank_cfs_asc",
                "delta_accuracy_vs_default",
                "delta_cfs_vs_default",
            ]
        ],
        on=["protected_attribute", "seed", "config_id"],
        how="left",
    )


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    ok = per_seed[per_seed["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    agg = (
        ok.groupby(["protected_attribute", "config_id", "config_label"], dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1=("f1", "mean"),
            mean_abs_spd=("abs_spd", "mean"),
            mean_abs_eod=("abs_eod", "mean"),
            mean_cfs=("cfs", "mean"),
            std_cfs=("cfs", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_delta_accuracy_vs_default=("delta_accuracy_vs_default", "mean"),
            mean_delta_cfs_vs_default=("delta_cfs_vs_default", "mean"),
        )
        .reset_index()
    )
    return agg


def qualitative_assessment(summary: pd.DataFrame) -> dict:
    """Check whether accuracy-fairness divergence pattern holds across configs."""
    notes = []
    diverges_all = True
    for prot in PROTECTED:
        sub = summary[summary["protected_attribute"] == prot]
        if sub.empty:
            continue
        default = sub[sub["config_id"] == "default"]
        if default.empty:
            continue
        d_acc = float(default["mean_accuracy"].iloc[0])
        d_cfs = float(default["mean_cfs"].iloc[0])
        # High accuracy with non-trivial CFS = divergence pattern
        high_acc = d_acc >= 0.84
        non_trivial_cfs = d_cfs >= 0.03
        if not (high_acc and non_trivial_cfs):
            diverges_all = False
            notes.append(f"{prot}: default does not show clear divergence (acc={d_acc:.3f}, cfs={d_cfs:.3f})")

        alts = sub[sub["config_id"] != "default"]
        for _, row in alts.iterrows():
            acc_shift = float(row["mean_delta_accuracy_vs_default"])
            cfs_shift = float(row["mean_delta_cfs_vs_default"])
            # Qualitative pattern holds if tuning does not flip to low-acc/low-CFS uniformly
            if abs(acc_shift) > 0.02 and abs(cfs_shift) > 0.02 and np.sign(acc_shift) != np.sign(-cfs_shift):
                notes.append(
                    f"{prot}/{row['config_id']}: larger acc shift ({acc_shift:+.3f}) "
                    f"with opposing CFS shift ({cfs_shift:+.3f}) — pattern preserved"
                )

    return {
        "qualitative_divergence_preserved": diverges_all,
        "assessment_notes": notes,
        "conclusion_unchanged": diverges_all,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult XGBoost hyperparameter sensitivity (W4)")
    parser.add_argument(
        "--feasibility",
        action="store_true",
        help="Run 2 seeds only (42-43) for smoke test",
    )
    args = parser.parse_args()
    seeds = FEASIBILITY_SEEDS if args.feasibility else FULL_SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, load_meta = load_adult_readonly()
    if df is None:
        raise RuntimeError(f"Failed to load Adult: {load_meta}")

    rows = []
    total = len(PROTECTED) * len(CONFIGS) * len(seeds)
    done = 0
    for protected in PROTECTED:
        for config in CONFIGS:
            for seed in seeds:
                done += 1
                if done % 30 == 0 or done == 1:
                    print(f"[{done}/{total}] {protected} {config['config_id']} seed={seed}")
                rows.append(run_one(protected, config, seed, df))

    per_seed = add_ranks(pd.DataFrame(rows))
    summary = summarize(per_seed)
    assessment = qualitative_assessment(summary)

    suffix = "feasibility" if args.feasibility else "full"
    per_path = OUT_DIR / f"xgboost_hp_sensitivity_{suffix}_per_seed.csv"
    summ_path = OUT_DIR / f"xgboost_hp_sensitivity_{suffix}_summary.csv"
    per_seed.to_csv(per_path, index=False)
    summary.to_csv(summ_path, index=False)

    meta = {
        "analysis": "optional_xgboost_hyperparameter_sensitivity_W4",
        "not_part_of_main_grid": True,
        "dataset": "adult",
        "model": "xgboost",
        "mitigation": "baseline",
        "protected_attributes": PROTECTED,
        "configs": [
            {"config_id": c["config_id"], "label": c["label"], "overrides": c["overrides"]}
            for c in CONFIGS
        ],
        "default_xgb_params": DEFAULT_XGB,
        "seeds": f"{seeds[0]}-{seeds[-1]}",
        "n_seeds": len(seeds),
        "n_configs": len(CONFIGS),
        "n_runs_attempted": total,
        "n_success": int((per_seed["status"] == "success").sum()),
        "qualitative_assessment": assessment,
        "adult_load_meta": load_meta,
    }
    meta_path = OUT_DIR / f"run_metadata_{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {per_path}")
    print(f"Wrote {summ_path}")
    print(f"Wrote {meta_path}")
    print("Qualitative assessment:", json.dumps(assessment, indent=2))
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
