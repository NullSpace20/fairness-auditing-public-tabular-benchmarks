"""Revision R2B: Equalized Odds calibration-split robustness (analysis only).

Compares target-stratified calibration (original Q1 protocol) with
protected-aware joint-stratified calibration where feasible.
Does not modify original Q1 result files.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from fairness_utils import compute_cfs
from loaders import (
    ACS_PHASE4A_SUBSAMPLE_ROWS,
    load_acs_income_ca_2018,
    load_adult_readonly,
    load_uci_bank_additional,
)
from mitigations_aif360 import apply_equalized_odds
from pipeline_core import (
    TEST_SIZE,
    VAL_FRACTION,
    ProtectedSpec,
    build_preprocessor,
    fairness_metrics,
    get_acs_protected_specs,
    get_adult_protected_specs,
    get_bank_protected_specs,
    get_non_feature_columns,
    performance_metrics,
)
from run_phase4b import get_specs

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "revision_R2B_eo_calibration"
FIG_DIR = OUT_DIR / "figures"
ORIGINAL_CSV = (
    Q1_ROOT
    / "results"
    / "phase5_postprocessing"
    / "supplementary"
    / "supp_per_seed_full_results.csv"
)

SEEDS = list(range(42, 72))
MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "mlp",
]

DATASET_PLAN = {
    "adult": {
        "label": "Adult Income",
        "target": "income",
        "non_feature_key": "adult",
        "variant": "kaggle_mirror_train_partition",
        "protected_settings": ["sex", "race", "age_group"],
    },
    "bank_uci": {
        "label": "UCI Bank Marketing",
        "target": "label",
        "non_feature_key": "bank_uci",
        "variant": "uci_bank_additional_full",
        "protected_settings": ["age_group", "job_group"],
    },
    "acs_income_ca_2018": {
        "label": "ACSIncome CA 2018",
        "target": "label",
        "non_feature_key": "acs_income",
        "variant": "acs_ca_2018_stratified",
        "protected_settings": ["sex", "age_group"],
    },
}

# Minimum train-full joint-cell count to attempt protected-aware calibration split.
MIN_JOINT_TRAIN_CELL = 8
SMALL_CELL_THRESHOLDS = (10, 20, 30)

# Classification thresholds (documented in report).
CFS_STABLE = 0.02
CFS_MODERATE = 0.05
ACC_STABLE = 0.005
ACC_MODERATE = 0.02


def joint_label(y: np.ndarray, prot: np.ndarray) -> np.ndarray:
    return np.asarray(y).astype(int).ravel() * 2 + np.asarray(prot).astype(int).ravel()


def prepare_eo_splits(
    df: pd.DataFrame,
    target_col: str,
    protected_cols: List[str],
    spec: ProtectedSpec,
    seed: int,
    calibration_strategy: str,
) -> Dict[str, Any]:
    """Reproduce EO splits; calibration_strategy in {'target_stratified', 'protected_aware'}."""
    df_run = df
    if spec.filter_column and spec.allowed_filter_values:
        df_run = df[df[spec.filter_column].isin(spec.allowed_filter_values)].copy()

    feature_cols = [c for c in df_run.columns if c not in [target_col] + protected_cols]
    X = df_run[feature_cols]
    y = df_run[target_col].values
    protected = df_run[spec.column].values.astype(int)

    X_train_full, X_test, y_train_full, y_test, prot_train_full, prot_test = train_test_split(
        X, y, protected, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    preprocessor = build_preprocessor(pd.concat([X_train_full, X_test]), feature_cols)
    X_train_full_p = preprocessor.fit_transform(X_train_full)
    X_test_p = preprocessor.transform(X_test)

    joint_train_full = joint_label(y_train_full, prot_train_full)
    joint_counts_train_full = {
        int(k): int(v) for k, v in zip(*np.unique(joint_train_full, return_counts=True))
    }
    all_four = all(joint_counts_train_full.get(i, 0) > 0 for i in range(4))
    min_joint_train = min(joint_counts_train_full.get(i, 0) for i in range(4)) if all_four else 0
    joint_feasible = all_four and min_joint_train >= MIN_JOINT_TRAIN_CELL

    used_strategy = calibration_strategy
    fallback_reason = ""
    if calibration_strategy == "protected_aware" and not joint_feasible:
        used_strategy = "target_stratified_fallback"
        fallback_reason = (
            f"joint stratification infeasible: min_joint_train_cell={min_joint_train}, "
            f"all_four_present={all_four}"
        )

    if used_strategy in ("target_stratified", "target_stratified_fallback"):
        stratify_cal = y_train_full
    else:
        stratify_cal = joint_train_full

    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X_train_full_p,
        y_train_full,
        prot_train_full,
        test_size=VAL_FRACTION,
        stratify=stratify_cal,
        random_state=seed,
    )

    cal_counts = calibration_cell_counts(y_val, prot_val, spec)
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test_p,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "prot_train": prot_train,
        "prot_val": prot_val,
        "prot_test": prot_test,
        "n_rows": len(df_run),
        "calibration_strategy_requested": calibration_strategy,
        "calibration_strategy_used": used_strategy,
        "joint_feasible": joint_feasible,
        "fallback_reason": fallback_reason,
        "joint_counts_train_full": joint_counts_train_full,
        "min_joint_train_cell": min_joint_train,
        **cal_counts,
    }


def calibration_cell_counts(
    y_val: np.ndarray, prot_val: np.ndarray, spec: ProtectedSpec
) -> Dict[str, Any]:
    y_val = np.asarray(y_val).astype(int).ravel()
    prot_val = np.asarray(prot_val).astype(int).ravel()
    priv = spec.privileged_value
    unpriv = spec.unprivileged_value

    def count_mask(mask: np.ndarray) -> int:
        return int(mask.sum())

    priv_mask = prot_val == priv
    unpriv_mask = prot_val == unpriv
    pos_mask = y_val == 1
    neg_mask = y_val == 0

    cells = {
        "cal_priv_pos": count_mask(priv_mask & pos_mask),
        "cal_priv_neg": count_mask(priv_mask & neg_mask),
        "cal_unpriv_pos": count_mask(unpriv_mask & pos_mask),
        "cal_unpriv_neg": count_mask(unpriv_mask & neg_mask),
        "cal_priv_total": count_mask(priv_mask),
        "cal_unpriv_total": count_mask(unpriv_mask),
        "cal_pos_total": count_mask(pos_mask),
        "cal_neg_total": count_mask(neg_mask),
        "cal_total": int(len(y_val)),
    }
    joint_cells = [cells["cal_priv_pos"], cells["cal_priv_neg"],
                   cells["cal_unpriv_pos"], cells["cal_unpriv_neg"]]
    cells["cal_min_joint_cell"] = int(min(joint_cells))
    cells["cal_min_group_total"] = int(min(cells["cal_priv_total"], cells["cal_unpriv_total"]))
    for thr in SMALL_CELL_THRESHOLDS:
        cells[f"flag_joint_lt_{thr}"] = cells["cal_min_joint_cell"] < thr
        cells[f"flag_group_lt_{thr}"] = cells["cal_min_group_total"] < thr
    return cells


def run_eo_config(
    split_pack: Dict[str, Any],
    model: str,
    spec: ProtectedSpec,
    seed: int,
) -> Dict[str, float]:
    y_pred = apply_equalized_odds(
        model,
        split_pack["X_train"],
        split_pack["y_train"],
        split_pack["prot_train"],
        split_pack["X_val"],
        split_pack["y_val"],
        split_pack["prot_val"],
        split_pack["X_test"],
        split_pack["prot_test"],
        spec,
        seed,
    )
    perf = performance_metrics(split_pack["y_test"], y_pred)
    fair = fairness_metrics(
        split_pack["y_test"], y_pred, split_pack["prot_test"], spec
    )
    perf.update(fair)
    perf["cfs"] = compute_cfs(perf["spd"], perf["di"], perf["eod"], perf["aod"])
    perf["abs_spd"] = abs(perf["spd"])
    perf["abs_eod"] = abs(perf["eod"])
    perf["abs_aod"] = abs(perf["aod"])
    perf["abs_di_violation"] = abs(1.0 - perf["di"])
    return perf


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    adult_df, _ = load_adult_readonly()
    bank_df, _ = load_uci_bank_additional()
    if bank_df is None:
        raise RuntimeError("UCI Bank not found")
    acs_df, _ = load_acs_income_ca_2018(max_rows=ACS_PHASE4A_SUBSAMPLE_ROWS)
    if acs_df is None:
        raise RuntimeError("ACSIncome not found")
    return {"adult": adult_df, "bank_uci": bank_df, "acs_income_ca_2018": acs_df}


def spec_by_name(dataset: str, name: str) -> ProtectedSpec:
    for s in get_specs(dataset):
        if s.name == name:
            return s
    raise KeyError(f"{name} not in {dataset}")


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd else np.nan


def bootstrap_ci(diff: np.ndarray, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return (np.nan, np.nan)
    boots = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def classify_setting(summary_row: pd.Series, feasible: bool) -> str:
    if not feasible:
        return "not feasible"
    cfs_d = abs(summary_row["mean_cfs_diff_protected_minus_target"])
    acc_d = abs(summary_row["mean_accuracy_diff_protected_minus_target"])
    if cfs_d < CFS_STABLE and acc_d < ACC_STABLE:
        return "stable"
    if cfs_d < CFS_MODERATE and acc_d < ACC_MODERATE:
        return "moderately sensitive"
    return "sensitive"


def make_figures(
    count_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)

    # Heatmap: min calibration joint cell by dataset × protected (median over seeds)
    hm = (
        count_summary.groupby(["dataset_label", "protected_attribute"])["cal_min_joint_cell_median"]
        .mean()
        .reset_index()
        .pivot(index="dataset_label", columns="protected_attribute", values="cal_min_joint_cell_median")
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd_r", ax=ax,
                cbar_kws={"label": "Median min joint cell count (cal set)"})
    ax.set_title("EO calibration: smallest protected×target cell\n(target-stratified split; median over seeds)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_eo_cal_min_cell_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # CFS comparison: original vs protected-aware
    plot_df = comparison_summary.copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    x_labels = plot_df["dataset_label"] + " / " + plot_df["protected_attribute"]
    plot_df = plot_df.assign(x_label=x_labels)
    idx = np.arange(len(plot_df))
    w = 0.35
    ax.bar(idx - w / 2, plot_df["mean_cfs_target_stratified"], w, label="Target-stratified cal.")
    ax.bar(idx + w / 2, plot_df["mean_cfs_protected_aware"], w, label="Protected-aware cal.")
    ax.set_xticks(idx)
    ax.set_xticklabels(plot_df["x_label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean CFS (model-averaged)")
    ax.set_title("Equalized Odds: calibration strategy comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_eo_cal_cfs_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    t0 = time.perf_counter()

    count_rows: List[Dict[str, Any]] = []
    eo_rows: List[Dict[str, Any]] = []

    total_eo = (
        len(DATASET_PLAN) * sum(len(v["protected_settings"]) for v in DATASET_PLAN.values())
        * len(MODELS) * len(SEEDS) * 2
    )
    done = 0

    for dataset_key, plan in DATASET_PLAN.items():
        df = datasets[dataset_key]
        target_col = plan["target"]
        non_feature = get_non_feature_columns(plan["non_feature_key"])

        for protected_name in plan["protected_settings"]:
            spec = spec_by_name(dataset_key, protected_name)

            for seed in SEEDS:
                split_target = prepare_eo_splits(
                    df, target_col, non_feature, spec, seed, "target_stratified"
                )
                split_prot = prepare_eo_splits(
                    df, target_col, non_feature, spec, seed, "protected_aware"
                )

                for model in MODELS:
                    count_rows.append({
                        "dataset": dataset_key,
                        "dataset_label": plan["label"],
                        "protected_attribute": protected_name,
                        "model": model,
                        "seed": seed,
                        "calibration_strategy_used": split_target["calibration_strategy_used"],
                        "joint_feasible": split_target["joint_feasible"],
                        "min_joint_train_cell": split_target["min_joint_train_cell"],
                        **{k: split_target[k] for k in split_target if k.startswith("cal_") or k.startswith("flag_")},
                    })

                    for strategy_key, split_pack in (
                        ("target_stratified", split_target),
                        ("protected_aware", split_prot),
                    ):
                        metrics = run_eo_config(split_pack, model, spec, seed)
                        eo_rows.append({
                            "dataset": dataset_key,
                            "dataset_label": plan["label"],
                            "dataset_variant": plan["variant"],
                            "protected_attribute": protected_name,
                            "model": model,
                            "seed": seed,
                            "calibration_strategy": strategy_key,
                            "calibration_strategy_used": split_pack["calibration_strategy_used"],
                            "joint_feasible": split_pack["joint_feasible"],
                            "fallback_reason": split_pack.get("fallback_reason", ""),
                            "cal_min_joint_cell": split_pack["cal_min_joint_cell"],
                            "status": "success",
                            **metrics,
                        })
                        done += 1
                        if done % 100 == 0:
                            pd.DataFrame(eo_rows).to_csv(
                                OUT_DIR / "eo_calibration_robustness_per_seed_partial.csv", index=False
                            )
                            print(f"  EO runs {done}/{total_eo}", flush=True)

    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(OUT_DIR / "eo_calibration_group_counts_per_seed.csv", index=False)

    count_summary = (
        count_df.groupby(["dataset", "dataset_label", "protected_attribute"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            cal_min_joint_cell_min=("cal_min_joint_cell", "min"),
            cal_min_joint_cell_median=("cal_min_joint_cell", "median"),
            cal_min_joint_cell_mean=("cal_min_joint_cell", "mean"),
            cal_min_joint_cell_p5=("cal_min_joint_cell", lambda s: float(np.percentile(s, 5))),
            cal_min_group_total_min=("cal_min_group_total", "min"),
            cal_min_group_total_median=("cal_min_group_total", "median"),
            cal_priv_pos_min=("cal_priv_pos", "min"),
            cal_unpriv_pos_min=("cal_unpriv_pos", "min"),
            cal_unpriv_neg_min=("cal_unpriv_neg", "min"),
            pct_seeds_joint_lt_10=("flag_joint_lt_10", "mean"),
            pct_seeds_joint_lt_20=("flag_joint_lt_20", "mean"),
            pct_seeds_joint_lt_30=("flag_joint_lt_30", "mean"),
            joint_feasible_all_seeds=("joint_feasible", "min"),
        )
    )
    count_summary.to_csv(OUT_DIR / "eo_calibration_group_counts_summary.csv", index=False)

    eo_df = pd.DataFrame(eo_rows)
    partial = OUT_DIR / "eo_calibration_robustness_per_seed_partial.csv"
    if partial.exists():
        partial.unlink()
    eo_df.to_csv(OUT_DIR / "eo_calibration_robustness_per_seed.csv", index=False)

    metrics = ["accuracy", "cfs", "abs_spd", "abs_eod", "abs_aod"]
    agg_rows = []
    for keys, grp in eo_df.groupby(
        ["dataset", "dataset_label", "protected_attribute", "model", "calibration_strategy"]
    ):
        row = dict(zip(
            ["dataset", "dataset_label", "protected_attribute", "model", "calibration_strategy"], keys
        ))
        for m in metrics:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_sd"] = grp[m].std(ddof=1)
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(OUT_DIR / "eo_calibration_robustness_aggregate.csv", index=False)

    # Comparison summary
    comp_rows = []
    stat_rows = []
    for dataset_key, plan in DATASET_PLAN.items():
        for protected_name in plan["protected_settings"]:
            for model in MODELS:
                sub_t = eo_df[
                    (eo_df["dataset"] == dataset_key)
                    & (eo_df["protected_attribute"] == protected_name)
                    & (eo_df["model"] == model)
                    & (eo_df["calibration_strategy"] == "target_stratified")
                ].set_index("seed")
                sub_p = eo_df[
                    (eo_df["dataset"] == dataset_key)
                    & (eo_df["protected_attribute"] == protected_name)
                    & (eo_df["model"] == model)
                    & (eo_df["calibration_strategy"] == "protected_aware")
                ].set_index("seed")
                joined = sub_t.join(sub_p, lsuffix="_t", rsuffix="_p", how="inner")
                feasible = bool(
                    (sub_p["calibration_strategy_used"] == "protected_aware").all()
                )
                row = {
                    "dataset": dataset_key,
                    "dataset_label": plan["label"],
                    "protected_attribute": protected_name,
                    "model": model,
                    "joint_feasible_all_seeds": feasible,
                    "n_pairs": len(joined),
                }
                for m in metrics:
                    diff = joined[f"{m}_p"] - joined[f"{m}_t"]
                    row[f"mean_{m}_diff_protected_minus_target"] = float(diff.mean())
                    row[f"sd_{m}_diff"] = float(diff.std(ddof=1))
                    row[f"mean_{m}_target_stratified"] = float(joined[f"{m}_t"].mean())
                    row[f"mean_{m}_protected_aware"] = float(joined[f"{m}_p"].mean())
                    ci_lo, ci_hi = bootstrap_ci(diff.values, seed=hash((dataset_key, protected_name, model, m)) % 10000)
                    stat_rows.append({
                        "dataset": dataset_key,
                        "protected_attribute": protected_name,
                        "model": model,
                        "metric": m,
                        "n_pairs": len(diff),
                        "mean_diff_protected_minus_target": float(diff.mean()),
                        "sd_diff": float(diff.std(ddof=1)),
                        "cohen_dz": cohen_dz(diff.values),
                        "bootstrap_ci_low": ci_lo,
                        "bootstrap_ci_high": ci_hi,
                        "joint_feasible_all_seeds": feasible,
                    })
                comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "eo_calibration_comparison_summary.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(OUT_DIR / "eo_calibration_stat_tests.csv", index=False)

    # Setting-level classification (averaged over models)
    class_rows = []
    for (dataset_key, protected_name), grp in comp_df.groupby(["dataset", "protected_attribute"]):
        feasible = bool(grp["joint_feasible_all_seeds"].all())
        avg = {
            "dataset": dataset_key,
            "dataset_label": grp["dataset_label"].iloc[0],
            "protected_attribute": protected_name,
            "joint_feasible_all_seeds": feasible,
            "mean_cfs_diff_protected_minus_target": grp["mean_cfs_diff_protected_minus_target"].mean(),
            "mean_accuracy_diff_protected_minus_target": grp["mean_accuracy_diff_protected_minus_target"].mean(),
            "mean_abs_spd_diff_protected_minus_target": grp["mean_abs_spd_diff_protected_minus_target"].mean(),
            "mean_abs_eod_diff_protected_minus_target": grp["mean_abs_eod_diff_protected_minus_target"].mean(),
            "max_abs_cfs_diff_any_model": grp["mean_cfs_diff_protected_minus_target"].abs().max(),
        }
        avg["classification"] = classify_setting(pd.Series(avg), feasible)
        cs = count_summary[
            (count_summary["dataset"] == dataset_key)
            & (count_summary["protected_attribute"] == protected_name)
        ]
        if not cs.empty:
            avg["cal_min_joint_cell_median"] = float(cs["cal_min_joint_cell_median"].iloc[0])
            avg["cal_min_joint_cell_p5"] = float(cs["cal_min_joint_cell_p5"].iloc[0])
            avg["pct_seeds_joint_lt_10"] = float(cs["pct_seeds_joint_lt_10"].iloc[0])
        class_rows.append(avg)
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(OUT_DIR / "eo_calibration_robustness_classification.csv", index=False)

    # Validate original EO vs Q1
    validation_rows = []
    if ORIGINAL_CSV.exists():
        orig = pd.read_csv(ORIGINAL_CSV)
        orig_eo = orig[orig["mitigation"] == "equalized_odds"]
        new_t = eo_df[eo_df["calibration_strategy"] == "target_stratified"]
        merged = new_t.merge(
            orig_eo, on=["dataset", "model", "protected_attribute", "seed"], suffixes=("_new", "_orig")
        )
        for col in ["accuracy", "cfs", "spd", "eod", "aod"]:
            d = merged[f"{col}_new"] - merged[f"{col}_orig"]
            validation_rows.append({
                "metric": col,
                "n_pairs": len(d),
                "max_abs_diff": float(d.abs().max()),
                "mean_abs_diff": float(d.abs().mean()),
            })
        pd.DataFrame(validation_rows).to_csv(
            OUT_DIR / "eo_calibration_original_validation.csv", index=False
        )

    # Setting-level comparison for figures (model-averaged CFS)
    fig_comp = (
        comp_df.groupby(["dataset", "dataset_label", "protected_attribute"], as_index=False)
        .agg(
            mean_cfs_target_stratified=("mean_cfs_target_stratified", "mean"),
            mean_cfs_protected_aware=("mean_cfs_protected_aware", "mean"),
            mean_cfs_diff=("mean_cfs_diff_protected_minus_target", "mean"),
        )
    )
    make_figures(count_summary, fig_comp)

    meta = {
        "phase": "R2B",
        "n_eo_runs": len(eo_df),
        "n_success": int((eo_df["status"] == "success").sum()),
        "wall_seconds": time.perf_counter() - t0,
        "min_joint_train_cell_threshold": MIN_JOINT_TRAIN_CELL,
        "classification_thresholds": {
            "cfs_stable": CFS_STABLE,
            "cfs_moderate": CFS_MODERATE,
            "acc_stable": ACC_STABLE,
            "acc_moderate": ACC_MODERATE,
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
