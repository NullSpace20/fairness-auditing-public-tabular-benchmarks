"""Revision R2A: baseline age-binning robustness (analysis only).

Reruns baseline models under three age codings on the Q1-upgrade datasets,
using seeds 42-71 and the same preprocessing/split protocol as phase 4B.
Does not modify original Q1 result files.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fairness_utils import compute_cfs
from loaders import (
    ACS_PHASE4A_SUBSAMPLE_ROWS,
    load_acs_income_ca_2018,
    load_adult_readonly,
    load_uci_bank_additional,
)
from pipeline_core import (
    ProtectedSpec,
    fairness_metrics,
    fit_baseline,
    get_non_feature_columns,
    performance_metrics,
    prepare_split,
)

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "revision_R2_age_robustness"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = Q1_ROOT / "reports" / "revision_R2A"
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

DATASETS = {
    "adult": {
        "label": "Adult Income",
        "target": "income",
        "non_feature_key": "adult",
        "variant": "kaggle_mirror_train_partition",
        "age_source": "age (int, UCI Adult training partition)",
    },
    "bank_uci": {
        "label": "UCI Bank Marketing",
        "target": "label",
        "non_feature_key": "bank_uci",
        "variant": "uci_bank_additional_full",
        "age_source": "age (int, canonical UCI bank-additional-full.csv)",
    },
    "acs_income_ca_2018": {
        "label": "ACSIncome CA 2018",
        "target": "label",
        "non_feature_key": "acs_income",
        "variant": "acs_ca_2018_stratified",
        "age_source": "AGEP (int, Folktables ACSIncome California 2018)",
    },
}

AGE_CODING_META = {
    "middle_vs_rest": {
        "column": "age_group",
        "privileged_rule": "age 31-60",
        "unprivileged_rule": "age <=30 or age >=61",
        "privileged_value": 1,
        "unprivileged_value": 0,
    },
    "young_vs_rest": {
        "column": "young_vs_rest",
        "privileged_rule": "age >30",
        "unprivileged_rule": "age <=30",
        "privileged_value": 1,
        "unprivileged_value": 0,
    },
    "older_vs_rest": {
        "column": "older_vs_rest",
        "privileged_rule": "age <61",
        "unprivileged_rule": "age >=61",
        "privileged_value": 1,
        "unprivileged_value": 0,
    },
}


def add_alternative_age_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bands = out["_age_band"]
    out["young_vs_rest"] = np.where(bands == "young", 0, 1)
    out["older_vs_rest"] = np.where(bands == "old", 0, 1)
    return out


def protected_spec_for_coding(coding: str) -> ProtectedSpec:
    meta = AGE_CODING_META[coding]
    return ProtectedSpec(
        name=coding,
        column=meta["column"],
        privileged_value=meta["privileged_value"],
        unprivileged_value=meta["unprivileged_value"],
        description=f"{meta['privileged_rule']} privileged",
    )


def load_all_datasets() -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    loaded: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {}
    adult_df, adult_meta = load_adult_readonly()
    loaded["adult"] = (add_alternative_age_columns(adult_df), adult_meta)
    bank_df, bank_meta = load_uci_bank_additional()
    if bank_df is None:
        raise RuntimeError("UCI Bank dataset not found.")
    loaded["bank_uci"] = (add_alternative_age_columns(bank_df), bank_meta)
    acs_df, acs_meta = load_acs_income_ca_2018(max_rows=ACS_PHASE4A_SUBSAMPLE_ROWS)
    if acs_df is None:
        raise RuntimeError("ACSIncome dataset not found (folktables required).")
    loaded["acs_income_ca_2018"] = (add_alternative_age_columns(acs_df), acs_meta)
    return loaded


def build_group_counts(loaded: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset_key, (df, meta) in loaded.items():
        info = DATASETS[dataset_key]
        pos_rate = float(df[info["target"]].mean())
        for coding, meta_c in AGE_CODING_META.items():
            col = meta_c["column"]
            vc = df[col].value_counts().to_dict()
            priv = int(vc.get(meta_c["privileged_value"], 0))
            unpriv = int(vc.get(meta_c["unprivileged_value"], 0))
            rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_label": info["label"],
                    "age_coding": coding,
                    "age_column": col,
                    "age_source": info["age_source"],
                    "n_rows": int(len(df)),
                    "target": info["target"],
                    "positive_rate": pos_rate,
                    "privileged_count": priv,
                    "unprivileged_count": unpriv,
                    "privileged_share": priv / len(df) if len(df) else np.nan,
                    "unprivileged_share": unpriv / len(df) if len(df) else np.nan,
                    "privileged_rule": meta_c["privileged_rule"],
                    "unprivileged_rule": meta_c["unprivileged_rule"],
                    "loader_meta_json": json.dumps(
                        {k: meta.get(k) for k in ("status", "source_path", "sha256", "positive_rate", "final_sample_rows", "full_ca_rows_before_subsample")},
                        default=str,
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_baseline_row(
    dataset_key: str,
    df: pd.DataFrame,
    model: str,
    coding: str,
    seed: int,
) -> Dict[str, Any]:
    info = DATASETS[dataset_key]
    spec = protected_spec_for_coding(coding)
    extra_cols = ["young_vs_rest", "older_vs_rest"]
    non_feature = get_non_feature_columns(info["non_feature_key"]) + extra_cols
    target_col = info["target"]

    t0 = time.perf_counter()
    try:
        X_tr, X_te, y_tr, y_te, p_tr, p_te, n_rows = prepare_split(
            df, target_col, non_feature, spec, seed, with_val=False
        )
        y_pred = fit_baseline(model, X_tr, y_tr, X_te, seed)
        perf = performance_metrics(y_te, y_pred)
        fair = fairness_metrics(y_te, y_pred, p_te, spec)
        perf.update(fair)
        perf["cfs"] = compute_cfs(perf["spd"], perf["di"], perf["eod"], perf["aod"])
        perf["abs_spd"] = abs(perf["spd"])
        perf["abs_eod"] = abs(perf["eod"])
        perf["abs_aod"] = abs(perf["aod"])
        perf["abs_di_violation"] = abs(1.0 - perf["di"])
        status = "success"
        err = ""
    except Exception as exc:
        perf = {}
        n_rows = 0
        status = "failed"
        err = str(exc)

    return {
        "dataset": dataset_key,
        "dataset_label": info["label"],
        "dataset_variant": info["variant"],
        "model": model,
        "mitigation": "baseline",
        "age_coding": coding,
        "protected_attribute": coding,
        "seed": seed,
        "n_rows": n_rows,
        "runtime_seconds": time.perf_counter() - t0,
        "status": status,
        "error_message": err,
        **{k: perf.get(k) for k in (
            "accuracy", "precision", "recall", "f1",
            "spd", "di", "eod", "aod", "cfs",
            "abs_spd", "abs_eod", "abs_aod", "abs_di_violation",
        )},
        "analysis_method": "baseline_rerun_same_protocol",
    }


def aggregate_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    ok = per_seed[per_seed["status"] == "success"].copy()
    metrics = ["accuracy", "precision", "recall", "f1", "spd", "di", "eod", "aod",
               "cfs", "abs_spd", "abs_eod", "abs_aod", "abs_di_violation"]
    rows = []
    for keys, grp in ok.groupby(["dataset", "dataset_label", "model", "age_coding"], sort=True):
        row = dict(zip(["dataset", "dataset_label", "model", "age_coding"], keys))
        row["n_seeds"] = len(grp)
        row["n_rows_mean"] = grp["n_rows"].mean()
        for m in metrics:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_sd"] = grp[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(aggregate: pd.DataFrame, group_counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in aggregate["dataset"].unique():
        for model in MODELS:
            sub = aggregate[(aggregate["dataset"] == dataset) & (aggregate["model"] == model)]
            if sub.empty:
                continue
            gc = group_counts[(group_counts["dataset"] == dataset)]
            base = {
                "dataset": dataset,
                "dataset_label": sub["dataset_label"].iloc[0],
                "model": model,
            }
            for coding in AGE_CODING_META:
                s = sub[sub["age_coding"] == coding]
                if s.empty:
                    continue
                r = s.iloc[0]
                gc_row = gc[gc["age_coding"] == coding].iloc[0]
                rows.append(
                    {
                        **base,
                        "age_coding": coding,
                        "privileged_count": gc_row["privileged_count"],
                        "unprivileged_count": gc_row["unprivileged_count"],
                        "mean_abs_spd": r["abs_spd_mean"],
                        "sd_abs_spd": r["abs_spd_sd"],
                        "mean_abs_eod": r["abs_eod_mean"],
                        "sd_abs_eod": r["abs_eod_sd"],
                        "mean_abs_aod": r["abs_aod_mean"],
                        "sd_abs_aod": r["abs_aod_sd"],
                        "mean_cfs": r["cfs_mean"],
                        "sd_cfs": r["cfs_sd"],
                        "mean_accuracy": r["accuracy_mean"],
                    }
                )
    return pd.DataFrame(rows)


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(diff.mean() / sd)


def bootstrap_ci_mean(diff: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return (np.nan, np.nan)
    boots = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def run_stat_tests(per_seed: pd.DataFrame) -> pd.DataFrame:
    ok = per_seed[per_seed["status"] == "success"].copy()
    rows = []
    comparisons = [
        ("middle_vs_rest", "young_vs_rest"),
        ("middle_vs_rest", "older_vs_rest"),
    ]
    metrics = [
        ("abs_spd", "abs_spd"),
        ("abs_eod", "abs_eod"),
        ("abs_aod", "abs_aod"),
        ("cfs", "cfs"),
    ]
    for dataset in ok["dataset"].unique():
        for model in MODELS:
            for coding_a, coding_b in comparisons:
                for metric_col, metric_label in metrics:
                    a = ok[
                        (ok["dataset"] == dataset)
                        & (ok["model"] == model)
                        & (ok["age_coding"] == coding_a)
                    ].set_index("seed")[metric_col]
                    b = ok[
                        (ok["dataset"] == dataset)
                        & (ok["model"] == model)
                        & (ok["age_coding"] == coding_b)
                    ].set_index("seed")[metric_col]
                    joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
                    if len(joined) < 2:
                        continue
                    diff = joined["a"] - joined["b"]
                    ci_lo, ci_hi = bootstrap_ci_mean(diff.values, seed=42 + hash((dataset, model, coding_a, coding_b, metric_col)) % 1000)
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "comparison": f"{coding_a}_minus_{coding_b}",
                            "coding_a": coding_a,
                            "coding_b": coding_b,
                            "metric": metric_label,
                            "n_pairs": len(joined),
                            "mean_diff_a_minus_b": float(diff.mean()),
                            "sd_diff": float(diff.std(ddof=1)),
                            "cohen_dz": cohen_dz(diff.values),
                            "bootstrap_ci_low": ci_lo,
                            "bootstrap_ci_high": ci_hi,
                        }
                    )
    return pd.DataFrame(rows)


def validate_against_original(per_seed: pd.DataFrame) -> pd.DataFrame:
    if not ORIGINAL_CSV.exists():
        return pd.DataFrame()
    orig = pd.read_csv(ORIGINAL_CSV)
    orig_age = orig[
        (orig["mitigation"] == "baseline")
        & (orig["protected_attribute"] == "age_group")
    ].copy()
    new_mid = per_seed[
        (per_seed["age_coding"] == "middle_vs_rest") & (per_seed["status"] == "success")
    ].copy()
    merged = new_mid.merge(
        orig_age,
        on=["dataset", "model", "seed"],
        suffixes=("_new", "_orig"),
        how="inner",
    )
    rows = []
    for col in ["accuracy", "cfs", "spd", "eod", "aod"]:
        d = merged[f"{col}_new"] - merged[f"{col}_orig"]
        rows.append(
            {
                "metric": col,
                "n_pairs": len(d),
                "max_abs_diff": float(d.abs().max()),
                "mean_abs_diff": float(d.abs().mean()),
                "matches_within_1e6": bool((d.abs() < 1e-6).all()),
            }
        )
    return pd.DataFrame(rows)


def classify_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level classification using mean CFS across models (cautious rules)."""
    rows = []
    for dataset in summary["dataset"].unique():
        sub = summary[summary["dataset"] == dataset]
        by_coding = sub.groupby("age_coding")["mean_cfs"].mean()
        mid = float(by_coding.get("middle_vs_rest", np.nan))
        young = float(by_coding.get("young_vs_rest", np.nan))
        older = float(by_coding.get("older_vs_rest", np.nan))

        # Visible disparity: mean CFS across models above a modest floor
        floor = 0.08
        mid_vis = mid >= floor
        young_vis = young >= floor
        older_vis = older >= floor

        if mid_vis and young_vis and older_vis:
            label = "robust"
            note = (
                "Mean CFS stays above a modest floor under all three codings, "
                "so age-related disparity remains visible under alternative groupings."
            )
        elif mid_vis and (young_vis or older_vis):
            label = "partially robust"
            note = (
                "Age-related disparity remains under the original coding and at least one "
                "alternative, but weakens under the other alternative coding."
            )
        elif mid_vis and not young_vis and not older_vis:
            label = "sensitive"
            note = (
                "The headline age disparity is concentrated in the middle-vs-rest coding; "
                "alternative young-vs-rest and older-vs-rest codings show weaker composite violations."
            )
        else:
            label = "partially robust"
            note = "Mixed pattern across models; see per-model summary."

        rows.append(
            {
                "dataset": dataset,
                "dataset_label": sub["dataset_label"].iloc[0],
                "classification": label,
                "mean_cfs_middle_vs_rest": mid,
                "mean_cfs_young_vs_rest": young,
                "mean_cfs_older_vs_rest": older,
                "interpretation_note": note,
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, classification: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)

    # Heatmap: dataset x age_coding, averaged over models, colored by mean CFS
    pivot = (
        summary.groupby(["dataset_label", "age_coding"])["mean_cfs"]
        .mean()
        .reset_index()
        .pivot(index="dataset_label", columns="age_coding", values="mean_cfs")
        .astype(float)
    )
    col_order = ["middle_vs_rest", "young_vs_rest", "older_vs_rest"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean CFS"})
    ax.set_title("Age-binning robustness: mean CFS by dataset and age coding\n(averaged over five baseline models, 30 seeds)")
    ax.set_xlabel("Age coding")
    ax.set_ylabel("")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_age_robustness_heatmap_cfs.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Grouped bar: per dataset, three codings, mean |SPD| averaged over models
    plot_df = (
        summary.groupby(["dataset_label", "age_coding"], as_index=False)
        .agg(mean_abs_spd=("mean_abs_spd", "mean"), mean_cfs=("mean_cfs", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = col_order
    sns.barplot(
        data=plot_df, x="dataset_label", y="mean_abs_spd", hue="age_coding",
        hue_order=order, ax=axes[0], palette="Set2",
    )
    axes[0].set_title("Mean |SPD| (model-averaged)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)
    sns.barplot(
        data=plot_df, x="dataset_label", y="mean_cfs", hue="age_coding",
        hue_order=order, ax=axes[1], palette="Set2",
    )
    axes[1].set_title("Mean CFS (model-averaged)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)
    for ax in axes:
        ax.legend(title="Age coding", fontsize=8, title_fontsize=9)
    fig.suptitle("Baseline age disparity under alternative age codings", y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_age_robustness_bars.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...", flush=True)
    loaded = load_all_datasets()

    print("Building group counts...", flush=True)
    group_counts = build_group_counts(loaded)
    group_counts.to_csv(OUT_DIR / "age_group_counts.csv", index=False)

    partial_path = OUT_DIR / "age_robustness_per_seed_partial.csv"
    rows: List[Dict[str, Any]] = []
    total = len(DATASETS) * len(MODELS) * len(AGE_CODING_META) * len(SEEDS)
    done = 0
    t_all = time.perf_counter()

    for dataset_key, (df, _) in loaded.items():
        for model in MODELS:
            for coding in AGE_CODING_META:
                for seed in SEEDS:
                    row = run_baseline_row(dataset_key, df, model, coding, seed)
                    rows.append(row)
                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"  {done}/{total} configs ({done/total:.1%})", flush=True)
                        pd.DataFrame(rows).to_csv(partial_path, index=False)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_DIR / "age_robustness_per_seed.csv", index=False)
    if partial_path.exists():
        partial_path.unlink()

    aggregate = aggregate_results(per_seed)
    aggregate.to_csv(OUT_DIR / "age_robustness_aggregate.csv", index=False)

    summary = build_summary(aggregate, group_counts)
    summary.to_csv(OUT_DIR / "age_robustness_summary.csv", index=False)

    stat_tests = run_stat_tests(per_seed)
    stat_tests.to_csv(OUT_DIR / "age_robustness_stat_tests.csv", index=False)

    validation = validate_against_original(per_seed)
    if not validation.empty:
        validation.to_csv(OUT_DIR / "age_robustness_original_validation.csv", index=False)

    classification = classify_robustness(summary)
    classification.to_csv(OUT_DIR / "age_robustness_classification.csv", index=False)

    meta = {
        "phase": "R2A",
        "analysis_method": "baseline_rerun_same_protocol",
        "seeds": SEEDS,
        "models": MODELS,
        "age_codings": list(AGE_CODING_META.keys()),
        "n_configs": total,
        "n_success": int((per_seed["status"] == "success").sum()),
        "n_failed": int((per_seed["status"] == "failed").sum()),
        "wall_seconds": time.perf_counter() - t_all,
        "outputs": {
            "per_seed": str(OUT_DIR / "age_robustness_per_seed.csv"),
            "aggregate": str(OUT_DIR / "age_robustness_aggregate.csv"),
            "summary": str(OUT_DIR / "age_robustness_summary.csv"),
            "stat_tests": str(OUT_DIR / "age_robustness_stat_tests.csv"),
            "group_counts": str(OUT_DIR / "age_group_counts.csv"),
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    make_figures(summary, classification)
    print(json.dumps(meta, indent=2))
    return 0 if meta["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
