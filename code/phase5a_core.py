"""Phase 5A core: loading, validation, aggregation, statistics, ranking helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from fairness_utils import ACFS_WEIGHT_PRESETS
from pipeline_core import Q1_ROOT

RESULTS_IN_DIR = Q1_ROOT / "results" / "phase4b_optionB_30seed"
RESULTS_CSV = RESULTS_IN_DIR / "phase4b_results.csv"
MANIFEST_CSV = RESULTS_IN_DIR / "phase4b_config_manifest.csv"
RUNTIME_CSV = RESULTS_IN_DIR / "phase4b_runtime_summary.csv"
DATASET_SUMMARIES_JSON = RESULTS_IN_DIR / "phase4b_dataset_summaries_corrected.json"
COMPLETION_JSON = RESULTS_IN_DIR / "phase4c_completion_summary.json"

OUT_DIR = Q1_ROOT / "results" / "phase5_postprocessing"
TABLES_DIR = OUT_DIR / "tables"
SUPP_DIR = OUT_DIR / "supplementary"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = Q1_ROOT / "reports" / "phase5a"

GROUP_COLS = ["dataset", "model", "mitigation", "protected_attribute"]
EXPECTED_TOTAL = 3690
EXPECTED_SEEDS = list(range(42, 72))
EXPECTED_DATASETS = ["acs_income_ca_2018", "adult", "bank_uci"]
EXPECTED_MODELS = [
    "gradient_boosting",
    "logistic_regression",
    "mlp",
    "random_forest",
    "xgboost",
]
EXPECTED_MITIGATIONS = [
    "ExponentiatedGradient_DP",
    "ExponentiatedGradient_EO",
    "baseline",
    "equalized_odds",
    "reweighing",
]

DATASET_LABEL = {
    "adult": "Adult",
    "bank_uci": "UCI Bank",
    "acs_income_ca_2018": "ACSIncome (CA 2018)",
}
MODEL_LABEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}
MITIGATION_LABEL = {
    "baseline": "Baseline",
    "reweighing": "Reweighing",
    "equalized_odds": "Equalized Odds",
    "ExponentiatedGradient_DP": "EG (Demographic Parity)",
    "ExponentiatedGradient_EO": "EG (Equalized Odds)",
}


def ensure_dirs() -> None:
    for d in (OUT_DIR, TABLES_DIR, SUPP_DIR, FIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    return df


def load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def validate(df: pd.DataFrame) -> Dict[str, Any]:
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "spd",
        "di",
        "eod",
        "aod",
        "cfs",
        "acfs_balanced",
    ]
    dup = int(
        df.duplicated(
            subset=["run_block", "dataset", "model", "mitigation", "protected_attribute", "seed"]
        ).sum()
    )
    impossible: List[str] = []
    for col, lo, hi in [
        ("accuracy", 0.0, 1.0),
        ("precision", 0.0, 1.0),
        ("recall", 0.0, 1.0),
        ("f1", 0.0, 1.0),
        ("spd", -1.0, 1.0),
        ("eod", -1.0, 1.0),
        ("aod", -1.0, 1.0),
        ("di", 0.0, np.inf),
        ("cfs", 0.0, np.inf),
        ("acfs_balanced", 0.0, np.inf),
    ]:
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad):
            impossible.append(f"{col}: {len(bad)} rows outside [{lo}, {hi}]")

    missing = {c: int(df[c].isna().sum()) for c in metric_cols if df[c].isna().any()}
    non_numeric = [c for c in ("cfs", "acfs_balanced") if not np.issubdtype(df[c].dtype, np.number)]

    report = {
        "total_rows": int(len(df)),
        "expected_total": EXPECTED_TOTAL,
        "total_ok": int(len(df)) == EXPECTED_TOTAL,
        "n_success": int((df["status"] == "success").sum()),
        "n_failed": int((df["status"] == "failed").sum()),
        "n_skipped": int((df["status"] == "skipped").sum()),
        "duplicate_rows": dup,
        "datasets": sorted(df["dataset"].unique().tolist()),
        "datasets_ok": sorted(df["dataset"].unique().tolist()) == EXPECTED_DATASETS,
        "models": sorted(df["model"].unique().tolist()),
        "models_ok": sorted(df["model"].unique().tolist()) == EXPECTED_MODELS,
        "mitigations": sorted(df["mitigation"].unique().tolist()),
        "mitigations_ok": sorted(df["mitigation"].unique().tolist()) == EXPECTED_MITIGATIONS,
        "protected_attributes": sorted(df["protected_attribute"].unique().tolist()),
        "seeds": sorted(df["seed"].unique().tolist()),
        "seeds_ok": sorted(df["seed"].unique().tolist()) == EXPECTED_SEEDS,
        "cfs_acfs_numeric": len(non_numeric) == 0,
        "missing_metric_values": missing,
        "impossible_metric_values": impossible,
    }
    report["all_checks_passed"] = (
        report["total_ok"]
        and report["n_success"] == EXPECTED_TOTAL
        and report["n_failed"] == 0
        and report["n_skipped"] == 0
        and report["duplicate_rows"] == 0
        and report["datasets_ok"]
        and report["models_ok"]
        and report["mitigations_ok"]
        and report["seeds_ok"]
        and report["cfs_acfs_numeric"]
        and not missing
        and not impossible
    )
    return report


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_spd"] = df["spd"].abs()
    df["abs_eod"] = df["eod"].abs()
    df["abs_aod"] = df["aod"].abs()
    df["di_violation"] = (1.0 - df["di"]).abs()
    for preset, (w1, w2, w3, w4) in ACFS_WEIGHT_PRESETS.items():
        df[f"acfs_{preset}"] = (
            w1 * df["abs_spd"] + w2 * df["abs_eod"] + w3 * df["abs_aod"] + w4 * df["di_violation"]
        )
    return df


def build_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "spd",
        "di",
        "eod",
        "aod",
        "cfs",
        "acfs_balanced",
    ]
    abs_metrics = ["abs_spd", "abs_eod", "abs_aod"]
    records: List[Dict[str, Any]] = []
    for keys, g in df.groupby(GROUP_COLS):
        row: Dict[str, Any] = dict(zip(GROUP_COLS, keys))
        row["run_block"] = g["run_block"].iloc[0]
        row["n_seeds"] = int(g["seed"].nunique())
        for m in metrics:
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"std_{m}"] = float(g[m].std(ddof=1))
        for m in abs_metrics:
            row[f"mean_{m}"] = float(g[m].mean())
        records.append(row)
    agg = pd.DataFrame(records)
    return agg.sort_values(GROUP_COLS).reset_index(drop=True)


def cohen_dz(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 12345
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_correction(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def paired_comparison(
    df: pd.DataFrame,
    metric: str,
    baseline_mit: str = "baseline",
    compare_mits: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Paired comparisons across seeds within dataset/model/protected/mitigation cells."""
    if compare_mits is None:
        compare_mits = [
            "reweighing",
            "equalized_odds",
            "ExponentiatedGradient_DP",
            "ExponentiatedGradient_EO",
        ]
    records: List[Dict[str, Any]] = []
    strata = ["dataset", "model", "protected_attribute"]
    for keys, g in df.groupby(strata):
        base = g[g["mitigation"] == baseline_mit].set_index("seed")[metric]
        if base.empty:
            continue
        for mit in compare_mits:
            cur = g[g["mitigation"] == mit].set_index("seed")[metric]
            if cur.empty:
                continue
            common = base.index.intersection(cur.index)
            if len(common) < 3:
                continue
            b = base.loc[common].to_numpy(dtype=float)
            c = cur.loc[common].to_numpy(dtype=float)
            diff = c - b
            try:
                w_stat, w_p = stats.wilcoxon(c, b, zero_method="wilcox", correction=False)
            except ValueError:
                w_stat, w_p = float("nan"), 1.0
            t_stat, t_p = stats.ttest_rel(c, b)
            lo, hi = bootstrap_ci(diff)
            rec = dict(zip(strata, keys))
            rec.update(
                {
                    "metric": metric,
                    "baseline": baseline_mit,
                    "comparison": mit,
                    "n_pairs": int(len(common)),
                    "mean_baseline": float(b.mean()),
                    "mean_comparison": float(c.mean()),
                    "mean_diff": float(diff.mean()),
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "cohen_dz": cohen_dz(diff),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(w_p),
                    "ttest_stat": float(t_stat),
                    "ttest_p": float(t_p),
                }
            )
            records.append(rec)
    res = pd.DataFrame(records)
    if res.empty:
        return res
    for fam_key, fam in res.groupby(["metric", "comparison"]):
        idx = fam.index.tolist()
        res.loc[idx, "wilcoxon_p_holm"] = holm_correction(fam["wilcoxon_p"].tolist())
        res.loc[idx, "ttest_p_holm"] = holm_correction(fam["ttest_p"].tolist())
    return res


def pareto_front(sub: pd.DataFrame, acc_col: str, fair_col: str) -> pd.DataFrame:
    """Higher accuracy is better; lower fairness score is better."""
    pts = sub[[acc_col, fair_col]].to_numpy()
    keep = np.ones(len(sub), dtype=bool)
    for i in range(len(sub)):
        for j in range(len(sub)):
            if i == j:
                continue
            if (
                pts[j, 0] >= pts[i, 0]
                and pts[j, 1] <= pts[i, 1]
                and (pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1])
            ):
                keep[i] = False
                break
    return sub[keep]
