"""R1: SAM-Fair selection stability under DI-scaled composite variants.

Post-processing only: reads stored per-seed results and acfs_rankings candidate
scope. No model retraining; does not modify the 3,690-run main grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sam_fair_select import (
    MODEL_SHORT,
    MIT_SHORT,
    RUNTIME_DATASET,
    select_sam,
)

REPO = Path(__file__).resolve().parents[2]
PKG = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
)
DATA = PKG / "data_tables"
OUT = REPO / "q1_upgrade" / "results" / "r1_samfair_di_scaled_validation"
REPORT_DIR = REPO / "q1_upgrade" / "reports" / "r1_samfair_di_scaled_validation"

GROUP_COLS = ["dataset", "model", "mitigation", "protected_attribute"]
SETTINGS = [
    ("acs_income_ca_2018", "age_group", "ACSIncome (CA 2018)"),
    ("acs_income_ca_2018", "sex", "ACSIncome (CA 2018)"),
    ("adult", "age_group", "Adult"),
    ("adult", "race", "Adult"),
    ("adult", "sex", "Adult"),
    ("bank_uci", "age_group", "UCI Bank"),
    ("bank_uci", "job_group", "UCI Bank"),
]

FOUR_FIFTHS_LOW = 0.8
FOUR_FIFTHS_HIGH = 1.25

VARIANTS = {
    "original": "mean_acfs_original",
    "di_clipped": "mean_acfs_di_clipped",
    "four_fifths": "mean_acfs_four_fifths",
}

DATASET_LABEL = {v[0]: v[2] for v in SETTINGS}
PROTECTED_BY_DS = {}
for ds, prot, _ in SETTINGS:
    PROTECTED_BY_DS.setdefault(ds, []).append(prot)


def di_dev_clipped(di: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(1.0 - di), 1.0)


def four_fifths_penalty(di: np.ndarray) -> np.ndarray:
    di = np.asarray(di, dtype=float)
    below = np.maximum(FOUR_FIFTHS_LOW - di, 0.0)
    above = np.maximum(di - FOUR_FIFTHS_HIGH, 0.0)
    return np.minimum(below + above, 1.0)


def balanced_acfs(
    spd: np.ndarray,
    di_term: np.ndarray,
    eod: np.ndarray,
    aod: np.ndarray,
) -> np.ndarray:
    """Balanced ACFS (= CFS when weights are 0.25 each)."""
    return 0.25 * (
        np.abs(spd) + np.abs(eod) + np.abs(aod) + np.asarray(di_term, dtype=float)
    )


def aggregate_variants(per_seed: pd.DataFrame) -> pd.DataFrame:
    spd = per_seed["spd"].to_numpy()
    di = per_seed["di"].to_numpy()
    eod = per_seed["eod"].to_numpy()
    aod = per_seed["aod"].to_numpy()

    per_seed = per_seed.copy()
    per_seed["acfs_original"] = balanced_acfs(spd, np.abs(1.0 - di), eod, aod)
    per_seed["acfs_di_clipped"] = balanced_acfs(spd, di_dev_clipped(di), eod, aod)
    per_seed["acfs_four_fifths"] = balanced_acfs(spd, four_fifths_penalty(di), eod, aod)

    rows = []
    for keys, g in per_seed.groupby(GROUP_COLS):
        rec = dict(zip(GROUP_COLS, keys))
        rec["mean_accuracy"] = float(g["accuracy"].mean())
        rec["mean_acfs_original"] = float(g["acfs_original"].mean())
        rec["mean_acfs_di_clipped"] = float(g["acfs_di_clipped"].mean())
        rec["mean_acfs_four_fifths"] = float(g["acfs_four_fifths"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def build_acfs_df(agg_var: pd.DataFrame, acfs_col: str) -> pd.DataFrame:
    """Match acfs_rankings.csv schema for select_sam."""
    base = pd.read_csv(DATA / "acfs_rankings.csv")
    merged = base.merge(
        agg_var[GROUP_COLS + ["mean_accuracy", acfs_col]],
        on=GROUP_COLS,
        how="left",
        suffixes=("_base", ""),
    )
    out = base.copy()
    out["mean_accuracy"] = merged["mean_accuracy"]
    out["mean_acfs_balanced"] = merged[acfs_col]
    return out


def display_model(model: str) -> str:
    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }
    return labels.get(model, model)


def display_mitigation(mit: str) -> str:
    labels = {
        "baseline": "Baseline",
        "reweighing": "Reweighing",
        "equalized_odds": "Equalized Odds",
        "ExponentiatedGradient_DP": "EG (DP)",
        "ExponentiatedGradient_EO": "EG (EO)",
    }
    return labels.get(mit, mit)


def interpret_row(
    ds: str,
    prot: str,
    orig: dict,
    clip: dict,
    ff: dict,
) -> str:
    changed_clip = (
        orig.get("model") != clip.get("model")
        or orig.get("mitigation") != clip.get("mitigation")
    )
    changed_ff = (
        orig.get("model") != ff.get("model")
        or orig.get("mitigation") != ff.get("mitigation")
    )
    if not changed_clip and not changed_ff:
        if ds == "bank_uci" and prot == "age_group":
            return "Stable; RF+EO under all three composites."
        return "Stable under DI-scaled composites."
    parts = []
    if changed_clip:
        parts.append("DI-clipped composite shifts selection")
    if changed_ff:
        parts.append("four-fifths composite shifts selection")
    if ds == "bank_uci" and prot == "age_group":
        parts.append("low-base-rate DI sensitivity")
    return "; ".join(parts) + "."


def run_validation(delta: float = 0.03, lam: float = 0.0, rho: float = 0.0) -> pd.DataFrame:
    per_seed = pd.read_csv(DATA / "per_seed_full_results_3690.csv")
    if len(per_seed) != 3690:
        raise ValueError(f"Expected 3690 per-seed rows, got {len(per_seed)}")

    agg_var = aggregate_variants(per_seed)
    agg = pd.read_csv(DATA / "full_aggregate_30seed.csv")
    runtime = pd.read_csv(DATA / "runtime_summary.csv")

    # Verify original variant matches stored balanced ACFS on candidate rows.
    base = pd.read_csv(DATA / "acfs_rankings.csv")
    chk = base.merge(
        agg_var[GROUP_COLS + ["mean_acfs_original"]],
        on=GROUP_COLS,
        how="left",
    )
    diff = (chk["mean_acfs_balanced"] - chk["mean_acfs_original"]).abs().max()
    if diff > 1e-3:
        raise ValueError(f"Original ACFS mismatch vs acfs_rankings: max diff {diff}")

    selections = {v: {} for v in VARIANTS}
    for variant, col in VARIANTS.items():
        acfs_df = build_acfs_df(agg_var, col)
        for ds, prot, _ in SETTINGS:
            selections[variant][(ds, prot)] = select_sam(
                acfs_df,
                agg,
                runtime,
                ds,
                prot,
                "mean_acfs_balanced",
                lam,
                rho,
                delta,
            )

    rows = []
    for ds, prot, dlabel in SETTINGS:
        o = selections["original"][(ds, prot)]
        c = selections["di_clipped"][(ds, prot)]
        f = selections["four_fifths"][(ds, prot)]
        changed = (
            o.get("model") != c.get("model")
            or o.get("mitigation") != c.get("mitigation")
            or o.get("model") != f.get("model")
            or o.get("mitigation") != f.get("mitigation")
        )
        rows.append(
            {
                "dataset": ds,
                "dataset_label": dlabel,
                "protected_attribute": prot,
                "orig_model": o.get("model", ""),
                "orig_mitigation": o.get("mitigation", ""),
                "orig_model_display": display_model(o.get("model", "")),
                "orig_mitigation_display": display_mitigation(o.get("mitigation", "")),
                "di_clipped_model": c.get("model", ""),
                "di_clipped_mitigation": c.get("mitigation", ""),
                "di_clipped_model_display": display_model(c.get("model", "")),
                "di_clipped_mitigation_display": display_mitigation(c.get("mitigation", "")),
                "four_fifths_model": f.get("model", ""),
                "four_fifths_mitigation": f.get("mitigation", ""),
                "four_fifths_model_display": display_model(f.get("model", "")),
                "four_fifths_mitigation_display": display_mitigation(f.get("mitigation", "")),
                "selection_changed": "yes" if changed else "no",
                "di_clipped_changed": (
                    "yes"
                    if o.get("model") != c.get("model")
                    or o.get("mitigation") != c.get("mitigation")
                    else "no"
                ),
                "four_fifths_changed": (
                    "yes"
                    if o.get("model") != f.get("model")
                    or o.get("mitigation") != f.get("mitigation")
                    else "no"
                ),
                "interpretation": interpret_row(ds, prot, o, c, f),
                "orig_mean_acfs": o.get("mean_acfs"),
                "di_clipped_mean_acfs": c.get("mean_acfs"),
                "four_fifths_mean_acfs": f.get("mean_acfs"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pkg_out = PKG / "revision_robustness" / "sam_fair_di_scaled"
    pkg_out.mkdir(parents=True, exist_ok=True)

    summary = run_validation()
    summary.to_csv(OUT / "sam_fair_di_scaled_stability.csv", index=False)
    summary.to_csv(
        pkg_out / "sam_fair_di_scaled_stability.csv",
        index=False,
    )

    meta = {
        "analysis": "r1_samfair_di_scaled_validation",
        "source_per_seed": str(DATA / "per_seed_full_results_3690.csv"),
        "source_acfs_rankings": str(DATA / "acfs_rankings.csv"),
        "candidate_scope": (
            "Main-grid mitigations (baseline, reweighing, equalized_odds) on all five "
            "model families plus scoped Exponentiated Gradient (LR all settings; RF Adult "
            "sex and UCI Bank age_group only). XGBoost-EG probe excluded."
        ),
        "n_candidate_rows_acfs_rankings": int(len(pd.read_csv(DATA / "acfs_rankings.csv"))),
        "delta": 0.03,
        "lambda": 0.0,
        "rho": 0.0,
        "variants": list(VARIANTS.keys()),
        "n_settings": 7,
        "n_unchanged_di_clipped": int((summary["di_clipped_changed"] == "no").sum()),
        "n_unchanged_four_fifths": int((summary["four_fifths_changed"] == "no").sum()),
        "n_any_change": int((summary["selection_changed"] == "yes").sum()),
    }
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (
        pkg_out / "run_metadata.json"
    ).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(summary[["dataset_label", "protected_attribute", "selection_changed", "interpretation"]])
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
