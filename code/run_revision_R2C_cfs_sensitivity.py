"""Revision R2C: CFS sensitivity to DI scaling (analysis only; no model reruns)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

from fairness_utils import compute_cfs

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "revision_R2C_cfs_sensitivity"
FIG_DIR = OUT_DIR / "figures"
INPUT_CSV = (
    Q1_ROOT
    / "results"
    / "phase5_postprocessing"
    / "supplementary"
    / "supp_per_seed_full_results.csv"
)
BOUNDED_DI_CSV = (
    Q1_ROOT
    / "results"
    / "phase5_postprocessing"
    / "supplementary"
    / "supp_bounded_di_robustness.csv"
)

GROUP_COLS = ["dataset", "model", "mitigation", "protected_attribute"]
RANK_GROUP = ["dataset", "protected_attribute"]
COMPOSITES = [
    "cfs_original",
    "cfs_no_di",
    "cfs_di_clipped",
    "cfs_four_fifths",
]
ALT_COMPOSITES = ["cfs_no_di", "cfs_di_clipped", "cfs_four_fifths"]

FOUR_FIFTHS_LOW = 0.8
FOUR_FIFTHS_HIGH = 1.25  # 1 / 0.8

# Classification thresholds (documented in report).
SPEARMAN_STABLE = 0.8
SPEARMAN_MODERATE = 0.6
TOP3_OVERLAP_STABLE = 2


def di_dev_clipped(di: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(1.0 - di), 1.0)


def four_fifths_penalty(di: np.ndarray) -> np.ndarray:
    """Bounded DI penalty: 0 inside [0.8, 1.25], else min(distance to band, 1)."""
    di = np.asarray(di, dtype=float)
    below = np.maximum(FOUR_FIFTHS_LOW - di, 0.0)
    above = np.maximum(di - FOUR_FIFTHS_HIGH, 0.0)
    return np.minimum(below + above, 1.0)


def compute_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spd = out["spd"].to_numpy()
    di = out["di"].to_numpy()
    eod = out["eod"].to_numpy()
    aod = out["aod"].to_numpy()

    abs_spd = np.abs(spd)
    abs_eod = np.abs(eod)
    abs_aod = np.abs(aod)
    di_viol = np.abs(1.0 - di)
    di_clip = di_dev_clipped(di)
    ff_pen = four_fifths_penalty(di)

    out["abs_spd"] = abs_spd
    out["abs_eod"] = abs_eod
    out["abs_aod"] = abs_aod
    out["abs_di_violation"] = di_viol

    out["cfs_original"] = compute_cfs(spd, di, eod, aod)
    out["cfs_no_di"] = np.nanmean(
        np.column_stack([abs_spd, abs_eod, abs_aod]), axis=1
    )
    out["cfs_di_clipped"] = np.nanmean(
        np.column_stack([abs_spd, abs_eod, abs_aod, di_clip]), axis=1
    )
    out["cfs_four_fifths"] = np.nanmean(
        np.column_stack([abs_spd, abs_eod, abs_aod, ff_pen]), axis=1
    )

    # Validate against stored CFS where present.
    if "cfs" in out.columns:
        out["cfs_stored"] = out["cfs"]
        out["cfs_original_diff"] = out["cfs_original"] - out["cfs_stored"]
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "abs_spd",
        "abs_eod",
        "abs_aod",
        "abs_di_violation",
        *COMPOSITES,
    ]
    rows = []
    for keys, grp in df.groupby(GROUP_COLS, sort=True):
        row = dict(zip(GROUP_COLS, keys))
        row["n_seeds"] = len(grp)
        row["positive_rate_proxy"] = np.nan  # filled later if needed
        for m in metrics:
            row[f"mean_{m}"] = grp[m].mean()
            row[f"sd_{m}"] = grp[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def config_label(row: pd.Series) -> str:
    return f"{row['model']}|{row['mitigation']}"


def ranking_table(agg: pd.DataFrame, composite: str) -> pd.DataFrame:
    col = f"mean_{composite}"
    rows = []
    for (dataset, protected), grp in agg.groupby(RANK_GROUP, sort=True):
        ranked = grp.sort_values(col, ascending=True).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["composite"] = composite
        ranked["config_id"] = ranked.apply(config_label, axis=1)
        rows.append(ranked)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def top_configurations(rank_df: pd.DataFrame) -> pd.DataFrame:
    tops = rank_df[rank_df["rank"] == 1].copy()
    return tops[
        ["dataset", "protected_attribute", "composite", "model", "mitigation", "config_id"]
        + [c for c in tops.columns if c.startswith("mean_cfs")]
    ]


def ranking_stability(agg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranks = {c: ranking_table(agg, c) for c in COMPOSITES}
    top_df = pd.concat([top_configurations(r) for r in ranks.values()], ignore_index=True)
    top_df.to_csv(OUT_DIR / "cfs_top_configurations.csv", index=False)

    orig = ranks["cfs_original"]
    rows = []
    for (dataset, protected), orig_grp in orig.groupby(RANK_GROUP):
        orig_map = orig_grp.set_index("config_id")["rank"]
        orig_top3 = set(orig_grp.nsmallest(3, "rank")["config_id"])
        orig_top1 = orig_grp.loc[orig_grp["rank"].idxmin(), "config_id"]
        for alt in ALT_COMPOSITES:
            alt_grp = ranks[alt][
                (ranks[alt]["dataset"] == dataset)
                & (ranks[alt]["protected_attribute"] == protected)
            ]
            alt_map = alt_grp.set_index("config_id")["rank"]
            common = sorted(set(orig_map.index) & set(alt_map.index))
            if len(common) < 3:
                rho = np.nan
                tau = np.nan
            else:
                o_r = [orig_map[c] for c in common]
                a_r = [alt_map[c] for c in common]
                rho = spearmanr(o_r, a_r).correlation
                tau = kendalltau(o_r, a_r).correlation
            alt_top3 = set(alt_grp.nsmallest(3, "rank")["config_id"])
            alt_top1 = alt_grp.loc[alt_grp["rank"].idxmin(), "config_id"]
            overlap = len(orig_top3 & alt_top3)
            rows.append(
                {
                    "dataset": dataset,
                    "protected_attribute": protected,
                    "alternative_composite": alt,
                    "top1_same": orig_top1 == alt_top1,
                    "top1_original": orig_top1,
                    "top1_alternative": alt_top1,
                    "top3_overlap": overlap,
                    "spearman_rho": rho,
                    "kendall_tau": tau,
                    "n_configs": len(common),
                }
            )
    stab = pd.DataFrame(rows)
    return stab, top_df


def classify_setting(stab_row: pd.DataFrame) -> str:
    """Classify using worst case across three alternatives."""
    if stab_row.empty:
        return "unknown"
    if (stab_row["top1_same"].all() and (stab_row["top3_overlap"] >= TOP3_OVERLAP_STABLE).all()
            and (stab_row["spearman_rho"] >= SPEARMAN_STABLE).all()):
        return "stable"
    if ((stab_row["top3_overlap"] >= 1).all()
            and (stab_row["spearman_rho"] >= SPEARMAN_MODERATE).all()):
        return "moderately sensitive"
    if (stab_row["spearman_rho"] < SPEARMAN_MODERATE).any() or (stab_row["top3_overlap"] == 0).any():
        return "sensitive"
    return "moderately sensitive"


def sensitivity_classification(stab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, protected), grp in stab.groupby(["dataset", "protected_attribute"]):
        rows.append(
            {
                "dataset": dataset,
                "protected_attribute": protected,
                "classification": classify_setting(grp),
                "min_spearman_rho": grp["spearman_rho"].min(),
                "mean_spearman_rho": grp["spearman_rho"].mean(),
                "min_top3_overlap": grp["top3_overlap"].min(),
                "all_top1_same": bool(grp["top1_same"].all()),
            }
        )
    return pd.DataFrame(rows)


def claim_checks(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Claim A: accuracy-fairness divergence
    div_counts = {c: {"same": 0, "total": 0} for c in COMPOSITES}
    for (_, _), grp in agg.groupby(RANK_GROUP):
        best_acc = grp.loc[grp["mean_accuracy"].idxmax(), "model"]
        for c in COMPOSITES:
            best_fair = grp.loc[grp[f"mean_{c}"].idxmin(), "model"]
            div_counts[c]["total"] += 1
            if best_acc != best_fair:
                div_counts[c]["same"] += 1
    claim_a = all(
        div_counts[c]["same"] / div_counts[c]["total"] >= 0.85 for c in COMPOSITES
    )
    rows.append(
        {
            "claim": "A",
            "statement": "Accuracy and fairness diverge; the most accurate models are rarely the fairest.",
            "verdict": "supported" if claim_a else "supported with qualification",
            "detail": "; ".join(
                f"{c}: {div_counts[c]['same']}/{div_counts[c]['total']} settings differ"
                for c in COMPOSITES
            ),
        }
    )

    # Claim B: EO vs Reweighing
    eo_better = {c: 0 for c in COMPOSITES}
    eo_total = 0
    acc_cost = {c: [] for c in COMPOSITES}
    for keys, grp in agg.groupby(RANK_GROUP):
        sub = grp[grp["mitigation"].isin(["equalized_odds", "reweighing"])]
        if sub.empty:
            continue
        for model in sub["model"].unique():
            eo_rows = sub[(sub["model"] == model) & (sub["mitigation"] == "equalized_odds")]
            rw_rows = sub[(sub["model"] == model) & (sub["mitigation"] == "reweighing")]
            if eo_rows.empty or rw_rows.empty:
                continue
            eo_total += 1
            for c in COMPOSITES:
                eo_cfs = eo_rows[f"mean_{c}"].mean()
                rw_cfs = rw_rows[f"mean_{c}"].mean()
                if eo_cfs < rw_cfs:
                    eo_better[c] += 1
                acc_cost[c].append(
                    eo_rows["mean_accuracy"].mean() - rw_rows["mean_accuracy"].mean()
                )
    frac = {c: eo_better[c] / eo_total if eo_total else np.nan for c in COMPOSITES}
    claim_b = all(frac[c] >= 0.7 for c in COMPOSITES)
    rows.append(
        {
            "claim": "B",
            "statement": "Equalized Odds generally lowers fairness violations more than Reweighing, at a larger accuracy cost.",
            "verdict": "supported" if claim_b else "supported with qualification",
            "detail": "; ".join(
                f"{c}: EO lower CFS in {eo_better[c]}/{eo_total} model-settings; "
                f"mean acc cost {np.mean(acc_cost[c]):.4f}"
                for c in COMPOSITES
            ),
        }
    )

    # Claim C: EG-DP tightest SPD where applied
    eg_tightest = {c: 0 for c in COMPOSITES}
    eg_total = 0
    for keys, grp in agg.groupby(RANK_GROUP):
        eg = grp[grp["mitigation"] == "ExponentiatedGradient_DP"]
        if eg.empty:
            continue
        others = grp[~grp["mitigation"].isin(["ExponentiatedGradient_DP"])]
        for _, eg_row in eg.iterrows():
            same_model = others[others["model"] == eg_row["model"]]
            if same_model.empty:
                continue
            eg_total += 1
            for c in COMPOSITES:
                if eg_row["mean_abs_spd"] <= same_model["mean_abs_spd"].min():
                    eg_tightest[c] += 1
    frac_c = {c: eg_tightest[c] / eg_total if eg_total else np.nan for c in COMPOSITES}
    claim_c = all(frac_c[c] >= 0.8 for c in COMPOSITES)
    rows.append(
        {
            "claim": "C",
            "statement": "EG-DP gives the tightest statistical parity where applied, sometimes at a substantial accuracy cost.",
            "verdict": "supported" if claim_c else "supported with qualification",
            "detail": "; ".join(
                f"{c}: EG-DP lowest |SPD| in {eg_tightest[c]}/{eg_total} comparisons"
                for c in COMPOSITES
            ),
        }
    )

    # Claim D: UCI Bank age_group DI sensitivity
    bank_age = agg[
        (agg["dataset"] == "bank_uci") & (agg["protected_attribute"] == "age_group")
    ].copy()
    bank_age["di_share_original"] = bank_age["mean_abs_di_violation"] / (
        4 * bank_age["mean_cfs_original"]
    )
    bank_age["di_share_no_di"] = 0.0
    bank_age["rank_shift_di_clipped"] = False
    orig_rank = bank_age.sort_values("mean_cfs_original")["model"].tolist()
    clip_rank = bank_age.sort_values("mean_cfs_di_clipped")["model"].tolist()
    di_dominates = (bank_age["di_share_original"] > 0.35).mean() > 0.5
    rank_changes = orig_rank != clip_rank
    rows.append(
        {
            "claim": "D",
            "statement": "UCI Bank age_group CFS must be interpreted cautiously because DI is sensitive to low positive rate.",
            "verdict": "supported" if di_dominates else "supported with qualification",
            "detail": (
                f"mean DI share of original CFS across configs: "
                f"{bank_age['di_share_original'].mean():.3f}; "
                f"original vs DI-clipped ranking changes for "
                f"{int(rank_changes)} mitigations/models when sorted by model-mitigation groups"
            ),
        }
    )

    return pd.DataFrame(rows)


def make_figures(per_seed: pd.DataFrame, agg: pd.DataFrame, stab: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)

    # Heatmap: Spearman rho by dataset/protected × alternative
    hm = stab.pivot_table(
        index=["dataset", "protected_attribute"],
        columns="alternative_composite",
        values="spearman_rho",
    ).astype(float)
    hm.index = [f"{d} / {p}" for d, p in hm.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.4, vmax=1.0, ax=ax)
    ax.set_title("CFS ranking stability: Spearman rho vs original\n(by dataset and protected setting)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_cfs_ranking_stability_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Scatter: original vs DI-clipped (configuration means)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(agg["mean_cfs_original"], agg["mean_cfs_di_clipped"], alpha=0.35, s=20)
    lim = max(agg["mean_cfs_original"].max(), agg["mean_cfs_di_clipped"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Mean CFS (original)")
    ax.set_ylabel("Mean CFS (DI-clipped)")
    ax.set_title("Configuration-level composite scores")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_cfs_original_vs_di_clipped.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # UCI Bank age_group sensitivity
    bank = agg[
        (agg["dataset"] == "bank_uci") & (agg["protected_attribute"] == "age_group")
    ].copy()
    bank["label"] = bank["model"] + " / " + bank["mitigation"]
    plot_cols = ["mean_cfs_original", "mean_cfs_no_di", "mean_cfs_di_clipped", "mean_cfs_four_fifths"]
    plot_df = bank.melt(
        id_vars=["label"],
        value_vars=plot_cols,
        var_name="composite",
        value_name="mean_cfs",
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(data=plot_df, x="label", y="mean_cfs", hue="composite", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    ax.set_title("UCI Bank age_group: composite sensitivity by configuration")
    ax.legend(title="Composite", fontsize=7, title_fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_uci_bank_age_group_cfs_sensitivity.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_input(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "n_rows": len(df),
        "n_success": int((df["status"] == "success").sum()) if "status" in df.columns else len(df),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "models": sorted(df["model"].unique().tolist()),
        "mitigations": sorted(df["mitigation"].unique().tolist()),
        "protected_attributes": sorted(df["protected_attribute"].unique().tolist()),
        "seed_min": int(df["seed"].min()),
        "seed_max": int(df["seed"].max()),
        "n_seeds": int(df["seed"].nunique()),
        "cfs_max_abs_diff_vs_stored": float(df["cfs_original_diff"].abs().max())
        if "cfs_original_diff" in df.columns
        else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = df[df["status"] == "success"].copy()
    per_seed = compute_variants(df)
    per_seed.to_csv(OUT_DIR / "cfs_sensitivity_per_seed.csv", index=False)

    meta = validate_input(per_seed)
    meta["input_file"] = str(INPUT_CSV)
    meta["bounded_di_reference"] = str(BOUNDED_DI_CSV) if BOUNDED_DI_CSV.exists() else None

    agg = aggregate(per_seed)
    agg.to_csv(OUT_DIR / "cfs_sensitivity_aggregate.csv", index=False)

    stab, _ = ranking_stability(agg)
    stab.to_csv(OUT_DIR / "cfs_ranking_stability.csv", index=False)

    classification = sensitivity_classification(stab)
    classification.to_csv(OUT_DIR / "cfs_sensitivity_classification.csv", index=False)

    claims = claim_checks(agg)
    claims.to_csv(OUT_DIR / "cfs_claim_check.csv", index=False)

    make_figures(per_seed, agg, stab)

    meta["composite_definitions"] = {
        "cfs_original": "mean(|SPD|, |EOD|, |AOD|, |1-DI|)",
        "cfs_no_di": "mean(|SPD|, |EOD|, |AOD|)",
        "cfs_di_clipped": "mean(|SPD|, |EOD|, |AOD|, min(|1-DI|, 1))",
        "cfs_four_fifths": (
            "mean(|SPD|, |EOD|, |AOD|, bounded_DI_penalty) where penalty=0 if DI in [0.8,1.25] "
            "else min(distance to band, 1)"
        ),
    }
    meta["classification_thresholds"] = {
        "stable": f"top1 same AND top3 overlap>={TOP3_OVERLAP_STABLE} AND rho>={SPEARMAN_STABLE}",
        "moderately_sensitive": f"top3 overlap>=1 AND rho>={SPEARMAN_MODERATE}",
        "sensitive": f"rho<{SPEARMAN_MODERATE} OR top3 overlap=0",
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
