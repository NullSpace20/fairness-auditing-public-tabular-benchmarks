"""Phase 5A: post-processing and analysis (no manuscript edits)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase5a_core as core
from fairness_utils import ACFS_WEIGHT_PRESETS


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def _fmt_cell(value: Any, float_fmt: str) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if np.isnan(value):
            return "--"
        return float_fmt % value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _latex_escape(value)


def df_to_latex(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.3f") -> str:
    col_spec = "l" * len(df.columns)
    header = " & ".join(_latex_escape(c) for c in df.columns) + " \\\\"
    body_lines = []
    for _, row in df.iterrows():
        cells = [_fmt_cell(row[c], float_fmt) for c in df.columns]
        body_lines.append(" & ".join(cells) + " \\\\")
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{_latex_escape(caption)}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
        f"{header}\n\\hline\n"
        f"{body}\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def save_table(df: pd.DataFrame, name: str, directory: Path, caption: str, latex: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / f"{name}.csv", index=False)
    if latex:
        (directory / f"{name}.tex").write_text(
            df_to_latex(df, caption, f"tab:{name}"), encoding="utf-8"
        )


# --------------------------------------------------------------------------
# Main tables
# --------------------------------------------------------------------------


def table_dataset_provenance(summaries: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ds in ["adult", "bank_uci", "acs_income_ca_2018"]:
        meta = summaries.get(ds, {})
        rows.append(
            {
                "dataset": core.DATASET_LABEL[ds],
                "key": ds,
                "rows": meta.get("rows", ""),
                "target": meta.get("target_column", meta.get("loader_summary", {}).get("target_column", "")),
                "positive_rate": meta.get("positive_rate", ""),
                "protected_settings": "; ".join(meta.get("protected_settings", [])),
                "sha256": meta.get("sha256", ""),
                "sample_policy": meta.get("sample_policy", "full dataset" if ds != "acs_income_ca_2018" else ""),
            }
        )
    return pd.DataFrame(rows)


def table_model_mitigation_scope(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in core.EXPECTED_MODELS:
        sub = df[df["model"] == model]
        mits = sorted(sub["mitigation"].unique())
        ds = sorted(sub["dataset"].unique())
        has_eg = any("ExponentiatedGradient" in m for m in mits)
        eg_ds = sorted(
            sub[sub["mitigation"].str.startswith("ExponentiatedGradient")]["dataset"].unique()
        )
        rows.append(
            {
                "model": core.MODEL_LABEL[model],
                "datasets": "; ".join(core.DATASET_LABEL[d] for d in ds),
                "aif360_mitigations": "baseline; reweighing; equalized_odds",
                "exponentiated_gradient": "yes" if has_eg else "no",
                "eg_datasets": "; ".join(core.DATASET_LABEL[d] for d in eg_ds) if has_eg else "--",
                "scope": "full" if has_eg and len(eg_ds) == 3 else ("limited" if has_eg else "main-grid only"),
            }
        )
    return pd.DataFrame(rows)


def table_baseline_accuracy_fairness(agg: pd.DataFrame) -> pd.DataFrame:
    sub = agg[agg["mitigation"] == "baseline"].copy()
    cols = [
        "dataset",
        "model",
        "protected_attribute",
        "mean_accuracy",
        "std_accuracy",
        "mean_f1",
        "mean_abs_spd",
        "mean_abs_eod",
        "mean_abs_aod",
        "mean_cfs",
        "std_cfs",
    ]
    out = sub[cols].copy()
    out["dataset"] = out["dataset"].map(core.DATASET_LABEL)
    out["model"] = out["model"].map(core.MODEL_LABEL)
    return out.round(4).sort_values(["dataset", "model", "protected_attribute"])


def table_mitigation_tradeoffs(agg: pd.DataFrame) -> pd.DataFrame:
    """Change in accuracy and CFS relative to baseline, per dataset/model/protected."""
    rows = []
    base = agg[agg["mitigation"] == "baseline"].set_index(
        ["dataset", "model", "protected_attribute"]
    )
    for mit in ["reweighing", "equalized_odds"]:
        cur = agg[agg["mitigation"] == mit].set_index(
            ["dataset", "model", "protected_attribute"]
        )
        for idx in cur.index:
            if idx not in base.index:
                continue
            rows.append(
                {
                    "dataset": core.DATASET_LABEL[idx[0]],
                    "model": core.MODEL_LABEL[idx[1]],
                    "protected_attribute": idx[2],
                    "mitigation": core.MITIGATION_LABEL[mit],
                    "delta_accuracy": round(cur.loc[idx, "mean_accuracy"] - base.loc[idx, "mean_accuracy"], 4),
                    "delta_cfs": round(cur.loc[idx, "mean_cfs"] - base.loc[idx, "mean_cfs"], 4),
                    "delta_abs_spd": round(cur.loc[idx, "mean_abs_spd"] - base.loc[idx, "mean_abs_spd"], 4),
                    "baseline_cfs": round(base.loc[idx, "mean_cfs"], 4),
                    "mitigated_cfs": round(cur.loc[idx, "mean_cfs"], 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "protected_attribute", "mitigation"])


def table_eg_comparison(agg: pd.DataFrame) -> pd.DataFrame:
    eg = agg[agg["mitigation"].str.startswith("ExponentiatedGradient")].copy()
    cols = [
        "dataset",
        "model",
        "protected_attribute",
        "mitigation",
        "mean_accuracy",
        "std_accuracy",
        "mean_abs_spd",
        "mean_abs_eod",
        "mean_cfs",
        "std_cfs",
    ]
    out = eg[cols].copy()
    out["dataset"] = out["dataset"].map(core.DATASET_LABEL)
    out["model"] = out["model"].map(core.MODEL_LABEL)
    out["mitigation"] = out["mitigation"].map(core.MITIGATION_LABEL)
    return out.round(4).sort_values(["dataset", "model", "protected_attribute", "mitigation"])


def table_runtime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["run_block", "dataset", "model", "mitigation"]):
        rows.append(
            {
                "run_block": keys[0],
                "dataset": core.DATASET_LABEL[keys[1]],
                "model": core.MODEL_LABEL[keys[2]],
                "mitigation": core.MITIGATION_LABEL.get(keys[3], keys[3]),
                "n": int(len(g)),
                "mean_runtime_s": round(g["runtime_seconds"].mean(), 3),
                "sum_runtime_s": round(g["runtime_seconds"].sum(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["run_block", "dataset", "model", "mitigation"])


def table_rq_alignment() -> pd.DataFrame:
    rows = [
        {
            "research_question": "RQ1: Do fairness interventions reduce disparity across datasets?",
            "evidence": "Mitigation trade-off table; paired baseline-vs-mitigation tests on CFS/|SPD|.",
            "scope": "3 datasets, 5 models, 3 AIF360 mitigations, 30 seeds.",
        },
        {
            "research_question": "RQ2: How do in-processing (EG) and pre/post-processing methods compare?",
            "evidence": "EG comparison table; EG vs AIF360 figure; LR full scope, RF Adult+Bank.",
            "scope": "EG on LR (all datasets) and RF (Adult, UCI Bank).",
        },
        {
            "research_question": "RQ3: Are accuracy-fairness trade-offs consistent across model families?",
            "evidence": "Pareto front; five-model main grid; variability boxplots.",
            "scope": "LR, RF, GB, XGBoost, MLP.",
        },
        {
            "research_question": "RQ4: Does the composite score (CFS/ACFS) support model selection?",
            "evidence": "CFS/ACFS rankings; best model-mitigation per setting; ACFS presets.",
            "scope": "All settings, 4 ACFS weighting presets.",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Supplementary tables
# --------------------------------------------------------------------------


def table_acfs_rankings(df: pd.DataFrame) -> pd.DataFrame:
    presets = list(ACFS_WEIGHT_PRESETS.keys())
    rows = []
    for keys, g in df.groupby(core.GROUP_COLS):
        rec = dict(zip(core.GROUP_COLS, keys))
        rec["mean_accuracy"] = round(g["accuracy"].mean(), 4)
        for p in presets:
            rec[f"mean_acfs_{p}"] = round(g[f"acfs_{p}"].mean(), 4)
        rows.append(rec)
    out = pd.DataFrame(rows)
    for p in presets:
        out[f"rank_acfs_{p}"] = out.groupby(["dataset", "protected_attribute"])[
            f"mean_acfs_{p}"
        ].rank(method="min")
    return out.sort_values(["dataset", "protected_attribute", "mean_acfs_balanced"])


def table_pareto(agg: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for keys, g in agg.groupby(["dataset", "protected_attribute"]):
        front = core.pareto_front(g, "mean_accuracy", "mean_cfs").copy()
        front["dataset"] = keys[0]
        front["protected_attribute"] = keys[1]
        frames.append(front)
    out = pd.concat(frames, ignore_index=True)
    cols = [
        "dataset",
        "protected_attribute",
        "model",
        "mitigation",
        "mean_accuracy",
        "mean_cfs",
        "mean_abs_spd",
    ]
    out = out[cols].copy()
    out["dataset"] = out["dataset"].map(core.DATASET_LABEL)
    out["model"] = out["model"].map(core.MODEL_LABEL)
    out["mitigation"] = out["mitigation"].map(core.MITIGATION_LABEL)
    return out.round(4).sort_values(["dataset", "protected_attribute", "mean_accuracy"], ascending=[True, True, False])


def table_bounded_di(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Fraction of seeds meeting the four-fifths (0.8) disparate-impact rule."""
    rows = []
    for keys, g in df.groupby(core.GROUP_COLS):
        di = g["di"].to_numpy()
        within = np.mean((di >= threshold) & (di <= 1.0 / threshold))
        rec = dict(zip(core.GROUP_COLS, keys))
        rec["mean_di"] = round(float(np.mean(di)), 4)
        rec["frac_seeds_within_4_5_rule"] = round(float(within), 4)
        rec["n_seeds"] = int(len(g))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(core.GROUP_COLS)


def table_seed_variability(agg: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "model",
        "mitigation",
        "protected_attribute",
        "std_accuracy",
        "std_f1",
        "std_spd",
        "std_cfs",
        "n_seeds",
    ]
    out = agg[cols].copy()
    return out.round(4).sort_values(["dataset", "model", "mitigation", "protected_attribute"])


def table_best_pairs(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in agg.groupby(["dataset", "protected_attribute"]):
        best = g.loc[g["mean_cfs"].idxmin()]
        best_acc = g.loc[g["mean_accuracy"].idxmax()]
        rows.append(
            {
                "dataset": core.DATASET_LABEL[keys[0]],
                "protected_attribute": keys[1],
                "best_fairness_model": core.MODEL_LABEL[best["model"]],
                "best_fairness_mitigation": core.MITIGATION_LABEL[best["mitigation"]],
                "best_fairness_cfs": round(best["mean_cfs"], 4),
                "best_fairness_accuracy": round(best["mean_accuracy"], 4),
                "best_accuracy_model": core.MODEL_LABEL[best_acc["model"]],
                "best_accuracy_mitigation": core.MITIGATION_LABEL[best_acc["mitigation"]],
                "best_accuracy_value": round(best_acc["mean_accuracy"], 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def save_fig(fig, name: str) -> None:
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(core.FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(core.FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_pareto(agg: pd.DataFrame) -> None:
    datasets = core.EXPECTED_DATASETS
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    markers = {
        "baseline": "o",
        "reweighing": "s",
        "equalized_odds": "^",
        "ExponentiatedGradient_DP": "D",
        "ExponentiatedGradient_EO": "P",
    }
    for ax, ds in zip(axes, datasets):
        sub = agg[agg["dataset"] == ds]
        for mit, mk in markers.items():
            s = sub[sub["mitigation"] == mit]
            if s.empty:
                continue
            ax.scatter(
                s["mean_accuracy"].to_numpy(),
                s["mean_cfs"].to_numpy(),
                marker=mk,
                s=45,
                alpha=0.75,
                label=core.MITIGATION_LABEL[mit],
            )
        front = core.pareto_front(sub, "mean_accuracy", "mean_cfs").sort_values("mean_accuracy")
        ax.plot(front["mean_accuracy"].to_numpy(), front["mean_cfs"].to_numpy(), "k--", lw=1, alpha=0.6)
        ax.set_title(core.DATASET_LABEL[ds])
        ax.set_xlabel("Mean accuracy")
        ax.set_ylabel("Mean CFS (lower = fairer)")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Accuracy vs. composite fairness (30-seed means)")
    fig.tight_layout()
    save_fig(fig, "fig_pareto_accuracy_fairness")


def fig_mitigation_slope(agg: pd.DataFrame) -> None:
    order = ["baseline", "reweighing", "equalized_odds"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, core.EXPECTED_DATASETS):
        sub = agg[(agg["dataset"] == ds) & (agg["mitigation"].isin(order))]
        for (model, prot), g in sub.groupby(["model", "protected_attribute"]):
            g = g.set_index("mitigation").reindex(order)
            ax.plot(np.arange(len(order)), g["mean_cfs"].to_numpy(), marker="o", alpha=0.6,
                    label=f"{core.MODEL_LABEL[model]}/{prot}")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([core.MITIGATION_LABEL[m] for m in order], rotation=20)
        ax.set_title(core.DATASET_LABEL[ds])
        ax.set_ylabel("Mean CFS")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Mitigation trade-off slopes (composite fairness)")
    fig.tight_layout()
    save_fig(fig, "fig_mitigation_tradeoff_slope")


def fig_variability_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    order = core.EXPECTED_MITIGATIONS
    data = [df[df["mitigation"] == m]["cfs"].to_numpy() for m in order]
    ax.boxplot(data, labels=[core.MITIGATION_LABEL[m] for m in order], showmeans=True)
    ax.set_ylabel("CFS across all settings and seeds")
    ax.set_title("Composite fairness variability by mitigation")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_fairness_variability_box")


def fig_dataset_heatmap(agg: pd.DataFrame) -> None:
    piv = agg[agg["mitigation"] == "baseline"].pivot_table(
        index="model", columns="dataset", values="mean_cfs", aggfunc="mean"
    )
    piv = piv[[d for d in core.EXPECTED_DATASETS if d in piv.columns]]
    fig, ax = plt.subplots(figsize=(8, 6))
    piv_vals = piv.to_numpy()
    im = ax.imshow(piv_vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([core.DATASET_LABEL[c] for c in piv.columns], rotation=15)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([core.MODEL_LABEL[m] for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean baseline CFS")
    ax.set_title("Baseline composite fairness by model and dataset")
    fig.tight_layout()
    save_fig(fig, "fig_dataset_comparison_heatmap")


def fig_eg_vs_aif(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = agg[agg["model"].isin(["logistic_regression", "random_forest"])].copy()
    sub["group"] = sub["dataset"].map(core.DATASET_LABEL) + " / " + sub["model"].map(core.MODEL_LABEL) + " / " + sub["protected_attribute"]
    mit_order = ["baseline", "reweighing", "equalized_odds", "ExponentiatedGradient_DP", "ExponentiatedGradient_EO"]
    groups = sorted(sub["group"].unique())
    x = np.arange(len(groups))
    width = 0.16
    for i, mit in enumerate(mit_order):
        vals = []
        for grp in groups:
            r = sub[(sub["group"] == grp) & (sub["mitigation"] == mit)]
            vals.append(r["mean_cfs"].iloc[0] if not r.empty else np.nan)
        ax.bar(x + (i - 2) * width, vals, width, label=core.MITIGATION_LABEL[mit])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.set_ylabel("Mean CFS")
    ax.set_title("EG vs. AIF360 mitigations (LR and RF)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_eg_vs_aif360")


def fig_runtime(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df.groupby(["model", "mitigation"])["runtime_seconds"].mean().unstack()
    means = means.reindex(core.EXPECTED_MODELS)
    means.plot(kind="bar", ax=ax, logy=True)
    ax.set_ylabel("Mean runtime (s, log scale)")
    ax.set_xticklabels([core.MODEL_LABEL[m] for m in means.index], rotation=20)
    ax.set_title("Mean runtime by model and mitigation")
    ax.legend([core.MITIGATION_LABEL.get(c, c) for c in means.columns], fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_runtime_comparison")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    core.ensure_dirs()
    df_raw = core.load_results()
    validation = core.validate(df_raw)
    (core.OUT_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2, default=str), encoding="utf-8"
    )

    df = core.add_derived(df_raw)
    summaries = core.load_json(core.DATASET_SUMMARIES_JSON)
    completion = core.load_json(core.COMPLETION_JSON)

    agg = core.build_aggregate(df)
    save_table(agg, "full_aggregate_30seed", core.SUPP_DIR, "Full 30-seed aggregate table", latex=False)

    # Main tables
    save_table(table_dataset_provenance(summaries), "main_table_dataset_provenance", core.TABLES_DIR, "Dataset provenance")
    save_table(table_model_mitigation_scope(df), "main_table_model_mitigation_scope", core.TABLES_DIR, "Model and mitigation scope")
    save_table(table_baseline_accuracy_fairness(agg), "main_table_baseline_accuracy_fairness", core.TABLES_DIR, "Baseline accuracy and fairness (30-seed means)")
    save_table(table_mitigation_tradeoffs(agg), "main_table_mitigation_tradeoffs", core.TABLES_DIR, "Mitigation trade-offs relative to baseline")
    save_table(table_eg_comparison(agg), "main_table_eg_comparison", core.TABLES_DIR, "Exponentiated Gradient comparison")
    save_table(table_runtime_summary(df), "main_table_runtime_summary", core.TABLES_DIR, "Runtime summary")
    save_table(table_rq_alignment(), "main_table_research_question_alignment", core.TABLES_DIR, "Research question alignment")

    # Supplementary tables
    save_table(df_raw, "supp_per_seed_full_results", core.SUPP_DIR, "Per-seed full results", latex=False)
    save_table(table_acfs_rankings(df), "supp_acfs_rankings", core.SUPP_DIR, "ACFS rankings by preset")
    save_table(table_pareto(agg), "supp_pareto_optimal", core.SUPP_DIR, "Pareto-optimal configurations")
    save_table(table_bounded_di(df), "supp_bounded_di_robustness", core.SUPP_DIR, "Four-fifths rule robustness")
    save_table(table_seed_variability(agg), "supp_seed_variability", core.SUPP_DIR, "Seed-level variability")
    save_table(table_best_pairs(agg), "supp_best_model_mitigation_pairs", core.SUPP_DIR, "Best model-mitigation pairs")
    save_table(table_model_mitigation_scope(df), "supp_model_scope", core.SUPP_DIR, "Model scope", latex=False)

    # Statistical analysis
    stats_frames = []
    for metric in ["cfs", "abs_spd", "accuracy"]:
        res = core.paired_comparison(df, metric)
        if not res.empty:
            stats_frames.append(res)
    if stats_frames:
        stats_all = pd.concat(stats_frames, ignore_index=True)
        save_table(stats_all, "supp_paired_statistical_tests", core.SUPP_DIR, "Paired statistical comparisons with Holm correction", latex=False)
    else:
        stats_all = pd.DataFrame()

    # Figures
    fig_pareto(agg)
    fig_mitigation_slope(agg)
    fig_variability_box(df)
    fig_dataset_heatmap(agg)
    fig_eg_vs_aif(agg)
    fig_runtime(df)

    manifest = {
        "phase": "5A",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
        "n_aggregate_cells": int(len(agg)),
        "n_stat_comparisons": int(len(stats_all)),
        "main_tables": sorted(p.stem for p in core.TABLES_DIR.glob("*.csv")),
        "supplementary_tables": sorted(p.stem for p in core.SUPP_DIR.glob("*.csv")),
        "figures": sorted(p.name for p in core.FIG_DIR.glob("*.png")),
        "completion_summary": completion,
    }
    (core.OUT_DIR / "phase5a_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    print(f"Phase 5A: validation passed = {validation['all_checks_passed']}", flush=True)
    print(f"Aggregate cells: {len(agg)}; stat comparisons: {len(stats_all)}", flush=True)
    print(f"Figures: {len(manifest['figures'])}", flush=True)


if __name__ == "__main__":
    main()
