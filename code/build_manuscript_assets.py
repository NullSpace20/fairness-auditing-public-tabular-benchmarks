"""Build Q1-upgrade manuscript LaTeX tables and copy figures from Phase 5A outputs.

Reads only Phase 5A CSVs. Writes into Paper_Springer_JBigData_Q1Upgrade/.
Does not touch original manuscript folders.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import phase5a_core as core
from pipeline_core import Q1_ROOT

PROJECT_ROOT = Q1_ROOT.parent
PP = Q1_ROOT / "results" / "phase5_postprocessing"
TABLES = PP / "tables"
SUPP = PP / "supplementary"
FIGS = PP / "figures"

MS = PROJECT_ROOT / "Paper_Springer_JBigData_Q1Upgrade"
MS_TABLES = MS / "Q1_Tables"
MS_FIGS = MS / "Q1_Figures"


def esc(x) -> str:
    s = str(x)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def tabular(df: pd.DataFrame, colspec: str, headers, rows_fmt) -> str:
    lines = [
        "\\begin{tabular}{" + colspec + "}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(rows_fmt(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def wrap_table(inner: str, caption: str, label: str, resize: bool = True, small: bool = False) -> str:
    pre = "\\begin{table}[htbp]\n\\centering\n"
    if small:
        pre += "\\footnotesize\n"
    pre += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    if resize:
        pre += "\\resizebox{\\textwidth}{!}{%\n" + inner + "\n}\n"
    else:
        pre += inner + "\n"
    pre += "\\end{table}\n"
    return pre


def write(name: str, content: str) -> None:
    MS_TABLES.mkdir(parents=True, exist_ok=True)
    (MS_TABLES / f"{name}.tex").write_text(content, encoding="utf-8")


def build_dataset_provenance() -> None:
    df = pd.read_csv(TABLES / "main_table_dataset_provenance.csv")
    def fmt(r):
        return " & ".join([
            esc(r["dataset"]), f"\\texttt{{{esc(r['key'])}}}", str(r["rows"]),
            esc(r["target"]), f"{float(r['positive_rate']):.4f}",
            esc(r["protected_settings"]),
            f"\\texttt{{{esc(str(r['sha256'])[:12])}}}" if str(r["sha256"]) not in ("", "nan") else "--",
        ])
    inner = tabular(
        df, "lllllll",
        ["Dataset", "Key", "Rows", "Target", "Pos. rate", "Protected settings", "SHA-256 (first 12)"],
        fmt,
    )
    note = (
        "\n\\vspace{2pt}\n{\\footnotesize \\textbf{Note.} ACSIncome is a 2018 ACS "
        "California 1-Year sample. The full California pool contains 195{,}665 person "
        "records; a stratified subsample capped at 50{,}000 rows (final 49{,}999) is drawn "
        "with a fixed seed, and experiment seeds affect only train/test splits. UCI Bank "
        "is the canonical \\texttt{bank-additional-full.csv}; \\texttt{duration} is dropped "
        "before training.}\n"
    )
    write("tab_dataset_provenance", wrap_table(inner, "Dataset provenance and fingerprints for the exact files used in all 3{,}690 runs.", "tab:dataset-summary") + note)


def build_model_scope() -> None:
    df = pd.read_csv(TABLES / "main_table_model_mitigation_scope.csv")
    def fmt(r):
        return " & ".join([
            esc(r["model"]), esc(r["datasets"]), esc(r["aif360_mitigations"]),
            esc(r["exponentiated_gradient"]), esc(r["eg_datasets"]), esc(r["scope"]),
        ])
    inner = tabular(
        df, "lp{3.2cm}p{3.0cm}llp{2.2cm}",
        ["Model", "Datasets", "AIF360 mitigations", "EG", "EG datasets", "Scope"],
        fmt,
    )
    write("tab_model_scope", wrap_table(inner, "Model and mitigation scope. Exponentiated Gradient is restricted to Logistic Regression (all datasets) and Random Forest (Adult, UCI Bank).", "tab:model-scope"))


def build_baseline() -> None:
    df = pd.read_csv(TABLES / "main_table_baseline_accuracy_fairness.csv")
    df = df.sort_values(["dataset", "model", "protected_attribute"])
    def fmt(r):
        return " & ".join([
            esc(r["dataset"]), esc(r["model"]), esc(r["protected_attribute"]),
            f"{r['mean_accuracy']:.3f}", f"{r['mean_f1']:.3f}",
            f"{r['mean_abs_spd']:.3f}", f"{r['mean_abs_eod']:.3f}",
            f"{r['mean_cfs']:.3f}", f"{r['std_cfs']:.3f}",
        ])
    inner = tabular(
        df, "lllrrrrrr",
        ["Dataset", "Model", "Protected", "Acc", "F1", "|SPD|", "|EOD|", "CFS", "SD(CFS)"],
        fmt,
    )
    write("tab_baseline", wrap_table(inner, "Baseline predictive performance and fairness, 30-seed means. CFS is the balanced composite fairness score; lower is fairer.", "tab:baseline-fair", small=True))


def build_tradeoffs() -> None:
    df = pd.read_csv(TABLES / "main_table_mitigation_tradeoffs.csv")
    summ = df.groupby("mitigation")[["delta_accuracy", "delta_cfs", "delta_abs_spd"]].mean().reset_index()
    def fmt(r):
        return " & ".join([
            esc(r["mitigation"]), f"{r['delta_accuracy']:+.4f}",
            f"{r['delta_cfs']:+.4f}", f"{r['delta_abs_spd']:+.4f}",
        ])
    inner = tabular(
        summ, "lrrr",
        ["Mitigation", "$\\Delta$ Accuracy", "$\\Delta$ CFS", "$\\Delta$ |SPD|"],
        fmt,
    )
    write("tab_tradeoffs", wrap_table(inner, "Mean change relative to baseline, averaged over all dataset/model/protected settings and 30 seeds. Negative CFS and |SPD| indicate reduced fairness violation.", "tab:mitigation-tradeoffs", resize=False))


def build_eg() -> None:
    df = pd.read_csv(TABLES / "main_table_eg_comparison.csv")
    df = df.sort_values(["dataset", "model", "protected_attribute", "mitigation"])
    def fmt(r):
        return " & ".join([
            esc(r["dataset"]), esc(r["model"]), esc(r["protected_attribute"]),
            esc(r["mitigation"]), f"{r['mean_accuracy']:.3f}",
            f"{r['mean_abs_spd']:.3f}", f"{r['mean_abs_eod']:.3f}", f"{r['mean_cfs']:.3f}",
        ])
    inner = tabular(
        df, "llllrrrr",
        ["Dataset", "Model", "Protected", "Constraint", "Acc", "|SPD|", "|EOD|", "CFS"],
        fmt,
    )
    write("tab_eg", wrap_table(inner, "Exponentiated Gradient results, 30-seed means. EG covers Logistic Regression on all datasets and Random Forest on Adult (sex) and UCI Bank (age\\_group) only.", "tab:eg-comparison", small=True))


def build_runtime() -> None:
    df = pd.read_csv(TABLES / "main_table_runtime_summary.csv")
    g = df.groupby("model").agg(
        total_runtime=("sum_runtime_s", "sum"),
        n=("n", "sum"),
    ).reset_index()
    g["mean_runtime"] = g["total_runtime"] / g["n"]
    g = g[["model", "n", "mean_runtime", "total_runtime"]]
    def fmt(r):
        return " & ".join([
            esc(r["model"]), str(int(r["n"])),
            f"{r['mean_runtime']:.2f}", f"{r['total_runtime']/3600:.2f}",
        ])
    inner = tabular(
        g, "lrrr",
        ["Model", "Runs", "Mean per run (s)", "Total (h)"],
        fmt,
    )
    write("tab_runtime", wrap_table(inner, "Runtime by model (mean per run and total). The Random Forest mean is inflated by its Exponentiated Gradient runs; among the main-grid families the multilayer perceptron is the most expensive.", "tab:runtime", resize=False))


def build_pareto() -> None:
    df = pd.read_csv(SUPP / "supp_pareto_optimal.csv")
    def fmt(r):
        return " & ".join([
            esc(r["dataset"]), esc(r["protected_attribute"]), esc(r["model"]),
            esc(r["mitigation"]), f"{r['mean_accuracy']:.3f}", f"{r['mean_cfs']:.3f}",
        ])
    inner = tabular(
        df, "llllrr",
        ["Dataset", "Protected", "Model", "Mitigation", "Acc", "CFS"],
        fmt,
    )
    write("tab_pareto", wrap_table(inner, "Pareto-optimal accuracy--fairness configurations per dataset and protected setting (30-seed means).", "tab:pareto-optimal", small=True))


def build_best_pairs() -> None:
    df = pd.read_csv(SUPP / "supp_best_model_mitigation_pairs.csv")
    def fmt(r):
        return " & ".join([
            esc(r["dataset"]), esc(r["protected_attribute"]),
            esc(r["best_fairness_model"]), esc(r["best_fairness_mitigation"]),
            f"{r['best_fairness_cfs']:.3f}", f"{r['best_fairness_accuracy']:.3f}",
            esc(r["best_accuracy_model"]), f"{r['best_accuracy_value']:.3f}",
        ])
    inner = tabular(
        df, "llllrrlr",
        ["Dataset", "Protected", "Fairest model", "Fairest mit.", "CFS", "Acc",
         "Most acc. model", "Acc"],
        fmt,
    )
    write("tab_best_pairs", wrap_table(inner, "Lowest-CFS and highest-accuracy configurations per setting (30-seed means).", "tab:best-pairs", small=True))


def build_stats() -> None:
    df = pd.read_csv(SUPP / "supp_paired_statistical_tests.csv")
    cfs = df[df["metric"] == "cfs"].copy()
    rows = []
    for comp, g in cfs.groupby("comparison"):
        n_sig = int((g["wilcoxon_p_holm"] < 0.05).sum())
        rows.append({
            "comparison": comp,
            "n": len(g),
            "n_sig": n_sig,
            "mean_diff": g["mean_diff"].mean(),
            "mean_dz": g["cohen_dz"].mean(),
        })
    out = pd.DataFrame(rows)
    label = {
        "reweighing": "Reweighing",
        "equalized_odds": "Equalized Odds",
        "ExponentiatedGradient_DP": "EG (DP)",
        "ExponentiatedGradient_EO": "EG (EO)",
    }
    out["comparison"] = out["comparison"].map(label)
    def fmt(r):
        return " & ".join([
            esc(r["comparison"]), f"{r['n_sig']}/{r['n']}",
            f"{r['mean_diff']:+.3f}", f"{r['mean_dz']:+.2f}",
        ])
    inner = tabular(
        out, "lrrr",
        ["Comparison (vs baseline)", "Sig. (Holm)", "Mean $\\Delta$CFS", "Mean $d_z$"],
        fmt,
    )
    write("tab_stats", wrap_table(inner, "Paired comparisons of CFS against baseline across 30 seeds. Significance is Holm-adjusted Wilcoxon within each comparison family; $d_z$ is Cohen's paired effect size.", "tab:significance", resize=False))


def copy_figures() -> None:
    MS_FIGS.mkdir(parents=True, exist_ok=True)
    for pdf in FIGS.glob("*.pdf"):
        shutil.copy2(pdf, MS_FIGS / pdf.name)


# --------------------------------------------------------------------------
# Journal-compliant figure regeneration (titles moved to LaTeX captions;
# panel/dataset labels kept as in-graphic keys). Written into the manuscript
# figure folder only; Phase 5A figure outputs are left untouched.
# --------------------------------------------------------------------------

DS_SHORT = {"adult": "Adult", "bank_uci": "Bank", "acs_income_ca_2018": "ACS"}
MODEL_SHORT = {
    "logistic_regression": "LR", "random_forest": "RF",
    "gradient_boosting": "GB", "xgboost": "XGB", "mlp": "MLP",
}


def _save(fig, name: str) -> None:
    MS_FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(MS_FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(MS_FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_agg() -> pd.DataFrame:
    return pd.read_csv(SUPP / "full_aggregate_30seed.csv")


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(Q1_ROOT / "results" / "phase4b_optionB_30seed" / "phase4b_results.csv")
    return df


def fig_pipeline() -> None:
    stages = [
        ("Data", "Adult + UCI Bank + ACSIncome\n(ACSIncome: 2018 CA 50k sample)"),
        ("Preprocessing", "one-hot + scaling;\ndrop Bank duration;\n80/20 stratified split"),
        ("Models + mitigations", "5 models + scoped mitigations\n(Baseline, Reweighing, EO;\nEG-DP/EO where in scope)"),
        ("Fairness + accuracy", "SPD, DI, EOD, AOD;\naccuracy, F1; CFS / ACFS"),
        ("Repeated splits", "30 seeds (42\u201371);\nbootstrap CIs, Cohen's $d_z$,\nHolm-adjusted paired tests"),
        ("Ranking + audit", "Pareto frontier,\ncomposite ranking,\naudit-support selection"),
    ]
    palette = [
        ("#dbeafe", "#1d4ed8"),
        ("#e0f2fe", "#0369a1"),
        ("#ede9fe", "#6d28d9"),
        ("#f3e8ff", "#7e22ce"),
        ("#fef3c7", "#b45309"),
        ("#dcfce7", "#15803d"),
    ]
    fig, ax = plt.subplots(figsize=(14.2, 4.8))
    ax.axis("off")
    n = len(stages)
    bw, bh, gap = 1.55, 1.7, 0.48
    y0 = 0.15
    for i, ((head, body), (fill, edge)) in enumerate(zip(stages, palette)):
        x = i * (bw + gap)
        card = FancyBboxPatch(
            (x, y0),
            bw,
            bh,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(card)
        accent = plt.Rectangle((x, y0 + bh - 0.18), bw, 0.18, facecolor=edge, edgecolor=edge, lw=0)
        ax.add_patch(accent)
        ax.text(
            x + 0.10,
            y0 + bh - 0.30,
            head,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            x + 0.10,
            y0 + bh - 0.58,
            body,
            ha="left",
            va="top",
            fontsize=7.35,
            color="#1f2937",
            linespacing=1.18,
        )
        if i < n - 1:
            ymid = y0 + bh / 2
            ax.annotate(
                "",
                xy=(x + bw + gap - 0.06, ymid),
                xytext=(x + bw + 0.05, ymid),
                arrowprops=dict(arrowstyle="-|>", color="#4b5563", lw=1.5, shrinkA=0, shrinkB=0),
            )
    ax.set_xlim(-0.15, n * (bw + gap) - 0.25)
    ax.set_ylim(0, 2.1)
    fig.tight_layout()
    _save(fig, "fig_pipeline_workflow")


def fig_pareto(agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    markers = {"baseline": "o", "reweighing": "s", "equalized_odds": "^",
               "ExponentiatedGradient_DP": "D", "ExponentiatedGradient_EO": "P"}
    for ax, ds in zip(axes, core.EXPECTED_DATASETS):
        sub = agg[agg["dataset"] == ds]
        for mit, mk in markers.items():
            s = sub[sub["mitigation"] == mit]
            if s.empty:
                continue
            ax.scatter(s["mean_accuracy"].to_numpy(), s["mean_cfs"].to_numpy(),
                       marker=mk, s=45, alpha=0.75, label=core.MITIGATION_LABEL[mit])
        front = core.pareto_front(sub, "mean_accuracy", "mean_cfs").sort_values("mean_accuracy")
        ax.plot(front["mean_accuracy"].to_numpy(), front["mean_cfs"].to_numpy(), "k--", lw=1, alpha=0.6)
        ax.set_title(core.DATASET_LABEL[ds])  # panel key (dataset), kept
        ax.set_xlabel("Mean accuracy")
        ax.set_ylabel("Mean CFS (lower = fairer)")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    _save(fig, "fig_pareto_accuracy_fairness")


def fig_mitigation_slope(agg: pd.DataFrame) -> None:
    order = ["baseline", "reweighing", "equalized_odds"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, core.EXPECTED_DATASETS):
        sub = agg[(agg["dataset"] == ds) & (agg["mitigation"].isin(order))]
        for (model, prot), g in sub.groupby(["model", "protected_attribute"]):
            g = g.set_index("mitigation").reindex(order)
            ax.plot(np.arange(len(order)), g["mean_cfs"].to_numpy(), marker="o", alpha=0.6,
                    label=f"{MODEL_SHORT[model]}/{prot}")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([core.MITIGATION_LABEL[m] for m in order], rotation=20)
        ax.set_title(core.DATASET_LABEL[ds])  # panel key (dataset), kept
        ax.set_ylabel("Mean CFS")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_mitigation_tradeoff_slope")


def fig_variability_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    order = core.EXPECTED_MITIGATIONS
    data = [df[df["mitigation"] == m]["cfs"].to_numpy() for m in order]
    ax.boxplot(data, labels=[core.MITIGATION_LABEL[m] for m in order], showmeans=True)
    ax.set_ylabel("CFS across all settings and seeds")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_fairness_variability_box")


def fig_dataset_heatmap(agg: pd.DataFrame) -> None:
    piv = agg[agg["mitigation"] == "baseline"].pivot_table(
        index="model", columns="dataset", values="mean_cfs", aggfunc="mean")
    piv = piv[[d for d in core.EXPECTED_DATASETS if d in piv.columns]]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([core.DATASET_LABEL[c] for c in piv.columns], rotation=15)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([core.MODEL_LABEL[m] for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean baseline CFS")
    fig.tight_layout()
    _save(fig, "fig_dataset_comparison_heatmap")


def fig_eg_vs_aif(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = agg[agg["model"].isin(["logistic_regression", "random_forest"])].copy()
    sub["group"] = (sub["dataset"].map(DS_SHORT) + "\u00b7" + sub["model"].map(MODEL_SHORT)
                    + "\u00b7" + sub["protected_attribute"])
    mit_order = ["baseline", "reweighing", "equalized_odds",
                 "ExponentiatedGradient_DP", "ExponentiatedGradient_EO"]
    # Keep only settings where EG was actually applied (the EG comparison target).
    eg_groups = set(sub[sub["mitigation"].str.startswith("ExponentiatedGradient")]["group"])
    sub = sub[sub["group"].isin(eg_groups)]
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
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean CFS")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_eg_vs_aif360")


def fig_runtime(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df.groupby(["model", "mitigation"])["runtime_seconds"].mean().unstack()
    means = means.reindex(core.EXPECTED_MODELS)
    means.plot(kind="bar", ax=ax, logy=True)
    ax.set_ylabel("Mean runtime (s, log scale)")
    ax.set_xticklabels([core.MODEL_LABEL[m] for m in means.index], rotation=20)
    ax.legend([core.MITIGATION_LABEL.get(c, c) for c in means.columns], fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_runtime_comparison")


def build_figures() -> None:
    agg = _load_agg()
    raw = _load_raw()
    fig_pipeline()
    fig_pareto(agg)
    fig_mitigation_slope(agg)
    fig_variability_box(raw)
    fig_dataset_heatmap(agg)
    fig_eg_vs_aif(agg)
    fig_runtime(raw)


def main() -> None:
    if not MS.exists():
        raise FileNotFoundError(f"Manuscript folder missing: {MS}")
    build_dataset_provenance()
    build_model_scope()
    build_baseline()
    build_tradeoffs()
    build_eg()
    build_runtime()
    build_pareto()
    build_best_pairs()
    build_stats()
    build_figures()
    print("Tables:", sorted(p.name for p in MS_TABLES.glob("*.tex")))
    print("Figures:", sorted(p.name for p in MS_FIGS.glob("*.pdf")))


if __name__ == "__main__":
    main()
