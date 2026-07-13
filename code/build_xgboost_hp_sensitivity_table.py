"""Build optional SI table for Adult XGBoost hyperparameter sensitivity."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "q1_upgrade" / "results" / "xgboost_hyperparameter_sensitivity"
OUT_SI = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
    / "si_tables"
    / "s10_xgboost_hp_sensitivity.tex"
)


def load_summary() -> pd.DataFrame:
    full = RESULTS / "xgboost_hp_sensitivity_full_summary.csv"
    if full.exists():
        return pd.read_csv(full)
    return pd.read_csv(RESULTS / "xgboost_hp_sensitivity_feasibility_summary.csv")


def main() -> None:
    summ = load_summary()
    if summ.empty:
        raise FileNotFoundError("No summary CSV found; run run_xgboost_hyperparameter_sensitivity.py first")

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Optional Adult XGBoost hyperparameter sensitivity (baseline only; not part of the 3{,}690-run main grid). "
        r"Thirty seeds (42--71); alternative configurations are pre-specified overrides relative to the main-grid default.}",
        r"\label{tab:si-xgboost-hp-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{Protected} & \textbf{Configuration} & \textbf{Acc.} & \textbf{F1} & $|\mathrm{SPD}|$ & $|\mathrm{EOD}|$ & \textbf{CFS} & $\Delta$\textbf{CFS} \\",
        r"\midrule",
    ]
    for prot in ["sex", "race", "age_group"]:
        sub = summ[summ["protected_attribute"] == prot].copy()
        if sub.empty:
            continue
        prot_tex = prot.replace("_", r"\_")
        for i, row in sub.iterrows():
            label = row["config_label"].replace("_", r"\_")
            delta = row.get("mean_delta_cfs_vs_default", float("nan"))
            delta_s = f"{delta:+.3f}" if pd.notna(delta) else "--"
            prefix = prot_tex if i == sub.index[0] else ""
            if i != sub.index[0]:
                prefix = ""
            # group by protected: only show protected name on first row of group
            first_in_group = row.name == sub.index[0]
            prot_cell = prot_tex if first_in_group else ""
            lines.append(
                f"{prot_cell} & {label} & {row['mean_accuracy']:.3f} & {row['mean_f1']:.3f} & "
                f"{row['mean_abs_spd']:.3f} & {row['mean_abs_eod']:.3f} & "
                f"{row['mean_cfs']:.3f} & {delta_s} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    OUT_SI.parent.mkdir(parents=True, exist_ok=True)
    OUT_SI.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT_SI)


if __name__ == "__main__":
    main()
