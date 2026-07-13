"""Build optional SI table for Adult sex×race mitigated subgroup sensitivity."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "q1_upgrade" / "results" / "subgroup_mitigation_sensitivity"
OUT_SI = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
    / "si_tables"
    / "s11_subgroup_mitigation_sensitivity.tex"
)

LABELS = {
    "baseline": "Baseline",
    "equalized_odds_sex": "EO (fitted on sex)",
    "equalized_odds_race": "EO (fitted on race)",
}


def load_summary() -> pd.DataFrame:
    full = RESULTS / "subgroup_mitigation_full_summary.csv"
    if full.exists():
        return pd.read_csv(full)
    return pd.read_csv(RESULTS / "subgroup_mitigation_feasibility_summary.csv")


def main() -> None:
    summ = load_summary()
    if summ.empty:
        raise FileNotFoundError("No summary CSV; run run_subgroup_mitigation_sensitivity.py first")

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Optional Adult \texttt{sex}$\times$\texttt{race} mitigated subgroup sensitivity "
        r"(not part of the 3{,}690-run main grid). Equalized Odds is fitted on a single marginal "
        r"binary attribute (\texttt{sex} or \texttt{race}); evaluation uses four \texttt{sex}$\times$\texttt{race} "
        r"cells on the test partition (30 seeds 42--71). Illustrative only; not full subgroup mitigation.}",
        r"\label{tab:si-subgroup-mitigation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Condition} & \textbf{Acc.} & Marg.\ $|\mathrm{SPD}|$ & Marg.\ $|\mathrm{EOD}|$ "
        r"& \textbf{PR gap} & \textbf{TPR gap} & \textbf{Min cell} & \textbf{Seeds} \\",
        r"\midrule",
    ]
    order = ["baseline", "equalized_odds_sex", "equalized_odds_race"]
    for model in ["logistic_regression", "xgboost"]:
        model_label = "Logistic Regression" if model == "logistic_regression" else "XGBoost"
        first = True
        for cid in order:
            row = summ[(summ["model"] == model) & (summ["condition_id"] == cid)]
            if row.empty:
                continue
            row = row.iloc[0]
            cond = LABELS.get(cid, cid)
            model_cell = model_label if first else ""
            first = False
            lines.append(
                f"{model_cell} & {cond} & {row['mean_accuracy']:.3f} & "
                f"{row['mean_marginal_abs_spd']:.3f} & {row['mean_marginal_abs_eod']:.3f} & "
                f"{row['mean_subgroup_pr_gap']:.3f} & {row['mean_subgroup_tpr_gap']:.3f} & "
                f"{int(row['min_cell_n_overall'])} & {int(row['n_seeds'])} \\\\"
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
