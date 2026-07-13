"""Build optional SI table for extra XGBoost EG probe cell (W2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "q1_upgrade" / "results" / "xgboost_eg_probe_extra_cell"
OUT_SI = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
    / "si_tables"
    / "s12_extra_eg_probe_cell.tex"
)

CONSTRAINT_LABELS = {"DP": "EG-DP (extra probe)", "EO": "EG-EO (extra probe)"}


def load_summary() -> pd.DataFrame:
    full = RESULTS / "extra_eg_probe_cell_full_summary.csv"
    if full.exists():
        return pd.read_csv(full)
    return pd.read_csv(RESULTS / "extra_eg_probe_cell_feasibility_summary.csv")


def load_metadata() -> dict:
    import json

    meta_path = RESULTS / "run_metadata_full.json"
    if not meta_path.exists():
        meta_path = RESULTS / "run_metadata_feasibility.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    summ = load_summary()
    meta = load_metadata()
    if summ.empty:
        raise FileNotFoundError("No summary CSV; run run_xgboost_eg_probe_extra_cell.py first")

    ds = summ["dataset"].iloc[0]
    ds_label = "Adult" if ds == "adult" else "ACSIncome"
    prot_tex = summ["protected_attribute"].iloc[0].replace("_", r"\_")

    caption = (
        "Extra XGBoost EG probe cell (not part of the 3{,}690-run main grid; "
        "not included in Figure~\\ref{fig:si-pareto} or the manuscript pooled main-grid mitigation table). "
        f"Setting: {ds_label} \\texttt{{{prot_tex}}}; 30 seeds (42--71)."
    )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        r"\label{tab:si-extra-eg-probe-cell}",
        r"\resizebox{0.85\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Constraint} & \textbf{Acc.} & $|\mathrm{SPD}|$ & $|\mathrm{EOD}|$ & \textbf{CFS} & \textbf{Seeds} \\",
        r"\midrule",
    ]
    for _, row in summ.iterrows():
        c = CONSTRAINT_LABELS.get(row["constraint"], row["constraint"])
        lines.append(
            f"{c} & {row['mean_accuracy']:.3f} & {row['mean_abs_spd']:.3f} & "
            f"{row['mean_abs_eod']:.3f} & {row['mean_cfs']:.3f} & {int(row['n_seeds'])} \\\\"
        )
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
