"""Build XGBoost-EG probe comparison table (LaTeX) from probe + main-grid aggregates."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AGG = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
    / "data_tables"
    / "full_aggregate_30seed.csv"
)
PROBE = REPO / "q1_upgrade" / "results" / "xgboost_eg_probe"
OUT = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Q1_Tables"
    / "tab_xgboost_eg_probe.tex"
)

SETTINGS = [
    ("acs_income_ca_2018", "age_group", "ACSIncome"),
    ("adult", "sex", "Adult"),
    ("bank_uci", "age_group", "UCI Bank"),
]

MAIN_MITS = [
    ("baseline", "Baseline"),
    ("reweighing", "Reweighing"),
    ("equalized_odds", "Equalized Odds"),
]

EG_MAP = {
    "DP": ("ExponentiatedGradient_DP", "EG-DP (probe)"),
    "EO": ("ExponentiatedGradient_EO", "EG-EO (probe)"),
}


def load_probe_summary() -> pd.DataFrame:
    full = PROBE / "xgboost_eg_probe_full_summary.csv"
    if full.exists():
        return pd.read_csv(full)
    return pd.read_csv(PROBE / "xgboost_eg_probe_feasibility_summary.csv")


def main() -> None:
    agg = pd.read_csv(AGG)
    probe = load_probe_summary()
    rows = []
    for ds, prot, label in SETTINGS:
        for mit_key, mit_label in MAIN_MITS:
            r = agg[
                (agg["dataset"] == ds)
                & (agg["protected_attribute"] == prot)
                & (agg["model"] == "xgboost")
                & (agg["mitigation"] == mit_key)
            ]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(
                {
                    "dataset": label,
                    "protected": prot.replace("_", r"\_"),
                    "condition": mit_label,
                    "acc": r["mean_accuracy"],
                    "spd": abs(r["mean_spd"]),
                    "eod": abs(r["mean_eod"]),
                    "cfs": r["mean_cfs"],
                    "rt": "main grid",
                }
            )
        for c_code, (mit_name, mit_label) in EG_MAP.items():
            pr = probe[
                (probe["dataset"] == ds)
                & (probe["protected_attribute"] == prot)
                & (probe["constraint"] == c_code)
            ]
            if pr.empty:
                continue
            pr = pr.iloc[0]
            n = int(pr["n"])
            cfs_val = pr["mean_cfs"]
            if isinstance(cfs_val, str):
                cfs_val = float(cfs_val.strip("[]"))
            tag = f"probe ({n} seeds)"
            rows.append(
                {
                    "dataset": label,
                    "protected": prot.replace("_", r"\_"),
                    "condition": mit_label,
                    "acc": pr["mean_accuracy"],
                    "spd": pr["mean_abs_spd"],
                    "eod": pr["mean_abs_eod"],
                    "cfs": cfs_val,
                    "rt": tag,
                }
            )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{XGBoost boosted-tree EG sensitivity probe on three selected settings (not part of the 3{,}690-run main grid). Main-grid XGBoost rows use 30 seeds; EG-DP/EG-EO rows use the separate probe run documented in Additional file~1.}",
        r"\label{tab:xgboost-eg-probe}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrl}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Protected} & \textbf{Condition} & \textbf{Acc.} & $|\mathrm{SPD}|$ & $|\mathrm{EOD}|$ & \textbf{CFS} & \textbf{Runs} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['protected']} & {row['condition']} & "
            f"{row['acc']:.3f} & {row['spd']:.3f} & {row['eod']:.3f} & "
            f"{row['cfs']:.3f} & {row['rt']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT, "rows", len(rows))


if __name__ == "__main__":
    main()
