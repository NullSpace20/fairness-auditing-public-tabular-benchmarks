#!/usr/bin/env python3
"""SAM-Fair audit-support selection over stored aggregate outputs.

SAM-Fair ranks model--mitigation configurations; it is not a mitigation method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_AGG = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
    / "data_tables"
    / "full_aggregate_30seed.csv"
)
DEFAULT_ACFS = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
    / "data_tables"
    / "acfs_rankings.csv"
)
DEFAULT_RT = (
    REPO
    / "Paper_Springer_JBigData_Q1Upgrade"
    / "Supplementary_Materials_and_Reproducibility_Package"
    / "data_tables"
    / "runtime_summary.csv"
)

PRESETS = {
    "balanced": "mean_acfs_balanced",
    "statistical_parity": "mean_acfs_statistical_parity",
    "equal_opportunity": "mean_acfs_equal_opportunity",
    "di_sensitive": "mean_acfs_regulatory_risk",
}

MIT_SHORT = {
    "baseline": "Baseline",
    "reweighing": "Reweighing",
    "equalized_odds": "EO",
    "ExponentiatedGradient_DP": "EG-DP",
    "ExponentiatedGradient_EO": "EG-EO",
}

MODEL_SHORT = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "gradient_boosting": "GB",
    "xgboost": "XGB",
    "mlp": "MLP",
}

RUNTIME_DATASET = {
    "acs_income_ca_2018": "ACSIncome (CA 2018)",
    "adult": "Adult",
    "bank_uci": "UCI Bank",
}

RUNTIME_MODEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


def label_row(model: str, mitigation: str) -> str:
    return f"{MODEL_SHORT.get(model, model)}+{MIT_SHORT.get(mitigation, mitigation)}"


def pareto_filter(df: pd.DataFrame, acc_col: str, acfs_col: str) -> pd.DataFrame:
    idx = []
    for i, r in df.iterrows():
        dominated = False
        for j, s in df.iterrows():
            if i == j:
                continue
            if (
                s[acc_col] >= r[acc_col]
                and s[acfs_col] <= r[acfs_col]
                and (s[acc_col] > r[acc_col] or s[acfs_col] < r[acfs_col])
            ):
                dominated = True
                break
        if not dominated:
            idx.append(i)
    return df.loc[idx]


def select_sam(
    acfs_df: pd.DataFrame,
    agg: pd.DataFrame,
    runtime: pd.DataFrame,
    dataset: str,
    protected: str,
    acfs_col: str,
    lam: float,
    rho: float,
    delta: float = 0.03,
) -> dict:
    g = acfs_df[
        (acfs_df["dataset"] == dataset) & (acfs_df["protected_attribute"] == protected)
    ].copy()
    std_map = (
        agg[(agg["dataset"] == dataset) & (agg["protected_attribute"] == protected)][
            ["model", "mitigation", "std_acfs_balanced"]
        ]
        .set_index(["model", "mitigation"])["std_acfs_balanced"]
    )
    baselines = (
        g[g["mitigation"] == "baseline"][["model", "mean_accuracy"]]
        .set_index("model")["mean_accuracy"]
    )
    g = g[g["mitigation"] != "baseline"].copy()
    g["base_acc"] = g["model"].map(baselines)
    g["acc_loss"] = g["base_acc"] - g["mean_accuracy"]
    g = g[g["acc_loss"] <= delta + 1e-9]
    if g.empty:
        return {"selection": "NONE", "candidates": 0}
    g["score_std"] = g.apply(
        lambda r: std_map.get((r["model"], r["mitigation"]), 0.0), axis=1
    )
    rt = runtime.copy()
    rt["ds_key"] = rt["dataset"].map({v: k for k, v in RUNTIME_DATASET.items()})
    rt["model_key"] = rt["model"].map({v: k for k, v in RUNTIME_MODEL.items()})
    rt["mit_key"] = rt["mitigation"].map(
        {
            "Baseline": "baseline",
            "Reweighing": "reweighing",
            "Equalized Odds": "equalized_odds",
            "EG (Demographic Parity)": "ExponentiatedGradient_DP",
            "EG (Equalized Odds)": "ExponentiatedGradient_EO",
        }
    )
    rt_map = rt.set_index(["ds_key", "model_key", "mit_key"])["mean_runtime_s"].to_dict()
    g["runtime"] = g.apply(
        lambda r: rt_map.get((dataset, r["model"], r["mitigation"]), 0.0), axis=1
    )
    max_rt = max(g["runtime"].max(), 1e-9)
    g["runtime_norm"] = g["runtime"] / max_rt
    g["sam_score"] = g[acfs_col] + lam * g["score_std"] + rho * g["runtime_norm"]
    nd = pareto_filter(g, "mean_accuracy", acfs_col)
    pick = nd.sort_values(["sam_score", "mean_accuracy"], ascending=[True, False]).iloc[0]
    return {
        "selection": label_row(pick["model"], pick["mitigation"]),
        "model": pick["model"],
        "mitigation": pick["mitigation"],
        "mean_accuracy": float(pick["mean_accuracy"]),
        "mean_acfs": float(pick[acfs_col]),
        "acc_loss": float(pick["acc_loss"]),
        "sam_score": float(pick["sam_score"]),
        "runtime_s": float(pick["runtime"]),
        "candidates": int(len(g)),
        "pareto_candidates": int(len(nd)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM-Fair selection over stored benchmark outputs (audit-support; not mitigation)."
    )
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGG)
    parser.add_argument("--acfs", type=Path, default=DEFAULT_ACFS)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RT)
    parser.add_argument("--dataset", required=True, help="e.g. adult, bank_uci, acs_income_ca_2018")
    parser.add_argument("--protected", required=True, help="e.g. sex, age_group")
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=sorted(PRESETS.keys()),
        help="ACFS weight preset",
    )
    parser.add_argument("--delta", type=float, default=0.03, help="Accuracy-loss budget")
    parser.add_argument("--lambda-uncertainty", type=float, default=0.0, dest="lam")
    parser.add_argument("--rho-runtime", type=float, default=0.0, dest="rho")
    parser.add_argument("--output", type=Path, help="Write JSON result to path")
    parser.add_argument("--csv", action="store_true", help="Print one-line CSV to stdout")
    args = parser.parse_args()

    agg = pd.read_csv(args.aggregate)
    acfs_df = pd.read_csv(args.acfs)
    runtime = pd.read_csv(args.runtime)
    acfs_col = PRESETS[args.preset]

    result = select_sam(
        acfs_df,
        agg,
        runtime,
        args.dataset,
        args.protected,
        acfs_col,
        args.lam,
        args.rho,
        args.delta,
    )
    out = {
        "dataset": args.dataset,
        "protected_attribute": args.protected,
        "preset": args.preset,
        "acfs_column": acfs_col,
        "delta": args.delta,
        "lambda_uncertainty": args.lam,
        "rho_runtime": args.rho,
        **result,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    if args.csv:
        flat = {k: v for k, v in out.items()}
        print(",".join(flat.keys()))
        print(",".join(str(flat[k]) for k in flat))
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
