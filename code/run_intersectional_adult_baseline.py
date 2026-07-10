"""Limited intersectional baseline sensitivity: Adult sex x race (probe analysis).

Reruns baseline Logistic Regression and XGBoost on Adult with seeds 42-71.
Does not modify the original 3,690-run main-grid CSVs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fairness_utils import compute_cfs
from loaders import load_adult_readonly
from sklearn.model_selection import train_test_split

from pipeline_core import (
    TEST_SIZE,
    build_preprocessor,
    fairness_metrics,
    fit_baseline,
    get_adult_protected_specs,
    get_non_feature_columns,
    performance_metrics,
)

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "intersectional_adult_baseline"
SEEDS = list(range(42, 72))
MODELS = ["logistic_regression", "xgboost"]
MIN_CELL_N = 30


def cell_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sex: np.ndarray,
    race: np.ndarray,
) -> dict:
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true).astype(int).ravel(),
            "pred": np.asarray(y_pred).astype(int).ravel(),
            "sex": np.asarray(sex).astype(int).ravel(),
            "race": np.asarray(race).astype(int).ravel(),
        }
    )
    df["cell"] = df["sex"].astype(str) + df["race"].astype(str)
    rows = []
    for cell, g in df.groupby("cell"):
        n = len(g)
        if n < MIN_CELL_N:
            continue
        pr = float(g["pred"].mean())
        pos = g[g["y"] == 1]
        neg = g[g["y"] == 0]
        tpr = float(pos["pred"].mean()) if len(pos) else float("nan")
        fpr = float(neg["pred"].mean()) if len(neg) else float("nan")
        rows.append({"cell": cell, "n": n, "pr": pr, "tpr": tpr, "fpr": fpr})
    if len(rows) < 2:
        return {
            "pr_range": float("nan"),
            "tpr_range": float("nan"),
            "fpr_range": float("nan"),
            "n_cells": len(rows),
            "min_cell_n": min((r["n"] for r in rows), default=0),
        }
    prs = [r["pr"] for r in rows]
    tprs = [r["tpr"] for r in rows if not np.isnan(r["tpr"])]
    fprs = [r["fpr"] for r in rows if not np.isnan(r["fpr"])]
    return {
        "pr_range": max(prs) - min(prs),
        "tpr_range": (max(tprs) - min(tprs)) if len(tprs) >= 2 else float("nan"),
        "fpr_range": (max(fprs) - min(fprs)) if len(fprs) >= 2 else float("nan"),
        "n_cells": len(rows),
        "min_cell_n": min(r["n"] for r in rows),
    }


def run_seed(model: str, seed: int, df: pd.DataFrame, specs) -> dict:
    sex_spec = next(s for s in specs if s.name == "sex")
    race_spec = next(s for s in specs if s.name == "race")
    prot_cols = get_non_feature_columns("adult")
    target = "income"
    feature_cols = [c for c in df.columns if c not in [target] + prot_cols]
    X = df[feature_cols]
    y = df[target].values
    sex_all = df["sex_binary"].values.astype(int)
    race_all = df["race_binary"].values.astype(int)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        sex_train,
        sex_test,
        race_train,
        race_test,
    ) = train_test_split(
        X,
        y,
        sex_all,
        race_all,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )
    preprocessor = build_preprocessor(pd.concat([X_train, X_test]), feature_cols)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    t0 = time.perf_counter()
    y_pred = fit_baseline(model, X_train_p, y_train, X_test_p, seed)
    runtime = time.perf_counter() - t0

    sex_f = fairness_metrics(y_test, y_pred, sex_test, sex_spec)
    race_f = fairness_metrics(y_test, y_pred, race_test, race_spec)
    cells = cell_metrics(y_test, y_pred, sex_test, race_test)
    perf = performance_metrics(y_test, y_pred)
    cfs = compute_cfs(sex_f["spd"], sex_f["di"], sex_f["eod"], sex_f["aod"])

    return {
        "dataset": "adult",
        "model": model,
        "mitigation": "baseline",
        "seed": seed,
        "accuracy": perf["accuracy"],
        "cfs": cfs,
        "sex_abs_spd": abs(sex_f["spd"]),
        "sex_abs_eod": abs(sex_f["eod"]),
        "race_abs_spd": abs(race_f["spd"]),
        "race_abs_eod": abs(race_f["eod"]),
        "max_marginal_abs_spd": max(abs(sex_f["spd"]), abs(race_f["spd"])),
        "max_marginal_abs_eod": max(abs(sex_f["eod"]), abs(race_f["eod"])),
        "intersectional_pr_range": cells["pr_range"],
        "intersectional_tpr_range": cells["tpr_range"],
        "intersectional_fpr_range": cells["fpr_range"],
        "n_intersectional_cells": cells["n_cells"],
        "min_cell_n": cells["min_cell_n"],
        "runtime_seconds": runtime,
        "status": "success",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = load_adult_readonly()
    specs = get_adult_protected_specs()
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            try:
                rows.append(run_seed(model, seed, df, specs))
            except Exception as exc:
                rows.append(
                    {
                        "dataset": "adult",
                        "model": model,
                        "mitigation": "baseline",
                        "seed": seed,
                        "status": f"error: {exc}",
                    }
                )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_DIR / "intersectional_per_seed.csv", index=False)

    ok = per_seed[per_seed["status"] == "success"]
    summary = (
        ok.groupby("model")
        .agg(
            n_seeds=("seed", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_max_marginal_abs_spd=("max_marginal_abs_spd", "mean"),
            mean_max_marginal_abs_eod=("max_marginal_abs_eod", "mean"),
            mean_intersectional_pr_range=("intersectional_pr_range", "mean"),
            mean_intersectional_tpr_range=("intersectional_tpr_range", "mean"),
            mean_intersectional_fpr_range=("intersectional_fpr_range", "mean"),
            mean_min_cell_n=("min_cell_n", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "intersectional_summary.csv", index=False)

    meta_out = {
        "analysis": "intersectional_adult_baseline",
        "cells": "sex_binary x race_binary (4 groups)",
        "models": MODELS,
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "min_cell_n_threshold": MIN_CELL_N,
        "n_success": int((per_seed["status"] == "success").sum()),
        "loader": meta.get("loader_summary"),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print("Wrote", OUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
