"""Optional Adult sex×race mitigated subgroup sensitivity (W3).

Baseline and Equalized Odds on marginal sex or race; evaluate sex×race cells.
Does NOT modify the 3,690-run main grid or claim full subgroup mitigation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fairness_utils import compute_cfs
from loaders import load_adult_readonly
from mitigations_aif360 import apply_equalized_odds
from pipeline_core import (
    TEST_SIZE,
    VAL_FRACTION,
    ProtectedSpec,
    build_preprocessor,
    fairness_metrics,
    fit_baseline,
    get_adult_protected_specs,
    get_non_feature_columns,
    performance_metrics,
)

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "subgroup_mitigation_sensitivity"
SEEDS_FULL = list(range(42, 72))
SEEDS_FEASIBILITY = [42, 43]
MODELS = ["logistic_regression", "xgboost"]
MIN_CELL_N = 30

# EO is applied on a single binary protected attribute; cells are sex×race for evaluation.
CONDITIONS = [
    {
        "condition_id": "baseline",
        "mitigation": "baseline",
        "eo_protected_attribute": None,
        "label": "baseline",
    },
    {
        "condition_id": "equalized_odds_sex",
        "mitigation": "equalized_odds",
        "eo_protected_attribute": "sex",
        "label": "Equalized Odds (fitted on sex)",
    },
    {
        "condition_id": "equalized_odds_race",
        "mitigation": "equalized_odds",
        "eo_protected_attribute": "race",
        "label": "Equalized Odds (fitted on race)",
    },
]


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
            "subgroup_pr_gap": float("nan"),
            "subgroup_tpr_gap": float("nan"),
            "subgroup_fpr_gap": float("nan"),
            "n_cells": len(rows),
            "min_cell_n": min((r["n"] for r in rows), default=0),
        }
    prs = [r["pr"] for r in rows]
    tprs = [r["tpr"] for r in rows if not np.isnan(r["tpr"])]
    fprs = [r["fpr"] for r in rows if not np.isnan(r["fpr"])]
    return {
        "subgroup_pr_gap": max(prs) - min(prs),
        "subgroup_tpr_gap": (max(tprs) - min(tprs)) if len(tprs) >= 2 else float("nan"),
        "subgroup_fpr_gap": (max(fprs) - min(fprs)) if len(fprs) >= 2 else float("nan"),
        "n_cells": len(rows),
        "min_cell_n": min(r["n"] for r in rows),
    }


def prepare_intersectional_split(df: pd.DataFrame, seed: int, with_val: bool = False) -> dict:
    prot_cols = get_non_feature_columns("adult")
    target = "income"
    feature_cols = [c for c in df.columns if c not in [target] + prot_cols]
    X = df[feature_cols]
    y = df[target].values
    sex = df["sex_binary"].values.astype(int)
    race = df["race_binary"].values.astype(int)

    X_train_full, X_test, y_train_full, y_test, sex_train_full, sex_test, race_train_full, race_test = (
        train_test_split(
            X, y, sex, race, test_size=TEST_SIZE, stratify=y, random_state=seed
        )
    )
    preprocessor = build_preprocessor(pd.concat([X_train_full, X_test]), feature_cols)
    X_train_full_p = preprocessor.fit_transform(X_train_full)
    X_test_p = preprocessor.transform(X_test)

    out = {
        "X_test": X_test_p,
        "y_test": y_test,
        "sex_test": sex_test,
        "race_test": race_test,
        "n_rows": len(df),
    }
    if not with_val:
        out["X_train"] = X_train_full_p
        out["y_train"] = y_train_full
        out["sex_train"] = sex_train_full
        out["race_train"] = race_train_full
        return out

    (
        X_train,
        X_val,
        y_train,
        y_val,
        sex_train,
        sex_val,
        race_train,
        race_val,
    ) = train_test_split(
        X_train_full_p,
        y_train_full,
        sex_train_full,
        race_train_full,
        test_size=VAL_FRACTION,
        stratify=y_train_full,
        random_state=seed,
    )
    out.update(
        {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "sex_train": sex_train,
            "sex_val": sex_val,
            "race_train": race_train,
            "race_val": race_val,
        }
    )
    return out


def get_spec(name: str) -> ProtectedSpec:
    return next(s for s in get_adult_protected_specs() if s.name == name)


def run_one(model: str, condition: dict, seed: int, df: pd.DataFrame) -> dict:
    sex_spec = get_spec("sex")
    race_spec = get_spec("race")
    base = {
        "analysis": "optional_subgroup_mitigation_sensitivity",
        "dataset": "adult",
        "evaluation_cells": "sex_binary x race_binary",
        "model": model,
        "condition_id": condition["condition_id"],
        "mitigation": condition["mitigation"],
        "eo_protected_attribute": condition["eo_protected_attribute"] or "",
        "eo_protected_column": "",
        "seed": seed,
    }

    t0 = time.perf_counter()
    try:
        if condition["mitigation"] == "baseline":
            split = prepare_intersectional_split(df, seed, with_val=False)
            y_pred = fit_baseline(
                model, split["X_train"], split["y_train"], split["X_test"], seed
            )
            y_test = split["y_test"]
            sex_test = split["sex_test"]
            race_test = split["race_test"]
        else:
            eo_attr = condition["eo_protected_attribute"]
            spec = get_spec(eo_attr)
            base["eo_protected_column"] = spec.column
            split = prepare_intersectional_split(df, seed, with_val=True)
            if eo_attr == "sex":
                prot_train, prot_val, prot_test = (
                    split["sex_train"],
                    split["sex_val"],
                    split["sex_test"],
                )
            else:
                prot_train, prot_val, prot_test = (
                    split["race_train"],
                    split["race_val"],
                    split["race_test"],
                )
            y_pred = apply_equalized_odds(
                model,
                split["X_train"],
                split["y_train"],
                prot_train,
                split["X_val"],
                split["y_val"],
                prot_val,
                split["X_test"],
                prot_test,
                spec,
                seed,
            )
            y_test = split["y_test"]
            sex_test = split["sex_test"]
            race_test = split["race_test"]

        runtime = time.perf_counter() - t0
        sex_f = fairness_metrics(y_test, y_pred, sex_test, sex_spec)
        race_f = fairness_metrics(y_test, y_pred, race_test, race_spec)
        cells = cell_metrics(y_test, y_pred, sex_test, race_test)
        perf = performance_metrics(y_test, y_pred)

        row = {
            **base,
            "accuracy": perf["accuracy"],
            "f1": perf["f1"],
            "sex_abs_spd": abs(sex_f["spd"]),
            "sex_abs_eod": abs(sex_f["eod"]),
            "race_abs_spd": abs(race_f["spd"]),
            "race_abs_eod": abs(race_f["eod"]),
            "marginal_abs_spd": max(abs(sex_f["spd"]), abs(race_f["spd"])),
            "marginal_abs_eod": max(abs(sex_f["eod"]), abs(race_f["eod"])),
            "subgroup_pr_gap": cells["subgroup_pr_gap"],
            "subgroup_tpr_gap": cells["subgroup_tpr_gap"],
            "subgroup_fpr_gap": cells["subgroup_fpr_gap"],
            "n_cells": cells["n_cells"],
            "min_cell_n": cells["min_cell_n"],
            "cfs_sex": float(
                compute_cfs(sex_f["spd"], sex_f["di"], sex_f["eod"], sex_f["aod"])
            ),
            "runtime_seconds": runtime,
            "status": "success",
            "error": "",
        }
        return row
    except Exception as exc:
        return {
            **base,
            "status": f"error: {exc}",
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - t0,
        }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    ok = per_seed[per_seed["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby(["model", "condition_id", "eo_protected_attribute"], dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_f1=("f1", "mean"),
            mean_marginal_abs_spd=("marginal_abs_spd", "mean"),
            mean_marginal_abs_eod=("marginal_abs_eod", "mean"),
            mean_subgroup_pr_gap=("subgroup_pr_gap", "mean"),
            mean_subgroup_tpr_gap=("subgroup_tpr_gap", "mean"),
            mean_min_cell_n=("min_cell_n", "mean"),
            min_cell_n_overall=("min_cell_n", "min"),
        )
        .reset_index()
    )


def gap_assessment(summary: pd.DataFrame) -> dict:
    """Compare EO conditions to baseline on subgroup gaps."""
    notes: List[str] = []
    narrowed_any = False
    for model in MODELS:
        base = summary[
            (summary["model"] == model) & (summary["condition_id"] == "baseline")
        ]
        if base.empty:
            continue
        b_pr = float(base["mean_subgroup_pr_gap"].iloc[0])
        b_tpr = float(base["mean_subgroup_tpr_gap"].iloc[0])
        for cid in ("equalized_odds_sex", "equalized_odds_race"):
            sub = summary[(summary["model"] == model) & (summary["condition_id"] == cid)]
            if sub.empty:
                continue
            pr = float(sub["mean_subgroup_pr_gap"].iloc[0])
            tpr = float(sub["mean_subgroup_tpr_gap"].iloc[0])
            eo_attr = sub["eo_protected_attribute"].iloc[0]
            pr_delta = pr - b_pr
            tpr_delta = tpr - b_tpr
            if pr_delta < -0.01 or tpr_delta < -0.01:
                narrowed_any = True
            notes.append(
                f"{model}/{cid} (EO on {eo_attr}): PR gap {pr:.3f} vs baseline {b_pr:.3f} "
                f"({pr_delta:+.3f}); TPR gap {tpr:.3f} vs baseline {b_tpr:.3f} ({tpr_delta:+.3f})"
            )
    return {
        "subgroup_gaps_narrowed_under_eo": narrowed_any,
        "assessment_notes": notes,
        "intersectional_eo_not_applied": True,
        "eo_applied_on_marginal_attributes_only": ["sex", "race"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult sex×race mitigated subgroup sensitivity (W3)")
    parser.add_argument("--feasibility", action="store_true", help="Run seeds 42-43 only")
    args = parser.parse_args()
    seeds = SEEDS_FEASIBILITY if args.feasibility else SEEDS_FULL

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, load_meta = load_adult_readonly()
    if df is None:
        raise RuntimeError(f"Failed to load Adult: {load_meta}")

    rows = []
    total = len(MODELS) * len(CONDITIONS) * len(seeds)
    done = 0
    for model in MODELS:
        for condition in CONDITIONS:
            for seed in seeds:
                done += 1
                if done % 30 == 0 or done == 1:
                    print(f"[{done}/{total}] {model} {condition['condition_id']} seed={seed}")
                rows.append(run_one(model, condition, seed, df))

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    assessment = gap_assessment(summary) if not summary.empty else {}

    suffix = "feasibility" if args.feasibility else "full"
    per_path = OUT_DIR / f"subgroup_mitigation_{suffix}_per_seed.csv"
    summ_path = OUT_DIR / f"subgroup_mitigation_{suffix}_summary.csv"
    per_seed.to_csv(per_path, index=False)
    summary.to_csv(summ_path, index=False)

    meta = {
        "analysis": "optional_subgroup_mitigation_sensitivity_W3",
        "not_part_of_main_grid": True,
        "dataset": "adult",
        "evaluation_cells": "sex_binary x race_binary (4 cells)",
        "models": MODELS,
        "conditions": CONDITIONS,
        "eo_note": (
            "Equalized Odds post-processing uses AIF360 EqOddsPostprocessing on a single "
            "binary protected attribute (sex or race). It is NOT applied to intersectional "
            "four-cell groups. Sex×race cell metrics are evaluated after mitigation."
        ),
        "seeds": f"{seeds[0]}-{seeds[-1]}",
        "n_runs_attempted": total,
        "n_success": int((per_seed["status"] == "success").sum()),
        "gap_assessment": assessment,
        "limitations": [
            "Illustrative sensitivity only; not full subgroup mitigation",
            "EO targets marginal binary attribute, not intersectional cells",
            "Single dataset and cross-attribute setting (Adult sex×race)",
        ],
        "adult_load_meta": load_meta,
    }
    meta_path = OUT_DIR / f"run_metadata_{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {per_path}")
    print(f"Wrote {summ_path}")
    print(f"Wrote {meta_path}")
    if not summary.empty:
        print(summary.to_string(index=False))
    print("Gap assessment:", json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
