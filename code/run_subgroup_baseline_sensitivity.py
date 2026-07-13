"""Descriptive subgroup / intersectional baseline sensitivity (R5).

Baseline Logistic Regression and XGBoost only; seeds 42-71.
Does not modify the 3,690-run main-grid CSVs or run mitigations.

Analyses:
  - adult_sex_race      (intersectional; reuses existing CSV if present)
  - adult_sex_age       (optional descriptive check)
  - acs_sex_age         (subgroup check)
  - bank_age_job        (subgroup check)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from loaders import load_acs_income_ca_2018, load_adult_readonly, load_uci_bank_additional
from pipeline_core import (
    TEST_SIZE,
    ProtectedSpec,
    build_preprocessor,
    fairness_metrics,
    fit_baseline,
    get_acs_protected_specs,
    get_adult_protected_specs,
    get_bank_protected_specs,
    get_non_feature_columns,
    performance_metrics,
)

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "subgroup_baseline_sensitivity"
LEGACY_ADULT = Q1_ROOT / "results" / "intersectional_adult_baseline"
SEEDS = list(range(42, 72))
MODELS = ["logistic_regression", "xgboost"]
MIN_CELL_N = 30


@dataclass(frozen=True)
class SubgroupAnalysis:
    analysis_id: str
    dataset_key: str
    cross_label: str
    attr_a: str
    attr_b: str
    marginal_spec_names: Tuple[str, str]
    intersectional: bool
    get_specs: Callable[[], List[ProtectedSpec]]
    load_df: Callable[[], Tuple[pd.DataFrame, Dict]]
    target_col: str


def cell_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attr_a: np.ndarray,
    attr_b: np.ndarray,
) -> dict:
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true).astype(int).ravel(),
            "pred": np.asarray(y_pred).astype(int).ravel(),
            "a": np.asarray(attr_a).astype(int).ravel(),
            "b": np.asarray(attr_b).astype(int).ravel(),
        }
    )
    df["cell"] = df["a"].astype(str) + df["b"].astype(str)
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
            "subgroup_pr_range": float("nan"),
            "subgroup_tpr_range": float("nan"),
            "subgroup_fpr_range": float("nan"),
            "n_cells": len(rows),
            "min_cell_n": min((r["n"] for r in rows), default=0),
        }
    prs = [r["pr"] for r in rows]
    tprs = [r["tpr"] for r in rows if not np.isnan(r["tpr"])]
    fprs = [r["fpr"] for r in rows if not np.isnan(r["fpr"])]
    return {
        "subgroup_pr_range": max(prs) - min(prs),
        "subgroup_tpr_range": (max(tprs) - min(tprs)) if len(tprs) >= 2 else float("nan"),
        "subgroup_fpr_range": (max(fprs) - min(fprs)) if len(fprs) >= 2 else float("nan"),
        "n_cells": len(rows),
        "min_cell_n": min(r["n"] for r in rows),
    }


def get_analyses() -> List[SubgroupAnalysis]:
    return [
        SubgroupAnalysis(
            "adult_sex_race",
            "adult",
            "sex x race",
            "sex_binary",
            "race_binary",
            ("sex", "race"),
            True,
            get_adult_protected_specs,
            load_adult_readonly,
            "income",
        ),
        SubgroupAnalysis(
            "adult_sex_age",
            "adult",
            "sex x age_group",
            "sex_binary",
            "age_group",
            ("sex", "age_group"),
            True,
            get_adult_protected_specs,
            load_adult_readonly,
            "income",
        ),
        SubgroupAnalysis(
            "acs_sex_age",
            "acs_income",
            "sex x age_group",
            "sex_binary",
            "age_group",
            ("sex", "age_group"),
            False,
            get_acs_protected_specs,
            load_acs_income_ca_2018,
            "label",
        ),
        SubgroupAnalysis(
            "bank_age_job",
            "bank_uci",
            "age_group x job_group",
            "age_group",
            "job_group",
            ("age_group", "job_group"),
            False,
            get_bank_protected_specs,
            load_uci_bank_additional,
            "label",
        ),
    ]


def run_seed(
    analysis: SubgroupAnalysis,
    model: str,
    seed: int,
    df: pd.DataFrame,
    specs: List[ProtectedSpec],
) -> dict:
    spec_map = {s.name: s for s in specs}
    spec_a = spec_map[analysis.marginal_spec_names[0]]
    spec_b = spec_map[analysis.marginal_spec_names[1]]
    prot_cols = get_non_feature_columns(analysis.dataset_key)
    target = analysis.target_col
    feature_cols = [c for c in df.columns if c not in [target] + prot_cols]
    X = df[feature_cols]
    y = df[target].values
    a_all = df[analysis.attr_a].values.astype(int)
    b_all = df[analysis.attr_b].values.astype(int)

    X_train, X_test, y_train, y_test, a_train, a_test, b_train, b_test = train_test_split(
        X,
        y,
        a_all,
        b_all,
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

    fa = fairness_metrics(y_test, y_pred, a_test, spec_a)
    fb = fairness_metrics(y_test, y_pred, b_test, spec_b)
    cells = cell_metrics(y_test, y_pred, a_test, b_test)

    return {
        "analysis_id": analysis.analysis_id,
        "dataset": analysis.dataset_key,
        "cross_label": analysis.cross_label,
        "intersectional": analysis.intersectional,
        "model": model,
        "mitigation": "baseline",
        "seed": seed,
        "accuracy": performance_metrics(y_test, y_pred)["accuracy"],
        "attr_a_abs_spd": abs(fa["spd"]),
        "attr_a_abs_eod": abs(fa["eod"]),
        "attr_b_abs_spd": abs(fb["spd"]),
        "attr_b_abs_eod": abs(fb["eod"]),
        "max_marginal_abs_spd": max(abs(fa["spd"]), abs(fb["spd"])),
        "max_marginal_abs_eod": max(abs(fa["eod"]), abs(fb["eod"])),
        "subgroup_pr_range": cells["subgroup_pr_range"],
        "subgroup_tpr_range": cells["subgroup_tpr_range"],
        "subgroup_fpr_range": cells["subgroup_fpr_range"],
        "n_subgroup_cells": cells["n_cells"],
        "min_cell_n": cells["min_cell_n"],
        "runtime_seconds": runtime,
        "status": "success",
    }


def load_legacy_adult_sex_race() -> Optional[pd.DataFrame]:
    path = LEGACY_ADULT / "intersectional_per_seed.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ok = df[df["status"] == "success"].copy()
    if ok.empty:
        return None
    ok["analysis_id"] = "adult_sex_race"
    ok["dataset"] = "adult"
    ok["cross_label"] = "sex x race"
    ok["intersectional"] = True
    ok["attr_a_abs_spd"] = ok["sex_abs_spd"]
    ok["attr_a_abs_eod"] = ok["sex_abs_eod"]
    ok["attr_b_abs_spd"] = ok["race_abs_spd"]
    ok["attr_b_abs_eod"] = ok["race_abs_eod"]
    ok["subgroup_pr_range"] = ok["intersectional_pr_range"]
    ok["subgroup_tpr_range"] = ok["intersectional_tpr_range"]
    ok["subgroup_fpr_range"] = ok["intersectional_fpr_range"]
    ok["n_subgroup_cells"] = ok["n_intersectional_cells"]
    return ok


def run_analysis(analysis: SubgroupAnalysis, reuse_legacy: bool = True) -> pd.DataFrame:
    if analysis.analysis_id == "adult_sex_race" and reuse_legacy:
        legacy = load_legacy_adult_sex_race()
        if legacy is not None:
            print(f"Reusing legacy results for {analysis.analysis_id}")
            return legacy

    df, meta = analysis.load_df()
    if df is None:
        raise RuntimeError(f"Failed to load dataset for {analysis.analysis_id}: {meta}")
    specs = analysis.get_specs()
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            try:
                rows.append(run_seed(analysis, model, seed, df, specs))
            except Exception as exc:
                rows.append(
                    {
                        "analysis_id": analysis.analysis_id,
                        "dataset": analysis.dataset_key,
                        "cross_label": analysis.cross_label,
                        "model": model,
                        "mitigation": "baseline",
                        "seed": seed,
                        "status": f"error: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    ok = per_seed[per_seed["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby(["analysis_id", "dataset", "cross_label", "intersectional", "model"], dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_max_marginal_abs_spd=("max_marginal_abs_spd", "mean"),
            mean_max_marginal_abs_eod=("max_marginal_abs_eod", "mean"),
            mean_subgroup_pr_range=("subgroup_pr_range", "mean"),
            mean_subgroup_tpr_range=("subgroup_tpr_range", "mean"),
            mean_min_cell_n=("min_cell_n", "mean"),
            min_cell_n_overall=("min_cell_n", "min"),
        )
        .reset_index()
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for analysis in get_analyses():
        print(f"Running {analysis.analysis_id} ...")
        per = run_analysis(analysis)
        per.to_csv(OUT_DIR / f"{analysis.analysis_id}_per_seed.csv", index=False)
        summ = summarize(per)
        if not summ.empty:
            summaries.append(summ)
        all_rows.append(per)

    per_seed = pd.concat(all_rows, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    per_seed.to_csv(OUT_DIR / "subgroup_per_seed_all.csv", index=False)
    summary.to_csv(OUT_DIR / "subgroup_summary_all.csv", index=False)

    meta = {
        "analysis": "subgroup_baseline_sensitivity_R5",
        "models": MODELS,
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "min_cell_n_threshold": MIN_CELL_N,
        "analyses": [a.analysis_id for a in get_analyses()],
        "n_success": int((per_seed["status"] == "success").sum()),
        "n_total": int(len(per_seed)),
        "legacy_reuse": "adult_sex_race from intersectional_adult_baseline if present",
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", OUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
