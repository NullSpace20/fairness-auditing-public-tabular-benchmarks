"""Phase 4B: final 30-seed Option B (LR+RF main grid, LR-EG full, RF-EG Adult+Bank only)."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from exponentiated_gradient_mitigation import fit_predict_exponentiated_gradient
from fairness_utils import compute_acfs, compute_cfs
from loaders import (
    ACS_PHASE4A_SUBSAMPLE_ROWS,
    ACS_PILOT_SUBSAMPLE_ROWS,
    load_acs_income_ca_2018,
    load_adult_readonly,
    load_uci_bank_additional,
)
from mitigations_aif360 import apply_equalized_odds, apply_reweighing
from pipeline_core import (
    Q1_ROOT,
    ProtectedSpec,
    fairness_metrics,
    fit_baseline,
    get_acs_protected_specs,
    get_adult_protected_specs,
    get_bank_protected_specs,
    get_non_feature_columns,
    performance_metrics,
    prepare_split,
)

PHASE4B_SEEDS = list(range(42, 72))
MAIN_MODELS = ["logistic_regression", "random_forest"]
OPTIONAL_MODELS = ["gradient_boosting", "xgboost", "mlp"]
MAIN_MITIGATIONS = ["baseline", "reweighing", "equalized_odds"]
MAX_WALL_SECONDS_REQUIRED = 18 * 3600
MAX_WALL_SECONDS_TOTAL = 24 * 3600
FAILURE_RATE_STOP = 0.05
RF_EG_SLOW_THRESHOLD = 700.0
RF_EG_SLOW_MAX_BEFORE_SKIP = 3
OPTIONAL_MODEL_FAIL_THRESHOLD = 5
ACS_PROBE_BASELINE_THRESHOLD = 90.0

RESULTS_DIR = Q1_ROOT / "results" / "phase4b_optionB_30seed"
REPORTS_DIR = Q1_ROOT / "reports" / "phase4b"
PHASE5_READY_DIR = RESULTS_DIR / "phase5_ready"
PARTIAL_CSV = RESULTS_DIR / "phase4b_results_partial.csv"
FINAL_CSV = RESULTS_DIR / "phase4b_results.csv"

DATASET_PLAN = {
    "adult": {
        "variant": "kaggle_mirror_train_partition",
        "target": "income",
        "non_feature_key": "adult",
        "protected_main": ["sex", "race", "age_group"],
        "eg_lr": ["sex", "race", "age_group"],
        "eg_rf": ["sex"],
    },
    "bank_uci": {
        "variant": "uci_bank_additional_full",
        "target": "label",
        "non_feature_key": "bank_uci",
        "protected_main": ["age_group", "job_group"],
        "eg_lr": ["age_group", "job_group"],
        "eg_rf": ["age_group"],
    },
    "acs_income_ca_2018": {
        "variant": "acs_ca_2018_stratified",
        "target": "label",
        "non_feature_key": "acs_income",
        "protected_main": ["sex", "age_group"],
        "eg_lr": ["sex", "age_group"],
        "eg_rf": [],
    },
}


def scalar_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.asarray(value).ravel()[0])
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def config_key(
    dataset: str,
    model: str,
    mitigation: str,
    protected: str,
    seed: int,
    run_block: str = "required",
) -> str:
    return f"{run_block}|{dataset}|{model}|{mitigation}|{protected}|{seed}"


def make_result_row(
    *,
    dataset: str,
    dataset_variant: str,
    model: str,
    mitigation: str,
    protected_attribute: str,
    seed: int,
    metrics: Dict[str, float],
    runtime_seconds: float,
    status: str,
    error_message: str = "",
    n_rows: int = 0,
    run_block: str = "required",
) -> Dict[str, Any]:
    return {
        "run_block": run_block,
        "dataset": dataset,
        "dataset_variant": dataset_variant,
        "model": model,
        "mitigation": mitigation,
        "protected_attribute": protected_attribute,
        "seed": int(seed),
        "n_rows": int(n_rows),
        "accuracy": scalar_float(metrics.get("accuracy")),
        "precision": scalar_float(metrics.get("precision")),
        "recall": scalar_float(metrics.get("recall")),
        "f1": scalar_float(metrics.get("f1")),
        "spd": scalar_float(metrics.get("spd")),
        "di": scalar_float(metrics.get("di")),
        "eod": scalar_float(metrics.get("eod")),
        "aod": scalar_float(metrics.get("aod")),
        "cfs": scalar_float(metrics.get("cfs")),
        "acfs_balanced": scalar_float(metrics.get("acfs_balanced")),
        "runtime_seconds": float(runtime_seconds),
        "status": status,
        "error_message": error_message,
    }


def get_specs(dataset: str) -> List[ProtectedSpec]:
    if dataset == "adult":
        return get_adult_protected_specs()
    if dataset == "bank_uci":
        return get_bank_protected_specs()
    return get_acs_protected_specs()


def run_main_mitigation(
    model: str,
    mitigation: str,
    df: pd.DataFrame,
    target_col: str,
    non_feature_cols: List[str],
    spec: ProtectedSpec,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if mitigation == "equalized_odds":
        pack = prepare_split(df, target_col, non_feature_cols, spec, seed, with_val=True)
        X_tr, X_val, X_te, y_tr, y_val, y_te, p_tr, p_val, p_te, n_rows = pack
        y_pred = apply_equalized_odds(
            model, X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, p_te, spec, seed
        )
        return y_te, y_pred, p_te, n_rows

    X_tr, X_te, y_tr, y_te, p_tr, p_te, n_rows = prepare_split(
        df, target_col, non_feature_cols, spec, seed, with_val=False
    )
    if mitigation == "baseline":
        y_pred = fit_baseline(model, X_tr, y_tr, X_te, seed)
    elif mitigation == "reweighing":
        y_pred = apply_reweighing(model, X_tr, y_tr, p_tr, X_te, spec, seed)
    else:
        raise ValueError(mitigation)
    return y_te, y_pred, p_te, n_rows


def execute_config(
    dataset: str,
    dataset_variant: str,
    df: pd.DataFrame,
    target_col: str,
    non_feature_cols: List[str],
    spec: ProtectedSpec,
    model: str,
    mitigation: str,
    seed: int,
    eg_constraint: Optional[str] = None,
    run_block: str = "required",
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    mit_label = mitigation
    if eg_constraint:
        mit_label = f"ExponentiatedGradient_{eg_constraint}"
    try:
        if eg_constraint:
            X_tr, X_te, y_tr, y_te, p_tr, p_te, n_rows = prepare_split(
                df, target_col, non_feature_cols, spec, seed, with_val=False
            )
            y_pred, mit_name = fit_predict_exponentiated_gradient(
                X_tr,
                y_tr,
                p_tr,
                X_te,
                constraint=eg_constraint,
                seed=seed,
                model_name=model,
            )
            mitigation = mit_name
        else:
            y_te, y_pred, p_te, n_rows = run_main_mitigation(
                model, mitigation, df, target_col, non_feature_cols, spec, seed
            )

        perf = performance_metrics(y_te, y_pred)
        fair = fairness_metrics(y_te, y_pred, p_te, spec)
        perf.update(fair)
        perf["cfs"] = compute_cfs(perf["spd"], perf["di"], perf["eod"], perf["aod"])
        perf["acfs_balanced"] = compute_acfs(perf["spd"], perf["di"], perf["eod"], perf["aod"])

        for k in ("accuracy", "spd", "di", "eod", "aod", "cfs"):
            v = perf.get(k)
            if v is not None and (np.isnan(v) or np.isinf(v)):
                raise ValueError(f"Impossible metric {k}={v}")

        return make_result_row(
            dataset=dataset,
            dataset_variant=dataset_variant,
            model=model,
            mitigation=mitigation,
            protected_attribute=spec.name,
            seed=seed,
            metrics=perf,
            runtime_seconds=time.perf_counter() - t0,
            status="success",
            n_rows=n_rows,
            run_block=run_block,
        )
    except Exception as exc:
        return make_result_row(
            dataset=dataset,
            dataset_variant=dataset_variant,
            model=model,
            mitigation=mit_label,
            protected_attribute=spec.name,
            seed=seed,
            metrics={},
            runtime_seconds=time.perf_counter() - t0,
            status="failed",
            error_message=f"{exc}\n{traceback.format_exc()}",
            run_block=run_block,
        )


def build_required_manifest() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset, plan in DATASET_PLAN.items():
        for protected in plan["protected_main"]:
            for model in MAIN_MODELS:
                for mitigation in MAIN_MITIGATIONS:
                    for seed in PHASE4B_SEEDS:
                        rows.append(
                            {
                                "run_block": "required",
                                "dataset": dataset,
                                "model": model,
                                "mitigation": mitigation,
                                "protected_attribute": protected,
                                "seed": seed,
                                "run_type": "main",
                            }
                        )
        for protected in plan["eg_lr"]:
            for constraint in ["DP", "EO"]:
                for seed in PHASE4B_SEEDS:
                    rows.append(
                        {
                            "run_block": "required",
                            "dataset": dataset,
                            "model": "logistic_regression",
                            "mitigation": f"ExponentiatedGradient_{constraint}",
                            "protected_attribute": protected,
                            "seed": seed,
                            "run_type": "eg_lr",
                        }
                    )
        for protected in plan["eg_rf"]:
            for constraint in ["DP", "EO"]:
                for seed in PHASE4B_SEEDS:
                    rows.append(
                        {
                            "run_block": "required",
                            "dataset": dataset,
                            "model": "random_forest",
                            "mitigation": f"ExponentiatedGradient_{constraint}",
                            "protected_attribute": protected,
                            "seed": seed,
                            "run_type": "eg_rf",
                        }
                    )
    return pd.DataFrame(rows)


def build_optional_manifest() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset, plan in DATASET_PLAN.items():
        for protected in plan["protected_main"]:
            for model in OPTIONAL_MODELS:
                for mitigation in MAIN_MITIGATIONS:
                    for seed in PHASE4B_SEEDS:
                        rows.append(
                            {
                                "run_block": "optional_main_grid_extension",
                                "dataset": dataset,
                                "model": model,
                                "mitigation": mitigation,
                                "protected_attribute": protected,
                                "seed": seed,
                                "run_type": "optional_main",
                            }
                        )
    return pd.DataFrame(rows)


def choose_acs_sample_cap() -> Tuple[int, str]:
    from pipeline_core import fit_lr_baseline

    df, meta = load_acs_income_ca_2018(ACS_PHASE4A_SUBSAMPLE_ROWS)
    if df is None:
        return ACS_PILOT_SUBSAMPLE_ROWS, "folktables_load_failed_using_8k_fallback"
    specs = {s.name: s for s in get_acs_protected_specs()}
    spec = specs["sex"]
    non_feature = get_non_feature_columns("acs_income")
    t0 = time.perf_counter()
    try:
        X_tr, X_te, y_tr, y_te, p_tr, p_te, _ = prepare_split(
            df, "label", non_feature, spec, 42, with_val=False
        )
        fit_lr_baseline(X_tr, y_tr, X_te, 42)
        elapsed = time.perf_counter() - t0
        if elapsed > ACS_PROBE_BASELINE_THRESHOLD:
            return ACS_PILOT_SUBSAMPLE_ROWS, (
                f"50k probe baseline took {elapsed:.1f}s > {ACS_PROBE_BASELINE_THRESHOLD}s; "
                "using 8000-row cap"
            )
        return ACS_PHASE4A_SUBSAMPLE_ROWS, f"50k cap accepted (probe {elapsed:.1f}s)"
    except Exception as exc:
        return ACS_PILOT_SUBSAMPLE_ROWS, f"50k probe failed ({exc}); using 8000-row cap"


def load_datasets(acs_cap: int) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    summaries: Dict[str, Any] = {}
    frames: Dict[str, pd.DataFrame] = {}

    adult_df, adult_meta = load_adult_readonly()
    summaries["adult"] = adult_meta
    frames["adult"] = adult_df

    bank_df, bank_meta = load_uci_bank_additional()
    summaries["bank_uci"] = bank_meta
    if bank_df is not None:
        frames["bank_uci"] = bank_df

    acs_df, acs_meta = load_acs_income_ca_2018(acs_cap)
    summaries["acs_income_ca_2018"] = acs_meta
    if acs_df is not None:
        frames["acs_income_ca_2018"] = acs_df

    return summaries, frames


def run_manifest(
    manifest: pd.DataFrame,
    frames: Dict[str, pd.DataFrame],
    specs_cache: Dict[str, Dict[str, ProtectedSpec]],
    rows: List[Dict[str, Any]],
    done: Set[str],
    wall_start: float,
    wall_limit: float,
    rf_eg_slow_counts: Dict[Tuple[str, str], int],
    skip_rf_eg_keys: Set[Tuple[str, str]],
    skipped_optional_models: Set[str],
    optional_model_stats: Dict[str, Dict[str, int]],
) -> Tuple[Optional[str], bool]:
    """Run configs from manifest. Returns (stopped_reason, required_completed)."""
    stopped_reason = None
    required_completed = False

    for _, cfg in manifest.iterrows():
        elapsed_wall = time.perf_counter() - wall_start
        run_block = cfg.get("run_block", "required")
        if run_block == "required" and elapsed_wall > MAX_WALL_SECONDS_REQUIRED:
            stopped_reason = "18_hour_required_wall_clock_limit"
            required_completed = True
            break
        if elapsed_wall > wall_limit:
            stopped_reason = "24_hour_total_wall_clock_limit"
            break

        dataset = cfg["dataset"]
        if dataset not in frames:
            continue

        model = cfg["model"]
        mitigation = cfg["mitigation"]
        protected = cfg["protected_attribute"]
        seed = int(cfg["seed"])
        key = config_key(dataset, model, mitigation, protected, seed, run_block)
        if key in done:
            continue

        if run_block == "optional_main_grid_extension" and model in skipped_optional_models:
            row = make_result_row(
                dataset=dataset,
                dataset_variant=DATASET_PLAN[dataset]["variant"],
                model=model,
                mitigation=mitigation,
                protected_attribute=protected,
                seed=seed,
                metrics={},
                runtime_seconds=0.0,
                status="skipped",
                error_message=f"Optional model {model} skipped after repeated failures",
                run_block=run_block,
            )
            rows.append(row)
            done.add(key)
            pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)
            continue

        eg_constraint = None
        if str(mitigation).startswith("ExponentiatedGradient_"):
            eg_constraint = mitigation.replace("ExponentiatedGradient_", "")
            rf_key = (dataset, eg_constraint)
            if rf_key in skip_rf_eg_keys and model == "random_forest":
                row = make_result_row(
                    dataset=dataset,
                    dataset_variant=DATASET_PLAN[dataset]["variant"],
                    model=model,
                    mitigation=mitigation,
                    protected_attribute=protected,
                    seed=seed,
                    metrics={},
                    runtime_seconds=0.0,
                    status="skipped",
                    error_message=(
                        f"RF-EG skipped: >= {RF_EG_SLOW_MAX_BEFORE_SKIP} runs exceeded "
                        f"{RF_EG_SLOW_THRESHOLD}s for {dataset}/{eg_constraint}"
                    ),
                    run_block=run_block,
                )
                rows.append(row)
                done.add(key)
                pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)
                continue

        plan = DATASET_PLAN[dataset]
        spec = specs_cache[dataset][protected]
        non_feature = get_non_feature_columns(plan["non_feature_key"])

        row = execute_config(
            dataset,
            plan["variant"],
            frames[dataset],
            plan["target"],
            non_feature,
            spec,
            model,
            mitigation,
            seed,
            eg_constraint=eg_constraint,
            run_block=run_block,
        )
        rows.append(row)
        done.add(key)
        pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)

        completed = [r for r in rows if r.get("status") in ("success", "failed")]
        if len(completed) >= 20:
            fail_rate = sum(1 for r in completed if r["status"] == "failed") / len(completed)
            if fail_rate > FAILURE_RATE_STOP:
                stopped_reason = f"failure_rate_{fail_rate:.2%}_exceeds_5%"
                break

        if (
            model == "random_forest"
            and eg_constraint
            and row["status"] == "success"
            and row["runtime_seconds"] > RF_EG_SLOW_THRESHOLD
        ):
            rf_key = (dataset, eg_constraint)
            rf_eg_slow_counts[rf_key] = rf_eg_slow_counts.get(rf_key, 0) + 1
            if rf_eg_slow_counts[rf_key] >= RF_EG_SLOW_MAX_BEFORE_SKIP:
                skip_rf_eg_keys.add(rf_key)

        if run_block == "optional_main_grid_extension":
            stats = optional_model_stats.setdefault(model, {"success": 0, "failed": 0})
            if row["status"] == "success":
                stats["success"] += 1
            elif row["status"] == "failed":
                stats["failed"] += 1
                total = stats["success"] + stats["failed"]
                if stats["failed"] >= OPTIONAL_MODEL_FAIL_THRESHOLD and (
                    stats["failed"] / total > 0.10
                ):
                    skipped_optional_models.add(model)

    return stopped_reason, required_completed


def write_summaries(
    results_df: pd.DataFrame,
    manifest: pd.DataFrame,
    summaries: Dict[str, Any],
    acs_cap: int,
    acs_cap_reason: str,
    wall_start: float,
    stopped_reason: Optional[str],
    optional_started: bool,
    rf_eg_slow_counts: Dict[Tuple[str, str], int],
    skip_rf_eg_keys: Set[Tuple[str, str]],
    skipped_optional_models: Set[str],
) -> Dict[str, Any]:
    ok = results_df[results_df["status"] == "success"]
    runtime_summary = (
        ok.groupby(["run_block", "dataset", "model", "mitigation"])["runtime_seconds"]
        .agg(["count", "mean", "sum"])
        .reset_index()
    )
    runtime_summary.to_csv(RESULTS_DIR / "phase4b_runtime_summary.csv", index=False)

    n_dup = int(
        results_df.duplicated(
            subset=[
                "run_block",
                "dataset",
                "model",
                "mitigation",
                "protected_attribute",
                "seed",
            ]
        ).sum()
    )

    req = results_df[results_df["run_block"] == "required"]
    opt = results_df[results_df["run_block"] == "optional_main_grid_extension"]

    summary = {
        "phase": "4B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": PHASE4B_SEEDS,
        "planned_required_configurations": int(len(manifest[manifest["run_block"] == "required"])),
        "planned_optional_configurations": 1890 if optional_started else 0,
        "completed_rows": int(len(results_df)),
        "required_completed_rows": int(len(req)),
        "optional_completed_rows": int(len(opt)),
        "n_success": int((results_df["status"] == "success").sum()),
        "n_failed": int((results_df["status"] == "failed").sum()),
        "n_skipped": int((results_df["status"] == "skipped").sum()),
        "n_duplicate_rows": n_dup,
        "required_n_success": int((req["status"] == "success").sum()),
        "required_n_failed": int((req["status"] == "failed").sum()),
        "required_n_skipped": int((req["status"] == "skipped").sum()),
        "optional_n_success": int((opt["status"] == "success").sum()) if not opt.empty else 0,
        "optional_n_failed": int((opt["status"] == "failed").sum()) if not opt.empty else 0,
        "optional_n_skipped": int((opt["status"] == "skipped").sum()) if not opt.empty else 0,
        "acs_sample_cap_rows": acs_cap,
        "acs_cap_reason": acs_cap_reason,
        "rf_eg_slow_counts": {f"{d}|{c}": v for (d, c), v in rf_eg_slow_counts.items()},
        "skip_rf_eg_keys": [f"{d}|{c}" for d, c in skip_rf_eg_keys],
        "skipped_optional_models": sorted(skipped_optional_models),
        "optional_extension_started": optional_started,
        "stopped_reason": stopped_reason,
        "total_wall_clock_seconds": time.perf_counter() - wall_start,
        "mean_runtime_success": float(ok["runtime_seconds"].mean()) if not ok.empty else None,
    }
    (RESULTS_DIR / "phase4b_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h ({seconds:.0f} s)"


def _mean_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(group_cols)[["accuracy", "spd", "cfs", "runtime_seconds"]]
        .mean()
        .reset_index()
    )


def generate_report(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    summaries: Dict[str, Any],
    manifest: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    req = results_df[results_df["run_block"] == "required"]
    opt = results_df[results_df["run_block"] == "optional_main_grid_extension"]
    ok_req = req[req["status"] == "success"]

    lines: List[str] = [
        "# Q1 Upgrade — Phase 4B Option B (30-seed) Report",
        "",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "Scope: final 30-seed Option B run (LR + RF main grid, LR-EG full scope, RF-EG Adult + UCI Bank only).",
        "",
        "---",
        "",
        "## 1. Final confirmation",
        "",
        "- **Final 30-seed Q1-upgrade experiment was run** (or partially completed; see stop reason).",
        "- **No manuscript files were modified** (`Paper/`, `Paper_Springer_JBigData/` untouched).",
        "- **No old result CSVs were overwritten** (`Code/results_30seeds/` untouched).",
        "- **All outputs are isolated under `q1_upgrade/`.**",
        "",
        "---",
        "",
        "## 2. Run design",
        "",
        "| Item | Setting |",
        "|------|---------|",
        f"| **Seeds** | 42–71 (30 seeds) |",
        "| **Models (required)** | Logistic Regression, Random Forest |",
        "| **Models (optional extension)** | Gradient Boosting, XGBoost, MLP |",
        "| **Main mitigations** | baseline, reweighing, equalized_odds |",
        "| **EG mitigations** | ExponentiatedGradient_DP, ExponentiatedGradient_EO |",
        f"| **Planned required configurations** | **{summary['planned_required_configurations']}** |",
        f"| **Required completed rows** | **{summary['required_completed_rows']}** |",
        f"| **Optional extension started** | **{summary['optional_extension_started']}** |",
        f"| **Optional completed rows** | **{summary['optional_completed_rows']}** |",
        f"| **Successful** | **{summary['n_success']}** |",
        f"| **Failed** | **{summary['n_failed']}** |",
        f"| **Skipped** | **{summary['n_skipped']}** |",
        f"| **Duplicate rows** | **{summary['n_duplicate_rows']}** |",
        f"| **Stopped reason** | {summary.get('stopped_reason') or 'none (completed)'} |",
        "",
        "### Datasets and protected settings",
        "",
        "| Dataset key | Protected (main + LR-EG) | RF-EG only |",
        "|-------------|--------------------------|------------|",
        "| `adult` | sex, race, age_group | sex |",
        "| `bank_uci` | age_group, job_group | age_group |",
        "| `acs_income_ca_2018` | sex, age_group | *(none — excluded)* |",
        "",
        "### Configuration counts (required)",
        "",
        "| Block | Count |",
        "|-------|------:|",
        "| Main grid (7 protected × 2 models × 3 mit × 30 seeds) | 1,260 |",
        "| LR-EG (7 protected × 2 constraints × 30 seeds) | 420 |",
        "| RF-EG (2 protected × 2 constraints × 30 seeds) | 120 |",
        "| **Total** | **1,800** |",
        "",
        "---",
        "",
        "## 3. Dataset summaries",
        "",
        f"Full JSON: `q1_upgrade/results/phase4b_optionB_30seed/phase4b_dataset_summaries.json`",
        "",
    ]

    for ds_key, label in [
        ("adult", "Adult"),
        ("bank_uci", "UCI Bank"),
        ("acs_income_ca_2018", "ACSIncome"),
    ]:
        meta = summaries.get(ds_key, {})
        lines.append(f"### {label}")
        lines.append("")
        if ds_key == "bank_uci":
            lines.append(f"- File: `{meta.get('expected_path', '')}`")
            lines.append(f"- SHA-256: `{meta.get('sha256', 'n/a')}`")
        if ds_key == "acs_income_ca_2018":
            lines.append(f"- Year/state: 2018 ACS, California")
            lines.append(f"- Full CA rows before sampling: {meta.get('full_ca_rows', 'n/a')}")
            lines.append(f"- Sample cap: {summary['acs_sample_cap_rows']}")
            lines.append(f"- Final sample: {meta.get('rows', 'n/a')} rows")
            lines.append(f"- Sample policy: {meta.get('sample_policy', 'n/a')}")
        lines.append(f"- Rows: {meta.get('rows', 'n/a')}")
        if "target_distribution" in meta:
            td = meta["target_distribution"]
            pos = td.get("1", td.get(1, "n/a"))
            lines.append(f"- Target positive count: {pos}")
        lines.append("")

    lines.extend(["---", "", "## 4. Results summary", ""])

    for block_name, block_df in [("Required", req), ("Optional extension", opt)]:
        if block_df.empty:
            continue
        lines.append(f"### {block_name}")
        lines.append("")
        lines.append(f"- Success: {(block_df['status'] == 'success').sum()}")
        lines.append(f"- Failed: {(block_df['status'] == 'failed').sum()}")
        lines.append(f"- Skipped: {(block_df['status'] == 'skipped').sum()}")
        lines.append("")

    if not ok_req.empty:
        seed42 = ok_req[ok_req["seed"] == 42]
        bl = seed42[
            (seed42["mitigation"] == "baseline") & (seed42["model"] == "logistic_regression")
        ]
        if not bl.empty:
            lines.append("### Key baseline metrics (LR, seed 42)")
            lines.append("")
            lines.append("| Dataset | Protected | Accuracy | SPD | CFS |")
            lines.append("|---------|-----------|----------|-----|-----|")
            for _, r in bl.iterrows():
                lines.append(
                    f"| {r['dataset']} | {r['protected_attribute']} | "
                    f"{r['accuracy']:.3f} | {r['spd']:.3f} | {r['cfs']:.3f} |"
                )
            lines.append("")

        eg = seed42[seed42["mitigation"].str.startswith("ExponentiatedGradient")]
        if not eg.empty:
            lines.append("### EG highlights (seed 42)")
            lines.append("")
            lines.append("| Dataset | Model | Mitigation | Accuracy | SPD | Runtime (s) |")
            lines.append("|---------|-------|------------|----------|-----|-------------|")
            for _, r in eg.iterrows():
                lines.append(
                    f"| {r['dataset']} | {r['model']} | {r['mitigation']} | "
                    f"{r['accuracy']:.3f} | {r['spd']:.3f} | {r['runtime_seconds']:.1f} |"
                )
            lines.append("")

    failed = results_df[results_df["status"] == "failed"]
    if not failed.empty:
        lines.append("### Failed configurations")
        lines.append("")
        for _, r in failed.head(20).iterrows():
            lines.append(
                f"- `{r['run_block']}` {r['dataset']}/{r['model']}/{r['mitigation']}/"
                f"{r['protected_attribute']}/seed={r['seed']}"
            )
        lines.append("")

    lines.extend(["---", "", "## 5. Runtime summary", ""])
    lines.append(f"- **Total wall clock:** {_fmt_hours(summary['total_wall_clock_seconds'])}")
    lines.append("")

    ok_all = results_df[results_df["status"] == "success"]
    for label, col in [
        ("dataset", "dataset"),
        ("model", "model"),
        ("mitigation", "mitigation"),
    ]:
        if not ok_all.empty:
            means = ok_all.groupby(col)["runtime_seconds"].mean()
            lines.append(f"**Mean runtime by {label}:**")
            for k, v in means.items():
                lines.append(f"- {k}: {v:.2f} s")
            lines.append("")

    lr_eg = ok_all[
        (ok_all["model"] == "logistic_regression")
        & (ok_all["mitigation"].str.startswith("ExponentiatedGradient"))
    ]
    rf_eg = ok_all[
        (ok_all["model"] == "random_forest")
        & (ok_all["mitigation"].str.startswith("ExponentiatedGradient"))
    ]
    if not lr_eg.empty:
        lines.append(f"- **LR-EG mean runtime:** {lr_eg['runtime_seconds'].mean():.2f} s")
    if not rf_eg.empty:
        lines.append(f"- **RF-EG mean runtime:** {rf_eg['runtime_seconds'].mean():.2f} s")
    lines.append("")

    for block in ["required", "optional_main_grid_extension"]:
        block_ok = ok_all[ok_all["run_block"] == block]
        if not block_ok.empty:
            lines.append(
                f"- **{block} runtime:** {_fmt_hours(block_ok['runtime_seconds'].sum())}"
            )
    lines.append("")

    lines.extend(["---", "", "## 6. Recommendation", ""])
    req_ok = summary["required_n_success"]
    req_planned = summary["planned_required_configurations"]
    fail_rate = summary["required_n_failed"] / max(req_ok + summary["required_n_failed"], 1)
    if req_ok >= req_planned * 0.95 and fail_rate < 0.05:
        rec = "Proceed to **Phase 5** post-processing and manuscript rewrite."
    elif fail_rate >= 0.05:
        rec = "Review failures and consider a targeted rerun before Phase 5."
    else:
        rec = "Complete remaining required configurations, then proceed to Phase 5."
    lines.append(rec)
    lines.append("")

    retain_optional = (
        summary["optional_extension_started"]
        and summary["optional_n_success"] > 0
        and summary.get("stopped_reason") != "18_hour_required_wall_clock_limit"
    )
    if summary["optional_extension_started"]:
        lines.append(
            f"**Optional extension retention:** "
            f"{'Retain in manuscript for five-model comparability' if retain_optional else 'Do not retain — insufficient coverage or time budget exceeded'}."
        )
        lines.append("")

    lines.extend(["---", "", "## 7. Manuscript implication (no edits made)", ""])
    lines.extend(
        [
            "If Phase 5 proceeds, the following sections would need revision:",
            "",
            "- **Abstract:** Update dataset list (UCI Bank full, ACSIncome CA 2018 50k), model scope (LR/RF + EG; optional GB/XGB/MLP), and headline fairness findings from 30-seed aggregates.",
            "- **Dataset Description:** Document UCI Bank canonical file, duration removal, ACS 50k stratified cap, and protected-attribute definitions.",
            "- **Methods:** State LR-EG on all protected settings; RF-EG limited to Adult (sex) and UCI Bank (age_group); no RF-EG on ACSIncome.",
            "- **Results:** Replace 10-seed pilot aggregates with 30-seed tables; add EG blocks; optional extension tables if retained.",
            "- **Discussion:** Interpret ACS RF-EG exclusion; compare AIF360 vs Fairlearn EG trade-offs.",
            "- **Limitations:** Note ACS subsample cap, RF-EG scope restriction, and optional model coverage.",
            "- **Supplementary:** Full config manifest, runtime fact sheet, and per-seed result CSV.",
            "",
        ]
    )

    report_path = REPORTS_DIR / "q1_upgrade_phase4b_optionB_30seed_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_phase5_ready(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    summaries: Dict[str, Any],
) -> None:
    PHASE5_READY_DIR.mkdir(parents=True, exist_ok=True)
    ok = results_df[results_df["status"] == "success"]

    checklist = """# Phase 5 Analysis Readiness Checklist

## Post-processing scripts to update
- Add Phase 4B result loader pointing to `q1_upgrade/results/phase4b_optionB_30seed/phase4b_results.csv`
- Filter by `run_block` when aggregating required vs optional extension
- Update aggregation scripts to handle `bank_uci` and `acs_income_ca_2018` dataset keys
- Regenerate fairness metric summaries with 30-seed means and confidence intervals
- Separate LR-EG and RF-EG blocks in tables

## Manuscript sections that must change
- Abstract, Dataset Description, Methods, Results, Discussion, Limitations
- Supplementary: config manifest, runtime fact sheet, full result CSV

## Tables and figures to regenerate
- Main results table (baseline, reweighing, equalized odds) per dataset × protected × model
- EG comparison table (LR-EG full scope; RF-EG Adult + Bank only)
- Optional: GB/XGB/MLP main-grid table if extension retained
- Runtime summary figure or table
- CFS/ACFS comparison charts

## Claims now supported (pending Phase 5 aggregation)
- 30-seed Option B results on Adult, full UCI Bank, and ACSIncome CA 2018 (50k cap)
- LR-EG on all seven protected settings
- RF-EG on Adult (sex) and UCI Bank (age_group) only
- AIF360 mitigations (baseline, reweighing, equalized odds) on LR and RF

## Claims to remove or soften
- Any statement implying RF-EG on ACSIncome
- Any 10-seed Phase 4A numbers presented as final
- Five-model comparability unless optional extension is retained and aggregated
- Cross-dataset generalization beyond the three benchmark datasets tested
"""
    (PHASE5_READY_DIR / "analysis_readiness_checklist.md").write_text(checklist, encoding="utf-8")

    prov_rows = []
    for ds_key, source, target, protected in [
        ("adult", "Code/data/adult.csv (read-only)", "income", "sex; race; age_group"),
        (
            "bank_uci",
            "q1_upgrade/data/raw/uci_bank/bank-additional-full.csv",
            "label (from y)",
            "age_group; job_group",
        ),
        (
            "acs_income_ca_2018",
            "Folktables ACS 2018 1-Year CA person",
            "label",
            "sex; age_group",
        ),
    ]:
        meta = summaries.get(ds_key, {})
        rows_n = meta.get("rows", "")
        td = meta.get("target_distribution", {})
        pos = td.get("1", td.get(1, ""))
        total = meta.get("rows", 1)
        pos_rate = f"{int(pos) / int(total):.3f}" if pos and total else ""
        prov_rows.append(
            {
                "dataset_key": ds_key,
                "source": source,
                "row_count": rows_n,
                "target": target,
                "positive_rate": pos_rate,
                "protected_settings": protected,
                "sha256_fingerprint": meta.get("sha256", ""),
                "sample_policy": meta.get(
                    "sample_policy",
                    "full dataset" if ds_key != "acs_income_ca_2018" else "",
                ),
            }
        )
    pd.DataFrame(prov_rows).to_csv(PHASE5_READY_DIR / "dataset_provenance_table.csv", index=False)

    method_rows = [
        {
            "model": "logistic_regression",
            "mitigation": "baseline; reweighing; equalized_odds",
            "datasets": "adult; bank_uci; acs_income_ca_2018",
            "protected_settings": "all planned per dataset",
            "scope": "full",
            "limitation_reason": "",
        },
        {
            "model": "random_forest",
            "mitigation": "baseline; reweighing; equalized_odds",
            "datasets": "adult; bank_uci; acs_income_ca_2018",
            "protected_settings": "all planned per dataset",
            "scope": "full",
            "limitation_reason": "",
        },
        {
            "model": "logistic_regression",
            "mitigation": "ExponentiatedGradient_DP; ExponentiatedGradient_EO",
            "datasets": "adult; bank_uci; acs_income_ca_2018",
            "protected_settings": "all planned per dataset",
            "scope": "full",
            "limitation_reason": "",
        },
        {
            "model": "random_forest",
            "mitigation": "ExponentiatedGradient_DP; ExponentiatedGradient_EO",
            "datasets": "adult; bank_uci",
            "protected_settings": "adult: sex; bank_uci: age_group",
            "scope": "limited",
            "limitation_reason": "ACS RF-EG excluded after Phase 4A runtime probe",
        },
        {
            "model": "gradient_boosting; xgboost; mlp",
            "mitigation": "baseline; reweighing; equalized_odds",
            "datasets": "adult; bank_uci; acs_income_ca_2018",
            "protected_settings": "all planned per dataset",
            "scope": "optional extension",
            "limitation_reason": "No EG; only if optional block completed within time budget",
        },
    ]
    pd.DataFrame(method_rows).to_csv(PHASE5_READY_DIR / "method_scope_table.csv", index=False)

    fact_lines = [
        "# Phase 4B Runtime Fact Sheet",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"## Total runtime",
        f"- Wall clock: {_fmt_hours(summary['total_wall_clock_seconds'])}",
        "",
    ]
    for block in ["required", "optional_main_grid_extension"]:
        block_df = ok[ok["run_block"] == block]
        if not block_df.empty:
            fact_lines.append(f"## {block}")
            fact_lines.append(f"- Sum runtime: {_fmt_hours(block_df['runtime_seconds'].sum())}")
            fact_lines.append(f"- Configurations: {len(block_df)}")
            fact_lines.append("")

    for label, col in [("dataset", "dataset"), ("model", "model"), ("mitigation", "mitigation")]:
        if not ok.empty:
            fact_lines.append(f"## Mean runtime by {label}")
            for k, v in ok.groupby(col)["runtime_seconds"].mean().items():
                fact_lines.append(f"- {k}: {v:.2f} s")
            fact_lines.append("")

    lr_eg = ok[
        (ok["model"] == "logistic_regression")
        & (ok["mitigation"].str.startswith("ExponentiatedGradient"))
    ]
    rf_eg = ok[
        (ok["model"] == "random_forest")
        & (ok["mitigation"].str.startswith("ExponentiatedGradient"))
    ]
    if not lr_eg.empty:
        fact_lines.append(f"## LR-EG")
        fact_lines.append(f"- Mean: {lr_eg['runtime_seconds'].mean():.2f} s")
        fact_lines.append(f"- Sum: {_fmt_hours(lr_eg['runtime_seconds'].sum())}")
        fact_lines.append("")
    if not rf_eg.empty:
        fact_lines.append(f"## RF-EG")
        fact_lines.append(f"- Mean: {rf_eg['runtime_seconds'].mean():.2f} s")
        fact_lines.append(f"- Sum: {_fmt_hours(rf_eg['runtime_seconds'].sum())}")
        fact_lines.append("")

    skipped = results_df[results_df["status"].isin(["skipped", "failed"])]
    if not skipped.empty:
        fact_lines.append("## Skipped / failed configurations")
        for _, r in skipped.iterrows():
            fact_lines.append(
                f"- [{r['status']}] {r['run_block']} {r['dataset']}/{r['model']}/"
                f"{r['mitigation']}/{r['protected_attribute']}/seed={r['seed']}"
            )

    (PHASE5_READY_DIR / "runtime_fact_sheet.md").write_text("\n".join(fact_lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    acs_cap, acs_cap_reason = choose_acs_sample_cap()

    summaries, frames = load_datasets(acs_cap)
    summaries["acs_sample_cap_decision"] = {"cap_rows": acs_cap, "reason": acs_cap_reason}
    (RESULTS_DIR / "phase4b_dataset_summaries.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )

    required_manifest = build_required_manifest()
    optional_manifest = build_optional_manifest()
    full_manifest = pd.concat([required_manifest, optional_manifest], ignore_index=True)
    full_manifest.to_csv(RESULTS_DIR / "phase4b_config_manifest.csv", index=False)

    done: Set[str] = set()
    rows: List[Dict[str, Any]] = []
    if PARTIAL_CSV.exists():
        prev = pd.read_csv(PARTIAL_CSV)
        rows = prev.to_dict(orient="records")
        for r in rows:
            done.add(
                config_key(
                    r["dataset"],
                    r["model"],
                    r["mitigation"],
                    r["protected_attribute"],
                    int(r["seed"]),
                    r.get("run_block", "required"),
                )
            )

    specs_cache = {d: {s.name: s for s in get_specs(d)} for d in DATASET_PLAN}
    rf_eg_slow_counts: Dict[Tuple[str, str], int] = {}
    skip_rf_eg_keys: Set[Tuple[str, str]] = set()
    skipped_optional_models: Set[str] = set()
    optional_model_stats: Dict[str, Dict[str, int]] = {}
    stopped_reason: Optional[str] = None
    optional_started = False

    stopped_reason, _ = run_manifest(
        required_manifest,
        frames,
        specs_cache,
        rows,
        done,
        wall_start,
        MAX_WALL_SECONDS_TOTAL,
        rf_eg_slow_counts,
        skip_rf_eg_keys,
        skipped_optional_models,
        optional_model_stats,
    )

    required_done = {k for k in done if k.startswith("required|")}
    required_complete = len(required_done) >= len(required_manifest)

    if (
        stopped_reason is None
        and required_complete
        and (time.perf_counter() - wall_start) < MAX_WALL_SECONDS_REQUIRED
    ):
        optional_started = True
        opt_stopped, _ = run_manifest(
            optional_manifest,
            frames,
            specs_cache,
            rows,
            done,
            wall_start,
            MAX_WALL_SECONDS_TOTAL,
            rf_eg_slow_counts,
            skip_rf_eg_keys,
            skipped_optional_models,
            optional_model_stats,
        )
        if opt_stopped:
            stopped_reason = opt_stopped

    results_df = pd.DataFrame(rows)
    results_df.to_csv(FINAL_CSV, index=False)

    run_summary = write_summaries(
        results_df,
        full_manifest,
        summaries,
        acs_cap,
        acs_cap_reason,
        wall_start,
        stopped_reason,
        optional_started,
        rf_eg_slow_counts,
        skip_rf_eg_keys,
        skipped_optional_models,
    )

    generate_report(results_df, run_summary, summaries, full_manifest)
    generate_phase5_ready(results_df, run_summary, summaries)

    print(
        f"Phase 4B: {run_summary['n_success']}/{run_summary['completed_rows']} succeeded "
        f"(required planned {run_summary['planned_required_configurations']})",
        flush=True,
    )
    if optional_started:
        print(
            f"Optional extension: {run_summary['optional_n_success']} success, "
            f"{run_summary['optional_n_failed']} failed",
            flush=True,
        )
    if stopped_reason:
        print(f"Stopped early: {stopped_reason}", flush=True)


if __name__ == "__main__":
    main()
