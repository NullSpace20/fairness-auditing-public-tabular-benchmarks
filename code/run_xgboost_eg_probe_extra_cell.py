"""Extra XGBoost EG probe cell (W2) — supplementary sensitivity only.

Primary: Adult age_group. Fallback: ACSIncome sex if primary is unstable/costly.
Not part of the 3,690-run main grid; not plotted in Figure 6 or Table 7.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd

from exponentiated_gradient_mitigation import fit_predict_exponentiated_gradient
from fairness_utils import compute_cfs
from loaders import load_acs_income_ca_2018, load_adult_readonly
from pipeline_core import (
    fairness_metrics,
    get_acs_protected_specs,
    get_adult_protected_specs,
    get_non_feature_columns,
    performance_metrics,
    prepare_split,
)

Q1_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Q1_ROOT / "results" / "xgboost_eg_probe_extra_cell"

PRIMARY = {
    "setting_id": "adult_age_group",
    "dataset": "adult",
    "protected_attribute": "age_group",
    "loader": load_adult_readonly,
    "specs_fn": get_adult_protected_specs,
    "target": "income",
    "dataset_key": "adult",
}

FALLBACK = {
    "setting_id": "acs_income_sex",
    "dataset": "acs_income_ca_2018",
    "protected_attribute": "sex",
    "loader": load_acs_income_ca_2018,
    "specs_fn": get_acs_protected_specs,
    "target": "label",
    "dataset_key": "acs_income",
}

CONSTRAINTS = [("DP", "EG-DP"), ("EO", "EG-EO")]
FULL_SEEDS = list(range(42, 72))
FEASIBILITY_SEEDS = [42, 43]
# If more than this fraction of feasibility runs fail, switch to fallback.
MAX_FAILURE_RATE = 0.0  # any failure on primary triggers fallback check


def get_spec(setting: dict):
    specs = setting["specs_fn"]()
    return next(s for s in specs if s.name == setting["protected_attribute"])


def run_one(setting: dict, constraint: str, seed: int, df) -> dict:
    spec = get_spec(setting)
    prot_cols = get_non_feature_columns(setting["dataset_key"])
    split = prepare_split(df, setting["target"], prot_cols, spec, seed, with_val=False)
    X_train, X_test, y_train, y_test, prot_train, prot_test = split[:6]

    t0 = time.perf_counter()
    base = {
        "analysis": "extra_eg_probe",
        "setting_id": setting["setting_id"],
        "dataset": setting["dataset"],
        "protected_attribute": setting["protected_attribute"],
        "model": "xgboost",
        "constraint": constraint,
        "mitigation": f"extra_EG-{constraint}",
        "seed": seed,
        "not_part_of_main_grid": True,
    }
    try:
        y_pred, mit_name = fit_predict_exponentiated_gradient(
            X_train,
            y_train,
            prot_train,
            X_test,
            constraint=constraint,
            seed=seed,
            model_name="xgboost",
        )
        fair = fairness_metrics(y_test, y_pred, prot_test, spec)
        perf = performance_metrics(y_test, y_pred)
        cfs = compute_cfs(
            abs(fair["spd"]),
            abs(1 - fair["di"]) if fair["di"] else 0.0,
            abs(fair["eod"]),
            abs(fair["aod"]),
        )
        return {
            **base,
            "mitigation": mit_name,
            "accuracy": perf["accuracy"],
            "abs_spd": abs(fair["spd"]),
            "abs_eod": abs(fair["eod"]),
            "cfs": float(cfs),
            "runtime_seconds": time.perf_counter() - t0,
            "status": "success",
            "error": "",
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - t0,
        }


def run_setting(setting: dict, seeds: List[int]) -> Tuple[pd.DataFrame, dict]:
    df, load_meta = setting["loader"]()
    if df is None:
        raise RuntimeError(f"Failed to load {setting['setting_id']}: {load_meta}")
    rows = []
    for c_code, _ in CONSTRAINTS:
        for seed in seeds:
            rows.append(run_one(setting, c_code, seed, df))
            r = rows[-1]
            print(
                setting["setting_id"],
                c_code,
                seed,
                r["status"],
                f"{r.get('runtime_seconds', 0):.1f}s",
                flush=True,
            )
    per_seed = pd.DataFrame(rows)
    n_fail = int((per_seed["status"] == "error").sum())
    meta = {
        "setting_id": setting["setting_id"],
        "dataset": setting["dataset"],
        "protected_attribute": setting["protected_attribute"],
        "seeds": f"{seeds[0]}-{seeds[-1]}",
        "n_runs": len(per_seed),
        "n_success": int((per_seed["status"] == "success").sum()),
        "n_error": n_fail,
        "load_meta": load_meta,
    }
    return per_seed, meta


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    ok = per_seed[per_seed["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby(["setting_id", "dataset", "protected_attribute", "constraint"])
        .agg(
            n_seeds=("seed", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_abs_spd=("abs_spd", "mean"),
            mean_abs_eod=("abs_eod", "mean"),
            mean_cfs=("cfs", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )


def choose_setting(seeds: List[int], force: str = "") -> Tuple[dict, str]:
    if force == "primary":
        return PRIMARY, "forced primary (Adult age_group)"
    if force == "fallback":
        return FALLBACK, "forced fallback (ACSIncome sex)"

    print("Feasibility check on primary: Adult age_group ...")
    primary_df, primary_meta = run_setting(PRIMARY, seeds)
    fail_rate = primary_meta["n_error"] / max(primary_meta["n_runs"], 1)
    if fail_rate <= MAX_FAILURE_RATE and primary_meta["n_success"] == primary_meta["n_runs"]:
        return PRIMARY, "primary accepted (all feasibility runs succeeded)"

    print(
        f"Primary had {primary_meta['n_error']} failures; "
        "switching to fallback: ACSIncome sex ..."
    )
    return FALLBACK, (
        f"fallback used: Adult age_group had {primary_meta['n_error']} error(s) "
        f"in {primary_meta['n_runs']} feasibility runs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra XGBoost EG probe cell (W2)")
    parser.add_argument("--feasibility", action="store_true", help="Seeds 42-43 only")
    parser.add_argument(
        "--setting",
        choices=["auto", "primary", "fallback"],
        default="auto",
        help="Probe setting (default: auto with fallback on failure)",
    )
    args = parser.parse_args()
    seeds = FEASIBILITY_SEEDS if args.feasibility else FULL_SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.setting == "auto" and not args.feasibility:
        # Quick 2-seed check then full run on chosen setting
        setting, reason = choose_setting(FEASIBILITY_SEEDS)
        if setting["setting_id"] != PRIMARY["setting_id"]:
            pass  # already know fallback
        elif reason.startswith("primary accepted"):
            setting = PRIMARY
        else:
            setting = FALLBACK
        print(f"Selected setting: {setting['setting_id']} ({reason})")
        per_seed, run_meta = run_setting(setting, FULL_SEEDS)
        selection_reason = reason
    elif args.setting == "auto":
        setting, reason = choose_setting(seeds)
        per_seed, run_meta = run_setting(setting, seeds)
        selection_reason = reason
    else:
        setting = PRIMARY if args.setting == "primary" else FALLBACK
        per_seed, run_meta = run_setting(setting, seeds)
        selection_reason = f"manual --setting {args.setting}"

    summary = summarize(per_seed)
    tag = "feasibility" if args.feasibility else "full"

    per_path = OUT_DIR / f"extra_eg_probe_cell_{tag}_per_seed.csv"
    summ_path = OUT_DIR / f"extra_eg_probe_cell_{tag}_summary.csv"
    per_seed.to_csv(per_path, index=False)
    summary.to_csv(summ_path, index=False)

    meta = {
        "analysis": "optional_extra_eg_probe_cell_W2",
        "label": "extra EG probe",
        "not_part_of_main_grid": True,
        "not_in_figure_6": True,
        "not_in_table_7_pooled": True,
        "model": "xgboost",
        "constraints": ["DP", "EO"],
        "primary_setting": PRIMARY["setting_id"],
        "fallback_setting": FALLBACK["setting_id"],
        "selected_setting": setting["setting_id"],
        "selection_reason": selection_reason,
        "seeds": f"{seeds[0]}-{seeds[-1]}",
        "n_runs_attempted": int(len(per_seed)),
        "n_success": int((per_seed["status"] == "success").sum()),
        "n_error": int((per_seed["status"] == "error").sum()),
        "run_meta": run_meta,
    }
    meta_path = OUT_DIR / f"run_metadata_{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {per_path}")
    print(f"Wrote {summ_path}")
    print(f"Wrote {meta_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
