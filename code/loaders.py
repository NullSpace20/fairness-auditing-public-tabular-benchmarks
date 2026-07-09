"""Q1-upgrade dataset loaders (isolated; read-only on Code/data for Adult)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline_core import (
    Q1_ROOT,
    CODE_DATA_DIR,
    add_age_pairwise_columns,
    age_band_label,
    age_group_value,
    get_non_feature_columns,
    job_group_value,
)

RAW_UCI_BANK_PATH = Q1_ROOT / "data" / "raw" / "uci_bank" / "bank-additional-full.csv"
FOLKTABLES_CACHE = Q1_ROOT / "data" / "folktables_cache"
ACS_PILOT_SUBSAMPLE_ROWS = 8000
ACS_PHASE4A_SUBSAMPLE_ROWS = 50000
ACS_PILOT_SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_loader(df: pd.DataFrame, target_col: str, protected_cols: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "target_column": target_col,
        "missing_values_total": int(df.isna().sum().sum()),
        "target_distribution": {
            str(k): int(v) for k, v in df[target_col].value_counts(dropna=False).items()
        },
    }
    protected = {}
    for col in protected_cols:
        if col in df.columns:
            protected[col] = {
                "n_unique": int(df[col].nunique(dropna=True)),
                "value_counts": {
                    str(k): int(v)
                    for k, v in df[col].value_counts(dropna=False).head(8).items()
                },
            }
    summary["candidate_protected_attributes"] = protected
    return summary


def load_uci_bank_additional() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    meta: Dict[str, Any] = {"expected_path": str(RAW_UCI_BANK_PATH)}
    if not RAW_UCI_BANK_PATH.exists():
        meta["status"] = "skipped"
        meta["reason"] = "Full UCI Bank file not found; smoke test skipped."
        return None, meta

    df = pd.read_csv(RAW_UCI_BANK_PATH, sep=";")
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    df["label"] = df["y"].map({"yes": 1, "no": 0})
    df = df.drop(columns=["y"])
    df["_age_band"] = df["age"].apply(age_band_label)
    df["age_group"] = df["age"].apply(age_group_value)
    df = add_age_pairwise_columns(df)
    df["job_group"] = df["job"].apply(job_group_value)
    drop_cols = ["age", "job"]
    non_feature = get_non_feature_columns("bank_uci")
    feature_cols = [c for c in df.columns if c not in drop_cols + ["label"] + non_feature]
    out = df[feature_cols + ["label"] + non_feature]
    meta["status"] = "loaded"
    meta["dataset_key"] = "bank_uci"
    meta["sha256"] = sha256_file(RAW_UCI_BANK_PATH)
    meta["positive_rate"] = float(out["label"].mean())
    meta["duration_dropped"] = True
    meta["loader_summary"] = summarize_loader(
        out, "label", ["age_group", "job_group"]
    )
    meta["age_group_nan_count"] = int(out["age_group"].isna().sum())
    meta["job_group_nan_count"] = int(out["job_group"].isna().sum())
    return out, meta


def load_adult_readonly() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = CODE_DATA_DIR / "adult.csv"
    df = pd.read_csv(path)
    df = df.replace("?", "Unknown")
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    df["sex_binary"] = df["sex"].map({"Male": 1, "Female": 0})
    df["race_binary"] = df["race"].apply(lambda r: 1 if r == "White" else 0)
    df["_age_band"] = df["age"].apply(age_band_label)
    df["age_group"] = df["age"].apply(age_group_value)
    df = add_age_pairwise_columns(df)
    drop_cols = ["fnlwgt", "sex", "age", "race"]
    non_feature = get_non_feature_columns("adult")
    feature_cols = [c for c in df.columns if c not in drop_cols + ["income"] + non_feature]
    out = df[feature_cols + ["income"] + non_feature]
    meta = {
        "status": "loaded",
        "source_path": str(path),
        "read_only": True,
        "loader_summary": summarize_loader(out, "income", ["sex_binary", "age_group"]),
    }
    return out, meta


def _load_acs_ca_frame(max_rows: int) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError as exc:
        return None, {
            "status": "skipped",
            "reason": "folktables not installed",
            "install_command": "pip install folktables",
            "error": str(exc),
        }

    data_source = ACSDataSource(
        survey_year="2018",
        horizon="1-Year",
        survey="person",
        root_dir=str(FOLKTABLES_CACHE),
    )
    acs_df = data_source.get_data(states=["CA"], download=True)
    features, labels, _group = ACSIncome.df_to_pandas(acs_df)
    df = features.copy()
    df["label"] = labels.astype(int)
    df["AGEP"] = df["AGEP"].astype(int)
    df["_age_band"] = df["AGEP"].apply(age_band_label)
    df["age_group"] = df["AGEP"].apply(age_group_value)
    df = add_age_pairwise_columns(df)
    df["sex_binary"] = np.where(df["SEX"] == 1, 1, 0)
    df["race_binary"] = np.where(df["RAC1P"] == 1, 1, 0)

    full_rows = int(len(df))
    if len(df) > max_rows:
        df = (
            df.groupby("label", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, int(max_rows * len(g) / len(df))),
                    random_state=ACS_PILOT_SEED,
                )
            )
            .reset_index(drop=True)
        )
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=ACS_PILOT_SEED)

    drop_cols = ["SEX", "RAC1P", "AGEP"]
    non_feature = get_non_feature_columns("acs_income")
    feature_cols = [c for c in df.columns if c not in drop_cols + ["label"] + non_feature]
    out = df[feature_cols + ["label"] + non_feature]
    meta = {
        "full_ca_rows_before_subsample": full_rows,
        "subsample_rows_cap": max_rows,
        "final_sample_rows": int(len(out)),
        "sample_seed": ACS_PILOT_SEED,
    }
    return out, meta


def load_acs_income_pilot() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "dataset_key": "acs_income_ca_2018_pilot",
        "pilot": "ACSIncome 2018 California 1-Year person survey",
        "survey_year": "2018",
        "state": "CA",
        "subsample_rows": ACS_PILOT_SUBSAMPLE_ROWS,
        "sample_policy": (
            "Stratified subsample capped at 8000 rows with fixed seed 42 on full CA load; "
            "pilot experiment seeds (42-44) affect train/test splits only, not the row sample."
        ),
    }
    out, frame_meta = _load_acs_ca_frame(ACS_PILOT_SUBSAMPLE_ROWS)
    if out is None:
        meta.update(frame_meta)
        return None, meta
    meta.update(frame_meta)
    meta["status"] = "loaded"
    meta["loader_summary"] = summarize_loader(out, "label", ["sex_binary", "age_group"])
    meta["sex_binary_nan_count"] = int(out["sex_binary"].isna().sum())
    meta["age_group_nan_count"] = int(out["age_group"].isna().sum())
    return out, meta


def load_acs_income_ca_2018(max_rows: int = ACS_PHASE4A_SUBSAMPLE_ROWS) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "dataset_key": "acs_income_ca_2018",
        "survey_year": "2018",
        "state": "CA",
        "requested_cap": max_rows,
        "sample_policy": (
            f"Stratified subsample capped at {max_rows} rows with fixed seed 42 on full CA load; "
            "experiment seeds affect train/test splits only."
        ),
    }
    out, frame_meta = _load_acs_ca_frame(max_rows)
    if out is None:
        meta.update(frame_meta)
        return None, meta
    meta.update(frame_meta)
    meta["status"] = "loaded"
    meta["loader_summary"] = summarize_loader(out, "label", ["sex_binary", "age_group"])
    meta["sex_binary_nan_count"] = int(out["sex_binary"].isna().sum())
    meta["age_group_nan_count"] = int(out["age_group"].isna().sum())
    return out, meta


def save_loader_summary(name: str, meta: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"loader_summary_{name}.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
