"""Minimal pipeline helpers copied/adapted from Code/fairness_experiments.py for Q1 smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairness_utils import compute_acfs, compute_cfs

Q1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Q1_ROOT.parent
CODE_DATA_DIR = PROJECT_ROOT / "Code" / "data"

TEST_SIZE = 0.2
VAL_FRACTION = 0.25
SMOKE_SEED = 42
PRIVILEGED_JOBS = {"management", "admin.", "technician"}


@dataclass(frozen=True)
class ProtectedSpec:
    name: str
    column: str
    privileged_value: int
    unprivileged_value: int
    description: str
    filter_column: Optional[str] = None
    allowed_filter_values: Optional[Tuple[str, ...]] = None


def age_band_label(age: int) -> str:
    if age <= 30:
        return "young"
    if age >= 61:
        return "old"
    return "middle"


def age_group_value(age: int) -> int:
    if 31 <= age <= 60:
        return 1
    return 0


def add_age_pairwise_columns(df: pd.DataFrame) -> pd.DataFrame:
    bands = df["_age_band"]
    df["age_young_vs_middle"] = np.where(
        bands.isin(["young", "middle"]),
        np.where(bands == "middle", 1, 0),
        np.nan,
    )
    df["age_old_vs_middle"] = np.where(
        bands.isin(["old", "middle"]),
        np.where(bands == "middle", 1, 0),
        np.nan,
    )
    return df


def job_group_value(job: str) -> int:
    return 1 if job in PRIVILEGED_JOBS else 0


def get_non_feature_columns(dataset: str) -> List[str]:
    age_pairwise = ["age_young_vs_middle", "age_old_vs_middle"]
    if dataset == "acs_income":
        return ["sex_binary", "race_binary", "age_group", *age_pairwise, "_age_band"]
    if dataset == "adult":
        return ["sex_binary", "race_binary", "age_group", *age_pairwise, "_age_band"]
    return ["age_group", "job_group", *age_pairwise, "_age_band"]


def get_bank_protected_specs() -> List[ProtectedSpec]:
    age_young_middle = ProtectedSpec(
        "age_young_vs_middle",
        "age_young_vs_middle",
        1,
        0,
        "Young vs middle",
        filter_column="_age_band",
        allowed_filter_values=("young", "middle"),
    )
    return [
        ProtectedSpec("age_group", "age_group", 1, 0, "Middle-aged privileged"),
        ProtectedSpec("job_group", "job_group", 1, 0, "White-collar privileged"),
        age_young_middle,
    ]


def get_acs_protected_specs() -> List[ProtectedSpec]:
    return [
        ProtectedSpec("sex", "sex_binary", 1, 0, "Male privileged (ACS SEX=1)"),
        ProtectedSpec("age_group", "age_group", 1, 0, "Middle-aged privileged"),
    ]


def get_adult_protected_specs() -> List[ProtectedSpec]:
    return [
        ProtectedSpec("sex", "sex_binary", 1, 0, "Male privileged"),
        ProtectedSpec("race", "race_binary", 1, 0, "White privileged (1), non-White unprivileged (0)"),
        ProtectedSpec("age_group", "age_group", 1, 0, "Middle-aged privileged"),
    ]


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    categorical_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )


def densify(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    spec: ProtectedSpec,
) -> Dict[str, float]:
    df = pd.DataFrame(
        {
            "label": np.asarray(y_true).astype(int).ravel(),
            "prediction": np.asarray(y_pred).astype(int).ravel(),
            spec.column: np.asarray(protected).astype(int).ravel(),
        }
    )
    privileged_groups = [{spec.column: spec.privileged_value}]
    unprivileged_groups = [{spec.column: spec.unprivileged_value}]
    dataset_true = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df.drop(columns=["prediction"]),
        label_names=["label"],
        protected_attribute_names=[spec.column],
    )
    dataset_pred = dataset_true.copy()
    dataset_pred.labels = df["prediction"].values.reshape(-1, 1)
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    return {
        "spd": metric.statistical_parity_difference(),
        "di": metric.disparate_impact(),
        "eod": metric.equal_opportunity_difference(),
        "aod": metric.average_odds_difference(),
    }


def prepare_split(
    df: pd.DataFrame,
    target_col: str,
    protected_cols: List[str],
    spec: ProtectedSpec,
    seed: int,
    with_val: bool = False,
):
    df_run = df
    if spec.filter_column and spec.allowed_filter_values:
        df_run = df[df[spec.filter_column].isin(spec.allowed_filter_values)].copy()
    if len(df_run) < 50:
        raise ValueError(f"Too few rows ({len(df_run)}) after filtering for {spec.name}")

    # age_group and job_group use integer columns with no NaN; pairwise columns are excluded
    if df_run[spec.column].isna().any():
        raise ValueError(
            f"Protected column {spec.column} contains NaN for setting {spec.name}; "
            "filter rows or use a non-pairwise setting."
        )

    feature_cols = [c for c in df_run.columns if c not in [target_col] + protected_cols]
    X = df_run[feature_cols]
    y = df_run[target_col].values
    protected = df_run[spec.column].values.astype(int)

    X_train_full, X_test, y_train_full, y_test, prot_train_full, prot_test = train_test_split(
        X, y, protected, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    preprocessor = build_preprocessor(pd.concat([X_train_full, X_test]), feature_cols)
    X_train_full_p = preprocessor.fit_transform(X_train_full)
    X_test_p = preprocessor.transform(X_test)

    if not with_val:
        return (
            X_train_full_p,
            X_test_p,
            y_train_full,
            y_test,
            prot_train_full,
            prot_test,
            len(df_run),
        )

    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X_train_full_p,
        y_train_full,
        prot_train_full,
        test_size=VAL_FRACTION,
        stratify=y_train_full,
        random_state=seed,
    )
    return (
        X_train,
        X_val,
        X_test_p,
        y_train,
        y_val,
        y_test,
        prot_train,
        prot_val,
        prot_test,
        len(df_run),
    )


def fit_lr_baseline(X_train, y_train, X_test, seed: int) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    clf.fit(densify(X_train), y_train)
    return clf.predict(densify(X_test))


def fit_rf_baseline(X_train, y_train, X_test, seed: int) -> np.ndarray:
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(densify(X_train), y_train)
    return clf.predict(densify(X_test))


def fit_gb_baseline(X_train, y_train, X_test, seed: int) -> np.ndarray:
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(densify(X_train), y_train)
    return clf.predict(densify(X_test))


def fit_xgb_baseline(X_train, y_train, X_test, seed: int) -> np.ndarray:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required for the xgboost model.") from exc
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(densify(X_train), y_train)
    return clf.predict(densify(X_test))


def fit_mlp_baseline(X_train, y_train, X_test, seed: int) -> np.ndarray:
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=400,
        early_stopping=True,
        random_state=seed,
    )
    clf.fit(densify(X_train), y_train)
    return clf.predict(densify(X_test))


def fit_baseline(model_name: str, X_train, y_train, X_test, seed: int) -> np.ndarray:
    if model_name == "logistic_regression":
        return fit_lr_baseline(X_train, y_train, X_test, seed)
    if model_name == "random_forest":
        return fit_rf_baseline(X_train, y_train, X_test, seed)
    if model_name == "gradient_boosting":
        return fit_gb_baseline(X_train, y_train, X_test, seed)
    if model_name == "xgboost":
        return fit_xgb_baseline(X_train, y_train, X_test, seed)
    if model_name == "mlp":
        return fit_mlp_baseline(X_train, y_train, X_test, seed)
    raise ValueError(f"Unknown model: {model_name}")
