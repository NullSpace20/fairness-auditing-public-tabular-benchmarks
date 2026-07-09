"""AIF360 mitigations for Q1 isolated pilot (adapted from Code/fairness_experiments.py)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from pipeline_core import ProtectedSpec, densify, fit_baseline, fit_lr_baseline, fit_rf_baseline


def weighted_resample(X, y, weights, random_state: int):
    rng = np.random.default_rng(random_state)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    indices = rng.choice(len(y), size=len(y), replace=True, p=weights)
    return X[indices], y[indices]


def _fit_reweighed(model_name: str, X_tr, y_train, weights, X_te, random_state: int) -> np.ndarray:
    X_tr = densify(X_tr)
    X_te = densify(X_te)
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state)
        clf.fit(X_tr, y_train, sample_weight=weights)
        return clf.predict(X_te)
    if model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        clf.fit(X_tr, y_train, sample_weight=weights)
        return clf.predict(X_te)
    if model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=random_state)
        clf.fit(X_tr, y_train, sample_weight=weights)
        return clf.predict(X_te)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost is required for the xgboost model.") from exc
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_train, sample_weight=weights)
        return clf.predict(X_te)
    if model_name == "mlp":
        X_w, y_w = weighted_resample(X_tr, y_train, weights, random_state)
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=400,
            early_stopping=True,
            random_state=random_state,
        )
        clf.fit(X_w, y_w)
        return clf.predict(X_te)
    raise ValueError(f"Reweighing not supported for model: {model_name}")


def to_aif_dataset(
    X,
    y: np.ndarray,
    protected: np.ndarray,
    spec: ProtectedSpec,
) -> BinaryLabelDataset:
    X_arr = densify(X)
    df = __import__("pandas").DataFrame(X_arr)
    df.columns = [f"f{i}" for i in range(X_arr.shape[1])]
    df["label"] = np.asarray(y).astype(int).ravel()
    df[spec.column] = np.asarray(protected).astype(int).ravel()
    return BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=["label"],
        protected_attribute_names=[spec.column],
    )


def apply_reweighing(
    model_name: str,
    X_train,
    y_train,
    prot_train: np.ndarray,
    X_test,
    spec: ProtectedSpec,
    random_state: int,
) -> np.ndarray:
    privileged_groups = [{spec.column: spec.privileged_value}]
    unprivileged_groups = [{spec.column: spec.unprivileged_value}]
    train_ds = to_aif_dataset(X_train, y_train, prot_train, spec)
    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    rw.fit(train_ds)
    weights = rw.transform(train_ds).instance_weights.ravel()
    return _fit_reweighed(model_name, X_train, y_train, weights, X_test, random_state)


def apply_equalized_odds(
    model_name: str,
    X_train,
    y_train,
    prot_train: np.ndarray,
    X_val,
    y_val,
    prot_val: np.ndarray,
    X_test,
    prot_test: np.ndarray,
    spec: ProtectedSpec,
    random_state: int,
) -> np.ndarray:
    privileged_groups = [{spec.column: spec.privileged_value}]
    unprivileged_groups = [{spec.column: spec.unprivileged_value}]

    if model_name in (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "mlp",
    ):
        val_pred = fit_baseline(model_name, X_train, y_train, X_val, random_state)
        test_pred = fit_baseline(model_name, X_train, y_train, X_test, random_state)
    else:
        raise ValueError(f"Equalized Odds not supported for model: {model_name}")

    val_ds = to_aif_dataset(X_val, y_val, prot_val, spec)
    val_pred_ds = to_aif_dataset(X_val, val_pred, prot_val, spec)
    test_pred_ds = to_aif_dataset(X_test, test_pred, prot_test, spec)

    eq = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        seed=random_state,
    )
    eq.fit(val_ds, val_pred_ds)
    return eq.predict(test_pred_ds).labels.ravel().astype(int)
