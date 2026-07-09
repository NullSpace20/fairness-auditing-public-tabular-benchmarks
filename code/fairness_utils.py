"""Copied from Code/fairness_utils.py for isolated Q1 smoke tests (unchanged logic)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

CFS_COMPONENTS = ("abs_spd", "abs_eod", "abs_aod", "di_violation")

ACFS_WEIGHT_PRESETS: Dict[str, tuple] = {
    "balanced": (0.25, 0.25, 0.25, 0.25),
    "statistical_parity": (0.50, 0.20, 0.15, 0.15),
    "equal_opportunity": (0.15, 0.50, 0.20, 0.15),
    "regulatory_risk": (0.20, 0.25, 0.25, 0.30),
}


def abs_spd(spd: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.abs(spd)


def abs_eod(eod: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.abs(eod)


def abs_aod(aod: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.abs(aod)


def di_violation(di: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.abs(1.0 - di)


def compute_cfs(
    spd: Union[float, np.ndarray],
    di: Union[float, np.ndarray],
    eod: Union[float, np.ndarray],
    aod: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    parts = np.column_stack(
        [abs_spd(spd), abs_eod(eod), abs_aod(aod), di_violation(di)]
    )
    return np.nanmean(parts, axis=1) if parts.ndim > 1 else float(np.nanmean(parts))


def compute_acfs(
    spd: Union[float, np.ndarray],
    di: Union[float, np.ndarray],
    eod: Union[float, np.ndarray],
    aod: Union[float, np.ndarray],
    weights: tuple = ACFS_WEIGHT_PRESETS["balanced"],
) -> Union[float, np.ndarray]:
    w1, w2, w3, w4 = weights
    parts = np.column_stack(
        [
            w1 * abs_spd(spd),
            w2 * abs_eod(eod),
            w3 * abs_aod(aod),
            w4 * di_violation(di),
        ]
    )
    return np.nansum(parts, axis=1) if parts.ndim > 1 else float(np.nansum(parts))
