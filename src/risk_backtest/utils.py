"""
Utilities Module
~~~~~~~~~~~~~~~~

Helper functions for overshoot computation, windowing, and data preparation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .config import BacktestConfig


def compute_overshoots(
    returns: ArrayLike,
    risk_series: ArrayLike,
    config: BacktestConfig | None = None,
) -> np.ndarray:
    """
    Compute binary overshoot series from returns and a risk measure.

    Parameters
    ----------
    returns : array-like
        Daily return series.
    risk_series : array-like
        Risk measure series (VaR, volatility, etc.).
    config : BacktestConfig, optional
        Configuration for scaling. If None, assumes daily VaR at 99%.

    Returns
    -------
    np.ndarray
        Boolean array where True = breach.
    """
    if config is None:
        config = BacktestConfig()
    return config.compute_overshoots(returns, risk_series)


def create_windows(
    series: ArrayLike, window_sizes: list[int]
) -> dict[int, np.ndarray]:
    """
    Extract trailing windows of specified sizes from a series.

    Parameters
    ----------
    series : array-like
        The full series.
    window_sizes : list[int]
        Desired window sizes (in observations).

    Returns
    -------
    dict[int, np.ndarray]
        Mapping of window_size → windowed array (most recent observations).
        Only includes windows where sufficient data exists.
    """
    arr = np.asarray(series)
    result = {}
    for ws in window_sizes:
        if len(arr) >= ws:
            result[ws] = arr[-ws:]
    return result


def validate_inputs(
    returns: ArrayLike, risk_series: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and align return and risk series.

    Parameters
    ----------
    returns : array-like
        Return series.
    risk_series : array-like
        Risk measure series.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated arrays of equal length.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or are empty.
    """
    ret = np.asarray(returns, dtype=float)
    risk = np.asarray(risk_series, dtype=float)

    if len(ret) == 0:
        raise ValueError("returns series is empty")
    if len(risk) == 0:
        raise ValueError("risk_series is empty")
    if len(ret) != len(risk):
        raise ValueError(
            f"Length mismatch: returns has {len(ret)} elements, "
            f"risk_series has {len(risk)} elements"
        )
    return ret, risk
