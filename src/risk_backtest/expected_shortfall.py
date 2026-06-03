"""
Expected Shortfall (ES / CVaR) Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standalone Expected Shortfall (a.k.a. Conditional VaR) estimators. ES is the
expected loss given that the VaR threshold has been breached; under FRTB it
replaces VaR as the regulatory tail risk measure.

All estimators return a positive number representing the magnitude of the
expected tail loss.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def historical_es(
    returns: ArrayLike,
    confidence_level: float = 0.975,
) -> float:
    """
    Compute Expected Shortfall using historical simulation.

    ES is the average loss conditional on the return being worse than the
    empirical VaR quantile.

    Parameters
    ----------
    returns : array-like
        Historical return series.
    confidence_level : float, optional (default=0.975)
        Confidence level. FRTB uses 97.5%; Basel III uses 99%.

    Returns
    -------
    float
        Expected Shortfall (positive magnitude).
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        raise ValueError("Historical ES requires at least 10 observations.")
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    var_threshold = float(np.quantile(arr, 1 - confidence_level))
    tail = arr[arr <= var_threshold]
    if len(tail) == 0:
        return -float(var_threshold)
    return -float(tail.mean())


def normal_es(
    returns: ArrayLike,
    confidence_level: float = 0.975,
) -> float:
    """
    Compute Expected Shortfall assuming returns are Gaussian.

    Closed-form formula: ES = -(μ - σ · φ(z) / (1 - C))
    where z = Φ⁻¹(1 - C), φ is the standard normal PDF.

    Parameters
    ----------
    returns : array-like
        Historical return series.
    confidence_level : float, optional (default=0.975)
        Confidence level.

    Returns
    -------
    float
        Expected Shortfall (positive magnitude).
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        raise ValueError("Normal ES requires at least 10 observations.")
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    alpha = 1 - confidence_level
    z = stats.norm.ppf(alpha)
    es = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return float(es)


def es_from_var_series(
    returns: ArrayLike,
    var_series: ArrayLike,
) -> float:
    """
    Estimate Expected Shortfall as the average breach magnitude from a
    realised return series and a (time-varying) VaR forecast series.

    Useful for post-hoc ES backtesting given a model's daily VaR.

    Parameters
    ----------
    returns : array-like
        Realised returns (decimal).
    var_series : array-like
        Forecast VaR (positive loss magnitudes).

    Returns
    -------
    float
        Mean loss magnitude on breach days. Returns 0.0 if no breaches occurred.
    """
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(var_series, dtype=np.float64)
    if r.shape != v.shape:
        raise ValueError(
            f"returns and var_series shape mismatch: {r.shape} vs {v.shape}"
        )
    mask = r < -v
    if not mask.any():
        return 0.0
    return -float(r[mask].mean())
