"""
Configuration Module
~~~~~~~~~~~~~~~~~~~~

Provides BacktestConfig for normalizing different risk measures
(VaR, volatility, ES) at various horizons into a daily breach probability
suitable for backtesting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm


class RiskMeasure(str, Enum):
    """Supported risk measure types."""

    VAR = "VaR"
    VOLATILITY = "volatility"
    ES = "ES"


class Horizon(str, Enum):
    """Time horizon of the provided risk measure."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUAL = "annual"


# Trading days per horizon (for sqrt-T scaling)
_HORIZON_DAYS: dict[Horizon, int] = {
    Horizon.DAILY: 1,
    Horizon.WEEKLY: 5,
    Horizon.MONTHLY: 21,
    Horizon.ANNUAL: 252,
}


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration for backtesting a risk measure.

    Normalizes any risk measure into a daily breach probability and
    provides scaling logic to convert the risk series to daily units.

    Parameters
    ----------
    risk_measure : str
        Type of risk measure: "VaR", "volatility", or "ES".
    confidence_level : float
        Confidence level expressed as a probability in (0, 1).
        For VaR: 0.99 means 99% VaR (1% breach probability).
        For volatility: 0.8413 means 1-sigma one-sided (~15.87% breach prob).
    horizon : str
        Time horizon of the provided series: "daily", "weekly", "monthly", "annual".
    scaling_method : str
        Method for time-scaling: "sqrt_t" (default) or "none".
    custom_days : int | None
        Override trading days for the horizon (e.g. 260 for some markets).

    Examples
    --------
    >>> # Daily 99% VaR (default)
    >>> config = BacktestConfig()

    >>> # Annual 1-sigma volatility
    >>> config = BacktestConfig(
    ...     risk_measure="volatility",
    ...     confidence_level=0.8413,
    ...     horizon="annual",
    ... )

    >>> # Monthly 95% VaR
    >>> config = BacktestConfig(confidence_level=0.95, horizon="monthly")
    """

    risk_measure: Literal["VaR", "volatility", "ES"] = "VaR"
    confidence_level: float = 0.99
    horizon: Literal["daily", "weekly", "monthly", "annual"] = "daily"
    scaling_method: Literal["sqrt_t", "none"] = "sqrt_t"
    custom_days: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.custom_days is not None and self.custom_days <= 0:
            raise ValueError(f"custom_days must be positive, got {self.custom_days}")

    @property
    def daily_breach_probability(self) -> float:
        """
        Expected daily breach probability derived from the risk measure config.

        For VaR at confidence c: P = 1 - c
        For volatility at confidence c: P = 1 - c (interpreted as one-sided quantile)
        For ES: same as VaR (breach is still defined as exceeding the quantile)
        """
        return 1.0 - self.confidence_level

    @property
    def horizon_days(self) -> int:
        """Number of trading days corresponding to the configured horizon."""
        if self.custom_days is not None:
            return self.custom_days
        return _HORIZON_DAYS[Horizon(self.horizon)]

    @property
    def scaling_factor(self) -> float:
        """
        Factor to scale the risk series from its native horizon to daily.

        For sqrt_t scaling: divide by sqrt(horizon_days).
        For 'none': no scaling (factor = 1).
        """
        if self.scaling_method == "none" or self.horizon == "daily":
            return 1.0
        return 1.0 / math.sqrt(self.horizon_days)

    def scale_risk_series(self, risk_series: ArrayLike) -> np.ndarray:
        """
        Scale the risk measure series from native horizon to daily.

        Parameters
        ----------
        risk_series : array-like
            The risk measure values in their native horizon.

        Returns
        -------
        np.ndarray
            Scaled daily risk values.
        """
        return np.asarray(risk_series, dtype=float) * self.scaling_factor

    def compute_overshoots(
        self, returns: ArrayLike, risk_series: ArrayLike
    ) -> np.ndarray:
        """
        Compute binary overshoot (breach) series.

        A breach occurs when the return falls below the negative of the
        (scaled) risk measure.

        Parameters
        ----------
        returns : array-like
            Daily return series.
        risk_series : array-like
            Risk measure series (in native horizon, will be scaled).

        Returns
        -------
        np.ndarray
            Boolean array where True indicates a breach.
        """
        returns_arr = np.asarray(returns, dtype=float)
        scaled_risk = self.scale_risk_series(risk_series)
        return returns_arr < -scaled_risk

    def summary(self) -> dict[str, object]:
        """Return a summary dict of the configuration."""
        return {
            "risk_measure": self.risk_measure,
            "confidence_level": self.confidence_level,
            "horizon": self.horizon,
            "scaling_method": self.scaling_method,
            "horizon_days": self.horizon_days,
            "scaling_factor": self.scaling_factor,
            "daily_breach_probability": self.daily_breach_probability,
        }
