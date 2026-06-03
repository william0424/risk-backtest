"""
Regulatory Backtesting Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tools used by supervisors (Basel, CSSF, FCA, etc.) when evaluating internal
VaR models.

Currently implemented:

- **Basel Traffic Light**: zone classification based on 1-year (250-day) breach
  counts at 99% VaR. Returns the zone and the supervisory capital multiplier
  add-on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.stats import binom


@dataclass(frozen=True)
class TrafficLightResult:
    """
    Result of a Basel traffic light test.

    Attributes
    ----------
    zone : str
        One of 'green', 'yellow', 'red'.
    breaches : int
        Number of overshoots observed.
    n_obs : int
        Number of observations used (typically 250).
    cumulative_probability : float
        P(X <= breaches) under the binomial null hypothesis.
    multiplier_addon : float
        Supervisory capital multiplier add-on (k - 3.0). Ranges 0.00–1.00.
    multiplier : float
        Full Basel multiplier (3.0 + add-on).
    """

    zone: Literal["green", "yellow", "red"]
    breaches: int
    n_obs: int
    cumulative_probability: float
    multiplier_addon: float
    multiplier: float


# Basel Committee plus-factor schedule (BCBS, "Supervisory framework for the
# use of backtesting in conjunction with the internal models approach to
# market risk capital requirements", 1996) for 250 observations @ 99% VaR.
_BASEL_PLUS_FACTOR_250 = {
    0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00,
    5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85,
}


def basel_traffic_light(
    breaches: int,
    n_obs: int = 250,
    confidence_level: float = 0.99,
) -> TrafficLightResult:
    """
    Classify a backtest result into the Basel traffic-light zones.

    Zones (for the standard 250-day, 99% VaR setup):

    - **Green** (≤4 breaches): model is acceptable; no add-on.
    - **Yellow** (5–9 breaches): increasing supervisory scrutiny and capital
      multiplier add-on (0.40 → 0.85).
    - **Red** (≥10 breaches): model is rejected; full add-on of 1.00.

    For non-standard `n_obs` or `confidence_level`, the boundaries are
    derived from the binomial cumulative distribution by matching the
    one-sided p-value cut-offs implied by the Basel table (~95% and ~99.99%).

    Parameters
    ----------
    breaches : int
        Number of observed VaR overshoots.
    n_obs : int, optional (default=250)
        Number of observations in the backtest window.
    confidence_level : float, optional (default=0.99)
        VaR confidence level.

    Returns
    -------
    TrafficLightResult

    References
    ----------
    BCBS (1996). *Supervisory framework for the use of "backtesting" in
    conjunction with the internal models approach to market risk capital
    requirements*. Basel Committee on Banking Supervision.
    """
    if breaches < 0:
        raise ValueError(f"breaches must be non-negative, got {breaches}")
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    p = 1 - confidence_level
    cum_prob = float(binom.cdf(breaches, n_obs, p))

    if n_obs == 250 and abs(confidence_level - 0.99) < 1e-9:
        # Use the canonical Basel schedule
        if breaches <= 4:
            zone: Literal["green", "yellow", "red"] = "green"
            addon = 0.00
        elif breaches <= 9:
            zone = "yellow"
            addon = _BASEL_PLUS_FACTOR_250[breaches]
        else:
            zone = "red"
            addon = 1.00
    else:
        # Generic derivation: green if cum_prob < 95%, red if > 99.99%
        if cum_prob < 0.95:
            zone = "green"
            addon = 0.00
        elif cum_prob < 0.9999:
            zone = "yellow"
            # Linearly interpolate add-on between 0.40 and 0.85
            span = (cum_prob - 0.95) / (0.9999 - 0.95)
            addon = round(0.40 + 0.45 * span, 2)
        else:
            zone = "red"
            addon = 1.00

    return TrafficLightResult(
        zone=zone,
        breaches=breaches,
        n_obs=n_obs,
        cumulative_probability=cum_prob,
        multiplier_addon=addon,
        multiplier=3.0 + addon,
    )
