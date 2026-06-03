"""
risk-backtest
~~~~~~~~~~~~~

A data-agnostic backtesting toolbox for VaR and other risk measures.

Basic usage:

    >>> from risk_backtest import run_backtest, BacktestConfig
    >>> results = run_backtest(returns, var_series)
    >>> results.summary

Advanced (annual 1-sigma volatility, batch, parallel):

    >>> config = BacktestConfig(
    ...     risk_measure="volatility",
    ...     confidence_level=0.8413,
    ...     horizon="annual",
    ... )
    >>> results = run_backtest(
    ...     returns={"A": ret_a, "B": ret_b},
    ...     risk_series={"A": vol_a, "B": vol_b},
    ...     config=config,
    ...     n_jobs=-1,
    ... )
"""

from .backtest import BacktestResult, run_backtest
from .calibration import calculate_bias_and_q_statistics, calculate_bias_q_batch
from .cluster import count_clusters, detect_cluster
from .config import BacktestConfig, Horizon, RiskMeasure
from .expected_shortfall import es_from_var_series, historical_es, normal_es
from .plotting import plot_var_vs_returns
from .regulatory import TrafficLightResult, basel_traffic_light
from .sensitivity import cluster_threshold_sensitivity
from .statistical_tests import VaRBacktest
from .utils import compute_overshoots, create_windows, validate_inputs
from .var_models import (
    EVTResult,
    GARCH_MODELS,
    GARCHResult,
    VAR_METHODS,
    cornish_fisher_var,
    estimate_var,
    evt_var,
    garch_var,
    historical_var,
    normal_var,
    recursive_garch_variance,
)

__version__ = "0.1.0"

__all__ = [
    # High-level API
    "run_backtest",
    "BacktestResult",
    # Configuration
    "BacktestConfig",
    "RiskMeasure",
    "Horizon",
    # Statistical tests
    "VaRBacktest",
    # Calibration statistics
    "calculate_bias_and_q_statistics",
    "calculate_bias_q_batch",
    # Cluster detection
    "detect_cluster",
    "count_clusters",
    # Sensitivity analysis
    "cluster_threshold_sensitivity",
    # Expected Shortfall
    "historical_es",
    "normal_es",
    "es_from_var_series",
    # Regulatory helpers
    "basel_traffic_light",
    "TrafficLightResult",
    # VaR estimation models
    "historical_var",
    "normal_var",
    "cornish_fisher_var",
    "evt_var",
    "garch_var",
    "estimate_var",
    "recursive_garch_variance",
    "EVTResult",
    "GARCHResult",
    "GARCH_MODELS",
    "VAR_METHODS",
    # Utilities
    "compute_overshoots",
    "create_windows",
    "validate_inputs",
    # Plotting
    "plot_var_vs_returns",
]
