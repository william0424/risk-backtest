"""
Backtest Module
~~~~~~~~~~~~~~~

High-level API for running complete backtesting analyses with support for
parallelization, multiple window sizes, and flexible risk measure configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .cluster import count_clusters, detect_cluster
from .config import BacktestConfig
from .parallel import BacktestTask, build_tasks, run_parallel
from .statistical_tests import VaRBacktest
from .utils import validate_inputs


@dataclass
class BacktestResult:
    """
    Container for backtesting results.

    Attributes
    ----------
    summary : pd.DataFrame
        DataFrame with test results indexed by (name, window_size, cluster_adj).
    config : BacktestConfig
        Configuration used for the backtest.
    """

    summary: pd.DataFrame
    config: BacktestConfig

    def __repr__(self) -> str:
        n_series = self.summary.index.get_level_values(0).nunique()
        n_rows = len(self.summary)
        return (
            f"BacktestResult(series={n_series}, rows={n_rows}, "
            f"config={self.config.risk_measure}@{self.config.confidence_level})"
        )

    @property
    def pass_rates(self) -> pd.DataFrame:
        """Compute pass rates across all tests at 5% significance."""
        pval_cols = [c for c in self.summary.columns if c.endswith("_P")]
        pass_df = self.summary[pval_cols] >= 0.05
        return pass_df.groupby(level=["window_size", "cluster_adj"]).mean() * 100


def _execute_task(
    task: BacktestTask,
    config: BacktestConfig,
    cluster_threshold: int,
) -> list[dict[str, Any]]:
    """
    Execute a single backtesting task (one series × one window).

    Returns both unadjusted and cluster-adjusted results.
    """
    # Compute overshoots
    overshoots = config.compute_overshoots(task.returns, task.risk_series)

    # Window the overshoot series
    os_window = overshoots[-task.window_size :]
    T = len(os_window)

    # Initialize test engine
    var_test = VaRBacktest(P=config.daily_breach_probability)

    results = []

    # --- Unadjusted tests ---
    try:
        unadj = var_test.run_tests(os_window, T=T)
    except ValueError:
        return results  # Skip if insufficient data

    # Cluster detection
    cluster_starts, isolated, real_cluster, cluster_adj = detect_cluster(
        os_window, threshold=cluster_threshold
    )
    n_clusters, n_isolated = count_clusters(cluster_starts, isolated, window_size=T)

    results.append(
        {
            "name": task.name,
            "window_size": task.window_size,
            "cluster_adj": "Unadjusted",
            "N_Clusters": n_clusters,
            "N_Isolated": n_isolated,
            **unadj,
        }
    )

    # --- Cluster-adjusted tests ---
    try:
        adj = var_test.run_tests(cluster_adj, T=T)
    except ValueError:
        pass
    else:
        # Re-detect clusters on the adjusted series to confirm none remain
        adj_starts, adj_iso, _, _ = detect_cluster(
            cluster_adj, threshold=cluster_threshold
        )
        n_clusters_adj, n_isolated_adj = count_clusters(
            adj_starts, adj_iso, window_size=T
        )
        results.append(
            {
                "name": task.name,
                "window_size": task.window_size,
                "cluster_adj": "Cluster_Adjusted",
                "N_Clusters": n_clusters_adj,
                "N_Isolated": n_isolated_adj,
                **adj,
            }
        )

    return results


def run_backtest(
    returns: dict[str, ArrayLike] | ArrayLike,
    risk_series: dict[str, ArrayLike] | ArrayLike,
    config: BacktestConfig | None = None,
    cluster_threshold: int = 5,
    window_sizes: list[int] | None = None,
    n_jobs: int = 1,
) -> BacktestResult:
    """
    Run a complete backtesting analysis.

    Supports single series or batch mode (dict of named series).
    Parallelizes across (series, window_size) combinations.

    Parameters
    ----------
    returns : dict[str, array-like] or array-like
        Return series. Use a dict for batch mode (keys = series names).
    risk_series : dict[str, array-like] or array-like
        Risk measure series. Must match structure of `returns`.
    config : BacktestConfig, optional
        Risk measure configuration. Defaults to daily 99% VaR.
    cluster_threshold : int, optional (default=5)
        Threshold for cluster detection.
    window_sizes : list[int], optional
        Observation windows to test. Defaults to [250, 500, 750].
    n_jobs : int, optional (default=1)
        Number of parallel jobs (-1 for all cores).

    Returns
    -------
    BacktestResult
        Container with summary DataFrame and configuration.

    Examples
    --------
    >>> from risk_backtest import run_backtest, BacktestConfig
    >>> results = run_backtest(returns_series, var_series)
    >>> results.summary

    >>> # Batch mode with custom config
    >>> config = BacktestConfig(risk_measure="volatility", confidence_level=0.8413, horizon="annual")
    >>> results = run_backtest(
    ...     returns={"A": ret_a, "B": ret_b},
    ...     risk_series={"A": vol_a, "B": vol_b},
    ...     config=config,
    ...     window_sizes=[250, 500],
    ...     n_jobs=-1,
    ... )
    """
    if config is None:
        config = BacktestConfig()
    if window_sizes is None:
        window_sizes = [250, 500, 750]

    # Validate inputs
    if isinstance(returns, dict):
        if not isinstance(risk_series, dict):
            raise TypeError(
                "If returns is a dict, risk_series must also be a dict."
            )
        missing = set(returns.keys()) - set(risk_series.keys())
        if missing:
            raise ValueError(f"risk_series missing keys: {missing}")
        for name in returns:
            validate_inputs(returns[name], risk_series[name])
    else:
        validate_inputs(returns, risk_series)

    # Build task list
    tasks = build_tasks(returns, risk_series, window_sizes)

    if not tasks:
        return BacktestResult(
            summary=pd.DataFrame(),
            config=config,
        )

    # Execute tasks
    def _worker(task: BacktestTask) -> list[dict]:
        return _execute_task(task, config, cluster_threshold)

    if n_jobs == 1:
        all_results = []
        for task in tasks:
            all_results.extend(_worker(task))
    else:
        task_results = run_parallel(_worker, tasks, n_jobs=n_jobs)
        all_results = []
        for res_list in task_results:
            all_results.extend(res_list)

    if not all_results:
        return BacktestResult(summary=pd.DataFrame(), config=config)

    # Build summary DataFrame
    df = pd.DataFrame(all_results)
    df = df.set_index(["name", "window_size", "cluster_adj"])
    df = df.sort_index()

    return BacktestResult(summary=df, config=config)
