"""
Sensitivity Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cluster threshold sensitivity analysis for VaR backtesting.
Runs backtesting across a range of thresholds and reports how test results vary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .backtest import BacktestResult, run_backtest
from .config import BacktestConfig


def cluster_threshold_sensitivity(
    returns: dict[str, ArrayLike] | ArrayLike,
    risk_series: dict[str, ArrayLike] | ArrayLike,
    thresholds: list[int] | range | None = None,
    config: BacktestConfig | None = None,
    window_sizes: list[int] | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run backtesting across multiple cluster thresholds and compare results.

    For each threshold value, runs a full backtest and collects summary statistics
    including pass rates, average breach counts, and cluster counts.

    Parameters
    ----------
    returns : dict[str, array-like] or array-like
        Return series (single or batch mode).
    risk_series : dict[str, array-like] or array-like
        Risk measure series matching `returns`.
    thresholds : list[int] or range, optional
        Cluster thresholds to test. Defaults to range(1, 16).
    config : BacktestConfig, optional
        Risk measure configuration. Defaults to daily 99% VaR.
    window_sizes : list[int], optional
        Observation windows. Defaults to [250, 500, 750].
    n_jobs : int, optional (default=1)
        Parallel jobs for each backtest run.
    verbose : bool, optional (default=False)
        If True, print progress and intermediate results per threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by threshold with columns:
        - N_Series: Number of series analyzed
        - Avg_Breaches: Mean breach count across all results
        - Avg_Clusters: Mean cluster count
        - Avg_Isolated: Mean isolated breach count
        - Pass_Rate_Kupiec: % of tests passing Kupiec at 5%
        - Pass_Rate_Christoffersen: % passing Christoffersen at 5%
        - Pass_Rate_Joint: % passing joint test at 5%
        - Pass_Rate_Martingale: % passing martingale test at 5%
        - Total_Results: Total result rows

    Examples
    --------
    >>> from risk_backtest import cluster_threshold_sensitivity
    >>> df = cluster_threshold_sensitivity(returns, var_series, thresholds=range(1, 11))
    >>> df[['Avg_Clusters', 'Pass_Rate_Kupiec']].plot()
    """
    if thresholds is None:
        thresholds = range(1, 16)

    thresholds = list(thresholds)

    rows = []

    for i, ct in enumerate(thresholds):
        if verbose:
            print(f"[{i+1}/{len(thresholds)}] Cluster Threshold = {ct}", end="")

        result = run_backtest(
            returns=returns,
            risk_series=risk_series,
            config=config,
            cluster_threshold=ct,
            window_sizes=window_sizes,
            n_jobs=n_jobs,
        )

        row = _summarize_result(result, ct)
        rows.append(row)

        if verbose:
            print(
                f"  →  clusters={row['Avg_Clusters']:.2f}, "
                f"isolated={row['Avg_Isolated']:.2f}, "
                f"Kupiec pass={row['Pass_Rate_Kupiec']:.1f}%, "
                f"Joint pass={row['Pass_Rate_Joint']:.1f}%"
            )

    df = pd.DataFrame(rows).set_index("Threshold")
    return df


def _summarize_result(result: BacktestResult, threshold: int) -> dict[str, Any]:
    """Extract summary statistics from a BacktestResult."""
    summary = result.summary

    if summary.empty:
        return {
            "Threshold": threshold,
            "N_Series": 0,
            "Total_Results": 0,
            "Avg_Breaches": 0.0,
            "Avg_Clusters": 0.0,
            "Avg_Isolated": 0.0,
            "Pass_Rate_Kupiec": 0.0,
            "Pass_Rate_Christoffersen": 0.0,
            "Pass_Rate_Joint": 0.0,
            "Pass_Rate_Martingale": 0.0,
        }

    # Only use cluster-adjusted rows for cluster/isolated counts
    adj_mask = summary.index.get_level_values("cluster_adj") == "Cluster_Adjusted"
    adj_df = summary[adj_mask]
    unadj_df = summary[~adj_mask]

    n_series = summary.index.get_level_values(0).nunique()

    # Breach counts from unadjusted rows
    avg_breaches = unadj_df["N_Overshoots"].mean() if "N_Overshoots" in unadj_df.columns else 0.0

    # Cluster info
    avg_clusters = unadj_df["N_Clusters"].mean() if "N_Clusters" in unadj_df.columns else 0.0
    avg_isolated = unadj_df["N_Isolated"].mean() if "N_Isolated" in unadj_df.columns else 0.0

    # Pass rates at 5% significance (across ALL rows: both adj and unadj)
    def _pass_rate(col: str) -> float:
        if col not in summary.columns:
            return 0.0
        valid = summary[col].dropna()
        if len(valid) == 0:
            return 0.0
        return (valid >= 0.05).mean() * 100

    return {
        "Threshold": threshold,
        "N_Series": n_series,
        "Total_Results": len(summary),
        "Avg_Breaches": float(avg_breaches),
        "Avg_Clusters": float(avg_clusters),
        "Avg_Isolated": float(avg_isolated),
        "Pass_Rate_Kupiec": _pass_rate("Kupiec_P"),
        "Pass_Rate_Christoffersen": _pass_rate("Christoffersen_P"),
        "Pass_Rate_Joint": _pass_rate("Joint_Test_P"),
        "Pass_Rate_Martingale": _pass_rate("Martingale_Test_P"),
    }
