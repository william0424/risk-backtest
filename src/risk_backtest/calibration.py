"""
Calibration statistics for VaR model assessment.

Provides rolling bias and Q-statistics to evaluate how well
VaR forecasts are calibrated relative to realised returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_bias_and_q_statistics(
    returns: np.ndarray | pd.Series,
    var_forecasts: np.ndarray | pd.Series,
    *,
    window_length: int = 60,
    min_periods_ratio: float = 0.85,
    confidence_level: float = 0.99,
    convert_var_to_std: bool = True,
    verbose: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate rolling bias and Q-statistics for VaR backtesting.

    These statistics measure the calibration of VaR forecasts:

    - **Bias Statistic**: rolling std(z) where z = r / sigma
      - Bias ≈ 1.0: Well-calibrated model
      - Bias > 1.0: Model under-predicts risk (returns more volatile than forecast)
      - Bias < 1.0: Model over-predicts risk (returns less volatile than forecast)

    - **Q Statistic**: rolling mean(z² − ln(z²))
      - Under correct calibration, E[Q] = 1 + Euler–Mascheroni constant ≈ 1.577
      - Deviations indicate mis-specification of the VaR distribution

    Parameters
    ----------
    returns : array-like
        Daily returns series.
    var_forecasts : array-like
        Daily VaR forecasts (absolute values used). These represent the
        risk forecast for the *next* day, so they are lagged by 1 internally.
    window_length : int, default 60
        Rolling window length in trading days (~3 months at 60).
    min_periods_ratio : float, default 0.85
        Minimum fraction of window_length required for a valid calculation.
    confidence_level : float, default 0.99
        VaR confidence level (e.g. 0.99 for 99% VaR). Used to convert
        VaR to standard deviation when ``convert_var_to_std=True``.
    convert_var_to_std : bool, default True
        If True, divides ``|var_forecasts|`` by the normal Z-score for
        ``confidence_level`` to obtain sigma. Set to False if var_forecasts
        are already in standard-deviation units.
    verbose : bool, default False
        If True, prints diagnostic information during calculation.

    Returns
    -------
    bias_series : pd.Series
        Rolling bias statistic (std of standardised returns).
    q_stat_series : pd.Series
        Rolling Q statistic.

    Examples
    --------
    >>> from risk_backtest import calculate_bias_and_q_statistics
    >>> bias, q = calculate_bias_and_q_statistics(
    ...     returns=fund_returns,
    ...     var_forecasts=var_predictions,
    ...     window_length=60,
    ...     confidence_level=0.99,
    ... )
    """
    returns = pd.Series(np.asarray(returns, dtype=np.float64)).reset_index(drop=True)
    var_forecasts = pd.Series(np.asarray(var_forecasts, dtype=np.float64)).reset_index(drop=True)

    min_periods = int(min_periods_ratio * window_length)

    if verbose:
        print(f"  Bias/Q-stat: {len(returns)} obs, window={window_length}, "
              f"min_periods={min_periods}, confidence={confidence_level}")

    # Lag VaR forecasts by 1 day (forecast made at t-1 for day t)
    sigma = var_forecasts.shift(1).abs()

    # Convert VaR to standard deviation if needed
    if convert_var_to_std:
        z_score = norm.ppf(confidence_level)
        sigma = sigma / z_score
        if verbose:
            print(f"  Converting VaR to std using z-score={z_score:.4f}")

    # Standardised returns: z = r / sigma
    z = returns / sigma

    # Rolling bias statistic: std(z)
    bias_series = z.rolling(window=window_length, min_periods=min_periods).std()

    # Q statistic: mean(z² - ln(z²))
    z_squared = z ** 2
    z_squared_safe = z_squared.replace(0, np.nan)
    q_components = z_squared - np.log(z_squared_safe)
    q_stat_series = q_components.rolling(window=window_length, min_periods=min_periods).mean()

    if verbose:
        valid_bias = bias_series.notna().sum()
        valid_q = q_stat_series.notna().sum()
        print(f"  Results: {valid_bias} bias values, {valid_q} Q-stat values")

    return bias_series, q_stat_series


def calculate_bias_q_batch(
    daily_results: pd.DataFrame,
    *,
    return_columns: dict[str, str] | None = None,
    var_column: str = "VAR",
    fund_column: str = "FUNDCODE",
    date_column: str = "VALUATIONDATE",
    window_length: int = 60,
    min_periods_ratio: float = 0.85,
    confidence_level: float = 0.99,
    convert_var_to_std: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calculate bias and Q-statistics for all funds in a daily results DataFrame.

    Adds columns ``BIAS_{type}`` and ``Q_STAT_{type}`` for each return type.

    Parameters
    ----------
    daily_results : pd.DataFrame
        DataFrame with fund-level daily data.
    return_columns : dict, optional
        Mapping of label → column name for return columns.
        Default: ``{"DIRTY": "RETURN_DIRTY", "CLEAN": "RETURN_CLEAN"}``.
    var_column : str, default "VAR"
        Column name for VaR forecasts.
    fund_column : str, default "FUNDCODE"
        Column name for fund identifier.
    date_column : str, default "VALUATIONDATE"
        Column name for date.
    window_length : int, default 60
        Rolling window length in trading days.
    min_periods_ratio : float, default 0.85
        Minimum fraction of window required for valid calculation.
    confidence_level : float, default 0.99
        VaR confidence level for VaR-to-std conversion.
    convert_var_to_std : bool, default True
        Whether to convert VaR to std (divide by z-score).
    verbose : bool, default False
        Print progress information.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added BIAS_* and Q_STAT_* columns.

    Examples
    --------
    >>> from risk_backtest import calculate_bias_q_batch
    >>> daily_results = calculate_bias_q_batch(
    ...     daily_results,
    ...     window_length=60,
    ...     confidence_level=0.99,
    ...     verbose=True,
    ... )
    """
    if return_columns is None:
        return_columns = {"DIRTY": "RETURN_DIRTY", "CLEAN": "RETURN_CLEAN"}

    df = daily_results.copy()
    fund_list = df[fund_column].unique()
    min_obs = int(0.5 * window_length)

    if verbose:
        print(f"Calculating bias and Q-statistics: {len(fund_list)} funds, "
              f"window={window_length}, confidence={confidence_level}")

    for label, ret_col in return_columns.items():
        if ret_col not in df.columns:
            if verbose:
                print(f"  Skipping {label}: column '{ret_col}' not found")
            continue

        bias_col = f"BIAS_{label}"
        q_col = f"Q_STAT_{label}"

        bias_list = []
        q_list = []

        for fund in fund_list:
            fund_mask = df[fund_column] == fund
            fund_data = df.loc[fund_mask].sort_values(date_column)

            if len(fund_data) < min_obs:
                bias_list.extend([np.nan] * len(fund_data))
                q_list.extend([np.nan] * len(fund_data))
                if verbose:
                    print(f"  Skipping {fund} ({label}): insufficient data ({len(fund_data)} < {min_obs})")
                continue

            if fund_data[ret_col].notna().sum() == 0:
                bias_list.extend([np.nan] * len(fund_data))
                q_list.extend([np.nan] * len(fund_data))
                continue

            bias, q = calculate_bias_and_q_statistics(
                fund_data[ret_col].values,
                fund_data[var_column].values,
                window_length=window_length,
                min_periods_ratio=min_periods_ratio,
                confidence_level=confidence_level,
                convert_var_to_std=convert_var_to_std,
                verbose=False,
            )
            bias_list.extend(bias.tolist())
            q_list.extend(q.tolist())

        df[bias_col] = bias_list
        df[q_col] = q_list

        if verbose:
            valid = df[bias_col].notna().sum()
            print(f"  {label}: {valid:,} valid bias/Q-stat values")

    return df
