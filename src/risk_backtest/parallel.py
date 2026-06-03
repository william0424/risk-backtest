"""
Parallelization Module
~~~~~~~~~~~~~~~~~~~~~~

Provides utilities for running backtesting tasks in parallel across
multiple series, windows, or configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class BacktestTask:
    """
    A single backtesting unit of work.

    Parameters
    ----------
    name : str
        Identifier for this task (e.g. fund code).
    returns : np.ndarray
        Return series.
    risk_series : np.ndarray
        Risk measure series.
    window_size : int
        Number of observations to use.
    """

    name: str
    returns: np.ndarray
    risk_series: np.ndarray
    window_size: int


def _try_import_joblib():
    """Try to import joblib, return None if not available."""
    try:
        import joblib
        return joblib
    except ImportError:
        return None


def run_parallel(
    func: Callable[[BacktestTask], Any],
    tasks: list[BacktestTask],
    n_jobs: int = 1,
) -> list[Any]:
    """
    Execute backtesting tasks in parallel.

    Parameters
    ----------
    func : callable
        Function that takes a BacktestTask and returns a result.
    tasks : list[BacktestTask]
        List of tasks to execute.
    n_jobs : int, optional (default=1)
        Number of parallel jobs:
        - 1: sequential (no parallelism)
        - -1: use all available cores
        - n > 1: use n cores

    Returns
    -------
    list
        Results from each task, in the same order as input.

    Notes
    -----
    Uses joblib if installed and n_jobs != 1, otherwise falls back to
    concurrent.futures.ProcessPoolExecutor, or sequential execution.
    """
    if n_jobs == 1 or len(tasks) <= 1:
        return [func(task) for task in tasks]

    joblib = _try_import_joblib()
    if joblib is not None:
        return joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(func)(task) for task in tasks
        )

    # Fallback to concurrent.futures
    import concurrent.futures
    import os

    max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, tasks))
    return results


def build_tasks(
    returns: dict[str, ArrayLike] | ArrayLike,
    risk_series: dict[str, ArrayLike] | ArrayLike,
    window_sizes: list[int],
) -> list[BacktestTask]:
    """
    Build a list of BacktestTasks from inputs.

    Supports:
    - Single series: one task per window size
    - Dict of series: tasks for each (name, window_size) combination

    Parameters
    ----------
    returns : dict or array-like
        Single return series or dict mapping names to series.
    risk_series : dict or array-like
        Single risk series or dict mapping names to series.
    window_sizes : list[int]
        Window sizes to test.

    Returns
    -------
    list[BacktestTask]
    """
    tasks = []

    if isinstance(returns, dict):
        # Batch mode: multiple named series
        names = list(returns.keys())
        for name in names:
            ret = np.asarray(returns[name], dtype=float)
            risk = np.asarray(risk_series[name], dtype=float)
            for ws in window_sizes:
                if len(ret) >= ws:
                    tasks.append(BacktestTask(
                        name=name,
                        returns=ret,
                        risk_series=risk,
                        window_size=ws,
                    ))
    else:
        # Single series mode
        ret = np.asarray(returns, dtype=float)
        risk = np.asarray(risk_series, dtype=float)
        for ws in window_sizes:
            if len(ret) >= ws:
                tasks.append(BacktestTask(
                    name="default",
                    returns=ret,
                    risk_series=risk,
                    window_size=ws,
                ))

    return tasks
