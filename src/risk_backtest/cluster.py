"""
Cluster Detection Module
~~~~~~~~~~~~~~~~~~~~~~~~

Identifies clusters of breaches in a binary time series based on proximity.
Useful for adjusting backtesting statistics to avoid double-counting
correlated breaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def detect_cluster(
    series: ArrayLike, threshold: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Detect clusters in a boolean/binary series based on proximity threshold.

    A cluster is a group of True values (breaches) where the sum of distances
    to the nearest True values on both sides is less than or equal to `threshold`.

    Parameters
    ----------
    series : array-like
        Binary series (True/False or 1/0) indicating breaches.
    threshold : int
        Maximum sum of left and right distances for points to be considered
        part of the same cluster.

    Returns
    -------
    tuple of pd.Series
        - cluster_starts: marks the first breach in each cluster
        - isolated: marks breaches not part of any cluster
        - real_cluster: marks all points within true clusters (excl. isolated)
        - cluster_adj: union of cluster_starts and isolated (for adjusted tests)

    Examples
    --------
    >>> breaches = [False, True, True, False, False, True, False]
    >>> starts, isolated, real, adj = detect_cluster(breaches, threshold=5)
    """
    series = pd.Series(series).reset_index(drop=True)
    result = pd.Series([False] * len(series), index=series.index)

    true_indices = series[series == True].index.tolist()  # noqa: E712

    if len(true_indices) == 0:
        return result, result.copy(), result.copy(), result.copy()

    for idx in series.index:
        if series[idx]:
            left_d = 0
            right_d = 0
        else:
            left_trues = [i for i in true_indices if i < idx]
            left_d = idx - max(left_trues) if left_trues else float("inf")

            right_trues = [i for i in true_indices if i > idx]
            right_d = min(right_trues) - idx if right_trues else float("inf")

        if left_d + right_d <= threshold:
            result[idx] = True

    # Identify isolated breaches (no adjacent cluster members)
    isolated = pd.Series([False] * len(series), index=series.index)
    for i in range(len(result)):
        if result.iloc[i]:
            left_is_false = i == 0 or not result.iloc[i - 1]
            right_is_false = i == len(result) - 1 or not result.iloc[i + 1]
            if left_is_false and right_is_false:
                isolated.iloc[i] = True

    # Cluster starting points
    cluster_starts = pd.Series([False] * len(series), index=series.index)
    for i in range(len(result)):
        if result.iloc[i] and not isolated.iloc[i]:
            if (i < len(result) - 1 and result.iloc[i + 1]) and (
                i == 0 or not result.iloc[i - 1]
            ):
                cluster_starts.iloc[i] = True

    real_cluster = result & ~isolated
    cluster_adj = cluster_starts | isolated

    return cluster_starts, isolated, real_cluster, cluster_adj


def count_clusters(
    cluster_starts: ArrayLike,
    isolated: ArrayLike,
    window_size: int | None = None,
) -> tuple[int, int]:
    """
    Count clusters and isolated breaches, optionally within a trailing window.

    Parameters
    ----------
    cluster_starts : array-like
        Boolean series marking cluster starts.
    isolated : array-like
        Boolean series marking isolated breaches.
    window_size : int, optional
        If provided, count only within the last `window_size` observations.

    Returns
    -------
    tuple[int, int]
        (n_clusters, n_isolated)
    """
    if window_size is not None:
        cluster_starts = cluster_starts[-window_size:]
        isolated = isolated[-window_size:]

    return int(np.sum(cluster_starts)), int(np.sum(isolated))
