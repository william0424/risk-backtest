"""Tests for cluster detection module."""

import numpy as np
import pandas as pd
import pytest

from risk_backtest import count_clusters, detect_cluster


class TestDetectCluster:
    def test_no_breaches(self):
        series = [False] * 100
        starts, isolated, real, adj = detect_cluster(series, threshold=5)
        assert starts.sum() == 0
        assert isolated.sum() == 0
        assert real.sum() == 0
        assert adj.sum() == 0

    def test_single_breach_is_isolated(self):
        series = [False] * 50 + [True] + [False] * 49
        starts, isolated, real, adj = detect_cluster(series, threshold=5)
        assert isolated.sum() == 1
        assert starts.sum() == 0
        assert adj.sum() == 1  # isolated counts in adj

    def test_adjacent_breaches_form_cluster(self):
        series = [False] * 10 + [True, True, True] + [False] * 87
        starts, isolated, real, adj = detect_cluster(series, threshold=5)
        assert real.sum() > 0
        assert starts.sum() >= 1

    def test_distant_breaches_remain_isolated(self):
        series = [False] * 100
        series[10] = True
        series[90] = True
        starts, isolated, real, adj = detect_cluster(series, threshold=5)
        assert isolated.sum() == 2
        assert real.sum() == 0

    def test_threshold_1_strict(self):
        # Only immediately adjacent breaches cluster at threshold=1
        series = [False, True, False, True, False]
        starts, isolated, real, adj = detect_cluster(series, threshold=1)
        # At threshold=1, gap of 1 between positions 1 and 3 means
        # left_d + right_d = 2 > 1 for position 2, so they don't cluster
        assert isolated.sum() == 2

    def test_threshold_large_clusters_everything(self):
        series = [False] * 20
        series[5] = True
        series[15] = True
        starts, isolated, real, adj = detect_cluster(series, threshold=20)
        # Large threshold should cluster them
        assert real.sum() > 0

    def test_returns_pandas_series(self):
        series = [False, True, False, True, False]
        starts, isolated, real, adj = detect_cluster(series, threshold=5)
        assert isinstance(starts, pd.Series)
        assert isinstance(isolated, pd.Series)
        assert isinstance(real, pd.Series)
        assert isinstance(adj, pd.Series)

    def test_accepts_numpy_array(self):
        series = np.array([False, True, True, False, True])
        starts, isolated, real, adj = detect_cluster(series, threshold=5)
        assert len(starts) == 5


class TestCountClusters:
    def test_basic_count(self):
        starts = pd.Series([False, True, False, False, True])
        isolated = pd.Series([False, False, False, True, False])
        n_c, n_i = count_clusters(starts, isolated)
        assert n_c == 2
        assert n_i == 1

    def test_with_window(self):
        starts = pd.Series([True, False, False, False, True])
        isolated = pd.Series([False, True, False, False, False])
        n_c, n_i = count_clusters(starts, isolated, window_size=3)
        # Last 3 elements: starts=[False, False, True], isolated=[False, False, False]
        assert n_c == 1
        assert n_i == 0

    def test_empty_series(self):
        starts = pd.Series([], dtype=bool)
        isolated = pd.Series([], dtype=bool)
        n_c, n_i = count_clusters(starts, isolated)
        assert n_c == 0
        assert n_i == 0
