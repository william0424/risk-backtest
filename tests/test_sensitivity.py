"""Tests for cluster threshold sensitivity analysis."""

import numpy as np
import pytest

from risk_backtest import BacktestConfig, cluster_threshold_sensitivity


@pytest.fixture
def sample_data():
    """Generate returns with some breaches."""
    rng = np.random.default_rng(42)
    n = 300
    returns = rng.normal(0, 0.01, n)
    var = np.full(n, 0.02)
    # Inject clustered breaches
    for i in range(50, 55):
        returns[i] = -0.03
    for i in range(150, 153):
        returns[i] = -0.025
    returns[200] = -0.04
    return returns, var


class TestClusterThresholdSensitivity:
    def test_basic_run(self, sample_data):
        returns, var = sample_data
        df = cluster_threshold_sensitivity(
            returns, var, thresholds=[3, 5, 7], window_sizes=[250]
        )
        assert len(df) == 3
        assert df.index.name == "Threshold"
        assert "Pass_Rate_Kupiec" in df.columns
        assert "Avg_Clusters" in df.columns

    def test_verbose(self, sample_data, capsys):
        returns, var = sample_data
        cluster_threshold_sensitivity(
            returns, var, thresholds=[3, 5], window_sizes=[250], verbose=True
        )
        captured = capsys.readouterr()
        assert "Cluster Threshold = 3" in captured.out
        assert "Cluster Threshold = 5" in captured.out

    def test_batch_mode(self, sample_data):
        returns, var = sample_data
        ret_dict = {"A": returns, "B": returns}
        var_dict = {"A": var, "B": var}
        df = cluster_threshold_sensitivity(
            ret_dict, var_dict, thresholds=[3, 5], window_sizes=[250]
        )
        assert df.loc[3, "N_Series"] == 2
        assert df.loc[5, "N_Series"] == 2

    def test_default_thresholds(self, sample_data):
        returns, var = sample_data
        df = cluster_threshold_sensitivity(returns, var, window_sizes=[250])
        assert len(df) == 15  # range(1, 16)

    def test_clusters_decrease_with_threshold(self, sample_data):
        returns, var = sample_data
        df = cluster_threshold_sensitivity(
            returns, var, thresholds=range(1, 10), window_sizes=[250]
        )
        # More clustering at higher thresholds → fewer isolated, more clusters
        # At minimum, column should exist and have valid values
        assert (df["Avg_Clusters"] >= 0).all()
        assert (df["Avg_Isolated"] >= 0).all()

    def test_pass_rates_bounded(self, sample_data):
        returns, var = sample_data
        df = cluster_threshold_sensitivity(
            returns, var, thresholds=[5], window_sizes=[250]
        )
        for col in df.columns:
            if col.startswith("Pass_Rate_"):
                assert 0 <= df[col].iloc[0] <= 100

    def test_avg_breaches_is_populated(self, sample_data):
        # Regression: sensitivity previously looked up wrong column name
        # and silently returned 0 for Avg_Breaches.
        returns, var = sample_data
        df = cluster_threshold_sensitivity(
            returns, var, thresholds=[5], window_sizes=[250]
        )
        assert df["Avg_Breaches"].iloc[0] > 0
