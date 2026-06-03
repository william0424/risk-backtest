"""Tests for the high-level run_backtest API."""

import numpy as np
import pytest

from risk_backtest import BacktestConfig, BacktestResult, run_backtest


@pytest.fixture
def sample_data():
    """Generate sample returns and VaR series."""
    rng = np.random.default_rng(42)
    n = 300
    returns = rng.normal(0, 0.01, n)
    var = np.full(n, 0.02)  # 2% daily VaR
    # Inject some breaches
    returns[10] = -0.03
    returns[50] = -0.025
    returns[100] = -0.04
    returns[150] = -0.03
    returns[200] = -0.05
    return returns, var


class TestRunBacktestSingle:
    def test_basic_run(self, sample_data):
        returns, var = sample_data
        result = run_backtest(returns, var, window_sizes=[250])
        assert isinstance(result, BacktestResult)
        assert len(result.summary) > 0

    def test_default_config(self, sample_data):
        returns, var = sample_data
        result = run_backtest(returns, var, window_sizes=[250])
        assert result.config.confidence_level == 0.99

    def test_custom_config(self, sample_data):
        returns, var = sample_data
        config = BacktestConfig(confidence_level=0.95)
        result = run_backtest(returns, var, config=config, window_sizes=[250])
        assert result.config.daily_breach_probability == pytest.approx(0.05)

    def test_multiple_windows(self, sample_data):
        returns, var = sample_data
        result = run_backtest(returns, var, window_sizes=[200, 250])
        # Should have results for both windows (unadjusted + adjusted = 4 rows)
        assert len(result.summary) == 4

    def test_window_too_large_excluded(self, sample_data):
        returns, var = sample_data
        result = run_backtest(returns, var, window_sizes=[500])
        # 300 data points < 500 window, so no results
        assert len(result.summary) == 0

    def test_result_has_test_columns(self, sample_data):
        returns, var = sample_data
        result = run_backtest(returns, var, window_sizes=[250])
        cols = result.summary.columns.tolist()
        assert "Kupiec_P" in cols
        assert "Christoffersen_P" in cols
        assert "Martingale_Test_P" in cols
        assert "N_Clusters" in cols

    def test_cluster_adj_both_present(self, sample_data):
        returns, var = sample_data
        result = run_backtest(returns, var, window_sizes=[250])
        adj_values = result.summary.index.get_level_values("cluster_adj").unique()
        assert "Unadjusted" in adj_values
        assert "Cluster_Adjusted" in adj_values


class TestRunBacktestBatch:
    def test_batch_mode(self, sample_data):
        returns, var = sample_data
        ret_dict = {"A": returns, "B": returns * 1.1}
        var_dict = {"A": var, "B": var}
        result = run_backtest(ret_dict, var_dict, window_sizes=[250])
        names = result.summary.index.get_level_values("name").unique()
        assert "A" in names
        assert "B" in names

    def test_batch_missing_key_raises(self, sample_data):
        returns, var = sample_data
        with pytest.raises(ValueError, match="missing keys"):
            run_backtest(
                returns={"A": returns, "B": returns},
                risk_series={"A": var},
                window_sizes=[250],
            )

    def test_batch_type_mismatch_raises(self, sample_data):
        returns, var = sample_data
        with pytest.raises(TypeError):
            run_backtest(
                returns={"A": returns},
                risk_series=var,
                window_sizes=[250],
            )


class TestRunBacktestParallel:
    def test_parallel_matches_sequential(self, sample_data):
        returns, var = sample_data
        ret_dict = {"A": returns, "B": returns, "C": returns}
        var_dict = {"A": var, "B": var, "C": var}

        seq = run_backtest(ret_dict, var_dict, window_sizes=[250], n_jobs=1)
        # n_jobs=2 uses threading/multiprocessing
        par = run_backtest(ret_dict, var_dict, window_sizes=[250], n_jobs=2)

        # Results should be identical
        assert len(seq.summary) == len(par.summary)
        for col in seq.summary.columns:
            if seq.summary[col].dtype in [float, np.float64]:
                np.testing.assert_array_almost_equal(
                    seq.summary[col].values,
                    par.summary[col].values,
                    decimal=10,
                )


class TestPassRates:
    def test_pass_rates_property(self, sample_data):
        returns, var = sample_data
        result = run_backtest(returns, var, window_sizes=[250])
        pr = result.pass_rates
        assert len(pr) > 0
        # Values should be percentages between 0 and 100
        assert (pr >= 0).all().all()
        assert (pr <= 100).all().all()


class TestInputValidation:
    def test_empty_returns_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_backtest(np.array([]), np.array([]), window_sizes=[250])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            run_backtest(np.zeros(100), np.zeros(50), window_sizes=[50])
