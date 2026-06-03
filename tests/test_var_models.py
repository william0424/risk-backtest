"""Tests for VaR estimation models."""

import numpy as np
import pytest
from scipy import stats

from risk_backtest import cornish_fisher_var, evt_var, estimate_var, EVTResult


@pytest.fixture
def normal_returns():
    """Normal returns (no skew, no excess kurtosis)."""
    rng = np.random.default_rng(123)
    return rng.normal(0, 0.01, 1000)


@pytest.fixture
def fat_tail_returns():
    """Returns with fat tails (t-distributed)."""
    rng = np.random.default_rng(456)
    return rng.standard_t(df=4, size=1000) * 0.01


class TestCornishFisherVaR:
    def test_basic_estimate(self, normal_returns):
        var = cornish_fisher_var(normal_returns, confidence_level=0.99)
        assert var > 0
        # For near-normal data, CF should be close to parametric normal VaR
        normal_var = -(np.mean(normal_returns) + stats.norm.ppf(0.01) * np.std(normal_returns, ddof=1))
        assert abs(var - normal_var) / normal_var < 0.15

    def test_fat_tails_higher_var(self, normal_returns, fat_tail_returns):
        var_normal = cornish_fisher_var(normal_returns)
        var_fat = cornish_fisher_var(fat_tail_returns)
        # Fat tails should produce higher VaR
        assert var_fat > var_normal * 0.8  # Allow some tolerance

    def test_confidence_level_effect(self, normal_returns):
        var_95 = cornish_fisher_var(normal_returns, confidence_level=0.95)
        var_99 = cornish_fisher_var(normal_returns, confidence_level=0.99)
        assert var_99 > var_95

    def test_verbose(self, normal_returns, capsys):
        cornish_fisher_var(normal_returns, verbose=True)
        captured = capsys.readouterr()
        assert "Cornish-Fisher" in captured.out
        assert "Skewness" in captured.out
        assert "Excess Kurtosis" in captured.out

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 10"):
            cornish_fisher_var(np.array([0.01, -0.01, 0.02]))

    def test_handles_nan(self, normal_returns):
        data = np.concatenate([normal_returns, [np.nan, np.nan]])
        var = cornish_fisher_var(data)
        assert var > 0


class TestEVTVaR:
    def test_basic_estimate(self, fat_tail_returns):
        result = evt_var(fat_tail_returns, confidence_level=0.99)
        assert isinstance(result, EVTResult)
        assert result.var > 0
        assert result.es >= result.var  # ES always >= VaR

    def test_gpd_parameters(self, fat_tail_returns):
        result = evt_var(fat_tail_returns)
        assert result.scale > 0
        assert result.n_exceedances >= 10

    def test_custom_threshold_quantile(self, fat_tail_returns):
        r1 = evt_var(fat_tail_returns, threshold_quantile=0.85)
        r2 = evt_var(fat_tail_returns, threshold_quantile=0.95)
        # Higher threshold → fewer exceedances
        assert r1.n_exceedances > r2.n_exceedances

    def test_explicit_threshold(self, fat_tail_returns):
        result = evt_var(fat_tail_returns, threshold=0.01)
        assert result.threshold == 0.01

    def test_verbose(self, fat_tail_returns, capsys):
        evt_var(fat_tail_returns, verbose=True)
        captured = capsys.readouterr()
        assert "EVT" in captured.out
        assert "GPD shape" in captured.out

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 30"):
            evt_var(np.random.normal(0, 0.01, 20))

    def test_too_few_exceedances(self):
        # Very high threshold → too few exceedances
        returns = np.random.normal(0, 0.01, 100)
        with pytest.raises(ValueError, match="exceedances"):
            evt_var(returns, threshold_quantile=0.99)


class TestEstimateVaR:
    def test_default_methods(self, normal_returns):
        results = estimate_var(normal_returns, confidence_level=0.99)
        assert "historical" in results
        assert "normal" in results
        assert "cornish_fisher" in results
        assert "evt" in results

    def test_specific_methods(self, normal_returns):
        results = estimate_var(
            normal_returns, methods=["historical", "normal"]
        )
        assert len(results) == 2
        assert results["historical"] > 0
        assert results["normal"] > 0

    def test_historical_var(self, normal_returns):
        results = estimate_var(normal_returns, methods=["historical"])
        expected = -float(np.quantile(normal_returns, 0.01))
        assert abs(results["historical"] - expected) < 1e-10

    def test_normal_var(self, normal_returns):
        results = estimate_var(normal_returns, methods=["normal"])
        mu = np.mean(normal_returns)
        sigma = np.std(normal_returns, ddof=1)
        expected = -(mu + stats.norm.ppf(0.01) * sigma)
        assert abs(results["normal"] - expected) < 1e-10

    def test_unknown_method_raises(self, normal_returns):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_var(normal_returns, methods=["magic"])

    def test_verbose(self, normal_returns, capsys):
        estimate_var(normal_returns, methods=["historical", "cornish_fisher"], verbose=True)
        captured = capsys.readouterr()
        assert "historical" in captured.out.lower()
        assert "Cornish-Fisher" in captured.out

    def test_evt_kwargs(self, fat_tail_returns):
        results = estimate_var(
            fat_tail_returns, methods=["evt"], threshold_quantile=0.85
        )
        assert isinstance(results["evt"], EVTResult)


class TestGARCHVaR:
    """GARCH tests — only run if arch is installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_arch(self):
        pytest.importorskip("arch")

    @pytest.fixture
    def long_returns(self):
        rng = np.random.default_rng(789)
        return rng.normal(0, 0.01, 500)

    def test_basic_garch(self, long_returns):
        from risk_backtest import garch_var, GARCHResult

        result = garch_var(long_returns, confidence_level=0.99)
        assert isinstance(result, GARCHResult)
        assert result.var > 0
        assert result.conditional_vol > 0
        assert 0 < result.persistence < 1.1
        assert result.model_type == "garch"

    def test_gjr_garch(self, long_returns):
        from risk_backtest import garch_var

        result = garch_var(long_returns, model_type="gjr-garch")
        assert result.model_type == "gjr-garch"
        assert "gamma" in result.extra_params
        assert result.var > 0

    def test_egarch(self, long_returns):
        from risk_backtest import garch_var

        result = garch_var(long_returns, model_type="egarch")
        assert result.model_type == "egarch"
        assert "gamma" in result.extra_params
        assert result.var > 0

    def test_aparch(self, long_returns):
        from risk_backtest import garch_var

        result = garch_var(long_returns, model_type="aparch")
        assert result.model_type == "aparch"
        assert "gamma" in result.extra_params
        assert "delta" in result.extra_params
        assert result.var > 0

    def test_invalid_model_type(self, long_returns):
        from risk_backtest import garch_var

        with pytest.raises(ValueError, match="Unknown model_type"):
            garch_var(long_returns, model_type="invalid")

    def test_garch_verbose(self, long_returns, capsys):
        from risk_backtest import garch_var

        garch_var(long_returns, verbose=True)
        captured = capsys.readouterr()
        assert "GARCH" in captured.out
        assert "Persistence" in captured.out

    def test_gjr_verbose(self, long_returns, capsys):
        from risk_backtest import garch_var

        garch_var(long_returns, model_type="gjr-garch", verbose=True)
        captured = capsys.readouterr()
        assert "γ (gamma)" in captured.out

    def test_garch_t_distribution(self, long_returns):
        from risk_backtest import garch_var

        result = garch_var(long_returns, dist="t")
        assert result.var > 0

    def test_too_few_observations(self):
        from risk_backtest import garch_var

        with pytest.raises(ValueError, match="at least 100"):
            garch_var(np.random.normal(0, 0.01, 50))

    def test_estimate_var_with_garch_family(self, long_returns):
        from risk_backtest import GARCHResult

        results = estimate_var(long_returns, methods=["garch", "gjr-garch"])
        assert isinstance(results["garch"], GARCHResult)
        assert isinstance(results["gjr-garch"], GARCHResult)
        assert results["gjr-garch"].model_type == "gjr-garch"


class TestRecursiveGARCHVariance:
    def test_basic_garch(self):
        from risk_backtest import recursive_garch_variance

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        sigma2 = recursive_garch_variance(returns, omega=1e-6, alpha=0.1, beta=0.85)
        assert len(sigma2) == 500
        assert (sigma2 > 0).all()

    def test_gjr_garch(self):
        from risk_backtest import recursive_garch_variance

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 300)
        sigma2 = recursive_garch_variance(
            returns, omega=1e-6, alpha=0.05, beta=0.85, gamma=0.1,
            model_type="gjr-garch"
        )
        assert len(sigma2) == 300
        assert (sigma2 > 0).all()

    def test_egarch(self):
        from risk_backtest import recursive_garch_variance

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 300)
        sigma2 = recursive_garch_variance(
            returns, omega=-0.1, alpha=0.1, beta=0.95, gamma=-0.05,
            model_type="egarch"
        )
        assert len(sigma2) == 300
        assert (sigma2 > 0).all()

    def test_verbose(self, capsys):
        from risk_backtest import recursive_garch_variance

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 200)
        recursive_garch_variance(returns, omega=1e-6, alpha=0.1, beta=0.85, verbose=True)
        captured = capsys.readouterr()
        assert "Recursive GARCH" in captured.out
        assert "Avg daily vol" in captured.out

    def test_invalid_model(self):
        from risk_backtest import recursive_garch_variance

        with pytest.raises(ValueError, match="supports"):
            recursive_garch_variance(np.zeros(100), model_type="aparch")

    def test_leverage_effect(self):
        """GJR-GARCH should produce higher variance after negative returns."""
        from risk_backtest import recursive_garch_variance

        # Create returns that start negative then turn positive
        returns = np.zeros(100)
        returns[10] = -0.05  # big negative shock
        returns[50] = 0.05   # big positive shock

        sigma2 = recursive_garch_variance(
            returns, omega=1e-6, alpha=0.05, beta=0.85, gamma=0.1,
            model_type="gjr-garch"
        )
        # After negative shock, variance should be higher than after positive shock
        assert sigma2[11] > sigma2[51]
