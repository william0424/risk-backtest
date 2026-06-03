"""Tests for Expected Shortfall estimators."""

import numpy as np
import pytest

from risk_backtest import es_from_var_series, historical_es, normal_es


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.01, 1000)


class TestHistoricalES:
    def test_positive(self, returns):
        es = historical_es(returns, confidence_level=0.975)
        assert es > 0

    def test_es_greater_than_var(self, returns):
        from risk_backtest import historical_var

        var = historical_var(returns, confidence_level=0.975)
        es = historical_es(returns, confidence_level=0.975)
        assert es >= var

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            historical_es(np.zeros(5))

    def test_invalid_cl_raises(self, returns):
        with pytest.raises(ValueError, match="confidence_level"):
            historical_es(returns, confidence_level=1.5)


class TestNormalES:
    def test_positive(self, returns):
        assert normal_es(returns, confidence_level=0.975) > 0

    def test_closed_form_matches_simulation(self):
        rng = np.random.default_rng(0)
        big_sample = rng.normal(0, 1, 200_000)
        analytical = normal_es(big_sample, confidence_level=0.975)
        empirical = historical_es(big_sample, confidence_level=0.975)
        assert abs(analytical - empirical) / empirical < 0.05

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            normal_es(np.zeros(5))


class TestESFromVarSeries:
    def test_no_breaches_returns_zero(self):
        r = np.full(100, 0.01)
        v = np.full(100, 0.02)
        assert es_from_var_series(r, v) == 0.0

    def test_average_of_breaches(self):
        r = np.array([0.01, -0.05, 0.0, -0.10])
        v = np.array([0.02, 0.02, 0.02, 0.02])
        # Breaches at indices 1 (-0.05) and 3 (-0.10), mean magnitude = 0.075
        assert es_from_var_series(r, v) == pytest.approx(0.075)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            es_from_var_series(np.zeros(10), np.zeros(5))
