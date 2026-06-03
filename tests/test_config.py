"""Tests for BacktestConfig."""

import math

import numpy as np
import pytest

from risk_backtest import BacktestConfig, Horizon, RiskMeasure


class TestBacktestConfig:
    def test_default_config(self):
        config = BacktestConfig()
        assert config.risk_measure == "VaR"
        assert config.confidence_level == 0.99
        assert config.horizon == "daily"
        assert config.daily_breach_probability == pytest.approx(0.01)
        assert config.scaling_factor == 1.0

    def test_annual_volatility(self):
        config = BacktestConfig(
            risk_measure="volatility",
            confidence_level=0.8413,
            horizon="annual",
        )
        assert config.daily_breach_probability == pytest.approx(0.1587, rel=1e-3)
        assert config.scaling_factor == pytest.approx(1 / math.sqrt(252), rel=1e-6)
        assert config.horizon_days == 252

    def test_monthly_var(self):
        config = BacktestConfig(
            confidence_level=0.95,
            horizon="monthly",
        )
        assert config.daily_breach_probability == pytest.approx(0.05)
        assert config.scaling_factor == pytest.approx(1 / math.sqrt(21), rel=1e-6)

    def test_weekly_horizon(self):
        config = BacktestConfig(horizon="weekly")
        assert config.horizon_days == 5
        assert config.scaling_factor == pytest.approx(1 / math.sqrt(5), rel=1e-6)

    def test_no_scaling(self):
        config = BacktestConfig(horizon="annual", scaling_method="none")
        assert config.scaling_factor == 1.0

    def test_custom_days(self):
        config = BacktestConfig(horizon="annual", custom_days=260)
        assert config.horizon_days == 260
        assert config.scaling_factor == pytest.approx(1 / math.sqrt(260), rel=1e-6)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            BacktestConfig(confidence_level=0)
        with pytest.raises(ValueError):
            BacktestConfig(confidence_level=1)
        with pytest.raises(ValueError):
            BacktestConfig(confidence_level=1.5)

    def test_invalid_custom_days(self):
        with pytest.raises(ValueError):
            BacktestConfig(custom_days=0)
        with pytest.raises(ValueError):
            BacktestConfig(custom_days=-1)


class TestScaleRiskSeries:
    def test_daily_no_scaling(self):
        config = BacktestConfig()
        risk = np.array([0.01, 0.02, 0.015])
        scaled = config.scale_risk_series(risk)
        np.testing.assert_array_almost_equal(scaled, risk)

    def test_annual_scaling(self):
        config = BacktestConfig(horizon="annual")
        risk = np.array([0.10])  # 10% annual vol
        scaled = config.scale_risk_series(risk)
        expected = 0.10 / math.sqrt(252)
        assert scaled[0] == pytest.approx(expected, rel=1e-6)


class TestComputeOvershoots:
    def test_basic_breach(self):
        config = BacktestConfig()
        returns = np.array([-0.02, 0.01, -0.005, -0.015])
        var = np.array([0.01, 0.01, 0.01, 0.01])
        overshoots = config.compute_overshoots(returns, var)
        # Breach when return < -VaR
        expected = np.array([True, False, False, True])
        np.testing.assert_array_equal(overshoots, expected)

    def test_annual_vol_scaling(self):
        config = BacktestConfig(
            risk_measure="volatility",
            confidence_level=0.8413,
            horizon="annual",
        )
        annual_vol = np.array([0.15])  # 15% annual
        daily_vol = 0.15 / math.sqrt(252)
        returns = np.array([-daily_vol * 1.1])  # breach
        overshoots = config.compute_overshoots(returns, annual_vol)
        assert overshoots[0] == True  # noqa: E712

    def test_summary(self):
        config = BacktestConfig(
            risk_measure="volatility",
            confidence_level=0.8413,
            horizon="annual",
        )
        s = config.summary()
        assert s["risk_measure"] == "volatility"
        assert s["horizon"] == "annual"
        assert "scaling_factor" in s
        assert "daily_breach_probability" in s
