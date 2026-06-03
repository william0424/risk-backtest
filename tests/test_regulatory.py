"""Tests for Basel traffic-light regulatory backtesting."""

import pytest

from risk_backtest import TrafficLightResult, basel_traffic_light


class TestBaselTrafficLight:
    @pytest.mark.parametrize("breaches", [0, 1, 2, 3, 4])
    def test_green_zone(self, breaches):
        res = basel_traffic_light(breaches)
        assert res.zone == "green"
        assert res.multiplier_addon == 0.0
        assert res.multiplier == 3.0

    @pytest.mark.parametrize(
        "breaches,addon",
        [(5, 0.40), (6, 0.50), (7, 0.65), (8, 0.75), (9, 0.85)],
    )
    def test_yellow_zone_schedule(self, breaches, addon):
        res = basel_traffic_light(breaches)
        assert res.zone == "yellow"
        assert res.multiplier_addon == addon
        assert res.multiplier == pytest.approx(3.0 + addon)

    @pytest.mark.parametrize("breaches", [10, 12, 25])
    def test_red_zone(self, breaches):
        res = basel_traffic_light(breaches)
        assert res.zone == "red"
        assert res.multiplier_addon == 1.0

    def test_returns_dataclass(self):
        res = basel_traffic_light(5)
        assert isinstance(res, TrafficLightResult)
        assert res.breaches == 5
        assert res.n_obs == 250
        assert 0 < res.cumulative_probability < 1

    def test_generic_nonstandard_window(self):
        # Should still classify and return a zone
        res = basel_traffic_light(2, n_obs=100, confidence_level=0.99)
        assert res.zone in {"green", "yellow", "red"}

    def test_invalid_breaches(self):
        with pytest.raises(ValueError, match="non-negative"):
            basel_traffic_light(-1)

    def test_invalid_n_obs(self):
        with pytest.raises(ValueError, match="positive"):
            basel_traffic_light(0, n_obs=0)

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="confidence_level"):
            basel_traffic_light(0, confidence_level=1.5)
