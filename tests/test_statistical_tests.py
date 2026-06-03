"""Tests for the VaRBacktest statistical tests class."""

import math

import numpy as np
import pytest

from risk_backtest import VaRBacktest


@pytest.fixture
def bt():
    return VaRBacktest(P=0.01)


class TestBinomial:
    def test_zero_breaches(self, bt):
        prob = bt.binomial_probability(0, 250)
        assert prob == pytest.approx(0.99**250, rel=1e-6)

    def test_cumulative_one_or_more(self, bt):
        cum = bt.binomial_cumulative_prob(1, 250)
        assert cum == pytest.approx(1 - 0.99**250, rel=1e-6)

    def test_cumulative_zero(self, bt):
        cum = bt.binomial_cumulative_prob(0, 250)
        assert cum == pytest.approx(1.0, rel=1e-10)

    def test_probability_sums_to_one(self, bt):
        T = 50
        total = sum(bt.binomial_probability(n, T) for n in range(T + 1))
        assert total == pytest.approx(1.0, rel=1e-6)


class TestZTest:
    def test_expected_breaches(self, bt):
        # N = P*T = 0.01*250 = 2.5, so N=2 or 3 should be near zero
        stat = bt.z_test_statistic(2, 250)
        assert stat < 0  # fewer than expected

        stat = bt.z_test_statistic(3, 250)
        assert stat > 0  # more than expected

    def test_p_value_range(self, bt):
        p = bt.z_test(5, 250)
        assert 0 <= p <= 1


class TestKupiec:
    def test_zero_breaches(self, bt):
        stat = bt.kupiec_statistic(0, 250)
        expected = -2 * math.log(0.99**250)
        assert stat == pytest.approx(expected, rel=1e-6)

    def test_p_value_range(self, bt):
        p = bt.kupiec_test(3, 250)
        assert 0 <= p <= 1

    def test_many_breaches_rejects(self, bt):
        p = bt.kupiec_test(20, 250)
        assert p < 0.01


class TestChristoffersen:
    def test_independent_breaches(self, bt):
        # No clustering: N4=0
        stat = bt.christoffersen_statistic(240, 5, 5, 0)
        assert stat >= 0

    def test_perfect_clustering(self, bt):
        # All breaches follow breaches: high statistic
        stat = bt.christoffersen_statistic(240, 0, 1, 9)
        assert stat > 0

    def test_p_value_range(self, bt):
        p = bt.christoffersen_test(200, 10, 10, 5)
        assert 0 <= p <= 1


class TestJoint:
    def test_combines_uc_and_ind(self, bt):
        joint = bt.joint_statistic(200, 10, 10, 5)
        uc = bt.kupiec_statistic(15, 225)
        ind = bt.christoffersen_statistic(200, 10, 10, 5)
        assert joint == pytest.approx(uc + ind, rel=1e-6)

    def test_p_value_range(self, bt):
        p = bt.joint_test(200, 10, 10, 5)
        assert 0 <= p <= 1


class TestMartingale:
    def test_no_breaches(self, bt):
        serie = np.zeros(250)
        stat, p = bt.martingale_test(serie)
        assert stat == 0
        assert p == 1.0

    def test_random_breaches(self, bt):
        rng = np.random.default_rng(42)
        serie = rng.random(250) < 0.01
        stat, p = bt.martingale_test(serie)
        assert stat >= 0
        assert 0 <= p <= 1

    def test_autocorrelated_breaches(self, bt):
        # Create highly autocorrelated series
        serie = np.zeros(250)
        serie[50:60] = 1  # Cluster of breaches
        stat, p = bt.martingale_test(serie)
        assert stat > 0


class TestRunTests:
    def test_returns_all_keys(self, bt):
        serie = np.zeros(250)
        serie[10] = 1
        serie[100] = 1
        result = bt.run_tests(serie, T=250)

        expected_keys = [
            "N_Obs", "N_Overshoots",
            "Binomial_Probability", "Binomial_Cumulative_Prob",
            "Z_Test_Statistic", "Z_Test_P",
            "Kupiec_Statistic", "Kupiec_P",
            "Christoffersen_Statistic", "Christoffersen_P",
            "Joint_Test_Statistic", "Joint_Test_P",
            "Martingale_Test_Statistic", "Martingale_Test_P",
        ]
        for key in expected_keys:
            assert key in result

    def test_insufficient_data_raises(self, bt):
        serie = np.zeros(100)
        with pytest.raises(ValueError, match="Insufficient data"):
            bt.run_tests(serie, T=250)

    def test_n_obs_correct(self, bt):
        serie = np.zeros(300)
        serie[250] = 1
        result = bt.run_tests(serie, T=250)
        assert result["N_Obs"] == 250


class TestValidation:
    def test_invalid_P(self):
        with pytest.raises(ValueError):
            VaRBacktest(P=0)
        with pytest.raises(ValueError):
            VaRBacktest(P=1)
        with pytest.raises(ValueError):
            VaRBacktest(P=-0.1)


class TestLegacyAliases:
    def test_legacy_methods_exist(self, bt):
        assert bt.Binomial_probability(2, 250) == bt.binomial_probability(2, 250)
        assert bt.LR_uc(2, 250) == bt.kupiec_statistic(2, 250)
        assert bt.Markov_statistic(200, 10, 10, 5) == bt.christoffersen_statistic(200, 10, 10, 5)
