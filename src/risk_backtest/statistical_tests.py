"""
Statistical Tests Module
~~~~~~~~~~~~~~~~~~~~~~~~

Provides the VaRBacktest class implementing comprehensive statistical tests
for evaluating risk model accuracy.

Tests included:
- Binomial test (exact probability)
- Z-test (proportion test)
- Kupiec test (Likelihood Ratio Unconditional Coverage)
- Christoffersen test (Independence / Markov)
- Joint test (UC + Independence)
- Martingale test (autocorrelation in breach sequence)
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2, norm


class VaRBacktest:
    """
    A comprehensive backtesting framework for VaR and other risk measures.

    Implements various statistical tests to evaluate the accuracy of risk
    predictions. All methods accept simple numeric inputs (breach counts,
    observation counts) or binary arrays, making the class fully data-agnostic.

    Parameters
    ----------
    P : float, optional (default=0.01)
        The expected breach probability (e.g., 0.01 for 99% VaR).

    Examples
    --------
    >>> bt = VaRBacktest(P=0.01)
    >>> results = bt.run_tests(overshoot_series, T=250)
    >>> print(results['Kupiec_P'])
    """

    def __init__(self, P: float = 0.01) -> None:
        if not 0 < P < 1:
            raise ValueError(f"P must be in (0, 1), got {P}")
        self.P = P
        # Caches for repeated computations (thread-safe for read-heavy workloads)
        self._binom_prob_cache: dict[tuple, float] = {}
        self._binom_cum_cache: dict[tuple, float] = {}
        self._z_stat_cache: dict[tuple, float] = {}
        self._lr_uc_cache: dict[tuple, float] = {}
        self._markov_cache: dict[tuple, float] = {}

    def binomial_probability(self, N: int, T: int) -> float:
        """
        Calculate exact binomial probability P(X = N).

        Parameters
        ----------
        N : int
            Number of breaches observed.
        T : int
            Total number of observations.

        Returns
        -------
        float
        """
        key = (N, T, self.P)
        if key not in self._binom_prob_cache:
            self._binom_prob_cache[key] = (
                math.comb(T, N) * self.P**N * (1 - self.P) ** (T - N)
            )
        return self._binom_prob_cache[key]

    def binomial_cumulative_prob(self, N: int, T: int) -> float:
        """
        Calculate cumulative binomial probability P(X >= N).

        Parameters
        ----------
        N : int
            Number of breaches observed.
        T : int
            Total number of observations.

        Returns
        -------
        float
        """
        key = (N, T, self.P)
        if key not in self._binom_cum_cache:
            self._binom_cum_cache[key] = 1 - sum(
                self.binomial_probability(x, T) for x in range(N)
            )
        return self._binom_cum_cache[key]

    def z_test_statistic(self, N: int, T: int) -> float:
        """
        Calculate Z-test statistic for proportion test.

        Parameters
        ----------
        N : int
            Number of breaches observed.
        T : int
            Total number of observations.

        Returns
        -------
        float
        """
        key = (N, T, self.P)
        if key not in self._z_stat_cache:
            self._z_stat_cache[key] = (N - self.P * T) / math.sqrt(
                T * self.P * (1 - self.P)
            )
        return self._z_stat_cache[key]

    def z_test(self, N: int, T: int) -> float:
        """
        Perform Z-test and return one-sided p-value.

        Parameters
        ----------
        N : int
            Number of breaches observed.
        T : int
            Total number of observations.

        Returns
        -------
        float
            One-sided p-value (probability of observing this many or more breaches).
        """
        s = self.z_test_statistic(N, T)
        return float(1 - norm.cdf(s))

    def kupiec_statistic(self, N: int, T: int) -> float:
        """
        Calculate Kupiec's Likelihood Ratio statistic for Unconditional Coverage.

        Parameters
        ----------
        N : int
            Number of breaches observed.
        T : int
            Total number of observations.

        Returns
        -------
        float
        """
        key = (N, T, self.P)
        if key not in self._lr_uc_cache:
            if N == 0:
                self._lr_uc_cache[key] = -2 * math.log((1 - self.P) ** T)
            elif N == T:
                self._lr_uc_cache[key] = -2 * math.log(self.P**T)
            else:
                self._lr_uc_cache[key] = -2 * math.log(
                    ((1 - self.P) ** (T - N) * self.P**N)
                    / ((1 - N / T) ** (T - N) * (N / T) ** N)
                )
        return self._lr_uc_cache[key]

    def kupiec_test(self, N: int, T: int) -> float:
        """
        Perform Kupiec's unconditional coverage test and return p-value.

        Parameters
        ----------
        N : int
            Number of breaches observed.
        T : int
            Total number of observations.

        Returns
        -------
        float
            p-value (chi-squared with 1 degree of freedom).
        """
        return float(1 - chi2.cdf(self.kupiec_statistic(N, T), df=1))

    def christoffersen_statistic(self, N1: int, N2: int, N3: int, N4: int) -> float:
        """
        Calculate Christoffersen's independence (Markov) test statistic.

        Tests whether breaches are serially independent using a first-order
        Markov chain model.

        Parameters
        ----------
        N1 : int
            Transitions: no-breach → no-breach.
        N2 : int
            Transitions: breach → no-breach.
        N3 : int
            Transitions: no-breach → breach.
        N4 : int
            Transitions: breach → breach.

        Returns
        -------
        float
        """
        key = (N1, N2, N3, N4)
        if key not in self._markov_cache:
            T = N1 + N2 + N3 + N4
            pi = (N3 + N4) / T if T > 0 else 0
            pi0 = N3 / (N1 + N3) if (N1 + N3) > 0 else 0
            pi1 = N4 / (N2 + N4) if (N2 + N4) > 0 else 0

            if pi0 == 0 or pi1 == 0 or pi == 0:
                self._markov_cache[key] = 0
            else:
                self._markov_cache[key] = -2 * math.log(
                    ((1 - pi) ** (N1 + N2) * pi ** (N3 + N4))
                    / ((1 - pi0) ** N1 * pi0**N3 * (1 - pi1) ** N2 * pi1**N4)
                )
        return self._markov_cache[key]

    def christoffersen_test(self, N1: int, N2: int, N3: int, N4: int) -> float:
        """
        Perform Christoffersen's independence test and return p-value.

        Parameters
        ----------
        N1, N2, N3, N4 : int
            Transition counts (see christoffersen_statistic).

        Returns
        -------
        float
            p-value (chi-squared with 1 degree of freedom).
        """
        return float(
            1 - chi2.cdf(self.christoffersen_statistic(N1, N2, N3, N4), df=1)
        )

    def joint_statistic(self, N1: int, N2: int, N3: int, N4: int) -> float:
        """
        Calculate joint test statistic (UC + Independence).

        Parameters
        ----------
        N1, N2, N3, N4 : int
            Transition counts.

        Returns
        -------
        float
        """
        ind = self.christoffersen_statistic(N1, N2, N3, N4)
        uc = self.kupiec_statistic(N=N3 + N4, T=N1 + N2 + N3 + N4)
        return ind + uc

    def joint_test(self, N1: int, N2: int, N3: int, N4: int) -> float:
        """
        Perform joint test and return p-value.

        Parameters
        ----------
        N1, N2, N3, N4 : int
            Transition counts.

        Returns
        -------
        float
            p-value (chi-squared with 2 degrees of freedom).
        """
        return float(1 - chi2.cdf(self.joint_statistic(N1, N2, N3, N4), df=2))

    def martingale_test(self, serie: ArrayLike, K: int = 5) -> tuple[float, float]:
        """
        Perform Martingale (Ljung-Box style) test for autocorrelation in breaches.

        Parameters
        ----------
        serie : array-like
            Binary series of breaches (True/1 = breach).
        K : int, optional (default=5)
            Number of lags to test.

        Returns
        -------
        tuple[float, float]
            (statistic, p-value)
        """
        serie_arr = np.asarray(serie, dtype=float)
        N = len(serie_arr)

        if not np.any(serie_arr):
            return 0.0, 1.0

        serie_adjusted = serie_arr - self.P
        statistic = 0.0

        for i in range(1, K + 1):
            if N <= i:
                continue
            try:
                corr_matrix = np.corrcoef(serie_adjusted[:-i], serie_adjusted[i:])
                corr_matrix = np.asarray(corr_matrix)
                if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                    r = corr_matrix[0, 1]
                    if np.isnan(r):
                        r = 0.0
                else:
                    r = 0.0
            except Exception:
                r = 0.0

            statistic += N * (N + 2) / (N - i) * r**2

        return float(statistic), float(1 - chi2.cdf(statistic, df=K))

    def run_tests(
        self,
        overshoot_serie: ArrayLike,
        T: int,
        min_completeness: float = 0.85,
    ) -> dict[str, Any]:
        """
        Run all backtesting tests on a binary overshoot series.

        Parameters
        ----------
        overshoot_serie : array-like
            Binary series where True/1 indicates a breach.
        T : int
            Number of observations to use (takes most recent T).
        min_completeness : float, optional (default=0.85)
            Minimum fraction of T that must be present.

        Returns
        -------
        dict[str, Any]
            Dictionary with all test results.

        Raises
        ------
        ValueError
            If insufficient data available.
        """
        serie = np.asarray(overshoot_serie, dtype=float)
        if T is None:
            T = len(serie)
        else:
            serie = serie[-T:].copy()
        N = int(serie.sum())

        if len(serie) < min_completeness * T:
            raise ValueError(
                f"Insufficient data: {len(serie)} observations, "
                f"need at least {min_completeness * T:.0f} (T={T})."
            )

        # Markov transition counts
        serie_lag = np.roll(serie, 1)
        serie_lag[0] = 0

        N4 = int((serie * serie_lag).sum())
        N3 = N - N4
        N2 = int(((1 - serie) * serie_lag).sum())
        N1 = T - N - N2

        martingale_stat, martingale_p = self.martingale_test(serie, K=5)

        return {
            "N_Obs": T,
            "N_Overshoots": N,
            "Binomial_Probability": self.binomial_probability(N, T),
            "Binomial_Cumulative_Prob": self.binomial_cumulative_prob(N, T),
            "Z_Test_Statistic": self.z_test_statistic(N, T),
            "Z_Test_P": self.z_test(N, T),
            "Kupiec_Statistic": self.kupiec_statistic(N, T),
            "Kupiec_P": self.kupiec_test(N, T),
            "Christoffersen_Statistic": self.christoffersen_statistic(N1, N2, N3, N4),
            "Christoffersen_P": self.christoffersen_test(N1, N2, N3, N4),
            "Joint_Test_Statistic": self.joint_statistic(N1, N2, N3, N4),
            "Joint_Test_P": self.joint_test(N1, N2, N3, N4),
            "Martingale_Test_Statistic": martingale_stat,
            "Martingale_Test_P": martingale_p,
        }

    # --- Legacy API aliases (backward compatibility) ---

    Binomial_probability = binomial_probability
    Binomial_cumulative_prob = binomial_cumulative_prob
    Z_test_statistic = z_test_statistic
    Z_test = z_test
    LR_uc = kupiec_statistic
    LR_uc_test = kupiec_test
    Markov_statistic = christoffersen_statistic
    Markov_test = christoffersen_test
