"""
VaR Estimation Models Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standalone Value-at-Risk estimation methods, each usable independently:

- ``historical_var`` — Historical Simulation (empirical quantile)
- ``normal_var`` — Parametric Normal (Gaussian)
- ``cornish_fisher_var`` — Parametric with skewness/kurtosis adjustment
- ``evt_var`` — Extreme Value Theory (Peaks Over Threshold / GPD)
- ``garch_var`` — GARCH-family conditional volatility VaR

Universal dispatcher:

- ``estimate_var`` — Run one or more methods and compare results

All individual methods share the signature pattern:
    func(returns, confidence_level=0.99, verbose=False, ...) -> float | ResultObject
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


# =============================================================================
# Available methods registry (used by estimate_var dispatcher)
# =============================================================================

VAR_METHODS = (
    "historical",
    "normal",
    "cornish_fisher",
    "evt",
    "garch",
    "gjr-garch",
    "egarch",
    "aparch",
)


# =============================================================================
# Historical Simulation
# =============================================================================


def historical_var(
    returns: ArrayLike,
    confidence_level: float = 0.99,
    verbose: bool = False,
) -> float:
    """
    Compute VaR using Historical Simulation (empirical quantile).

    The simplest non-parametric approach: VaR is the (1-CL) quantile of the
    empirical loss distribution.

    Parameters
    ----------
    returns : array-like
        Historical return series.
    confidence_level : float, optional (default=0.99)
        Confidence level (e.g., 0.99 for 99% VaR).
    verbose : bool, optional (default=False)
        Print diagnostic information.

    Returns
    -------
    float
        VaR estimate (positive number representing loss magnitude).
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        raise ValueError("Historical simulation requires at least 10 observations.")

    var_val = -float(np.quantile(returns, 1 - confidence_level))

    if verbose:
        print(f"  Historical Simulation VaR ({confidence_level*100:.0f}%)")
        print(f"    N observations:  {len(returns)}")
        print(f"    VaR:             {var_val:.6f}")

    return var_val


# =============================================================================
# Normal (Parametric Gaussian)
# =============================================================================


def normal_var(
    returns: ArrayLike,
    confidence_level: float = 0.99,
    verbose: bool = False,
) -> float:
    """
    Compute VaR using the Parametric Normal (Gaussian) method.

    Assumes returns follow a normal distribution. VaR is computed as:
        VaR = -(μ + z * σ)
    where z is the normal quantile for the left tail.

    Parameters
    ----------
    returns : array-like
        Historical return series.
    confidence_level : float, optional (default=0.99)
        Confidence level (e.g., 0.99 for 99% VaR).
    verbose : bool, optional (default=False)
        Print diagnostic information.

    Returns
    -------
    float
        VaR estimate (positive number representing loss magnitude).
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        raise ValueError("Normal VaR requires at least 10 observations.")

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = stats.norm.ppf(1 - confidence_level)
    var_val = -(mu + z * sigma)

    if verbose:
        print(f"  Normal (Parametric) VaR ({confidence_level*100:.0f}%)")
        print(f"    N observations:  {len(returns)}")
        print(f"    Mean:            {mu:.6f}")
        print(f"    Std Dev:         {sigma:.6f}")
        print(f"    z:               {z:.4f}")
        print(f"    VaR:             {var_val:.6f}")

    return float(var_val)


# =============================================================================
# Cornish-Fisher Expansion
# =============================================================================


def cornish_fisher_var(
    returns: ArrayLike,
    confidence_level: float = 0.99,
    verbose: bool = False,
) -> float:
    """
    Compute VaR using the Cornish-Fisher expansion.

    Adjusts the normal quantile for skewness and excess kurtosis of the
    empirical return distribution.

    Parameters
    ----------
    returns : array-like
        Historical return series.
    confidence_level : float, optional (default=0.99)
        Confidence level (e.g., 0.99 for 99% VaR).
    verbose : bool, optional (default=False)
        Print intermediate calculations.

    Returns
    -------
    float
        VaR estimate (positive number representing loss magnitude).

    Notes
    -----
    The Cornish-Fisher expansion approximates the quantile of a non-normal
    distribution using the first four moments:

        z_cf = z + (z² - 1) * S/6 + (z³ - 3z) * K/24 - (2z³ - 5z) * S²/36

    where z is the normal quantile, S is skewness, K is excess kurtosis.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        raise ValueError("Cornish-Fisher requires at least 10 observations.")

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skew = float(stats.skew(returns))
    excess_kurt = float(stats.kurtosis(returns))  # excess kurtosis (Fisher)

    # Normal quantile (left tail)
    z = stats.norm.ppf(1 - confidence_level)

    # Cornish-Fisher adjustment
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * excess_kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )

    var = -(mu + z_cf * sigma)

    if verbose:
        print(f"  Cornish-Fisher VaR ({confidence_level*100:.0f}%)")
        print(f"    Mean:            {mu:.6f}")
        print(f"    Std Dev:         {sigma:.6f}")
        print(f"    Skewness:        {skew:.4f}")
        print(f"    Excess Kurtosis: {excess_kurt:.4f}")
        print(f"    Normal z:        {z:.4f}")
        print(f"    CF-adjusted z:   {z_cf:.4f}")
        print(f"    VaR:             {var:.6f}")

    return float(var)


# =============================================================================
# Extreme Value Theory — Peaks Over Threshold (GPD)
# =============================================================================


@dataclass(frozen=True)
class EVTResult:
    """
    Result of EVT-based VaR estimation.

    Attributes
    ----------
    var : float
        VaR estimate.
    es : float
        Expected Shortfall (CVaR) estimate.
    shape : float
        GPD shape parameter (xi).
    scale : float
        GPD scale parameter (beta).
    n_exceedances : int
        Number of threshold exceedances used for fitting.
    threshold : float
        Threshold value used.
    """

    var: float
    es: float
    shape: float
    scale: float
    n_exceedances: int
    threshold: float


def evt_var(
    returns: ArrayLike,
    confidence_level: float = 0.99,
    threshold_quantile: float = 0.90,
    threshold: float | None = None,
    verbose: bool = False,
) -> EVTResult:
    """
    Compute VaR using Extreme Value Theory (Peaks Over Threshold / GPD).

    Fits a Generalized Pareto Distribution to exceedances above a threshold
    in the loss distribution, then extrapolates to the desired quantile.

    Parameters
    ----------
    returns : array-like
        Historical return series.
    confidence_level : float, optional (default=0.99)
        Confidence level for VaR.
    threshold_quantile : float, optional (default=0.90)
        Quantile of the loss distribution used as threshold (between 0 and 1).
        Higher values use fewer but more extreme observations.
    threshold : float, optional
        Explicit threshold value (overrides threshold_quantile if provided).
    verbose : bool, optional (default=False)
        Print intermediate results.

    Returns
    -------
    EVTResult
        Named result with VaR, ES, shape/scale parameters, and diagnostics.

    Notes
    -----
    The Peaks Over Threshold method:
    1. Convert returns to losses (negate)
    2. Set threshold u at the given quantile of losses
    3. Fit GPD to exceedances (losses - u) for observations > u
    4. Compute VaR: u + (beta/xi) * [(n/Nu * (1-CL))^(-xi) - 1]
    5. Compute ES: VaR/(1-xi) + (beta - xi*u)/(1-xi)

    References
    ----------
    McNeil, A.J. & Frey, R. (2000). "Estimation of tail-related risk measures
    for heteroscedastic financial time series: an extreme value approach."
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 30:
        raise ValueError("EVT requires at least 30 observations.")

    # Work with losses (positive values = losses)
    losses = -returns
    n = len(losses)

    # Determine threshold
    if threshold is None:
        u = float(np.quantile(losses, threshold_quantile))
    else:
        u = float(threshold)

    # Exceedances above threshold
    exceedances = losses[losses > u] - u
    n_exceed = len(exceedances)

    if n_exceed < 10:
        raise ValueError(
            f"Only {n_exceed} exceedances found (need >= 10). "
            f"Try lowering threshold_quantile (currently {threshold_quantile})."
        )

    # Fit GPD to exceedances using MLE
    xi, loc, beta = stats.genpareto.fit(exceedances, floc=0)

    # Compute VaR using the GPD quantile formula
    p = 1 - confidence_level  # tail probability
    Fu = n_exceed / n  # fraction exceeding threshold

    if abs(xi) < 1e-10:
        # Exponential case (xi → 0)
        var_val = u + beta * np.log(Fu / p)
    else:
        var_val = u + (beta / xi) * ((Fu / p) ** xi - 1)

    # Compute Expected Shortfall
    if xi < 1:
        es_val = var_val / (1 - xi) + (beta - xi * u) / (1 - xi)
    else:
        es_val = np.inf  # ES not defined for xi >= 1

    if verbose:
        print(f"  EVT (Peaks Over Threshold) VaR ({confidence_level*100:.0f}%)")
        print(f"    N observations:  {n}")
        print(f"    Threshold (u):   {u:.6f}")
        print(f"    N exceedances:   {n_exceed} ({n_exceed/n*100:.1f}%)")
        print(f"    GPD shape (ξ):   {xi:.4f}")
        print(f"    GPD scale (β):   {beta:.6f}")
        print(f"    VaR:             {var_val:.6f}")
        print(f"    ES (CVaR):       {es_val:.6f}")

    return EVTResult(
        var=float(var_val),
        es=float(es_val),
        shape=float(xi),
        scale=float(beta),
        n_exceedances=n_exceed,
        threshold=u,
    )


# =============================================================================
# GARCH Family — Conditional VaR
# =============================================================================

# Supported volatility model types
GARCH_MODELS = ("garch", "gjr-garch", "egarch", "aparch")


@dataclass(frozen=True)
class GARCHResult:
    """
    Result of GARCH-family VaR estimation.

    Attributes
    ----------
    var : float
        1-step-ahead VaR estimate.
    conditional_vol : float
        1-step-ahead conditional volatility forecast.
    omega : float
        Estimated omega parameter.
    alpha : float
        Estimated alpha (ARCH) parameter.
    beta : float
        Estimated beta (GARCH) parameter.
    persistence : float
        Alpha + beta (persistence measure).
    unconditional_vol : float
        Long-run (unconditional) daily volatility.
    log_likelihood : float
        Model log-likelihood.
    model_type : str
        Model variant used (garch, gjr-garch, egarch, aparch).
    extra_params : dict
        Additional model-specific parameters (gamma for GJR/EGARCH/APARCH, delta for APARCH).
    """

    var: float
    conditional_vol: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    unconditional_vol: float
    log_likelihood: float
    model_type: str = "garch"
    extra_params: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.extra_params is None:
            object.__setattr__(self, "extra_params", {})


def garch_var(
    returns: ArrayLike,
    confidence_level: float = 0.99,
    model_type: str = "garch",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    mean: str = "constant",
    verbose: bool = False,
) -> GARCHResult:
    """
    Compute VaR using a GARCH-family model.

    Supports GARCH, GJR-GARCH, EGARCH, and APARCH specifications.

    Parameters
    ----------
    returns : array-like
        Historical return series (decimal, e.g., 0.01 = 1%).
    confidence_level : float, optional (default=0.99)
        Confidence level for VaR.
    model_type : str, optional (default='garch')
        Volatility model: 'garch', 'gjr-garch', 'egarch', or 'aparch'.
    p : int, optional (default=1)
        GARCH lag order (lagged conditional variances).
    q : int, optional (default=1)
        ARCH lag order (lagged squared innovations).
    dist : str, optional (default='normal')
        Error distribution: 'normal', 't', or 'skewt'.
    mean : str, optional (default='constant')
        Mean model: 'zero', 'constant', or 'ar'.
    verbose : bool, optional (default=False)
        Print model parameters and diagnostics.

    Returns
    -------
    GARCHResult
        Named result with VaR, conditional vol, and estimated parameters.

    Raises
    ------
    ImportError
        If `arch` package is not installed.

    Notes
    -----
    Requires the `arch` package: pip install risk-backtest[models]

    Model specifications:
    - **GARCH(1,1)**: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    - **GJR-GARCH**: σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
    - **EGARCH**: log(σ²_t) = ω + α·(|z_{t-1}| - E|z|) + γ·z_{t-1} + β·log(σ²_{t-1})
    - **APARCH**: σ^δ_t = ω + α·(|ε_{t-1}| - γ·ε_{t-1})^δ + β·σ^δ_{t-1}

    Examples
    --------
    >>> result = garch_var(returns, model_type='gjr-garch', dist='t')
    >>> print(f"VaR: {result.var:.4%}, leverage: {result.extra_params.get('gamma', 0):.4f}")
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError(
            "GARCH estimation requires the 'arch' package. "
            "Install it with: pip install risk-backtest[models]"
        )

    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 100:
        raise ValueError("GARCH requires at least 100 observations.")

    model_type = model_type.lower()
    if model_type not in GARCH_MODELS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {', '.join(GARCH_MODELS)}"
        )

    # Scale to percentage returns for arch package
    returns_pct = returns * 100

    # Map mean model
    mean_map = {"zero": "Zero", "constant": "Constant", "ar": "ARX"}
    mean_spec = mean_map.get(mean, "Constant")

    # Map distribution
    dist_map = {"normal": "normal", "t": "t", "skewt": "skewt"}
    dist_spec = dist_map.get(dist, "normal")

    # Map model type to arch package vol spec
    vol_map = {
        "garch": "Garch",
        "gjr-garch": "Garch",
        "egarch": "EGARCH",
        "aparch": "APARCH",
    }
    vol_spec = vol_map[model_type]

    # Build model kwargs
    model_kwargs: dict[str, Any] = {
        "mean": mean_spec,
        "vol": vol_spec,
        "p": p,
        "q": q,
        "dist": dist_spec,
    }
    if model_type == "gjr-garch":
        model_kwargs["o"] = 1  # one asymmetric lag

    # Fit model
    model = arch_model(returns_pct, **model_kwargs)
    result = model.fit(disp="off", show_warning=False)

    # Extract common parameters
    omega = result.params.get("omega", 0)
    alpha_val = sum(result.params.get(f"alpha[{i}]", 0) for i in range(1, q + 1))
    beta_val = sum(result.params.get(f"beta[{i}]", 0) for i in range(1, p + 1))

    # Extract model-specific params
    extra_params: dict[str, float] = {}
    if model_type == "gjr-garch":
        gamma = sum(result.params.get(f"gamma[{i}]", 0) for i in range(1, 2))
        extra_params["gamma"] = float(gamma)
        persistence = alpha_val + beta_val + 0.5 * gamma
    elif model_type == "egarch":
        gamma = sum(result.params.get(f"gamma[{i}]", 0) for i in range(1, q + 1))
        extra_params["gamma"] = float(gamma)
        persistence = beta_val  # EGARCH persistence is just beta
    elif model_type == "aparch":
        gamma = sum(result.params.get(f"gamma[{i}]", 0) for i in range(1, q + 1))
        delta = result.params.get("delta", 2.0)
        extra_params["gamma"] = float(gamma)
        extra_params["delta"] = float(delta)
        persistence = alpha_val + beta_val
    else:
        persistence = alpha_val + beta_val

    # 1-step ahead forecast
    forecast = result.forecast(horizon=1)
    cond_var = forecast.variance.iloc[-1, 0]  # in pct^2
    cond_vol_pct = np.sqrt(cond_var)
    cond_vol = cond_vol_pct / 100  # back to decimal

    # Unconditional variance
    if model_type == "egarch":
        # EGARCH unconditional vol from forecast mean
        uncond_vol = float(np.std(returns, ddof=1))  # fallback to sample
    elif persistence < 1:
        uncond_var = omega / (1 - persistence)
        uncond_vol = np.sqrt(uncond_var) / 100  # decimal
    else:
        uncond_vol = np.nan

    # Compute quantile based on distribution
    alpha_tail = 1 - confidence_level
    if dist_spec == "normal":
        z = stats.norm.ppf(alpha_tail)
    elif dist_spec == "t":
        nu = result.params.get("nu", 5)
        z = stats.t.ppf(alpha_tail, df=nu)
        z = z * np.sqrt((nu - 2) / nu)
    elif dist_spec == "skewt":
        z = stats.norm.ppf(alpha_tail)
    else:
        z = stats.norm.ppf(alpha_tail)

    # Mean forecast
    mean_forecast = result.params.get("mu", 0) / 100 if "mu" in result.params else 0

    # VaR (positive = loss)
    var_val = -(mean_forecast + z * cond_vol)

    if verbose:
        model_label = model_type.upper().replace("-", "_")
        print(f"  {model_label}({p},{q}) VaR ({confidence_level*100:.0f}%)")
        print(f"    Distribution:      {dist_spec}")
        print(f"    Mean model:        {mean_spec}")
        print(f"    N observations:    {len(returns)}")
        print(f"    ω (omega):         {omega:.6f}")
        print(f"    α (alpha):         {alpha_val:.6f}")
        print(f"    β (beta):          {beta_val:.6f}")
        if "gamma" in extra_params:
            print(f"    γ (gamma):         {extra_params['gamma']:.6f}")
        if "delta" in extra_params:
            print(f"    δ (delta):         {extra_params['delta']:.4f}")
        print(f"    Persistence:       {persistence:.6f}")
        print(f"    Cond. Vol (daily): {cond_vol:.6f} ({cond_vol*100:.4f}%)")
        print(f"    Uncond. Vol:       {uncond_vol:.6f} ({uncond_vol*100:.4f}%)")
        print(f"    Log-Likelihood:    {result.loglikelihood:.2f}")
        print(f"    VaR:               {var_val:.6f}")

    return GARCHResult(
        var=float(var_val),
        conditional_vol=float(cond_vol),
        omega=float(omega),
        alpha=float(alpha_val),
        beta=float(beta_val),
        persistence=float(persistence),
        unconditional_vol=float(uncond_vol),
        log_likelihood=float(result.loglikelihood),
        model_type=model_type,
        extra_params=extra_params,
    )


# =============================================================================
# Recursive Variance Calculation (no external dependency)
# =============================================================================


def recursive_garch_variance(
    returns: ArrayLike,
    omega: float = 0.00001,
    alpha: float = 0.1,
    beta: float = 0.85,
    gamma: float = 0.0,
    model_type: str = "garch",
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute conditional variance series using recursive GARCH-family equations.

    This is a lightweight, dependency-free implementation that computes the full
    conditional variance path given fixed parameters (e.g., from a prior estimation).
    Uses O(n) recursive computation — no matrix operations or optimization.

    Parameters
    ----------
    returns : array-like
        Return series (decimal).
    omega : float
        Baseline variance constant.
    alpha : float
        ARCH coefficient (shock sensitivity).
    beta : float
        GARCH coefficient (variance persistence).
    gamma : float, optional (default=0.0)
        Asymmetry/leverage parameter (used by GJR-GARCH and EGARCH).
    model_type : str, optional (default='garch')
        Model type: 'garch', 'gjr-garch', or 'egarch'.
    verbose : bool, optional (default=False)
        Print summary statistics of the variance series.

    Returns
    -------
    np.ndarray
        Array of conditional variances (same length as returns).

    Notes
    -----
    Recursive equations (O(n) complexity):
    - GARCH:     σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
    - GJR-GARCH: σ²_t = ω + (α + γ·I_{t-1})·r²_{t-1} + β·σ²_{t-1}
    - EGARCH:    log(σ²_t) = ω + α·(|z_{t-1}|-√(2/π)) + γ·z_{t-1} + β·log(σ²_{t-1})

    where I_{t-1} = 1 if r_{t-1} < 0 (leverage indicator).

    Examples
    --------
    >>> # Use parameters from a fitted model
    >>> result = garch_var(returns, model_type='gjr-garch', verbose=True)
    >>> var_series = recursive_garch_variance(
    ...     returns, omega=result.omega, alpha=result.alpha,
    ...     beta=result.beta, gamma=result.extra_params['gamma'],
    ...     model_type='gjr-garch'
    ... )
    >>> daily_vol = np.sqrt(var_series)
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    model_type = model_type.lower()

    if model_type not in ("garch", "gjr-garch", "egarch"):
        raise ValueError(
            f"Recursive calculation supports: garch, gjr-garch, egarch. Got '{model_type}'."
        )

    # ------------------------------------------------------------------
    # Unit handling
    # ------------------------------------------------------------------
    # ``garch_var`` fits arch_model on returns * 100 (percentage scale), so the
    # returned parameters (omega, alpha, beta, gamma) are calibrated in pct/pct²
    # space. To stay consistent and produce a conditional variance that matches
    # ``garch_var.conditional_vol`` (decimal scale), we run the recursion on
    # percentage returns internally, then convert the resulting σ² (pct²) back
    # to decimal² by dividing by SCALE² = 10000.
    # ------------------------------------------------------------------
    SCALE = 100.0
    returns_pct = returns * SCALE

    sigma2_pct = np.empty(n, dtype=np.float64)
    init_var_pct = np.var(returns_pct[:min(50, n)])
    if init_var_pct <= 0:
        if model_type != "egarch" and (1 - alpha - beta) > 0:
            init_var_pct = omega / (1 - alpha - beta)
        else:
            init_var_pct = 1.0
    sigma2_pct[0] = init_var_pct

    if model_type == "egarch":
        # Work in log-variance space (pct²)
        log_sigma2 = np.empty(n, dtype=np.float64)
        log_sigma2[0] = np.log(sigma2_pct[0])
        e_abs_z = np.sqrt(2.0 / np.pi)  # E[|z|] for standard normal

        for t in range(1, n):
            z_prev = returns_pct[t - 1] / np.sqrt(np.exp(log_sigma2[t - 1]))
            log_sigma2[t] = (
                omega
                + alpha * (abs(z_prev) - e_abs_z)
                + gamma * z_prev
                + beta * log_sigma2[t - 1]
            )
            # Clamp to prevent overflow
            log_sigma2[t] = np.clip(log_sigma2[t], -30, 20)

        sigma2_pct = np.exp(log_sigma2)

    elif model_type == "gjr-garch":
        for t in range(1, n):
            r_prev_sq = returns_pct[t - 1] ** 2
            indicator = 1.0 if returns_pct[t - 1] < 0 else 0.0
            sigma2_pct[t] = (
                omega
                + (alpha + gamma * indicator) * r_prev_sq
                + beta * sigma2_pct[t - 1]
            )
            sigma2_pct[t] = max(sigma2_pct[t], 1e-20)

    else:  # standard GARCH
        for t in range(1, n):
            sigma2_pct[t] = (
                omega + alpha * returns_pct[t - 1] ** 2 + beta * sigma2_pct[t - 1]
            )
            sigma2_pct[t] = max(sigma2_pct[t], 1e-20)

    # Convert pct² → decimal²
    sigma2 = sigma2_pct / (SCALE ** 2)

    if verbose:
        vol = np.sqrt(sigma2)
        print(f"  Recursive {model_type.upper()} Variance (n={n})")
        print(f"    ω={omega:.2e}, α={alpha:.4f}, β={beta:.4f}", end="")
        if gamma != 0:
            print(f", γ={gamma:.4f}")
        else:
            print()
        print(f"    Avg daily vol:   {vol.mean():.6f} ({vol.mean()*100:.4f}%)")
        print(f"    Min daily vol:   {vol.min():.6f}")
        print(f"    Max daily vol:   {vol.max():.6f}")
        print(f"    Persistence:     {alpha + beta:.4f}")

    return sigma2


# =============================================================================
# Convenience: Compare All Methods
# =============================================================================


def estimate_var(
    returns: ArrayLike,
    confidence_level: float = 0.99,
    methods: list[str] | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, float | EVTResult | GARCHResult]:
    """
    Universal dispatcher: estimate VaR using one or more methods.

    Each method delegates to its standalone function. Use this for quick
    comparison across approaches, or call individual functions directly
    for finer control.

    Parameters
    ----------
    returns : array-like
        Historical return series.
    confidence_level : float, optional (default=0.99)
        Confidence level for VaR.
    methods : list[str], optional
        Methods to use. Defaults to all non-GARCH methods.
        Available: {VAR_METHODS}
    verbose : bool, optional (default=False)
        Print intermediate results for each method.
    **kwargs
        Additional keyword arguments passed to individual methods:
        - threshold_quantile: for EVT (default 0.90)
        - garch_p, garch_q: GARCH order (default 1, 1)
        - garch_dist: GARCH error distribution (default 'normal')
        - garch_mean: mean model (default 'constant')

    Returns
    -------
    dict[str, float | EVTResult | GARCHResult]
        Dictionary mapping method name to VaR estimate (or result object).
        Use ``result.var`` for EVTResult/GARCHResult, or ``result`` directly
        for float methods.

    Examples
    --------
    >>> results = estimate_var(returns, methods=['cornish_fisher', 'evt', 'gjr-garch'])
    >>> for method, var in results.items():
    ...     val = var.var if hasattr(var, 'var') else var
    ...     print(f"{method}: {val:.4%}")

    >>> # Or call individual functions directly:
    >>> from risk_backtest import historical_var, normal_var, cornish_fisher_var
    >>> historical_var(returns, confidence_level=0.99)
    """
    if methods is None:
        methods = ["historical", "normal", "cornish_fisher", "evt"]

    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    results: dict[str, Any] = {}
    garch_family = set(GARCH_MODELS)

    for method in methods:
        if verbose:
            print(f"\n{'─'*50}")
            print(f"  Method: {method}")
            print(f"{'─'*50}")

        if method == "historical":
            results[method] = historical_var(
                returns, confidence_level=confidence_level, verbose=verbose
            )

        elif method == "normal":
            results[method] = normal_var(
                returns, confidence_level=confidence_level, verbose=verbose
            )

        elif method == "cornish_fisher":
            results[method] = cornish_fisher_var(
                returns, confidence_level=confidence_level, verbose=verbose
            )

        elif method == "evt":
            threshold_q = kwargs.get("threshold_quantile", 0.90)
            results[method] = evt_var(
                returns,
                confidence_level=confidence_level,
                threshold_quantile=threshold_q,
                verbose=verbose,
            )

        elif method in garch_family:
            garch_p = kwargs.get("garch_p", 1)
            garch_q = kwargs.get("garch_q", 1)
            garch_dist = kwargs.get("garch_dist", "normal")
            garch_mean = kwargs.get("garch_mean", "constant")
            results[method] = garch_var(
                returns,
                confidence_level=confidence_level,
                model_type=method,
                p=garch_p,
                q=garch_q,
                dist=garch_dist,
                mean=garch_mean,
                verbose=verbose,
            )

        else:
            raise ValueError(
                f"Unknown method '{method}'. Available: {', '.join(VAR_METHODS)}"
            )

    return results
