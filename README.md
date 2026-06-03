# risk-backtest

A data-agnostic Python toolbox for **Value-at-Risk (VaR) backtesting**, VaR/ES
estimation, calibration statistics, cluster-adjusted hypothesis tests, and
Basel regulatory zone classification.

Designed for portfolio risk teams that need to validate internal models
against UCITS / CSSF / BCBS regulations and produce graphical reports.

## Installation

```bash
pip install risk-backtest                  # core
pip install risk-backtest[parallel]        # + joblib
pip install risk-backtest[models]          # + arch (for GARCH family)
pip install risk-backtest[plotting]        # + matplotlib
pip install risk-backtest[dev]             # everything + pytest, build, twine
```

## Quick start

```python
import numpy as np
from risk_backtest import run_backtest, BacktestConfig

returns  = np.random.normal(0, 0.01, 500)
var      = np.full(500, 0.02)

result = run_backtest(returns, var, window_sizes=[250])
print(result.summary)
print(result.pass_rates)
```

---

## Public API at a glance

| Group | Function / Class | Purpose |
|-------|------------------|---------|
| **High-level** | `run_backtest` | One-call backtest (single or batch, parallel-ready) |
| | `BacktestResult` | Container with `summary`, `pass_rates`, `config` |
| **Config** | `BacktestConfig` | Risk measure + horizon + scaling |
| | `RiskMeasure`, `Horizon` | Enums |
| **Statistical tests** | `VaRBacktest` | Binomial / Z / Kupiec / Christoffersen / Joint / Martingale |
| **Calibration** | `calculate_bias_and_q_statistics` | Rolling bias & Q-statistic (single series) |
| | `calculate_bias_q_batch` | Same, applied across a fund-level DataFrame |
| **Cluster detection** | `detect_cluster` | Identify breach clusters by proximity |
| | `count_clusters` | Count cluster starts and isolated breaches |
| **Sensitivity** | `cluster_threshold_sensitivity` | Sweep cluster thresholds and compare pass-rates |
| **Expected Shortfall** | `historical_es` | Empirical CVaR |
| | `normal_es` | Closed-form Gaussian ES |
| | `es_from_var_series` | Mean breach magnitude vs forecast VaR |
| **Regulatory** | `basel_traffic_light` | Green/Yellow/Red zone + capital multiplier add-on |
| | `TrafficLightResult` | Dataclass result |
| **VaR estimators** | `historical_var` | Empirical quantile |
| | `normal_var` | Parametric Gaussian |
| | `cornish_fisher_var` | Skew/kurtosis-adjusted quantile |
| | `evt_var` (`EVTResult`) | Peaks-Over-Threshold / GPD |
| | `garch_var` (`GARCHResult`) | GARCH / GJR / EGARCH / APARCH |
| | `recursive_garch_variance` | Dependency-free variance recursion |
| | `estimate_var` | Dispatcher — run several methods at once |
| | `VAR_METHODS`, `GARCH_MODELS` | Available method names |
| **Utilities** | `compute_overshoots` | Build the breach boolean series |
| | `create_windows` | Trailing-window slices |
| | `validate_inputs` | Align/length-check arrays |
| **Plotting** | `plot_var_vs_returns` | Report-style returns + VaR bands chart |

---

## High-level backtesting

### Single fund

```python
from risk_backtest import run_backtest

result = run_backtest(returns, var, window_sizes=[250, 500])
result.summary       # MultiIndex (name, window_size, cluster_adj)
result.pass_rates    # % of tests passing at 5% significance
```

### Batch + parallel

```python
ret_dict = {"Fund_A": ret_a, "Fund_B": ret_b}
var_dict = {"Fund_A": var_a, "Fund_B": var_b}

result = run_backtest(ret_dict, var_dict,
                      window_sizes=[250, 500, 750],
                      n_jobs=-1)
```

### Custom risk measure (annual 1-σ volatility)

```python
from risk_backtest import BacktestConfig, run_backtest

config = BacktestConfig(
    risk_measure="volatility",
    confidence_level=0.8413,   # 1-sigma one-sided
    horizon="annual",          # sqrt-T scaled to daily internally
)
result = run_backtest(returns, annual_vol, config=config, window_sizes=[252])
```

---

## Statistical tests (VaRBacktest)

Six tests, each run twice — once on raw breaches and once cluster-adjusted:

| Test | What it measures |
|------|------------------|
| Binomial | Exact probability of observed breach count |
| Z-test | Normal approximation of breach frequency |
| Kupiec (LR-UC) | Unconditional coverage |
| Christoffersen (LR-IND) | Independence of breaches (first-order Markov) |
| Joint (LR-CC) | Coverage + independence combined |
| Martingale | Ljung-Box style autocorrelation in breach series |

```python
from risk_backtest import VaRBacktest

bt = VaRBacktest(P=0.01)
res = bt.run_tests(overshoots, T=250)
print(res["Kupiec_P"], res["Christoffersen_P"])
```

---

## VaR estimation

```python
from risk_backtest import (
    historical_var, normal_var, cornish_fisher_var,
    evt_var, garch_var, estimate_var,
)

historical_var(returns, confidence_level=0.99)
normal_var(returns, confidence_level=0.99)
cornish_fisher_var(returns, confidence_level=0.99)

evt = evt_var(returns, threshold_quantile=0.90)
print(evt.var, evt.es, evt.shape, evt.scale)

garch = garch_var(returns, model_type="gjr-garch", dist="t")
print(garch.var, garch.conditional_vol, garch.persistence)

# Compare several methods at once
estimate_var(returns, methods=["historical", "cornish_fisher", "evt"])
```

Dependency-free conditional-variance recursion using fitted parameters:

```python
from risk_backtest import recursive_garch_variance
sigma2 = recursive_garch_variance(returns,
                                  omega=garch.omega,
                                  alpha=garch.alpha,
                                  beta=garch.beta,
                                  gamma=garch.extra_params.get("gamma", 0),
                                  model_type="gjr-garch")
```

---

## Expected Shortfall

```python
from risk_backtest import historical_es, normal_es, es_from_var_series

historical_es(returns, confidence_level=0.975)   # FRTB default CL
normal_es(returns, confidence_level=0.975)       # closed-form Gaussian
es_from_var_series(returns, var_series)          # mean breach magnitude
```

---

## Calibration: bias and Q-statistics

```python
from risk_backtest import calculate_bias_and_q_statistics, calculate_bias_q_batch

bias, q = calculate_bias_and_q_statistics(
    returns=fund_returns,
    var_forecasts=var_series,
    window_length=60,
    confidence_level=0.99,
)

# Or on a multi-fund DataFrame, adds BIAS_* and Q_STAT_* columns
df = calculate_bias_q_batch(daily_results, window_length=60)
```

Interpretation: bias ≈ 1.0 well-calibrated; Q ≈ 1.577 (1 + Euler-Mascheroni)
under correct calibration.

---

## Cluster detection & sensitivity

```python
from risk_backtest import detect_cluster, count_clusters, cluster_threshold_sensitivity

starts, isolated, real_cluster, cluster_adj = detect_cluster(overshoots, threshold=5)
n_clusters, n_isolated = count_clusters(starts, isolated, window_size=250)

# How sensitive are pass-rates to the cluster threshold?
df = cluster_threshold_sensitivity(returns, var, thresholds=range(1, 11))
```

---

## Regulatory: Basel traffic light

```python
from risk_backtest import basel_traffic_light

res = basel_traffic_light(breaches=6, n_obs=250)
res.zone               # 'yellow'
res.multiplier_addon   # 0.50
res.multiplier         # 3.50 (Basel k = 3 + add-on)
res.cumulative_probability
```

| Breaches (250d, 99% VaR) | Zone | Add-on |
|--------------------------|------|--------|
| 0 – 4 | green  | 0.00 |
| 5 | yellow | 0.40 |
| 6 | yellow | 0.50 |
| 7 | yellow | 0.65 |
| 8 | yellow | 0.75 |
| 9 | yellow | 0.85 |
| ≥10 | red | 1.00 |

For non-standard windows, the add-on is interpolated from the binomial CDF.

---

## Utilities & plotting

```python
from risk_backtest import compute_overshoots, create_windows, validate_inputs

compute_overshoots(returns, var)            # boolean breach array
create_windows(series, window_sizes=[250, 500])
validate_inputs(returns, var)               # raises on mismatch / empty
```

```python
from risk_backtest import plot_var_vs_returns

fig = plot_var_vs_returns(returns, var, dates=dates,
                          title="Fund A — 1-day 99% VaR")
fig.savefig("fund_a.png", dpi=150)
```

Requires `pip install risk-backtest[plotting]`.

---

## License

MIT
