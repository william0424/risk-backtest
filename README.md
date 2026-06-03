# risk-backtest

A Python library for Value at Risk (VaR) model backtesting with statistical tests, cluster detection, and parallel batch processing.

## Installation

```bash
pip install risk-backtest
```

For parallel processing support:
```bash
pip install risk-backtest[parallel]
```

## Quick Start

```python
import numpy as np
from risk_backtest import run_backtest, BacktestConfig

# Single fund
returns = np.random.normal(0, 0.01, 500)
var_series = np.full(500, 0.02)

result = run_backtest(returns, var_series, window_sizes=[250])
print(result.summary)
print(result.pass_rates)
```

## Batch Mode

```python
# Multiple funds in one call
returns_dict = {"Fund_A": returns_a, "Fund_B": returns_b}
var_dict = {"Fund_A": var_a, "Fund_B": var_b}

result = run_backtest(returns_dict, var_dict, window_sizes=[250, 500], n_jobs=4)
```

## Custom Risk Measures

```python
from risk_backtest import BacktestConfig

# Annual volatility backtesting (auto-scales to daily)
config = BacktestConfig(
    risk_measure="volatility",
    confidence_level=0.8413,  # 1-sigma
    horizon="annual",
)

result = run_backtest(returns, annual_vol, config=config, window_sizes=[252])
```

## Statistical Tests

Six tests are applied per window:

| Test | What it measures |
|------|-----------------|
| Binomial | Exact probability of observed breaches |
| Z-test | Normal approximation of breach frequency |
| Kupiec (LR-UC) | Unconditional coverage |
| Christoffersen (LR-IND) | Independence of breaches |
| Joint (LR-CC) | Combined coverage + independence |
| Martingale | Predictability in breach sequence |

Each test runs twice: once on raw breaches and once cluster-adjusted (avoiding double-counting correlated breaches).

## Features

- **Cluster detection**: Identifies breach clusters to avoid inflating test statistics
- **Parallelization**: Process hundreds of funds efficiently with joblib or concurrent.futures
- **Flexible horizons**: Daily, weekly, monthly, annual with sqrt-T scaling
- **Multiple risk measures**: VaR, volatility, Expected Shortfall
- **Expected Shortfall**: `historical_es`, `normal_es`, `es_from_var_series`
- **Regulatory helpers**: `basel_traffic_light` for Basel green/yellow/red zone classification

## Regulatory Backtesting

```python
from risk_backtest import basel_traffic_light

result = basel_traffic_light(breaches=6, n_obs=250)
print(result.zone)              # 'yellow'
print(result.multiplier_addon)  # 0.50
print(result.multiplier)        # 3.50
```

## License

MIT
