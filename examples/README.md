# risk-backtest Examples

Example notebooks demonstrating the main features of the `risk-backtest` package.

## Notebooks

| Notebook | Topic | Key Functions |
|----------|-------|---------------|
| [01_backtesting.ipynb](01_backtesting.ipynb) | VaR Backtesting | `run_backtest()`, `BacktestConfig`, batch mode, parallel execution |
| [02_var_models.ipynb](02_var_models.ipynb) | VaR Estimation | `cornish_fisher_var()`, `evt_var()`, `garch_var()`, `recursive_garch_variance()` |
| [03_sensitivity_analysis.ipynb](03_sensitivity_analysis.ipynb) | Cluster Sensitivity | `cluster_threshold_sensitivity()`, `detect_cluster()` |

## Prerequisites

```bash
# Install package (editable mode for development)
pip install -e path/to/risk-backtest

# For GARCH models
pip install -e "path/to/risk-backtest[models]"

# For charts in notebooks
pip install matplotlib
```

## Quick Reference

```python
# Backtesting
from risk_backtest import run_backtest, BacktestConfig
result = run_backtest(returns, var_series, window_sizes=[252], n_jobs=-1)

# VaR Models
from risk_backtest import estimate_var
results = estimate_var(returns, methods=["cornish_fisher", "evt", "gjr-garch"])

# Sensitivity
from risk_backtest import cluster_threshold_sensitivity
df = cluster_threshold_sensitivity(returns_dict, var_dict, thresholds=range(1, 16))
```
