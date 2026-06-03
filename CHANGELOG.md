# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Expected Shortfall** module (`expected_shortfall.py`):
  - `historical_es` — empirical CVaR from a return series
  - `normal_es` — closed-form Gaussian ES
  - `es_from_var_series` — mean breach magnitude given realised returns and a VaR series
- **Regulatory** module (`regulatory.py`):
  - `basel_traffic_light` — green/yellow/red classification with capital multiplier add-on
  - `TrafficLightResult` dataclass
- Optional `plotting` extra (`pip install risk-backtest[plotting]`).
- `dev` extra now bundles `build` and `twine` for releases.
- Additional project URLs (Repository, Changelog) in package metadata.

### Fixed
- `cluster_threshold_sensitivity` reported `Avg_Breaches = 0` because it looked
  up a non-existent column. Now uses `N_Overshoots`.
- `VaRBacktest.run_tests` raised `TypeError` when called with `T=None`; it now
  defaults `T` to the series length.

## [0.1.0] - 2026-06-03

### Added
- Initial release.
- Six statistical tests: Binomial, Z, Kupiec, Christoffersen, Joint, Martingale.
- Cluster detection with adjusted tests.
- Batch + parallel backtesting via `run_backtest`.
- VaR estimators: historical, normal, Cornish-Fisher, EVT (POT/GPD), GARCH family.
- Bias and Q-statistic calibration.
- Cluster-threshold sensitivity analysis.
- Plotly-free matplotlib plotting helper.
