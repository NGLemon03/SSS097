# Mainline Workflow

This document defines the official maintainable runtime chain and quality scope.

## Runtime Baseline

- Python: `3.12`
- Mainline lint target: `ruff.toml` include list
- Mainline tests: `pytest` (configured via `pytest.ini`)

## Official Entry Points

- `analysis/optuna_16.py`
- `convert_results_to_trades.py`
- `run_oos_analysis.py`
- `run_enhanced_ensemble.py`
- `run_batch_smart_leverage.py`
- `app_dash.py`

## Official Pipeline Sequence

1. `analysis/optuna_16.py`
2. `convert_results_to_trades.py`
3. `analysis/optimize_ensemble.py`
4. `init_warehouse.py`
5. `run_oos_analysis.py` or `run_enhanced_ensemble.py`
6. `run_batch_smart_leverage.py` (optional)
7. `app_dash.py` for UI analysis

## Active Analysis Modules

- `analysis/optuna_16.py`
- `analysis/optimize_ensemble.py`
- `analysis/strategy_manager.py`
- `analysis/data_loader.py`
- `analysis/config.py`
- `analysis/metrics.py`
- `analysis/logging_config.py`

## Compatibility Wrappers

These files are kept for backward compatibility and now print deprecation guidance:

- `auto_pipeline.py` -> historical wrapper (target script archived)
- `run_workflow.py` -> forwards to `run_enhanced_ensemble.py`
- `run_workflow_example.py` -> forwards to `run_enhanced_ensemble.py`

## Archived Entry Points

These scripts were moved out of root and are preserved in archive snapshots:

- `run_full_pipeline_5.py` -> `archive/test_series_one_folder_20260306_182919/`
- `run_full_pipeline.txt` -> `archive/test_series_one_folder_20260306_182919/`
- `run_round3*`/`run_round4*`/`run_round5*`/`run_round6*` -> `archive/test_series_one_folder_20260306_182919/`
- `run_hybrid_backtest*.py` -> `archive/test_series_one_folder_20260306_182919/`

## Legacy Split

Archived historical code is under `legacy/` and excluded from default lint/test:

- `legacy/analysis/old`
- `legacy/analysis/past`
- `legacy/analysis/Optuna`

See `legacy/manifest.json` for archive mapping.

## Validation Commands

```powershell
python -m pytest --collect-only
python -m pytest -q
ruff check .
ruff check . --select F821,E722
```
