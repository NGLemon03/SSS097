# Mainline Workflow

This document defines the official maintainable runtime chain and quality scope.

## Runtime Baseline

- Python: `3.12`
- Mainline lint target: `ruff.toml` include list
- Mainline tests: `pytest` (configured via `pytest.ini`)

## Official Entry Points

- `run_full_pipeline_5.py`
- `run_oos_analysis.py`
- `run_enhanced_ensemble.py`
- `run_batch_smart_leverage.py`
- `app_dash.py`

## Official Pipeline Sequence

1. `run_full_pipeline_5.py`
2. `analysis/optuna_16.py`
3. `convert_results_to_trades.py`
4. `analysis/optimize_ensemble.py`
5. `init_warehouse.py`
6. `run_oos_analysis.py` or `run_enhanced_ensemble.py`
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

- `auto_pipeline.py` -> forwards to `run_full_pipeline_5.py`
- `run_workflow.py` -> forwards to `run_enhanced_ensemble.py`
- `run_workflow_example.py` -> forwards to `run_enhanced_ensemble.py`

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
