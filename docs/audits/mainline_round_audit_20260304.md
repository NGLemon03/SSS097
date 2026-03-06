# Mainline + Round Derived Scripts Audit (2026-03-04)

## Scope
- Included official entrypoints (5):
  - `run_full_pipeline_5.py`
  - `run_oos_analysis.py`
  - `run_enhanced_ensemble.py`
  - `run_batch_smart_leverage.py`
  - `app_dash.py`
- Included commonly used derived scripts (10):
  - `run_round3_regime_anti_overfit_test.py`
  - `run_round3_sequential_new_logic_tests.py`
  - `run_round4_overlay_test.py`
  - `run_round5_crash_shield_robustness.py`
  - `run_round6_core_strategy_focus.py`
  - `run_round6_followup_tests.py`
  - `run_anti_overfit_full_period_test.py`
  - `run_beta_test.py`
  - `ensemble_wrapper.py`
  - `predict_tomorrow.py`
- Excluded: `legacy/`, `archive/`

## Validation Executed
1. Static compile smoke:
   - `py_compile` for non-legacy Python files: pass
2. Mainline pytest smoke:
   - `python -m pytest -q tests/test_mainline_smoke.py` -> `14 passed`
   - `python -m pytest -q tests/test_app_dash_subprocess_safety.py tests/test_app_dash_vote_diagnostics.py` -> `4 passed`
3. CLI help smoke for argparse scripts in this scope:
   - Round scripts + anti-overfit + beta + predict_tomorrow: `--help` pass
4. Core import alignment:
   - `from/import SSSv096 | sss_core | SSS_EnsembleTab` symbol checks: `118` references checked, `2` missing references found in `app_dash.py` (`SSS_EnsembleTab.trade_contribution_by_phase`)

## Status Rules Used
- `Green`: runtime contract aligned with mainline expectations and has direct lint/test coverage
- `Yellow`: executable but maintenance/contract risk exists (hardcoded runtime, no standardized CLI, no mainline test/lint coverage, or import mismatch in non-smoke-covered feature paths)
- `Red`: executable contract is materially weaker than mainline (side-effectful demo entrypoint, no stable CLI contract)

## Inventory Table
The required deliverable table is in:
- [mainline_round_audit_20260304.csv](/c:/Stock_reserach/002g/docs/audits/mainline_round_audit_20260304.csv)

## Functional Inventory Notes
1. `run_full_pipeline_5.py`
   - Purpose: orchestrates optuna -> trade conversion -> ensemble optimization -> warehouse init.
   - Startup: `python run_full_pipeline_5.py`.
   - Inputs: `analysis/optuna_16.py`, existing `results/`, config constants in file.
   - Outputs: archived CSVs under `archive/*`, generated results and warehouse references.
   - Coupling: strong process-chain coupling to analysis pipeline scripts.
   - Last updated: 2026-02-23.
   - Mainline lint/test: in `ruff` include; pytest compile coverage only.
2. `run_oos_analysis.py`
   - Purpose: OOS comparison plotting for ensemble variants.
   - Startup: `python run_oos_analysis.py --split_date ...`.
   - Inputs: active warehouse, strategy trades from `sss_backtest_outputs`/`archive`, market data.
   - Outputs: `sss_backtest_outputs/ensemble_daily_state_*.csv`, interactive plot.
   - Coupling: `SSS_EnsembleTab`, `analysis.strategy_manager`, `sss_core.logic`.
   - Last updated: 2026-01-05.
   - Mainline lint/test: in `ruff` include; pytest compile + `--help` coverage.
3. `run_enhanced_ensemble.py`
   - Purpose: enhanced/walk-forward ensemble execution mode.
   - Startup: `python run_enhanced_ensemble.py [args]`.
   - Inputs: `trades_from_results_*.csv` in `sss_backtest_outputs`.
   - Outputs: runtime logs (no strong standardized artifact contract in script body).
   - Coupling: `SSS_EnsembleTab`.
   - Last updated: 2025-12-20.
   - Mainline lint/test: in `ruff` include; pytest compile + `--help` coverage.
4. `run_batch_smart_leverage.py`
   - Purpose: batch evaluate smart-leverage portfolio against warehouse snapshots.
   - Startup: `python run_batch_smart_leverage.py`.
   - Inputs: warehouse JSON list, trade CSVs, target/safe asset prices (includes fetch/update).
   - Outputs: refreshed safe-asset CSV cache and evaluation summary artifacts.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `analysis.strategy_manager`.
   - Last updated: 2026-02-23.
   - Mainline lint/test: in `ruff` include; pytest compile coverage only.
5. `app_dash.py`
   - Purpose: interactive UI + integrated strategy backtesting/analysis.
   - Startup: `python app_dash.py`.
   - Inputs: warehouse, trades, market data, mainline strategy modules.
   - Outputs: UI exports (`csv` download artifacts), temp trade/debug files.
   - Coupling: `SSSv096`, `SSS_EnsembleTab`, multiple `sss_core` submodules.
   - Last updated: 2026-03-03.
   - Mainline lint/test: in `ruff` include; dedicated app_dash pytest coverage exists.
   - Contract note: callbacks at `app_dash.py:4266` and `app_dash.py:4594` import missing symbol `SSS_EnsembleTab.trade_contribution_by_phase` (currently caught by `try/except`, so feature degrades instead of crashing).
6. `run_round3_regime_anti_overfit_test.py`
   - Purpose: regime-aware anti-overfit full-period walk-forward scoring.
   - Startup: CLI argparse.
   - Inputs: warehouse, trades, market data.
   - Outputs: summary/weights/equity/selection/score/regime CSVs in `results/`.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `analysis.strategy_manager`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
7. `run_round3_sequential_new_logic_tests.py`
   - Purpose: sequential round-3 logic checks and output comparison.
   - Startup: CLI argparse.
   - Inputs: warehouse, trades, market data.
   - Outputs: summary, weights, equity, selected strategy CSVs.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `analysis.strategy_manager`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
8. `run_round4_overlay_test.py`
   - Purpose: overlay robustness test and parameter variants.
   - Startup: CLI argparse.
   - Inputs: warehouse, trades, market data.
   - Outputs: result slices + weights/equity CSV outputs.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `analysis.strategy_manager`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
9. `run_round5_crash_shield_robustness.py`
   - Purpose: crash-shield robustness/sensitivity diagnostics.
   - Startup: CLI argparse.
   - Inputs: split schedules, warehouse-derived strategy files, market data.
   - Outputs: wide set of result/sensitivity/activity/debug CSV outputs.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `run_round4_overlay_test`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
10. `run_round6_core_strategy_focus.py`
   - Purpose: focused evaluation for core strategy families.
   - Startup: CLI argparse.
   - Inputs: warehouse + market data.
   - Outputs: core summary/inactive/weights/equity CSV outputs.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `run_round4_overlay_test`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
11. `run_round6_followup_tests.py`
   - Purpose: follow-up robustness tests and turn-stage diagnostics.
   - Startup: CLI argparse.
   - Inputs: warehouse + multi-source market data.
   - Outputs: follow-up result CSVs and weight dump.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `run_round4_overlay_test`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
12. `run_anti_overfit_full_period_test.py`
   - Purpose: full-period anti-overfitting walk-forward selection backtest.
   - Startup: CLI argparse.
   - Inputs: warehouse + trades + market data.
   - Outputs: summary/weights/equity/selection CSVs.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `analysis.strategy_manager`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
13. `run_beta_test.py`
   - Purpose: beta strategy selection/evaluation against benchmark behavior.
   - Startup: CLI argparse.
   - Inputs: active warehouse, trades, benchmark prices.
   - Outputs: result CSV in `results/`.
   - Coupling: `SSS_EnsembleTab`, `sss_core.logic`, `analysis.strategy_manager`.
   - Last updated: 2026-02-24.
   - Mainline lint/test: not in `ruff` include, no pytest target.
14. `ensemble_wrapper.py`
   - Purpose: utility wrapper for converting optuna results into trade files and ad-hoc ensemble calls.
   - Startup: `__main__` demo block (not parameterized CLI).
   - Inputs: `results/optuna_results_*.csv`, market data, trade output dirs.
   - Outputs: generated `trades_*.csv` and demo-run artifacts.
   - Coupling: `SSS_EnsembleTab`, `SSSv096`.
   - Last updated: 2026-02-25.
   - Mainline lint/test: not in `ruff` include, no pytest target.
15. `predict_tomorrow.py`
   - Purpose: daily signal inference from warehouse and latest data/trades.
   - Startup: CLI argparse (`--ticker`, `--warehouse`).
   - Inputs: `analysis/*warehouse*.json`, `results/`, `sss_backtest_outputs/`, live data loader.
   - Outputs: console prediction summary.
   - Coupling: `sss_core`, `analysis.strategy_manager`.
   - Last updated: 2026-02-12.
   - Mainline lint/test: not in `ruff` include, no pytest target.

## Risk List (Priority Ordered)
### P1 - Mainline risks
1. `app_dash.py`: phase-contribution feature path imports missing `SSS_EnsembleTab.trade_contribution_by_phase`; runtime falls back to error text and this path is not covered by current app pytest.
2. `run_full_pipeline_5.py`: official entrypoint but no CLI contract (`argparse`, `--help`, `--dry-run`), and runtime knobs are hardcoded constants.
3. `run_batch_smart_leverage.py`: official entrypoint but no CLI contract; high side-effect profile (data refresh/network/filesystem) tied to hardcoded defaults.

### P2 - Fixable drift in commonly used derived scripts
1. Round scripts (`run_round3*`, `run_round4*`, `run_round5*`, `run_round6*`, `run_anti_overfit_full_period_test.py`, `run_beta_test.py`) are executable and smoke-clean, but not in mainline lint/test scope.
2. `run_oos_analysis.py` and `run_enhanced_ensemble.py` are official but still rely on internal hardcoded behavior that is not regression-tested by behavior-level pytest.
3. `ensemble_wrapper.py` lacks a stable CLI contract and runs side-effectful demo logic in `__main__`; highest drift risk among derived scripts.

### P3 - Tooling boundary/maintenance hygiene
1. `predict_tomorrow.py` is operational but outside mainline quality gates; path assumptions and no automated test target.

## Consolidation Recommendations
1. `Keep`
   - (none)
2. `Harden`
   - Mainline: `app_dash.py`, `run_full_pipeline_5.py`, `run_oos_analysis.py`, `run_enhanced_ensemble.py`, `run_batch_smart_leverage.py`
   - Derived: all Round scripts, `run_anti_overfit_full_period_test.py`, `run_beta_test.py`, `predict_tomorrow.py`
3. `Merge`
   - `ensemble_wrapper.py` -> fold supported flows into a single maintained CLI (suggest: `run_enhanced_ensemble.py` subcommands or `runners/` entrypoint), then leave wrapper as deprecation shim.
4. `Deprecate` (after merge path exists)
   - `ensemble_wrapper.py` current demo-style `__main__` execution mode.

## Backward-Compatible Sync Actions (Next Stage)
1. Add argparse contracts to non-CLI official scripts:
   - `run_full_pipeline_5.py`: add `--mode --train_end_date --n_trials --score_mode --dry-run`
   - `run_batch_smart_leverage.py`: add `--target --safe --start_date --top_n --dry-run`
2. Add `--dry-run` to Round scripts and `run_beta_test.py` to validate I/O paths without full backtest compute.
3. Add pytest smoke for commonly used derived scripts:
   - compile + `--help` + minimal dry-run return code checks.
4. Keep deprecation compatibility instead of hard removals.
5. In `app_dash.py`, replace `from SSS_EnsembleTab import trade_contribution_by_phase` with maintained source (`sss_core.trade_diagnostics`) or add a backward-compatible shim export in `SSS_EnsembleTab.py`.

## Notes
- No public API/function signature changes were applied in this audit stage.
- This stage is audit-only with reproducible evidence and decision-ready outputs.
