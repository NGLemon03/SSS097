# Legacy Archive

This directory stores historical code that is intentionally excluded from the active mainline workflow.

## Scope

- `legacy/analysis/old`: older pipeline and experiment snapshots.
- `legacy/analysis/past`: clustering and research history snapshots.
- `legacy/analysis/Optuna`: archived Optuna version history.

## Policy

- Treat files here as read-only historical records.
- Do not import legacy modules from mainline runtime code.
- Default `ruff` and `pytest` settings exclude this directory.
- Mainline Optuna entrypoint is `analysis/optuna_16.py`.

## Traceability

See `legacy/manifest.json` for source-to-destination mapping and archive metadata.
