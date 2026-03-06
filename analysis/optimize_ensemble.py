# analysis/optimize_ensemble.py
# -*- coding: utf-8 -*-
"""
SSS096 ensemble optimizer.

P0 fixes in this version:
- Use fixed benchmark index for position reconstruction (reproducible runs).
- Enforce trade schema normalization with observable failure accounting.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path bootstrap for standalone script execution.
sys.path.append(os.getcwd())
try:
    from sss_core.logic import load_data
    from SSS_EnsembleTab import CostParams, EnsembleParams, RunConfig, run_ensemble
except ImportError:
    print("Failed to import project modules. Please run from project root.")
    sys.exit(1)

# Constants
RESULTS_DIR = Path("results")
TRADES_DIR = Path("sss_backtest_outputs")
RECOMMEND_FILE = Path("analysis/recommended_strategies.json")
TICKER = "00631L.TW"
TEST_K_LIST = [3, 5, 7, 10]
MAX_CORRELATION = 0.95

DATE_ALIASES = ("trade_date", "date", "signal_date")
TYPE_ALIASES = ("type", "action", "side")
TYPE_MAP = {
    "buy": "buy",
    "entry": "buy",
    "long": "buy",
    "add": "buy",
    "sell": "sell",
    "exit": "sell",
    "sell_forced": "sell",
    "forced_sell": "sell",
    "short": "sell",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Optimizer")


def _new_parse_stats(total_files: int = 0) -> dict[str, object]:
    return {
        "total_files": int(total_files),
        "parsed_ok": 0,
        "missing_type_col": 0,
        "missing_date_col": 0,
        "invalid_rows": 0,
        "skipped_candidates": 0,
        "reasons_by_strategy": {},
    }


def _record_failure(
    stats: dict[str, object] | None,
    strategy_id: str,
    reason: str,
    *,
    count_as_skip: bool = True,
) -> None:
    if stats is None:
        return
    reasons = stats.setdefault("reasons_by_strategy", {})
    reasons[strategy_id] = reason
    if count_as_skip:
        stats["skipped_candidates"] = int(stats.get("skipped_candidates", 0)) + 1


def _normalize_trade_schema(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None, int]:
    if df is None or df.empty:
        return None, "empty_trade_file", 0

    cols = {str(c).lower().strip(): c for c in df.columns}
    date_col = next((cols[a] for a in DATE_ALIASES if a in cols), None)
    type_col = next((cols[a] for a in TYPE_ALIASES if a in cols), None)

    if date_col is None:
        return None, "missing_date_col", 0
    if type_col is None:
        return None, "missing_type_col", 0

    out = pd.DataFrame()
    out["trade_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    out["type"] = (
        df[type_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(TYPE_MAP)
    )

    invalid_mask = out["trade_date"].isna() | out["type"].isna()
    invalid_rows = int(invalid_mask.sum())
    out = out.loc[~invalid_mask, ["trade_date", "type"]].sort_values("trade_date")

    if out.empty:
        return None, "no_valid_rows", invalid_rows
    return out, None, invalid_rows


def _build_position_index(df_bench: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.to_datetime(df_bench.index, errors="coerce")
    idx = pd.DatetimeIndex(idx).normalize()
    idx = idx[idx.notna()]
    # Unique + sorted for deterministic alignment
    return pd.DatetimeIndex(pd.Index(idx).unique().sort_values())


def get_strategy_position_series(
    trade_file_path: Path,
    index: pd.DatetimeIndex,
    stats: dict[str, object] | None = None,
    strategy_id: str = "",
) -> pd.Series | None:
    try:
        df = pd.read_csv(trade_file_path)
    except Exception as exc:
        logger.warning("[schema] failed to read trades: %s (%s)", trade_file_path, exc)
        _record_failure(stats, strategy_id or str(trade_file_path), "read_csv_failed")
        return None

    normalized, error_code, invalid_rows = _normalize_trade_schema(df)
    if stats is not None:
        stats["invalid_rows"] = int(stats.get("invalid_rows", 0)) + invalid_rows

    if error_code is not None:
        if stats is not None:
            if error_code == "missing_type_col":
                stats["missing_type_col"] = int(stats.get("missing_type_col", 0)) + 1
            elif error_code == "missing_date_col":
                stats["missing_date_col"] = int(stats.get("missing_date_col", 0)) + 1
        logger.warning(
            "[schema] %s for %s (strategy=%s)",
            error_code,
            trade_file_path,
            strategy_id or "unknown",
        )
        _record_failure(stats, strategy_id or str(trade_file_path), error_code)
        return None

    daily_pos = pd.Series(0.0, index=index, dtype=float)
    for row in normalized.itertuples(index=False):
        loc_idx = index.searchsorted(row.trade_date)
        if loc_idx >= len(index):
            continue
        if row.type == "buy":
            daily_pos.iloc[loc_idx:] = 1.0
        elif row.type == "sell":
            daily_pos.iloc[loc_idx:] = 0.0

    if stats is not None:
        stats["parsed_ok"] = int(stats.get("parsed_ok", 0)) + 1
    return daily_pos


def select_diverse_strategies_by_group(
    candidates: list[dict[str, object]],
    k: int,
    file_map: dict[str, Path],
    position_index: pd.DatetimeIndex,
) -> list[dict[str, object]]:
    """Round-robin by strategy type, then filter by position correlation."""
    groups: dict[str, list[dict[str, object]]] = {}
    for cand in candidates:
        stype = str(cand["type"])
        groups.setdefault(stype, []).append(cand)

    if not groups:
        return []

    pool: list[dict[str, object]] = []
    max_len = max(len(v) for v in groups.values())
    for i in range(max_len):
        for stype in groups:
            if i < len(groups[stype]):
                pool.append(groups[stype][i])

    unique_files = {str(p.resolve()) for p in file_map.values()}
    parse_stats = _new_parse_stats(total_files=len(unique_files))
    selected_cands: list[dict[str, object]] = []
    selected_positions: list[pd.Series] = []

    for cand in pool:
        if len(selected_cands) >= k:
            break

        strategy_id = str(cand.get("id") or cand.get("full_id") or "unknown")
        fpath = file_map.get(str(cand["id"])) or file_map.get(str(cand.get("full_id")))
        if not fpath:
            logger.warning("[selection] missing trade file for %s", strategy_id)
            _record_failure(parse_stats, strategy_id, "missing_trade_file")
            continue

        pos = get_strategy_position_series(
            fpath,
            position_index,
            stats=parse_stats,
            strategy_id=strategy_id,
        )
        if pos is None:
            continue
        if pos.std() == 0:
            logger.warning("[selection] zero-variance position series: %s", strategy_id)
            _record_failure(parse_stats, strategy_id, "zero_variance_position")
            continue

        is_duplicate = False
        for existing_pos in selected_positions:
            if pos.corr(existing_pos) > MAX_CORRELATION:
                is_duplicate = True
                break

        if is_duplicate:
            _record_failure(parse_stats, strategy_id, "high_correlation")
            continue

        selected_cands.append(cand)
        selected_positions.append(pos)

    logger.info(
        "[schema] total_files=%d parsed_ok=%d missing_type_col=%d missing_date_col=%d invalid_rows=%d skipped_candidates=%d",
        int(parse_stats["total_files"]),
        int(parse_stats["parsed_ok"]),
        int(parse_stats["missing_type_col"]),
        int(parse_stats["missing_date_col"]),
        int(parse_stats["invalid_rows"]),
        int(parse_stats["skipped_candidates"]),
    )
    return selected_cands


def load_all_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    files = sorted(RESULTS_DIR.glob("**/*.csv"), key=lambda p: str(p))
    logger.info("Scanning %d Optuna result files...", len(files))

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            logger.warning("Skip unreadable result file %s: %s", f.name, exc)
            continue

        if "value" not in df.columns or "parameters" not in df.columns:
            continue

        fname = f.name
        stype = "Unknown"
        if "RMA" in fname:
            stype = "RMA"
        elif "ssma_turn" in fname:
            stype = "ssma_turn"
        elif "single" in fname:
            stype = "single"

        extra = ""
        if "Factor" in fname:
            if "2412" in fname:
                extra = "_Factor_TWII_2412"
            elif "2414" in fname:
                extra = "_Factor_TWII_2414"

        for _, row in df.iterrows():
            if float(row["value"]) < 0:
                continue

            try:
                params = ast.literal_eval(row["parameters"])
            except Exception as exc:
                logger.warning(
                    "Skip candidate due to invalid parameters literal (%s, trial=%s): %s",
                    fname,
                    row.get("trial_number"),
                    exc,
                )
                continue

            tid = row["trial_number"]
            candidates.append(
                {
                    "id": f"{stype}_trial{tid}",
                    "full_id": f"{stype}{extra}_trial{tid}",
                    "type": stype,
                    "params": params,
                    "score": row["value"],
                    "source_file": fname,
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _build_trade_file_map() -> dict[str, Path]:
    files = sorted(TRADES_DIR.glob("trades_*.csv"), key=lambda p: p.name)
    file_map: dict[str, Path] = {}
    for f in files:
        name = f.stem.replace("trades_", "")
        file_map[name] = f

        match = re.search(r"(.*)_trial(\d+)", name)
        if not match:
            continue

        base = match.group(1)
        tid = match.group(2)
        stype = "Unknown"
        if name.startswith("RMA"):
            stype = "RMA"
        elif name.startswith("single"):
            stype = "single"
        elif name.startswith("ssma_turn"):
            stype = "ssma_turn"

        file_map[f"{stype}_trial{tid}"] = f
        if "Factor_TWII" in base:
            if "2414" in base:
                file_map[f"{stype}_Factor_TWII_2414_trial{tid}"] = f
            elif "2412" in base:
                file_map[f"{stype}_Factor_TWII_2412_trial{tid}"] = f

    return file_map


def simulate_ensemble(selected_cands: list[dict[str, object]], file_map: dict[str, Path]) -> pd.Series | None:
    strategies_names = []
    for cand in selected_cands:
        fpath = file_map.get(str(cand["id"])) or file_map.get(str(cand.get("full_id")))
        if not fpath:
            continue
        key = fpath.stem.replace("trades_from_results_", "").replace("trades_", "")
        strategies_names.append(key)

    if not strategies_names:
        return None

    ens_params = EnsembleParams(
        floor=0.2,
        ema_span=3,
        delta_cap=0.3,
        majority_k=max(1, int(len(strategies_names) * 0.55)),
        min_cooldown_days=1,
    )
    cost_params = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30)
    cfg = RunConfig(
        ticker=TICKER,
        method="majority",
        strategies=strategies_names,
        params=ens_params,
        cost=cost_params,
        file_map=file_map,
    )

    try:
        _, _, _, _, _, equity, _, _ = run_ensemble(cfg)
        return equity
    except Exception as exc:
        logger.warning("Ensemble simulation failed: %s", exc)
        return None


def analyze_performance_vs_benchmark(
    equity: pd.Series,
    label: str,
    df_bench: pd.DataFrame,
) -> tuple[float, float, float]:
    if equity is None or equity.empty:
        return -999, -999, -999

    start_dt = equity.index[0]
    end_dt = equity.index[-1]
    bench = df_bench.loc[start_dt:end_dt]["close"]
    if bench.empty:
        return -999, -999, -999

    strat_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    bench_ret = (bench.iloc[-1] / bench.iloc[0]) - 1
    excess_ret = strat_ret - bench_ret

    roll_max = equity.cummax()
    mdd = ((equity / roll_max) - 1).min()

    pct = equity.pct_change().dropna()
    sharpe = pct.mean() / pct.std() * (252**0.5) if pct.std() > 0 else 0

    status = "WIN" if excess_ret > 0 else "LOSE"
    print(
        f"{label:<10} | Ret: {strat_ret*100:6.2f}% (B&H: {bench_ret*100:6.2f}%) {status} | "
        f"MDD: {mdd*100:6.2f}% | Sharpe: {sharpe:.2f}"
    )
    return sharpe, excess_ret, mdd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_date", type=str, default="", help="OOS split date")
    parser.add_argument("--mode", type=str, default="", help="run mode (OOS/IS)")
    parser.add_argument("--score_mode", type=str, default="", help="score mode")
    args = parser.parse_args()

    print("SSS096 Ensemble optimizer (v3)")
    if args.split_date:
        print(
            f"Pipeline args: mode={args.mode}, split={args.split_date}, score={args.score_mode}"
        )
    print("=" * 70)

    print(f"Loading benchmark: {TICKER}")
    df_bench, _ = load_data(TICKER)
    position_index = _build_position_index(df_bench)
    if position_index.empty:
        print("No benchmark index available; abort.")
        return

    candidates = load_all_candidates()
    if not candidates:
        print("No valid candidates found. Please run pipeline first.")
        return

    file_map = _build_trade_file_map()
    if not file_map:
        print("No trade files found under sss_backtest_outputs.")
        return

    print("-" * 70)
    print(f"{'Group':<10} | {'Performance (vs B&H)':<35} | {'Risk'}")
    print("-" * 70)

    best_k = 0
    best_score = -999
    best_combination: list[dict[str, object]] = []

    for k in TEST_K_LIST:
        selected_cands = select_diverse_strategies_by_group(
            candidates,
            k,
            file_map,
            position_index,
        )
        if len(selected_cands) < 2:
            print(f"Top {k}: insufficient usable strategies ({len(selected_cands)})")
            continue

        label = f"Top {len(selected_cands)}"
        equity = simulate_ensemble(selected_cands, file_map)
        if equity is None:
            continue

        _sharpe, excess, mdd = analyze_performance_vs_benchmark(equity, label, df_bench)
        score = excess - abs(mdd)
        if score > best_score:
            best_score = score
            best_k = len(selected_cands)
            best_combination = selected_cands

    print("-" * 70)
    print(f"Best combination: Top {best_k}")

    if best_combination:
        save_data = {
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_k": best_k,
            "score": best_score,
            "metadata": {
                "split_date": args.split_date,
                "mode": args.mode,
                "score_mode": args.score_mode,
            },
            "strategies": [],
        }

        for cand in best_combination:
            fpath = file_map.get(str(cand["id"])) or file_map.get(str(cand.get("full_id")))
            if not fpath:
                continue
            save_data["strategies"].append(
                {
                    "name": fpath.stem,
                    "type": cand["type"],
                    "params": cand["params"],
                }
            )

        with open(RECOMMEND_FILE, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4)
        print(f"Saved recommendations to {RECOMMEND_FILE}")

    print("=" * 70)


if __name__ == "__main__":
    main()
