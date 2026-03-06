from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from SSS_EnsembleTab import (
    CostParams,
    EnsembleParams,
    RunConfig,
    calculate_performance,
    run_ensemble,
)
from analysis.strategy_manager import manager
from sss_core.logic import load_data


@dataclass(frozen=True)
class WalkForwardMode:
    name: str
    top_k: int
    floor: float
    min_trades: int
    active_lookback_days: int
    max_corr: float
    score_kind: str  # "ret_only" or "robust"
    ema_span: int
    delta_cap: float
    min_trade_dw: float


def _load_strategy_names(warehouse_path: Path | None) -> list[str]:
    if warehouse_path and warehouse_path.exists():
        payload = json.loads(warehouse_path.read_text(encoding="utf-8"))
        out = [str(x.get("name", "")).replace(".csv", "") for x in payload.get("strategies", [])]
        out = [x for x in out if x]
        if out:
            return out

    # fallback to active warehouse
    payload = manager.load_strategies()
    out = [str(x.get("name", "")).replace(".csv", "") for x in payload]
    return [x for x in out if x]


def _resolve_trade_files(names: Iterable[str], include_archive: bool) -> dict[str, Path]:
    roots = [Path("sss_backtest_outputs")]
    if include_archive:
        roots.append(Path("archive"))

    file_map: dict[str, Path] = {}
    for name in names:
        for root in roots:
            matches = list(root.rglob(f"*{name}*.csv"))
            if not matches:
                continue
            file_map[name] = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            break
    return file_map


def _load_positions(trade_path: Path, index: pd.DatetimeIndex) -> pd.Series | None:
    try:
        df = pd.read_csv(trade_path)
    except Exception:
        return None

    cols = {str(c).lower().strip(): c for c in df.columns}
    date_col = cols.get("trade_date", cols.get("date"))
    type_col = cols.get("type", cols.get("action"))
    if date_col is None or type_col is None:
        return None

    dt = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    side = df[type_col].astype(str).str.lower().str.strip()

    pos = pd.Series(0.0, index=index, dtype=float)
    for d, s in zip(dt, side):
        if pd.isna(d):
            continue
        loc = index.searchsorted(d)
        if loc >= len(index):
            continue
        if s in ("buy", "entry", "long", "add"):
            pos.iloc[loc:] = 1.0
        elif s in ("sell", "exit", "short", "sell_forced", "forced_sell"):
            pos.iloc[loc:] = 0.0
    return pos


def _compute_metrics(eq: pd.Series, bench: pd.Series) -> dict[str, float]:
    idx = eq.index.intersection(bench.index)
    e = eq.reindex(idx).ffill().dropna()
    b = bench.reindex(idx).ffill().dropna()
    if len(e) < 2 or len(b) < 2:
        return {}

    ret = float(e.iloc[-1] / e.iloc[0] - 1)
    b_ret = float(b.iloc[-1] / b.iloc[0] - 1)
    alpha = ret - b_ret

    dd = e / e.cummax() - 1
    mdd = float(dd.min())

    b_dd = b / b.cummax() - 1
    b_mdd = float(b_dd.min())

    dr = e.pct_change().dropna()
    vol = float(dr.std() * np.sqrt(252)) if len(dr) > 1 else float("nan")
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan")

    years = max((e.index[-1] - e.index[0]).days / 365.25, 1 / 252)
    ann = float((1 + ret) ** (1 / years) - 1) if (1 + ret) > 0 else float("nan")

    return {
        "ret": ret,
        "ann_ret": ann,
        "mdd": mdd,
        "sharpe": sharpe,
        "vol": vol,
        "bh_ret": b_ret,
        "bh_mdd": b_mdd,
        "alpha_vs_bh": alpha,
    }


def _score_strategy(
    train_open: pd.Series,
    train_pos: pd.Series,
    mode: WalkForwardMode,
    cost: CostParams,
) -> tuple[float, dict[str, float]]:
    eq, trades, _, _ = calculate_performance(train_open, train_pos, cost)
    if len(eq) < 2:
        return float("-inf"), {}

    ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    dr = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

    buy_count = int((train_pos.diff() > 0).sum())
    trade_count = int(len(trades))

    if trade_count < mode.min_trades:
        return float("-inf"), {
            "ret": ret,
            "mdd": mdd,
            "sharpe": sharpe,
            "buy_count": buy_count,
            "trade_count": trade_count,
        }

    if mode.active_lookback_days > 0:
        recent = train_pos.iloc[-mode.active_lookback_days :]
        recent_buy = int((recent.diff() > 0).sum())
        if recent_buy <= 0:
            return float("-inf"), {
                "ret": ret,
                "mdd": mdd,
                "sharpe": sharpe,
                "buy_count": buy_count,
                "trade_count": trade_count,
            }

    if mode.score_kind == "ret_only":
        score = ret
    else:
        # Robust score: reward return/sharpe, penalize drawdown.
        score = ret - 0.8 * abs(mdd) + 0.05 * sharpe

    return score, {
        "ret": ret,
        "mdd": mdd,
        "sharpe": sharpe,
        "buy_count": buy_count,
        "trade_count": trade_count,
    }


def _smooth_weights(
    w: pd.Series,
    ema_span: int,
    delta_cap: float,
    min_trade_dw: float,
) -> pd.Series:
    if w.empty:
        return w

    out = w.copy()
    if ema_span > 1:
        out = out.ewm(span=ema_span, adjust=False).mean()

    capped = out.copy()
    for i in range(1, len(capped)):
        prev = float(capped.iloc[i - 1])
        curr = float(capped.iloc[i])
        delta = curr - prev
        if abs(delta) > delta_cap:
            capped.iloc[i] = prev + (delta_cap if delta > 0 else -delta_cap)

    # dead-band for small changes
    final = capped.copy()
    for i in range(1, len(final)):
        prev = float(final.iloc[i - 1])
        curr = float(final.iloc[i])
        if abs(curr - prev) < min_trade_dw:
            final.iloc[i] = prev
    return final.clip(0, 1)


def _month_start_indices(index: pd.DatetimeIndex) -> list[int]:
    if len(index) == 0:
        return []
    out = [0]
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month or index[i].year != index[i - 1].year:
            out.append(i)
    return out


def _build_walkforward_weights(
    open_px: pd.Series,
    pos_df: pd.DataFrame,
    mode: WalkForwardMode,
    train_days: int,
    cost: CostParams,
) -> tuple[pd.Series, pd.DataFrame]:
    idx = open_px.index
    month_starts = _month_start_indices(idx)
    month_starts = [i for i in month_starts if i >= train_days]
    if not month_starts:
        w = pd.Series(mode.floor, index=idx, dtype=float)
        return w, pd.DataFrame()

    w = pd.Series(0.0, index=idx, dtype=float)
    logs: list[dict[str, object]] = []

    for k, start_i in enumerate(month_starts):
        end_i = month_starts[k + 1] if k + 1 < len(month_starts) else len(idx)
        train_slice = slice(start_i - train_days, start_i)

        train_open = open_px.iloc[train_slice]
        train_pos_df = pos_df.iloc[train_slice]

        cands: list[tuple[str, float]] = []
        stats_map: dict[str, dict[str, float]] = {}
        for name in train_pos_df.columns:
            train_pos = train_pos_df[name]
            if train_pos.std() == 0:
                continue
            score, st = _score_strategy(train_open, train_pos, mode, cost)
            if score == float("-inf"):
                continue
            cands.append((name, score))
            stats_map[name] = st

        cands.sort(key=lambda x: x[1], reverse=True)
        selected: list[str] = []
        for name, _ in cands:
            if len(selected) >= mode.top_k:
                break
            if mode.max_corr < 0.999 and selected:
                corr_ok = True
                for s in selected:
                    corr = train_pos_df[name].corr(train_pos_df[s])
                    if pd.notna(corr) and corr > mode.max_corr:
                        corr_ok = False
                        break
                if not corr_ok:
                    continue
            selected.append(name)

        if not selected and cands:
            selected = [cands[0][0]]

        seg_idx = idx[start_i:end_i]
        if selected:
            raw = pos_df[selected].loc[seg_idx].mean(axis=1)
            seg_w = mode.floor + (1 - mode.floor) * raw
        else:
            seg_w = pd.Series(mode.floor, index=seg_idx, dtype=float)
        w.loc[seg_idx] = seg_w.values

        logs.append(
            {
                "rebalance_date": str(idx[start_i].date()),
                "mode": mode.name,
                "num_candidates": len(cands),
                "num_selected": len(selected),
                "selected": "|".join(selected),
            }
        )

    # no look-ahead: before first eligible rebalance, keep floor exposure only
    first_start = month_starts[0]
    w.iloc[:first_start] = mode.floor

    w = _smooth_weights(
        w=w,
        ema_span=mode.ema_span,
        delta_cap=mode.delta_cap,
        min_trade_dw=mode.min_trade_dw,
    )
    return w, pd.DataFrame(logs)


def _run_baseline_current_proportional(
    ticker: str,
    strategy_names: list[str],
    file_map: dict[str, Path],
    cost: CostParams,
) -> tuple[pd.Series, pd.Series]:
    params = EnsembleParams(
        floor=0.5,
        ema_span=3,
        delta_cap=0.1,
        min_cooldown_days=1,
        min_trade_dw=0.01,
    )
    cfg = RunConfig(
        ticker=ticker,
        method="proportional",
        strategies=strategy_names,
        file_map=file_map,
        params=params,
        cost=cost,
    )
    open_px, w, _, _, _, equity, _, _ = run_ensemble(cfg)
    if equity is None or equity.empty:
        raise RuntimeError("Failed to run baseline current proportional.")
    return equity, w


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-period anti-overfitting walk-forward test.")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--warehouse", default="analysis/strategy_warehouse.json")
    parser.add_argument("--split_date", default="2025-06-30")
    parser.add_argument("--train_days", type=int, default=504)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--include_archive", action="store_true")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    split_date = pd.Timestamp(args.split_date)
    warehouse_path = Path(args.warehouse)

    strategy_names = _load_strategy_names(warehouse_path)
    if not strategy_names:
        raise SystemExit("No strategies loaded from warehouse.")

    file_map = _resolve_trade_files(strategy_names, include_archive=args.include_archive)
    strategy_names = [x for x in strategy_names if x in file_map]
    if len(strategy_names) < 2:
        raise SystemExit("Need at least 2 strategy files for anti-overfit test.")

    df_price, _ = load_data(args.ticker)
    if df_price.empty or "open" not in df_price.columns or "close" not in df_price.columns:
        raise SystemExit(f"Price data unavailable for {args.ticker}.")

    open_px = pd.to_numeric(df_price["open"], errors="coerce").dropna().sort_index()
    close_px = pd.to_numeric(df_price["close"], errors="coerce").dropna().sort_index()
    idx = open_px.index.intersection(close_px.index)
    open_px = open_px.reindex(idx).dropna()
    close_px = close_px.reindex(idx).dropna()

    pos_map: dict[str, pd.Series] = {}
    for n in strategy_names:
        p = _load_positions(file_map[n], open_px.index)
        if p is None or p.std() == 0:
            continue
        pos_map[n] = p.reindex(open_px.index).ffill().fillna(0.0)
    if len(pos_map) < 2:
        raise SystemExit("Not enough valid non-zero position strategies.")
    pos_df = pd.DataFrame(pos_map, index=open_px.index)

    print(f"Ticker: {args.ticker}")
    print(f"Strategies loaded: {len(strategy_names)} | usable: {pos_df.shape[1]}")
    print(f"Backtest range: {open_px.index[0].date()} -> {open_px.index[-1].date()} ({len(open_px)} bars)")
    print(f"Walk-forward train_days: {args.train_days}, split_date: {split_date.date()}")

    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)

    # Baseline (current style)
    base_eq, base_w = _run_baseline_current_proportional(args.ticker, list(pos_df.columns), file_map, cost)

    # Naive walk-forward: ret-only selector
    naive_mode = WalkForwardMode(
        name="walkforward_naive",
        top_k=args.top_k,
        floor=0.20,
        min_trades=0,
        active_lookback_days=0,
        max_corr=0.999,
        score_kind="ret_only",
        ema_span=3,
        delta_cap=0.20,
        min_trade_dw=0.005,
    )
    naive_w, naive_sel = _build_walkforward_weights(open_px, pos_df, naive_mode, args.train_days, cost)
    naive_eq, _, _, _ = calculate_performance(open_px, naive_w, cost)

    # Robust walk-forward: anti-overfitting selector
    robust_mode = WalkForwardMode(
        name="walkforward_anti_overfit",
        top_k=args.top_k,
        floor=0.20,
        min_trades=8,
        active_lookback_days=126,
        max_corr=0.90,
        score_kind="robust",
        ema_span=3,
        delta_cap=0.15,
        min_trade_dw=0.01,
    )
    robust_w, robust_sel = _build_walkforward_weights(open_px, pos_df, robust_mode, args.train_days, cost)
    robust_eq, _, _, _ = calculate_performance(open_px, robust_w, cost)

    # Benchmark curve (buy-and-hold on close)
    bh_eq = close_px / close_px.iloc[0]
    bh_eq = bh_eq * 1_000_000.0

    # Metrics
    summary_rows: list[dict[str, object]] = []
    for name, eq, w in [
        ("baseline_current_proportional", base_eq, base_w),
        ("walkforward_naive", naive_eq, naive_w),
        ("walkforward_anti_overfit", robust_eq, robust_w),
    ]:
        full = _compute_metrics(eq, bh_eq)
        oos = _compute_metrics(eq[eq.index > split_date], bh_eq[bh_eq.index > split_date])
        row = {"strategy": name}
        row.update({f"full_{k}": v for k, v in full.items()})
        row.update({f"oos_{k}": v for k, v in oos.items()})
        row["time_in_market"] = float((w > 0.05).mean()) if len(w) else float("nan")
        row["turnover_py"] = float(w.diff().abs().sum() / len(w) * 252) if len(w) else float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="oos_alpha_vs_bh", ascending=False)

    # Save artifacts
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = out_dir / f"anti_overfit_fullperiod_summary_{ts}.csv"
    weights_path = out_dir / f"anti_overfit_fullperiod_weights_{ts}.csv"
    sel_path = out_dir / f"anti_overfit_fullperiod_selections_{ts}.csv"
    equity_path = out_dir / f"anti_overfit_fullperiod_equity_{ts}.csv"

    summary_df.to_csv(summary_path, index=False)

    weights_df = pd.DataFrame(
        {
            "baseline_w": base_w.reindex(open_px.index).ffill().fillna(0.0),
            "naive_w": naive_w.reindex(open_px.index).ffill().fillna(0.0),
            "anti_overfit_w": robust_w.reindex(open_px.index).ffill().fillna(0.0),
        },
        index=open_px.index,
    )
    weights_df.to_csv(weights_path, index_label="date")

    eq_df = pd.DataFrame(
        {
            "baseline_equity": base_eq.reindex(open_px.index).ffill(),
            "naive_equity": naive_eq.reindex(open_px.index).ffill(),
            "anti_overfit_equity": robust_eq.reindex(open_px.index).ffill(),
            "bh_equity": bh_eq.reindex(open_px.index).ffill(),
        },
        index=open_px.index,
    )
    eq_df.to_csv(equity_path, index_label="date")

    sel_df = pd.concat([naive_sel, robust_sel], ignore_index=True) if not naive_sel.empty or not robust_sel.empty else pd.DataFrame()
    sel_df.to_csv(sel_path, index=False)

    # Console summary
    show_cols = [
        "strategy",
        "full_ret",
        "full_mdd",
        "full_alpha_vs_bh",
        "oos_ret",
        "oos_mdd",
        "oos_alpha_vs_bh",
        "oos_sharpe",
        "time_in_market",
        "turnover_py",
    ]
    view = summary_df[show_cols].copy()
    for c in [
        "full_ret",
        "full_mdd",
        "full_alpha_vs_bh",
        "oos_ret",
        "oos_mdd",
        "oos_alpha_vs_bh",
        "time_in_market",
    ]:
        view[c] = view[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "nan")
    view["oos_sharpe"] = view["oos_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
    view["turnover_py"] = view["turnover_py"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")

    print("\nFull-period anti-overfitting test result:")
    print(view.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved weights: {weights_path}")
    print(f"Saved equity: {equity_path}")
    print(f"Saved selections: {sel_path}")


if __name__ == "__main__":
    main()
