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
class Round3Config:
    top_k: int = 5
    train_days: int = 504
    min_trades: int = 8
    active_lookback_days: int = 126
    max_corr: float = 0.90
    ema_span: int = 3
    delta_cap: float = 0.12
    min_trade_dw: float = 0.01


REGIME_BOUNDS = {
    "risk_on": (0.45, 1.00),
    "neutral": (0.30, 0.75),
    "risk_off": (0.10, 0.40),
}


def _load_strategy_names(warehouse_path: Path | None) -> list[str]:
    if warehouse_path and warehouse_path.exists():
        try:
            payload = json.loads(warehouse_path.read_text(encoding="utf-8"))
            out = [str(x.get("name", "")).replace(".csv", "") for x in payload.get("strategies", [])]
            out = [x for x in out if x]
            if out:
                return out
        except Exception:
            pass

    payload = manager.load_strategies()
    out = [str(x.get("name", "")).replace(".csv", "") for x in payload]
    return [x for x in out if x]


def _resolve_trade_files(names: Iterable[str], include_archive: bool) -> dict[str, Path]:
    roots = [Path("sss_backtest_outputs")]
    if include_archive:
        roots.append(Path("archive"))

    out: dict[str, Path] = {}
    for name in names:
        for root in roots:
            matches = list(root.rglob(f"*{name}*.csv"))
            if not matches:
                continue
            out[name] = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            break
    return out


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


def _month_start_indices(index: pd.DatetimeIndex) -> list[int]:
    if len(index) == 0:
        return []
    out = [0]
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month or index[i].year != index[i - 1].year:
            out.append(i)
    return out


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
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan")
    vol = float(dr.std() * np.sqrt(252)) if len(dr) > 1 else float("nan")

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


def _build_regime(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    ma120 = close.rolling(120).mean()
    ma60 = close.rolling(60).mean()
    ma60_slope = ma60.diff(20) / 20.0

    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean()
    atr60 = tr.rolling(60).mean()
    atr_ratio = atr20 / atr60.replace(0, np.nan)

    drawdown = close / close.cummax() - 1

    raw_on = (close > ma120) & (ma60_slope > 0) & (atr_ratio < 1.25)
    raw_off = ((close < ma120) & (ma60_slope < 0)) | (drawdown < -0.08)
    raw_on = raw_on.fillna(False)
    raw_off = raw_off.fillna(False)

    state = "neutral"
    off_streak = 0
    calm_streak = 0
    states: list[str] = []

    for dt in close.index:
        if bool(raw_off.loc[dt]):
            off_streak += 1
            calm_streak = 0
        else:
            off_streak = 0
            if state == "risk_off":
                calm_streak += 1

        if state != "risk_off" and off_streak >= 2:
            state = "risk_off"
            calm_streak = 0
        elif state == "risk_off":
            if bool(raw_off.loc[dt]):
                pass
            elif calm_streak >= 5:
                state = "risk_on" if bool(raw_on.loc[dt]) else "neutral"
                calm_streak = 0
        else:
            if bool(raw_on.loc[dt]):
                state = "risk_on"
            elif not bool(raw_off.loc[dt]):
                state = "neutral"

        states.append(state)

    regime = pd.Series(states, index=close.index, name="regime")
    return pd.DataFrame(
        {
            "regime": regime,
            "raw_on": raw_on,
            "raw_off": raw_off,
            "atr_ratio": atr_ratio,
            "drawdown": drawdown,
        },
        index=close.index,
    )


def _score_candidate(
    train_open: pd.Series,
    train_close: pd.Series,
    train_pos: pd.Series,
    cfg: Round3Config,
    cost: CostParams,
) -> tuple[float, dict[str, float]]:
    if train_pos.std() == 0:
        return float("-inf"), {}

    eq, trades, _, _ = calculate_performance(train_open, train_pos, cost)
    if len(eq) < 63:
        return float("-inf"), {}

    trade_count = int(len(trades))
    if trade_count < cfg.min_trades:
        return float("-inf"), {"trade_count": trade_count}

    recent = train_pos.iloc[-cfg.active_lookback_days :]
    recent_buy = int((recent.diff() > 0).sum())
    if recent_buy <= 0:
        return float("-inf"), {"trade_count": trade_count, "recent_buy": recent_buy}

    strat_ret = eq.pct_change().dropna()
    bench_ret = train_close.reindex(eq.index).ffill().pct_change().dropna()
    common = strat_ret.index.intersection(bench_ret.index)
    strat_ret = strat_ret.reindex(common).dropna()
    bench_ret = bench_ret.reindex(common).dropna()
    if len(common) < 63:
        return float("-inf"), {"trade_count": trade_count, "recent_buy": recent_buy}

    roll_strat = (1 + strat_ret).rolling(63).apply(np.prod, raw=True) - 1
    roll_bench = (1 + bench_ret).rolling(63).apply(np.prod, raw=True) - 1
    roll_alpha = (roll_strat - roll_bench).dropna()

    total_alpha = float(eq.iloc[-1] / eq.iloc[0] - 1) - float(
        train_close.iloc[-1] / train_close.iloc[0] - 1
    )
    median_alpha = float(roll_alpha.median()) if not roll_alpha.empty else total_alpha

    mdd = float((eq / eq.cummax() - 1).min())
    turnover_py = float(train_pos.diff().abs().sum() / len(train_pos) * 252)
    turnover_pen = turnover_py / 20.0

    score = median_alpha - 0.8 * abs(mdd) - 0.2 * turnover_pen
    return score, {
        "median_alpha_63d": median_alpha,
        "mdd": mdd,
        "turnover_py": turnover_py,
        "trade_count": trade_count,
        "recent_buy": recent_buy,
    }


def _smooth_weights(w: pd.Series, cfg: Round3Config) -> pd.Series:
    out = w.copy()
    if cfg.ema_span > 1:
        out = out.ewm(span=cfg.ema_span, adjust=False).mean()

    capped = out.copy()
    for i in range(1, len(capped)):
        prev = float(capped.iloc[i - 1])
        curr = float(capped.iloc[i])
        delta = curr - prev
        if abs(delta) > cfg.delta_cap:
            capped.iloc[i] = prev + (cfg.delta_cap if delta > 0 else -cfg.delta_cap)

    final = capped.copy()
    for i in range(1, len(final)):
        prev = float(final.iloc[i - 1])
        curr = float(final.iloc[i])
        if abs(curr - prev) < cfg.min_trade_dw:
            final.iloc[i] = prev

    return final.clip(0, 1)


def _bounds_for_regime(regime: str) -> tuple[float, float]:
    return REGIME_BOUNDS.get(regime, REGIME_BOUNDS["neutral"])


def _build_round3_weights(
    open_px: pd.Series,
    close_px: pd.Series,
    pos_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    cfg: Round3Config,
    cost: CostParams,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    idx = open_px.index
    month_starts = [i for i in _month_start_indices(idx) if i >= cfg.train_days]

    w = pd.Series(0.0, index=idx, dtype=float)
    raw_all = pos_df.mean(axis=1).reindex(idx).ffill().fillna(0.0)

    selection_logs: list[dict[str, object]] = []
    score_logs: list[dict[str, object]] = []

    for k, start_i in enumerate(month_starts):
        end_i = month_starts[k + 1] if k + 1 < len(month_starts) else len(idx)
        seg_idx = idx[start_i:end_i]

        train_slice = slice(start_i - cfg.train_days, start_i)
        train_open = open_px.iloc[train_slice]
        train_close = close_px.iloc[train_slice]
        train_pos_df = pos_df.iloc[train_slice]

        cands: list[tuple[str, float]] = []
        stat_map: dict[str, dict[str, float]] = {}

        for name in train_pos_df.columns:
            score, st = _score_candidate(
                train_open=train_open,
                train_close=train_close,
                train_pos=train_pos_df[name],
                cfg=cfg,
                cost=cost,
            )
            if score == float("-inf"):
                continue
            cands.append((name, score))
            stat_map[name] = st
            score_logs.append(
                {
                    "rebalance_date": str(idx[start_i].date()),
                    "strategy": name,
                    "score": score,
                    **st,
                }
            )

        cands.sort(key=lambda x: x[1], reverse=True)
        selected: list[str] = []
        for name, _ in cands:
            if len(selected) >= cfg.top_k:
                break
            corr_ok = True
            for s in selected:
                corr = train_pos_df[name].corr(train_pos_df[s])
                if pd.notna(corr) and corr > cfg.max_corr:
                    corr_ok = False
                    break
            if not corr_ok:
                continue
            selected.append(name)

        if selected:
            raw_seg = pos_df[selected].loc[seg_idx].mean(axis=1)
        else:
            raw_seg = raw_all.loc[seg_idx]

        regime_seg = regime_df["regime"].reindex(seg_idx).fillna("neutral")
        floor_seg = regime_seg.map(lambda x: _bounds_for_regime(str(x))[0]).astype(float)
        cap_seg = regime_seg.map(lambda x: _bounds_for_regime(str(x))[1]).astype(float)
        seg_w = floor_seg + (cap_seg - floor_seg) * raw_seg.clip(0, 1)
        w.loc[seg_idx] = seg_w.values

        selection_logs.append(
            {
                "rebalance_date": str(idx[start_i].date()),
                "num_candidates": len(cands),
                "num_selected": len(selected),
                "selected": "|".join(selected),
            }
        )

    # Before first train-ready rebalance, use all-strategy average under regime caps.
    pre_end = month_starts[0] if month_starts else len(idx)
    pre_idx = idx[:pre_end]
    pre_regime = regime_df["regime"].reindex(pre_idx).fillna("neutral")
    pre_floor = pre_regime.map(lambda x: _bounds_for_regime(str(x))[0]).astype(float)
    pre_cap = pre_regime.map(lambda x: _bounds_for_regime(str(x))[1]).astype(float)
    pre_w = pre_floor + (pre_cap - pre_floor) * raw_all.loc[pre_idx].clip(0, 1)
    w.loc[pre_idx] = pre_w.values

    w = _smooth_weights(w, cfg)
    return w, pd.DataFrame(selection_logs), pd.DataFrame(score_logs)


def _run_baseline(
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
    _, w, _, _, _, eq, _, _ = run_ensemble(cfg)
    if eq is None or eq.empty:
        raise RuntimeError("Baseline run failed.")
    return eq, w


def main() -> None:
    parser = argparse.ArgumentParser(description="Round3: Regime + anti-overfitting full-period test.")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--warehouse", default="analysis/strategy_warehouse.json")
    parser.add_argument("--split_date", default="2025-06-30")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--train_days", type=int, default=504)
    parser.add_argument("--min_trades", type=int, default=8)
    parser.add_argument("--active_lookback_days", type=int, default=126)
    parser.add_argument("--max_corr", type=float, default=0.90)
    parser.add_argument("--ema_span", type=int, default=3)
    parser.add_argument("--delta_cap", type=float, default=0.12)
    parser.add_argument("--min_trade_dw", type=float, default=0.01)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--include_archive", action="store_true")
    args = parser.parse_args()

    split_date = pd.Timestamp(args.split_date)
    cfg = Round3Config(
        top_k=args.top_k,
        train_days=args.train_days,
        min_trades=args.min_trades,
        active_lookback_days=args.active_lookback_days,
        max_corr=args.max_corr,
        ema_span=args.ema_span,
        delta_cap=args.delta_cap,
        min_trade_dw=args.min_trade_dw,
    )

    names = _load_strategy_names(Path(args.warehouse))
    file_map = _resolve_trade_files(names, include_archive=args.include_archive)
    names = [n for n in names if n in file_map]
    if len(names) < 2:
        raise SystemExit("Need at least 2 strategy files.")

    df_price, _ = load_data(args.ticker)
    if df_price.empty:
        raise SystemExit(f"No price data for {args.ticker}.")

    required = {"open", "high", "low", "close"}
    missing = required - set(df_price.columns)
    if missing:
        raise SystemExit(f"Missing required price columns: {missing}")

    open_px = pd.to_numeric(df_price["open"], errors="coerce").dropna().sort_index()
    high_px = pd.to_numeric(df_price["high"], errors="coerce").dropna().sort_index()
    low_px = pd.to_numeric(df_price["low"], errors="coerce").dropna().sort_index()
    close_px = pd.to_numeric(df_price["close"], errors="coerce").dropna().sort_index()
    idx = open_px.index.intersection(high_px.index).intersection(low_px.index).intersection(close_px.index)
    open_px = open_px.reindex(idx).dropna()
    high_px = high_px.reindex(idx).dropna()
    low_px = low_px.reindex(idx).dropna()
    close_px = close_px.reindex(idx).dropna()

    pos_map: dict[str, pd.Series] = {}
    for n in names:
        s = _load_positions(file_map[n], open_px.index)
        if s is None or s.std() == 0:
            continue
        pos_map[n] = s.reindex(open_px.index).ffill().fillna(0.0)

    if len(pos_map) < 2:
        raise SystemExit("Not enough valid non-zero position strategies.")
    pos_df = pd.DataFrame(pos_map, index=open_px.index)

    regime_df = _build_regime(close_px, high_px, low_px)
    risk_off_pct = float((regime_df["regime"] == "risk_off").mean())
    risk_on_pct = float((regime_df["regime"] == "risk_on").mean())

    print(f"Ticker: {args.ticker}")
    print(f"Strategies loaded: {len(names)} | usable: {pos_df.shape[1]}")
    print(f"Backtest range: {open_px.index[0].date()} -> {open_px.index[-1].date()} ({len(open_px)} bars)")
    print(f"Split date: {split_date.date()}")
    print(f"Regime share: risk_on={risk_on_pct:.2%}, risk_off={risk_off_pct:.2%}")

    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)
    baseline_eq, baseline_w = _run_baseline(args.ticker, list(pos_df.columns), file_map, cost)

    r3_w, sel_df, score_df = _build_round3_weights(
        open_px=open_px,
        close_px=close_px,
        pos_df=pos_df,
        regime_df=regime_df,
        cfg=cfg,
        cost=cost,
    )
    r3_eq, _, _, _ = calculate_performance(open_px, r3_w, cost)

    bh_eq = close_px / close_px.iloc[0] * 1_000_000.0

    summary_rows: list[dict[str, object]] = []
    for name, eq, w in [
        ("baseline_current_proportional", baseline_eq, baseline_w),
        ("round3_regime_anti_overfit", r3_eq, r3_w),
    ]:
        full = _compute_metrics(eq, bh_eq)
        oos = _compute_metrics(eq[eq.index > split_date], bh_eq[bh_eq.index > split_date])
        row = {"strategy": name}
        row.update({f"full_{k}": v for k, v in full.items()})
        row.update({f"oos_{k}": v for k, v in oos.items()})
        row["time_in_market"] = float((w > 0.05).mean())
        row["turnover_py"] = float(w.diff().abs().sum() / len(w) * 252)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["risk_on_pct"] = risk_on_pct
    summary_df["risk_off_pct"] = risk_off_pct
    summary_df["selected_avg"] = float(sel_df["num_selected"].mean()) if not sel_df.empty else float("nan")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = out_dir / f"round3_regime_summary_{ts}.csv"
    weights_path = out_dir / f"round3_regime_weights_{ts}.csv"
    equity_path = out_dir / f"round3_regime_equity_{ts}.csv"
    sel_path = out_dir / f"round3_regime_selections_{ts}.csv"
    score_path = out_dir / f"round3_regime_scores_{ts}.csv"
    regime_path = out_dir / f"round3_regime_states_{ts}.csv"

    summary_df.to_csv(summary_path, index=False)

    weights_df = pd.DataFrame(
        {
            "baseline_w": baseline_w.reindex(open_px.index).ffill().fillna(0.0),
            "round3_w": r3_w.reindex(open_px.index).ffill().fillna(0.0),
        },
        index=open_px.index,
    )
    weights_df.to_csv(weights_path, index_label="date")

    eq_df = pd.DataFrame(
        {
            "baseline_equity": baseline_eq.reindex(open_px.index).ffill(),
            "round3_equity": r3_eq.reindex(open_px.index).ffill(),
            "bh_equity": bh_eq.reindex(open_px.index).ffill(),
        },
        index=open_px.index,
    )
    eq_df.to_csv(equity_path, index_label="date")

    sel_df.to_csv(sel_path, index=False)
    score_df.to_csv(score_path, index=False)
    regime_df.to_csv(regime_path, index_label="date")

    view = summary_df[
        [
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
            "risk_on_pct",
            "risk_off_pct",
            "selected_avg",
        ]
    ].copy()

    pct_cols = [
        "full_ret",
        "full_mdd",
        "full_alpha_vs_bh",
        "oos_ret",
        "oos_mdd",
        "oos_alpha_vs_bh",
        "time_in_market",
        "risk_on_pct",
        "risk_off_pct",
    ]
    for c in pct_cols:
        view[c] = view[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "nan")
    view["oos_sharpe"] = view["oos_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
    view["turnover_py"] = view["turnover_py"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
    view["selected_avg"] = view["selected_avg"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")

    print("\nRound3 full-period result:")
    print(view.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved weights: {weights_path}")
    print(f"Saved equity: {equity_path}")
    print(f"Saved selections: {sel_path}")
    print(f"Saved candidate scores: {score_path}")
    print(f"Saved regime states: {regime_path}")


if __name__ == "__main__":
    main()
