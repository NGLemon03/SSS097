from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from SSS_EnsembleTab import CostParams, EnsembleParams, RunConfig, calculate_performance, run_ensemble
from analysis.strategy_manager import manager
from sss_core.logic import load_data


@dataclass(frozen=True)
class Layer1Config:
    top_k: int = 5
    train_days: int = 504
    min_trades: int = 8
    active_lookback_days: int = 126
    corr_cluster_th: float = 0.90
    cluster_cap: float = 0.40
    floor: float = 0.20
    ema_span: int = 3
    delta_cap: float = 0.12
    min_trade_dw: float = 0.01
    target_vol: float = 0.22


def _load_strategy_names(warehouse_path: Path | None) -> list[str]:
    if warehouse_path and warehouse_path.exists():
        try:
            payload = json.loads(warehouse_path.read_text(encoding="utf-8"))
            names = [str(x.get("name", "")).replace(".csv", "") for x in payload.get("strategies", [])]
            names = [x for x in names if x]
            if names:
                return names
        except Exception:
            pass

    payload = manager.load_strategies()
    names = [str(x.get("name", "")).replace(".csv", "") for x in payload]
    return [x for x in names if x]


def _resolve_trade_files(names: Iterable[str], include_archive: bool) -> dict[str, Path]:
    roots = [Path("sss_backtest_outputs")]
    if include_archive:
        roots.append(Path("archive"))

    out: dict[str, Path] = {}
    for n in names:
        for r in roots:
            matches = list(r.rglob(f"*{n}*.csv"))
            if not matches:
                continue
            out[n] = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            break
    return out


def _load_positions_and_buys(
    trade_path: Path,
    index: pd.DatetimeIndex,
) -> tuple[pd.Series | None, np.ndarray]:
    try:
        df = pd.read_csv(trade_path)
    except Exception:
        return None, np.array([], dtype="datetime64[ns]")

    cols = {str(c).lower().strip(): c for c in df.columns}
    date_col = cols.get("trade_date", cols.get("date"))
    type_col = cols.get("type", cols.get("action"))
    if date_col is None or type_col is None:
        return None, np.array([], dtype="datetime64[ns]")

    dt = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    side = df[type_col].astype(str).str.lower().str.strip()

    pos = pd.Series(0.0, index=index, dtype=float)
    buy_mask = side.isin(["buy", "entry", "long", "add"]) & dt.notna()
    buy_dates = np.sort(dt[buy_mask].values.astype("datetime64[ns]"))

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
    return pos, buy_dates


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

    bdd = b / b.cummax() - 1
    bmdd = float(bdd.min())

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
        "bh_mdd": bmdd,
        "alpha_vs_bh": alpha,
    }


def _apply_t_plus_1(w: pd.Series) -> pd.Series:
    """Strict T+1 execution: signal generated after close, trade next day open."""
    if w.empty:
        return w
    return w.shift(1).fillna(0.0).clip(0, 1)


def _smooth_weights(w: pd.Series, ema_span: int, delta_cap: float, min_trade_dw: float) -> pd.Series:
    out = w.copy()
    if ema_span > 1:
        out = out.ewm(span=ema_span, adjust=False).mean()

    cap = out.copy()
    for i in range(1, len(cap)):
        prev = float(cap.iloc[i - 1])
        curr = float(cap.iloc[i])
        d = curr - prev
        if abs(d) > delta_cap:
            cap.iloc[i] = prev + (delta_cap if d > 0 else -delta_cap)

    final = cap.copy()
    for i in range(1, len(final)):
        prev = float(final.iloc[i - 1])
        curr = float(final.iloc[i])
        if abs(curr - prev) < min_trade_dw:
            final.iloc[i] = prev
    return final.clip(0, 1)


def _rolling_alpha_gate(eq: pd.Series, bench_close: pd.Series, win: int = 63) -> float:
    e = eq.pct_change().dropna()
    b = bench_close.reindex(eq.index).ffill().pct_change().dropna()
    idx = e.index.intersection(b.index)
    e = e.reindex(idx).dropna()
    b = b.reindex(idx).dropna()
    if len(idx) < win:
        return float("-inf")

    e_roll = (1 + e).rolling(win).apply(np.prod, raw=True) - 1
    b_roll = (1 + b).rolling(win).apply(np.prod, raw=True) - 1
    alpha = (e_roll - b_roll).dropna()
    if alpha.empty:
        return float("-inf")
    return float(alpha.median())


def _score_candidate(
    train_open: pd.Series,
    train_close: pd.Series,
    train_pos: pd.Series,
    cfg: Layer1Config,
    cost: CostParams,
) -> tuple[float, dict[str, float]]:
    eq, trades, _, _ = calculate_performance(train_open, train_pos, cost)
    if len(eq) < 80:
        return float("-inf"), {}

    trade_count = int(len(trades))
    if trade_count < cfg.min_trades:
        return float("-inf"), {"trade_count": trade_count}

    ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    dr = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    alpha_med = _rolling_alpha_gate(eq, train_close, win=63)

    if alpha_med <= 0:
        return float("-inf"), {
            "ret": ret,
            "mdd": mdd,
            "sharpe": sharpe,
            "alpha63_med": alpha_med,
            "trade_count": trade_count,
        }

    score = ret - 0.8 * abs(mdd) + 0.05 * sharpe + 0.8 * alpha_med
    return score, {
        "ret": ret,
        "mdd": mdd,
        "sharpe": sharpe,
        "alpha63_med": alpha_med,
        "trade_count": trade_count,
    }


def _days_since_last_buy(buy_dates: np.ndarray, ref_date: pd.Timestamp) -> int | None:
    if buy_dates.size == 0:
        return None
    ref = np.datetime64(ref_date.normalize().to_datetime64())
    i = np.searchsorted(buy_dates, ref, side="right") - 1
    if i < 0:
        return None
    last = pd.Timestamp(buy_dates[i])
    return int((ref_date.normalize() - last.normalize()).days)


def _build_clusters(names: list[str], corr: pd.DataFrame, th: float) -> list[list[str]]:
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            c = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
            if pd.notna(c) and c > th:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def _apply_cluster_cap(
    selected: list[str],
    weights: pd.Series,
    train_pos_df: pd.DataFrame,
    cfg: Layer1Config,
) -> pd.Series:
    if len(selected) <= 1:
        return weights
    corr = train_pos_df[selected].corr()
    clusters = _build_clusters(selected, corr, cfg.corr_cluster_th)
    out = weights.copy()
    for grp in clusters:
        grp_sum = float(out.reindex(grp).sum())
        if grp_sum > cfg.cluster_cap and grp_sum > 0:
            scale = cfg.cluster_cap / grp_sum
            out.loc[grp] = out.loc[grp] * scale
    return out


def _build_line1_weights(
    open_px: pd.Series,
    close_px: pd.Series,
    pos_df: pd.DataFrame,
    buy_dates_map: dict[str, np.ndarray],
    cfg: Layer1Config,
    cost: CostParams,
) -> tuple[pd.Series, pd.DataFrame]:
    idx = open_px.index
    month_starts = [i for i in _month_start_indices(idx) if i >= cfg.train_days]
    raw_all = pos_df.mean(axis=1).reindex(idx).ffill().fillna(0.0)

    w = pd.Series(cfg.floor, index=idx, dtype=float)
    logs: list[dict[str, object]] = []

    for k, start_i in enumerate(month_starts):
        end_i = month_starts[k + 1] if k + 1 < len(month_starts) else len(idx)
        seg_idx = idx[start_i:end_i]

        train_slice = slice(start_i - cfg.train_days, start_i)
        train_open = open_px.iloc[train_slice]
        train_close = close_px.iloc[train_slice]
        train_pos_df = pos_df.iloc[train_slice]

        cands: list[tuple[str, float]] = []
        for name in train_pos_df.columns:
            train_pos = train_pos_df[name]
            if train_pos.std() == 0:
                continue
            score, st = _score_candidate(train_open, train_close, train_pos, cfg, cost)
            if score == float("-inf"):
                continue
            # Active lookback guard by recent buys.
            recent = train_pos.iloc[-cfg.active_lookback_days :]
            recent_buy = int((recent.diff() > 0).sum())
            if recent_buy <= 0:
                continue
            cands.append((name, score))

        cands.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in cands[: cfg.top_k]]

        if not selected:
            seg_raw = raw_all.loc[seg_idx]
            w.loc[seg_idx] = cfg.floor + (1 - cfg.floor) * seg_raw.values
            logs.append(
                {
                    "rebalance_date": str(idx[start_i].date()),
                    "selected": "",
                    "num_selected": 0,
                }
            )
            continue

        score_arr = np.array([max(1e-8, x[1]) for x in cands[: cfg.top_k]], dtype=float)
        base_w = pd.Series(score_arr / score_arr.sum(), index=selected, dtype=float)

        # Stale penalty.
        penalty = pd.Series(1.0, index=selected, dtype=float)
        ref_date = idx[start_i]
        for n in selected:
            days = _days_since_last_buy(buy_dates_map.get(n, np.array([], dtype="datetime64[ns]")), ref_date)
            if days is None or days > 180:
                penalty.loc[n] = 0.0
            elif days > 90:
                penalty.loc[n] = 0.5
            else:
                penalty.loc[n] = 1.0

        w_sel = base_w * penalty
        if w_sel.sum() <= 0:
            w_sel = base_w.copy()
        else:
            w_sel = w_sel / w_sel.sum()

        # Correlation cluster cap.
        w_sel = _apply_cluster_cap(selected, w_sel, train_pos_df, cfg)

        seg_raw = (pos_df[selected].loc[seg_idx] * w_sel.reindex(selected).values).sum(axis=1)
        seg_w = cfg.floor + (1 - cfg.floor) * seg_raw.clip(0, 1)
        w.loc[seg_idx] = seg_w.values

        logs.append(
            {
                "rebalance_date": str(ref_date.date()),
                "selected": "|".join(selected),
                "num_selected": len(selected),
                "effective_weight_sum": float(w_sel.sum()),
            }
        )

    w = _smooth_weights(w, cfg.ema_span, cfg.delta_cap, cfg.min_trade_dw)

    # Vol target overlay.
    ret_o2o = open_px.pct_change().fillna(0.0)
    strat_ret = ret_o2o * w.shift(1).fillna(w.iloc[0])
    rolling_vol = strat_ret.rolling(20).std() * np.sqrt(252)
    scaler = (cfg.target_vol / rolling_vol.replace(0, np.nan)).clip(lower=0.55, upper=1.15).fillna(1.0)
    w_vt = (w * scaler).clip(0, 1)
    w_vt = _smooth_weights(w_vt, cfg.ema_span, cfg.delta_cap, cfg.min_trade_dw)

    return w_vt, pd.DataFrame(logs)


def _build_line2_robust_smaa(close_px: pd.Series) -> pd.Series:
    win = 90
    center = close_px.rolling(win).median()
    resid = close_px - center
    mad = resid.abs().rolling(win).median()
    scale = 1.4826 * mad.replace(0, np.nan)

    upper = center + 0.8 * scale
    lower = center - 0.8 * scale

    state = 0.0
    out = []
    for dt in close_px.index:
        c = close_px.loc[dt]
        u = upper.loc[dt]
        l = lower.loc[dt]
        if pd.notna(u) and c > u:
            state = 1.0
        elif pd.notna(l) and c < l:
            state = 0.0
        out.append(state)
    w = pd.Series(out, index=close_px.index, dtype=float)
    return _smooth_weights(w, ema_span=3, delta_cap=0.10, min_trade_dw=0.005)


def _kalman_1d(series: pd.Series, q: float = 1e-5, r: float = 2e-4) -> pd.Series:
    x = float(series.iloc[0])
    p = 1.0
    out = []
    for z in series.values:
        # predict
        p = p + q
        # update
        k = p / (p + r)
        x = x + k * (float(z) - x)
        p = (1 - k) * p
        out.append(x)
    return pd.Series(out, index=series.index, dtype=float)


def _build_line3_extreme(close_px: pd.Series) -> pd.Series:
    logc = np.log(close_px.replace(0, np.nan)).ffill().dropna()
    level = _kalman_1d(logc, q=1e-5, r=2e-4)
    slope_s = level.diff(5)
    slope_l = level.diff(20)
    resid = logc - level
    rz = resid / resid.rolling(40).std().replace(0, np.nan)

    state = 0.0
    out = []
    for dt in logc.index:
        s = slope_s.loc[dt]
        l = slope_l.loc[dt]
        z = rz.loc[dt]

        if state < 0.5:
            if pd.notna(s) and pd.notna(l) and pd.notna(z) and s > 0 and l > 0 and z > -0.5:
                state = 1.0
        else:
            if pd.notna(s) and pd.notna(z) and s < 0 and z < 0:
                state = 0.0
        out.append(state)

    w = pd.Series(out, index=logc.index, dtype=float).reindex(close_px.index).ffill().fillna(0.0)
    return _smooth_weights(w, ema_span=3, delta_cap=0.10, min_trade_dw=0.005)


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
    _, w, _, _, _, eq, _, _ = run_ensemble(cfg)
    if eq is None or eq.empty:
        raise RuntimeError("Baseline run failed.")
    return eq, w


def main() -> None:
    parser = argparse.ArgumentParser(description="Round3 sequential new logic tests (1/2/3).")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--warehouse", default="analysis/strategy_warehouse.json")
    parser.add_argument("--split_date", default="2025-06-30")
    parser.add_argument("--include_archive", action="store_true")
    parser.add_argument("--bh_target_mult", type=float, default=1.2)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    split_date = pd.Timestamp(args.split_date)
    l1_cfg = Layer1Config()

    names = _load_strategy_names(Path(args.warehouse))
    file_map = _resolve_trade_files(names, include_archive=args.include_archive)
    names = [n for n in names if n in file_map]
    if len(names) < 2:
        raise SystemExit("Need at least 2 strategy trade files.")

    df_price, _ = load_data(args.ticker)
    req = {"open", "close"}
    if df_price.empty or not req.issubset(df_price.columns):
        raise SystemExit(f"Price data unavailable or missing cols for {args.ticker}.")

    open_px = pd.to_numeric(df_price["open"], errors="coerce").dropna().sort_index()
    close_px = pd.to_numeric(df_price["close"], errors="coerce").dropna().sort_index()
    idx = open_px.index.intersection(close_px.index)
    open_px = open_px.reindex(idx).dropna()
    close_px = close_px.reindex(idx).dropna()

    pos_map: dict[str, pd.Series] = {}
    buy_map: dict[str, np.ndarray] = {}
    for n in names:
        pos, buy_dates = _load_positions_and_buys(file_map[n], open_px.index)
        if pos is None or pos.std() == 0:
            continue
        pos_map[n] = pos.reindex(open_px.index).ffill().fillna(0.0)
        buy_map[n] = buy_dates

    if len(pos_map) < 2:
        raise SystemExit("Not enough usable strategies after parsing.")

    pos_df = pd.DataFrame(pos_map, index=open_px.index)
    names = list(pos_df.columns)

    print(f"Ticker: {args.ticker}")
    print(f"Strategies loaded: {len(names)}")
    print(f"Range: {open_px.index[0].date()} -> {open_px.index[-1].date()} ({len(open_px)} bars)")
    print(f"Split: {split_date.date()}")

    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)

    base_eq, base_w = _run_baseline_current_proportional(args.ticker, names, file_map, cost)

    # Line 1: no signal formula changes, only anti-overfit allocator layer.
    l1_w, l1_sel = _build_line1_weights(
        open_px=open_px,
        close_px=close_px,
        pos_df=pos_df,
        buy_dates_map=buy_map,
        cfg=l1_cfg,
        cost=cost,
    )
    l1_eq, _, _, _ = calculate_performance(open_px, l1_w, cost)

    # Line 2: robust SMAA replacement.
    l2_w = _apply_t_plus_1(_build_line2_robust_smaa(close_px))
    l2_eq, _, _, _ = calculate_performance(open_px, l2_w, cost)

    # Line 3: extreme SMAA replacement.
    l3_w = _apply_t_plus_1(_build_line3_extreme(close_px))
    l3_eq, _, _, _ = calculate_performance(open_px, l3_w, cost)

    bh_eq = close_px / close_px.iloc[0] * 1_000_000.0

    rows: list[dict[str, object]] = []
    for name, eq, w in [
        ("baseline_current_proportional", base_eq, base_w),
        ("line1_allocator_anti_overfit", l1_eq, l1_w),
        ("line2_robust_smaa", l2_eq, l2_w),
        ("line3_extreme_kalman", l3_eq, l3_w),
    ]:
        full = _compute_metrics(eq, bh_eq)
        oos = _compute_metrics(eq[eq.index > split_date], bh_eq[bh_eq.index > split_date])
        row = {"strategy": name}
        row.update({f"full_{k}": v for k, v in full.items()})
        row.update({f"oos_{k}": v for k, v in oos.items()})
        row["time_in_market"] = float((w > 0.05).mean())
        row["turnover_py"] = float(w.diff().abs().sum() / len(w) * 252) if len(w) else float("nan")
        full_bh = row.get("full_bh_ret", np.nan)
        oos_bh = row.get("oos_bh_ret", np.nan)
        row["full_bh_multiple"] = (
            (row["full_ret"] / full_bh) if pd.notna(full_bh) and full_bh > 0 else np.nan
        )
        row["oos_bh_multiple"] = (
            (row["oos_ret"] / oos_bh) if pd.notna(oos_bh) and oos_bh > 0 else np.nan
        )
        row["full_target_pass"] = (
            bool(row["full_bh_multiple"] >= args.bh_target_mult)
            if pd.notna(row["full_bh_multiple"])
            else False
        )
        row["oos_target_pass"] = (
            bool(row["oos_bh_multiple"] >= args.bh_target_mult)
            if pd.notna(row["oos_bh_multiple"])
            else False
        )
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("oos_alpha_vs_bh", ascending=False)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = out_dir / f"round3_seq_summary_{ts}.csv"
    weights_path = out_dir / f"round3_seq_weights_{ts}.csv"
    equity_path = out_dir / f"round3_seq_equity_{ts}.csv"
    sel_path = out_dir / f"round3_seq_line1_selections_{ts}.csv"

    summary_df.to_csv(summary_path, index=False)

    weights_df = pd.DataFrame(
        {
            "baseline_w": base_w.reindex(open_px.index).ffill().fillna(0.0),
            "line1_w": l1_w.reindex(open_px.index).ffill().fillna(0.0),
            "line2_w": l2_w.reindex(open_px.index).ffill().fillna(0.0),
            "line3_w": l3_w.reindex(open_px.index).ffill().fillna(0.0),
        },
        index=open_px.index,
    )
    weights_df.to_csv(weights_path, index_label="date")

    eq_df = pd.DataFrame(
        {
            "baseline_equity": base_eq.reindex(open_px.index).ffill(),
            "line1_equity": l1_eq.reindex(open_px.index).ffill(),
            "line2_equity": l2_eq.reindex(open_px.index).ffill(),
            "line3_equity": l3_eq.reindex(open_px.index).ffill(),
            "bh_equity": bh_eq.reindex(open_px.index).ffill(),
        },
        index=open_px.index,
    )
    eq_df.to_csv(equity_path, index_label="date")
    l1_sel.to_csv(sel_path, index=False)

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
            "full_bh_multiple",
            "oos_bh_multiple",
            "full_target_pass",
            "oos_target_pass",
        ]
    ].copy()
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
    view["full_bh_multiple"] = view["full_bh_multiple"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "nan")
    view["oos_bh_multiple"] = view["oos_bh_multiple"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "nan")

    print("\nRound3 sequential (1->2->3) result:")
    print(f"Target: strategy_return >= B&H * {args.bh_target_mult:.2f}")
    print(view.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved weights: {weights_path}")
    print(f"Saved equity: {equity_path}")
    print(f"Saved line1 selections: {sel_path}")


if __name__ == "__main__":
    main()
