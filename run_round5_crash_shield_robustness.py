from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import run_round4_overlay_test as r4
from SSS_EnsembleTab import CostParams, calculate_performance
from sss_core.logic import load_data


@dataclass(frozen=True)
class CrashShieldParam:
    dd_window: int = 5
    dd_trigger: float = -0.08
    atr_short: int = 20
    atr_long: int = 60
    atr_ratio_trigger: float = 1.5
    use_gap: bool = False
    gap_trigger: float = -0.04
    entry_confirm_days: int = 2
    cap_shield: float = 0.35
    max_shield_days: int = 7
    exit_rebound_2d: float = 0.06
    exit_ma_window: int = 20
    ramp_caps: tuple[float, float, float] = (0.6, 0.8, 1.0)
    delta_cap: float = 0.12
    min_trade_dw: float = 0.01


def _wealth_mult(ret: float, bh_ret: float) -> float:
    if (1.0 + bh_ret) <= 0:
        return float("nan")
    return float((1.0 + ret) / (1.0 + bh_ret))


def _atr(high_px: pd.Series, low_px: pd.Series, close_px: pd.Series, window: int) -> pd.Series:
    prev_close = close_px.shift(1)
    tr = pd.concat(
        [
            high_px - low_px,
            (high_px - prev_close).abs(),
            (low_px - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _build_crash_shield_caps(
    close_px: pd.Series,
    open_px: pd.Series,
    high_px: pd.Series,
    low_px: pd.Series,
    p: CrashShieldParam,
) -> tuple[pd.Series, pd.DataFrame]:
    idx = close_px.index
    dd = close_px / close_px.rolling(p.dd_window).max() - 1.0
    atr_s = _atr(high_px, low_px, close_px, p.atr_short)
    atr_l = _atr(high_px, low_px, close_px, p.atr_long)
    atr_ratio = atr_s / atr_l.replace(0.0, np.nan)
    gap = open_px / close_px.shift(1) - 1.0

    event_dd_atr = (dd <= p.dd_trigger) & (atr_ratio >= p.atr_ratio_trigger)
    event_gap = (gap <= p.gap_trigger) if p.use_gap else pd.Series(False, index=idx, dtype=bool)
    event_raw = (event_dd_atr | event_gap).fillna(False)

    two_day_rebound = close_px.pct_change(2)
    up = (close_px.diff() > 0).astype(bool)
    up_prev = up.shift(1)
    up_prev = up_prev.where(up_prev.notna(), False).astype(bool)
    two_up = (up & up_prev).astype(bool)
    ma_exit = close_px.rolling(p.exit_ma_window).mean()
    exit_rebound = (two_day_rebound >= p.exit_rebound_2d).fillna(False)
    exit_ma = (close_px > ma_exit).fillna(False)
    exit_raw = (exit_rebound | two_up | exit_ma).fillna(False)

    caps = pd.Series(1.0, index=idx, dtype=float)
    mode_hist: list[str] = []
    in_shield_hist: list[bool] = []
    in_ramp_hist: list[bool] = []

    mode = "normal"
    entry_streak = 0
    shield_days = 0
    ramp_i = -1

    for i, dt in enumerate(idx):
        if i == 0:
            event_prev = False
            exit_prev = False
        else:
            prev_dt = idx[i - 1]
            event_prev = bool(event_raw.loc[prev_dt])
            exit_prev = bool(exit_raw.loc[prev_dt])

        if mode == "shield":
            shield_days += 1
            if exit_prev or shield_days >= p.max_shield_days:
                mode = "ramp"
                ramp_i = 0
                shield_days = 0
        elif mode == "ramp":
            if event_prev:
                entry_streak += 1
            else:
                entry_streak = 0

            if entry_streak >= p.entry_confirm_days:
                mode = "shield"
                shield_days = 0
                entry_streak = 0
                ramp_i = -1
            else:
                if ramp_i >= len(p.ramp_caps) - 1:
                    mode = "normal"
                    ramp_i = -1
                else:
                    ramp_i += 1
        else:
            if event_prev:
                entry_streak += 1
            else:
                entry_streak = 0
            if entry_streak >= p.entry_confirm_days:
                mode = "shield"
                shield_days = 0
                entry_streak = 0

        if mode == "shield":
            caps.iloc[i] = p.cap_shield
        elif mode == "ramp":
            caps.iloc[i] = p.ramp_caps[min(max(ramp_i, 0), len(p.ramp_caps) - 1)]
        else:
            caps.iloc[i] = 1.0

        mode_hist.append(mode)
        in_shield_hist.append(mode == "shield")
        in_ramp_hist.append(mode == "ramp")

    debug = pd.DataFrame(
        {
            "dd": dd,
            "atr_ratio": atr_ratio,
            "gap": gap,
            "event_dd_atr": event_dd_atr.astype(bool),
            "event_gap": event_gap.astype(bool),
            "event_raw": event_raw.astype(bool),
            "exit_rebound": exit_rebound.astype(bool),
            "exit_two_up": two_up.astype(bool),
            "exit_ma20": exit_ma.astype(bool),
            "exit_raw": exit_raw.astype(bool),
            "cap": caps,
            "mode": mode_hist,
            "in_shield": in_shield_hist,
            "in_ramp": in_ramp_hist,
        },
        index=idx,
    )
    return caps, debug


def _apply_crash_shield(
    base_w: pd.Series,
    close_px: pd.Series,
    open_px: pd.Series,
    high_px: pd.Series,
    low_px: pd.Series,
    p: CrashShieldParam,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    caps, debug = _build_crash_shield_caps(close_px, open_px, high_px, low_px, p)
    out = np.minimum(base_w, caps)
    # Crash shield should de-risk immediately and only limit re-risk speed.
    out = _smooth_reentry_only(out, up_delta_cap=p.delta_cap, min_trade_dw=p.min_trade_dw)
    return out, caps, debug


def _smooth_reentry_only(w: pd.Series, up_delta_cap: float, min_trade_dw: float) -> pd.Series:
    out = w.copy()
    for i in range(1, len(out)):
        prev = float(out.iloc[i - 1])
        curr = float(out.iloc[i])
        d = curr - prev
        if d > up_delta_cap:
            out.iloc[i] = prev + up_delta_cap
    for i in range(1, len(out)):
        prev = float(out.iloc[i - 1])
        curr = float(out.iloc[i])
        if abs(curr - prev) < min_trade_dw:
            out.iloc[i] = prev
    return out.clip(0, 1)


def _parse_split_dates(raw: str) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    for x in raw.split(","):
        s = x.strip()
        if not s:
            continue
        out.append(pd.Timestamp(s))
    return sorted(set(out))


def _eval_candidate(
    eq: pd.Series,
    bh_eq: pd.Series,
    split_dates: list[pd.Timestamp],
    main_split: pd.Timestamp,
    bh_target_mult: float,
) -> dict[str, float | bool]:
    full = r4._compute_metrics(eq, bh_eq)
    main = r4._compute_metrics(eq[eq.index > main_split], bh_eq[bh_eq.index > main_split])
    if not full or not main:
        return {}

    full_mult = _wealth_mult(full["ret"], full["bh_ret"])
    main_mult = _wealth_mult(main["ret"], main["bh_ret"])

    oos_mults: list[float] = []
    oos_mdd_imps: list[float] = []
    for sd in split_dates:
        st = r4._compute_metrics(eq[eq.index > sd], bh_eq[bh_eq.index > sd])
        if not st:
            continue
        mult = _wealth_mult(st["ret"], st["bh_ret"])
        mdd_imp = float(st["mdd"] - st["bh_mdd"])
        oos_mults.append(mult)
        oos_mdd_imps.append(mdd_imp)

    if not oos_mults:
        return {}

    median_oos_mult = float(np.nanmedian(oos_mults))
    worst_oos_mult = float(np.nanmin(oos_mults))
    median_oos_mdd_imp = float(np.nanmedian(oos_mdd_imps))
    worst_oos_mdd_imp = float(np.nanmin(oos_mdd_imps))
    robust_score = median_oos_mult + 0.25 * median_oos_mdd_imp + 0.05 * main["sharpe"]

    out: dict[str, float | bool] = {
        "full_ret": float(full["ret"]),
        "full_mdd": float(full["mdd"]),
        "full_bh_multiple": full_mult,
        "main_oos_ret": float(main["ret"]),
        "main_oos_mdd": float(main["mdd"]),
        "main_oos_bh_multiple": main_mult,
        "main_oos_mdd_improve_vs_bh": float(main["mdd"] - main["bh_mdd"]),
        "main_oos_sharpe": float(main["sharpe"]),
        "median_oos_multiple": median_oos_mult,
        "worst_oos_multiple": worst_oos_mult,
        "median_oos_mdd_improve": median_oos_mdd_imp,
        "worst_oos_mdd_improve": worst_oos_mdd_imp,
        "robust_score": robust_score,
        "full_target_pass": bool(pd.notna(full_mult) and full_mult >= bh_target_mult),
        "main_oos_target_pass": bool(pd.notna(main_mult) and main_mult >= bh_target_mult),
    }
    for i, sd in enumerate(split_dates):
        if i < len(oos_mults):
            out[f"oos_mult_{sd.date()}"] = float(oos_mults[i])
            out[f"oos_mdd_imp_{sd.date()}"] = float(oos_mdd_imps[i])
    return out


def _shift_weights(w: pd.Series, days: int) -> pd.Series:
    if days == 0:
        return w.copy()
    s = w.shift(days)
    if days > 0:
        return s.fillna(float(w.iloc[0]))
    return s.fillna(float(w.iloc[-1]))


def _load_activity_stats(
    file_map: dict[str, Path],
    asof: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lb_start = asof - pd.Timedelta(days=lookback_days)
    for name, p in file_map.items():
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {str(c).lower().strip(): c for c in df.columns}
        date_col = cols.get("trade_date", cols.get("date"))
        type_col = cols.get("type", cols.get("action"))
        if date_col is None or type_col is None:
            continue
        dt = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        side = df[type_col].astype(str).str.lower().str.strip()
        buy = side.isin(["buy", "entry", "long", "add"])
        sell = side.isin(["sell", "exit", "short", "sell_forced", "forced_sell"])

        recent = dt.between(lb_start, asof, inclusive="both")
        recent_buys = int((buy & recent & dt.notna()).sum())
        recent_sells = int((sell & recent & dt.notna()).sum())

        last_buy_dt = dt[buy & dt.notna()]
        last_sell_dt = dt[sell & dt.notna()]
        last_buy = pd.NaT if last_buy_dt.empty else last_buy_dt.max()
        last_sell = pd.NaT if last_sell_dt.empty else last_sell_dt.max()
        days_since_last_buy = None
        if pd.notna(last_buy):
            days_since_last_buy = int((asof - pd.Timestamp(last_buy)).days)

        rows.append(
            {
                "strategy": name,
                "recent_buy_count": recent_buys,
                "recent_sell_count": recent_sells,
                "last_buy_date": "" if pd.isna(last_buy) else str(pd.Timestamp(last_buy).date()),
                "last_sell_date": "" if pd.isna(last_sell) else str(pd.Timestamp(last_sell).date()),
                "days_since_last_buy": days_since_last_buy,
                "no_buy_recent": bool(recent_buys == 0),
                "trade_file": str(p),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["no_buy_recent", "days_since_last_buy", "recent_buy_count"],
        ascending=[False, False, True],
    )


def _grid() -> list[CrashShieldParam]:
    out: list[CrashShieldParam] = []
    for dd_t in [-0.08, -0.10]:
        for atr_t in [1.4, 1.5]:
            for use_gap in [False, True]:
                for entry_days in [1, 2]:
                    for cap in [0.25, 0.35, 0.45]:
                        for max_days in [5, 7]:
                            for exit_reb in [0.04, 0.06]:
                                out.append(
                                    CrashShieldParam(
                                        dd_trigger=dd_t,
                                        atr_ratio_trigger=atr_t,
                                        use_gap=use_gap,
                                        entry_confirm_days=entry_days,
                                        cap_shield=cap,
                                        max_shield_days=max_days,
                                        exit_rebound_2d=exit_reb,
                                    )
                                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Round5 crash-shield robustness test.")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--warehouse", default="analysis/strategy_warehouse.json")
    parser.add_argument("--include_archive", action="store_true")
    parser.add_argument("--base_mode", choices=["ensemble", "bh"], default="bh")
    parser.add_argument("--split_dates", default="2024-12-31,2025-03-31,2025-06-30,2025-09-30")
    parser.add_argument("--main_split_date", default="2025-06-30")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--bh_target_mult", type=float, default=1.2)
    parser.add_argument("--median_mult_min", type=float, default=1.02)
    parser.add_argument("--worst_mult_min", type=float, default=0.95)
    parser.add_argument("--lookback_days", type=int, default=180)
    parser.add_argument("--top_k_time_shift", type=int, default=5)
    args = parser.parse_args()

    split_dates = _parse_split_dates(args.split_dates)
    main_split = pd.Timestamp(args.main_split_date)
    if not split_dates:
        split_dates = [main_split]

    df_price, _ = load_data(args.ticker)
    if df_price.empty or "open" not in df_price.columns or "close" not in df_price.columns:
        raise SystemExit("Price data unavailable.")

    open_px = pd.to_numeric(df_price["open"], errors="coerce").dropna().sort_index()
    close_px = pd.to_numeric(df_price["close"], errors="coerce").dropna().sort_index()
    if "high" in df_price.columns:
        high_px = pd.to_numeric(df_price["high"], errors="coerce")
    else:
        high_px = close_px.copy()
    if "low" in df_price.columns:
        low_px = pd.to_numeric(df_price["low"], errors="coerce")
    else:
        low_px = close_px.copy()

    idx = open_px.index.intersection(close_px.index)
    idx = idx.intersection(high_px.dropna().index).intersection(low_px.dropna().index)
    open_px = open_px.reindex(idx).dropna()
    close_px = close_px.reindex(idx).dropna()
    high_px = high_px.reindex(idx).fillna(close_px)
    low_px = low_px.reindex(idx).fillna(close_px)

    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)
    file_map: dict[str, Path] = {}
    if args.base_mode == "ensemble":
        names = r4._load_strategy_names(Path(args.warehouse))
        file_map = r4._resolve_trade_files(names, include_archive=args.include_archive)
        names = [n for n in names if n in file_map]
        if len(names) < 2:
            raise SystemExit("Need >=2 strategy files for ensemble mode.")
        base_eq, base_w = r4._run_baseline(args.ticker, names, file_map, cost)
    else:
        base_w = pd.Series(1.0, index=open_px.index, dtype=float)
        base_eq, _, _, _ = calculate_performance(open_px, base_w, cost)

    base_w = base_w.reindex(open_px.index).ffill().fillna(0.0)
    bh_eq = open_px / open_px.iloc[0] * 1_000_000.0
    base_eval = _eval_candidate(base_eq, bh_eq, split_dates, main_split, args.bh_target_mult)
    if not base_eval:
        raise SystemExit("Baseline eval failed.")

    print(f"Ticker: {args.ticker}")
    print(f"Range: {open_px.index[0].date()} -> {open_px.index[-1].date()} ({len(open_px)} bars)")
    print(f"Base mode: {args.base_mode}")
    print(
        f"Baseline main OOS: ret={base_eval['main_oos_ret']:.2%}, "
        f"mdd={base_eval['main_oos_mdd']:.2%}, mult={base_eval['main_oos_bh_multiple']:.4f}"
    )
    print(f"Split dates: {', '.join([str(x.date()) for x in split_dates])}")

    grid = _grid()
    rows: list[dict[str, object]] = []
    w_cache: dict[int, pd.Series] = {}
    cap_cache: dict[int, pd.Series] = {}
    dbg_cache: dict[int, pd.DataFrame] = {}

    for i, gp in enumerate(grid, 1):
        w, cap, dbg = _apply_crash_shield(base_w, close_px, open_px, high_px, low_px, gp)
        eq, _, _, _ = calculate_performance(open_px, w, cost)
        ev = _eval_candidate(eq, bh_eq, split_dates, main_split, args.bh_target_mult)
        if not ev:
            continue

        robust_pass = bool(
            pd.notna(ev["median_oos_multiple"])
            and pd.notna(ev["worst_oos_multiple"])
            and float(ev["median_oos_multiple"]) >= args.median_mult_min
            and float(ev["worst_oos_multiple"]) >= args.worst_mult_min
        )

        row: dict[str, object] = {
            "candidate_id": i,
            "base_mode": args.base_mode,
            "robust_pass": robust_pass,
            "turnover_py": float(w.diff().abs().sum() / len(w) * 252),
            "time_in_market": float((w > 0.05).mean()),
            **asdict(gp),
            **ev,
        }
        rows.append(row)
        w_cache[i] = w
        cap_cache[i] = cap
        dbg_cache[i] = dbg

        if i % 30 == 0 or i == len(grid):
            print(f"Progress: {i}/{len(grid)}")

    if not rows:
        raise SystemExit("No candidate rows.")

    res = pd.DataFrame(rows).sort_values("robust_score", ascending=False)
    best = res.iloc[0]
    best_id = int(best["candidate_id"])
    best_w = w_cache[best_id]
    best_cap = cap_cache[best_id]
    best_dbg = dbg_cache[best_id]
    best_eq, _, _, _ = calculate_performance(open_px, best_w, cost)

    # Time-shift robustness for top-k candidates.
    shifts = [-3, -2, -1, 0, 1, 2, 3]
    shift_rows: list[dict[str, object]] = []
    topk = res.head(max(1, args.top_k_time_shift))
    for _, r in topk.iterrows():
        cid = int(r["candidate_id"])
        w0 = w_cache[cid]
        for sh in shifts:
            ws = _shift_weights(w0, sh)
            eqs, _, _, _ = calculate_performance(open_px, ws, cost)
            evs = _eval_candidate(eqs, bh_eq, split_dates, main_split, args.bh_target_mult)
            if not evs:
                continue
            shift_rows.append(
                {
                    "candidate_id": cid,
                    "shift_days": sh,
                    "median_oos_multiple": evs["median_oos_multiple"],
                    "worst_oos_multiple": evs["worst_oos_multiple"],
                    "main_oos_bh_multiple": evs["main_oos_bh_multiple"],
                    "main_oos_mdd_improve_vs_bh": evs["main_oos_mdd_improve_vs_bh"],
                }
            )
    shift_df = pd.DataFrame(shift_rows).sort_values(["candidate_id", "shift_days"]) if shift_rows else pd.DataFrame()

    # Local sensitivity around best params (about +/-10%).
    bp = CrashShieldParam(
        dd_window=int(best["dd_window"]),
        dd_trigger=float(best["dd_trigger"]),
        atr_short=int(best["atr_short"]),
        atr_long=int(best["atr_long"]),
        atr_ratio_trigger=float(best["atr_ratio_trigger"]),
        use_gap=bool(best["use_gap"]),
        gap_trigger=float(best["gap_trigger"]),
        entry_confirm_days=int(best["entry_confirm_days"]),
        cap_shield=float(best["cap_shield"]),
        max_shield_days=int(best["max_shield_days"]),
        exit_rebound_2d=float(best["exit_rebound_2d"]),
        exit_ma_window=int(best["exit_ma_window"]),
        ramp_caps=tuple(float(x) for x in str(best["ramp_caps"]).strip("()").split(",") if x.strip()),
        delta_cap=float(best["delta_cap"]),
        min_trade_dw=float(best["min_trade_dw"]),
    )
    if len(bp.ramp_caps) != 3:
        bp = CrashShieldParam(
            dd_window=bp.dd_window,
            dd_trigger=bp.dd_trigger,
            atr_short=bp.atr_short,
            atr_long=bp.atr_long,
            atr_ratio_trigger=bp.atr_ratio_trigger,
            use_gap=bp.use_gap,
            gap_trigger=bp.gap_trigger,
            entry_confirm_days=bp.entry_confirm_days,
            cap_shield=bp.cap_shield,
            max_shield_days=bp.max_shield_days,
            exit_rebound_2d=bp.exit_rebound_2d,
            exit_ma_window=bp.exit_ma_window,
            delta_cap=bp.delta_cap,
            min_trade_dw=bp.min_trade_dw,
        )

    sens: list[tuple[str, CrashShieldParam]] = [
        ("base", bp),
        ("cap_shield_-10%", CrashShieldParam(**{**asdict(bp), "cap_shield": max(0.05, bp.cap_shield * 0.9)})),
        ("cap_shield_+10%", CrashShieldParam(**{**asdict(bp), "cap_shield": min(0.95, bp.cap_shield * 1.1)})),
        ("dd_trigger_-10%", CrashShieldParam(**{**asdict(bp), "dd_trigger": bp.dd_trigger * 1.1})),
        ("dd_trigger_+10%", CrashShieldParam(**{**asdict(bp), "dd_trigger": bp.dd_trigger * 0.9})),
        (
            "atr_ratio_-10%",
            CrashShieldParam(**{**asdict(bp), "atr_ratio_trigger": max(1.1, bp.atr_ratio_trigger * 0.9)}),
        ),
        ("atr_ratio_+10%", CrashShieldParam(**{**asdict(bp), "atr_ratio_trigger": bp.atr_ratio_trigger * 1.1})),
        (
            "max_days_-1",
            CrashShieldParam(**{**asdict(bp), "max_shield_days": max(2, bp.max_shield_days - 1)}),
        ),
        ("max_days_+1", CrashShieldParam(**{**asdict(bp), "max_shield_days": bp.max_shield_days + 1})),
        (
            "exit_rebound_-10%",
            CrashShieldParam(**{**asdict(bp), "exit_rebound_2d": max(0.02, bp.exit_rebound_2d * 0.9)}),
        ),
        ("exit_rebound_+10%", CrashShieldParam(**{**asdict(bp), "exit_rebound_2d": bp.exit_rebound_2d * 1.1})),
    ]
    sens_rows: list[dict[str, object]] = []
    for tag, sp in sens:
        ws, _, _ = _apply_crash_shield(base_w, close_px, open_px, high_px, low_px, sp)
        eqs, _, _, _ = calculate_performance(open_px, ws, cost)
        evs = _eval_candidate(eqs, bh_eq, split_dates, main_split, args.bh_target_mult)
        if not evs:
            continue
        sens_rows.append(
            {
                "variant": tag,
                "main_oos_bh_multiple": evs["main_oos_bh_multiple"],
                "main_oos_mdd_improve_vs_bh": evs["main_oos_mdd_improve_vs_bh"],
                "median_oos_multiple": evs["median_oos_multiple"],
                "worst_oos_multiple": evs["worst_oos_multiple"],
                "robust_score": evs["robust_score"],
            }
        )
    sens_df = pd.DataFrame(sens_rows).sort_values("robust_score", ascending=False) if sens_rows else pd.DataFrame()

    # Recent diagnostics.
    lb = args.lookback_days
    recent_idx = best_w.index[best_w.index >= (best_w.index[-1] - pd.Timedelta(days=lb))]
    base_d = base_w.reindex(recent_idx).ffill()
    best_d = best_w.reindex(recent_idx).ffill()
    cap_d = best_cap.reindex(recent_idx).ffill()
    dbg_d = best_dbg.reindex(recent_idx).ffill()
    recent_diag = pd.DataFrame(
        [
            {
                "lookback_days": lb,
                "base_increase_days": int((base_d.diff() > 0).sum()),
                "blocked_increase_days": int(((base_d.diff() > 0) & (best_d < base_d - 1e-12)).sum()),
                "shield_days": int(dbg_d["in_shield"].fillna(False).sum()),
                "ramp_days": int(dbg_d["in_ramp"].fillna(False).sum()),
                "crash_event_days": int(dbg_d["event_raw"].fillna(False).sum()),
                "exit_event_days": int(dbg_d["exit_raw"].fillna(False).sum()),
                "avg_cap": float(cap_d.mean()),
                "min_cap": float(cap_d.min()),
                "avg_weight_base": float(base_d.mean()),
                "avg_weight_best": float(best_d.mean()),
            }
        ]
    )

    activity_df = _load_activity_stats(file_map, asof=open_px.index[-1], lookback_days=lb) if file_map else pd.DataFrame()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.base_mode}"

    all_path = out_dir / f"round5_crash_grid_{tag}_{ts}.csv"
    top_path = out_dir / f"round5_crash_top_{tag}_{ts}.csv"
    shift_path = out_dir / f"round5_crash_shift_{tag}_{ts}.csv"
    sens_path = out_dir / f"round5_crash_sensitivity_{tag}_{ts}.csv"
    diag_path = out_dir / f"round5_crash_recent_diag_{tag}_{ts}.csv"
    act_path = out_dir / f"round5_strategy_activity_{tag}_{ts}.csv"
    w_path = out_dir / f"round5_best_weights_{tag}_{ts}.csv"
    eq_path = out_dir / f"round5_best_equity_{tag}_{ts}.csv"
    dbg_path = out_dir / f"round5_best_debug_{tag}_{ts}.csv"

    res.to_csv(all_path, index=False)
    res.head(30).to_csv(top_path, index=False)
    if not shift_df.empty:
        shift_df.to_csv(shift_path, index=False)
    if not sens_df.empty:
        sens_df.to_csv(sens_path, index=False)
    recent_diag.to_csv(diag_path, index=False)
    if not activity_df.empty:
        activity_df.to_csv(act_path, index=False)

    pd.DataFrame(
        {
            "base_w": base_w.reindex(open_px.index).ffill().fillna(0.0),
            "best_w": best_w.reindex(open_px.index).ffill().fillna(0.0),
            "best_cap": best_cap.reindex(open_px.index).ffill().fillna(1.0),
        },
        index=open_px.index,
    ).to_csv(w_path, index_label="date")

    pd.DataFrame(
        {
            "base_equity": base_eq.reindex(open_px.index).ffill(),
            "best_equity": best_eq.reindex(open_px.index).ffill(),
            "bh_equity": bh_eq.reindex(open_px.index).ffill(),
        },
        index=open_px.index,
    ).to_csv(eq_path, index_label="date")

    best_dbg.to_csv(dbg_path, index_label="date")

    near_cnt = int(((res["main_oos_bh_multiple"] >= 0.85) & (res["main_oos_mdd_improve_vs_bh"] > 1e-6)).sum())
    strict_cnt = int(((res["main_oos_bh_multiple"] >= 0.95) & (res["main_oos_mdd_improve_vs_bh"] > 1e-6)).sum())
    robust_cnt = int(res["robust_pass"].sum())

    print("\nTop 10 candidates:")
    print(
        res[
            [
                "candidate_id",
                "dd_trigger",
                "atr_ratio_trigger",
                "use_gap",
                "entry_confirm_days",
                "cap_shield",
                "max_shield_days",
                "exit_rebound_2d",
                "main_oos_bh_multiple",
                "main_oos_mdd_improve_vs_bh",
                "median_oos_multiple",
                "worst_oos_multiple",
                "robust_pass",
                "robust_score",
            ]
        ]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(
        f"\nCounts: near(>=0.85 & MDD improve)={near_cnt}, "
        f"strict(>=0.95 & MDD improve)={strict_cnt}, robust_pass={robust_cnt}"
    )
    if not shift_df.empty:
        bshift = shift_df[shift_df["candidate_id"] == best_id]
        print(
            f"Best candidate shift robustness: "
            f"main_oos_mult min={bshift['main_oos_bh_multiple'].min():.4f}, "
            f"max={bshift['main_oos_bh_multiple'].max():.4f}"
        )
    if not sens_df.empty:
        base_s = sens_df[sens_df["variant"] == "base"]
        if not base_s.empty:
            print(
                f"Sensitivity (base): main_oos_mult={base_s.iloc[0]['main_oos_bh_multiple']:.4f}, "
                f"median_oos_mult={base_s.iloc[0]['median_oos_multiple']:.4f}"
            )

    print(f"\nSaved all: {all_path}")
    print(f"Saved top: {top_path}")
    if not shift_df.empty:
        print(f"Saved shift: {shift_path}")
    if not sens_df.empty:
        print(f"Saved sensitivity: {sens_path}")
    print(f"Saved recent diag: {diag_path}")
    if not activity_df.empty:
        print(f"Saved strategy activity: {act_path}")
    print(f"Saved best weights: {w_path}")
    print(f"Saved best equity: {eq_path}")
    print(f"Saved best debug: {dbg_path}")


if __name__ == "__main__":
    main()
