from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import run_round4_overlay_test as r4
from SSS_EnsembleTab import CostParams, EnsembleParams, RunConfig, calculate_performance, run_ensemble
from sss_core.logic import backtest_unified, calc_smaa, compute_single, load_data


def _load_warehouse(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for s in data.get("strategies", []):
        name = str(s.get("name", "")).replace(".csv", "").strip()
        st = str(s.get("type", "")).strip().lower()
        params = dict(s.get("params", {}))
        if name:
            out.append({"name": name, "type": st, "params": params})
    return out


def _parse_source_mode(strategy_name: str) -> str:
    if "Factor(TWII_2412.TW)" in strategy_name:
        return "Factor (^TWII / 2412.TW)"
    if "Factor(TWII_2414.TW)" in strategy_name:
        return "Factor (^TWII / 2414.TW)"
    return "Self"


def _parse_split_dates(raw: str) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    for x in raw.split(","):
        s = x.strip()
        if s:
            out.append(pd.Timestamp(s))
    return sorted(set(out))


def _wealth_mult(ret: float, bh_ret: float) -> float:
    if (1.0 + bh_ret) <= 0:
        return float("nan")
    return float((1.0 + ret) / (1.0 + bh_ret))


def _apply_t_plus_1(w: pd.Series) -> pd.Series:
    return w.shift(1).fillna(0.0).clip(0, 1)


def _eval(eq: pd.Series, bh_eq: pd.Series, main_split: pd.Timestamp, split_dates: list[pd.Timestamp]) -> dict[str, float]:
    full = r4._compute_metrics(eq, bh_eq)
    main = r4._compute_metrics(eq[eq.index > main_split], bh_eq[bh_eq.index > main_split])
    if not full or not main:
        return {}

    full_mult = _wealth_mult(full["ret"], full["bh_ret"])
    main_mult = _wealth_mult(main["ret"], main["bh_ret"])
    mults: list[float] = []
    mdd_imps: list[float] = []
    for sd in split_dates:
        st = r4._compute_metrics(eq[eq.index > sd], bh_eq[bh_eq.index > sd])
        if not st:
            continue
        mults.append(_wealth_mult(st["ret"], st["bh_ret"]))
        mdd_imps.append(float(st["mdd"] - st["bh_mdd"]))

    out = {
        "full_ret": float(full["ret"]),
        "full_mdd": float(full["mdd"]),
        "full_bh_multiple": full_mult,
        "main_oos_ret": float(main["ret"]),
        "main_oos_mdd": float(main["mdd"]),
        "main_oos_bh_multiple": main_mult,
        "main_oos_mdd_improve_vs_bh": float(main["mdd"] - main["bh_mdd"]),
        "main_oos_sharpe": float(main["sharpe"]),
        "median_oos_multiple": float(np.nanmedian(mults)) if mults else float("nan"),
        "worst_oos_multiple": float(np.nanmin(mults)) if mults else float("nan"),
        "median_oos_mdd_improve": float(np.nanmedian(mdd_imps)) if mdd_imps else float("nan"),
        "worst_oos_mdd_improve": float(np.nanmin(mdd_imps)) if mdd_imps else float("nan"),
    }
    for i, sd in enumerate(split_dates):
        if i < len(mults):
            out[f"oos_mult_{sd.date()}"] = float(mults[i])
            out[f"oos_mdd_imp_{sd.date()}"] = float(mdd_imps[i])
    return out


def _turn_stage_diagnose(
    df_price: pd.DataFrame,
    df_factor: pd.DataFrame,
    params: dict[str, Any],
    lookback_days: int,
) -> dict[str, Any]:
    source_df = df_factor if not df_factor.empty else df_price
    if source_df.empty or "close" not in source_df.columns or "volume" not in source_df.columns:
        return {"diag_error": "source_missing_close_or_volume"}
    if "volume" not in df_price.columns:
        return {"diag_error": "target_missing_volume"}

    try:
        linlen = int(params.get("linlen"))
        factor = float(params.get("factor"))
        smaalen = int(params.get("smaalen"))
        prom_factor = float(params.get("prom_factor"))
        min_dist = int(params.get("min_dist"))
        buy_shift = max(0, int(params.get("buy_shift", 0)))
        exit_shift = max(0, int(params.get("exit_shift", 0)))
        vol_window = int(params.get("vol_window", 20))
        quantile_win = max(int(params.get("quantile_win", 100)), vol_window)
        signal_cooldown_days = int(params.get("signal_cooldown_days", 10))
        volume_target_pass_rate = float(params.get("volume_target_pass_rate", 0.65))
        volume_target_pass_rate = min(max(volume_target_pass_rate, 0.05), 0.95)
        volume_target_lookback = max(int(params.get("volume_target_lookback", 252)), vol_window, 20)
    except Exception:
        return {"diag_error": "param_cast_failed"}

    dfc = source_df.dropna(subset=["close"]).copy()
    dfc["close"] = pd.to_numeric(dfc["close"], errors="coerce")
    smaa = calc_smaa(dfc["close"].round(6), linlen, factor, smaalen)
    series_clean = smaa.dropna()
    if series_clean.empty or len(series_clean) < (quantile_win + 3):
        return {"diag_error": "insufficient_smaa_history"}

    prom = series_clean.rolling(window=min_dist + 1, min_periods=min_dist + 1).apply(lambda x: x.ptp(), raw=True)
    initial_threshold = prom.quantile(prom_factor / 100.0) if len(prom.dropna()) > 0 else prom.median()
    threshold_series = (
        prom.rolling(window=quantile_win, min_periods=quantile_win)
        .quantile(prom_factor / 100.0)
        .shift(1)
        .ffill()
        .fillna(initial_threshold)
    )

    peaks: list[pd.Timestamp] = []
    valleys: list[pd.Timestamp] = []
    # Match existing logic behavior: last_signal_dt is never updated there.
    last_signal_dt = None
    for i in range(quantile_win, len(series_clean)):
        if last_signal_dt is not None and (series_clean.index[i] - last_signal_dt).days <= signal_cooldown_days:
            continue
        window_data = series_clean.iloc[max(0, i - quantile_win) : i + 1].to_numpy()
        if len(window_data) < min_dist + 1:
            continue
        current_threshold = threshold_series.iloc[i]
        wpeaks, _ = find_peaks(window_data, distance=min_dist, prominence=current_threshold)
        wvalleys, _ = find_peaks(-window_data, distance=min_dist, prominence=current_threshold)
        wstart = max(0, i - quantile_win)
        for p_idx in wpeaks:
            d = series_clean.index[wstart + p_idx]
            if d not in peaks:
                peaks.append(d)
        for v_idx in wvalleys:
            d = series_clean.index[wstart + v_idx]
            if d not in valleys:
                valleys.append(d)

    vol_series = pd.to_numeric(df_price["volume"], errors="coerce")
    target_quantile = 1.0 - volume_target_pass_rate
    vol_threshold = (
        vol_series
        .rolling(volume_target_lookback, min_periods=volume_target_lookback)
        .quantile(target_quantile)
        .shift(1)
    )
    vol_ma = vol_series.rolling(vol_window, min_periods=vol_window).mean().shift(1)
    vol_threshold = vol_threshold.combine_first(vol_ma).ffill()

    def _filter_with_volume(dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
        valid: list[pd.Timestamp] = []
        for d in dates:
            if d in vol_threshold.index and d in df_price.index:
                if pd.notna(vol_threshold.loc[d]) and pd.notna(vol_series.loc[d]) and vol_series.loc[d] > vol_threshold.loc[d]:
                    valid.append(d)
        return valid

    vol_peaks = _filter_with_volume(peaks)
    vol_valleys = _filter_with_volume(valleys)

    def _apply_cooldown(dates: list[pd.Timestamp], cooldown_days: int) -> list[pd.Timestamp]:
        filtered: list[pd.Timestamp] = []
        last_date = pd.Timestamp("1900-01-01")
        for d in sorted(dates):
            if (d - last_date).days >= cooldown_days:
                filtered.append(d)
                last_date = d
        return filtered

    cd_peaks = _apply_cooldown(vol_peaks, signal_cooldown_days)
    cd_valleys = _apply_cooldown(vol_valleys, signal_cooldown_days)

    buy_dates: list[pd.Timestamp] = []
    sell_dates: list[pd.Timestamp] = []
    for dt in cd_valleys:
        if dt in df_price.index:
            i = df_price.index.get_loc(dt) + 1 + buy_shift
            if 0 <= i < len(df_price):
                buy_dates.append(df_price.index[i])
    for dt in cd_peaks:
        if dt in df_price.index:
            i = df_price.index.get_loc(dt) + 1 + exit_shift
            if 0 <= i < len(df_price):
                sell_dates.append(df_price.index[i])

    asof = df_price.index.max()
    recent_start = asof - pd.Timedelta(days=lookback_days)

    def _recent_count(dates: list[pd.Timestamp]) -> int:
        return int(sum(1 for d in dates if d >= recent_start))

    raw_v_r = _recent_count(valleys)
    vol_v_r = _recent_count(vol_valleys)
    cd_v_r = _recent_count(cd_valleys)
    buy_r = _recent_count(buy_dates)

    if raw_v_r <= 0:
        blocker = "no_raw_valley_signal"
    elif vol_v_r <= 0:
        blocker = "blocked_by_volume_filter"
    elif cd_v_r <= 0:
        blocker = "blocked_by_signal_cooldown"
    elif buy_r <= 0:
        blocker = "blocked_by_buy_shift_or_calendar"
    else:
        blocker = "active_recently"

    last_buy = max(buy_dates) if buy_dates else pd.NaT
    days_since_last_buy = None if pd.isna(last_buy) else int((asof.normalize() - last_buy.normalize()).days)

    return {
        "source_rows": int(len(source_df)),
        "source_factor_used": bool(not df_factor.empty),
        "raw_peaks": int(len(peaks)),
        "raw_valleys": int(len(valleys)),
        "volume_peaks": int(len(vol_peaks)),
        "volume_valleys": int(len(vol_valleys)),
        "cooldown_peaks": int(len(cd_peaks)),
        "cooldown_valleys": int(len(cd_valleys)),
        "final_buy_dates": int(len(buy_dates)),
        "final_sell_dates": int(len(sell_dates)),
        "raw_valleys_recent": int(raw_v_r),
        "volume_valleys_recent": int(vol_v_r),
        "cooldown_valleys_recent": int(cd_v_r),
        "final_buys_recent": int(buy_r),
        "days_since_last_buy": days_since_last_buy,
        "last_buy_date": "" if pd.isna(last_buy) else str(last_buy.date()),
        "blocker_reason": blocker,
        "volume_filter_mode": "rolling_quantile_target_pass",
        "volume_target_pass_rate": float(volume_target_pass_rate),
        "volume_target_lookback": int(volume_target_lookback),
        "volume_threshold_quantile": float(target_quantile),
        "valley_pass_volume_rate": float(len(vol_valleys) / len(valleys)) if len(valleys) > 0 else np.nan,
        "valley_pass_cooldown_rate": float(len(cd_valleys) / len(vol_valleys)) if len(vol_valleys) > 0 else np.nan,
        "valley_pass_shift_rate": float(len(buy_dates) / len(cd_valleys)) if len(cd_valleys) > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Follow-up tests: ensemble/single perturbation + turn hit diagnostics.")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--warehouse", default="analysis/strategy_warehouse.json")
    parser.add_argument("--include_archive", action="store_true")
    parser.add_argument("--main_split_date", default="2025-06-30")
    parser.add_argument("--split_dates", default="2024-12-31,2025-03-31,2025-06-30,2025-09-30")
    parser.add_argument("--lookback_days", type=int, default=180)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    split_dates = _parse_split_dates(args.split_dates)
    main_split = pd.Timestamp(args.main_split_date)

    items = _load_warehouse(Path(args.warehouse))
    if not items:
        raise SystemExit("No strategies loaded.")

    names = [x["name"] for x in items]
    type_map = {x["name"]: x["type"] for x in items}
    param_map = {x["name"]: x["params"] for x in items}
    file_map = r4._resolve_trade_files(names, include_archive=args.include_archive)
    names = [n for n in names if n in file_map]
    if len(names) < 2:
        raise SystemExit("Need >=2 strategy files.")

    df_price, _ = load_data(args.ticker, smaa_source="Self")
    if df_price.empty or "open" not in df_price.columns:
        raise SystemExit("Target price unavailable.")
    open_px = pd.to_numeric(df_price["open"], errors="coerce").dropna().sort_index()
    idx = open_px.index
    bh_eq = open_px / open_px.iloc[0] * 1_000_000.0
    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)

    # 1) Ensemble local perturbation.
    ens_rows: list[dict[str, Any]] = []
    ens_weights: dict[str, pd.Series] = {}
    ens_grid = []
    for floor in [0.45, 0.50, 0.55]:
        for ema_span in [2, 3, 4]:
            for delta_cap in [0.08, 0.10, 0.12]:
                for min_trade_dw in [0.005, 0.010, 0.015]:
                    ens_grid.append((floor, ema_span, delta_cap, min_trade_dw))

    for i, (floor, ema_span, delta_cap, min_trade_dw) in enumerate(ens_grid, 1):
        p = EnsembleParams(
            floor=floor,
            ema_span=ema_span,
            delta_cap=delta_cap,
            min_cooldown_days=1,
            min_trade_dw=min_trade_dw,
        )
        cfg = RunConfig(
            ticker=args.ticker,
            method="proportional",
            strategies=names,
            file_map=file_map,
            params=p,
            cost=cost,
        )
        _, w, _, _, _, _, _, _ = run_ensemble(cfg)
        if w is None or w.empty:
            continue
        w_t1 = _apply_t_plus_1(w.reindex(idx).ffill().fillna(0.0))
        eq, _, _, _ = calculate_performance(open_px, w_t1, cost)
        ev = _eval(eq, bh_eq, main_split, split_dates)
        if not ev:
            continue
        tag = f"f{floor:.2f}_e{ema_span}_d{delta_cap:.3f}_m{min_trade_dw:.3f}"
        ens_weights[tag] = w_t1
        ens_rows.append(
            {
                "param_tag": tag,
                "floor": floor,
                "ema_span": ema_span,
                "delta_cap": delta_cap,
                "min_trade_dw": min_trade_dw,
                "turnover_py": float(w_t1.diff().abs().sum() / len(w_t1) * 252),
                "time_in_market": float((w_t1 > 0.05).mean()),
                **ev,
            }
        )
        if i % 20 == 0 or i == len(ens_grid):
            print(f"Ensemble perturb progress: {i}/{len(ens_grid)}")

    ens_df = pd.DataFrame(ens_rows).sort_values("main_oos_bh_multiple", ascending=False) if ens_rows else pd.DataFrame()

    # 2) Single local perturbation.
    singles = [n for n in names if type_map.get(n, "") == "single"]
    single_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    single_weights: dict[str, pd.Series] = {}
    family_weights: dict[str, pd.Series] = {}

    data_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    single_grid = []
    for dev_m in [0.9, 1.0, 1.1]:
        for buy_m in [0.9, 1.0, 1.1]:
            for sell_m in [0.9, 1.0, 1.1]:
                for sl_m in [0.8, 1.0, 1.2]:
                    single_grid.append((dev_m, buy_m, sell_m, sl_m))

    combo_w: dict[tuple[float, float, float, float], list[pd.Series]] = {}
    for sname in singles:
        p0 = dict(param_map.get(sname, {}))
        mode = _parse_source_mode(sname)
        if mode not in data_cache:
            data_cache[mode] = load_data(args.ticker, smaa_source=mode)
        dfp, dff = data_cache[mode]
        if dfp.empty:
            continue
        dfi = dfp.reindex(idx).dropna(subset=["open", "close"])

        for j, (dev_m, buy_m, sell_m, sl_m) in enumerate(single_grid, 1):
            devwin = max(5, int(round(float(p0.get("devwin", 30)) * dev_m)))
            buy_mult = float(p0.get("buy_mult", 1.0)) * buy_m
            sell_mult = float(p0.get("sell_mult", 1.0)) * sell_m
            stop_loss = max(0.0, float(p0.get("stop_loss", 0.0)) * sl_m)

            df_ind = compute_single(
                dfi,
                dff.reindex(dfi.index).dropna() if not dff.empty else pd.DataFrame(),
                linlen=int(p0.get("linlen", 20)),
                factor=float(p0.get("factor", 10)),
                smaalen=int(p0.get("smaalen", 20)),
                devwin=devwin,
                smaa_source=mode,
            )
            if df_ind.empty:
                continue

            bt_params = {
                "buy_mult": buy_mult,
                "sell_mult": sell_mult,
                "stop_loss": stop_loss,
            }
            bt = backtest_unified(
                df_ind=df_ind,
                strategy_type="single",
                params=bt_params.copy(),
                buy_dates=None,
                sell_dates=None,
                discount=0.30,
                trade_cooldown_bars=3,
                bad_holding=(stop_loss > 0),
                use_leverage=False,
                lev_params=None,
            )
            w = bt.get("weight_curve", pd.Series(dtype=float))
            if w is None or w.empty:
                continue
            wv = w.reindex(idx).ffill().fillna(0.0).clip(0, 1)
            eq, _, _, _ = calculate_performance(open_px, wv, cost)
            ev = _eval(eq, bh_eq, main_split, split_dates)
            if not ev:
                continue

            key = (dev_m, buy_m, sell_m, sl_m)
            combo_w.setdefault(key, []).append(wv)
            tag = f"{sname}|dev{dev_m:.1f}|buy{buy_m:.1f}|sell{sell_m:.1f}|sl{sl_m:.1f}"
            single_weights[tag] = wv
            single_rows.append(
                {
                    "strategy": sname,
                    "source_mode": mode,
                    "dev_m": dev_m,
                    "buy_m": buy_m,
                    "sell_m": sell_m,
                    "sl_m": sl_m,
                    "devwin": devwin,
                    "buy_mult": buy_mult,
                    "sell_mult": sell_mult,
                    "stop_loss": stop_loss,
                    "turnover_py": float(wv.diff().abs().sum() / len(wv) * 252),
                    "time_in_market": float((wv > 0.05).mean()),
                    **ev,
                }
            )
            if j % 30 == 0 or j == len(single_grid):
                print(f"Single perturb progress ({sname}): {j}/{len(single_grid)}")

    # Single family equal-weight (for each perturb combo).
    for (dev_m, buy_m, sell_m, sl_m), arr in combo_w.items():
        if not arr:
            continue
        wf = pd.concat(arr, axis=1).mean(axis=1).clip(0, 1)
        eq, _, _, _ = calculate_performance(open_px, wf, cost)
        ev = _eval(eq, bh_eq, main_split, split_dates)
        if not ev:
            continue
        tag = f"single_family|dev{dev_m:.1f}|buy{buy_m:.1f}|sell{sell_m:.1f}|sl{sl_m:.1f}"
        family_weights[tag] = wf
        family_rows.append(
            {
                "dev_m": dev_m,
                "buy_m": buy_m,
                "sell_m": sell_m,
                "sl_m": sl_m,
                "turnover_py": float(wf.diff().abs().sum() / len(wf) * 252),
                "time_in_market": float((wf > 0.05).mean()),
                **ev,
            }
        )

    single_df = pd.DataFrame(single_rows).sort_values("main_oos_bh_multiple", ascending=False) if single_rows else pd.DataFrame()
    family_df = pd.DataFrame(family_rows).sort_values("main_oos_bh_multiple", ascending=False) if family_rows else pd.DataFrame()

    # 3) Turn stage hit-rate diagnostics.
    turns = [n for n in names if type_map.get(n, "") == "ssma_turn"]
    turn_rows: list[dict[str, Any]] = []
    for tname in turns:
        mode = _parse_source_mode(tname)
        if mode not in data_cache:
            data_cache[mode] = load_data(args.ticker, smaa_source=mode)
        dfi, dff = data_cache[mode]
        if dfi.empty:
            continue
        dfi = dfi.reindex(idx).dropna(subset=["open", "close", "volume"])
        diag = _turn_stage_diagnose(dfi, dff.reindex(dfi.index).dropna() if not dff.empty else pd.DataFrame(), param_map[tname], args.lookback_days)
        turn_rows.append({"strategy": tname, "source_mode": mode, **diag})

    turn_df = pd.DataFrame(turn_rows).sort_values(["final_buys_recent", "raw_valleys_recent"], ascending=[True, True]) if turn_rows else pd.DataFrame()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ens_path = out_dir / f"round6_followup_ensemble_perturb_{ts}.csv"
    single_path = out_dir / f"round6_followup_single_perturb_{ts}.csv"
    fam_path = out_dir / f"round6_followup_single_family_perturb_{ts}.csv"
    turn_path = out_dir / f"round6_followup_turn_diagnose_{ts}.csv"
    weight_path = out_dir / f"round6_followup_weights_{ts}.csv"

    if not ens_df.empty:
        ens_df.to_csv(ens_path, index=False)
    if not single_df.empty:
        single_df.to_csv(single_path, index=False)
    if not family_df.empty:
        family_df.to_csv(fam_path, index=False)
    if not turn_df.empty:
        turn_df.to_csv(turn_path, index=False)

    # Save only best representative weight curves.
    w_dump: dict[str, pd.Series] = {}
    if not ens_df.empty:
        k = str(ens_df.iloc[0]["param_tag"])
        if k in ens_weights:
            w_dump["best_ensemble_perturb_w"] = ens_weights[k]
    if not family_df.empty:
        k = (
            f"single_family|dev{family_df.iloc[0]['dev_m']:.1f}|buy{family_df.iloc[0]['buy_m']:.1f}|"
            f"sell{family_df.iloc[0]['sell_m']:.1f}|sl{family_df.iloc[0]['sl_m']:.1f}"
        )
        if k in family_weights:
            w_dump["best_single_family_perturb_w"] = family_weights[k]
    if w_dump:
        pd.DataFrame(w_dump, index=idx).to_csv(weight_path, index_label="date")

    print(f"Ticker: {args.ticker}")
    print(f"Main split: {main_split.date()}, multi-split: {', '.join([str(x.date()) for x in split_dates])}")
    print(f"Single strategies: {len(singles)}, turn strategies: {len(turns)}")

    if not ens_df.empty:
        top = ens_df.iloc[0]
        print(
            f"\nBest ensemble perturb: {top['param_tag']} | "
            f"main_mult={top['main_oos_bh_multiple']:.4f}, main_mdd={top['main_oos_mdd']:.4f}, "
            f"mdd_imp={top['main_oos_mdd_improve_vs_bh']:.4f}"
        )
    if not family_df.empty:
        top = family_df.iloc[0]
        print(
            f"Best single-family perturb: dev={top['dev_m']:.1f}, buy={top['buy_m']:.1f}, "
            f"sell={top['sell_m']:.1f}, sl={top['sl_m']:.1f} | "
            f"main_mult={top['main_oos_bh_multiple']:.4f}, main_mdd={top['main_oos_mdd']:.4f}, "
            f"mdd_imp={top['main_oos_mdd_improve_vs_bh']:.4f}"
        )
    if not turn_df.empty:
        print("\nTurn diagnostics:")
        show_cols = [
            "strategy",
            "source_mode",
            "raw_valleys_recent",
            "volume_valleys_recent",
            "cooldown_valleys_recent",
            "final_buys_recent",
            "days_since_last_buy",
            "blocker_reason",
            "valley_pass_volume_rate",
            "valley_pass_cooldown_rate",
            "valley_pass_shift_rate",
        ]
        print(turn_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not ens_df.empty:
        print(f"\nSaved ensemble perturb: {ens_path}")
    if not single_df.empty:
        print(f"Saved single perturb: {single_path}")
    if not family_df.empty:
        print(f"Saved single-family perturb: {fam_path}")
    if not turn_df.empty:
        print(f"Saved turn diagnostics: {turn_path}")
    if w_dump:
        print(f"Saved best weights: {weight_path}")


if __name__ == "__main__":
    main()
