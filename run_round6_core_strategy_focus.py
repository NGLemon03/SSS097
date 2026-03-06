from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import run_round4_overlay_test as r4
from SSS_EnsembleTab import CostParams, EnsembleParams, RunConfig, calculate_performance, run_ensemble
from sss_core.logic import load_data


@dataclass(frozen=True)
class StrategyItem:
    name: str
    strategy_type: str


def _load_warehouse(path: Path) -> list[StrategyItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[StrategyItem] = []
    for s in data.get("strategies", []):
        name = str(s.get("name", "")).replace(".csv", "").strip()
        st = str(s.get("type", "")).strip().lower()
        if name:
            out.append(StrategyItem(name=name, strategy_type=st))
    return out


def _load_position_and_buys(trade_path: Path, index: pd.DatetimeIndex) -> tuple[pd.Series | None, np.ndarray]:
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
    buy_mask = side.isin(["buy", "entry", "long", "add"]) & dt.notna()
    buy_dates = np.sort(dt[buy_mask].values.astype("datetime64[ns]"))

    pos = pd.Series(0.0, index=index, dtype=float)
    for d, s in zip(dt, side):
        if pd.isna(d):
            continue
        i = index.searchsorted(d)
        if i >= len(index):
            continue
        if s in ("buy", "entry", "long", "add"):
            pos.iloc[i:] = 1.0
        elif s in ("sell", "exit", "short", "sell_forced", "forced_sell"):
            pos.iloc[i:] = 0.0
    return pos, buy_dates


def _apply_t_plus_1(w: pd.Series) -> pd.Series:
    return w.shift(1).fillna(0.0).clip(0, 1)


def _wealth_mult(ret: float, bh_ret: float) -> float:
    if (1.0 + bh_ret) <= 0:
        return float("nan")
    return float((1.0 + ret) / (1.0 + bh_ret))


def _parse_split_dates(raw: str) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    for x in raw.split(","):
        s = x.strip()
        if s:
            out.append(pd.Timestamp(s))
    return sorted(set(out))


def _eval_metrics(
    eq: pd.Series,
    bh_eq: pd.Series,
    main_split: pd.Timestamp,
    split_dates: list[pd.Timestamp],
) -> dict[str, float]:
    full = r4._compute_metrics(eq, bh_eq)
    oos = r4._compute_metrics(eq[eq.index > main_split], bh_eq[bh_eq.index > main_split])
    if not full or not oos:
        return {}

    full_mult = _wealth_mult(full["ret"], full["bh_ret"])
    oos_mult = _wealth_mult(oos["ret"], oos["bh_ret"])

    oos_mults: list[float] = []
    oos_mdd_imps: list[float] = []
    for sd in split_dates:
        st = r4._compute_metrics(eq[eq.index > sd], bh_eq[bh_eq.index > sd])
        if not st:
            continue
        oos_mults.append(_wealth_mult(st["ret"], st["bh_ret"]))
        oos_mdd_imps.append(float(st["mdd"] - st["bh_mdd"]))

    out: dict[str, float] = {
        "full_ret": float(full["ret"]),
        "full_mdd": float(full["mdd"]),
        "full_sharpe": float(full["sharpe"]),
        "full_bh_multiple": full_mult,
        "main_oos_ret": float(oos["ret"]),
        "main_oos_mdd": float(oos["mdd"]),
        "main_oos_sharpe": float(oos["sharpe"]),
        "main_oos_mdd_improve_vs_bh": float(oos["mdd"] - oos["bh_mdd"]),
        "main_oos_bh_multiple": oos_mult,
        "median_oos_multiple": float(np.nanmedian(oos_mults)) if oos_mults else float("nan"),
        "worst_oos_multiple": float(np.nanmin(oos_mults)) if oos_mults else float("nan"),
        "median_oos_mdd_improve": float(np.nanmedian(oos_mdd_imps)) if oos_mdd_imps else float("nan"),
        "worst_oos_mdd_improve": float(np.nanmin(oos_mdd_imps)) if oos_mdd_imps else float("nan"),
    }
    for i, sd in enumerate(split_dates):
        if i < len(oos_mults):
            out[f"oos_mult_{sd.date()}"] = float(oos_mults[i])
            out[f"oos_mdd_imp_{sd.date()}"] = float(oos_mdd_imps[i])
    return out


def _recent_buy_stats(buy_dates: np.ndarray, asof: pd.Timestamp, lookback_days: int) -> tuple[int, int | None]:
    if buy_dates.size == 0:
        return 0, None
    s = pd.to_datetime(buy_dates)
    recent = s[(s >= (asof - pd.Timedelta(days=lookback_days))) & (s <= asof)]
    last_buy = pd.Timestamp(s.max())
    return int(len(recent)), int((asof.normalize() - last_buy.normalize()).days)


def _run_weight(
    label: str,
    strategy_type: str,
    w: pd.Series,
    open_px: pd.Series,
    bh_eq: pd.Series,
    cost: CostParams,
    main_split: pd.Timestamp,
    split_dates: list[pd.Timestamp],
    recent_buy_count: int | None = None,
    days_since_last_buy: int | None = None,
) -> tuple[dict[str, object], pd.Series, pd.Series]:
    ww = w.reindex(open_px.index).ffill().fillna(0.0).clip(0, 1)
    eq, _, _, _ = calculate_performance(open_px, ww, cost)
    ev = _eval_metrics(eq, bh_eq, main_split, split_dates)
    if not ev:
        return {}, ww, eq

    row: dict[str, object] = {
        "label": label,
        "type": strategy_type,
        "turnover_py": float(ww.diff().abs().sum() / len(ww) * 252),
        "time_in_market": float((ww > 0.05).mean()),
        "recent_buy_count_180d": recent_buy_count,
        "days_since_last_buy": days_since_last_buy,
        **ev,
    }
    return row, ww, eq


def main() -> None:
    parser = argparse.ArgumentParser(description="Round6 focused test for single/RMA/turn/ensemble.")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--warehouse", default="analysis/strategy_warehouse.json")
    parser.add_argument("--include_archive", action="store_true")
    parser.add_argument("--main_split_date", default="2025-06-30")
    parser.add_argument("--split_dates", default="2024-12-31,2025-03-31,2025-06-30,2025-09-30")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--strict_t1", action="store_true", default=True)
    parser.add_argument("--lookback_days", type=int, default=180)
    args = parser.parse_args()

    warehouse_path = Path(args.warehouse)
    items = _load_warehouse(warehouse_path)
    if not items:
        raise SystemExit("No strategies in warehouse.")

    names = [x.name for x in items]
    type_map = {x.name: x.strategy_type for x in items}
    file_map = r4._resolve_trade_files(names, include_archive=args.include_archive)
    names = [n for n in names if n in file_map]
    if len(names) < 2:
        raise SystemExit("Need >=2 resolved strategy files.")

    df_price, _ = load_data(args.ticker)
    need = {"open", "close"}
    if df_price.empty or not need.issubset(df_price.columns):
        raise SystemExit("Price data unavailable.")

    open_px = pd.to_numeric(df_price["open"], errors="coerce").dropna().sort_index()
    close_px = pd.to_numeric(df_price["close"], errors="coerce").dropna().sort_index()
    idx = open_px.index.intersection(close_px.index)
    open_px = open_px.reindex(idx).dropna()
    close_px = close_px.reindex(idx).dropna()

    bh_eq = open_px / open_px.iloc[0] * 1_000_000.0
    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)
    main_split = pd.Timestamp(args.main_split_date)
    split_dates = _parse_split_dates(args.split_dates)
    if not split_dates:
        split_dates = [main_split]

    rows: list[dict[str, object]] = []
    w_cols: dict[str, pd.Series] = {}
    eq_cols: dict[str, pd.Series] = {}
    type_buckets: dict[str, list[pd.Series]] = {"single": [], "rma": [], "ssma_turn": []}

    asof = open_px.index[-1]
    for n in names:
        pos, buy_dates = _load_position_and_buys(file_map[n], open_px.index)
        if pos is None:
            continue
        w = _apply_t_plus_1(pos) if args.strict_t1 else pos
        st = type_map.get(n, "unknown").lower()

        recent_buy_count, days_since_last_buy = _recent_buy_stats(buy_dates, asof, args.lookback_days)
        row, ww, eq = _run_weight(
            label=n,
            strategy_type=st,
            w=w,
            open_px=open_px,
            bh_eq=bh_eq,
            cost=cost,
            main_split=main_split,
            split_dates=split_dates,
            recent_buy_count=recent_buy_count,
            days_since_last_buy=days_since_last_buy,
        )
        if row:
            rows.append(row)
            w_cols[n] = ww
            eq_cols[n] = eq
            if st in type_buckets:
                type_buckets[st].append(ww)

    # Family equal-weight aggregations.
    for st, arr in type_buckets.items():
        if not arr:
            continue
        w = pd.concat(arr, axis=1).mean(axis=1).clip(0, 1)
        label = f"{st}_family_eqw"
        row, ww, eq = _run_weight(
            label=label,
            strategy_type=f"{st}_family",
            w=w,
            open_px=open_px,
            bh_eq=bh_eq,
            cost=cost,
            main_split=main_split,
            split_dates=split_dates,
        )
        if row:
            rows.append(row)
            w_cols[label] = ww
            eq_cols[label] = eq

    # Ensemble equal-weight over all warehouse strategies.
    if w_cols:
        base_names = [n for n in names if n in w_cols]
        if base_names:
            w_ens_eqw = pd.concat([w_cols[n] for n in base_names], axis=1).mean(axis=1).clip(0, 1)
            row, ww, eq = _run_weight(
                label="ensemble_eqw",
                strategy_type="ensemble",
                w=w_ens_eqw,
                open_px=open_px,
                bh_eq=bh_eq,
                cost=cost,
                main_split=main_split,
                split_dates=split_dates,
            )
            if row:
                rows.append(row)
                w_cols["ensemble_eqw"] = ww
                eq_cols["ensemble_eqw"] = eq

    # Ensemble current proportional (same baseline style), with strict T+1 applied on weights.
    p = EnsembleParams(
        floor=0.5,
        ema_span=3,
        delta_cap=0.1,
        min_cooldown_days=1,
        min_trade_dw=0.01,
    )
    cfg = RunConfig(
        ticker=args.ticker,
        method="proportional",
        strategies=names,
        file_map=file_map,
        params=p,
        cost=cost,
    )
    open_run, w_run, *_ = run_ensemble(cfg)
    if w_run is not None and not w_run.empty:
        ww = w_run.reindex(open_px.index).ffill().fillna(0.0).clip(0, 1)
        if args.strict_t1:
            ww = _apply_t_plus_1(ww)
        row, ww2, eq2 = _run_weight(
            label="ensemble_current_proportional",
            strategy_type="ensemble",
            w=ww,
            open_px=open_px,
            bh_eq=bh_eq,
            cost=cost,
            main_split=main_split,
            split_dates=split_dates,
        )
        if row:
            rows.append(row)
            w_cols["ensemble_current_proportional"] = ww2
            eq_cols["ensemble_current_proportional"] = eq2

    if not rows:
        raise SystemExit("No rows generated.")

    res = pd.DataFrame(rows).sort_values(["type", "main_oos_bh_multiple"], ascending=[True, False])
    core = res[res["label"].isin(["single_family_eqw", "rma_family_eqw", "ssma_turn_family_eqw", "ensemble_current_proportional", "ensemble_eqw"])].copy()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_path = out_dir / f"round6_core_focus_all_{ts}.csv"
    core_path = out_dir / f"round6_core_focus_core4_{ts}.csv"
    w_path = out_dir / f"round6_core_focus_weights_{ts}.csv"
    eq_path = out_dir / f"round6_core_focus_equity_{ts}.csv"
    inactive_path = out_dir / f"round6_core_focus_inactive_180d_{ts}.csv"

    res.to_csv(all_path, index=False)
    core.to_csv(core_path, index=False)
    pd.DataFrame(w_cols, index=open_px.index).to_csv(w_path, index_label="date")
    pd.DataFrame(eq_cols, index=open_px.index).to_csv(eq_path, index_label="date")

    inact = res[(res["type"].isin(["single", "rma", "ssma_turn"])) & (res["recent_buy_count_180d"].fillna(0) <= 0)]
    inact.to_csv(inactive_path, index=False)

    print(f"Ticker: {args.ticker}")
    print(f"Range: {open_px.index[0].date()} -> {open_px.index[-1].date()} ({len(open_px)} bars)")
    print(f"Strict T+1: {args.strict_t1}")
    print(f"Main split: {main_split.date()}, multi-split: {', '.join([str(x.date()) for x in split_dates])}")
    print("\nCore 4 summary:")
    if core.empty:
        print("No core rows.")
    else:
        print(
            core[
                [
                    "label",
                    "main_oos_ret",
                    "main_oos_mdd",
                    "main_oos_bh_multiple",
                    "main_oos_mdd_improve_vs_bh",
                    "median_oos_multiple",
                    "worst_oos_multiple",
                    "turnover_py",
                    "time_in_market",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print(f"\nInactive strategies (180d no buy): {len(inact)}")
    print(f"\nSaved all: {all_path}")
    print(f"Saved core4: {core_path}")
    print(f"Saved weights: {w_path}")
    print(f"Saved equity: {eq_path}")
    print(f"Saved inactive: {inactive_path}")


if __name__ == "__main__":
    main()

