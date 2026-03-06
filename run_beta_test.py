from __future__ import annotations

import argparse
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
    compute_risk_valve_signals,
    run_ensemble,
)
from analysis.strategy_manager import manager
from sss_core.logic import load_data


@dataclass(frozen=True)
class BetaConfig:
    label: str
    method: str
    floor: float
    delta_cap: float
    ema_span: int = 3
    min_cooldown_days: int = 1
    min_trade_dw: float = 0.01
    majority_k_pct: float | None = None


DEFAULT_CONFIGS: tuple[BetaConfig, ...] = (
    BetaConfig(
        label="current_majority",
        method="majority",
        floor=0.2,
        delta_cap=0.3,
        majority_k_pct=0.55,
    ),
    BetaConfig(
        label="current_proportional",
        method="proportional",
        floor=0.5,
        delta_cap=0.1,
    ),
    BetaConfig(
        label="beta_majority_f05_d01",
        method="majority",
        floor=0.5,
        delta_cap=0.1,
        majority_k_pct=0.55,
    ),
    BetaConfig(
        label="beta_majority_f05_d02",
        method="majority",
        floor=0.5,
        delta_cap=0.2,
        majority_k_pct=0.55,
    ),
    BetaConfig(
        label="beta_proportional_f03_d01",
        method="proportional",
        floor=0.3,
        delta_cap=0.1,
    ),
)


def _load_active_strategies() -> list[str]:
    payload = manager.load_strategies()
    names: list[str] = []
    for item in payload:
        name = str(item.get("name", "")).replace(".csv", "")
        if name:
            names.append(name)
    return names


def _build_file_map(strategy_names: Iterable[str], search_dirs: list[Path]) -> dict[str, Path]:
    file_map: dict[str, Path] = {}
    for s_name in strategy_names:
        for folder in search_dirs:
            matches = list(folder.rglob(f"*{s_name}*.csv"))
            if not matches:
                continue
            best = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            file_map[s_name] = best
            break
    return file_map


def _build_position_series(trade_path: Path, index: pd.DatetimeIndex) -> pd.Series | None:
    try:
        df = pd.read_csv(trade_path)
    except Exception:
        return None

    cols = {str(c).lower().strip(): c for c in df.columns}
    date_col = cols.get("trade_date", cols.get("date"))
    type_col = cols.get("type", cols.get("action"))
    if date_col is None or type_col is None:
        return None

    out = pd.Series(0.0, index=index, dtype=float)
    dt = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    side = df[type_col].astype(str).str.lower().str.strip()

    for d, s in zip(dt, side):
        if pd.isna(d):
            continue
        loc = index.searchsorted(d)
        if loc >= len(index):
            continue
        if s in ("buy", "entry", "long", "add"):
            out.iloc[loc:] = 1.0
        elif s in ("sell", "exit", "short", "sell_forced", "forced_sell"):
            out.iloc[loc:] = 0.0

    return out


def _dedupe_by_corr(
    strategy_names: list[str],
    file_map: dict[str, Path],
    benchmark_index: pd.DatetimeIndex,
    max_corr: float,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    kept: list[str] = []
    kept_pos: dict[str, pd.Series] = {}
    dropped: list[tuple[str, str, float]] = []

    for name in strategy_names:
        path = file_map.get(name)
        if path is None:
            continue

        pos = _build_position_series(path, benchmark_index)
        if pos is None or pos.std() == 0:
            dropped.append((name, "invalid_or_zero_variance", float("nan")))
            continue

        too_close = False
        for kept_name, kept_series in kept_pos.items():
            corr = pos.corr(kept_series)
            if pd.notna(corr) and corr > max_corr:
                dropped.append((name, kept_name, float(corr)))
                too_close = True
                break

        if too_close:
            continue

        kept.append(name)
        kept_pos[name] = pos

    return kept, dropped


def _calc_metrics(
    equity: pd.Series,
    bench_close: pd.Series,
    split_date: pd.Timestamp,
) -> dict[str, float]:
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()
    eq = eq[eq.index > split_date]
    if len(eq) < 2:
        return {}

    bench = bench_close.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench.sort_index()
    bench = bench[bench.index > split_date]

    common_idx = eq.index.intersection(bench.index)
    if len(common_idx) < 2:
        return {}

    eq = eq.reindex(common_idx).ffill().dropna()
    bench = bench.reindex(common_idx).ffill().dropna()
    if len(eq) < 2 or len(bench) < 2:
        return {}

    ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1)
    alpha = ret - bench_ret

    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())

    bench_dd = bench / bench.cummax() - 1
    bench_mdd = float(bench_dd.min())

    dr = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan")
    vol = float(dr.std() * np.sqrt(252)) if len(dr) > 1 else float("nan")

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 252)
    annual_ret = float((1 + ret) ** (1 / years) - 1) if (1 + ret) > 0 else float("nan")

    return {
        "oos_ret": ret,
        "oos_ann_ret": annual_ret,
        "oos_mdd": mdd,
        "oos_sharpe": sharpe,
        "oos_vol": vol,
        "oos_alpha_vs_bh": alpha,
        "bh_ret": bench_ret,
        "bh_mdd": bench_mdd,
    }


def _run_one_config(
    ticker: str,
    strategy_names: list[str],
    file_map: dict[str, Path],
    cfg: BetaConfig,
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    ens_params = EnsembleParams(
        floor=cfg.floor,
        ema_span=cfg.ema_span,
        delta_cap=cfg.delta_cap,
        min_cooldown_days=cfg.min_cooldown_days,
        min_trade_dw=cfg.min_trade_dw,
    )
    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)
    run_cfg = RunConfig(
        ticker=ticker,
        method=cfg.method,
        strategies=strategy_names,
        file_map=file_map,
        params=ens_params,
        cost=cost,
        majority_k_pct=cfg.majority_k_pct,
    )
    open_px, w, _, _, _, equity, _, _ = run_ensemble(run_cfg)
    if equity is None or equity.empty:
        return None, None, None
    return equity, w, open_px


def _build_hybrid_weights(
    w_aggr: pd.Series,
    w_def: pd.Series,
    benchmark_df: pd.DataFrame,
    risk_cap: float = 0.45,
    trigger_in_days: int = 2,
    release_days: int = 3,
    max_daily_dw: float = 0.15,
) -> tuple[pd.Series, pd.Series]:
    # Risk trigger: trend weakness OR volatility expansion.
    risk_df = compute_risk_valve_signals(
        benchmark_df,
        slope20_thresh=0.0,
        slope60_thresh=0.0,
        atr_win=20,
        atr_ref_win=60,
        atr_ratio_mult=1.35,
        use_slopes=True,
        slope_method="polyfit",
        atr_cmp="gt",
    )
    mask = risk_df["risk_trigger"].reindex(w_aggr.index).fillna(False).astype(bool)

    wa = w_aggr.reindex(w_def.index).ffill().fillna(0.0)
    wd = w_def.reindex(w_aggr.index).ffill().fillna(0.0)
    idx = wa.index.intersection(wd.index)
    wa = wa.reindex(idx).ffill().fillna(0.0)
    wd = wd.reindex(idx).ffill().fillna(0.0)
    mask = mask.reindex(idx).fillna(False).astype(bool)

    target = pd.Series(index=idx, dtype=float)
    state_defense = False
    risk_count = 0
    calm_count = 0

    for dt in idx:
        if bool(mask.loc[dt]):
            risk_count += 1
            calm_count = 0
        else:
            calm_count += 1
            risk_count = 0

        if (not state_defense) and risk_count >= trigger_in_days:
            state_defense = True
        elif state_defense and calm_count >= release_days:
            state_defense = False

        if state_defense:
            target.loc[dt] = min(float(wa.loc[dt]), float(wd.loc[dt]), risk_cap)
        else:
            target.loc[dt] = float(wa.loc[dt])

    out = target.copy()
    for i in range(1, len(out)):
        prev = float(out.iloc[i - 1])
        cur = float(out.iloc[i])
        delta = cur - prev
        if abs(delta) > max_daily_dw:
            out.iloc[i] = prev + (max_daily_dw if delta > 0 else -max_daily_dw)

    return out.clip(0.0, 1.0), mask.reindex(out.index).fillna(False).astype(bool)


def _print_compact(df: pd.DataFrame) -> None:
    if df.empty:
        print("No valid runs.")
        return

    show_cols = [
        "set_name",
        "config",
        "n_strategies",
        "oos_ret",
        "oos_mdd",
        "oos_alpha_vs_bh",
        "oos_sharpe",
    ]
    if "risk_trigger_pct" in df.columns:
        show_cols.append("risk_trigger_pct")
    view = df[show_cols].copy()
    for c in ("oos_ret", "oos_mdd", "oos_alpha_vs_bh"):
        view[c] = view[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "nan")
    view["oos_sharpe"] = view["oos_sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
    if "risk_trigger_pct" in view.columns:
        view["risk_trigger_pct"] = view["risk_trigger_pct"].map(
            lambda x: f"{x:.2%}" if pd.notna(x) else "nan"
        )
    print(view.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run beta strategy tests on active warehouse.")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--split_date", default="2025-06-30")
    parser.add_argument("--max_corr", type=float, default=0.95)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--include_archive", action="store_true")
    args = parser.parse_args()

    split_date = pd.Timestamp(args.split_date)

    active_names = _load_active_strategies()
    if not active_names:
        raise SystemExit("No active strategies in analysis/strategy_warehouse.json")

    search_dirs = [Path("sss_backtest_outputs")]
    if args.include_archive:
        search_dirs.append(Path("archive"))

    file_map = _build_file_map(active_names, search_dirs)
    if not file_map:
        raise SystemExit("No strategy trade files found under search dirs.")

    df_bench, _ = load_data(args.ticker)
    if df_bench.empty or "close" not in df_bench.columns:
        raise SystemExit(f"Benchmark data unavailable for {args.ticker}")
    bench_close = pd.to_numeric(df_bench["close"], errors="coerce").dropna()
    bench_index = pd.DatetimeIndex(pd.to_datetime(bench_close.index).normalize().unique()).sort_values()

    raw_names = [n for n in active_names if n in file_map]
    dedup_names, dropped = _dedupe_by_corr(raw_names, file_map, bench_index, args.max_corr)

    print("Active strategies:", len(active_names))
    print("Found trade files:", len(raw_names))
    print("Dedup kept:", len(dedup_names), "dropped:", len(dropped))

    if dropped:
        print("\nDropped strategies:")
        for name, reason, corr in dropped:
            if pd.isna(corr):
                print(f"- {name} ({reason})")
            else:
                print(f"- {name} (corr={corr:.4f} vs {reason})")

    run_sets = {
        "raw": raw_names,
        "dedup": dedup_names if dedup_names else raw_names,
    }

    rows: list[dict[str, object]] = []
    run_cache: dict[str, dict[str, dict[str, object]]] = {"raw": {}, "dedup": {}}
    for set_name, names in run_sets.items():
        for cfg in DEFAULT_CONFIGS:
            equity, w, open_px = _run_one_config(
                ticker=args.ticker,
                strategy_names=names,
                file_map=file_map,
                cfg=cfg,
            )
            if equity is None:
                continue

            metrics = _calc_metrics(equity, bench_close, split_date)
            if not metrics:
                continue

            row = {
                "set_name": set_name,
                "config": cfg.label,
                "method": cfg.method,
                "floor": cfg.floor,
                "delta_cap": cfg.delta_cap,
                "n_strategies": len(names),
            }
            row.update(metrics)
            row["score_ret_mdd"] = float(row["oos_ret"] - 0.8 * abs(row["oos_mdd"]))
            rows.append(row)
            run_cache[set_name][cfg.label] = {
                "cfg": cfg,
                "equity": equity,
                "w": w,
                "open_px": open_px,
                "metrics": metrics,
            }

    # Build hybrid rows: choose "aggressive" as highest OOS return and
    # "defensive" as lowest drawdown among positive-return configs.
    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)
    for set_name, cache in run_cache.items():
        if len(cache) < 2:
            continue

        candidates = list(cache.items())
        aggressive_name, aggressive_item = max(
            candidates,
            key=lambda kv: float(kv[1]["metrics"].get("oos_ret", float("-inf"))),
        )

        positive_candidates = [
            kv for kv in candidates if float(kv[1]["metrics"].get("oos_ret", -1.0)) > 0
        ]
        if not positive_candidates:
            positive_candidates = candidates
        defensive_name, defensive_item = min(
            positive_candidates,
            key=lambda kv: abs(float(kv[1]["metrics"].get("oos_mdd", -1.0))),
        )

        wa = aggressive_item.get("w")
        wd = defensive_item.get("w")
        open_px = aggressive_item.get("open_px")
        if wa is None or wd is None or open_px is None:
            continue

        w_hybrid, trigger_mask = _build_hybrid_weights(
            w_aggr=wa,
            w_def=wd,
            benchmark_df=df_bench,
            risk_cap=0.45,
            trigger_in_days=2,
            release_days=3,
            max_daily_dw=0.15,
        )
        eq_hybrid, _, _, _ = calculate_performance(open_px, w_hybrid, cost)
        hybrid_metrics = _calc_metrics(eq_hybrid, bench_close, split_date)
        if not hybrid_metrics:
            continue

        row = {
            "set_name": set_name,
            "config": f"hybrid_{aggressive_name}_x_{defensive_name}",
            "method": "hybrid",
            "floor": np.nan,
            "delta_cap": np.nan,
            "n_strategies": len(run_sets[set_name]),
            "risk_trigger_pct": float(trigger_mask.mean()) if len(trigger_mask) else np.nan,
            "hybrid_aggressive_leg": aggressive_name,
            "hybrid_defensive_leg": defensive_name,
        }
        row.update(hybrid_metrics)
        row["score_ret_mdd"] = float(row["oos_ret"] - 0.8 * abs(row["oos_mdd"]))
        rows.append(row)

    if not rows:
        raise SystemExit("No valid test rows produced.")

    out_df = pd.DataFrame(rows).sort_values(
        by=["oos_alpha_vs_bh", "score_ret_mdd"],
        ascending=[False, False],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"beta_test_report_{ts}.csv"
    out_df.to_csv(out_path, index=False)

    print("\nOOS comparison:")
    _print_compact(out_df)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
