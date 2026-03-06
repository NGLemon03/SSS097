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
class OverlayParam:
    target_vol: float
    cap_light: float
    cap_deep: float
    dd_light: float
    dd_deep: float
    trend_gate: bool
    delta_cap: float
    min_trade_dw: float


def _load_strategy_names(warehouse_path: Path | None) -> list[str]:
    if warehouse_path and warehouse_path.exists():
        try:
            data = json.loads(warehouse_path.read_text(encoding="utf-8"))
            names = [str(s.get("name", "")).replace(".csv", "") for s in data.get("strategies", [])]
            names = [x for x in names if x]
            if names:
                return names
        except Exception:
            pass
    payload = manager.load_strategies()
    names = [str(s.get("name", "")).replace(".csv", "") for s in payload]
    return [x for x in names if x]


def _resolve_trade_files(names: Iterable[str], include_archive: bool) -> dict[str, Path]:
    roots = [Path("sss_backtest_outputs")]
    if include_archive:
        roots.append(Path("archive"))
    out: dict[str, Path] = {}
    for n in names:
        for r in roots:
            cands = list(r.rglob(f"*{n}*.csv"))
            if not cands:
                continue
            out[n] = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            break
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
    mdd = float((e / e.cummax() - 1).min())
    b_mdd = float((b / b.cummax() - 1).min())
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


def _smooth_weights(w: pd.Series, delta_cap: float, min_trade_dw: float) -> pd.Series:
    out = w.copy()
    for i in range(1, len(out)):
        prev = float(out.iloc[i - 1])
        curr = float(out.iloc[i])
        d = curr - prev
        if abs(d) > delta_cap:
            out.iloc[i] = prev + (delta_cap if d > 0 else -delta_cap)
    for i in range(1, len(out)):
        prev = float(out.iloc[i - 1])
        curr = float(out.iloc[i])
        if abs(curr - prev) < min_trade_dw:
            out.iloc[i] = prev
    return out.clip(0, 1)


def _apply_vol_target(w: pd.Series, open_px: pd.Series, target_vol: float) -> pd.Series:
    # w[t] is the position held from open[t] to open[t+1].
    # Use realized returns available up to t-1 to decide scaling on day t.
    o2o_ret = open_px.pct_change().fillna(0.0)
    strat_ret_est = o2o_ret * w.shift(1).fillna(0.0)
    roll_vol = strat_ret_est.rolling(20).std() * np.sqrt(252)
    scaler = (target_vol / roll_vol.shift(1).replace(0, np.nan)).clip(lower=0.50, upper=1.20).fillna(1.0)
    return (w * scaler).clip(0, 1)


def _apply_vol_target_conditional(
    w: pd.Series, open_px: pd.Series, target_vol: float, active_mask: pd.Series | None
) -> pd.Series:
    out = _apply_vol_target(w, open_px, target_vol)
    if active_mask is None:
        return out
    scaler = (out / w.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scaler = scaler.where(active_mask, 1.0)
    return (w * scaler).clip(0, 1)


def _apply_dd_caps(
    w: pd.Series,
    open_px: pd.Series,
    cost: CostParams,
    dd_light: float,
    dd_deep: float,
    cap_light: float,
    cap_deep: float,
) -> pd.Series:
    eq, _, _, _ = calculate_performance(open_px, w, cost)
    dd = eq / eq.cummax() - 1
    cap = np.where(dd <= -dd_deep, cap_deep, np.where(dd <= -dd_light, cap_light, 1.0))
    # DD is known after the bar closes; apply cap on next trading day.
    cap_s = pd.Series(cap, index=w.index, dtype=float).shift(1).fillna(1.0)
    return np.minimum(w, cap_s)


def _apply_dd_caps_conditional(
    w: pd.Series,
    open_px: pd.Series,
    cost: CostParams,
    dd_light: float,
    dd_deep: float,
    cap_light: float,
    cap_deep: float,
    active_mask: pd.Series | None,
) -> pd.Series:
    out = _apply_dd_caps(
        w=w,
        open_px=open_px,
        cost=cost,
        dd_light=dd_light,
        dd_deep=dd_deep,
        cap_light=cap_light,
        cap_deep=cap_deep,
    )
    if active_mask is None:
        return out
    cap_ratio = (out / w.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    cap_ratio = cap_ratio.where(active_mask, 1.0)
    return np.minimum(w, w * cap_ratio).clip(0, 1)


def _apply_trend_gate(w: pd.Series, close_px: pd.Series) -> pd.Series:
    ma120 = close_px.rolling(120).mean()
    # Trend state from close[t-1] gates position changes on day t.
    gate = (close_px < ma120).shift(1)
    gate = gate.where(gate.notna(), False).astype(bool)
    out = w.copy()
    for i in range(1, len(out)):
        if gate.iloc[i]:
            if out.iloc[i] > out.iloc[i - 1]:
                out.iloc[i] = out.iloc[i - 1]
    return out


def _risk_off_mask(close_px: pd.Series) -> pd.Series:
    ma120 = close_px.rolling(120).mean()
    mask = (close_px < ma120).shift(1)
    return mask.where(mask.notna(), False).astype(bool)


def _run_baseline(
    ticker: str,
    strategy_names: list[str],
    file_map: dict[str, Path],
    cost: CostParams,
) -> tuple[pd.Series, pd.Series]:
    p = EnsembleParams(
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
        params=p,
        cost=cost,
    )
    open_px, w, _, _, _, eq, _, _ = run_ensemble(cfg)
    if eq is None or eq.empty:
        raise RuntimeError("Baseline run failed.")
    return eq, w.reindex(open_px.index).ffill().fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round4 overlay test (A1/A2/A3).")
    parser.add_argument("--ticker", default="00631L.TW")
    parser.add_argument("--warehouse", default="analysis/strategy_warehouse.json")
    parser.add_argument("--split_date", default="2025-06-30")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--include_archive", action="store_true")
    parser.add_argument("--bh_target_mult", type=float, default=1.2)
    parser.add_argument("--base_mode", choices=["ensemble", "bh"], default="ensemble")
    parser.add_argument("--conditional_overlay", action="store_true")
    args = parser.parse_args()

    split = pd.Timestamp(args.split_date)

    df_price, _ = load_data(args.ticker)
    if df_price.empty or not {"open", "close"}.issubset(df_price.columns):
        raise SystemExit("Price data unavailable.")

    open_px = pd.to_numeric(df_price["open"], errors="coerce").dropna().sort_index()
    close_px = pd.to_numeric(df_price["close"], errors="coerce").dropna().sort_index()
    idx = open_px.index.intersection(close_px.index)
    open_px = open_px.reindex(idx).dropna()
    close_px = close_px.reindex(idx).dropna()

    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30.0)
    if args.base_mode == "ensemble":
        names = _load_strategy_names(Path(args.warehouse))
        file_map = _resolve_trade_files(names, include_archive=args.include_archive)
        names = [n for n in names if n in file_map]
        if len(names) < 2:
            raise SystemExit("Need >=2 strategy files.")
        base_eq, base_w = _run_baseline(args.ticker, names, file_map, cost)
    else:
        base_w = pd.Series(1.0, index=open_px.index, dtype=float)
        base_eq, _, _, _ = calculate_performance(open_px, base_w, cost)
    # Use open-to-open benchmark for fair comparison with open-executed strategy.
    bh_eq = open_px / open_px.iloc[0] * 1_000_000.0

    base_full = _compute_metrics(base_eq, bh_eq)
    base_oos = _compute_metrics(base_eq[base_eq.index > split], bh_eq[bh_eq.index > split])
    risk_off = _risk_off_mask(close_px) if args.conditional_overlay else None

    print(f"Ticker: {args.ticker}")
    print(f"Range: {open_px.index[0].date()} -> {open_px.index[-1].date()} ({len(open_px)} bars)")
    print(f"Base mode: {args.base_mode}, conditional overlay: {args.conditional_overlay}")
    print(f"Baseline OOS: ret={base_oos['ret']:.2%}, mdd={base_oos['mdd']:.2%}, alpha={base_oos['alpha_vs_bh']:.2%}")

    grid: list[OverlayParam] = []
    for tv in [0.20, 0.22, 0.25]:
        for cl in [0.60, 0.70]:
            for cd in [0.25, 0.30, 0.35]:
                for gate in [True, False]:
                    grid.append(
                        OverlayParam(
                            target_vol=tv,
                            cap_light=cl,
                            cap_deep=cd,
                            dd_light=0.06,
                            dd_deep=0.10,
                            trend_gate=gate,
                            delta_cap=0.12,
                            min_trade_dw=0.01,
                        )
                    )

    rows: list[dict[str, object]] = []
    best_w: pd.Series | None = None
    best_score = float("-inf")

    for i, gp in enumerate(grid, 1):
        w = base_w.copy()
        w = _apply_vol_target_conditional(
            w=w,
            open_px=open_px,
            target_vol=gp.target_vol,
            active_mask=risk_off,
        )
        w = _apply_dd_caps_conditional(
            w, open_px, cost,
            dd_light=gp.dd_light,
            dd_deep=gp.dd_deep,
            cap_light=gp.cap_light,
            cap_deep=gp.cap_deep,
            active_mask=risk_off,
        )
        if gp.trend_gate:
            w = _apply_trend_gate(w, close_px)
        w = _smooth_weights(w, gp.delta_cap, gp.min_trade_dw)

        eq, _, _, _ = calculate_performance(open_px, w, cost)
        full = _compute_metrics(eq, bh_eq)
        oos = _compute_metrics(eq[eq.index > split], bh_eq[bh_eq.index > split])
        if not full or not oos:
            continue

        # Compare terminal wealth, not raw return ratios.
        full_mult = ((1.0 + full["ret"]) / (1.0 + full["bh_ret"])) if (1.0 + full["bh_ret"]) > 0 else np.nan
        oos_mult = ((1.0 + oos["ret"]) / (1.0 + oos["bh_ret"])) if (1.0 + oos["bh_ret"]) > 0 else np.nan

        # Near-BH objective: keep return close while reducing MDD.
        near_score = (
            (oos_mult if pd.notna(oos_mult) else -999.0)
            + 0.25 * (oos["mdd"] - oos["bh_mdd"])
            + 0.05 * oos["sharpe"]
        )

        row = {
            "target_vol": gp.target_vol,
            "cap_light": gp.cap_light,
            "cap_deep": gp.cap_deep,
            "trend_gate": gp.trend_gate,
            "base_mode": args.base_mode,
            "conditional_overlay": args.conditional_overlay,
            "oos_ret": oos["ret"],
            "oos_mdd": oos["mdd"],
            "oos_alpha_vs_bh": oos["alpha_vs_bh"],
            "oos_sharpe": oos["sharpe"],
            "oos_wealth_mult_vs_bh": oos_mult,
            # Backward-compat column name; value uses wealth multiple semantics.
            "oos_ret_ratio_vs_bh": oos_mult,
            "oos_mdd_improve_vs_bh": (oos["mdd"] - oos["bh_mdd"]),
            "full_ret": full["ret"],
            "full_mdd": full["mdd"],
            "full_alpha_vs_bh": full["alpha_vs_bh"],
            "full_bh_multiple": full_mult,
            "oos_bh_multiple": oos_mult,
            "full_target_pass": bool(pd.notna(full_mult) and full_mult >= args.bh_target_mult),
            "oos_target_pass": bool(pd.notna(oos_mult) and oos_mult >= args.bh_target_mult),
            "turnover_py": float(w.diff().abs().sum() / len(w) * 252),
            "time_in_market": float((w > 0.05).mean()),
            "near_bh_score": near_score,
        }
        rows.append(row)

        if near_score > best_score:
            best_score = near_score
            best_w = w.copy()

        if i % 6 == 0 or i == len(grid):
            print(f"Progress: {i}/{len(grid)}")

    if not rows:
        raise SystemExit("No result rows generated.")

    res = pd.DataFrame(rows).sort_values("near_bh_score", ascending=False)
    near = res[(res["oos_wealth_mult_vs_bh"] >= 0.85) & (res["oos_mdd_improve_vs_bh"] > 1e-6)]
    strict = res[(res["oos_wealth_mult_vs_bh"] >= 0.95) & (res["oos_mdd_improve_vs_bh"] > 1e-6)]

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.base_mode}_{'cond' if args.conditional_overlay else 'always'}"
    all_path = out_dir / f"round4_overlay_scan_{run_tag}_{ts}.csv"
    near_path = out_dir / f"round4_overlay_near_bh_{run_tag}_{ts}.csv"
    strict_path = out_dir / f"round4_overlay_strict_near_bh_{run_tag}_{ts}.csv"
    w_path = out_dir / f"round4_overlay_best_weights_{run_tag}_{ts}.csv"
    eq_path = out_dir / f"round4_overlay_best_equity_{run_tag}_{ts}.csv"

    res.to_csv(all_path, index=False)
    near.to_csv(near_path, index=False)
    strict.to_csv(strict_path, index=False)

    if best_w is not None:
        best_eq, _, _, _ = calculate_performance(open_px, best_w, cost)
        pd.DataFrame(
            {
                "baseline_w": base_w.reindex(open_px.index).ffill().fillna(0.0),
                "best_overlay_w": best_w.reindex(open_px.index).ffill().fillna(0.0),
            },
            index=open_px.index,
        ).to_csv(w_path, index_label="date")
        pd.DataFrame(
            {
                "baseline_equity": base_eq.reindex(open_px.index).ffill(),
                "best_overlay_equity": best_eq.reindex(open_px.index).ffill(),
                "bh_equity": bh_eq.reindex(open_px.index).ffill(),
            },
            index=open_px.index,
        ).to_csv(eq_path, index_label="date")

    print("\nTop 10 (near-BH objective):")
    print(
        res[
            [
                "target_vol",
                "cap_light",
                "cap_deep",
                "trend_gate",
                "oos_ret",
                "oos_mdd",
                "oos_wealth_mult_vs_bh",
                "oos_mdd_improve_vs_bh",
                "oos_sharpe",
                "turnover_py",
                "oos_bh_multiple",
            ]
        ]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print(f"\nNear-BH candidates (>=0.85x and lower MDD): {len(near)}")
    print(f"Strict near-BH candidates (>=0.95x and lower MDD): {len(strict)}")
    print(f"\nSaved all: {all_path}")
    print(f"Saved near-BH: {near_path}")
    print(f"Saved strict near-BH: {strict_path}")
    if best_w is not None:
        print(f"Saved best weights: {w_path}")
        print(f"Saved best equity: {eq_path}")


if __name__ == "__main__":
    main()
