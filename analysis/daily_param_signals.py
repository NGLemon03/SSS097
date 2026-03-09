from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis import config as cfg
from SSS_EnsembleTab import EnsembleParams, weights_majority, weights_proportional
from SSSv096 import (
    backtest_unified,
    compute_RMA,
    compute_single,
    compute_ssma_turn_combined,
    load_data,
    param_presets,
)

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

HISTORY_FILE = Path("analysis/signal_history.csv")

HISTORY_COLUMNS = [
    "date",
    "strategy_name",
    "signal",
    "price",
    "timestamp",
    "engine_family",
    "mode",
    "source",
    "semantics",
    "estimated_close",
    "estimated_volume",
    "estimated_inputs_json",
    "trigger_side",
    "triggered",
    "projected_exec_date",
    "impact_compare",
]

REQUIREMENT_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "Single 2": [
        {"symbol": "$TICKER", "field": "close", "min_bars": 177},
    ],
    "single_1887": [
        {"symbol": "$TICKER", "field": "close", "min_bars": 177},
    ],
    "Single 3": [
        {"symbol": "$TICKER", "field": "close", "min_bars": 177},
    ],
    "RMA_669": [
        {"symbol": "$TICKER", "field": "close", "min_bars": 383},
    ],
    "RMA_69": [
        {"symbol": "^TWII", "field": "close", "min_bars": 366},
        {"symbol": "2414.TW", "field": "close", "min_bars": 366},
    ],
    "TV_RMAv2": [
        {"symbol": "^TWII", "field": "close", "min_bars": 192},
        {"symbol": "2412.TW", "field": "close", "min_bars": 192},
    ],
    "STM1": [
        {"symbol": "$TICKER", "field": "close", "min_bars": 119},
        {"symbol": "$TICKER", "field": "volume", "min_bars": 41},
    ],
    "STM3": [
        {"symbol": "$TICKER", "field": "close", "min_bars": 114},
        {"symbol": "$TICKER", "field": "volume", "min_bars": 46},
    ],
    "STM_1939": [
        {"symbol": "$TICKER", "field": "close", "min_bars": 434},
        {"symbol": "$TICKER", "field": "volume", "min_bars": 81},
    ],
    "STM0": [
        {"symbol": "^TWII", "field": "close", "min_bars": 199},
        {"symbol": "2414.TW", "field": "close", "min_bars": 199},
        {"symbol": "$TICKER", "field": "volume", "min_bars": 91},
        {"symbol": "2414.TW", "field": "volume", "min_bars": 91},
    ],
    "STM4": [
        {"symbol": "^TWII", "field": "close", "min_bars": 109},
        {"symbol": "2414.TW", "field": "close", "min_bars": 109},
        {"symbol": "$TICKER", "field": "volume", "min_bars": 41},
        {"symbol": "2414.TW", "field": "volume", "min_bars": 41},
    ],
    "STM_2414_273": [
        {"symbol": "^TWII", "field": "close", "min_bars": 349},
        {"symbol": "2414.TW", "field": "close", "min_bars": 349},
        {"symbol": "$TICKER", "field": "volume", "min_bars": 91},
        {"symbol": "2414.TW", "field": "volume", "min_bars": 91},
    ],
}


@dataclass
class DailySignalContext:
    ticker: str
    run_date: pd.Timestamp
    visible_strategies: List[str]
    requirements_df: pd.DataFrame
    input_schema: List[Dict[str, Any]]
    requirement_map: Dict[str, List[Dict[str, Any]]]


def _to_datetime_index(index: Iterable[Any]) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index, errors="coerce", format="mixed")
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _today_taipei() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()


def _strategy_visible(name: str, conf: Dict[str, Any], hidden_set: set[str]) -> bool:
    if name in hidden_set:
        return False
    if bool(conf.get("hidden", False)):
        return False
    if bool(conf.get("ui_hidden", False)):
        return False
    return True


def get_visible_strategy_names(hidden_strategy_presets: Optional[List[str]] = None) -> List[str]:
    hidden_set = set(hidden_strategy_presets or [])
    return [
        name
        for name, conf in param_presets.items()
        if _strategy_visible(name, conf, hidden_set)
    ]


def _factor_symbol_from_source(smaa_source: str) -> Optional[str]:
    src = str(smaa_source or "")
    if "2412" in src:
        return "2412.TW"
    if "2414" in src:
        return "2414.TW"
    return None


def _normalize_ssma_filter_mode(conf: Dict[str, Any]) -> str:
    mode_raw = str(conf.get("signal_filter_mode", "volume_ma") or "volume_ma").strip().lower()
    if mode_raw in {"none", "off", "disable", "disabled", "no_volume"}:
        return "none"
    if mode_raw in {"volume_target", "adaptive_volume", "volume_quantile"}:
        return "volume_target"
    return "volume_ma"


def _ssma_requires_volume(conf: Dict[str, Any]) -> bool:
    return _normalize_ssma_filter_mode(conf) in {"volume_ma", "volume_target"}


def _fallback_requirements(
    strategy_name: str,
    ticker: str,
    conf: Dict[str, Any],
) -> List[Dict[str, Any]]:
    strategy_type = str(conf.get("strategy_type", "")).strip().lower()
    smaa_source = conf.get("smaa_source", "Self")
    factor_symbol = _factor_symbol_from_source(smaa_source)
    reqs: List[Dict[str, Any]] = []
    if strategy_type in {"single", "rma"}:
        if factor_symbol:
            reqs.extend(
                [
                    {"symbol": "^TWII", "field": "close", "min_bars": 150},
                    {"symbol": factor_symbol, "field": "close", "min_bars": 150},
                ]
            )
        else:
            reqs.append({"symbol": ticker, "field": "close", "min_bars": 150})
    elif strategy_type == "ssma_turn":
        requires_volume = _ssma_requires_volume(conf)
        if factor_symbol:
            reqs.extend(
                [
                    {"symbol": "^TWII", "field": "close", "min_bars": 150},
                    {"symbol": factor_symbol, "field": "close", "min_bars": 150},
                ]
            )
            if requires_volume:
                reqs.extend(
                    [
                        {"symbol": ticker, "field": "volume", "min_bars": 40},
                        {"symbol": factor_symbol, "field": "volume", "min_bars": 40},
                    ]
                )
        else:
            reqs.extend(
                [
                    {"symbol": ticker, "field": "close", "min_bars": 150},
                ]
            )
            if requires_volume:
                reqs.append({"symbol": ticker, "field": "volume", "min_bars": 40})
    return reqs


def _expand_template(
    strategy_name: str,
    ticker: str,
    conf: Dict[str, Any],
) -> List[Dict[str, Any]]:
    template = REQUIREMENT_TEMPLATES.get(strategy_name)
    if template is None:
        return _fallback_requirements(strategy_name, ticker, conf)
    requires_volume = _ssma_requires_volume(conf) if str(conf.get("strategy_type", "")).strip().lower() == "ssma_turn" else True
    expanded = []
    for item in template:
        field = str(item["field"]).lower()
        if field == "volume" and not requires_volume:
            continue
        symbol = str(item["symbol"])
        if symbol == "$TICKER":
            symbol = ticker
        expanded.append(
            {
                "symbol": symbol,
                "field": field,
                "min_bars": int(item["min_bars"]),
            }
        )
    return expanded


def _symbol_csv_path(symbol: str, pine_parity_mode: bool = False) -> Path:
    suffix = "_data_raw_unadj.csv" if pine_parity_mode else "_data_raw.csv"
    return cfg.DATA_DIR / f"{symbol.replace(':', '_')}{suffix}"


def _read_symbol_market_df(symbol: str) -> pd.DataFrame:
    path = _symbol_csv_path(symbol, pine_parity_mode=False)
    if path.exists():
        try:
            df = pd.read_csv(path, index_col=0)
            df.columns = [str(c).lower().strip() for c in df.columns]
            df.index = _to_datetime_index(df.index)
            for col in ("open", "high", "low", "close", "volume"):
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.sort_index()
        except Exception:
            pass
    try:
        df, _ = load_data(symbol, smaa_source="Self")
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.index = _to_datetime_index(df.index)
            for col in ("open", "high", "low", "close", "volume"):
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.sort_index()
    except Exception:
        pass
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _series_count(df: pd.DataFrame, field: str) -> int:
    if df.empty or field not in df.columns:
        return 0
    return int(pd.to_numeric(df[field], errors="coerce").dropna().shape[0])


def _latest_value(df: pd.DataFrame, field: str) -> Optional[float]:
    if df.empty or field not in df.columns:
        return None
    s = pd.to_numeric(df[field], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _requirement_key(symbol: str, field: str) -> str:
    return f"{symbol}|{field}"


def build_daily_signal_context(
    ticker: str,
    hidden_strategy_presets: Optional[List[str]] = None,
    run_date: Optional[pd.Timestamp] = None,
) -> DailySignalContext:
    run_ts = pd.Timestamp(run_date) if run_date is not None else _today_taipei()
    run_ts = run_ts.tz_localize(None).normalize()
    visible = get_visible_strategy_names(hidden_strategy_presets=hidden_strategy_presets)

    rows: List[Dict[str, Any]] = []
    requirement_map: Dict[str, List[Dict[str, Any]]] = {}
    for strategy_name in visible:
        conf = param_presets.get(strategy_name, {})
        strategy_type = str(conf.get("strategy_type", "unknown")).lower()
        if strategy_type == "ensemble":
            requirement_map[strategy_name] = []
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "strategy_type": strategy_type,
                    "symbol": "-",
                    "field": "-",
                    "min_bars": 0,
                    "requirement_key": "-",
                    "available_bars": np.nan,
                    "latest_value": np.nan,
                    "ready": True,
                    "notes": "derived_from_member_strategies",
                }
            )
            continue
        reqs = _expand_template(strategy_name, ticker, conf)
        requirement_map[strategy_name] = reqs
        for req in reqs:
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "strategy_type": strategy_type,
                    "symbol": req["symbol"],
                    "field": req["field"],
                    "min_bars": int(req["min_bars"]),
                    "requirement_key": _requirement_key(req["symbol"], req["field"]),
                }
            )

    requirements_df = pd.DataFrame(rows)
    if requirements_df.empty:
        requirements_df = pd.DataFrame(
            columns=[
                "strategy_name",
                "strategy_type",
                "symbol",
                "field",
                "min_bars",
                "requirement_key",
                "available_bars",
                "latest_value",
                "ready",
                "notes",
            ]
        )

    non_ensemble = requirements_df[requirements_df["symbol"] != "-"].copy()
    symbols = sorted(set(non_ensemble["symbol"].tolist()))
    market_cache = {sym: _read_symbol_market_df(sym) for sym in symbols}

    if not non_ensemble.empty:
        available_bars = []
        latest_values = []
        readiness = []
        for _, row in non_ensemble.iterrows():
            symbol = str(row["symbol"])
            field = str(row["field"])
            need = int(row["min_bars"])
            df_market = market_cache.get(symbol, pd.DataFrame())
            avail = _series_count(df_market, field)
            latest = _latest_value(df_market, field)
            available_bars.append(avail)
            latest_values.append(latest if latest is not None else np.nan)
            readiness.append(bool(avail >= need))
        non_ensemble["available_bars"] = available_bars
        non_ensemble["latest_value"] = latest_values
        non_ensemble["ready"] = readiness
        non_ensemble["notes"] = ""
        requirements_df = pd.concat(
            [non_ensemble, requirements_df[requirements_df["symbol"] == "-"]],
            ignore_index=True,
        )

    input_schema: List[Dict[str, Any]] = []
    if not non_ensemble.empty:
        for (symbol, field), grp in non_ensemble.groupby(["symbol", "field"], sort=True):
            req_by = sorted(set(grp["strategy_name"].astype(str)))
            need = int(grp["min_bars"].max())
            avail = int(grp["available_bars"].max()) if "available_bars" in grp.columns else 0
            default_val = grp["latest_value"].dropna()
            latest = float(default_val.iloc[-1]) if not default_val.empty else None
            input_schema.append(
                {
                    "key": _requirement_key(symbol, field),
                    "symbol": symbol,
                    "field": field,
                    "min_bars": need,
                    "available_bars": avail,
                    "ready": bool(avail >= need),
                    "latest_value": latest,
                    "required_by": req_by,
                }
            )

    return DailySignalContext(
        ticker=ticker,
        run_date=run_ts,
        visible_strategies=visible,
        requirements_df=requirements_df.sort_values(
            ["strategy_name", "symbol", "field"], kind="stable"
        ).reset_index(drop=True),
        input_schema=input_schema,
        requirement_map=requirement_map,
    )


def normalize_estimate_map(raw_estimates: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not raw_estimates:
        return out
    for key, value in raw_estimates.items():
        if value is None:
            continue
        try:
            num = float(value)
        except Exception:
            continue
        if np.isfinite(num):
            out[str(key)] = num
    return out


def _find_missing_estimate_keys(
    context: DailySignalContext,
    estimate_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    missing = []
    for item in context.input_schema:
        key = str(item["key"])
        if key not in estimate_map or not np.isfinite(float(estimate_map[key])):
            missing.append(item)
    return missing


def _upsert_market_value(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    field: str,
    value: Optional[float],
) -> pd.DataFrame:
    if value is None or not np.isfinite(float(value)):
        return df
    if df is None or df.empty:
        return df
    field = str(field).lower()
    out = df.copy()
    out.index = _to_datetime_index(out.index)
    if field not in out.columns:
        out[field] = np.nan
    v = float(value)
    target = pd.Timestamp(target_date).tz_localize(None).normalize()
    if target not in out.index:
        new_row = out.iloc[-1].copy()
        out.loc[target] = new_row
    out.at[target, field] = v

    if field == "close":
        for col in ("open", "high", "low"):
            if col in out.columns and pd.isna(out.at[target, col]):
                out.at[target, col] = v
    out[field] = pd.to_numeric(out[field], errors="coerce")
    out = out.sort_index()
    return out


def _apply_experiment_to_price_df(
    df_price: pd.DataFrame,
    ticker: str,
    estimate_map: Dict[str, float],
    run_date: pd.Timestamp,
) -> pd.DataFrame:
    out = df_price.copy()
    out = _upsert_market_value(
        out, run_date, "close", estimate_map.get(_requirement_key(ticker, "close"))
    )
    out = _upsert_market_value(
        out, run_date, "volume", estimate_map.get(_requirement_key(ticker, "volume"))
    )
    return out


def _apply_experiment_to_factor_df(
    df_factor: pd.DataFrame,
    smaa_source: str,
    estimate_map: Dict[str, float],
    run_date: pd.Timestamp,
) -> pd.DataFrame:
    out = df_factor.copy()
    if out.empty:
        return out
    factor_symbol = _factor_symbol_from_source(smaa_source)
    if not factor_symbol:
        return out

    twii_close = estimate_map.get(_requirement_key("^TWII", "close"))
    factor_close = estimate_map.get(_requirement_key(factor_symbol, "close"))
    if (
        twii_close is not None
        and factor_close is not None
        and np.isfinite(float(twii_close))
        and np.isfinite(float(factor_close))
        and float(factor_close) > 0
    ):
        ratio = float(twii_close) / float(factor_close)
        out = _upsert_market_value(out, run_date, "close", ratio)

    factor_volume = estimate_map.get(_requirement_key(factor_symbol, "volume"))
    out = _upsert_market_value(out, run_date, "volume", factor_volume)
    return out


def _latest_price_volume(df_price: pd.DataFrame) -> Tuple[float, float]:
    if df_price is None or df_price.empty:
        return np.nan, np.nan
    close_s = pd.to_numeric(df_price.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    vol_s = pd.to_numeric(df_price.get("volume", pd.Series(dtype=float)), errors="coerce").dropna()
    close_v = float(close_s.iloc[-1]) if not close_s.empty else np.nan
    vol_v = float(vol_s.iloc[-1]) if not vol_s.empty else np.nan
    return close_v, vol_v


def _signal_from_indicator_row(
    row: pd.Series,
    buy_mult: float,
    sell_mult: float,
    pine_parity_mode: bool = False,
) -> str:
    smaa = pd.to_numeric(row.get("smaa", np.nan), errors="coerce")
    base = pd.to_numeric(row.get("base", np.nan), errors="coerce")
    sd = pd.to_numeric(row.get("sd", np.nan), errors="coerce")
    if pd.isna(smaa) or pd.isna(base) or pd.isna(sd):
        return HOLD
    if pine_parity_mode:
        buy_level = base - sd * float(buy_mult)
    else:
        buy_level = base + sd * float(buy_mult)
    sell_level = base + sd * float(sell_mult)
    if smaa < buy_level:
        return BUY
    if smaa > sell_level:
        return SELL
    return HOLD


def _extract_position_series(bt_result: Dict[str, Any]) -> pd.Series:
    if not bt_result:
        return pd.Series(dtype=float)
    ds = bt_result.get("daily_state")
    if isinstance(ds, pd.DataFrame) and not ds.empty and "w" in ds.columns:
        s = pd.to_numeric(ds["w"], errors="coerce").fillna(0.0).clip(0, 1)
        s.index = _to_datetime_index(s.index)
        return s.sort_index()
    wc = bt_result.get("weight_curve")
    if isinstance(wc, pd.Series) and not wc.empty:
        s = pd.to_numeric(wc, errors="coerce").fillna(0.0).clip(0, 1)
        s.index = _to_datetime_index(s.index)
        return s.sort_index()
    return pd.Series(dtype=float)


def _compute_non_ensemble_strategy(
    strategy_name: str,
    strategy_conf: Dict[str, Any],
    ticker: str,
    mode: str,
    estimate_map: Dict[str, float],
    run_date: pd.Timestamp,
) -> Tuple[Dict[str, Any], pd.Series]:
    strategy_type = str(strategy_conf.get("strategy_type", "unknown")).lower()
    smaa_source = strategy_conf.get("smaa_source", "Self")
    data_provider = strategy_conf.get("data_provider", "yfinance")
    pine_parity_mode = bool(strategy_conf.get("pine_parity_mode", False))

    base_row = {
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "mode": mode,
        "signal": HOLD,
        "price": np.nan,
        "volume": np.nan,
        "triggered": False,
        "trigger_side": HOLD,
        "impact_compare": "",
        "error": "",
        "data_date": None,
        "estimated_close": estimate_map.get(_requirement_key(ticker, "close"), np.nan)
        if mode == "experiment"
        else np.nan,
        "estimated_volume": estimate_map.get(_requirement_key(ticker, "volume"), np.nan)
        if mode == "experiment"
        else np.nan,
    }

    try:
        df_price, df_factor = load_data(
            ticker=ticker,
            start_date="2000-01-01",
            end_date=None,
            smaa_source=smaa_source,
            force_update=False,
            data_provider=data_provider,
            pine_parity_mode=pine_parity_mode,
        )
        if df_price is None or df_price.empty:
            base_row["error"] = "price_data_empty"
            return base_row, pd.Series(dtype=float)

        df_price = df_price.copy()
        df_price.index = _to_datetime_index(df_price.index)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df_price.columns:
                df_price[col] = np.nan
            df_price[col] = pd.to_numeric(df_price[col], errors="coerce")

        if isinstance(df_factor, pd.DataFrame) and not df_factor.empty:
            df_factor = df_factor.copy()
            df_factor.index = _to_datetime_index(df_factor.index)
            for col in ("close", "volume"):
                if col not in df_factor.columns:
                    df_factor[col] = np.nan
                df_factor[col] = pd.to_numeric(df_factor[col], errors="coerce")
        else:
            df_factor = pd.DataFrame()

        if mode == "experiment":
            df_price = _apply_experiment_to_price_df(df_price, ticker, estimate_map, run_date)
            if not df_factor.empty:
                df_factor = _apply_experiment_to_factor_df(
                    df_factor=df_factor,
                    smaa_source=smaa_source,
                    estimate_map=estimate_map,
                    run_date=run_date,
                )

        price_v, volume_v = _latest_price_volume(df_price)
        base_row["price"] = price_v
        base_row["volume"] = volume_v

        if strategy_type == "single":
            df_ind = compute_single(
                df_price,
                df_factor,
                strategy_conf["linlen"],
                strategy_conf["factor"],
                strategy_conf["smaalen"],
                strategy_conf["devwin"],
                smaa_source=smaa_source,
            )
            buy_dates: List[pd.Timestamp] = []
            sell_dates: List[pd.Timestamp] = []
        elif strategy_type == "rma":
            df_ind = compute_RMA(
                df_price,
                df_factor,
                strategy_conf["linlen"],
                strategy_conf["factor"],
                strategy_conf["smaalen"],
                strategy_conf["rma_len"],
                strategy_conf["dev_len"],
                smaa_source=smaa_source,
                pine_parity_mode=pine_parity_mode,
            )
            buy_dates = []
            sell_dates = []
        elif strategy_type == "ssma_turn":
            calc_keys = [
                "linlen",
                "factor",
                "smaalen",
                "prom_factor",
                "min_dist",
                "buy_shift",
                "exit_shift",
                "vol_window",
                "quantile_win",
                "signal_cooldown_days",
                "signal_filter_mode",
                "volume_target_pass_rate",
                "volume_target_lookback",
            ]
            ssma_params = {k: strategy_conf[k] for k in calc_keys if k in strategy_conf}
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                df_price,
                df_factor,
                **ssma_params,
                smaa_source=smaa_source,
            )
        else:
            base_row["error"] = f"unsupported_strategy_type:{strategy_type}"
            return base_row, pd.Series(dtype=float)

        if df_ind is None or df_ind.empty:
            base_row["error"] = "indicator_empty"
            return base_row, pd.Series(dtype=float)

        df_ind = df_ind.copy()
        df_ind.index = _to_datetime_index(df_ind.index)
        latest_dt = pd.Timestamp(df_ind.index[-1]).tz_localize(None)
        base_row["data_date"] = latest_dt.strftime("%Y-%m-%d")

        if strategy_type in {"single", "rma"}:
            sig = _signal_from_indicator_row(
                df_ind.iloc[-1],
                buy_mult=float(strategy_conf.get("buy_mult", 1.0)),
                sell_mult=float(strategy_conf.get("sell_mult", 1.0)),
                pine_parity_mode=pine_parity_mode,
            )
        else:
            buy_set = {pd.Timestamp(d).tz_localize(None) for d in buy_dates}
            sell_set = {pd.Timestamp(d).tz_localize(None) for d in sell_dates}
            if latest_dt in sell_set:
                sig = SELL
            elif latest_dt in buy_set:
                sig = BUY
            else:
                sig = HOLD

        bt_params = dict(strategy_conf)
        if strategy_type == "ssma_turn":
            bt_params["stop_loss"] = float(strategy_conf.get("stop_loss", 0.0))
            bt_result = backtest_unified(
                df_ind=df_ind,
                strategy_type=strategy_type,
                params=bt_params,
                buy_dates=buy_dates,
                sell_dates=sell_dates,
                discount=0.3,
                trade_cooldown_bars=3,
                bad_holding=False,
            )
        else:
            bt_result = backtest_unified(
                df_ind=df_ind,
                strategy_type=strategy_type,
                params=bt_params,
                discount=0.3,
                trade_cooldown_bars=3,
                bad_holding=False,
            )
        weight_series = _extract_position_series(bt_result)

        base_row["signal"] = sig
        base_row["triggered"] = bool(sig != HOLD)
        base_row["trigger_side"] = sig
        return base_row, weight_series
    except Exception as exc:
        base_row["error"] = f"exception:{exc}"
        return base_row, pd.Series(dtype=float)


def _build_member_position_matrix(member_weights: Dict[str, pd.Series]) -> pd.DataFrame:
    if not member_weights:
        return pd.DataFrame()
    union_idx = pd.DatetimeIndex([])
    for ser in member_weights.values():
        if ser is None or ser.empty:
            continue
        idx = _to_datetime_index(ser.index)
        union_idx = union_idx.union(idx)
    union_idx = union_idx.sort_values()
    if union_idx.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=union_idx)
    for name, ser in member_weights.items():
        if ser is None or ser.empty:
            continue
        s = pd.to_numeric(ser, errors="coerce")
        s.index = _to_datetime_index(s.index)
        s = s.reindex(union_idx).ffill().fillna(0.0)
        out[name] = (s > 1e-8).astype(float)
    return out


def _compute_ensemble_strategy(
    strategy_name: str,
    strategy_conf: Dict[str, Any],
    ticker: str,
    mode: str,
    estimate_map: Dict[str, float],
    run_date: pd.Timestamp,
    member_weights: Dict[str, pd.Series],
) -> Dict[str, Any]:
    row = {
        "strategy_name": strategy_name,
        "strategy_type": "ensemble",
        "mode": mode,
        "signal": HOLD,
        "price": np.nan,
        "volume": np.nan,
        "triggered": False,
        "trigger_side": HOLD,
        "impact_compare": "",
        "error": "",
        "data_date": None,
        "estimated_close": estimate_map.get(_requirement_key(ticker, "close"), np.nan)
        if mode == "experiment"
        else np.nan,
        "estimated_volume": estimate_map.get(_requirement_key(ticker, "volume"), np.nan)
        if mode == "experiment"
        else np.nan,
    }

    pos_df = _build_member_position_matrix(member_weights)
    if pos_df.empty or pos_df.shape[1] == 0:
        row["error"] = "no_member_positions"
        return row

    try:
        params_cfg = dict(strategy_conf.get("params", {}))
        ens_params = EnsembleParams(
            floor=float(params_cfg.get("floor", 0.2)),
            ema_span=int(params_cfg.get("ema_span", 3)),
            delta_cap=float(params_cfg.get("delta_cap", 0.3)),
            majority_k=int(params_cfg.get("majority_k", 1)),
            min_cooldown_days=int(params_cfg.get("min_cooldown_days", 1)),
            min_trade_dw=float(params_cfg.get("min_trade_dw", 0.01)),
            delta_cap_buy=params_cfg.get("delta_cap_buy"),
            delta_cap_sell=params_cfg.get("delta_cap_sell"),
            enable_asymmetric=bool(params_cfg.get("enable_asymmetric", False)),
        )
        method = str(strategy_conf.get("method", "majority")).strip().lower()
        n_members = pos_df.shape[1]
        if method == "majority":
            k_pct = strategy_conf.get("majority_k_pct", 0.55)
            try:
                k_pct_f = float(k_pct)
            except Exception:
                k_pct_f = 0.55
            if np.isfinite(k_pct_f) and k_pct_f > 0:
                ens_params.majority_k = max(1, int(math.ceil(n_members * k_pct_f)))
            w = weights_majority(pos_df, ens_params)
        else:
            w = weights_proportional(pos_df, ens_params)

        if w is None or w.empty:
            row["error"] = "ensemble_weight_empty"
            return row
        w = pd.to_numeric(w, errors="coerce").ffill().fillna(0.0).clip(0, 1)
        w.index = _to_datetime_index(w.index)
        row["data_date"] = pd.Timestamp(w.index[-1]).strftime("%Y-%m-%d")

        delta = float(w.iloc[-1] - w.iloc[-2]) if len(w) >= 2 else 0.0
        if delta > 0:
            sig = BUY
        elif delta < 0:
            sig = SELL
        else:
            sig = HOLD
        row["signal"] = sig
        row["triggered"] = bool(sig != HOLD)
        row["trigger_side"] = sig

        df_price, _ = load_data(
            ticker=ticker,
            start_date="2000-01-01",
            end_date=None,
            smaa_source="Self",
            force_update=False,
            data_provider="yfinance",
            pine_parity_mode=False,
        )
        if isinstance(df_price, pd.DataFrame) and not df_price.empty:
            df_price = df_price.copy()
            df_price.index = _to_datetime_index(df_price.index)
            if mode == "experiment":
                df_price = _apply_experiment_to_price_df(df_price, ticker, estimate_map, run_date)
            price_v, volume_v = _latest_price_volume(df_price)
            row["price"] = price_v
            row["volume"] = volume_v
    except Exception as exc:
        row["error"] = f"exception:{exc}"
    return row


def _compute_mode_signals(
    context: DailySignalContext,
    mode: str,
    estimate_map: Dict[str, float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    member_weights: Dict[str, pd.Series] = {}

    for strategy_name in context.visible_strategies:
        strategy_conf = param_presets.get(strategy_name, {})
        strategy_type = str(strategy_conf.get("strategy_type", "")).lower()
        if strategy_type == "ensemble":
            continue
        row, weight_series = _compute_non_ensemble_strategy(
            strategy_name=strategy_name,
            strategy_conf=strategy_conf,
            ticker=context.ticker,
            mode=mode,
            estimate_map=estimate_map,
            run_date=context.run_date,
        )
        rows.append(row)
        if weight_series is not None and not weight_series.empty:
            member_weights[strategy_name] = weight_series

    for strategy_name in context.visible_strategies:
        strategy_conf = param_presets.get(strategy_name, {})
        strategy_type = str(strategy_conf.get("strategy_type", "")).lower()
        if strategy_type != "ensemble":
            continue
        row = _compute_ensemble_strategy(
            strategy_name=strategy_name,
            strategy_conf=strategy_conf,
            ticker=context.ticker,
            mode=mode,
            estimate_map=estimate_map,
            run_date=context.run_date,
            member_weights=member_weights,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "strategy_name",
                "strategy_type",
                "mode",
                "signal",
                "price",
                "volume",
                "triggered",
                "trigger_side",
                "impact_compare",
                "error",
                "data_date",
                "estimated_close",
                "estimated_volume",
            ]
        )
    return out


def _apply_impact_compare(
    real_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real = real_df.copy()
    exp = experiment_df.copy()
    if real.empty:
        real["impact_compare"] = ""
        exp["impact_compare"] = ""
        return real, exp, pd.DataFrame(columns=["strategy_name", "real_signal", "experiment_signal"])

    real_sig_map = dict(zip(real["strategy_name"], real["signal"]))
    real["impact_compare"] = "REAL_BASELINE"

    if exp.empty:
        return real, exp, pd.DataFrame(columns=["strategy_name", "real_signal", "experiment_signal"])

    compare_rows = []
    impacts = []
    for _, r in exp.iterrows():
        name = r.get("strategy_name")
        exp_sig = str(r.get("signal", HOLD))
        real_sig = str(real_sig_map.get(name, HOLD))
        if real_sig == exp_sig:
            impacts.append("UNCHANGED")
        else:
            impacts.append(f"{real_sig}->{exp_sig}")
            compare_rows.append(
                {
                    "strategy_name": name,
                    "real_signal": real_sig,
                    "experiment_signal": exp_sig,
                }
            )
    exp["impact_compare"] = impacts
    changed_df = pd.DataFrame(compare_rows)
    return real, exp, changed_df


def _ensure_history_columns(df_hist: pd.DataFrame) -> pd.DataFrame:
    out = df_hist.copy()
    for col in HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _next_business_day(date_ts: pd.Timestamp) -> str:
    next_bd = (pd.Timestamp(date_ts).tz_localize(None).normalize() + pd.offsets.BDay(1)).date()
    return str(next_bd)


def upsert_experiment_history(
    experiment_df: pd.DataFrame,
    run_date: pd.Timestamp,
    estimate_map: Dict[str, float],
    history_file: Path = HISTORY_FILE,
) -> int:
    if experiment_df is None or experiment_df.empty:
        return 0

    run_date_str = pd.Timestamp(run_date).tz_localize(None).strftime("%Y-%m-%d")
    ts_now = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S")
    estimate_json = json.dumps(estimate_map, ensure_ascii=False, sort_keys=True)
    projected_exec_date = _next_business_day(pd.Timestamp(run_date))

    recs = []
    for _, row in experiment_df.iterrows():
        sig = str(row.get("signal", HOLD)).upper()
        if sig not in {BUY, SELL, HOLD}:
            sig = HOLD
        recs.append(
            {
                "date": run_date_str,
                "strategy_name": row.get("strategy_name"),
                "signal": sig,
                "price": row.get("price", np.nan),
                "timestamp": ts_now,
                "engine_family": "core",
                "mode": "experiment",
                "source": "daily_param_signals",
                "semantics": "param_preset_daily_signal",
                "estimated_close": row.get("estimated_close", np.nan),
                "estimated_volume": row.get("estimated_volume", np.nan),
                "estimated_inputs_json": estimate_json,
                "trigger_side": sig,
                "triggered": bool(sig != HOLD),
                "projected_exec_date": projected_exec_date,
                "impact_compare": row.get("impact_compare", ""),
            }
        )
    new_df = pd.DataFrame(recs)
    new_df = _ensure_history_columns(new_df)

    if history_file.exists():
        try:
            hist_df = pd.read_csv(history_file)
        except Exception:
            hist_df = pd.DataFrame(columns=HISTORY_COLUMNS)
    else:
        hist_df = pd.DataFrame(columns=HISTORY_COLUMNS)

    hist_df = _ensure_history_columns(hist_df)
    hist_date = hist_df["date"].astype(str)
    hist_mode = hist_df["mode"].astype(str).str.lower()
    mask_old = (
        (hist_date == run_date_str)
        & (hist_df["strategy_name"].astype(str).isin(new_df["strategy_name"].astype(str)))
        & (hist_mode == "experiment")
    )
    hist_df = hist_df.loc[~mask_old].copy()

    if hist_df.empty:
        out_df = new_df.copy()
    else:
        out_df = pd.concat([hist_df, new_df], ignore_index=True)
    out_df = _ensure_history_columns(out_df)
    out_df.to_csv(history_file, index=False)
    return int(new_df.shape[0])


def run_daily_param_signals(
    ticker: str,
    hidden_strategy_presets: Optional[List[str]] = None,
    estimates: Optional[Dict[str, Any]] = None,
    persist_experiment: bool = False,
    run_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    context = build_daily_signal_context(
        ticker=ticker,
        hidden_strategy_presets=hidden_strategy_presets,
        run_date=run_date,
    )
    estimate_map = normalize_estimate_map(estimates)
    missing_inputs = _find_missing_estimate_keys(context, estimate_map)

    real_df = _compute_mode_signals(
        context=context,
        mode="real",
        estimate_map={},
    )

    experiment_df = pd.DataFrame(columns=real_df.columns)
    if not missing_inputs:
        experiment_df = _compute_mode_signals(
            context=context,
            mode="experiment",
            estimate_map=estimate_map,
        )

    real_df, experiment_df, changed_df = _apply_impact_compare(real_df, experiment_df)

    persisted_rows = 0
    if persist_experiment and not experiment_df.empty and not missing_inputs:
        persisted_rows = upsert_experiment_history(
            experiment_df=experiment_df,
            run_date=context.run_date,
            estimate_map=estimate_map,
        )

    return {
        "context": context,
        "real_df": real_df,
        "experiment_df": experiment_df,
        "changed_df": changed_df,
        "missing_inputs": missing_inputs,
        "estimate_map": estimate_map,
        "persisted_rows": persisted_rows,
    }
