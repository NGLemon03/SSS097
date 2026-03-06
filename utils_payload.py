# utils_payload.py
from __future__ import annotations
import io
import pandas as pd
from typing import Tuple, Dict, Any

def _to_df(x):
    """DataFrame 或 JSON(orient='split') → DataFrame；其他回空表。"""
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, str) and x.strip().startswith('{'):
        return pd.read_json(io.StringIO(x), orient='split')
    return pd.DataFrame()

def _normalize_trade_cols(trades: pd.DataFrame) -> pd.DataFrame:
    """標準化交易欄位：trade_date / type / price (+ 可選 weight_change/delta_units/exec_notional)。"""
    if trades is None or len(trades) == 0:
        # 盡量保留可能會用到的關鍵欄位（不存在就略過）
        return pd.DataFrame(columns=[
            "trade_date","type","price","weight_change","delta_units","exec_notional",
            "w_before","w_after","shares_before","shares_after","equity_after","cash_after",
            "invested_pct","cash_pct","position_value","fee_buy","fee_sell","sell_tax","comment"
        ])
    t = trades.copy()
    # 日期
    if "trade_date" not in t.columns:
        if "date" in t.columns:
            t["trade_date"] = pd.to_datetime(t["date"], errors="coerce")
        elif isinstance(t.index, pd.DatetimeIndex):
            t = t.reset_index().rename(columns={"index":"trade_date"})
        else:
            t["trade_date"] = pd.NaT
    else:
        t["trade_date"] = pd.to_datetime(t["trade_date"], errors="coerce")
    # 動作
    if "type" not in t.columns:
        for c in ("side","action"):
            if c in t.columns:
                t["type"] = t[c].astype(str).str.lower(); break
        if "type" not in t.columns:
            t["type"] = "hold"
    else:
        t["type"] = t["type"].astype(str).str.lower()
    # 價格
    if "price" not in t.columns:
        for c in ("open","price_open","exec_price","px","entry_price","close"):
            if c in t.columns:
                t["price"] = pd.to_numeric(t[c], errors="coerce"); break
        if "price" not in t.columns:
            t["price"] = pd.NA
    else:
        t["price"] = pd.to_numeric(t["price"], errors="coerce")
    
    # 欄位別名統一：有其一就建立標準欄位
    alias = {
        "weight_change": ["weight_change","dw","delta_w"],
        "delta_units":   ["delta_units","units_delta","unit_delta","share_delta"],
        "exec_notional": ["exec_notional","notional","amount"],
        "w_before":      ["w_before","w_prev"],
        "w_after":       ["w_after","w_next"],
        "shares_before": ["shares_before","units_before"],
        "shares_after":  ["shares_after","units_after","units","shares"],
        "cash_after":    ["cash_after","cash_post","cash"],
        "equity_after":  ["equity_after","equity"],
        "sell_tax":      ["sell_tax","tax"],
        "comment":       ["comment","note","reason"],
    }
    for tgt, cands in alias.items():
        if tgt not in t.columns:
            for c in cands:
                if c in t.columns:
                    t[tgt] = pd.to_numeric(t[c], errors="coerce") if tgt in ["weight_change","delta_units","exec_notional","w_before","w_after","shares_before","shares_after","equity_after","cash_after","fee_buy","fee_sell","sell_tax"] else t[c]; break
    
    # 推算缺失欄位
    if "weight_change" not in t.columns and {"w_before","w_after"} <= set(t.columns):
        t["weight_change"] = t["w_after"] - t["w_before"]
    
    # 可選欄位（確保存在）
    for opt in ("weight_change","delta_units","exec_notional"):
        if opt not in t.columns:
            t[opt] = pd.NA
    
    cols = [
        "trade_date","type","price","weight_change","delta_units","exec_notional",
        "w_before","w_after","shares_before","shares_after","equity_after","cash_after",
        "invested_pct","cash_pct","position_value","fee_buy","fee_sell","sell_tax","comment"
    ]
    keep = [c for c in cols if c in t.columns]
    return t[keep].sort_values("trade_date")

def _normalize_daily_state(ds: pd.DataFrame) -> pd.DataFrame:
    """標準化 daily_state：index→DatetimeIndex，統一 equity/cash，並推導 rtn/cum_return。"""
    if ds is None or len(ds) == 0:
        return pd.DataFrame(columns=[
            "equity","cash","rtn","cum_return","w","invested_pct","cash_pct","position_value","units"
        ])
    x = ds.copy().rename(columns={
        "equity_after":"equity","cash_after":"cash",
        "total_equity":"equity","total_cash":"cash",
    })
    if "date" in x.columns and not isinstance(x.index, pd.DatetimeIndex):
        x["date"] = pd.to_datetime(x["date"], errors="coerce"); x = x.set_index("date")
    if not isinstance(x.index, pd.DatetimeIndex):
        try: x.index = pd.to_datetime(x.index, errors="coerce")
        except Exception: pass
    if "equity" not in x.columns: x["equity"] = pd.NA
    if "cash"   not in x.columns: x["cash"]   = pd.NA
    if "rtn" not in x.columns: x["rtn"] = x["equity"].pct_change()
    if "cum_return" not in x.columns: x["cum_return"] = (1 + x["rtn"].fillna(0)).cumprod() - 1
    
    # 欄位齊備（有就保留，沒有嘗試推算）
    if "w" not in x.columns and "weight" in x.columns:
        x["w"] = x["weight"]
    if "position_value" not in x.columns and {"equity","cash"} <= set(x.columns):
        x["position_value"] = x["equity"] - x["cash"]
    if "invested_pct" not in x.columns and {"position_value","equity"} <= set(x.columns):
        x["invested_pct"] = (x["position_value"] / x["equity"]).clip(lower=0, upper=1)
    if "cash_pct" not in x.columns and {"cash","equity"} <= set(x.columns):
        x["cash_pct"] = (x["cash"] / x["equity"]).clip(lower=0, upper=1)
    
    cols = ["equity","cash","rtn","cum_return","w","invested_pct","cash_pct","position_value","units"]
    return x[[c for c in cols if c in x.columns]]

def parse_ensemble_payload(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳 (df_raw, trades, daily_state)，欄位均已標準化且對 UI 友善。"""
    df_raw      = _to_df(payload.get("df_raw"))
    trades_in   = payload.get("trades") if payload.get("trades") is not None else payload.get("trade_df")
    trades      = _normalize_trade_cols(_to_df(trades_in))
    daily_state = _normalize_daily_state(_to_df(payload.get("daily_state")))
    return df_raw, trades, daily_state

__all__ = ["_normalize_trade_cols","_normalize_daily_state","parse_ensemble_payload"]
