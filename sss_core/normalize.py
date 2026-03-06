# sss_core/normalize.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

def normalize_trades_for_ui(trades: pd.DataFrame, weight_curve: Optional[pd.Series] = None) -> pd.DataFrame:
    """標準化交易資料用於 UI 顯示，保留計算欄位但隱藏費用相關欄位，並確保權重欄位存在"""
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    
    # 複製資料避免修改原始資料
    trades_ui = trades.copy()
    
    # 標準化欄位名稱
    trades_ui.columns = [str(c).lower() for c in trades_ui.columns]
    
    # 確保必要欄位存在
    if "trade_date" not in trades_ui.columns and "date" in trades_ui.columns:
        trades_ui["trade_date"] = pd.to_datetime(trades_ui["date"], errors="coerce")
    
    if "type" not in trades_ui.columns and "action" in trades_ui.columns:
        trades_ui["type"] = trades_ui["action"].astype(str).str.lower()
    
    if "price" not in trades_ui.columns:
        for c in ["open", "price_open", "exec_price", "px", "close"]:
            if c in trades_ui.columns:
                trades_ui["price"] = trades_ui[c]
                break
    
    # 隱藏費用與 shares 相關欄位（僅用於 UI 顯示，不影響計算）
    fee_columns = ['fee_buy', 'fee_sell', 'sell_tax', 'tax']
    shares_columns = ['shares_before', 'shares_after']
    hide_columns = fee_columns + shares_columns
    display_columns = [c for c in trades_ui.columns if c not in hide_columns]
    
    # 確保權重欄位存在（如果提供了 weight_curve）
    if weight_curve is not None:
        trades_ui = _ensure_weight_columns(trades_ui, weight_curve=weight_curve)
    
    return trades_ui[display_columns]

def normalize_trades_for_plots(
    trades: pd.DataFrame,
    price_series: Optional[pd.Series] = None,   # 新增（可選）
) -> pd.DataFrame:
    """
    標準化交易資料用於繪圖，保留所有欄位。
    若 trades 缺 price 欄，且提供了 price_series，則依 trade_date 對齊補價。
    """
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    
    trades_plot = trades.copy()
    trades_plot.columns = [str(c).lower() for c in trades_plot.columns]
    
    # trade_date / type 標準化
    if "trade_date" not in trades_plot.columns and "date" in trades_plot.columns:
        trades_plot["trade_date"] = pd.to_datetime(trades_plot["date"], errors="coerce")
    if "type" not in trades_plot.columns and "action" in trades_plot.columns:
        trades_plot["type"] = trades_plot["action"].astype(str).str.lower()
    
    # price 標準化：先靠常見欄位，其次用 price_series 對齊補價
    if "price" not in trades_plot.columns:
        for c in ["open", "price_open", "exec_price", "px", "close"]:
            if c in trades_plot.columns:
                trades_plot["price"] = trades_plot[c]
                break
    
    if "price" not in trades_plot.columns:
        trades_plot["price"] = pd.NA  # 先補 NA，若等下有 price_series 再補齊
    
    if price_series is not None:
        # 確保兩邊 index/型別能對齊
        ps = price_series.copy()
        if not isinstance(ps.index, pd.DatetimeIndex):
            ps.index = pd.to_datetime(ps.index, errors="coerce")
        # 以 trade_date 對齊補價（僅在 price 為空或無效時）
        if "trade_date" in trades_plot.columns:
            td = pd.to_datetime(trades_plot["trade_date"], errors="coerce")
            aligned = ps.reindex(td).reset_index(drop=True)
            # 只填那些目前是 NA 的
            need_fill = trades_plot["price"].isna() | ~pd.to_numeric(trades_plot["price"], errors="coerce").notna()
            trades_plot.loc[need_fill, "price"] = aligned[need_fill].values
    
    return trades_plot

def normalize_daily_state(daily_state: Optional[pd.DataFrame]) -> pd.DataFrame:
    """標準化 daily_state 資料並自動糾偏 equity/cash/position_value 的關係"""
    if daily_state is None or len(daily_state) == 0:
        return pd.DataFrame()
    
    ds = daily_state.copy()
    ds.columns = [str(c).lower() for c in ds.columns]
    
    # 統一欄位別名
    alias = {
        "equity_after": "equity",
        "cash_after": "cash",
        "total_equity": "equity",
        "total_cash": "cash",
        "weight": "w",
        "weight_target": "w"
    }
    for old, new in alias.items():
        if old in ds.columns and new not in ds.columns:
            ds[new] = ds[old]
    
    # --- 自動糾偏：equity 應 ≈ position_value + cash ---
    if {"position_value", "cash"}.issubset(ds.columns):
        # 若缺 equity 直接補
        if "equity" not in ds.columns:
            ds["equity"] = ds["position_value"] + ds["cash"]
        else:
            # 1) 用等式誤差檢查
            diff = (ds["equity"] - (ds["position_value"] + ds["cash"])).abs()
            need_fix_by_diff = diff.fillna(0).median() > 1e-6
            
            # 2) 用強負相關檢查（equity 若其實是 position_value，常見 corr(equity,cash) ≈ -1）
            try:
                corr = pd.Series(ds["equity"]).corr(pd.Series(ds["cash"]))
            except Exception:
                corr = None
            need_fix_by_corr = (corr is not None) and (corr < -0.95)
            
            if need_fix_by_diff or need_fix_by_corr:
                ds["equity"] = ds["position_value"] + ds["cash"]
    
    # --- 補強：只有 equity/cash 但強負相關時，把 equity 視為 position_value 再修正 ---
    if "position_value" not in ds.columns and {"equity", "cash"}.issubset(ds.columns):
        try:
            corr = pd.Series(ds["equity"]).corr(pd.Series(ds["cash"]))
        except Exception:
            corr = None
        if corr is not None and corr < -0.95:
            ds["position_value"] = pd.to_numeric(ds["equity"], errors="coerce")
            ds["equity"] = ds["position_value"] + pd.to_numeric(ds["cash"], errors="coerce")
    
    # 百分比欄位（若可計）
    if {"equity", "cash"}.issubset(ds.columns):
        tot = (pd.to_numeric(ds["equity"], errors="coerce")).replace(0, np.nan)
        cash = pd.to_numeric(ds["cash"], errors="coerce")
        posv = pd.to_numeric(ds.get("position_value", np.nan), errors="coerce")
        ds["cash_pct"] = (cash / tot).clip(lower=0, upper=1)
        # 沒有 position_value 就用 equity - cash 推
        if "position_value" not in ds.columns:
            posv = (pd.to_numeric(ds["equity"], errors="coerce") - cash)
            ds["position_value"] = posv
        ds["invested_pct"] = (posv / tot).clip(lower=0, upper=1)
    
    # 末尾加入一個簡單的平衡檢查欄位（方便 csv 直觀看）
    if {"equity","cash"}.issubset(ds.columns):
        try:
            pos = pd.to_numeric(ds.get("position_value", np.nan), errors="coerce")
            eq  = pd.to_numeric(ds["equity"], errors="coerce")
            ca  = pd.to_numeric(ds["cash"], errors="coerce")
            ds["debug_residual_eq_minus_pos_cash"] = (eq - (pos.fillna(0) + ca)).astype(float)
        except Exception:
            pass
    
    return ds

# --- 新增：統一的交易表格顯示格式化 ---
# 欄位顯示名稱對應（中文化）
DISPLAY_NAME = {
    'trade_date': '交易日期',
    'type': '交易類型',
    'price': '價格',
    'weight_change': '權重變化',
    'w_before': '交易前權重',
    'w_after': '交易後權重',
    'delta_units': '股數變化',
    'exec_notional': '執行金額',
    'equity_after': '交易後權益',
    'cash_after': '交易後現金',
    'equity_pct': '權益%',
    'cash_pct': '現金%',
}

# 欄位優先顯示順序
PREFER_ORDER = [
    'trade_date', 'type', 'price',
    'weight_change', 'w_before', 'w_after',
    'delta_units', 'exec_notional',
    'equity_after', 'cash_after', 'equity_pct', 'cash_pct',
]

# 隱藏的欄位（保留計算但不顯示）
HIDE_COLS = {
    'shares_before', 'shares_after', 'fee_buy', 'fee_sell', 'sell_tax',
    'date', 'open', 'equity_open_after_trade'
}

def _ensure_weight_columns(df: pd.DataFrame, weight_curve: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    強化版權重欄位推導，優先使用 weight_curve 來計算權重變化，
    避免「都變 1」的問題。
    
    Args:
        df: 交易 DataFrame
        weight_curve: 權重時間序列（可選），用於更精確的權重推導
        
    Returns:
        包含 weight_change, w_before, w_after 的 DataFrame
    """
    import numpy as np
    d = df.copy()

    # 已有就直接用
    if {'weight_change', 'w_before', 'w_after'}.issubset(d.columns):
        return d

    # 0) 優先使用 weight_curve（如果提供）
    if weight_curve is not None and len(weight_curve) > 0:
        # 確保 weight_curve 有正確的索引
        wc = weight_curve.copy()
        if not isinstance(wc.index, pd.DatetimeIndex):
            wc.index = pd.to_datetime(wc.index, errors='coerce')
        
        # 如果有 trade_date，對齊 weight_curve
        if 'trade_date' in d.columns:
            td = pd.to_datetime(d['trade_date'], errors='coerce')
            # 對齊權重曲線
            aligned_weights = wc.reindex(td)
            
            # 計算權重變化
            d['w_after'] = aligned_weights.values
            d['w_before'] = aligned_weights.shift(1).values
            d['weight_change'] = d['w_after'] - d['w_before']
            
            # 清理 NaN 值
            d['w_before'] = d['w_before'].fillna(0.0)
            d['weight_change'] = d['weight_change'].fillna(0.0)
            
            return d

    # 1) 若已有 w_before/w_after 其中之一，先補另一個與 weight_change
    if 'w_before' in d.columns and 'w_after' in d.columns:
        d['weight_change'] = d['w_after'] - d['w_before']
        return d
    if 'w_after' in d.columns and 'w_before' not in d.columns:
        # 推前值
        d['w_before'] = (d['w_after'].shift(1)).fillna(0.0)
        d['weight_change'] = d['w_after'] - d['w_before']
        return d
    if 'w_before' in d.columns and 'w_after' not in d.columns:
        d['weight_change'] = d.get('dw') \
            if 'dw' in d.columns else d.get('delta_w')
        if 'weight_change' in d.columns and d['weight_change'].notna().any():
            d['w_after'] = (d['w_before'] + d['weight_change']).clip(-1, 1)
            return d

    # 2) 常見欄位：dw / delta_w / target_w
    for dw_col in ['weight_change', 'dw', 'delta_w']:
        if dw_col in d.columns:
            d['weight_change'] = pd.to_numeric(d[dw_col], errors='coerce')
            # 若也提供 target_w / w 之類欄位，補 before/after
            tgt = None
            for wcol in ['w', 'target_w', 'weight', 'w_target']:
                if wcol in d.columns:
                    tgt = pd.to_numeric(d[wcol], errors='coerce')
                    break
            if tgt is not None:
                d['w_after'] = tgt
                d['w_before'] = (tgt - d['weight_change']).fillna(method='ffill').fillna(0.0)
            else:
                d['w_after'] = d['weight_change'].cumsum().clip(-1, 1)
                d['w_before'] = (d['w_after'] - d['weight_change']).fillna(0.0)
            return d

    # 3) 用金額推導：exec_notional /（「交易當下的總資產」）
    #    優先使用「交易前」的資產，如果沒有，才用交易後。
    denom = None
    if {'equity_open_after_trade'}.issubset(d.columns):
        denom = pd.to_numeric(d['equity_open_after_trade'], errors='coerce')
    if denom is None and {'equity_before', 'cash_before'}.issubset(d.columns):
        denom = pd.to_numeric(d['equity_before'], errors='coerce') + pd.to_numeric(d['cash_before'], errors='coerce')
    if denom is None and {'equity_after', 'cash_after'}.issubset(d.columns):
        denom = pd.to_numeric(d['equity_after'], errors='coerce') + pd.to_numeric(d['cash_after'], errors='coerce')

    if denom is not None and 'exec_notional' in d.columns:
        exec_notional = pd.to_numeric(d['exec_notional'], errors='coerce').abs()
        d['weight_change'] = (exec_notional / denom).clip(upper=1.0)
        # 方向由 type 判斷
        if 'type' in d.columns:
            sign = d['type'].astype(str).str.lower().map({'buy': 1, 'long': 1, 'sell': -1, 'force_sell': -1, 'sell_forced': -1}).fillna(0)
            d['weight_change'] = d['weight_change'] * sign
        d['w_after'] = d['weight_change'].cumsum().clip(-1, 1)
        d['w_before'] = (d['w_after'] - d['weight_change']).fillna(0.0)
        return d

    # 4) 最後保底（二元全倉/全空），盡量不要走到這裡
    if 'type' in d.columns:
        m = d['type'].astype(str).str.lower().map({'buy': 1.0, 'long': 1.0, 'sell': -1.0, 'force_sell': -1.0, 'sell_forced': -1.0}).fillna(0.0)
        d['weight_change'] = m
        d['w_after'] = m.cumsum().clip(-1, 1)
        d['w_before'] = (d['w_after'] - d['weight_change']).fillna(0.0)

    return d

def format_trade_df_for_display(df: pd.DataFrame, weight_curve: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    統一的交易表格顯示格式化函式
    
    功能：
    1. 確保權重相關欄位存在（weight_change, w_before, w_after）
    2. 計算百分比欄位（equity_pct, cash_pct）
    3. 隱藏指定欄位（HIDE_COLS）
    4. 重新排序欄位（PREFER_ORDER）
    5. 中文化欄位名稱（DISPLAY_NAME）
    6. 格式化數值顯示
    
    Args:
        df: 原始交易 DataFrame
        weight_curve: 權重時間序列（可選），用於更精確的權重推導
        
    Returns:
        格式化後的 DataFrame，適合 UI 顯示
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    # 複製避免修改原始資料
    d = df.copy()
    
    # 標準化欄位名稱
    d.columns = [str(c).lower() for c in d.columns]
    
    # 1. 確保權重欄位存在
    d = _ensure_weight_columns(d, weight_curve=weight_curve)
    
    # 2. 計算百分比欄位（若有權益/現金）
    if {'equity_after', 'cash_after'}.issubset(d.columns):
        total = d['equity_after'] + d['cash_after']
        d['equity_pct'] = (d['equity_after'] / total).round(4)
        d['cash_pct'] = (d['cash_after'] / total).round(4)
    
    # 3. 隱藏指定欄位
    available_cols = [c for c in d.columns if c not in HIDE_COLS]
    d = d[available_cols]
    
    # 4. 重新排序欄位（優先顯示重要欄位）
    available_prefer = [c for c in PREFER_ORDER if c in d.columns]
    other_cols = [c for c in d.columns if c not in PREFER_ORDER]
    
    if available_prefer:
        d = d[available_prefer + other_cols]
    
    # 5. 中文化欄位名稱
    rename_dict = {old: new for old, new in DISPLAY_NAME.items() if old in d.columns}
    d = d.rename(columns=rename_dict)
    
    # 6. 格式化數值顯示
    # 日期欄位標準化
    date_cols = ['交易日期', 'trade_date', 'date']
    for col in date_cols:
        if col in d.columns:
            d[col] = pd.to_datetime(d[col], errors='coerce').dt.date
            break
    
    # 價格欄位保留兩位小數
    price_cols = ['價格', 'price']
    for col in price_cols:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            break
    
    # 百分比欄位格式化
    pct_cols = ['權益%', '現金%', 'equity_pct', 'cash_pct']
    for col in pct_cols:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    
    return d
