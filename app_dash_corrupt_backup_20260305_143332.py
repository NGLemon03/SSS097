import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import json
import io
from dash.dependencies import ALL
import shutil
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from analysis import config as cfg
import yfinance as yf
import logging
import numpy as np
from urllib.parse import quote as urlparse
from app.bootstrap import configure_runtime_environment
from app.callbacks import (
    register_market_callbacks,
    register_process_callbacks,
    register_strategy_callbacks,
    register_ui_callbacks,
)
from app.services.market_service import refresh_market_data
from app.services.process_service import run_prediction_job
from app.settings import (
    DEFAULT_TICKERS,
    LOG_LEVEL,
    MARKET_UPDATE_TICKERS,
    PRICE_UPDATE_INTERVAL_MS,
)
try:
    from analysis.strategy_manager import manager
except ImportError:
    manager = None # ?脣?
# ??蔭 logger - 雿輻?蝯曹??亥?蝟餌絞嚗?????????
from analysis.logging_config import get_logger, init_logging

configure_runtime_environment()

# ?脣??亥??剁??嗅?頛??
logger = get_logger("SSS.App")

def normalize_daily_state_columns(ds: pd.DataFrame) -> pd.DataFrame:
    """撠?????皞?? daily_state 甈??隤??蝯曹?嚗?
    - ??equity 撖衣????撣?潘??孵???position_value
    - 撱箇? portfolio_value = position_value + cash
    - 靽????invested_pct / cash_pct
    """
    if ds is None or ds.empty:
        return ds
    ds = ds.copy()

    # ?亙歇蝬?? position_value ??cash嚗???亙遣蝡?portfolio_value
    if {'position_value','cash'}.issubset(ds.columns):
        ds['portfolio_value'] = ds['position_value'] + ds['cash']

    # ??? equity + cash ???瘜?-> ?斗? equity ?舐蜇鞈????????
    elif {'equity','cash'}.issubset(ds.columns):
        # ?斗?閬??嚗?? equity/(equity+cash) ??葉雿??憿航? < 0.9嚗???????撣?潦?
        ratio = (ds['equity'] / (ds['equity'] + ds['cash'])).replace([np.inf, -np.inf], np.nan).clip(0,1)
        if ratio.median(skipna=True) < 0.9:
            # ??equity ?嗆????撣??
            ds = ds.rename(columns={'equity':'position_value'})
            ds['portfolio_value'] = ds['position_value'] + ds['cash']
        else:
            # equity 撌脫?蝮質??ｇ???????嚗??瘝?? position_value嚗?
            if 'position_value' not in ds.columns:
                ds['position_value'] = (ds['equity'] - ds['cash']).fillna(0.0)
            ds['portfolio_value'] = ds['equity']

    # ?曉?瘥??雿?絞銝
    if 'portfolio_value' in ds.columns:
        pv = ds['portfolio_value'].replace(0, np.nan)
        if 'invested_pct' not in ds.columns and 'position_value' in ds.columns:
            ds['invested_pct'] = (ds['position_value'] / pv).fillna(0.0).clip(0,1)
        if 'cash_pct' not in ds.columns and 'cash' in ds.columns:
            ds['cash_pct'] = (ds['cash'] / pv).fillna(0.0).clip(0,1)

    # ?箔?????詨捆嚗????equity = portfolio_value嚗????鼓???撘?蝙?剁?
    if 'portfolio_value' in ds.columns:
        ds['equity'] = ds['portfolio_value']

    return ds

def _initialize_app_logging():
    """初始化 App 端日誌設定。"""
    # ?芸?撖阡??閬?????憪??瑼???亥?
    init_logging(enable_file=True)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.info("App Dash ???")
    logger.info(f"Log level: {LOG_LEVEL}")
    logger.info(f"?亥??桅?: {os.path.abspath('analysis/log')}")
    return logger

# --- Smart Leverage 頛??閮???賢? ---
def calculate_smart_leverage_equity(daily_state, df_target, safe_ticker="0050.TW"):
    """
    ???閮?? Smart Leverage 甈???脩?
    撠?daily_state 銝剔? Cash ?其?璅⊥??箸???safe_ticker

    Args:
        daily_state: ???????亦???DataFrame (敹????'w' 甈??甈??)
        df_target: ?餅??扯??Ｙ??寞??豢? (敹????'close')
        safe_ticker: ?脣??扯??Ｙ?隞?Ⅳ (??身 0050.TW)

    Returns:
        靽格?敺?? daily_state (??????閮????equity, cash, position_value)
    """
    try:
        # 1. 皞???豢?
        if daily_state is None or daily_state.empty:
            return daily_state
        if 'w' not in daily_state.columns:
            logger.warning("Smart Leverage: daily_state 缺少 w 欄位，略過重算")
            return daily_state

        # 頛???脣?鞈?? (0050)
        safe_path = Path(f"data/{safe_ticker.replace(':', '_')}_data_raw.csv")
        if not safe_path.exists():
            # 憒??瘝??撠曹?頛?
            logger.info(f"銝?? {safe_ticker} ?冽? Smart Leverage...")
            import sys
            import io
            # 閮剔蔭 stdout 蝺函Ⅳ隞仿???Unicode ?航炊
            if sys.platform == 'win32':
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
            df_safe = yf.download(safe_ticker, start="2010-01-01", auto_adjust=True, progress=False)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            df_safe.to_csv(safe_path, encoding='utf-8')
            df_safe = pd.read_csv(safe_path, index_col=0, parse_dates=True, encoding='utf-8')
        else:
            # ??岫憭?車蝺函Ⅳ霈??CSV
            try:
                df_safe = pd.read_csv(safe_path, index_col=0, parse_dates=True, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df_safe = pd.read_csv(safe_path, index_col=0, parse_dates=True, encoding='cp950')
                except UnicodeDecodeError:
                    df_safe = pd.read_csv(safe_path, index_col=0, parse_dates=True, encoding='latin1')

        # 蝯曹?甈??撠?神
        df_safe.columns = [c.lower() for c in df_safe.columns]

        # ?? 靽桀儔嚗?撥?嗅????Index 頧??蝯曹???datetime ?澆?嚗???斗??嚗?
        # ????寞?嚗?aily_state.index ?航??臬?銝?("2015-07-27")嚗?f_target/df_safe ??datetime64[ns]
        # 銝???賢???timezone aware/naive 銝???渡??瘜?

        # ??? daily_state
        daily_state = daily_state.copy()
        if not isinstance(daily_state.index, pd.DatetimeIndex):
            daily_state.index = pd.to_datetime(daily_state.index, errors='coerce')
        if daily_state.index.tz is not None:
            daily_state.index = daily_state.index.tz_localize(None)

        # ??? df_target
        df_target = df_target.copy()
        if not isinstance(df_target.index, pd.DatetimeIndex):
            df_target.index = pd.to_datetime(df_target.index, errors='coerce')
        if df_target.index.tz is not None:
            df_target.index = df_target.index.tz_localize(None)
        # ?? 撘瑕?頧??嚗?Ⅱ靽?target ????文??舀?摮?
        if 'close' in df_target.columns:
            df_target['close'] = pd.to_numeric(df_target['close'], errors='coerce')

        # ??? df_safe
        if not isinstance(df_safe.index, pd.DatetimeIndex):
            df_safe.index = pd.to_datetime(df_safe.index, errors='coerce')
        if df_safe.index.tz is not None:
            df_safe.index = df_safe.index.tz_localize(None)
        # ?? 撘瑕?頧??嚗?Ⅱ靽?safe ????文??舀?摮?
        if 'close' in df_safe.columns:
            df_safe['close'] = pd.to_numeric(df_safe['close'], errors='coerce')

        # 2. 撠?????頠?(?曉???府?賣迤蝣箏?朣?
        common_idx = daily_state.index.intersection(df_target.index).intersection(df_safe.index)

        # ?? 憿??靽??嚗?Ⅱ靽?common_idx ???摨?
        common_idx = common_idx.sort_values()

        if len(common_idx) == 0:
            logger.warning(f"Smart Leverage: ?⊥?撠?????頠?(daily_state: {daily_state.index.min()} ~ {daily_state.index.max()}, "
                          f"df_target: {df_target.index.min()} ~ {df_target.index.max()}, "
                          f"df_safe: {df_safe.index.min()} ~ {df_safe.index.max()})")
            return daily_state

        logger.info(f"??Smart Leverage: ???撠?????頠賂???{len(common_idx)} 憭?({common_idx[0].date()} ~ {common_idx[-1].date()})")

        # ?芸??豢?
        w = daily_state.loc[common_idx, 'w'].fillna(0).values
        # ?餅?鞈???亙???
        r_target = df_target.loc[common_idx, 'close'].pct_change().fillna(0).values
        # ?脣?鞈???亙???
        r_safe = df_safe.loc[common_idx, 'close'].pct_change().fillna(0).values

        # 3. ???閮???唳楊??
        initial_equity = daily_state.loc[common_idx[0], 'equity']
        smart_equity = np.zeros(len(common_idx))
        smart_equity[0] = initial_equity

        # 鈭斗???????
        FEE_RATE = 0.001425 * 0.3
        TAX_RATE = 0.001  # ETF 蝔??

        w_prev = 0.0

        for i in range(1, len(common_idx)):
            curr_eq = smart_equity[i-1]
            w_prev_val = w[i-1]
            w_curr = w[i] if i < len(w) else w_prev_val

            # 閮??甈??霈???Ｙ??????
            delta_w = w_curr - w_prev
            cost = 0.0
            if abs(delta_w) > 0.001:
                cost = abs(delta_w) * curr_eq * (FEE_RATE + TAX_RATE)

            post_cost_eq = curr_eq - cost

            # ?詨??砍?嚗?蜇?梢? = (?嗆??餅?甈?? * ?餅??梢?) + (?嗆??脣?甈?? * ?脣??梢?)
            combined_ret = (w_curr * r_target[i]) + ((1 - w_curr) * r_safe[i])

            smart_equity[i] = post_cost_eq * (1 + combined_ret)
            w_prev = w_curr

        # 4. ?踵?????豢?
        new_ds = daily_state.loc[common_idx].copy()
        new_ds['equity'] = smart_equity
        # ???閮?? cash (隞?”?脣??其??孵?
        new_ds['cash'] = new_ds['equity'] * (1 - new_ds['w'])
        new_ds['position_value'] = new_ds['equity'] * new_ds['w']

        logger.info(f"??Smart Leverage 閮??摰?? (雿輻? {safe_ticker}嚗??蝯???? {smart_equity[-1]:,.0f})")
        return new_ds

    except Exception as e:
        logger.error(f"??Smart Leverage 閮??憭望?: {e}")
        import traceback
        traceback.print_exc()
        return daily_state  # 憭望?????喳?璅?

def plot_trade_returns_bar(trades_df):
    """
    蝜芾ˊ?桃?鈭斗??????(LIFO - ??脣??箸?)
    ?桃?嚗????????漱????????踹?鋡思????摨?????????雿?仃隤扎?
    """
    if trades_df is None or trades_df.empty:
        return go.Figure()
    
    # 撱箇??舀?
    df = trades_df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    # 蝣箔?敹??甈??摮??
    if 'type' not in df.columns or 'price' not in df.columns:
        return go.Figure()

    # --- 憒??瘝?? return 甈??嚗????LIFO 瞍??瘜??蝞?---
    if 'return' not in df.columns:
        # 撠???賊?甈??
        qty_col = None
        if 'shares' in df.columns: qty_col = 'shares'
        elif 'weight_change' in df.columns: qty_col = 'weight_change'
        elif 'delta_units' in df.columns: qty_col = 'delta_units'
        
        if qty_col:
            # === LIFO (??脣??? 閮???詨? ===
            # inventory 蝯??: list of dict {'price': float, 'qty': float}
            inventory = [] 
            returns = []
            
            # 蝣箔??????迤摨????
            df = df.sort_values('trade_date')
            
            for idx, row in df.iterrows():
                try:
                    trade_type = str(row['type']).lower()
                    price = float(row['price'])
                    qty = abs(float(row[qty_col])) # ???撠?潭?靘輯?蝞?
                    
                    if qty == 0:
                        returns.append(np.nan)
                        continue

                    if 'buy' in trade_type or 'add' in trade_type or 'long' in trade_type:
                        # 鞎瑕?嚗???亙???(Push)
                        inventory.append({'price': price, 'qty': qty})
                        returns.append(np.nan)
                        
                    elif 'sell' in trade_type or 'exit' in trade_type:
                        # 鞈??嚗?????撠曄垢?????(Pop from end = LIFO)
                        remaining_sell_qty = qty
                        total_cost = 0.0
                        matched_qty = 0.0
                        
                        # ?????風 inventory
                        while remaining_sell_qty > 0 and inventory:
                            last_batch = inventory[-1] # ???敺??蝑?
                            
                            if last_batch['qty'] <= remaining_sell_qty:
                                # ???銝??鞈???券????嚗???曉?銝??
                                cost = last_batch['qty'] * last_batch['price']
                                total_cost += cost
                                matched_qty += last_batch['qty']
                                remaining_sell_qty -= last_batch['qty']
                                inventory.pop() # 蝘駁????
                            else:
                                # ???憭?都嚗??????典?嚗??銝?????
                                cost = remaining_sell_qty * last_batch['price']
                                total_cost += cost
                                matched_qty += remaining_sell_qty
                                inventory[-1]['qty'] -= remaining_sell_qty # ?湔?摨怠??賊?
                                remaining_sell_qty = 0 # 鞈??鈭?
                        
                        # 閮?????
                        if matched_qty > 0:
                            avg_buy_price = total_cost / matched_qty
                            ret = (price - avg_buy_price) / avg_buy_price
                            returns.append(ret)
                        else:
                            # ?潛?蝛箏澈摮?都???航??舀?蝛箇??交?鞈??蝻箸?)嚗??閮?? 0
                            returns.append(0.0)
                    else:
                        returns.append(np.nan)
                        
                except Exception:
                    returns.append(np.nan)
            
            df['return'] = returns
        else:
            return go.Figure()

    # --- 隞乩?蝜芸???摩 ---
    valid_trades = df[
        (df['type'].astype(str).str.contains('sell', case=False, na=False)) & 
        (df['return'].notna())
    ].copy()
    
    if valid_trades.empty:
        fig = go.Figure()
        fig.update_layout(title="?桃?鈭斗??梢????雿?(?∟都?箇???", template='plotly_dark', height=400)
        return fig

    valid_trades['return'] = pd.to_numeric(valid_trades['return'], errors='coerce').fillna(0)
    colors = ['#00CC96' if x > 0 else '#EF553B' for x in valid_trades['return']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=valid_trades['trade_date'],
        y=valid_trades['return'],
        marker_color=colors,
        name='?桃???? (LIFO)'
    ))
    
    fig.add_hline(y=0, line_color="white", line_width=1)

    fig.update_layout(
        title="餈??鈭斗???? (LIFO 瞍??瘜?",
        xaxis_title="鞈???交?",
        yaxis_title="?梢???(vs ?餈?眺?亙?)",
        yaxis_tickformat=".2%",
        template='plotly_dark',
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    fig.add_annotation(
        text="閮鳴??∪??脣??箸?(LIFO)嚗?????餈??蝑??雿?????",
        xref="paper", yref="paper",
        x=0, y=1.1, showarrow=False,
        font=dict(size=10, color="gray")
    )
    return fig


def _new_vote_diagnostics():
    return {
        "requested": 0,
        "loaded": 0,
        "failed": 0,
        "reasons_by_strategy": {},
    }


def compute_ensemble_vote_series(daily_state, strategy_name, warehouse_file="strategy_warehouse.json"):
    """Compute ensemble long-vote series aligned to daily_state index."""
    diagnostics = _new_vote_diagnostics()

    if not strategy_name or not str(strategy_name).startswith("Ensemble"):
        diagnostics["reasons_by_strategy"]["_global"] = "non_ensemble_strategy"
        return None, None, diagnostics
    if daily_state is None or daily_state.empty:
        diagnostics["reasons_by_strategy"]["_global"] = "empty_daily_state"
        return None, None, diagnostics
    if manager is None:
        diagnostics["reasons_by_strategy"]["_global"] = "strategy_manager_unavailable"
        return None, None, diagnostics

    ds = daily_state.copy()
    if "date" in ds.columns:
        idx = pd.to_datetime(ds["date"], errors="coerce")
    else:
        idx = pd.to_datetime(ds.index, errors="coerce")
    idx = pd.DatetimeIndex(idx).normalize()
    idx = idx[idx.notna()]
    if len(idx) == 0:
        diagnostics["reasons_by_strategy"]["_global"] = "invalid_daily_state_index"
        return None, None, diagnostics

    try:
        active_strats = manager.load_strategies(warehouse_file)
    except Exception as exc:
        logger.warning("Failed to load strategies from warehouse %s: %s", warehouse_file, exc)
        diagnostics["reasons_by_strategy"]["_global"] = "warehouse_load_failed"
        return None, None, diagnostics

    strat_list = [s.get("name", "").replace(".csv", "") for s in active_strats if s.get("name")]
    diagnostics["requested"] = len(strat_list)
    if not strat_list:
        diagnostics["reasons_by_strategy"]["_global"] = "no_active_strategies"
        return None, None, diagnostics

    temp_out_dir = Path("sss_backtest_outputs/dash_temp")
    search_paths = [temp_out_dir, Path("sss_backtest_outputs")]
    archive_dir = Path("archive")
    if archive_dir.exists():
        search_paths.extend(list(archive_dir.glob("*/sss_backtest_outputs")))

    file_map = {}
    for s_name in strat_list:
        found = None
        for search_dir in search_paths:
            if not search_dir.exists():
                continue
            candidates = list(search_dir.glob(f"*{s_name}*.csv"))
            if candidates:
                found = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                break
        if found:
            file_map[s_name] = found

    def _read_trades(path: Path, strategy_key: str):
        try:
            df = pd.read_csv(path)
            cols = {str(c).lower(): c for c in df.columns}
            date_col = cols.get("trade_date", cols.get("date"))
            type_col = cols.get("type", cols.get("action"))
            if not date_col or not type_col:
                logger.warning("[%s] Missing required columns in %s", strategy_key, path)
                return None, "missing_required_columns"

            df["trade_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
            type_map = {
                "buy": "buy",
                "entry": "buy",
                "long": "buy",
                "add": "buy",
                "sell": "sell",
                "exit": "sell",
                "short": "sell",
                "sell_forced": "sell",
                "forced_sell": "sell",
            }
            df["type"] = df[type_col].astype(str).str.lower().str.strip().map(type_map)
            invalid_rows = int((df["trade_date"].isna() | df["type"].isna()).sum())
            if invalid_rows > 0:
                logger.warning("[%s] %d invalid trade rows dropped from %s", strategy_key, invalid_rows, path)
            df = df.dropna(subset=["trade_date", "type"])
            if df.empty:
                logger.warning("[%s] No valid trade rows in %s", strategy_key, path)
                return None, "no_valid_trade_rows"
            return df[["trade_date", "type"]], None
        except Exception as exc:
            logger.warning("[%s] Failed reading %s: %s", strategy_key, path, exc)
            return None, "read_trade_file_failed"

    def _build_position_series(trades_df, index):
        pos = pd.Series(0.0, index=index)
        if trades_df is None or trades_df.empty:
            return pos
        for _, row in trades_df.iterrows():
            dt = row["trade_date"]
            act = row["type"]
            if dt in pos.index:
                if act == "buy":
                    pos.loc[dt:] = 1.0
                elif act == "sell":
                    pos.loc[dt:] = 0.0
        return pos

    vote_df = []
    for s_name in strat_list:
        path = file_map.get(s_name)
        if not path or not path.exists():
            diagnostics["failed"] += 1
            diagnostics["reasons_by_strategy"][s_name] = "trade_file_not_found"
            logger.warning("[%s] Trade file not found", s_name)
            continue

        trades, reason = _read_trades(path, s_name)
        if trades is None:
            diagnostics["failed"] += 1
            diagnostics["reasons_by_strategy"][s_name] = reason or "trade_parse_failed"
            continue

        pos = _build_position_series(trades, idx)
        vote_df.append(pos)
        diagnostics["loaded"] += 1

    if diagnostics["loaded"] == 0:
        diagnostics["reasons_by_strategy"]["_global"] = "no_vote_series_loaded"
        return None, None, diagnostics

    long_votes = pd.concat(vote_df, axis=1).sum(axis=1)
    threshold = None
    if strategy_name == "Ensemble_Majority":
        import math

        threshold = math.ceil(0.55 * diagnostics["loaded"])

    return long_votes, threshold, diagnostics


# ATR 閮???賣?
def calculate_atr(df, window):
    """計算 ATR (Average True Range)。"""
    try:
        if df is None or getattr(df, "empty", True):
            return pd.Series(dtype=float)

        cols_lower = {str(c).lower(): c for c in df.columns}
        high_col = cols_lower.get("high")
        low_col = cols_lower.get("low")
        close_col = cols_lower.get("close")

        # 若缺 High/Low，退化為以 close 變動近似 TR
        if high_col and low_col and close_col:
            high = pd.to_numeric(df[high_col], errors="coerce")
            low = pd.to_numeric(df[low_col], errors="coerce")
            close = pd.to_numeric(df[close_col], errors="coerce")
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        elif close_col:
            close = pd.to_numeric(df[close_col], errors="coerce")
            true_range = close.diff().abs()
        else:
            return pd.Series(index=df.index, dtype=float)

        atr = true_range.rolling(window=int(window), min_periods=int(window)).mean()
        return atr if atr is not None else pd.Series(index=df.index, dtype=float)
    except Exception:
        return pd.Series(index=getattr(df, "index", None), dtype=float)


def _compute_atr_ratio_series(df_raw, fast_win=20, slow_win=60):
    """閮?? ATR(20)/ATR(60) 瘥?澆?????怠??祆?瘣????""
    if df_raw is None or getattr(df_raw, "empty", True):
        return pd.Series(dtype=float)
    try:
        atr_fast = calculate_atr(df_raw, fast_win)
        atr_slow = calculate_atr(df_raw, slow_win)
        if atr_fast is None or atr_slow is None:
            return pd.Series(index=df_raw.index, dtype=float)
        ratio = pd.to_numeric(atr_fast, errors="coerce") / pd.to_numeric(atr_slow, errors="coerce").replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio.index = pd.to_datetime(ratio.index)
        return ratio
    except Exception:
        return pd.Series(index=df_raw.index, dtype=float)


def _apply_soft_risk_cap(w_series, mask, ratio_series, atr_threshold, risk_cap, softness=1.2):
    """Soft risk cap based on ATR exceedance severity."""
    if w_series is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    w = pd.Series(pd.to_numeric(w_series, errors="coerce"), index=w_series.index).fillna(0.0).clip(lower=0.0)
    mask_s = pd.Series(mask, index=w.index).fillna(False).astype(bool)
    ratio_s = (
        pd.to_numeric(ratio_series, errors="coerce").reindex(w.index)
        if ratio_series is not None
        else pd.Series(np.nan, index=w.index)
    )

    threshold = float(atr_threshold) if atr_threshold and float(atr_threshold) > 0 else 1.0
    floor_cap = float(risk_cap) if risk_cap is not None else 1.0
    floor_cap = float(np.clip(floor_cap, 0.0, 1.0))
    softness = float(softness) if softness is not None else 1.2

    dynamic_cap = pd.Series(1.0, index=w.index)
    exceed = (ratio_s / threshold - 1.0).clip(lower=0.0)
    cap_on_mask = floor_cap + (1.0 - floor_cap) * np.exp(-softness * exceed)
    dynamic_cap.loc[mask_s] = cap_on_mask.loc[mask_s].fillna(floor_cap)

    w_adj = w.copy()
    w_adj.loc[mask_s] = np.minimum(w_adj.loc[mask_s], dynamic_cap.loc[mask_s])
    return w_adj, dynamic_cap


def _build_benchmark_df(df_raw):
    """Build normalized OHLCV benchmark dataframe with lowercase columns."""
    if df_raw is None or getattr(df_raw, "empty", True):
        return pd.DataFrame()

    src = df_raw.copy()
    out = pd.DataFrame(index=pd.to_datetime(src.index))
    cols_lower = {str(c).lower(): c for c in src.columns}

    for col in ("open", "high", "low", "close", "volume"):
        src_col = cols_lower.get(col)
        if src_col is not None:
            out[col] = pd.to_numeric(src[src_col], errors="coerce")

    return out

def calculate_equity_curve(open_px, w, cap, atr_ratio):
    """Calculate simple equity proxy for diagnostics."""
    try:
        open_s = pd.to_numeric(open_px, errors="coerce")
        w_s = pd.to_numeric(w, errors="coerce").reindex(open_s.index).fillna(0.0)
        equity = (open_s * w_s * float(cap)).cumsum()
        return equity
    except Exception as e:
        logger.warning(f"calculate_equity_curve failed: {e}")
        return None

def calculate_trades_from_equity(equity_curve, open_px, w, cap, atr_ratio):
    """Infer trade rows from weight changes for diagnostics."""
    try:
        if equity_curve is None or getattr(equity_curve, "empty", True):
            return None
        w_s = pd.to_numeric(w, errors="coerce")
        changes = w_s.diff().abs()
        trade_dates = changes[changes > 0.01].index
        trades = [{"trade_date": d, "return": 0.0} for d in trade_dates]
        return pd.DataFrame(trades) if trades else pd.DataFrame(columns=["trade_date", "return"])
    except Exception as e:
        logger.warning(f"calculate_trades_from_equity failed: {e}")
        return None

def df_from_pack(data):
    """敺?pack_df 蝯????JSON 摮?葡閫?? DataFrame"""
    import io, json
    import pandas as pd
    
    # 憒??撌脩???DataFrame嚗???亥???
    if isinstance(data, pd.DataFrame):
        return data
    
    # 瑼Ｘ??臬???None ??征摮?葡
    if data is None:
        return pd.DataFrame()
    
    # 憒???臬?銝莎??脰?憿??瑼Ｘ?
    if isinstance(data, str):
        if data == "" or data == "[]":
            return pd.DataFrame()
        # ???閰?split ???????閮?
        for orient in ("split", None):
            try:
                kw = {"orient": orient} if orient else {}
                return pd.read_json(io.StringIO(data), **kw)
            except Exception:
                pass
        return pd.DataFrame()
    
    if isinstance(data, (list, dict)):
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()
    
    return pd.DataFrame()

def series_from_pack(data):
    """敺?pack_series 蝯????JSON 摮?葡閫?? Series"""
    import io
    import pandas as pd
    
    # 憒??撌脩???Series嚗???亥???
    if isinstance(data, pd.Series):
        return data
    
    # 瑼Ｘ??臬???None ??征摮?葡
    if data is None:
        return pd.Series(dtype=float)
    
    # 憒???臬?銝莎??脰?憿??瑼Ｘ?
    if isinstance(data, str):
        if data == "" or data == "[]":
            return pd.Series(dtype=float)
        # Series 銋??閰?split
        for orient in ("split", None):
            try:
                kw = {"orient": orient} if orient else {}
                return pd.read_json(io.StringIO(data), typ="series", **kw)
            except Exception:
                pass
        return pd.Series(dtype=float)
    
    if isinstance(data, (list, dict)):
        try:
            return pd.Series(data)
        except Exception:
            return pd.Series(dtype=float)
    
    return pd.Series(dtype=float)

from SSSv096 import (
    param_presets, load_data, compute_single, compute_dual, compute_RMA,
    compute_ssma_turn_combined, backtest_unified, plot_stock_price, plot_equity_cash, plot_weight_series,
    calculate_holding_periods, get_param_preset_names
)

# ?? 雿輻??啁??亙ㄞ摨????極?瘀?Pickle + Gzip + Base64嚗?
try:
    from sss_core.data_utils import pack_df_robust as pack_df, pack_series_robust as pack_series
    from sss_core.data_utils import unpack_df_robust as df_from_pack, unpack_series_robust as series_from_pack
except Exception:
    # Fallback ?啗???(??data_utils 銝????
    try:
        from sss_core.schemas import pack_df, pack_series
    except Exception:
        from schemas import pack_df, pack_series

# ?臬?甈??甈??蝣箔??賢?
try:
    from sss_core.normalize import _ensure_weight_columns
except Exception:
    # 憒???⊥??臬?嚗??蝢拐???征???撘????fallback
    def _ensure_weight_columns(df):
        return df

# ??身雿?? get_version_history_html
try:
    from version_history import get_version_history_html
except ImportError:
    def get_version_history_html() -> str:
        return "<b>?⊥?頛?????甇瑕?閮??</b>"

# --- 靽???暸?Store ?????JSON-safe ---
def _pack_any(x):
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return pack_df(x)          # orient="split" + date_format="iso"
    if isinstance(x, pd.Series):
        return pack_series(x)      # orient="split" + date_format="iso"
    return x

def _pack_result_for_store(result: dict) -> dict:
    # 蝯曹??????pandas ?拐辣頧??摮?葡嚗?SON嚗?
    keys = [
        'trade_df', 'trades_df', 'signals_df',
        'equity_curve', 'cash_curve', 'price_series',
        'daily_state', 'trade_ledger',
        'daily_state_std', 'trade_ledger_std',
        'weight_curve',
        # ???啣?嚗??摮??憟??? baseline
        'daily_state_base', 'trade_ledger_base', 'weight_curve_base',
        # ???啣?嚗??摮?valve ???
        'daily_state_valve', 'trade_ledger_valve', 'weight_curve_valve', 'equity_curve_valve'
    ]
    out = dict(result)
    for k in keys:
        if k in out:
            out[k] = _pack_any(out[k])
    # ?血???datetime tuple ??trades 頧??摨?????雿???砌????嚗?
    if 'trades' in out and isinstance(out['trades'], list):
        out['trades'] = [
            (str(t[0]), t[1], str(t[2])) if isinstance(t, tuple) and len(t) == 3 else t
            for t in out['trades']
        ]
    return out


def _first_non_empty_result_df(result: dict, keys: list[str]) -> pd.DataFrame:
    """靘????? result 銝剔洵銝???閫??銝??蝛箇? DataFrame??""
    if not isinstance(result, dict):
        return pd.DataFrame()
    for key in keys:
        raw = result.get(key)
        if raw is None or (isinstance(raw, str) and raw == ""):
            continue
        try:
            df = df_from_pack(raw)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()


def _first_non_empty_result_series(result: dict, keys: list[str]) -> pd.Series:
    """靘????? result 銝剔洵銝???閫??銝??蝛箇? Series??""
    if not isinstance(result, dict):
        return pd.Series(dtype=float)
    for key in keys:
        raw = result.get(key)
        if raw is None or (isinstance(raw, str) and raw == ""):
            continue
        try:
            s = series_from_pack(raw)
        except Exception:
            continue
        if isinstance(s, pd.Series) and not s.empty:
            return s
    return pd.Series(dtype=float)


def _is_valid_number(v) -> bool:
    try:
        return pd.notna(v) and np.isfinite(float(v))
    except Exception:
        return False


def _ensure_exposure_metrics(metrics: dict, result: dict, daily_state_hint: pd.DataFrame | None = None) -> None:
    """
    鋆???典?瘥??/撟游?????????Ensemble 撣貊撩????????
    ?芸????蝻箏潭?鋆??銝??????????潦?
    """
    if not isinstance(metrics, dict):
        return

    w = pd.Series(dtype=float)

    if isinstance(daily_state_hint, pd.DataFrame) and not daily_state_hint.empty and "w" in daily_state_hint.columns:
        w = pd.to_numeric(daily_state_hint["w"], errors="coerce").dropna()

    if w.empty:
        ds = _first_non_empty_result_df(result, ["daily_state_valve", "daily_state_std", "daily_state", "daily_state_base"])
        if not ds.empty and "w" in ds.columns:
            w = pd.to_numeric(ds["w"], errors="coerce").dropna()

    if w.empty:
        ws = _first_non_empty_result_series(result, ["weight_curve_valve", "weight_curve", "weight_curve_base"])
        if not ws.empty:
            w = pd.to_numeric(ws, errors="coerce").dropna()

    if w.empty:
        return

    if not _is_valid_number(metrics.get("time_in_market")):
        metrics["time_in_market"] = float((w > 1e-6).mean())
    if not _is_valid_number(metrics.get("turnover_py")):
        metrics["turnover_py"] = float(w.diff().abs().fillna(0.0).sum() / max(len(w), 1) * 252.0)

default_tickers = list(DEFAULT_TICKERS)

# ?∪??湔?皜??嚗???詨???摩?梁?嚗?
# 0050.TW ?冽? Smart Leverage嚗?????隞?????韏瑟???
TICKER_LIST = list(MARKET_UPDATE_TICKERS)


def get_strategy_names(include_legacy: bool = False):
    try:
        return get_param_preset_names(include_hidden=include_legacy)
    except Exception:
        # Fallback for backward compatibility.
        if include_legacy:
            return list(param_presets.keys())
        return [k for k, v in param_presets.items() if not bool(v.get("ui_hidden", False))]


all_strategy_names = get_strategy_names(include_legacy=True)
strategy_names = list(all_strategy_names)

# Visibility control: options and defaults are kept in-code for easy maintenance.
HIDE_PRESET_CHECKLIST_ORDER = [
    "single trial917",
    "Ensemble_Majority",
    "Ensemble_Proportional",
    "Single 2",
    "single_1887",
    "Single 3",
    "RMA_69",
    "RMA_669",
    "STM0",
    "STM1",
    "STM3",
    "STM4",
    "STM_1939",
    "STM_2414_273",
]
HIDE_PRESET_DEFAULTS = [

    "RMA_69",
    "RMA_669",
    "STM0",
    "STM3",
    "STM4",

]


def _build_hide_strategy_options(all_names):
    ordered = []
    seen = set()
    for name in HIDE_PRESET_CHECKLIST_ORDER:
        if name in all_names and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in all_names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return [{"label": name, "value": name} for name in ordered]


hide_strategy_options = _build_hide_strategy_options(all_strategy_names)
hide_strategy_defaults = [name for name in HIDE_PRESET_DEFAULTS if name in all_strategy_names]

# ??????澈銝???詨??賊?
warehouse_options = []
if manager:
    try:
        files = manager.list_warehouses()
        warehouse_options = [
            {'label': '?? ?曉蝴 (Active)' if f == "strategy_warehouse.json" else f, 'value': f}
            for f in files
        ]
    except Exception:
        warehouse_options = [{'label': '?? ?曉蝴 (Active)', 'value': 'strategy_warehouse.json'}]

theme_list = ['theme-dark', 'theme-light', 'theme-blue']

def get_theme_label(theme):
    if theme == 'theme-dark':
        return '?? 瘛梯?銝駁?'
    elif theme == 'theme-light':
        return '?? 瘛箄?銝駁?'
    else:
        return '?? ???銝駁?'

def get_column_display_name(column_name):
    """撠?????雿??頧???箔葉??＊蝷箏?蝔?""
    column_mapping = {
        'trade_date': '鈭斗??交?',
        'signal_date': '閮???交?',
        'type': '鈭斗?憿??',
        'price': '?寞?',
        'weight_change': '甈??霈??',
        'w_before': '鈭斗??????,
        'w_after': '鈭斗?敺????,
        'delta_units': '?⊥?霈??',
        'exec_notional': '?瑁????',
        'equity_after': '鈭斗?敺????,
        'cash_after': '鈭斗?敺????,
        'equity_pct': '甈??%',
        'cash_pct': '?暸?%',
        'invested_pct': '???瘥??',
        'position_value': '?其??孵?,
        'return': '?梢???,
        'comment': '??酉',
        'shares': '?⊥?',
        'reason': '???',
        'fee': '???鞎?,
        'net_amount': '瘛券?',
        'leverage_ratio': '瑽?▼瘥?,
        'strategy_version': '蝑?????',
        'indicator_smaa': '???SMAA',
        'indicator_base': '???BASE',
        'indicator_sd': '???SD',
    }
    return column_mapping.get(column_name, column_name)

# 憿舐內撅斗?????????銝剜?嚗?
DISPLAY_NAME = {
    'trade_date': '鈭斗??交?',
    'signal_date': '閮???交?',
    'type': '鈭斗?憿??',
    'price': '?寞?',
    'weight_change': '甈??霈??',
    'w_before': '鈭斗??????,
    'w_after': '鈭斗?敺????,
    'delta_units': '?⊥?霈??',
    'exec_notional': '?瑁????',
    'equity_after': '鈭斗?敺????,
    'cash_after': '鈭斗?敺????,
    'equity_pct': '甈??%',
    'cash_pct': '?暸?%',
    'invested_pct': '???瘥??',
    'position_value': '?其?撣??,
    'return': '?梢?',
    'shares': '?⊥?',
    'reason': '???',
    'fee': '???鞎?,
    'net_amount': '瘛券?',
    'leverage_ratio': '瑽?▼瘥?,
    'strategy_version': '蝑?????',
    'indicator_smaa': '???SMAA',
    'indicator_base': '???BASE',
    'indicator_sd': '???SD',
    'comment': '??酉',
}

# 憿舐內撅扎??????雿??閮??靽????I 銝?＊蝷綽?
HIDE_COLS = {
    'shares_before', 'shares_after', 'fee_buy', 'fee_sell', 'sell_tax', 'tax',
    'date', 'open', 'equity_open_after_trade'  # ??雿???啁????嚗?絞銝?梯?
}

# 憿舐內撅斗?雿??摨??摮?????嚗??摮??撠梯歲???
PREFER_ORDER = [
    'trade_date','signal_date','type','price',
    'weight_change','w_before','w_after',
    'return','shares','reason','fee','net_amount',
    'leverage_ratio','strategy_version',
    'indicator_smaa','indicator_base','indicator_sd',
    'delta_units','exec_notional',
    'equity_after','cash_after','equity_pct','cash_pct',
    'invested_pct','position_value','comment'
]

# 鈭斗???敦?詨?甈??嚗?nsemble/??nsemble??＊蝷綽?蝻箏潛???
UNIFIED_MIN_COLUMNS = [
    'trade_date', 'signal_date', 'type', 'price',
    'weight_change', 'w_before', 'w_after',
    'return', 'shares', 'reason', 'fee', 'net_amount',
    'leverage_ratio', 'strategy_version',
    'indicator_smaa', 'indicator_base', 'indicator_sd',
]

def format_trade_like_df_for_display(df):
    """憿舐內撅歹??梯???? ??鋆????? ???澆?????銝剜?甈?? ??摰?????"""
    import pandas as pd
    if df is None or len(df)==0:
        return df

    d = df.copy()

    # 1) ?梯????
    hide = [c for c in HIDE_COLS if c in d.columns]
    if hide:
        d = d.drop(columns=hide, errors='ignore')

    # 1.5) ?詨?甈??鋆??嚗?? Ensemble/??nsemble 甈??蝯??銝?湛?
    for col in UNIFIED_MIN_COLUMNS:
        if col not in d.columns:
            d[col] = pd.NA

    # 2) 敹??甈??鋆???曉?瘥???亙歇摮??撠梁????
    if {'equity_after','cash_after'}.issubset(d.columns):
        tot = d['equity_after'] + d['cash_after']
        if 'equity_pct' not in d.columns:
            d['equity_pct'] = d.apply(
                lambda r: "" if pd.isna(r['equity_after']) or pd.isna(tot.loc[r.name]) or tot.loc[r.name] <= 0
                          else f"{(r['equity_after']/tot.loc[r.name]):.2%}", axis=1)
        if 'cash_pct' not in d.columns:
            d['cash_pct'] = d.apply(
                lambda r: "" if pd.isna(r['cash_after']) or pd.isna(tot.loc[r.name]) or tot.loc[r.name] <= 0
                          else f"{(r['cash_after']/tot.loc[r.name]):.2%}", axis=1)

    # 3) ?澆???
    def _fmt_date_col(s):
        dt = pd.to_datetime(s, errors='coerce')
        return dt.dt.strftime('%Y-%m-%d').fillna("")

    for col in ['trade_date','signal_date']:
        if col in d.columns:
            d[col] = _fmt_date_col(d[col])

    if 'type' in d.columns:
        type_map = {
            'buy': '鞎瑕?',
            'add': '??Ⅳ',
            'long': '???',
            'sell': '鞈??',
            'sell_forced': '撘瑕?鞈??',
            'forced_sell': '撘瑕?鞈??',
            'exit': '?箏?',
            'hold': '???',
        }
        d['type'] = d['type'].astype(str).str.lower().map(type_map).fillna(d['type'])

    if 'price' in d.columns:
        d['price'] = d['price'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
    if 'weight_change' in d.columns:
        d['weight_change'] = d['weight_change'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    for col in ['w_before','w_after']:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{x:,.3f}" if pd.notnull(x) else "")
    for col in ['exec_notional','equity_after','cash_after','position_value']:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    if 'delta_units' in d.columns:
        d['delta_units'] = d['delta_units'].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    if 'shares' in d.columns:
        d['shares'] = d['shares'].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    if 'return' in d.columns:
        d['return'] = d['return'].apply(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
    for col in ['fee', 'net_amount']:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
    if 'leverage_ratio' in d.columns:
        d['leverage_ratio'] = d['leverage_ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    for col in ['indicator_smaa', 'indicator_base', 'indicator_sd']:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    if 'reason' in d.columns:
        reason_map = {
            'signal_entry': '閮???脣?',
            'signal_exit': '閮???箏?',
            'stop_loss': '???',
            'force_liquidate': '撘瑕?撟喳?,
            'forced_sell': '撘瑕?鞈??',
            'sell_forced': '撘瑕?鞈??',
            'loan_repayment': '??????',
            'end_of_period': '???撟喳?,
            'ensemble_rebalance_buy': '??像銵∟眺??,
            'ensemble_rebalance_sell': '??像銵∟都??,
        }
        reason_series = d['reason'].fillna("").astype(str).str.strip()
        d['reason'] = reason_series.map(reason_map).fillna(reason_series)
    if 'strategy_version' in d.columns:
        d['strategy_version'] = d['strategy_version'].fillna("").astype(str)

    # 4) 摰?????嚗??????函?甈??嚗?
    exist = [c for c in PREFER_ORDER if c in d.columns]
    others = [c for c in d.columns if c not in exist]
    d = d[exist + others]

    # 5) 銝剜?甈??
    d = d.rename(columns={k: DISPLAY_NAME.get(k, k) for k in d.columns})
    return d

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# --------- Dash Layout ---------
app.layout = html.Div([
    dcc.Store(id='theme-store', data='theme-dark'),
    dcc.Store(id='price-update-store'),
    dcc.Interval(id='price-update-interval', interval=PRICE_UPDATE_INTERVAL_MS, n_intervals=0),

    # === Header Controls ===
    html.Div([
        html.Button(id='theme-toggle', n_clicks=0, children='?? 瘛梯?銝駁?', className='btn btn-secondary main-header-bar'),
        html.Button(id='history-btn', n_clicks=0, children='?? ???瘝輸?', className='btn btn-secondary main-header-bar ms-2'),
    ], className='header-controls'),

    # ???瘝輸?璅⊥?獢?
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("????祆窒?拍???)),
        dbc.ModalBody([
            dcc.Markdown(get_version_history_html(), dangerously_allow_html=True)
        ], className='version-history-modal-body'),
        dbc.ModalFooter(
            dbc.Button("???", id="history-close", className="ms-auto", n_clicks=0)
        ),
    ], id="history-modal", size="lg", is_open=False),

    # === Top Control Bar (隞???湧?甈? ===
    dbc.Container([
        # Row 1: ?箸???葫???
        dbc.Row(id='ctrl-row-basic', children=[
            dbc.Col([
                html.Label("?∠巨隞??", className="small mb-1"),
                dcc.Dropdown(id='ticker-dropdown', options=[{'label': t, 'value': t} for t in default_tickers],
                            value=default_tickers[0]),
            ], width=2),
            dbc.Col([
                html.Label("韏瑕??交?", className="small mb-1"),
                dcc.Input(id='start-date', type='text', value='2010-01-01'),
            ], width=1),
            dbc.Col([
                html.Label("蝯???交?", className="small mb-1"),
                dcc.Input(id='end-date', type='text', value='', placeholder='??征=???),
            ], width=1),
            dbc.Col([
                html.Label("?詨????", className="small mb-1"),
                dcc.Slider(id='discount-slider', min=0.1, max=0.7, step=0.01, value=0.3,
                          marks={0.1:'0.1',0.3:'0.3',0.5:'0.5',0.7:'0.7'},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], width=3),
            dbc.Col([
                html.Label("?瑕???(bars)", className="small mb-1"),
                dcc.Input(id='cooldown-bars', type='number', min=0, max=20, value=3, style={'width': '70px'}),
            ], width=1),
            dbc.Col([
                html.Div([
                    dbc.Checkbox(id='bad-holding', value=False, label="鞈???梢???-20%,蝑??銝?活鞈??", className="small"),
                    dbc.Checkbox(id='auto-run', value=True, label="?芸????嚗???貉??????葫嚗?, className="small mt-1"),
                    html.Label("?暸?閬?????蝑??", className="small mt-2 mb-1"),
                    html.Div(
                        dbc.Checklist(
                            id='hide-strategy-presets',
                            options=hide_strategy_options,
                            value=hide_strategy_defaults,
                            inline=False,
                            className="small",
                        ),
                        style={
                            "maxHeight": "70px",
                            "overflowY": "auto",
                            "border": "1px solid #3a3a3a",
                            "padding": "6px",
                            "borderRadius": "6px",
                        },
                    ),
                ]),
            ], width=4),
        ], className='p-2 mb-2', style={'borderRadius': '4px'}),

        # === 撅??/?嗅???? ===
        html.Div([
            dbc.Button(
                "??? 憿舐內/?梯? ?脤???? (憸券??仿?????亙?澈??nsemble)",
                id="collapse-button",
                className="mb-2 w-100",
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
        ]),

        # === Collapsible Area: ??ㄨ Row 2 & Row 3 ===
        dbc.Collapse(
            id="collapse-settings",
            is_open=False,  # ??身???
            children=[
                # Row 2: ?典?憸券??仿??批?
                dbc.Row(id='ctrl-row-risk', children=[
                    dbc.Col([
                        html.Label("?? ?典?憸券??仿?", id='label-risk-title', className="small mb-1 fw-bold"),
                        dbc.Checkbox(id='global-apply-switch', value=False, label="????典????憟??", className="small"),
                    ], width=2),
                    dbc.Col([
                        html.Label("憸券? CAP", className="small mb-1"),
                        dcc.Input(id='risk-cap-input', type='number', min=0.1, max=1.0, step=0.1, value=0.3, style={'width': '70px'}),
                    ], width=1),
                    dbc.Col([
                        html.Label("ATR 瘥?潮?瑼?, className="small mb-1"),
                        dcc.Input(id='atr-ratio-threshold', type='number', min=0.5, max=2.0, step=0.1, value=1.0, style={'width': '70px'}),
                    ], width=1),
                    dbc.Col([
                        html.Label("皜祈岫?賊?", className="small mb-1"),
                        dbc.Checkbox(id='force-valve-trigger', value=False, label="撘瑕?閫貊??仿?", className="small text-danger"),
                        dbc.Checkbox(id='smart-leverage-switch', value=False, label="Smart Leverage (0050隞???暸?)", className="small text-success fw-bold"),
                    ], width=2),
                    dbc.Col([
                        html.Div(id='risk-valve-status', className="small p-2", style={"borderRadius":"4px"}),
                    ], width=6),
                ], className='p-2 mb-2', style={'borderRadius': '4px'}),

                # Row 3: 蝑????澈 & ?拍? Ensemble ??? + ?瑁????
                dbc.Row(id='ctrl-row-ensemble', children=[
                    # ??澈?豢?
                    dbc.Col([
                        html.Label("?? 蝑????澈???", className="small mb-1 fw-bold"),
                        dcc.Dropdown(id='warehouse-dropdown', options=warehouse_options, value='strategy_warehouse.json'),
                    ], width=2),

                    # Ensemble_Majority ??? (???)
                    dbc.Col([
                        html.Label("?? Ensemble Majority", id='label-maj-title', className="small mb-1 fw-bold"),
                        html.Div([
                            html.Span("floor:", className="small me-1"),
                            dcc.Input(id='majority-floor', type='number', min=0.0, max=1.0, step=0.05, value=0.2,
                                     style={'width': '55px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("ema:", className="small me-1"),
                            dcc.Input(id='majority-ema-span', type='number', min=1, max=20, step=1, value=3,
                                     style={'width': '50px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("?cap:", className="small me-1"),
                            dcc.Input(id='majority-delta-cap', type='number', min=0.0, max=1.0, step=0.05, value=0.3,
                                     style={'width': '55px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("cool:", className="small me-1"),
                            dcc.Input(id='majority-min-cooldown', type='number', min=0, max=30, step=1, value=3,
                                     style={'width': '50px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("dw:", className="small me-1"),
                            dcc.Input(id='majority-min-trade-dw', type='number', min=0.0, max=0.5, step=0.01, value=0.05,
                                     style={'width': '55px', 'display': 'inline-block'}),
                        ], style={'whiteSpace': 'nowrap'}),
                    ], width=4),

                    # Ensemble_Proportional ??? (蝬??)
                    dbc.Col([
                        html.Label("?? Ensemble Proportional", id='label-prop-title', className="small mb-1 fw-bold"),
                        html.Div([
                            html.Span("floor:", className="small me-1"),
                            dcc.Input(id='prop-floor', type='number', min=0.0, max=1.0, step=0.05, value=0.2,
                                     style={'width': '55px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("ema:", className="small me-1"),
                            dcc.Input(id='prop-ema-span', type='number', min=1, max=20, step=1, value=3,
                                     style={'width': '50px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("?cap:", className="small me-1"),
                            dcc.Input(id='prop-delta-cap', type='number', min=0.0, max=1.0, step=0.05, value=0.2,
                                     style={'width': '55px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("cool:", className="small me-1"),
                            dcc.Input(id='prop-min-cooldown', type='number', min=0, max=30, step=1, value=3,
                                     style={'width': '50px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("dw:", className="small me-1"),
                            dcc.Input(id='prop-min-trade-dw', type='number', min=0.0, max=0.5, step=0.01, value=0.03,
                                     style={'width': '55px', 'display': 'inline-block'}),
                        ], style={'whiteSpace': 'nowrap'}),
                    ], width=4),

                    # ?瑁????
                    dbc.Col([
                        html.Label("\u00A0", className="small mb-1"),
                        html.Button("?? 銝?萄?銵?????皜?, id='run-btn', n_clicks=0, className="btn btn-primary w-100"),
                    ], width=2),
                ], className='p-2 mb-2', style={'borderRadius': '4px'}),
            ]  # 蝯?? Collapse children
        ),  # 蝯?? Collapse

    ], fluid=True, className='mb-3'),

    # === Main Content Area ===
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Tabs(
                    id='main-tabs',
                    value='backtest',
                    children=[
                        dcc.Tab(label="蝑????葫", value="backtest"),
                        dcc.Tab(label="?????亥眺鞈??瘥??", value="compare"),
                        dcc.Tab(label="?? 憓?撥???", value="enhanced"),
                        dcc.Tab(label="??? 瘥??閮???唳?摰?, value="daily_signal"),
                    ],
                    className='main-tabs-bar'
                ),
                html.Div(id='tab-content', className='main-content-panel')
            ], width=12)
        ])
    ], fluid=True),

    # === Hidden Store ===
    dcc.Loading(dcc.Store(id='backtest-store'), type="circle"),

], id='main-bg', className='theme-dark')

# --------- Externalized callbacks (incremental split) ---------
register_market_callbacks(app, tickers=TICKER_LIST)

# --------- ?脤?????憛?撅??/?嗅??批? ---------
def toggle_advanced_settings(n, is_open):
    """?批??脤?????憛??撅??/?嗅?"""
    if n:
        return not is_open
    return is_open

# --------- ??澈??”?湔? ---------
def update_warehouse_list(n):
    """憛怠???澈銝???詨?"""
    if manager:
        files = manager.list_warehouses()
        options = [
            {'label': '?? ?曉蝴 (Active)' if f == "strategy_warehouse.json" else f, 'value': f}
            for f in files
        ]
        return options, "strategy_warehouse.json"
    return [], "strategy_warehouse.json"


register_ui_callbacks(
    app,
    toggle_advanced_settings_func=toggle_advanced_settings,
    update_warehouse_list_func=update_warehouse_list,
)

# --------- 憸券??仿???????---------
@app.callback(
    Output('risk-valve-status', 'children'),
    [
        Input('global-apply-switch', 'value'),
        Input('risk-cap-input', 'value'),
        Input('atr-ratio-threshold', 'value'),
        Input('force-valve-trigger', 'value'),
        Input('ticker-dropdown', 'value'),
        Input('start-date', 'value'),
        Input('end-date', 'value')
    ]
)
def update_risk_valve_status(global_apply, risk_cap, atr_ratio, force_trigger, ticker, start_date, end_date):
    """????湔?憸券??仿????＊蝷?""
    logger.debug(f"=== 憸券??仿???????===")
    logger.debug(f"global_apply: {global_apply}")
    logger.debug(f"risk_cap: {risk_cap}")
    logger.debug(f"atr_ratio: {atr_ratio}")
    logger.debug(f"force_trigger: {force_trigger}")
    logger.debug(f"ticker: {ticker}")
    logger.debug(f"start_date: {start_date}")
    logger.debug(f"end_date: {end_date}")
    
    if not global_apply:
        logger.debug("憸券??仿??芸???)
        return html.Div([
            html.Small("?? 憸券??仿??芸???, style={"color":"#dc3545","fontWeight":"bold"}),
            html.Br(),
            html.Small("暺??銝??銴??獢???典?撅憸券??批?", style={"color":"#666","fontSize":"10px"})
        ])
    
    # 憒?????嚗??閰西??交???蒂閮?? ATR 瘥??
    try:
        if ticker and start_date:
            logger.debug(f"???頛???豢?: ticker={ticker}, start_date={start_date}, end_date={end_date}")
            df_raw, _ = load_data(ticker, start_date, end_date if end_date else None, "Self")
            logger.debug(f"?豢?頛??蝯??: 蝛?{df_raw.empty}, 敶Ｙ?={df_raw.shape if not df_raw.empty else 'N/A'}")
            
            if not df_raw.empty:
                # 閮?? ATR 瘥??
                logger.debug("???閮?? ATR 瘥??)
                atr_20 = calculate_atr(df_raw, 20)
                atr_60 = calculate_atr(df_raw, 60)
                logger.debug(f"ATR 閮??摰??: atr_20={type(atr_20)}, atr_60={type(atr_60)}")
                
                # ????日?鞈??
                debug_info = []
                debug_info.append(f"?豢?甈??: {list(df_raw.columns)}")
                debug_info.append(f"?豢?銵??: {len(df_raw)}")
                debug_info.append(f"ATR(20) 憿??: {type(atr_20)}")
                debug_info.append(f"ATR(60) 憿??: {type(atr_60)}")
                
                if atr_20 is not None:
                    debug_info.append(f"ATR(20) ?瑕漲: {len(atr_20) if hasattr(atr_20, '__len__') else 'N/A'}")
                    debug_info.append(f"ATR(20) ??征?? {atr_20.notna().sum() if hasattr(atr_20, 'notna') else 'N/A'}")
                
                if atr_60 is not None:
                    debug_info.append(f"ATR(60) ?瑕漲: {len(atr_60) if hasattr(atr_60, '__len__') else 'N/A'}")
                    debug_info.append(f"ATR(60) ??征?? {atr_60.notna().sum() if hasattr(atr_60, 'notna') else 'N/A'}")
                
                # 蝣箔? ATR ?豢????
                if (atr_20 is not None and atr_60 is not None and 
                    hasattr(atr_20, 'empty') and hasattr(atr_60, 'empty') and
                    not atr_20.empty and not atr_60.empty):
                    
                    # 瑼Ｘ??臬???雲憭????征??
                    atr_20_valid = atr_20.dropna()
                    atr_60_valid = atr_60.dropna()
                    
                    if len(atr_20_valid) > 0 and len(atr_60_valid) > 0:
                        # ????啁? ATR ?潮脰?瘥??
                        atr_20_latest = atr_20_valid.iloc[-1]
                        atr_60_latest = atr_60_valid.iloc[-1]
                        
                        debug_info.append(f"ATR(20) ??啣? {atr_20_latest:.6f}")
                        debug_info.append(f"ATR(60) ??啣? {atr_60_latest:.6f}")
                        
                        if atr_60_latest > 0:
                            atr_ratio_current = atr_20_latest / atr_60_latest
                            debug_info.append(f"ATR 瘥?? {atr_ratio_current:.4f}")
                            
                            # ?斗??臬??閬?孛?潮◢?芷??
                            valve_triggered = atr_ratio_current > atr_ratio
                            
                            # 憒?????撘瑕?閫貊?嚗??撘瑕?閫貊?憸券??仿?
                            if force_trigger:
                                valve_triggered = True
                                logger.info(f"撘瑕?閫貊?憸券??仿????")
                            
                            # 閮??憸券??仿??????亥?
                            logger.debug(f"ATR 瘥?潸?蝞? {atr_20_latest:.6f} / {atr_60_latest:.6f} = {atr_ratio_current:.4f}")
                            logger.debug(f"憸券??仿??瑼? {atr_ratio}, ?嗅?瘥?? {atr_ratio_current:.4f}")
                            logger.debug(f"憸券??仿?閫貊?: {'?? if valve_triggered else '??}")
                            logger.debug(f"憸券??仿???? {'?? 閫貊?' if valve_triggered else '?? 甇?虜'}")
                            
                            status_color = "#dc3545" if valve_triggered else "#28a745"
                            status_icon = "??" if valve_triggered else "??"
                            status_text = "閫貊?" if valve_triggered else "甇?虜"
                            
                            # ???撘瑕?閫貊??????＊蝷?
                            force_status = ""
                            if force_trigger:
                                force_status = html.Br() + html.Small("?? 撘瑕?閫貊?撌脣???, style={"color":"#dc3545","fontWeight":"bold","fontSize":"10px"})
                            
                            return html.Div([
                                html.Div([
                                    html.Small(f"{status_icon} 憸券??仿???? {status_text}", 
                                              style={"color":status_color,"fontWeight":"bold","fontSize":"12px"}),
                                    force_status,
                                    html.Br(),
                                    html.Small(f"ATR(20)/ATR(60) = {atr_ratio_current:.2f}", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small(f"?瑼餃? {atr_ratio}", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small(f"憸券?CAP: {risk_cap*100:.0f}%", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small(f"?暸?靽??銝??: {(1-risk_cap)*100:.0f}%", style={"color":"#666","fontSize":"11px"}),
                                    html.Br(),
                                    html.Small("--- ?日?鞈?? ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                                    html.Small([html.Div(info) for info in debug_info], style={"color":"#999","fontSize":"9px"})
                                ])
                            ])
                        else:
                            logger.warning(f"ATR(60) ?潛? 0嚗??瘜??蝞???? {atr_60_latest:.6f}")
                            return html.Div([
                                html.Small("?? ATR 閮???啣虜", style={"color":"#ffc107","fontWeight":"bold"}),
                                html.Br(),
                                html.Small(f"ATR(60) ?潛? {atr_60_latest:.6f}嚗??瘜??蝞????, style={"color":"#666","fontSize":"10px"}),
                                html.Br(),
                                html.Small("--- ?日?鞈?? ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                                html.Small([html.Div(info) for info in debug_info], style={"color":"#999","fontSize":"9px"})
                            ])
                    else:
                        logger.warning(f"ATR ?豢?銝?雲: ATR(20) ?????{len(atr_20_valid)}, ATR(60) ?????{len(atr_60_valid)}")
                        return html.Div([
                            html.Small("?? ATR ?豢?銝?雲", style={"color":"#ffc107","fontWeight":"bold"}),
                            html.Br(),
                            html.Small(f"ATR(20) ????? {len(atr_20_valid)}, ATR(60) ????? {len(atr_60_valid)}", style={"color":"#666","fontSize":"10px"}),
                            html.Br(),
                            html.Small("--- ?日?鞈?? ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                            html.Small([html.Div(info) for info in debug_info], style={"color":"#999","fontSize":"9px"})
                        ])
                else:
                    logger.warning("ATR ?豢??⊥?嚗??瘜??蝞????)
                    return html.Div([
                        html.Small("?? ATR ?豢??⊥?", style={"color":"#ffc107","fontWeight":"bold"}),
                        html.Br(),
                        html.Small("?⊥?閮?? ATR 瘥??, style={"color":"#666","fontSize":"10px"}),
                        html.Br(),
                        html.Small("--- ?日?鞈?? ---", style={"color":"#999","fontSize":"10px","fontStyle":"italic"}),
                        html.Small([html.Div(info) for info in debug_info], style={"color":"#666","fontSize":"9px"})
                    ])

            else:
                logger.warning(f"?⊥?頛???豢?: ticker={ticker}, start_date={start_date}")
                return html.Div([
                    html.Small("?? ?⊥?頛???豢?", style={"color":"#ffc107","fontWeight":"bold"}),
                    html.Br(),
                    html.Small("隢???豢??∠巨隞???????, style={"color":"#666","fontSize":"10px"})
                ])
        else:
            logger.debug("蝑???豢?頛??嚗???豢??∠巨隞???????)
            return html.Div([
                html.Small("?? 蝑???豢?頛??", style={"color":"#ffc107","fontWeight":"bold"}),
                html.Br(),
                html.Small("隢?????蟡其誨????交?", style={"color":"#666","fontSize":"10px"})
            ])
    except Exception as e:
        logger.error(f"憸券??仿??????啣仃?? {e}")
        return html.Div([
            html.Small("?? 閮??銝?..", style={"color":"#ffc107","fontWeight":"bold"}),
            html.Br(),
            html.Small(f"?航炊: {str(e)}", style={"color":"#666","fontSize":"10px"})
        ])

# --------- ?瑁???葫銝血???Store ---------
def run_backtest(n_clicks, auto_run, _price_update, hidden_strategy_presets, ticker, start_date, end_date, discount, cooldown, bad_holding,
                global_apply, risk_cap, atr_ratio, force_trigger, warehouse_file,
                maj_floor, maj_ema, maj_delta, maj_cooldown, maj_dw,
                prop_floor, prop_ema, prop_delta, prop_cooldown, prop_dw,
                stored_data):
    # === 隤輯岫?亥?嚗????DEBUG 蝝????＊蝷綽?===
    logger.debug(f"run_backtest 鋡怨矽??- n_clicks: {n_clicks}, auto_run: {auto_run}, trigger: {ctx.triggered_id}")
    
    # 蝘駁??芸?敹怠?皜??嚗??????冽?銵??
    # 霈?joblib.Memory ?芸?蝞∠?敹怠?嚗???券?閬?????皜??
    ctx_trigger = ctx.triggered_id
    if n_clicks is None and not auto_run and ctx_trigger != 'price-update-store':
        logger.debug(f"?拇?餈??嚗?_clicks={n_clicks}, auto_run={auto_run}")
        return stored_data
    
    # 頛???豢?
    df_raw, df_factor = load_data(ticker, start_date, end_date, "Self")
    if df_raw.empty:
        logger.warning(f"?⊥?頛?? {ticker} ?????)
        return {"error": f"?⊥?頛?? {ticker} ?????}
    
    # ?芸? auto-run ??True ?????◤暺?????蝞?
    if not auto_run and ctx_trigger not in ('run-btn', 'price-update-store'):
        logger.debug(f"頝喲???葫嚗?uto_run={auto_run}, ctx_trigger={ctx_trigger}")
        return stored_data
    
    hidden_set = set(hidden_strategy_presets or [])
    active_strategy_names = [name for name in get_strategy_names(include_legacy=True) if name not in hidden_set]
    if not active_strategy_names:
        logger.warning("?∪??函??伐??桀?蝭拚?敺???交??桃?蝛箝?)
        return {"error": "?∪??函??伐?隢??瘨?????????格?瑼Ｘ? param_presets 閮剖???}

    logger.info(
        f"????瑁???葫 - ticker: {ticker}, 蝑???? {len(active_strategy_names)}, ?梯??? {len(hidden_set)}"
    )
    results = {}
    
    # === ?啣?嚗??撅憸券??仿?閫貊????蕭頩?===
    valve_triggered = False
    atr_ratio_current = None
    
    for strat in active_strategy_names:
        # ?芯蝙??param_presets 銝剔????
        strat_params = param_presets[strat].copy()
        strat_type = strat_params["strategy_type"]
        smaa_src = strat_params.get("smaa_source", "Self")
        atr_ratio_current = None
        
        # ?箸?????亥??亙?????豢?
        df_raw, df_factor = load_data(ticker, start_date, end_date if end_date else None, smaa_source=smaa_src)
        
        # ????典?憸券??仿?閮剖?嚗??????剁?
        logger.debug(f"[{strat}] 憸券??仿??????? global_apply={global_apply}, 憿??={type(global_apply)}")
        if global_apply:
            logger.info(f"[{strat}] ????典?憸券??仿?: CAP={risk_cap}, ATR瘥?潮?瑼?{atr_ratio}")
            
            # 閮?? ATR 瘥?潘?雿輻???唳????????潭?隤?＊蝷綽?
            try:
                atr_20 = calculate_atr(df_raw, 20)
                atr_60 = calculate_atr(df_raw, 60)
                
                # 蝣箔? ATR ?豢????
                if not atr_20.empty and not atr_60.empty:
                    atr_20_valid = atr_20.dropna()
                    atr_60_valid = atr_60.dropna()
                    
                    # 瑼Ｘ?璅???賊??臬?頞喳?
                    min_samples_20, min_samples_60 = 30, 60  # ?喳??閬?30 ??60 ??見??
                    if len(atr_20_valid) < min_samples_20 or len(atr_60_valid) < min_samples_60:
                        logger.warning(f"[{strat}] ATR 璅??銝?雲嚗?0??{len(atr_20_valid)}/{min_samples_20}, 60??{len(atr_60_valid)}/{min_samples_60}")
                        atr_ratio_current = None
                    
                    atr_20_latest = atr_20_valid.iloc[-1]
                    atr_60_latest = atr_60_valid.iloc[-1]
                    
                    # 瑼Ｘ? ATR ?潭??血???
                    if atr_60_latest <= 0 or not np.isfinite(atr_60_latest):
                        logger.warning(f"[{strat}] ATR(60) ?潛?撣? {atr_60_latest}嚗?歲??◢?芷??")
                        atr_ratio_current = None
                    
                    if atr_20_latest <= 0 or not np.isfinite(atr_20_latest):
                        logger.warning(f"[{strat}] ATR(20) ?潛?撣? {atr_20_latest}嚗?歲??◢?芷??")
                        atr_ratio_current = None
                    
                    if atr_60_latest > 0 and np.isfinite(atr_20_latest) and np.isfinite(atr_60_latest):
                        atr_ratio_current = atr_20_latest / atr_60_latest
                        logger.debug(f"[{strat}] ATR ratio={atr_ratio_current:.4f} (20={atr_20_latest:.4f}, 60={atr_60_latest:.4f})")
                    else:
                        atr_ratio_current = None
                else:
                    logger.warning(f"[{strat}] ATR 閮??蝯???箇征")
                    
                # 撘瑕?閫貊???身蝵格?閮?
                if force_trigger:
                    valve_triggered = True
                    logger.info(f"[{strat}] ?? 撘瑕?閫貊?憸券??仿????")
                    
            except Exception as e:
                logger.warning(f"[{strat}] ATR 閮??憭望?: {e}")
        else:
            logger.info(f"[{strat}] ?芸??典?撅憸券??仿?")
        
        if strat_type == 'ssma_turn':
            calc_keys = [
                'linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win',
                'volume_target_pass_rate', 'volume_target_lookback'
            ]
            ssma_params = {k: v for k, v in strat_params.items() if k in calc_keys}
            backtest_params = ssma_params.copy()
            backtest_params['stop_loss'] = strat_params.get('stop_loss', 0.0)
            
            # ???閮??蝑??靽∟?嚗???箏??詨??賢歇蝬?◤憸券??仿?隤踵?嚗?
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_raw, df_factor, **ssma_params, smaa_source=smaa_src)
            if df_ind.empty:
                continue
            result = backtest_unified(
                df_ind,
                strat_type,
                backtest_params,
                buy_dates,
                sell_dates,
                discount=discount,
                trade_cooldown_bars=cooldown,
                bad_holding=bad_holding,
            )
            
            # === ??ssma_turn 銋???券◢?芷??嚗?? Ensemble 銝?渡?敺?蔭閬?神嚗?===
            if global_apply:
                # ?斗??臬?閬?孛?潘??????ATR 瑼Ｘ???撥?嗉孛?潔??湛?
                valve_triggered_local = False
                ratio_local = None
                try:
                    atr_20 = calculate_atr(df_raw, 20)
                    atr_60 = calculate_atr(df_raw, 60)
                    if not atr_20.empty and not atr_60.empty:
                        a20 = atr_20.dropna().iloc[-1]
                        a60 = atr_60.dropna().iloc[-1]
                        if a60 > 0:
                            ratio_local = float(a20 / a60)
                            valve_triggered_local = (ratio_local > atr_ratio)  # ??脤????銝?湛?雿輻? ">"
                except Exception:
                    pass

                if force_trigger:
                    valve_triggered_local = True
                    if ratio_local is None:
                        ratio_local = 1.5

                if valve_triggered_local:
                    from SSS_EnsembleTab import risk_valve_backtest, CostParams
                    # ??? open ?對?df_raw 甈????迂?臬?撖?
                    open_px = df_raw['open'] if 'open' in df_raw.columns else df_raw['close']
                    # 敺??皜祈撓?箸? w嚗???冽?皞?? daily_state嚗???????停?典? daily_state嚗?
                    w_series = None
                    try:
                        ds_std = df_from_pack(result.get('daily_state_std'))
                        if ds_std is not None and not ds_std.empty and 'w' in ds_std.columns:
                            w_series = ds_std['w']
                    except Exception:
                        pass
                    if w_series is None:
                        ds = df_from_pack(result.get('daily_state'))
                        if ds is not None and not ds.empty and 'w' in ds.columns:
                            w_series = ds['w']

                    if w_series is not None:
                        # 鈭斗????嚗?? Ensemble ???銝?湛?
                        trade_cost = strat_params.get('trade_cost', {})
                        cost_params = CostParams(
                            buy_fee_bp=float(trade_cost.get("buy_fee_bp", 4.27)),
                            sell_fee_bp=float(trade_cost.get("sell_fee_bp", 4.27)),
                            sell_tax_bp=float(trade_cost.get("sell_tax_bp", 30.0))
                        )

                        ratio_series_local = _compute_atr_ratio_series(df_raw).reindex(w_series.index)
                        mask_series_local = ratio_series_local > float(atr_ratio)
                        if force_trigger:
                            mask_series_local[:] = True
                        w_series_soft, _ = _apply_soft_risk_cap(
                            w_series.astype(float),
                            mask_series_local,
                            ratio_series_local,
                            atr_ratio,
                            risk_cap,
                        )

                        # === ?典?憟??憸券??仿?嚗?Ⅱ靽???訾??湔?(2025/08/20) ===
                        global_valve_params = {
                            "open_px": open_px,
                            "w": w_series_soft,
                            "cost": cost_params,
                            "benchmark_df": df_raw,
                            "mode": "cap",
                            "cap_level": 1.0,
                            "slope20_thresh": 0.0, 
                            "slope60_thresh": 0.0,
                            "atr_win": 20, 
                            "atr_ref_win": 60,
                            "atr_ratio_mult": float(ratio_local if ratio_local is not None else atr_ratio),   # ?乩???local ratio嚗?停??local嚗?????撅 atr_ratio
                            "use_slopes": True,
                            "slope_method": "polyfit",
                            "atr_cmp": "gt"
                        }
                        
                        # 閮???典?憸券??仿???蔭
                        logger.info(f"[Global] 憸券??仿???蔭: cap_level={global_valve_params['cap_level']}, atr_ratio_mult={global_valve_params['atr_ratio_mult']}")
                        
                        rv = risk_valve_backtest(**global_valve_params)

                        # app_dash.py / 2025-08-22 15:30
                        # ?典?憸券??仿?嚗?????摮?baseline ??valve ???嚗??閬?神璅????
                        # 1) 靽?? valve ????啣??券?
                        result['equity_curve_valve']     = pack_series(rv["daily_state_valve"]["equity"])
                        result['daily_state_valve']      = pack_df(rv["daily_state_valve"])
                        result['trade_ledger_valve']     = pack_df(rv["trade_ledger_valve"])
                        result['weight_curve_valve']     = pack_series(rv["weights_valve"])
                        
                        # 2) 靽?? baseline ????啣??券?嚗?????瘝??嚗?
                        if "daily_state_base" not in result and result.get("daily_state") is not None:
                            result["daily_state_base"] = result["daily_state"]
                        if "trade_ledger_base" not in result and result.get("trade_ledger") is not None:
                            result["trade_ledger_base"] = result["trade_ledger"]
                        if "weight_curve_base" not in result and result.get("weight_curve") is not None:
                            result["weight_curve_base"] = result["weight_curve"]
                        # 蝯?UI ???閮??銝???蝭????堆?
                        result['valve'] = {
                            "applied": True,
                            "cap": float(risk_cap),
                            "atr_ratio": ratio_local
                        }
                        
                        logger.info(f"[{strat}] SSMA 憸券??仿?撌脣??剁?cap={risk_cap}, ratio={ratio_local:.4f}嚗?)
                    else:
                        logger.warning(f"[{strat}] SSMA ?⊥????甈??摨??嚗?歲??◢?芷??憟??")
                else:
                    logger.info(f"[{strat}] SSMA 憸券??仿??芾孛?潘?雿輻????蝯??")
                    # 蝯?UI ???閮???芾孛?潘?
                    result['valve'] = {
                        "applied": False,
                        "cap": float(risk_cap),
                        "atr_ratio": ratio_local if ratio_local is not None else "N/A"
                    }
        elif strat_type == 'ensemble':# 雿輻??啁? ensemble_runner ?踹?敺芰?靘?陷
            result = {}
            if manager:
                # ?? 雿輻??詨???warehouse_file
                active_strats = manager.load_strategies(warehouse_file)
                # ???蝑????迂 (?駁? .csv ?舀???
                strat_list = [s['name'].replace('.csv', '') for s in active_strats]

                # ?????? 憓?撥???獢??撠?- ?舀?敺?archive ?曉??箏仃??IS/OOS 瑼?? ??????
                TRADES_DIR = Path("sss_backtest_outputs")

                # 摰?儔???頝臬?嚗??????嗅??桅?嚗??銝????? archive
                search_paths = [TRADES_DIR]
                archive_dir = Path("archive")
                if archive_dir.exists():
                    # ?曉?????隞賜???葉??sss_backtest_outputs
                    archive_subdirs = list(archive_dir.glob("*/sss_backtest_outputs"))
                    search_paths.extend(archive_subdirs)
                    logger.debug(f"?? ?菜葫??{len(archive_subdirs)} ???隞賜????撠???交?撠????)

                file_map = {}
                missing_strategies = []

                for s_name in strat_list:
                    found_file = None
                    # ??風?????賜?頝臬?
                    for search_dir in search_paths:
                        if not search_dir.exists():
                            continue
                        # ?????摩嚗?祝擛?????
                        # s_name ?航???"trades_from_results_RMA_trial337" ??"RMA_trial337"
                        candidates = list(search_dir.glob(f"*{s_name}*.csv"))
                        if candidates:
                            # ?曉???啁?????憒?????銴??
                            found_file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                            location = "?嗅??桅?" if search_dir == TRADES_DIR else f"??遢 ({search_dir.parent.name})"
                            logger.debug(f"   ??[{s_name}] ?????? -> {found_file.name} (靘??: {location})")
                            break  # ?曉?鈭?停??迫????嗡??桅?

                    if found_file:
                        file_map[s_name] = found_file
                    else:
                        missing_strategies.append(s_name)
                        logger.warning(f"   ??? [{s_name}] 敺孵??箏仃嚗??瘜???嗅????摮???????CSV")

                # ?勗????蝯??
                logger.debug(f"?? [{strat}] 瑼?????蝯梯?: ??? {len(file_map)}/{len(strat_list)}嚗??憭?{len(missing_strategies)}")

                if not file_map:
                    logger.error(f"??[{strat}] ?⊥??曉?隞颱????????交?獢??Ensemble 蝯?迫")
                    # ?? ????航炊蝯????UI嚗????舫?暺?歲??
                    results[strat] = {
                        'error': f'?曆??唬遙雿???交?獢?(?梢?閬?{len(strat_list)} ??',
                        'trade_df': pack_df(pd.DataFrame()),
                        'daily_state': pack_df(pd.DataFrame()),
                        'metrics': {
                            'total_return': None,
                            'annual_return': None,
                            'sharpe_ratio': None,
                            'calmar_ratio': None,
                            'max_drawdown': None,
                            'num_trades': 0
                        },
                        'error_detail': f"??澈 {warehouse_file} 銝剔?蝑??瑼??撌脰◤撠?????憭晞??瑼Ｘ? archive/ ?桅???
                    }
                    continue

                if missing_strategies:
                    logger.warning(f"??? [{strat}] ?典?蝑??瑼???箏仃 ({len(missing_strategies)}/{len(strat_list)})嚗?nsemble 撠?蝙?典??函? {len(file_map)} ????亦匱蝥??銵?)
                    logger.warning(f"   ?箏仃皜??: {', '.join(missing_strategies[:5])}{'...' if len(missing_strategies) > 5 else ''}")

                if strat_list:
                    strat_params['strategies'] = strat_list
                    strat_params['file_map'] = file_map  # ?? ?喲?瑼??撠??銵?
                    logger.info(f"[{strat}] 敺?{warehouse_file} 瘜典?蝑??: {strat_list}")
                else:
                    logger.warning(f"[{strat}] ??澈 {warehouse_file} ?舐征???Ensemble ?航??⊥????")
            else:
                logger.error("?⊥?頛?? Strategy Manager")
                
            try:
                from runners.ensemble_runner import run_ensemble_backtest
                from SSS_EnsembleTab import EnsembleParams, CostParams, RunConfig
                
                # ??SSSv096 ??楷?????文像
                flat_params = {}
                flat_params.update(strat_params.get('params', {}))
                flat_params.update(strat_params.get('trade_cost', {}))
                flat_params['method'] = strat_params.get('method', 'majority')
                flat_params['ticker'] = ticker
                
                # 雿輻?瘥???瑼駁???N 霈????仃??
                if 'majority_k' in flat_params and flat_params.get('method') == 'majority':
                    flat_params['majority_k_pct'] = 0.55
                    flat_params.pop('majority_k', None)
                    logger.info(f"[Ensemble] 雿輻?瘥???瑼?majority_k_pct={flat_params['majority_k_pct']}")
                
                # ?萄遣??蔭 - ?寞?蝑??憿??雿輻?撠????UI ???
                # Ensemble_Majority 雿輻???????嚗?nsemble_Proportional 雿輻?蝬?????
                if strat == "Ensemble_Majority":
                    ui_floor = maj_floor if maj_floor is not None else 0.2
                    ui_ema = maj_ema if maj_ema is not None else 3
                    ui_delta = maj_delta if maj_delta is not None else 0.3
                    ui_cooldown = maj_cooldown if maj_cooldown is not None else 3
                    ui_dw = maj_dw if maj_dw is not None else 0.05
                elif strat == "Ensemble_Proportional":
                    ui_floor = prop_floor if prop_floor is not None else 0.2
                    ui_ema = prop_ema if prop_ema is not None else 3
                    ui_delta = prop_delta if prop_delta is not None else 0.2
                    ui_cooldown = prop_cooldown if prop_cooldown is not None else 3
                    ui_dw = prop_dw if prop_dw is not None else 0.03
                else:
                    # ?嗡? Ensemble 蝑??雿輻???身??
                    ui_floor = flat_params.get("floor", 0.2)
                    ui_ema = flat_params.get("ema_span", 3)
                    ui_delta = flat_params.get("delta_cap", 0.3)
                    ui_cooldown = flat_params.get("min_cooldown_days", 1)
                    ui_dw = flat_params.get("min_trade_dw", 0.01)

                ensemble_params = EnsembleParams(
                    floor=ui_floor,
                    ema_span=int(ui_ema),
                    delta_cap=ui_delta,
                    majority_k=flat_params.get("majority_k", 6),
                    min_cooldown_days=int(ui_cooldown),
                    min_trade_dw=ui_dw
                )
                logger.debug(f"[{strat}] UI???: floor={ui_floor}, ema_span={ui_ema}, delta_cap={ui_delta}, cooldown={ui_cooldown}, dw={ui_dw}")
                
                cost_params = CostParams(
                    buy_fee_bp=flat_params.get("buy_fee_bp", 4.27),
                    sell_fee_bp=flat_params.get("sell_fee_bp", 4.27),
                    sell_tax_bp=flat_params.get("sell_tax_bp", 30.0)
                )
                
                cfg = RunConfig(
                    ticker=ticker,
                    method=flat_params.get("method", "majority"),
                    strategies=list(file_map.keys()),  # ?? ?芸??交??曉?瑼??????亙?
                    file_map=file_map,  # ?? ?喳?頝臬?撠??銵?
                    params=ensemble_params,
                    cost=cost_params
                )
                
                # ?喲?瘥???瑼餃???
                if flat_params.get("majority_k_pct"):
                    cfg.majority_k_pct = flat_params.get("majority_k_pct")
                else:
                    cfg.majority_k_pct = 0.55
                    logger.info(f"[Ensemble] 撘瑕?閮剖? majority_k_pct=0.55")
                
                logger.info(f"[Ensemble] ?瑁???蔭: ticker={ticker}, method={flat_params.get('method')}, majority_k_pct={flat_params.get('majority_k_pct', 'N/A')}")
                
                # --- ?啣?嚗????ATR 閫貊?????券◢?芷?? ---
                valve_triggered = False
                ratio = None
                try:
                    atr_20 = calculate_atr(df_raw, 20)
                    atr_60 = calculate_atr(df_raw, 60)
                    
                    # 憓??閰喟敦??矽閰西?閮?
                    logger.debug(f"[{strat}] Ensemble ATR 閮??: atr_20={type(atr_20)}, atr_60={type(atr_60)}")
                    
                    if not atr_20.empty and not atr_60.empty:
                        atr_20_valid = atr_20.dropna()
                        atr_60_valid = atr_60.dropna()
                        
                        logger.debug(f"[{strat}] Ensemble ATR ????? atr_20={len(atr_20_valid)}, atr_60={len(atr_60_valid)}")
                        
                        if len(atr_20_valid) > 0 and len(atr_60_valid) > 0:
                            a20 = atr_20_valid.iloc[-1]
                            a60 = atr_60_valid.iloc[-1]
                            
                            logger.debug(f"[{strat}] Ensemble ATR ??啣? a20={a20:.6f}, a60={a60:.6f}")
                            
                            if a60 > 0:
                                ratio = float(a20 / a60)
                                valve_triggered = (ratio > atr_ratio)  # ??脤????銝?湛?雿輻? ">"
                                logger.debug(f"[{strat}] Ensemble ATR 瘥?? {ratio:.4f} (?瑼?{atr_ratio}) -> 閫貊?={valve_triggered}")
                                
                                # 憓??憸券??仿?閫貊???底蝝啗?閮?
                                if valve_triggered:
                                    logger.info(f"[{strat}] ?? 憸券??仿?閫貊?嚗?TR瘥??{ratio:.4f}) > ?瑼?{atr_ratio})")
                                else:
                                    logger.info(f"[{strat}] ?? 憸券??仿??芾孛?潘?ATR瘥??{ratio:.4f}) <= ?瑼?{atr_ratio})")
                            else:
                                logger.warning(f"[{strat}] Ensemble ATR(60) ?潛? 0嚗??瘜??蝞????)
                        else:
                            logger.warning(f"[{strat}] Ensemble ATR ?豢?銝?雲")
                    else:
                        logger.warning(f"[{strat}] Ensemble ATR 閮??蝯???箇征")
                        
                except Exception as e:
                    logger.warning(f"[{strat}] ?⊥?閮?? Ensemble ATR 瘥?? {e}")
                    logger.warning(f"[{strat}] ?航炊閰單?: {type(e).__name__}: {str(e)}")

                # 憒?????撘瑕?閫貊?嚗??撘瑕?閫貊?憸券??仿?
                if force_trigger:
                    valve_triggered = True
                    logger.info(f"[{strat}] ?? 撘瑕?閫貊?憸券??仿????")
                    if ratio is None:
                        ratio = 1.5  # 閮剖?銝???閮剖潛??潮＊蝷?

                # 雿輻??啁? ensemble_runner ?瑁?
                backtest_result = run_ensemble_backtest(cfg)

                # ?亙?撅??????銝??閫貊?璇?辣嚗???冽???????憟?? CAP
                if global_apply and valve_triggered:
                    from SSS_EnsembleTab import risk_valve_backtest
                    bench = df_raw  # 撌脣? open/high/low/close/volume
                    
                    logger.info(f"[{strat}] ?? ???憟??憸券??仿?: cap={risk_cap}, ratio={ratio:.4f}")
                    ratio_series = _compute_atr_ratio_series(df_raw).reindex(backtest_result.weight_curve.index)
                    mask_series = ratio_series > float(atr_ratio)
                    if force_trigger:
                        mask_series[:] = True
                    w_soft, _ = _apply_soft_risk_cap(
                        backtest_result.weight_curve.astype(float),
                        mask_series,
                        ratio_series,
                        atr_ratio,
                        risk_cap,
                    )
                    
                    rv = risk_valve_backtest(
                        open_px=backtest_result.price_series,
                        w=w_soft,
                        cost=cost_params,
                        benchmark_df=bench,
                        mode="cap",
                        cap_level=1.0,
                        slope20_thresh=0.0, slope60_thresh=0.0,
                        atr_win=20, atr_ref_win=60,
                        atr_ratio_mult=float(atr_ratio),   # ??UI ??ATR ?瑼?
                        use_slopes=True,                   # ??頝??撘瑕??????
                        slope_method="polyfit",            # ??頝??撘瑕??????
                        atr_cmp="gt"                       # ??頝??撘瑕?????湛???>嚗?
                    )
                    # app_dash.py / 2025-08-22 15:30
                    # ?典?憸券??仿?嚗?????摮?baseline ??valve ???嚗??閬?神璅????
                    # 1) 靽?? valve ????啣??券?
                    result['daily_state_valve'] = pack_df(rv["daily_state_valve"])
                    result['trade_ledger_valve'] = pack_df(rv["trade_ledger_valve"])
                    result['weight_curve_valve'] = pack_series(rv["weights_valve"])
                    result['equity_curve_valve'] = pack_series(rv["daily_state_valve"]["equity"])
                    
                    # 2) 靽?? baseline ????啣??券?嚗?????瘝??嚗?
                    if "daily_state_base" not in result and result.get("daily_state"):
                        result["daily_state_base"] = result["daily_state"]
                    if "trade_ledger_base" not in result and result.get("trade_ledger"):
                        result["trade_ledger_base"] = result["trade_ledger"]
                    if "weight_curve_base" not in result and result.get("weight_curve"):
                        result["weight_curve_base"] = result["weight_curve"]
                    
                    # 3) ?湔? backtest_result ?拐辣嚗???澆?蝥?????
                    backtest_result.daily_state = rv["daily_state_valve"]
                    backtest_result.ledger = rv["trade_ledger_valve"]
                    backtest_result.weight_curve = rv["weights_valve"]
                    backtest_result.equity_curve = rv["daily_state_valve"]["equity"]
                    logger.info(f"[{strat}] 憸券??仿?撌脣??剁?cap={risk_cap}, ratio={ratio:.4f}嚗?)
                    
                    # 憓??憸券??仿??????底蝝啗?閮?
                    if "metrics" in rv:
                        logger.info(f"[{strat}] 憸券??仿????: PF???={rv['metrics'].get('pf_orig', 'N/A'):.2f}, PF?仿?={rv['metrics'].get('pf_valve', 'N/A'):.2f}")
                        logger.info(f"[{strat}] 憸券??仿????: MDD???={rv['metrics'].get('mdd_orig', 'N/A'):.2f}%, MDD?仿?={rv['metrics'].get('mdd_valve', 'N/A'):.2f}%")
                    
                    # 蝯?UI ???閮????SSMA ???撠??嚗?
                    result['valve'] = {
                        "applied": True,
                        "cap": float(risk_cap),
                        "atr_ratio": ratio
                    }
                    
                    # ?啣?嚗???典??畾萇???歇憟????
                    result['_risk_valve_applied'] = True
                else:
                    if global_apply:
                        logger.info(f"[{strat}] ?? 憸券??仿??芾孛?潘?雿輻???????")
                        # 蝯?UI ???閮???芾孛?潘?
                        result['valve'] = {
                            "applied": False,
                            "cap": float(risk_cap),
                            "atr_ratio": ratio if ratio is not None else "N/A"
                        }
                    else:
                        logger.info(f"[{strat}] ???典?憸券??仿??芸???)
                        # 蝯?UI ???閮???芸??剁?
                        result['valve'] = {
                            "applied": False,
                            "cap": "N/A",
                            "atr_ratio": "N/A"
                        }
                
                # ?? 靽?? valve 鞈??嚗???????店嚗?
                valve_info = result.get('valve')
                valve_data = {
                    'daily_state_valve': result.get('daily_state_valve'),
                    'trade_ledger_valve': result.get('trade_ledger_valve'),
                    'weight_curve_valve': result.get('weight_curve_valve'),
                    'equity_curve_valve': result.get('equity_curve_valve'),
                    'daily_state_base': result.get('daily_state_base'),
                    'trade_ledger_base': result.get('trade_ledger_base'),
                    'weight_curve_base': result.get('weight_curve_base'),
                    '_risk_valve_applied': result.get('_risk_valve_applied')
                }

                # ?????? 鋆?撥嚗??蝞???渡蜀???璅??Ensemble ??????頛???伐???????
                metrics = backtest_result.stats.copy() if backtest_result.stats else {}
                equity = backtest_result.equity_curve

                if equity is not None and not equity.empty and len(equity) > 1:
                    # 1. ?箇???????
                    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
                    days = (equity.index[-1] - equity.index[0]).days
                    years = max(days / 365.25, 0.1)
                    ann_ret = (1 + total_ret) ** (1 / years) - 1

                    # 2. 憸券????
                    roll_max = equity.cummax()
                    dd = (equity / roll_max) - 1
                    mdd = dd.min()

                    # 3. 憭??瘥??
                    daily_ret = equity.pct_change().dropna()
                    if len(daily_ret) > 0 and daily_ret.std() != 0:
                        sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5)
                    else:
                        sharpe = 0

                    # 4. ?∠?瘥??
                    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

                    # 5. 蝝Ｘ?隢暹???
                    downside = daily_ret[daily_ret < 0]
                    if len(downside) > 0 and downside.std() != 0:
                        sortino = daily_ret.mean() / downside.std() * (252**0.5)
                    else:
                        sortino = 0

                    # 6. 瘜Ｗ???
                    if len(daily_ret) > 0:
                        annualized_volatility = daily_ret.std() * (252**0.5)
                    else:
                        annualized_volatility = 0

                    # 7. ?湔? metrics嚗??銝?撩撠?? key嚗?圾瘙?UI NoneType ?航炊嚗?
                    metrics.update({
                        'total_return': total_ret,
                        'annual_return': ann_ret,
                        'max_drawdown': mdd,
                        'calmar_ratio': calmar,
                        'sharpe_ratio': sharpe,
                        'sortino_ratio': sortino,
                        'annualized_volatility': annualized_volatility,
                        'num_trades': metrics.get('num_trades', len(backtest_result.ledger) if backtest_result.ledger is not None else 0)
                    })

                    logger.info(f"[Ensemble] ?? 鋆?撥敺??璅? 撟游????={ann_ret:.2%}, MDD={mdd:.2%}, Sharpe={sharpe:.2f}, Calmar={calmar:.2f}")
                else:
                    logger.warning(f"[Ensemble] ??? 甈???脩?銝?雲嚗??瘜??蝞???湔?璅?)

                # 頧???箄??澆?隞乩????摰寞?
                result = {
                    'trades': [],
                    'trade_df': pack_df(backtest_result.trades),
                    'trades_df': pack_df(backtest_result.trades),
                    'signals_df': pack_df(backtest_result.trades[['trade_date', 'type', 'price']].rename(columns={'type': 'action'}) if not backtest_result.trades.empty else pd.DataFrame(columns=['trade_date', 'action', 'price'])),
                    'metrics': metrics,  # ?? 雿輻?鋆?撥敺?? metrics
                    'equity_curve': pack_series(backtest_result.equity_curve),
                    'cash_curve': pack_series(backtest_result.cash_curve) if backtest_result.cash_curve is not None else "",
                    'weight_curve': pack_series(backtest_result.weight_curve) if backtest_result.weight_curve is not None else pack_series(pd.Series(0.0, index=backtest_result.equity_curve.index)),
                    'price_series': pack_series(backtest_result.price_series) if backtest_result.price_series is not None else pack_series(pd.Series(1.0, index=backtest_result.equity_curve.index)),
                    'daily_state': pack_df(backtest_result.daily_state),
                    'trade_ledger': pack_df(backtest_result.ledger),
                    'daily_state_std': pack_df(backtest_result.daily_state),
                    'trade_ledger_std': pack_df(backtest_result.ledger)
                }

                # ?? ?Ｗ儔 valve 鞈??
                if valve_info:
                    result['valve'] = valve_info
                for k, v in valve_data.items():
                    if v is not None:
                        result[k] = v

                logger.info(f"[Ensemble] ?瑁????: 甈???脩??瑕漲={len(backtest_result.equity_curve)}, 鈭斗???{len(backtest_result.ledger) if backtest_result.ledger is not None and not backtest_result.ledger.empty else 0}")

                # ?? 撽??鞈??摰????
                if backtest_result.daily_state is not None and not backtest_result.daily_state.empty:
                    logger.debug(f"[Ensemble] daily_state 甈??: {list(backtest_result.daily_state.columns)}, ????\n{backtest_result.daily_state.head(3)}")
                else:
                    logger.warning(f"[Ensemble] ??? daily_state ?舐征??? None")
                
            except Exception as e:
                logger.error(f"Ensemble 蝑???瑁?憭望?: {e}")
                # ?萄遣蝛箇?蝯??
                result = {
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'trades_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0},
                    'equity_curve': pd.Series(1.0, index=df_raw.index)
                }
            
            # === 靽桀儔 3嚗?溶??矽閰行?隤???詨?摮???仿?????虫???===
            logger.info(f"[Ensemble] ?瑁?摰??嚗?icker={ticker}, method={flat_params.get('method')}")
            if 'equity_curve' in result and hasattr(result['equity_curve'], 'shape'):
                logger.debug(f"[Ensemble] 甈???脩??瑕漲: {len(result['equity_curve'])}")
            if 'trade_df' in result and hasattr(result['trade_df'], 'shape'):
                logger.debug(f"[Ensemble] 鈭斗?閮???賊?: {len(result['trade_df'])}")
        else:
            if strat_type == 'single':
                df_ind = compute_single(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["devwin"], smaa_source=smaa_src)
            elif strat_type == 'dual':
                df_ind = compute_dual(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["short_win"], strat_params["long_win"], smaa_source=smaa_src)
            elif strat_type == 'RMA':
                df_ind = compute_RMA(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["rma_len"], strat_params["dev_len"], smaa_source=smaa_src)
            if df_ind.empty:
                continue
            result = backtest_unified(
                df_ind,
                strat_type,
                strat_params,
                discount=discount,
                trade_cooldown_bars=cooldown,
                bad_holding=bad_holding,
            )
            
            # ?箏?隞???仿???溶??valve 璅??
            if global_apply:
                result['valve'] = {
                    "applied": False,  # ?嗡?蝑??憿???急?銝???湧◢?芷??
                    "cap": float(risk_cap),
                    "atr_ratio": "N/A"
                }
            else:
                result['valve'] = {
                    "applied": False,
                    "cap": "N/A",
                    "atr_ratio": "N/A"
                }
        # 蝯曹?雿輻? orient="split" ???嚗?????銴?????
        # 瘜冽?嚗?nsemble 蝑??撌脩???pack_df/pack_series 銝剛????嚗??ㄐ?芾????蝑??
        if strat_type != 'ensemble':
            if hasattr(result.get('trade_df'), 'to_json'):
                result['trade_df'] = result['trade_df'].to_json(date_format='iso', orient='split')
            if 'signals_df' in result and hasattr(result['signals_df'], 'to_json'):
                result['signals_df'] = result['signals_df'].to_json(date_format='iso', orient='split')
            if 'trades_df' in result and hasattr(result['trades_df'], 'to_json'):
                result['trades_df'] = result['trades_df'].to_json(date_format='iso', orient='split')
            if 'equity_curve' in result and hasattr(result['equity_curve'], 'to_json'):
                result['equity_curve'] = result['equity_curve'].to_json(date_format='iso', orient='split')
        if 'trades' in result and isinstance(result['trades'], list):
            result['trades'] = [
                (str(t[0]), t[1], str(t[2])) if isinstance(t, tuple) and len(t) == 3 else t
                for t in result['trades']
            ]
        
        # << ?啣?嚗??敺???敺???芣????鋆?? daily_state / weight_curve 蝑?>>
        result = _pack_result_for_store(result)
        
        # === ?典?憸券??仿?嚗??????憟??嚗??憓?撥???銝?湛? ===
        if global_apply:
            # ?啣?嚗??蝑?????撌脩?憟??嚗?停銝?????銝甈?
            if result.get('_risk_valve_applied'):
                logger.info(f"[{strat}] 撌脩?蝑?????憟??憸券??仿?嚗?歲???撅??活憟??")
            else:
                # ????憛????ㄐ???
                # 1) ??ds嚗?aily_state嚗??銝西圾??
                ds_raw = result.get("daily_state_std") or result.get("daily_state")
                ds = df_from_pack(ds_raw)
                if ds is None or ds.empty or "w" not in ds.columns:
                    logger.warning(f"[{strat}] daily_state 銝?? 'w'嚗?歲???撅憸券??仿?")
                else:
                    # 2) 雿輻???脤????銝?渡?憸券??仿??斗???摩
                    ratio_series = _compute_atr_ratio_series(df_raw).reindex(ds.index)
                    try:
                        from SSS_EnsembleTab import compute_risk_valve_signals
                        
                        # 撱箇??箸?鞈??嚗??擃???孵停撣嗡?嚗?
                        bench = _build_benchmark_df(df_raw)
                        
                        # 雿輻??脤???????閮剖??賂?????瑼?0嚗?TR瘥??1.5嚗??頛?泵??">"
                        risk_signals = compute_risk_valve_signals(
                            benchmark_df=bench,
                            slope20_thresh=0.0,      # 20?交????瑼?
                            slope60_thresh=0.0,      # 60?交????瑼?
                            atr_win=20,              # ATR閮??蝒??
                            atr_ref_win=60,          # ATR??????
                            atr_ratio_mult=float(atr_ratio),  # ATR瘥?潮?瑼?
                            use_slopes=True,         # ??????璇?辣
                            slope_method="polyfit",   # 雿輻?憭??撘???????
                            atr_cmp="gt"             # 雿輻? ">" 瘥??蝚西?
                        )
                        
                        mask = risk_signals["risk_trigger"].reindex(ds.index).fillna(False)
                        ratio_series = _compute_atr_ratio_series(df_raw).reindex(ds.index)
                        logger.info(f"[{strat}] ?脤????憸券??仿?嚗?????隞嗅??剁?ATR瘥?潮?瑼?{atr_ratio}")
                        
                    except Exception as e:
                        logger.warning(f"[{strat}] ?⊥?雿輻??脤????憸券??仿?嚗?????ATR-only: {e}")
                        # ???啣??祉? ATR-only ??摩
                        ratio_series = _compute_atr_ratio_series(df_raw).reindex(ds.index)
                        if ratio_series is None or ratio_series.empty:
                            logger.warning(f"[{strat}] ATR20/60 unavailable; skip valve recompute and keep baseline weights.")
                            mask = pd.Series(False, index=ds.index)
                        else:
                            ratio = ratio_series
                            mask = (ratio > float(atr_ratio))  # ??倦????????皝??輯撒??">" ?伍??
                    
                    if force_trigger:
                        mask[:] = True  # 撘瑕??券??亙?憟?CAP

                    # ?典?撅憯?w 銋?????靽???芸??仿???baseline
                    if "daily_state_base" not in result and ds_raw is not None:
                        result["daily_state_base"] = ds_raw  # 靽???芸??仿???baseline
                    
                    # ??餈賢?隞乩??抵?嚗???典?銝畾萸??撖?w 銋??嚗?
                    if "trade_ledger_base" not in result and result.get("trade_ledger") is not None:
                        result["trade_ledger_base"] = result["trade_ledger"]
                    if "weight_curve_base" not in result and result.get("weight_curve") is not None:
                        result["weight_curve_base"] = result["weight_curve"]
                    
                    # 3) 撠????ds.index嚗???憯?w ??CAP
                    mask_aligned = mask.reindex(ds.index).fillna(False)
                    w_soft, dyn_cap = _apply_soft_risk_cap(
                        ds["w"].astype(float),
                        mask_aligned,
                        ratio_series,
                        atr_ratio,
                        risk_cap,
                    )
                    ds["w"] = w_soft.to_numpy()

                    # 4) ??神 ds嚗?蒂???鈭斗?/甈??
                    result["daily_state_std"] = pack_df(ds)

                    # open ?對?瘝?? open 撠梢????嗆活?冽??文?嚗?
                    open_px = (df_raw["open"] if "open" in df_raw.columns else df_raw.get("?嗥???)).astype(float)
                    open_px = open_px.reindex(ds.index).dropna()

                    # ?乩?瘝輻??暹???risk_valve_backtest嚗?策 cap_level=1.0 銵函內?? 撌脩??舐?璅?????
                    try:
                        from SSS_EnsembleTab import (
                            risk_valve_backtest,
                            CostParams,
                            _mdd_from_daily_equity,
                            _sell_returns_pct_from_ledger,
                        )
                        
                        # ??????
                        trade_cost = (strat_params.get("trade_cost", {}) 
                                      if isinstance(strat_params, dict) else {})
                        cost = CostParams(
                            buy_fee_bp=float(trade_cost.get("buy_fee_bp", 4.27)),
                            sell_fee_bp=float(trade_cost.get("sell_fee_bp", 4.27)),
                            sell_tax_bp=float(trade_cost.get("sell_tax_bp", 30.0)),
                        )
                        
                        # ?箸?嚗??擃???孵停撣嗡?嚗?
                        bench = _build_benchmark_df(df_raw)
                        
                        # === 憸券??仿???葫嚗?Ⅱ靽???訾??湔?(2025/08/20) ===
                        valve_params = {
                            "open_px": open_px,
                            "w": ds["w"].astype(float).reindex(open_px.index).fillna(0.0),
                            "cost": cost,
                            "benchmark_df": bench,
                            "mode": "cap",
                            "cap_level": 1.0,
                            "slope20_thresh": 0.0,         # ?? ??脤????銝?湛?20?交????瑼?
                            "slope60_thresh": 0.0,         # ?? ??脤????銝?湛?60?交????瑼?
                            "atr_win": 20, 
                            "atr_ref_win": 60,
                            "atr_ratio_mult": float(atr_ratio),   # ?? ???撅銝??
                            "use_slopes": True,            # ?? ??脤????銝?湛???????璇?辣
                            "slope_method": "polyfit",     # ?? ??脤????銝?湛?雿輻?憭??撘????
                            "atr_cmp": "gt"               # ?? ??脤????銝?湛?雿輻? ">" 瘥??蝚西?
                        }
                        
                        # 閮??憸券??仿???蔭?冽?閮箸?
                        logger.info(f"[{strat}] 憸券??仿???蔭: cap_level={valve_params['cap_level']}, atr_ratio_mult={valve_params['atr_ratio_mult']}")
                        
                        result_cap = risk_valve_backtest(**valve_params)
                    except Exception as e:
                        logger.warning(f"[{strat}] ?⊥?撠?? risk_valve_backtest: {e}")
                        result_cap = None

                    if result_cap:
                        # === 摰??閬?神嚗??????萎蒂鋆???圈? ===
                        logger.debug(f"[UI_CHECK] ?喳?閬?神嚗?ew_trades={len(result_cap.get('trade_ledger_valve', pd.DataFrame()))} rows, new_ds={len(result_cap.get('daily_state_valve', pd.DataFrame()))} rows")
                        
                        # app_dash.py / 2025-08-22 15:30
                        # ?典?憸券??仿?嚗?????摮?baseline ??valve ???嚗??閬?神璅????
                        # 1) 靽?? valve ????啣??券?
                        if 'trade_ledger_valve' in result_cap:
                            result['trade_ledger_valve'] = pack_df(result_cap['trade_ledger_valve'])
                        
                        if 'daily_state_valve' in result_cap:
                            result['daily_state_valve'] = pack_df(result_cap['daily_state_valve'])
                        
                        if 'weights_valve' in result_cap:
                            result['weight_curve_valve'] = pack_series(result_cap['weights_valve'])
                        
                        # 甈???脩?嚗????Series
                        if 'daily_state_valve' in result_cap and 'equity' in result_cap['daily_state_valve']:
                            try:
                                result['equity_curve_valve'] = pack_series(result_cap['daily_state_valve']['equity'])
                            except Exception:
                                # ?乩?摮????DataFrame
                                result['equity_curve_valve'] = pack_df(result_cap['daily_state_valve']['equity'].to_frame('equity'))
                        
                        # 2) 靽?? baseline ????啣??券?嚗?????瘝??嚗?
                        if "daily_state_base" not in result and result.get("daily_state") is not None:
                            result["daily_state_base"] = result["daily_state"]
                        if "trade_ledger_base" not in result and result.get("trade_ledger") is not None:
                            result["trade_ledger_base"] = result["trade_ledger"]
                        if "weight_curve_base" not in result and result.get("weight_curve") is not None:
                            result["weight_curve_base"] = result["weight_curve"]
                        
                        # 3) 皜???航????瘛瑟????敹怠?
                        for k in ['trades_ui', 'trade_df', 'trade_ledger_std', 'metrics']:
                            if k in result:
                                result.pop(k, None)
                        

                        
                        # ?啣?嚗??閮?valve ????敺??敹怠??斗?
                        result['valve'] = {
                            'applied': True,
                            'cap': float(risk_cap),
                            'atr_ratio_mult': float(atr_ratio),
                        }
                        
                        # ?啣?嚗????ensemble ???嚗???臬?敺??
                        # ?典?撅憸券??仿??憛?葉嚗???????cfg ?拐辣嚗???乩蝙?券?閮剖?
                        result["ensemble_params"] = {"majority_k_pct": 0.55}  # ??身??

                        # 2025-08-20 ??????隞乩???蜀???閮?#app_dash.py
                        ledger_valve = result_cap.get('trade_ledger_valve', pd.DataFrame())
                        ds_valve = result_cap.get('daily_state_valve', pd.DataFrame())
                        if not ledger_valve.empty and not ds_valve.empty and 'equity' in ds_valve:
                            r = _sell_returns_pct_from_ledger(ledger_valve)
                            eq = ds_valve['equity']
                            total_ret = eq.iloc[-1] / eq.iloc[0] - 1
                            years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1)
                            ann_ret = (1 + total_ret) ** (1 / years) - 1
                            mdd = _mdd_from_daily_equity(eq)
                            dd = eq / eq.cummax() - 1
                            blocks = (~(dd < 0)).cumsum()
                            dd_dur = int((dd.groupby(blocks).cumcount() + 1).where(dd < 0).max() or 0)
                            num_trades = len(r)
                            win_rate = (r > 0).sum() / num_trades if num_trades > 0 else 0
                            avg_win = r[r > 0].mean() if win_rate > 0 else np.nan
                            avg_loss = r[r < 0].mean() if win_rate < 1 else np.nan
                            payoff = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_win) else np.nan
                            daily_r = eq.pct_change().dropna()
                            sharpe = (daily_r.mean() * np.sqrt(252)) / daily_r.std() if daily_r.std() != 0 else np.nan
                            downside = daily_r[daily_r < 0]
                            sortino = (daily_r.mean() * np.sqrt(252)) / downside.std() if downside.std() != 0 else np.nan
                            ann_vol = daily_r.std() * np.sqrt(252) if len(daily_r) > 0 else np.nan
                            prof = r[r > 0].sum()
                            loss = abs(r[r < 0].sum())
                            pf = prof / loss if loss != 0 else np.nan
                            win_flag = r > 0
                            grp = (win_flag != win_flag.shift()).cumsum()
                            consec = win_flag.groupby(grp).cumcount() + 1
                            max_wins = int(consec[win_flag].max() if True in win_flag.values else 0)
                            max_losses = int(consec[~win_flag].max() if False in win_flag.values else 0)
                            result['metrics'] = {
                                'total_return': float(total_ret),
                                'annual_return': float(ann_ret),
                                'max_drawdown': float(mdd),
                                'max_drawdown_duration': dd_dur,
                                'calmar_ratio': float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
                                'num_trades': int(num_trades),
                                'win_rate': float(win_rate),
                                'avg_win': float(avg_win) if not np.isnan(avg_win) else np.nan,
                                'avg_loss': float(avg_loss) if not np.isnan(avg_loss) else np.nan,
                                'payoff_ratio': float(payoff) if not np.isnan(payoff) else np.nan,
                                'sharpe_ratio': float(sharpe) if not np.isnan(sharpe) else np.nan,
                                'sortino_ratio': float(sortino) if not np.isnan(sortino) else np.nan,
                                'max_consecutive_wins': max_wins,
                                'max_consecutive_losses': max_losses,
                                'annualized_volatility': float(ann_vol) if not np.isnan(ann_vol) else np.nan,
                                'profit_factor': float(pf) if not np.isnan(pf) else np.nan,
                            }
                        
                        # 3) 蝯?UI 銝???璅?????嚗?噶?潮＊蝷箝?歇憟????
                        result['_risk_valve_applied'] = True
                        atr20_last = None
                        atr60_last = None
                        try:
                            _atr20 = calculate_atr(df_raw, 20)
                            _atr60 = calculate_atr(df_raw, 60)
                            if _atr20 is not None and hasattr(_atr20, "dropna"):
                                _atr20_valid = _atr20.dropna()
                                if len(_atr20_valid) > 0:
                                    atr20_last = float(_atr20_valid.iloc[-1])
                            if _atr60 is not None and hasattr(_atr60, "dropna"):
                                _atr60_valid = _atr60.dropna()
                                if len(_atr60_valid) > 0:
                                    atr60_last = float(_atr60_valid.iloc[-1])
                        except Exception:
                            pass

                        result['_risk_valve_params'] = {
                            'cap': float(risk_cap),
                            'atr_ratio': float(atr_ratio),
                            'atr20_last': atr20_last,
                            'atr60_last': atr60_last,
                        }
                        
                        true_days = int(mask_aligned.sum())
                        logger.debug(f"[{strat}] ?典?憸券??仿?撌脣??剁????嚗??憸券?憭拇?={true_days}, CAP={risk_cap:.2f}")
                    else:
                        logger.warning(f"[{strat}] 憸券??仿????瘝??餈??蝯??")

        
        results[strat] = result
    
    # 雿輻?蝚砌?????亦??豢?雿??銝餉?憿舐內?豢?
    first_strat = list(results.keys())[0] if results else active_strategy_names[0]
    first_smaa_src = param_presets[first_strat].get("smaa_source", "Self")
    df_raw_main, _ = load_data(ticker, start_date, end_date if end_date else None, smaa_source=first_smaa_src)
    
    # 蝯曹?雿輻? orient="split" 摨?????蝣箔?銝?湔?
    payload = {
        'results': results, 
        'df_raw': df_raw_main.to_json(date_format='iso', orient='split'), 
        'ticker': ticker
    }
    
    # ?脣??扳炎?伐?憒?????蝬脩????????拐辣撠梯???????
    try:
        json.dumps(payload)
    except Exception as e:
        logger.exception("[BUG] backtest-store payload 隞??銝??摨?????隞塚?%s", e)
        # 憒??閬?撥?嗡??湛??臬? fallback嚗?son.dumps(..., default=str) 雿??虜銝?遣霅啣???
    
    # === ??葫摰???亥? ===
    logger.debug(f"??葫摰?? - 蝑???? {len(results)}, ticker: {ticker}, ?豢?銵??: {len(df_raw_main)}")
    logger.debug(f"蝑????”: {list(results.keys())}")
    
    return payload

# --------- 銝駁?蝐文?摰寥＊蝷?---------
def update_tab(data, tab, theme, smart_leverage_on):
    # 蝣箔? pandas ?舐?
    import pandas as pd

    # === 隤輯岫?亥?嚗????DEBUG 蝝????＊蝷綽?===
    logger.debug(f"update_tab 鋡怨矽??- tab: {tab}")
    
    if not data:
        logger.warning("??葫鞈??銝???剁?隢???瑁???葫")
        return html.Div("隢???瑁???葫")

    # Defensive: avoid blank UI when backtest-store is malformed or error-only payload.
    if not isinstance(data, dict):
        logger.warning(f"update_tab ?嗅???dict 鞈??: {type(data)}")
        return html.Div("??葫鞈???澆??航炊嚗??????瑁???葫??, style={"color": "#dc3545"})

    if data.get("error"):
        logger.warning(f"backtest-store ?航炊: {data.get('error')}")
        return html.Div(
            [
                html.H4("??葫憭望?", style={"color": "#dc3545", "marginBottom": "10px"}),
                html.Div(str(data.get("error")), style={"color": "#dc3545"}),
            ],
            style={"padding": "16px"},
        )

    # data ?航???dict嚗?歇??????嚗???湔?????喳?
    results = data.get("results")
    if not isinstance(results, dict) or not results:
        logger.warning(f"backtest-store 蝻箏? results ???蝛箝?eys={list(data.keys())}")
        return html.Div("??葫蝯???箇征嚗??蝣箄?蝑???臬??刻◤?梯?嚗??????瑁???葫??, style={"color": "#dc3545"})

    df_raw = df_from_pack(data.get("df_raw"))
    ticker = data.get("ticker", "N/A")
    strategy_names = list(results.keys())
    
    logger.debug(f"?豢?閫??摰?? - 蝑???? {len(strategy_names)}, ticker: {ticker}, ?豢?銵??: {len(df_raw) if df_raw is not None else 0}")
    # === ?寞?銝駁?瘙箏?憿??霈?? ===
    if theme == 'theme-light':
        # 瘛箄?璅∪????
        plotly_template = 'plotly_white'
        bg_color = '#ffffff'
        font_color = '#212529'

        # ?∠???”?潮???
        card_bg = '#f8f9fa'       # 瘛箇??∠?
        card_border = '#dee2e6'
        card_text = '#212529'

        table_header_bg = '#e9ecef'
        table_cell_bg = '#ffffff'
        table_text = '#212529'
        table_border = '#dee2e6'

        legend_bgcolor = 'rgba(255,255,255,0.8)'
        legend_bordercolor = '#444'
        legend_font_color = '#333'

    elif theme == 'theme-blue':
        # ???璅∪????
        plotly_template = 'plotly_dark'
        bg_color = '#001a33'
        font_color = '#ffe066'

        # ?∠???”?潮???
        card_bg = '#002b4d'       # 瘛梯??∠?
        card_border = '#004080'
        card_text = '#ffe066'

        table_header_bg = '#003366'
        table_cell_bg = '#001a33'
        table_text = '#ffe066'
        table_border = '#004080'

        legend_bgcolor = 'rgba(0, 26, 51, 0.8)'
        legend_bordercolor = '#ffe066'
        legend_font_color = '#ffe066'

    else:  # theme-dark (??身)
        plotly_template = 'plotly_dark'
        bg_color = '#121212'
        font_color = '#e0e0e0'

        # ?∠???”?潮???
        card_bg = '#1e1e1e'       # 瘛梁??∠?
        card_border = '#333'
        card_text = '#fff'

        table_header_bg = '#2d2d2d'
        table_cell_bg = '#1e1e1e'
        table_text = '#e0e0e0'
        table_border = '#444'

        legend_bgcolor = 'rgba(30,30,30,0.8)'
        legend_bordercolor = '#fff'
        legend_font_color = '#fff'
    
    if tab == "backtest":
        # ?萄遣蝑????葫?????惜
        strategy_tabs = []
        for strategy in strategy_names:
            result = results.get(strategy)
            if not result:
                continue

            # ?? 瑼Ｘ??臬????隤方??荔?靘?? Ensemble ?曆??唳?獢??
            if 'error' in result:
                error_msg = result.get('error', '?芰??航炊')
                error_detail = result.get('error_detail', '')
                strategy_tabs.append(
                    dcc.Tab(
                        label=f"??{strategy}",
                        value=strategy,
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        children=[
                            html.Div([
                                dbc.Alert([
                                    html.H4("??? ??葫?瑁?憭望?", className="alert-heading"),
                                    html.P(error_msg, style={"font-weight": "bold", "font-size": "16px"}),
                                    html.Hr(),
                                    html.P(error_detail, className="mb-0"),
                                    html.Br(),
                                    html.P([
                                        "?? ?航????嚗?,
                                        html.Ul([
                                            html.Li("蝑??瑼??撌脰◤撠????archive/ ?桅?"),
                                            html.Li("??澈?????????????寥?"),
                                            html.Li("蝑??撠???瑁???葫嚗?撩撠??閬?? CSV 瑼??"),
                                        ])
                                    ]),
                                    html.P([
                                        "?? 撱箄降???嚗?,
                                        html.Ul([
                                            html.Li("瑼Ｘ? archive/ ?桅?銝剜??行??賊???遢"),
                                            html.Li("????啣?隞??澈???閰西岫"),
                                            html.Li("????啣?銵?run_full_pipeline ?Ｙ??啁?蝑??瑼??"),
                                        ])
                                    ])
                                ], color="warning", style={"margin": "20px"})
                            ], style={"padding": "20px"})
                        ]
                    )
                )
                continue  # 頝喲?甇?虜??葡???蝔?

            # === 蝯曹??亙?嚗????漱??”???????????蝺?===
            # 霈鈭斗?銵函?蝯曹??亙?嚗???冽?皞??嚗?? fallback
            trade_df = None
            candidates = [
                result.get('trades'),      # ?典?閬?神敺??皞??
                result.get('trades_ui'),   # ???撘???仿?摮??嚗?
                result.get('trade_df'),    # ???蝑???芸葆
            ]
            
            for cand in candidates:
                if cand is None:
                    continue
                # cand ?航?撌脫? DataFrame ??????銝?
                df = df_from_pack(cand) if isinstance(cand, str) else cand
                if df is not None and getattr(df, 'empty', True) is False:
                    trade_df = df.copy()
                    break
            
            if trade_df is None:
                # 撱箇?蝛箄”?踹?敺??撏?
                trade_df = pd.DataFrame(columns=['trade_date','type','price','shares','return'])
            
            # app_dash.py / 2025-08-22 16:00
            # ??? daily_state嚗????蝙?典??亦??穿??嗆活???嚗??敺?baseline嚗?? O2 銝?湛?
            # ?? 閮箸?嚗?＊蝷箏??函? daily_state 靘??嚗??????冽改?
            available_sources = [k for k in ['daily_state_valve', 'daily_state_std', 'daily_state', 'daily_state_base'] if result.get(k) is not None]
            logger.debug(f"[{strategy}] ?舐???daily_state 靘??: {available_sources}")

            # ??洵銝????蝛箝??????踹?鋡怎征??daily_state_valve ?餅? fallback
            daily_state_std = _first_non_empty_result_df(
                result,
                ['daily_state_valve', 'daily_state_std', 'daily_state', 'daily_state_base']
            )
            if not daily_state_std.empty:
                logger.debug(f"[{strategy}] 雿輻???征 daily_state嚗?圾???敶Ｙ?: {daily_state_std.shape}")
            else:
                logger.warning(f"[{strategy}] ??? 瘝???曉?隞颱???征 daily_state嚗?蝙?函征 DataFrame")
            
            # app_dash.py / 2025-08-22 16:00
            # ??? trade_ledger嚗????蝙?典??亦??穿??嗆活???嚗??敺?baseline嚗?? O2 銝?湛?
            trade_ledger_std = None
            if result.get('trade_ledger_valve'):
                trade_ledger_std = df_from_pack(result['trade_ledger_valve'])
            elif result.get('trade_ledger_std'):
                trade_ledger_std = df_from_pack(result['trade_ledger_std'])
            elif result.get('trade_ledger'):
                trade_ledger_std = df_from_pack(result['trade_ledger'])
            elif result.get('trade_ledger_base'):
                trade_ledger_std = df_from_pack(result['trade_ledger_base'])
            else:
                trade_ledger_std = pd.DataFrame()
            
            # 閮??靘???豢?蝯??
            logger.debug(f"[UI] {strategy} trades 靘???芸?摨??trades -> trades_ui -> trade_df嚗?祕??蝙??{'trades' if 'trades' in result else ('trades_ui' if 'trades_ui' in result else 'trade_df')}")
            logger.debug(f"[UI] {strategy} 霈?????3 ??w: {daily_state_std['w'].head(3).tolist() if 'w' in daily_state_std.columns else 'N/A'}")
            
            # 璅????漱??????蝣箔???絞銝??trade_date/type/price 甈??
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
                logger.debug(f"璅????? trades_ui 甈??: {list(trade_df.columns)}")
            except Exception as e:
                logger.warning(f"?⊥?雿輻? sss_core 璅?????雿輻?敺???寞?: {e}")
                # 敺??璅?????獢?
                if trade_df is not None and len(trade_df) > 0:
                    trade_df = trade_df.copy()
                    trade_df.columns = [str(c).lower() for c in trade_df.columns]
                    
                    # 蝣箔???trade_date 甈?
                    if "trade_date" not in trade_df.columns:
                        if "date" in trade_df.columns:
                            trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
                        elif isinstance(trade_df.index, pd.DatetimeIndex):
                            trade_df = trade_df.reset_index().rename(columns={"index": "trade_date"})
                        else:
                            trade_df["trade_date"] = pd.NaT
                    else:
                        trade_df["trade_date"] = pd.to_datetime(trade_df["trade_date"], errors="coerce")
                    
                    # 蝣箔???type 甈?
                    if "type" not in trade_df.columns:
                        if "action" in trade_df.columns:
                            trade_df["type"] = trade_df["action"].astype(str).str.lower()
                        elif "side" in trade_df.columns:
                            trade_df["type"] = trade_df["side"].astype(str).str.lower()
                        else:
                            trade_df["type"] = "hold"
                    
                    # 蝣箔???price 甈?
                    if "price" not in trade_df.columns:
                        for c in ["open", "price_open", "exec_price", "px", "close"]:
                            if c in trade_df.columns:
                                trade_df["price"] = trade_df[c]
                                break
                        if "price" not in trade_df.columns:
                            trade_df["price"] = 0.0
            
            # ???撠??嚗??霅?trade_date ??Timestamp嚗?rice/shares ??float
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            if 'signal_date' in trade_df.columns:
                trade_df['signal_date'] = pd.to_datetime(trade_df['signal_date'])
            if 'price' in trade_df.columns:
                trade_df['price'] = pd.to_numeric(trade_df['price'], errors='coerce')
            if 'shares' in trade_df.columns:
                trade_df['shares'] = pd.to_numeric(trade_df['shares'], errors='coerce')
            
            # === ?堆??交? trade_ledger嚗????＊蝷箸?摰?????雿?===
            ledger_df = df_from_pack(result.get('trade_ledger'))
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                ledger_ui = norm(ledger_df) if ledger_df is not None and len(ledger_df)>0 else pd.DataFrame()
            except Exception:
                ledger_ui = ledger_df if ledger_df is not None else pd.DataFrame()
            
            # === 靽格迤嚗????蝙?冽?皞??敺?? trade_ledger_std ===
            # 雿輻? utils_payload 璅???????????蝣箔?甈??朣??
            if trade_ledger_std is not None and not trade_ledger_std.empty:
                base_df = trade_ledger_std
                if 'trade_date' in base_df.columns:
                    base_df['trade_date'] = pd.to_datetime(base_df['trade_date'], errors='coerce')
                if 'signal_date' in base_df.columns:
                    base_df['signal_date'] = pd.to_datetime(base_df['signal_date'], errors='coerce')
            elif ledger_ui is not None and not ledger_ui.empty:
                base_df = ledger_ui
                if 'trade_date' in base_df.columns:
                    base_df['trade_date'] = pd.to_datetime(base_df['trade_date'], errors='coerce')
                if 'signal_date' in base_df.columns:
                    base_df['signal_date'] = pd.to_datetime(base_df['signal_date'], errors='coerce')
            else:
                base_df = trade_df
            
            # 閮?????甈??嚗???舐?嚗?
            logger.debug("[UI] trade_df ???甈??嚗?s", list(base_df.columns) if base_df is not None else None)
            logger.debug("[UI] trade_ledger_std ???甈??嚗?s", list(trade_ledger_std.columns) if trade_ledger_std is not None else None)

            # ?箔? 100% 靽?? weight_change ?箇?嚗??蝣箔?甈??甈??
            base_df = _ensure_weight_columns(base_df)
            # 雿輻??啁?蝯曹??澆????撘?
            display_df = format_trade_like_df_for_display(base_df)

            # === 鈭斗?瘚?偌撣?ledger)銵冽?嚗??皞??憿舐內??===
            ledger_src = trade_ledger_std if (trade_ledger_std is not None and not trade_ledger_std.empty) else \
                         (ledger_ui if (ledger_ui is not None and not ledger_ui.empty) else pd.DataFrame())

            if ledger_src is not None and not ledger_src.empty:
                # ?箔? 100% 靽?? weight_change ?箇?嚗??蝣箔?甈??甈??
                ledger_src = _ensure_weight_columns(ledger_src)
                # 雿輻??啁?蝯曹??澆????撘?
                ledger_display = format_trade_like_df_for_display(ledger_src)
                ledger_columns = [{"name": i, "id": i} for i in ledger_display.columns]
                ledger_data = ledger_display.to_dict('records')
            else:
                ledger_columns = []
                ledger_data = []

            # ==============================================================================
            # ?? ???靽格迤嚗????? Smart Leverage 銝行???Metrics嚗????? KPI ?∠?
            # ==============================================================================

            # ??捱摰?蝙?典???daily_state
            daily_state = _first_non_empty_result_df(
                result,
                ['daily_state_valve', 'daily_state_std', 'daily_state', 'daily_state_base']
            )

            # ?芸?雿輻?璅??????????
            if daily_state_std is not None and not daily_state_std.empty:
                ds_for_calc = daily_state_std
                logger.debug(f"[UI] 雿輻?璅???????daily_state_std ?脰? Smart Leverage 閮??")
            else:
                ds_for_calc = daily_state
                logger.debug(f"[UI] 雿輻???? daily_state ?脰? Smart Leverage 閮??")

            # 憒???暸?鈭?Smart Leverage嚗???唾?蝞?蒂?湔? Metrics
            if smart_leverage_on and ds_for_calc is not None and not ds_for_calc.empty and 'w' in ds_for_calc.columns:
                logger.info(f"[{strategy}] ??? Smart Leverage 閮?? (?湔???????銵?...")

                # A. 閮???唳楊??(0050 ?蹂誨?暸?)
                smart_ds = calculate_smart_leverage_equity(ds_for_calc, df_raw, safe_ticker="0050.TW")

                # B. ?踵?????祉?霈??嚗???Ｙ??怠??????
                if daily_state_std is not None and not daily_state_std.empty:
                    daily_state_std = smart_ds
                else:
                    daily_state = smart_ds

                ds_for_calc = smart_ds

                # C. ??? Metrics (霈?KPI ?∠??詨?霈??)
                try:
                    eq = smart_ds['equity']

                    # 蝮賢???
                    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1

                    # 撟游??梢?
                    days = (eq.index[-1] - eq.index[0]).days
                    years = max(days / 365.25, 0.1)
                    ann_ret = (1 + total_ret) ** (1 / years) - 1

                    # MDD
                    roll_max = eq.cummax()
                    dd = (eq / roll_max) - 1
                    mdd = dd.min()

                    # Sharpe (蝪⊥???
                    daily_ret = eq.pct_change().dropna()
                    sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if daily_ret.std() != 0 else 0

                    # 撘瑕??湔? result['metrics']
                    if 'metrics' not in result:
                        result['metrics'] = {}

                    result['metrics']['total_return'] = total_ret
                    result['metrics']['annual_return'] = ann_ret
                    result['metrics']['max_drawdown'] = mdd
                    result['metrics']['sharpe_ratio'] = sharpe

                    logger.info(f"[{strategy}] Metrics ???摰??: Ret={total_ret:.2%}, Ann={ann_ret:.2%}, MDD={mdd:.2%}, Sharpe={sharpe:.2f}")

                except Exception as e:
                    logger.error(f"[{strategy}] ??? Metrics 憭望?: {e}")

            # ==============================================================================
            # ?? ?曉? result['metrics'] 撌脩??舀??啁?鈭???臭誑??? KPI ?∠?
            # ==============================================================================

            metrics = result.get('metrics', {})
            # ??Ensemble ?交?????典?/??????嚗??ㄐ隞交???????朣?
            _ensure_exposure_metrics(metrics, result, daily_state_hint=daily_state_std)
            tooltip = f"{strategy} 蝑??隤祆?"
            param_display = {k: v for k, v in param_presets[strategy].items() if k != "strategy_type"}
            param_str = ", ".join(f"{k}: {v}" for k, v in param_display.items())
            avg_holding = calculate_holding_periods(trade_df)
            metrics['avg_holding_period'] = avg_holding
            
            label_map = {
                "total_return": "蝮賢??梁?",
                "annual_return": "撟游??????,
                "win_rate": "???",
                "max_drawdown": "?憭批???,
                "max_drawdown_duration": "??????",
                "calmar_ratio": "?∠?瘥??",
                "sharpe_ratio": "憭??瘥??",
                "sortino_ratio": "蝝Ｘ?隢暹???,
                "payoff_ratio": "???瘥?,
                "profit_factor": "??????",
                "time_in_market": "?典?瘥??",
                "turnover_py": "撟游??????,
                "num_trades": "鈭斗?甈⊥?",
                "avg_holding_period": "撟喳????予??,
                "annualized_volatility": "撟游?瘜Ｗ???,
                "max_consecutive_wins": "?憭折?????",
                "max_consecutive_losses": "?憭折???扳?",
                "avg_win": "撟喳????",
                "avg_loss": "撟喳??扳?",
            }
            
            # ?? 摰???澆?????拙?撘?- ??? None/NaN/Inf ??
            def safe_fmt(val, is_pct=False, is_int=False, suffix=""):
                if val is None or pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                    return "N/A"
                try:
                    if is_pct:
                        return f"{val:.2%}"
                    if is_int:
                        return f"{int(val)}{suffix}"
                    return f"{val:.2f}{suffix}"
                except (TypeError, ValueError):
                    return str(val)

            metric_cards = []
            for k, v in metrics.items():
                if k in ["total_return", "annual_return", "win_rate", "max_drawdown", "annualized_volatility", "avg_win", "avg_loss", "time_in_market", "turnover_py"]:
                    txt = safe_fmt(v, is_pct=True)
                elif k in ["calmar_ratio", "sharpe_ratio", "sortino_ratio", "payoff_ratio", "profit_factor"]:
                    txt = safe_fmt(v, is_pct=False)
                elif k in ["max_drawdown_duration", "avg_holding_period"]:
                    txt = safe_fmt(v, is_pct=False, suffix=" 憭?)
                elif k in ["num_trades", "max_consecutive_wins", "max_consecutive_losses"]:
                    txt = safe_fmt(v, is_int=True)
                else:
                    txt = safe_fmt(v)
                metric_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(label_map.get(k, k), className="card-title-label", style={"color": card_text}),
                                html.Div(txt, style={"font-weight": "bold", "font-size": "18px", "color": card_text})
                            ])
                        ], style={"background": card_bg, "border": f"1px solid {card_border}", "border-radius": "8px", "margin-bottom": "6px", "width": "100%"})
                    ], xs=12, sm=6, md=4, lg="auto", xl="auto", style={"flex": "0 0 160px", "maxWidth": "160px", "margin-bottom": "6px"})
                )
            
            # ?? 雿輻?蝯曹???”蝟餌絞嚗?????甇亦葬?橘?
            from sss_core.plotting_unified import create_unified_dashboard

            # ?芸?雿輻?璅???????????撌脣?????湔????嚗?Ⅱ靽??雿????
            if daily_state_std is not None and not daily_state_std.empty:
                daily_state_display = daily_state_std
                logger.debug(f"[UI] 蝜芸?雿輻? daily_state_std嚗??雿? {list(daily_state_std.columns)}")
            else:
                daily_state_display = daily_state
                logger.debug(f"[UI] 蝜芸?雿輻???? daily_state嚗??雿? {list(daily_state.columns) if daily_state is not None else None}")

            # 瑼Ｘ?暺??敹恍???伐?
            logger.debug(f"[UI] trade_df cols={list(trade_df.columns)} head=\n{trade_df.head(3)}")

            # ??甈??隤??蝯曹?
            daily_state_display = normalize_daily_state_columns(daily_state_display)

            # ?? 靽格迤 log 瑼Ｘ?嚗???祇???daily_state.columns嚗?
            logger.debug(f"[UI] daily_state_display cols={list(daily_state_display.columns) if daily_state_display is not None else None}")
            if daily_state_display is not None:
                has_cols = {'equity','cash'}.issubset(daily_state_display.columns)
                logger.debug(f"[UI] daily_state_display head=\n{daily_state_display[['equity','cash']].head(3) if has_cols else 'Missing equity/cash columns'}")

            vote_series = None
            vote_threshold = None
            vote_diagnostics = _new_vote_diagnostics()
            vote_warning_component = html.Div()

            # === ?? 撱箇?蝯曹???”嚗?????甇伐? ===
            if daily_state_display is not None and not daily_state_display.empty and {'equity','cash'}.issubset(daily_state_display.columns):
                vote_series, vote_threshold, vote_diagnostics = compute_ensemble_vote_series(
                    daily_state_display,
                    strategy,
                )
                if str(strategy).startswith("Ensemble") and vote_series is None:
                    requested = vote_diagnostics.get("requested", 0)
                    loaded = vote_diagnostics.get("loaded", 0)
                    failed = vote_diagnostics.get("failed", 0)
                    vote_warning_component = dbc.Alert(
                        f"Ensemble vote series unavailable (loaded/requested={loaded}/{requested}, failed={failed}).",
                        color="warning",
                        style={"marginTop": "8px", "marginBottom": "8px"},
                    )

                # 雿輻?蝯曹???”蝟餌絞
                unified_fig = create_unified_dashboard(
                    df_raw=df_raw,
                    daily_state=daily_state_display,
                    trade_df=trade_df,
                    ticker=ticker,
                    theme='dark' if theme == 'theme-dark' else 'light',
                    votes_series=vote_series,
                    votes_threshold=vote_threshold,
                )

                # ???????Ｗ?銵剁?撌脰酉閫??
                # fig2 = plot_equity_cash(
                #     daily_state_display[['equity','cash']].copy(),
                #     df_raw
                # )
                # fig_w = plot_weight_series(daily_state_display, trade_df)
                # fig_w.update_layout(
                #     template=plotly_template,
                #     font_color=font_color,
                #     plot_bgcolor=bg_color,
                #     paper_bgcolor=bg_color,
                #     legend=dict(bgcolor=legend_bgcolor, bordercolor=legend_bordercolor, font=dict(color=legend_font_color))
                # )
                
                # === ?啣?嚗???????”??===
                # 雿輻?璅???????daily_state_display嚗?歇蝬??皞?????嚗?
                # 皞??鞈??甈??銵冽??豢?
                if not daily_state_display.empty:
                    # ?豢?閬?＊蝷箇?甈??嚗?? Streamlit 銝?湛?
                    display_cols = ['portfolio_value', 'position_value', 'cash', 'invested_pct', 'cash_pct', 'w']
                    available_cols = [col for col in display_cols if col in daily_state_display.columns]
                    
                    if available_cols:
                        # ?澆????????潮＊蝷?
                        display_daily_state = daily_state_display[available_cols].copy()
                        display_daily_state.index = display_daily_state.index.strftime('%Y-%m-%d')
                        
                        # ?澆??????
                        for col in ['portfolio_value','position_value','cash']:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(
                                    lambda x: f"{int(round(x)):,}" if pd.notnull(x) and not pd.isna(x) else ""
                                )
                        
                        for col in ['invested_pct','cash_pct']:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(
                                    lambda x: f"{x:.2%}" if pd.notnull(x) else ""
                                )
                        
                        for col in ['w']:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(
                                    lambda x: f"{x:.3f}" if pd.notnull(x) else ""
                                )
                        
                        # ?萄遣鞈??甈??銵冽?
                        daily_state_table = html.Div([
                            html.H5("蝮質??ａ?蝵?, style={"marginTop": "16px", "color": font_color}),
                            html.Div("瘥??鞈????蔭???????蝮質??Ｕ???撣?潦??????鞈??靘??",
                                     style={"fontSize": "14px", "color": font_color, "marginBottom": "8px"}),
                            dash_table.DataTable(
                                columns=[{"name": i, "id": i} for i in display_daily_state.columns],
                                data=display_daily_state.head(20).to_dict('records'),  # ?芷＊蝷箏?20蝑?
                                style_table={'overflowX': 'auto', 'backgroundColor': bg_color},
                                style_cell={'textAlign': 'center', 'backgroundColor': table_cell_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                                style_header={'backgroundColor': table_header_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                                id={'type': 'daily-state-table', 'strategy': strategy}
                            ),
                            html.Div(f"憿舐內??0蝑??????惋len(display_daily_state)}蝑?, 
                                     style={"fontSize": "12px", "color": "#888", "textAlign": "center", "marginTop": "8px"})
                        ])
                    else:
                        daily_state_table = html.Div("鞈??甈??鞈??銝?雲", style={"color": "#888", "fontStyle": "italic"})
                else:
                    daily_state_table = html.Div("鞈??甈??鞈???箇征", style={"color": "#888", "fontStyle": "italic"})
            else:
                # ??嚗????daily_state嚗?蝙?函征?賜絞銝??”
                logger.debug("[UI] 雿輻? fallback嚗?aily_state ?箇征嚗?遣蝡?征?賢?銵?)
                unified_fig = go.Figure()
                unified_fig.add_annotation(
                    text="鞈??銝?雲嚗??瘜?鼓鋆賢?銵?,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20, color="#888")
                )
                unified_fig.update_layout(
                    height=600,
                    template=plotly_template,
                    paper_bgcolor=bg_color
                )

                # ?????fallback ??摩嚗?歇閮餉圾嚗?
                # fig2 = plot_equity_cash(trade_df, df_raw)
                # if daily_state_display is not None and not daily_state_display.empty and 'w' in daily_state_display.columns:
                #     fig_w = plot_weight_series(daily_state_display['w'], title="???甈??霈??")
                #     fig_w.update_layout(...)
                # else:
                #     fig_w = go.Figure()

                daily_state_table = html.Div("雿輻?鈭斗?銵券?撱箇?甈??/?暸??脩?", style={"color": "#888", "fontStyle": "italic"})

            # ?????fig2 layout ?湔?嚗?歇銝??閬??
            # fig2.update_layout(
            #     template=plotly_template, font_color=font_color, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            #     legend_font_color=legend_font_color,
            #     legend=dict(bgcolor=legend_bgcolor, bordercolor=legend_bordercolor, font=dict(color=legend_font_color))
            # )
            
            # 蝯曹????摨虫誑 figure.layout.height ?箸?嚗???????◤?箏?擃?漲憯?葬???
            unified_fig_height = getattr(unified_fig.layout, "height", None)
            if not isinstance(unified_fig_height, (int, float)) or unified_fig_height <= 0:
                unified_fig_height = 1700 if vote_series is not None and len(vote_series) > 0 else 1500
            unified_graph_style = {'height': f'{int(unified_fig_height)}px'}

            # === 閮??憸券??仿?敺賜??批捆 ===
            valve = results.get(strategy, {}).get('valve', {}) or {}
            valve_badge_text = ("撌脣??? if valve.get("applied") else "?芸???)
            valve_badge_extra = []
            if isinstance(valve.get("cap"), (int, float)):
                valve_badge_extra.append(f"CAP={valve['cap']:.2f}")
            if isinstance(valve.get("atr_ratio"), (int, float)):
                valve_badge_extra.append(f"ATR瘥??{valve['atr_ratio']:.2f}")
            elif valve.get("atr_ratio") == "forced":
                valve_badge_extra.append("撘瑕?閫貊?")
            
            valve_badge = html.Span(
                "??儭?憸券??仿?嚗? + valve_badge_text + ((" | " + " | ".join(valve_badge_extra)) if valve_badge_extra else ""),
                style={
                    "marginLeft": "8px",
                    "color": ("#dc3545" if valve.get("applied") else "#6c757d"),
                    "fontWeight": "bold"
                }
            ) if valve else html.Span("")

            strategy_content = html.Div([
                html.H4([
                    f"??葫蝑??: {strategy} ",
                    html.Span("??, title=tooltip, style={"cursor": "help", "color": "#888"}),
                    valve_badge
                ]),
                html.Div(f"???閮剖?: {param_str}"),
                html.Br(),
                dbc.Row(
                    metric_cards,
                    style={"flexWrap": "wrap", "justifyContent": "flex-start", "columnGap": "8px", "rowGap": "6px"},
                    className='metrics-cards-row'
                ),
                html.Br(),
                # ?? 蝯曹???”嚗?????甇亦葬?橘?
                dcc.Graph(
                    figure=unified_fig,
                    config={'displayModeBar': True, 'scrollZoom': True},
                    className='unified-dashboard-graph',
                    style=unified_graph_style
                ),

                # ??????撘萄??Ｗ?銵剁?撌脰酉閫??
                # dcc.Graph(figure=fig1, config={'displayModeBar': True}, className='main-metrics-graph'),
                # dcc.Graph(figure=fig2, config={'displayModeBar': True}, className='main-cash-graph'),
                # dcc.Graph(figure=fig_w, config={'displayModeBar': True}, className='main-weight-graph'),
                # 撠?漱???蝝唳?憿??隤祆???蔥?箏?銝銵?
                html.Div([
                    html.H5("鈭斗???敦", style={"marginBottom": 0, "marginRight": "12px", "color": font_color}),
                    html.Div("撖阡?銝???亦?靽∟??亦???予嚗?+1嚗??靽格?隞?Ⅳ??蔣?踹?憭?惜?ｇ??思?靽格?",
                             style={"fontWeight": "bold", "fontSize": "16px", "color": font_color})
                ], style={"display": "flex", "alignItems": "center", "marginTop": "16px"}),

                dash_table.DataTable(
                    columns=[{"name": get_column_display_name(i), "id": i} for i in display_df.columns],
                    data=display_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'backgroundColor': bg_color},
                    style_cell={'textAlign': 'center', 'backgroundColor': table_cell_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                    style_header={'backgroundColor': table_header_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                    id={'type': 'strategy-table', 'strategy': strategy}
                ),
                
                # === ?啣?嚗?漱???蝝?CSV 銝????? ===
                html.Div([
                    html.Button(
                        "?? 銝??鈭斗???敦 CSV",
                        id={'type': 'download-trade-details-csv', 'strategy': strategy},
                        style={
                            'backgroundColor': '#28a745',
                            'color': 'white',
                            'border': 'none',
                            'padding': '8px 16px',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'marginTop': '8px',
                            'fontSize': '14px'
                        }
                    ),
                    dcc.Download(id={'type': 'download-trade-details-data', 'strategy': strategy})
                ], style={'textAlign': 'center', 'marginTop': '8px'}),
                

            ])
            
            strategy_tabs.append(dcc.Tab(label=strategy, value=f"strategy_{strategy}", children=strategy_content))
        
        # ?萄遣蝑????葫?????惜摰孵?
        return html.Div([
            dcc.Tabs(
                id='strategy-tabs',
                value=f"strategy_{strategy_names[0]}" if strategy_names else "no_strategy",
                children=strategy_tabs,
                className='strategy-tabs-bar'
            )
        ])
        
    elif tab == "compare":
        def _normalize_type_value(v):
            s = str(v).strip().lower()
            type_map = {
                'buy': 'buy', 'add': 'buy', 'long': 'buy', 'entry': 'buy', '鞎瑕?': 'buy', '??Ⅳ': 'buy', '???': 'buy',
                'sell': 'sell', 'exit': 'sell', 'short': 'sell', '鞈??': 'sell', '?箏?': 'sell',
                'sell_forced': 'sell_forced', 'forced_sell': 'sell_forced', 'force_sell': 'sell_forced',
                '撘瑕?鞈??': 'sell_forced', '撘瑕?撟喳?: 'sell_forced',
            }
            return type_map.get(s, s)

        def _normalize_reason_value(v):
            s = str(v).strip().lower()
            reason_map = {
                'end_of_period': 'end_of_period',
                'end_of_backtest': 'end_of_period',
                'backtest_end': 'end_of_period',
                '???撟喳?: 'end_of_period',
            }
            return reason_map.get(s, s)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['close'], name='Close Price', line=dict(color='dodgerblue')))
        colors = ['green', 'limegreen', 'red', 'orange', 'purple', 'blue', 'pink', 'cyan']
        
        # 摰?儔蝑???圈??脩????
        strategy_colors = {strategy: colors[i % len(colors)] for i, strategy in enumerate(strategy_names)}
        
        # ?箏?銵冽溶??眺鞈??
        for i, strategy in enumerate(strategy_names):
            result = results.get(strategy)
            if not result:
                continue
            # 雿輻?閫???典??賂??舀? pack_df ???蝯?JSON 摮?葡?拍車?澆?
            trade_df = df_from_pack(result.get('trade_df'))
            
            # 璅????漱?????
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
            except Exception:
                # 敺??璅?????獢?
                if trade_df is not None and len(trade_df) > 0:
                    trade_df = trade_df.copy()
                    trade_df.columns = [str(c).lower() for c in trade_df.columns]
                    if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                        trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
                    if "type" not in trade_df.columns and "action" in trade_df.columns:
                        trade_df["type"] = trade_df["action"].astype(str).str.lower()
                    if "price" not in trade_df.columns:
                        for c in ["open", "price_open", "exec_price", "px", "close"]:
                            if c in trade_df.columns:
                                trade_df["price"] = trade_df[c]
                                break
            
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            if 'type' in trade_df.columns:
                trade_df['type_norm'] = trade_df['type'].apply(_normalize_type_value)
            else:
                trade_df['type_norm'] = ''
            if 'reason' in trade_df.columns:
                trade_df['reason_norm'] = trade_df['reason'].apply(_normalize_reason_value)
            else:
                trade_df['reason_norm'] = ''
            if trade_df.empty:
                continue
            # ??????皜祆??怠撥?嗅像????嚗????炊撠????祕鈭斗?閮??
            synthetic_forced_mask = (trade_df['type_norm'] == 'sell_forced') & (trade_df['reason_norm'] == 'end_of_period')
            trade_df_plot = trade_df.loc[~synthetic_forced_mask].copy()

            buys = trade_df_plot[trade_df_plot['type_norm'] == 'buy']
            sells = trade_df_plot[trade_df_plot['type_norm'] == 'sell']
            sells_forced = trade_df_plot[trade_df_plot['type_norm'] == 'sell_forced']

            fig.add_trace(go.Scatter(x=buys['trade_date'], y=buys['price'], mode='markers', name=f'{strategy} Buy',
                                     marker=dict(symbol='cross', size=8, color=colors[i % len(colors)])))
            fig.add_trace(go.Scatter(x=sells['trade_date'], y=sells['price'], mode='markers', name=f'{strategy} Sell',
                                     marker=dict(symbol='x', size=8, color=colors[i % len(colors)])))
            if not sells_forced.empty:
                fig.add_trace(go.Scatter(x=sells_forced['trade_date'], y=sells_forced['price'], mode='markers',
                                         name=f'{strategy} Forced',
                                         marker=dict(symbol='square', size=8, color='gray')))
        
        # ?湔???”雿??
        fig.update_layout(
            title=f'{ticker} ?????亥眺鞈??瘥??',
            xaxis_title='Date', yaxis_title='?∪?', template=plotly_template,
            font_color=font_color, plot_bgcolor=bg_color, paper_bgcolor=bg_color, legend_font_color=legend_font_color,
            legend=dict(
                x=1.05, y=1, xanchor='left', yanchor='top',
                bordercolor=legend_bordercolor, borderwidth=1, bgcolor=legend_bgcolor,
                itemsizing='constant', orientation='v', font=dict(color=legend_font_color)
            )
        )
        
        # 皞??瘥??銵冽??豢?
        comparison_data = []
        for strategy in strategy_names:
            result = results.get(strategy)
            if not result:
                continue
            
            # 霈??漱?????
            trade_df = df_from_pack(result.get('trade_df'))
            
            # 璅????漱?????
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
            except Exception:
                # 敺??璅?????獢?
                if trade_df is not None and len(trade_df) > 0:
                    trade_df = trade_df.copy()
                    trade_df.columns = [str(c).lower() for c in trade_df.columns]
                    if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                        trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
                    if "type" not in trade_df.columns and "action" in trade_df.columns:
                        trade_df["type"] = trade_df["action"].astype(str).str.lower()
                    if "price" not in trade_df.columns:
                        for c in ["open", "price_open", "exec_price", "px", "close"]:
                            if c in trade_df.columns:
                                trade_df["price"] = trade_df[c]
                                break
            
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            if 'type' in trade_df.columns:
                trade_df['type_norm'] = trade_df['type'].apply(_normalize_type_value)
            else:
                trade_df['type_norm'] = ''
            if 'reason' in trade_df.columns:
                trade_df['reason_norm'] = trade_df['reason'].apply(_normalize_reason_value)
            else:
                trade_df['reason_norm'] = ''

            # 閮??閰喟敦蝯梯?靽⊥?
            detailed_stats = calculate_strategy_detailed_stats(trade_df, df_raw)

            metrics = result['metrics']

            # 摰????澆??賂???? None -> 0 ???????踹??澆????隤?
            def safe_get(key, default=0.0):
                v = metrics.get(key)
                return v if v is not None else default

            comparison_data.append({
                '蝑??': strategy,
                '蝮賢??梁?': f"{safe_get('total_return'):.2%}",
                '撟游??????: f"{safe_get('annual_return'):.2%}",
                '?憭批???: f"{safe_get('max_drawdown'):.2%}",
                '?∠?瘥??': f"{safe_get('calmar_ratio'):.2f}",
                '鈭斗?甈⊥?': int(safe_get('num_trades', 0)),
                '???': f"{safe_get('win_rate'):.2%}",
                '???瘥?: f"{safe_get('payoff_ratio'):.2f}",
                '撟喳????憭拇?': f"{detailed_stats['avg_holding_days']:.1f}",
                '鞈??鞎瑕像??予??: f"{detailed_stats['avg_sell_to_buy_days']:.1f}",
                '?桀????: detailed_stats['current_status'],
                '頝??銝?活???憭拇?': f"{detailed_stats['days_since_last_action']}"
            })
        
        # 摰?儔憿??隤踵??賣?
        def adjust_color_for_theme(color, theme):
            # ???蝢拚??脣? RGB ???撠?
            color_to_rgb = {
                'green': '0, 128, 0',
                'limegreen': '50, 205, 50', 
                'red': '255, 0, 0',
                'orange': '255, 165, 0',
                'purple': '128, 0, 128',
                'blue': '0, 0, 255',
                'pink': '255, 192, 203',
                'cyan': '0, 255, 255'
            }
            
            rgb = color_to_rgb.get(color, '128, 128, 128')  # 暺???啗?
            
            if theme == 'theme-dark':
                return f'rgba({rgb}, 0.2)'  # ???摨?0.2
            elif theme == 'theme-light':
                return f'rgba({rgb}, 1)'    # ???摨?1
            else:  # theme-blue
                return f'rgba({rgb}, 0.5)'  # ???摨?0.5
        
        # ?萄遣瘥??銵冽?銝行??冽?隞嗆見撘?
        compare_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in comparison_data[0].keys()] if comparison_data else [],
            data=comparison_data,
            style_table={'overflowX': 'auto', 'backgroundColor': bg_color},
            style_cell={'textAlign': 'center', 'backgroundColor': table_cell_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
            style_header={'backgroundColor': table_header_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
            style_data_conditional=[
                {
                    'if': {'row_index': i},
                    'backgroundColor': adjust_color_for_theme(strategy_colors[row['蝑??']], theme),
                    'border': f'1px solid {strategy_colors[row['蝑??']]}'
                } for i, row in enumerate(comparison_data)
            ],
            id='compare-table'
        )
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': True}, className='compare-graph'),
            html.Hr(),
            compare_table
        ])
        
    elif tab == "enhanced":
        # === 憓?撥?????? ===
        enhanced_controls = html.Div([
            html.H4("?? 憓?撥???"),
            
            # === ?啣?嚗??撅???憟??????蝷?===
            html.Div([
                html.Div(id="enhanced-global-status", style={
                    "padding": "12px",
                    "marginBottom": "16px",
                    "borderRadius": "8px",
                    "border": "1px solid #dee2e6",
                    "backgroundColor": "#f8f9fa"
                })
            ]),
            
            # === ?啣?嚗????葫蝯??頛???憛?===
            html.Details([
                html.Summary("?? 敺??皜祉??????),
                html.Div([
                    html.Div("?豢?蝑??嚗????????ledger_std > ledger > trade_df嚗?, 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    dcc.Dropdown(
                        id="enhanced-strategy-selector",
                        placeholder="隢???瑁???葫...",
                        style={"width":"100%","marginBottom":"8px"}
                    ),
                    html.Button("頛???詨?蝑??", id="load-enhanced-strategy", n_clicks=0, 
                               style={"width":"100%","marginBottom":"8px"}),
                    html.Div("隢???瑁???葫嚗??頛??蝑????, id="enhanced-load-status", style={"fontSize":"12px","color":"#888"}),
                    html.Div("?? ??葫摰??敺???芸?敹怠??雿喟???, 
                             style={"fontSize":"11px","color":"#666","fontStyle":"italic","marginTop":"4px"})
                ])
            ], style={"marginBottom":"16px"}),
            
            # === ?梯???cache store ===
            dcc.Store(id="enhanced-trades-cache"),
            
            html.Details([
                html.Summary("憸券??仿???葫"),
                html.Div([
                    dcc.Dropdown(
                        id="rv-mode", options=[
                            {"label":"???銝?? (cap)","value":"cap"},
                            {"label":"蝳?迫??Ⅳ (ban_add)","value":"ban_add"},
                        ], value="cap", clearable=False, style={"width":"240px"}
                    ),
                    dcc.Slider(id="rv-cap", min=0.1, max=1.0, step=0.05, value=0.5,
                               tooltip={"placement":"bottom","always_visible":True}),
                    html.Div("ATR(20)/ATR(60) 瘥?潮?瑼?, style={"marginTop":"8px"}),
                    dcc.Slider(id="rv-atr-mult", min=1.0, max=2.0, step=0.05, value=1.3,
                               tooltip={"placement":"bottom","always_visible":True}),
                    html.Button("?瑁?憸券??仿???葫", id="run-rv", n_clicks=0, style={"marginTop":"8px"})
                ])
            ]),
            
            html.Div(id="rv-summary", style={"marginTop":"12px"}),
            dcc.Graph(id="rv-equity-chart", style={"height": "420px", "marginTop": "8px"}),
            dcc.Graph(id="rv-dd-chart", style={"height": "360px", "marginTop": "8px"}),
            
            # === ?啣?嚗?????撠????===
            html.Details([
                html.Summary("?? ?豢?瘥????那??),
                html.Div([
                    html.Div("?湔?頛詨?撖阡??豢??脰?瘥??嚗?那?瑕?撅憟????撥???????????????", 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Button("頛詨??豢?瘥???勗?", id="export-data-comparison", n_clicks=0, 
                               style={"width":"100%","marginBottom":"8px","backgroundColor":"#17a2b8","color":"white"}),
                    html.Div(id="data-comparison-output", style={"fontSize":"12px","color":"#666","marginTop":"8px"}),
                    dcc.Download(id="data-comparison-csv")
                ])
            ], style={"border":"1px solid #17a2b8","borderRadius":"8px","padding":"12px","marginTop":"12px"}),
            
            # === ?啣?嚗?◢???梢??啣?嚗?areto Map嚗??憛?===
            html.Details([
                html.Summary("?? 憸券?-?梢??啣?嚗?areto Map嚗?),
                html.Div([
                    html.Div("???蝑????◢???梢??????”", 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Button("??? Pareto Map", id="generate-pareto-map", n_clicks=0, 
                               style={"width":"100%","marginBottom":"8px"}),
                    html.Div(id="pareto-map-status", style={"fontSize":"12px","color":"#888","marginBottom":"8px"}),
                    dcc.Graph(id="pareto-map-graph", style={"height":"600px"}),
                    html.Div([
                        html.Button("?? 銝?? Pareto Map ?豢? (CSV)", id="download-pareto-csv", n_clicks=0,
                                   style={"width":"100%","marginBottom":"8px"}),
                        dcc.Download(id="pareto-csv-download"),
                        html.H6("??”隤芣?嚗?, style={"marginTop":"16px","marginBottom":"8px"}),
                        html.Ul([
                            html.Li("璈怨遘嚗??憭批??歹???椰??末嚗?),
                            html.Li("蝮梯遘嚗?F ?脣????嚗??銝??憟踝?"),
                            html.Li("憿??嚗??撠曇矽?游?摨佗?蝝??=????喳偏嚗?????曉之?喳偏嚗??箔葉蝺??"),
                            html.Li("暺?之撠??憸券?閫貊?憭拇?嚗??憭改?蝞∪?頞??嚗?),
                            html.Li("???????蝬?????獢??嚗??銝??撌艾???脫?餈?葉蝺???銝??憭批?隤?撐嚗?)
                        ], style={"fontSize":"12px","color":"#666"})
                    ])
                ])
            ], style={"border":"1px solid #333","borderRadius":"8px","padding":"12px","marginTop":"12px"}),
            # === 鈭斗?鞎Ｙ???圾?憛?===
            html.Details([
                html.Summary("?? 鈭斗?鞎Ｙ???圾"),
                html.Div([
                    html.Div("??圾鈭斗?鞎Ｙ?嚗????????蝣?皜?Ⅳ??挾??蜀??”??, 
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Div([
                        html.Div([
                            html.Label("?撠??頝?(憭?", style={"fontSize":"12px","color":"#888"}),
                            dcc.Input(id="phase-min-gap", type="number", value=5, min=0, max=30, step=1,
                                     style={"width":"80px","marginRight":"16px"})
                        ], style={"display":"inline-block","marginRight":"16px"}),
                                            html.Div([
                        html.Label("?瑕???(憭?", style={"fontSize":"12px","color":"#888"}),
                        dcc.Input(id="phase-cooldown", type="number", value=10, min=0, max=30, step=1,
                                 style={"width":"80px"})
                    ], style={"display":"inline-block"})
                ], style={"marginBottom":"8px"}),
                html.Div([
                    html.Button("?瑁?鈭斗?鞎Ｙ???圾", id="run-phase", n_clicks=0, 
                               style={"width":"48%","marginBottom":"8px","marginRight":"2%"}),
                    html.Button("?寥?皜祈岫???蝭??", id="run-batch-phase", n_clicks=0,
                               style={"width":"48%","marginBottom":"8px","marginLeft":"2%","backgroundColor":"#28a745","color":"white"})
                ], style={"display":"flex","justifyContent":"space-between"}),
                    html.Div([
                        html.H6("???隤芣?嚗?, style={"marginTop":"16px","marginBottom":"8px"}),
                        html.Ul([
                            html.Li("?撠??頝???拇活??Ⅳ?喳?閬????嗾憭抬?????函?閮??嚗??瞈曄?????喉?"),
                            html.Li("?瑕????瘥?活??Ⅳ敺??敹?????銋????迂銝??蝑??蝣潘??踹???漲???嚗?),
                            html.Li("?券??霈??閫???血?瘥?????蝢拍???Ⅳ瘜Ｘ挾嚗????◤?剜?撠??蝔??)
                        ], style={"fontSize":"12px","color":"#666","marginBottom":"16px"}),
                        html.Div(id="phase-table"),
                        html.Div([
                            html.H6("?寥?皜祈岫蝯??", style={"marginTop":"16px","marginBottom":"8px","color":"#28a745"}),
                            html.Div(id="batch-phase-results", style={"fontSize":"12px"})
                        ])
                    ])
                ])
            ], style={"border":"1px solid #333","borderRadius":"8px","padding":"12px","marginTop":"12px"})
        ], style={"maxWidth":"1240px", "margin":"0 auto", "padding":"8px 12px 16px"})
        
        return enhanced_controls
    elif tab == "daily_signal":
            # === 瘥??閮???唳?摰?(Dash ??祕雿? ===
            import os
            from datetime import datetime
            
            history_file = 'analysis/signal_history.csv'
            
            # 摰?儔?瑁????
            run_btn_layout = html.Div([
                html.Button("?? ?瑁?隞????葫 (?澆? predict_tomorrow.py)", id="btn-run-prediction", n_clicks=0, 
                        className="btn btn-danger", style={"marginBottom": "20px"}),
                html.Div(id="prediction-status-msg") # ?其?憿舐內?瑁?蝯??
            ])

            # 瑼Ｘ?瑼???臬?摮??
            if not os.path.exists(history_file):
                return html.Div([
                    html.H4("?? SSS096 瘥??????唳?摰?),
                    html.Div("撠??甇瑕?蝝???隢??????寞????銵??甈⊿?皜研?, style={"color": "orange"}),
                    run_btn_layout
                ], style={"padding": "20px"})
            
            # 霈?????
            try:
                df_hist = pd.read_csv(history_file)
                if df_hist.empty:
                    return html.Div(["蝝????箇征嚗??????瑁???葫??, run_btn_layout])
                
                # ?渡??豢?
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                df_hist = df_hist.sort_values('date', ascending=False)
                latest_date_dt = df_hist['date'].iloc[0]
                latest_date_str = latest_date_dt.strftime('%Y-%m-%d')
                
                # 蝭拚?隞??鞈??
                df_today = df_hist[df_hist['date'] == latest_date_dt].copy()
                
                # 閮??蝯梯?
                total_votes = len(df_today)
                long_votes = len(df_today[df_today['signal'] == 'LONG'])
                cash_votes = total_votes - long_votes
                
                # ?斗?撱箄降
                decision_text = "?? ?脣???? (LONG)" if long_votes > cash_votes else "??蝛箸?閫??(CASH)"
                decision_color = "#28a745" if long_votes > cash_votes else "#6c757d"
                
                # 鋆賭? KPI ?∠?
                def card(title, val, color):
                    return dbc.Card([
                        dbc.CardBody([
                            html.H5(title, className="card-title", style={"color": card_text}),
                            html.H2(val, style={"color": color, "fontWeight": "bold"})
                        ])
                    ], style={"backgroundColor": card_bg, "border": f"1px solid {color}"})

                kpi_row = dbc.Row([
                    dbc.Col(card("憭??蟡冽?", f"{long_votes} / {total_votes}", "#28a745"), width=4),
                    dbc.Col(card("蝛箸?蟡冽?", f"{cash_votes} / {total_votes}", "#ffc107"), width=4),
                    dbc.Col(card("?蝯?遣霅?, decision_text, decision_color), width=4),
                ], className="mb-4")

                # 鋆賭?閰喟敦銵冽?
                df_display = df_today[['strategy_name', 'signal', 'price']].copy()
                table = dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in df_display.columns],
                    data=df_display.to_dict('records'),
                    style_table={'backgroundColor': bg_color},
                    style_header={'backgroundColor': table_header_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                    style_cell={'backgroundColor': table_cell_bg, 'color': table_text, 'textAlign': 'center', 'border': f'1px solid {table_border}'},
                )

                # 鋆賭?甇瑕?頞典???
                daily_trend = df_hist.groupby('date')['signal'].apply(lambda x: (x=='LONG').sum()).reset_index()
                daily_trend.columns = ['date', 'long_votes']
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(
                    x=daily_trend['date'], y=daily_trend['long_votes'],
                    name='憭??蟡冽?', marker_color='#28a745'
                ))
                fig_trend.update_layout(
                    title="瘥??憭??蟡冽?霈??", 
                    template=plotly_template, 
                    font_color=font_color, 
                    plot_bgcolor=bg_color, 
                    paper_bgcolor=bg_color
                )

                return html.Div([
                    html.H3(f"?? ??啗???????{latest_date_str}"),
                    run_btn_layout,
                    kpi_row,
                    html.H4("閰喟敦??巨??敦"),
                    table,
                    html.Hr(),
                    dcc.Graph(figure=fig_trend)
                ], style={"padding": "20px"})

            except Exception as e:
                return html.Div([f"霈??????隤? {str(e)}", run_btn_layout])

# --------- Externalized strategy callbacks ---------
register_strategy_callbacks(
    app,
    run_backtest_func=run_backtest,
    update_tab_func=update_tab,
)

# --------- ???瘝輸?璅⊥?獢???嗅?銝駁???? ---------
@app.callback(
    Output("history-modal", "is_open"),
    Output('main-bg', 'className'),
    Output('theme-toggle', 'children'),
    Output('theme-store', 'data'),
    # [?啣?] ?批?銝????????璅??
    Output('ctrl-row-basic', 'style'),
    Output('ctrl-row-risk', 'style'),
    Output('ctrl-row-ensemble', 'style'),
    # [?啣?] ?批??寞?璅???????
    Output('label-risk-title', 'style'),
    Output('label-maj-title', 'style'),
    Output('label-prop-title', 'style'),
    [Input("history-btn", "n_clicks"), Input("history-close", "n_clicks"), Input('theme-toggle', 'n_clicks')],
    [State("history-modal", "is_open"), State('theme-store', 'data')],
    # 蝘駁? prevent_initial_call=True嚗??見?????????甇?Ⅱ皜脫?憿??
)
def toggle_history_modal_and_theme(history_btn, history_close, theme_btn, is_open, current_theme):
    ctx_trigger = ctx.triggered_id

    # 瘙箏?銝????蜓憿?
    next_theme = current_theme
    if ctx_trigger == "history-btn":
        next_theme = 'theme-dark'
        is_open = True
    elif ctx_trigger == "history-close":
        is_open = False
    elif ctx_trigger == "theme-toggle":
        if current_theme is None:
            next_theme = 'theme-dark'
        else:
            themes = ['theme-dark', 'theme-light', 'theme-blue']
            current_index = themes.index(current_theme)
            next_theme = themes[(current_index + 1) % len(themes)]

    # === 摰?儔璅??閮剖? (Style Presets) ===
    # ??身??????
    base_style = {'borderRadius': '4px', 'transition': 'background-color 0.3s'}

    if next_theme == 'theme-light':
        # 瘛箄?璅∪? (??????)
        style_basic = {**base_style, 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6', 'color': '#212529'}
        style_risk  = {**base_style, 'backgroundColor': '#fff3cd', 'border': '1px solid #ffeeba', 'color': '#856404'}
        style_ens   = {**base_style, 'backgroundColor': '#d1ecf1', 'border': '1px solid #bee5eb', 'color': '#0c5460'}

        # 璅??憿??
        c_risk = {'color': '#856404'} # 瘛梢?
        c_maj  = {'color': '#0066cc'} # 瘛梯?
        c_prop = {'color': '#28a745'} # 瘛梁?

        btn_label = '?? 瘛箄?銝駁?'

    elif next_theme == 'theme-blue':
        # ???璅∪?
        style_basic = {**base_style, 'backgroundColor': '#0b1e3a', 'border': '1px solid #446688', 'color': '#ffe066'}
        style_risk  = {**base_style, 'backgroundColor': '#3a2e05', 'border': '1px solid #886600', 'color': '#ffcc00'}
        style_ens   = {**base_style, 'backgroundColor': '#0f2b33', 'border': '1px solid #005566', 'color': '#00ccff'}

        c_risk = {'color': '#ffcc00'}
        c_maj  = {'color': '#3399ff'}
        c_prop = {'color': '#66ff66'}

        btn_label = '?? ???銝駁?'

    else: # theme-dark (??身)
        # 瘛梯?璅∪? (隤踵??箸楛??瘛梯?/瘛梢?嚗??摮????
        style_basic = {**base_style, 'backgroundColor': '#2b2b2b', 'border': '1px solid #444', 'color': '#e0e0e0'}

        # Risk: 瘛梯??脰???+ ????脤?獢?
        style_risk  = {**base_style, 'backgroundColor': '#2c2505', 'border': '1px solid #665200', 'color': '#e0e0e0'}

        # Ensemble: 瘛梢??脰???+ ??????
        style_ens   = {**base_style, 'backgroundColor': '#0c282e', 'border': '1px solid #0f4c5c', 'color': '#e0e0e0'}

        # 璅??憿??嚗??瘛梯????銝??閬?漁銝暺?
        c_risk = {'color': '#ffc107'} # 鈭桅? Warning
        c_maj  = {'color': '#66b2ff'} # 鈭株?
        c_prop = {'color': '#75df8a'} # 鈭桃?

        btn_label = '?? 瘛梯?銝駁?'

    return (
        is_open,
        next_theme,
        btn_label,
        next_theme,
        # ???璅??
        style_basic, style_risk, style_ens,
        c_risk, c_maj, c_prop
    )

# --------- 銝??鈭斗?蝝??---------
@app.callback(
    Output({'type': 'download-trade', 'strategy': ALL}, 'data'),
    Input({'type': 'download-btn', 'strategy': ALL}, 'n_clicks'),
    State({'type': 'strategy-table', 'strategy': ALL}, 'data'),
    State('backtest-store', 'data'),
    prevent_initial_call=True
)
def download_trade(n_clicks, table_data, backtest_data):
    ctx_trigger = ctx.triggered_id
    if not ctx_trigger or not backtest_data:
        return [None] * len(n_clicks)
    
    # 敺?孛?潛????ID銝剜?????亙?蝔?
    strategy = ctx_trigger['strategy']
    
    # 敺?acktest_data銝剔????????亦?鈭斗??豢?
    # backtest_data ?曉?撌脩???dict嚗???閬?json.loads
    results = backtest_data['results']
    result = results.get(strategy)
    
    if not result:
        return [None] * len(n_clicks)
    
    # 雿輻?閫???典??賂??舀? pack_df ???蝯?JSON 摮?葡?拍車?澆?
    trade_df = df_from_pack(result.get('trade_df'))
    
    # 璅????漱?????
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_df = norm(trade_df)
    except Exception:
        # 敺??璅?????獢?
        if trade_df is not None and len(trade_df) > 0:
            trade_df = trade_df.copy()
            trade_df.columns = [str(c).lower() for c in trade_df.columns]
            if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
            if "type" not in trade_df.columns and "action" in trade_df.columns:
                trade_df["type"] = trade_df["action"].astype(str).str.lower()
            if "price" not in trade_df.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_df.columns:
                        trade_df["price"] = trade_df[c]
                        break
    
    # ?萄遣銝???豢?
    def to_xlsx(bytes_io):
        with pd.ExcelWriter(bytes_io, engine='openpyxl') as writer:
            trade_df.to_excel(writer, sheet_name='鈭斗?蝝??, index=False)
    
    return [dcc.send_bytes(to_xlsx, f"{strategy}_鈭斗?蝝??xlsx") if i and i > 0 else None for i in n_clicks]

# --------- 銝??鈭斗???敦 CSV ---------
@app.callback(
    Output({'type': 'download-trade-details-data', 'strategy': ALL}, 'data'),
    Input({'type': 'download-trade-details-csv', 'strategy': ALL}, 'n_clicks'),
    State({'type': 'strategy-table', 'strategy': ALL}, 'data'),
    State('backtest-store', 'data'),
    prevent_initial_call=True
)
def download_trade_details_csv(n_clicks, table_data, backtest_data):
    """銝??鈭斗???敦??CSV ?澆?"""
    ctx_trigger = ctx.triggered_id
    if not ctx_trigger or not backtest_data:
        return [None] * len(n_clicks)
    
    # 敺?孛?潛????ID銝剜?????亙?蝔?
    strategy = ctx_trigger['strategy']
    
    # 敺?acktest_data銝剔????????亦?鈭斗??豢?
    results = backtest_data['results']
    result = results.get(strategy)
    
    if not result:
        return [None] * len(n_clicks)
    
    # 雿輻?閫???典??賂??舀? pack_df ???蝯?JSON 摮?葡?拍車?澆?
    trade_df = df_from_pack(result.get('trade_df'))
    
    # 璅????漱?????
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_df = norm(trade_df)
    except Exception:
        # 敺??璅?????獢?
        if trade_df is not None and len(trade_df) > 0:
            trade_df = trade_df.copy()
            trade_df.columns = [str(c).lower() for c in trade_df.columns]
            if "trade_date" not in trade_df.columns and "date" in trade_df.columns:
                trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
            if "type" not in trade_df.columns and "action" in trade_df.columns:
                trade_df["type"] = trade_df["action"].astype(str).str.lower()
            if "price" not in trade_df.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_df.columns:
                        trade_df["price"] = trade_df[c]
                        break
    
    # ?萄遣 CSV 銝???豢?
    def to_csv(bytes_io):
        # 雿輻? UTF-8 BOM 蝣箔? Excel ?賣迤蝣粹＊蝷箔葉??
        bytes_io.write('\ufeff'.encode('utf-8'))
        trade_df.to_csv(bytes_io, index=False, encoding='utf-8-sig')
    
    return [dcc.send_bytes(to_csv, f"{strategy}_鈭斗???敦.csv") if i and i > 0 else None for i in n_clicks]

def calculate_strategy_detailed_stats(trade_df, df_raw):
    """閮??蝑????底蝝啁絞閮?縑??""
    if trade_df.empty:
        return {
            'avg_holding_days': 0,
            'avg_sell_to_buy_days': 0,
            'current_status': '?芣???,
            'days_since_last_action': 0
        }
    
    def _normalize_reason_value(v):
        s = str(v).strip().lower()
        reason_map = {
            'end_of_period': 'end_of_period',
            'end_of_backtest': 'end_of_period',
            'backtest_end': 'end_of_period',
            '???撟喳?: 'end_of_period',
        }
        return reason_map.get(s, s)

    def _normalize_type_value(v):
        s = str(v).strip().lower()
        type_map = {
            'buy': 'buy', 'add': 'buy', 'long': 'buy', 'entry': 'buy', '鞎瑕?': 'buy', '??Ⅳ': 'buy', '???': 'buy',
            'sell': 'sell', 'exit': 'sell', 'short': 'sell', '鞈??': 'sell', '?箏?': 'sell',
            'sell_forced': 'sell_forced', 'forced_sell': 'sell_forced', 'force_sell': 'sell_forced',
            '撘瑕?鞈??': 'sell_forced', '撘瑕?撟喳?: 'sell_forced',
        }
        return type_map.get(s, s)

    def _is_synthetic_forced_exit(row):
        t = row.get('type_norm', '')
        r = row.get('reason_norm', '')
        return (t == 'sell_forced') and (r == 'end_of_period')

    # 蝣箔??交???? datetime 憿??
    if 'trade_date' in trade_df.columns:
        trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
    if 'type' in trade_df.columns:
        trade_df['type_norm'] = trade_df['type'].apply(_normalize_type_value)
    else:
        trade_df['type_norm'] = ''
    if 'reason' in trade_df.columns:
        trade_df['reason_norm'] = trade_df['reason'].apply(_normalize_reason_value)
    else:
        trade_df['reason_norm'] = ''
    
    # ??????摨?Ⅱ靽??摨?迤蝣?
    trade_df = trade_df.sort_values('trade_date').reset_index(drop=True)

    # ??????皜祆??怠撥?嗅像???????絞閮??撟脫?嚗????????撠曄垢憭??嚗?
    stats_df = trade_df.copy()
    while not stats_df.empty and _is_synthetic_forced_exit(stats_df.iloc[-1]):
        stats_df = stats_df.iloc[:-1].reset_index(drop=True)
    if stats_df.empty:
        stats_df = trade_df.copy()
    
    # 閮??撟喳????憭拇?嚗?眺?亙?鞈????予?賂?
    holding_periods = []
    for i in range(len(stats_df) - 1):
        current_type = stats_df.iloc[i]['type_norm']
        next_type = stats_df.iloc[i+1]['type_norm']
        if current_type == 'buy' and next_type in ['sell', 'sell_forced']:
            buy_date = stats_df.iloc[i]['trade_date']
            sell_date = stats_df.iloc[i+1]['trade_date']
            holding_days = (sell_date - buy_date).days
            holding_periods.append(holding_days)
    avg_holding_days = sum(holding_periods) / len(holding_periods) if holding_periods else 0
    
    # 閮??鞈??鞎瑕像??予?賂?鞈???唬?甈∟眺?亦?憭拇?嚗?
    sell_to_buy_periods = []
    for i in range(len(stats_df) - 1):
        current_type = stats_df.iloc[i]['type_norm']
        next_type = stats_df.iloc[i+1]['type_norm']
        if current_type in ['sell', 'sell_forced'] and next_type == 'buy':
            sell_date = stats_df.iloc[i]['trade_date']
            buy_date = stats_df.iloc[i+1]['trade_date']
            days_between = (buy_date - sell_date).days
            sell_to_buy_periods.append(days_between)
    avg_sell_to_buy_days = sum(sell_to_buy_periods) / len(sell_to_buy_periods) if sell_to_buy_periods else 0
    
    # ????敺??蝑??雿?
    last_trade = stats_df.iloc[-1] if not stats_df.empty else None
    if not df_raw.empty:
        current_date = df_raw.index[-1]
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
    else:
        current_date = datetime.now()
    
    if last_trade is not None:
        last_type = last_trade['type_norm']
        last_date = last_trade['trade_date']
        # ?交? w_after嚗????誑????????瑞????????舀迤蝣箄???Ensemble 摨???
        if 'w_after' in stats_df.columns and stats_df['w_after'].notna().any():
            last_w = pd.to_numeric(stats_df['w_after'], errors='coerce').dropna()
            is_holding = bool(len(last_w) > 0 and last_w.iloc[-1] > 1e-6)
            current_status = '???' if is_holding else '?芣???
            days_since_last_action = (current_date - last_date).days
        elif last_type == 'buy':
            current_status = '???'
            days_since_last_action = (current_date - last_date).days
        elif last_type in ['sell', 'sell_forced']:
            current_status = '?芣???
            days_since_last_action = (current_date - last_date).days
        else:
            current_status = '?芣???
            days_since_last_action = (current_date - last_date).days
    else:
        current_status = '?芣???
        days_since_last_action = 0
    
    return {
        'avg_holding_days': round(avg_holding_days, 1),
        'avg_sell_to_buy_days': round(avg_sell_to_buy_days, 1),
        'current_status': current_status,
        'days_since_last_action': days_since_last_action
    }

# --------- 憓?撥??? Callback嚗??撅?????????---------
@app.callback(
    Output("enhanced-global-status", "children"),
    [
        Input("global-apply-switch", "value"),
        Input("risk-cap-input", "value"),
        Input("atr-ratio-threshold", "value"),
        Input("force-valve-trigger", "value")
    ]
)
def update_enhanced_global_status(global_apply, risk_cap, atr_ratio, force_trigger):
    """?湔?憓?撥?????????撅???憟?????""
    if not global_apply:
        return html.Div([
            html.Small("?? ?典????憟???芸???, style={"color":"#dc3545","fontWeight":"bold","fontSize":"14px"}),
            html.Br(),
            html.Small("憓?撥???撠?蝙?券??Ｗ?撱箇????閮剖?", style={"color":"#666","fontSize":"12px"}),
            html.Br(),
            html.Small("?? 憒??雿輻??典?閮剖?嚗???典???????????典?撅???憟????, style={"color":"#666","fontSize":"11px","fontStyle":"italic"})
        ])
    
    # 憒??????典????憟??
    status_color = "#28a745" if not force_trigger else "#dc3545"
    status_icon = "??" if not force_trigger else "??"
    status_text = "甇?虜" if not force_trigger else "撘瑕?閫貊?"
    
    return html.Div([
        html.Small(f"{status_icon} ?典????憟??撌脣???, style={"color":status_color,"fontWeight":"bold","fontSize":"14px"}),
        html.Br(),
        html.Small(f"憸券??仿? CAP: {risk_cap}", style={"color":"#666","fontSize":"12px"}),
        html.Br(),
        html.Small(f"ATR瘥?潮?瑼? {atr_ratio}", style={"color":"#666","fontSize":"12px"}),
        html.Br(),
        html.Small(f"??? {status_text}", style={"color":status_color,"fontSize":"12px"}),
        html.Br(),
        html.Small("?? 憓?撥?????◢?芷????葫撠????蝙?券???典?閮剖?", style={"color":"#28a745","fontSize":"11px","fontStyle":"italic"})
    ])

# --------- 憓?撥??? Callback嚗?◢?芷????葫嚗?????嚗?---------
@app.callback(
    Output("rv-summary","children"),
    Output("rv-equity-chart","figure"),
    Output("rv-dd-chart","figure"),
    Input("run-rv","n_clicks"),
    State("rv-mode","value"),
    State("rv-cap","value"),
    State("rv-atr-mult","value"),
    State("enhanced-trades-cache","data"),
    # === ?啣?嚗?????撅???閮剖? ===
    State("global-apply-switch","value"),
    State("risk-cap-input","value"),
    State("atr-ratio-threshold","value"),
    # === ?啣?嚗?????撅憟???豢?皞?===
    State("backtest-store","data"),
    prevent_initial_call=True
)
def _run_rv(n_clicks, mode, cap_level, atr_mult, cache, global_apply, global_risk_cap, global_atr_ratio, backtest_data=None):
    if not n_clicks or not cache:
        return "隢??頛??蝑??鞈??", no_update, no_update

    # === 靽格迤嚗????蝙?典?撅???閮剖? ===
    if global_apply:
        # 憒??????典????憟??嚗????蝙?典?撅閮剖?
        effective_cap = global_risk_cap if global_risk_cap is not None else cap_level
        effective_atr_ratio = global_atr_ratio if global_atr_ratio is not None else atr_mult
        logger.debug(f"憓?撥???雿輻??典????嚗?AP={effective_cap}, ATR瘥?潮?瑼?{effective_atr_ratio}")
        
        # === ?啣?嚗?底蝝啁????撠???亥? ===
        logger.debug(f"=== 憓?撥??????撠?? ===")
        logger.debug(f"?典?閮剖?嚗?AP={global_risk_cap}, ATR瘥?潮?瑼?{global_atr_ratio}")
        logger.debug(f"???閮剖?嚗?AP={cap_level}, ATR瘥?潮?瑼?{atr_mult}")
        logger.debug(f"?蝯?蝙?剁?CAP={effective_cap}, ATR瘥?潮?瑼?{effective_atr_ratio}")
        
    else:
        # ?血?雿輻?憓?撥????????身摰?
        effective_cap = cap_level
        effective_atr_ratio = atr_mult
        logger.debug(f"憓?撥???雿輻???????嚗?AP={effective_cap}, ATR瘥?潮?瑼?{effective_atr_ratio}")
    
    # === ?游?嚗?蝙?刻??典?憟???詨??????? ===
    logger.debug(f"=== ?豢?撽?? ===")
    
    # ?芸?雿輻??典?憟????????嚗?Ⅱ靽???湔?
    if global_apply and backtest_data:
        # 敺?backtest-store ?脣??豢?嚗???典?憟??靽??銝??
        results = backtest_data.get("results", {})
        if results:
            # ?曉?撠??????亦???
            strategy_name = cache.get("strategy") if cache else None
            if strategy_name and strategy_name in results:
                result = results[strategy_name]
                df_raw = df_from_pack(backtest_data.get("df_raw"))
                daily_state = df_from_pack(result.get("daily_state_std") or result.get("daily_state"))
                logger.debug(f"雿輻??典?憟???豢?皞? {strategy_name}")
            else:
                # ???啣翰?????
                df_raw = df_from_pack(cache.get("df_raw"))
                daily_state = df_from_pack(cache.get("daily_state"))
                logger.debug("???啣翰??????")
        else:
            # ???啣翰?????
            df_raw = df_from_pack(cache.get("df_raw"))
            daily_state = df_from_pack(cache.get("daily_state"))
            logger.debug("???啣翰??????")
    else:
        # 雿輻?敹怠??豢?
        df_raw = df_from_pack(cache.get("df_raw"))
        daily_state = df_from_pack(cache.get("daily_state"))
        logger.debug("雿輻?敹怠??豢?皞?)
    
    # === ?啣?嚗??????湔扳炎??===
    if global_apply and backtest_data:
        logger.debug("=== ?豢?銝?湔扳炎??===")
        # 瑼Ｘ????撅憟???豢?????湔?
        global_df_raw = df_from_pack(backtest_data.get("df_raw"))
        if global_df_raw is not None and df_raw is not None:
            if len(global_df_raw) == len(df_raw):
                logger.debug(f"???豢??瑕漲銝?? {len(df_raw)}")
            else:
                logger.warning(f"???  ?豢??瑕漲銝???? ?典?={len(global_df_raw)}, 憓?撥???={len(df_raw)}")
        
        if daily_state is not None:
            logger.debug(f"??daily_state 頛?????: {len(daily_state)} 銵?)
        else:
            logger.warning("???  daily_state 頛??憭望?")
    
    # === ????豢?撽???亥? ===
    logger.debug(f"df_raw 敶Ｙ?: {df_raw.shape if df_raw is not None else 'None'}")
    logger.debug(f"daily_state 敶Ｙ?: {daily_state.shape if daily_state is not None else 'None'}")
    if daily_state is not None:
        logger.debug(f"daily_state 甈??: {list(daily_state.columns)}")
        logger.debug(f"daily_state 蝝Ｗ?蝭??: {daily_state.index.min()} ??{daily_state.index.max()}")
        if "w" in daily_state.columns:
            logger.debug(f"甈??甈??蝯梯?: ?撠??{daily_state['w'].min():.4f}, ?憭批?{daily_state['w'].max():.4f}, 撟喳???{daily_state['w'].mean():.4f}")
    
    if df_raw is None or df_raw.empty:
        return "?曆??啗??寡???, no_update, no_update
    
    if daily_state is None or daily_state.empty:
        return "?曆???daily_state嚗???亥???甈??嚗?, no_update, no_update

    # 甈??撠??
    c_open = "open" if "open" in df_raw.columns else _first_col(df_raw, ["Open","?????])
    c_close = "close" if "close" in df_raw.columns else _first_col(df_raw, ["Close","?嗥???])
    c_high  = "high" if "high" in df_raw.columns else _first_col(df_raw, ["High","?擃??"])
    c_low   = "low"  if "low"  in df_raw.columns else _first_col(df_raw, ["Low","?雿??"])

    if c_open is None or c_close is None:
        return "?∪?鞈??蝻箏? open/close 甈??", no_update, no_update

    open_px = pd.to_numeric(df_raw[c_open], errors="coerce").dropna()
    open_px.index = pd.to_datetime(df_raw.index)

    # 甈????? daily_state
    if "w" not in daily_state.columns:
        return "daily_state 蝻箏?甈??甈?? 'w'", no_update, no_update
    
    w = daily_state["w"].astype(float).reindex(open_px.index).ffill().fillna(0.0)

    # ??????嚗?蝙??SSS_EnsembleTab ??身嚗?
    cost = None

    # ?箸?嚗?? df_raw ?嗅?皞???喳?嚗???賢??賢??⊿?雿??????
    bench = pd.DataFrame({
        "?嗥???: pd.to_numeric(df_raw[c_close], errors="coerce"),
    }, index=pd.to_datetime(df_raw.index))
    if c_high and c_low:
        bench["?擃??"] = pd.to_numeric(df_raw[c_high], errors="coerce")
        bench["?雿??"] = pd.to_numeric(df_raw[c_low], errors="coerce")

    # ?閬????SSS_EnsembleTab ?扳?????賢?
    try:
        from SSS_EnsembleTab import risk_valve_backtest
        # === 憓?撥???憸券??仿?嚗?Ⅱ靽???訾??湔?(2025/08/20) ===
        enhanced_valve_params = {
            "open_px": open_px, 
            "w": w, 
            "cost": cost, 
            "benchmark_df": bench,
            "mode": mode, 
            "cap_level": float(effective_cap),  # === 靽格迤嚗?蝙?冽??????===
            "slope20_thresh": 0.0, 
            "slope60_thresh": 0.0,
            "atr_win": 20, 
            "atr_ref_win": 60, 
            "atr_ratio_mult": float(effective_atr_ratio),  # === 靽格迤嚗?蝙?冽??????===
            "use_slopes": True, 
            "slope_method": "polyfit", 
            "atr_cmp": "gt"
        }
        
        # 閮??憓?撥???憸券??仿???蔭
        logger.info(f"[Enhanced] 憸券??仿???蔭: cap_level={enhanced_valve_params['cap_level']}, atr_ratio_mult={enhanced_valve_params['atr_ratio_mult']}")
        
        out = risk_valve_backtest(**enhanced_valve_params)
    except Exception as e:
        return f"憸券??仿???葫?瑁?憭望?: {e}", no_update, no_update

    m = out["metrics"]
    
    # 閮??憸券?閫貊?憭拇?
    sig = out["signals"]["risk_trigger"]
    trigger_days = int(sig.fillna(False).sum())
    
    # === 靽格迤嚗?＊蝷箏祕??蝙?函???? ===
    summary = html.Div([
        html.Code(f"PF: ??? {m['pf_orig']:.2f} ???仿? {m['pf_valve']:.2f}"), html.Br(),
        html.Code(f"MDD: ??? {m['mdd_orig']:.2%} ???仿? {m['mdd_valve']:.2%}"), html.Br(),
        html.Code(f"?喳偏蝮賢?(>P90 甇????: ??? {m['right_tail_sum_orig']:.2f} ???仿? {m['right_tail_sum_valve']:.2f} (??m['right_tail_reduction']:.2f})"), html.Br(),
        html.Code(f"憸券?閫貊?憭拇?嚗?trigger_days} 憭?), html.Br(),
        html.Code(f"雿輻????嚗?AP={effective_cap}, ATR瘥?潮?瑼?{effective_atr_ratio}"), html.Br(),
        html.Code(f"???靘??嚗?'?典?閮剖?' if global_apply else '???閮剖?'}", style={"color": "#28a745" if global_apply else "#ffc107"})
    ])

    # 蝜芸?嚗???????????
    import plotly.graph_objects as go
    eq1 = out["daily_state_orig"]["equity"]
    eq2 = out["daily_state_valve"]["equity"]
    dd1 = eq1/eq1.cummax()-1
    dd2 = eq2/eq2.cummax()-1

    palette = {
        "orig":  {"color": "#1f77b4", "dash": "solid"},
        "valve": {"color": "#ff7f0e", "dash": "dot"},
    }

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq1.index, y=eq1, name="???",
        mode="lines", line=dict(color=palette["orig"]["color"], width=2, dash=palette["orig"]["dash"]),
        legendgroup="equity"
    ))
    fig_eq.add_trace(go.Scatter(
        x=eq2.index, y=eq2, name="?仿?",
        mode="lines", line=dict(color=palette["valve"]["color"], width=2, dash=palette["valve"]["dash"]),
        legendgroup="equity"
    ))
    fig_eq.update_layout(
        title="甈???脩?嚗?pen??pen嚗?,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(248,249,250,0.75)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        margin=dict(t=76, r=20, l=56, b=40),
        hovermode="x unified",
    )
    fig_eq.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig_eq.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd1.index, y=dd1, name="???",
        mode="lines", line=dict(color=palette["orig"]["color"], width=2, dash=palette["orig"]["dash"]),
        legendgroup="dd"
    ))
    fig_dd.add_trace(go.Scatter(
        x=dd2.index, y=dd2, name="?仿?",
        mode="lines", line=dict(color=palette["valve"]["color"], width=2, dash=palette["valve"]["dash"]),
        legendgroup="dd"
    ))
    fig_dd.update_layout(
        title="????脩?",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(248,249,250,0.75)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        yaxis_tickformat=".0%",
        margin=dict(t=76, r=20, l=56, b=40),
        hovermode="x unified",
    )
    fig_dd.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig_dd.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    return summary, fig_eq, fig_dd

# --------- 憓?撥??? Callback嚗?????撠????---------
@app.callback(
    Output("data-comparison-output", "children"),
    Output("data-comparison-csv", "data"),
    Input("export-data-comparison", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    State("backtest-store", "data"),
    State("global-apply-switch", "value"),
    State("risk-cap-input", "value"),
    State("atr-ratio-threshold", "value"),
    State("rv-cap", "value"),
    State("rv-atr-mult", "value"),
    prevent_initial_call=True
)
def generate_data_comparison_report(n_clicks, cache, backtest_data, global_apply, global_cap, global_atr, page_cap, page_atr):
    """????豢?瘥???勗?嚗?那?瑕?撅憟????撥??????????????? - 憓?撥??(2025/08/20)"""
    if not n_clicks:
        return "隢?????????????, no_update
    
    logger.debug(f"=== ???憓?撥?豢?瘥???勗? ===")
    
    # ?園????鞈??
    param_info = {
        "?典????憟??": "???" if global_apply else "?芸???,
        "?典?憸券??仿?CAP": global_cap,
        "?典?ATR瘥?潮?瑼?: global_atr,
        "???憸券??仿?CAP": page_cap,
        "???ATR瘥?潮?瑼?: page_atr,
        "?蝯?蝙?每AP": global_cap if global_apply else page_cap,
        "?蝯?蝙?杗TR瘥?潮?瑼?: global_atr if global_apply else page_atr,
        "???撌桃????": "CAP撌桃?={}, ATR撌桃?={}".format(
            abs((global_cap or 0) - (page_cap or 0)), 
            abs((global_atr or 0) - (page_atr or 0))
        )
    }
    
    # ?園??豢?鞈??
    data_info = {}
    
    if cache:
        df_raw = df_from_pack(cache.get("df_raw"))
        daily_state = df_from_pack(cache.get("daily_state"))
        trade_data = df_from_pack(cache.get("trade_data"))
        weight_curve = df_from_pack(cache.get("weight_curve"))
        
        data_info["enhanced_cache"] = {
            "df_raw_shape": df_raw.shape if df_raw is not None else None,
            "daily_state_shape": daily_state.shape if daily_state is not None else None,
            "trade_data_shape": trade_data.shape if trade_data is not None else None,
            "weight_curve_shape": weight_curve.shape if weight_curve is not None else None,
            "daily_state_columns": list(daily_state.columns) if daily_state is not None else None,
            "daily_state_index_range": f"{daily_state.index.min()} ??{daily_state.index.max()}" if daily_state is not None and not daily_state.empty else None
        }
        
        if daily_state is not None and "w" in daily_state.columns:
            data_info["enhanced_cache"]["weight_stats"] = {
                "min": float(daily_state["w"].min()),
                "max": float(daily_state["w"].max()),
                "mean": float(daily_state["w"].mean()),
                "std": float(daily_state["w"].std())
            }
    
    if backtest_data and backtest_data.get("results"):
        results = backtest_data["results"]
        data_info["backtest_store"] = {
            "available_strategies": list(results.keys()),
            "results_count": len(results)
        }
        
        # ?豢?蝚砌?????仿脰?閰喟敦???
        if results:
            first_strategy = list(results.keys())[0]
            result = results[first_strategy]
            
            data_info["backtest_store"]["first_strategy"] = {
                "name": first_strategy,
                "has_daily_state": result.get("daily_state") is not None,
                "has_daily_state_std": result.get("daily_state_std") is not None,
                "has_weight_curve": result.get("weight_curve") is not None,
                "valve_info": result.get("valve", {})
            }
    
    # ????勗?
    report_lines = []
    report_lines.append("=== ?豢?瘥???勗? ===")
    report_lines.append("")
    
    # ????典?
    report_lines.append("?? ???閮剖?:")
    for key, value in param_info.items():
        report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    
    # ?豢??典?
    report_lines.append("?? ?豢????")
    if "enhanced_cache" in data_info:
        report_lines.append("  Enhanced Cache:")
        for key, value in data_info["enhanced_cache"].items():
            report_lines.append(f"    {key}: {value}")
        report_lines.append("")
    
    if "backtest_store" in data_info:
        report_lines.append("  Backtest Store:")
        for key, value in data_info["backtest_store"].items():
            report_lines.append(f"    {key}: {value}")
        report_lines.append("")
    
    # 憓?撥閮箸?撱箄降 (2025/08/20)
    report_lines.append("?? 閰喟敦閮箸?撱箄降:")
    
    # ???銝?湔扳炎??
    if global_apply:
        cap_diff = abs((global_cap or 0) - (page_cap or 0))
        atr_diff = abs((global_atr or 0) - (page_atr or 0))
        if cap_diff > 0.001 or atr_diff > 0.001:
            report_lines.append(f"  ???  ?典?????Ｗ??詨榆?? CAP撌桃?={cap_diff:.4f}, ATR撌桃?={atr_diff:.4f}")
            report_lines.append("      ??撱箄降瑼Ｘ? UI 隞??????詨?甇交???)
        else:
            report_lines.append("  ???典????????Ｗ??訾???)
    else:
        report_lines.append("  ?對?  ?芸??典?撅???憟??嚗?蝙?券??Ｗ???)
        report_lines.append("      ??蝣箄??臬??閬???典?撅憟??隞乩?????湔?)
    
    # ?豢?摰???扳炎??
    enhanced_has_data = "enhanced_cache" in data_info and data_info["enhanced_cache"]["daily_state_shape"]
    backtest_has_data = "backtest_store" in data_info and data_info["backtest_store"]["results_count"] > 0
    
    if enhanced_has_data:
        report_lines.append("  ??Enhanced Cache ?????)
        if "weight_stats" in data_info["enhanced_cache"]:
            ws = data_info["enhanced_cache"]["weight_stats"]
            report_lines.append(f"      甈??蝭??: {ws['min']:.4f} ~ {ws['max']:.4f}, ??? {ws['mean']:.4f}")
    else:
        report_lines.append("  ??Enhanced Cache ?⊥???)
        report_lines.append("      ???航??閬???啣?銵??撘瑕???)
    
    if backtest_has_data:
        report_lines.append("  ??Backtest Store ?????)
    else:
        report_lines.append("  ??Backtest Store ?∠???)
        report_lines.append("      ???航??閬???啣?銵??皜砍???)
    
    # 憸券??仿???摩瑼Ｘ?
    effective_cap = global_cap if global_apply else page_cap
    effective_atr = global_atr if global_apply else page_atr
    
    report_lines.append("  ?? 憸券??仿???蔭:")
    report_lines.append(f"      ???CAP?? {effective_cap}")
    report_lines.append(f"      ???ATR?瑼? {effective_atr}")
    
    if effective_cap and effective_cap < 0.1:
        report_lines.append("      ???  CAP?潮?雿???航??????漲靽??")
    if effective_atr and effective_atr > 3.0:
        report_lines.append("      ???  ATR?瑼駁?擃???航?敺??閫貊?")
    
    # 銝?湔扳炎?亦蜇蝯?
    consistency_issues = []
    if global_apply and (cap_diff > 0.001 or atr_diff > 0.001):
        consistency_issues.append("???銝????)
    if not enhanced_has_data:
        consistency_issues.append("Enhanced Cache蝻箏仃")
    if not backtest_has_data:
        consistency_issues.append("Backtest Store蝻箏仃")
    
    if consistency_issues:
        report_lines.append(f"  ?? ?潛?銝?湔批?憿? {', '.join(consistency_issues)}")
        report_lines.append("      撱箄降?芸?閫?捱??????隞亦Ⅱ靽?????????湔?)
    else:
        report_lines.append("  ???芰??暹?憿臭??湔批?憿?)
    
    # ??? CSV ?豢?
    csv_data = []
    for key, value in param_info.items():
        csv_data.append({"???": key, "?詨?: str(value)})
    
    csv_data.append({"???": "", "?詨?: ""})
    csv_data.append({"???": "=== ?豢????===", "?詨?: ""})
    
    if "enhanced_cache" in data_info:
        for key, value in data_info["enhanced_cache"].items():
            csv_data.append({"???": f"Enhanced_{key}", "?詨?: str(value)})
    
    if "backtest_store" in data_info:
        for key, value in data_info["backtest_store"].items():
            csv_data.append({"???": f"Backtest_{key}", "?詨?: str(value)})
    
    # 餈???勗???CSV 銝??
    report_text = "\n".join(report_lines)
    csv_df = pd.DataFrame(csv_data)
    
    return report_text, dcc.send_data_frame(csv_df.to_csv, "data_comparison_report.csv", index=False)

def _first_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

# --------- 憓?撥??? Callback嚗?漱??甜?餅?閫??靽格迤??? ---------
@app.callback(
    Output("phase-table", "children"),
    Input("run-phase", "n_clicks"),
    State("phase-min-gap", "value"),
    State("phase-cooldown", "value"),
    State("enhanced-trades-cache", "data"),
    State("theme-store", "data"),   # ?交???theme-store嚗????????theme ?賊??舐宏??
    prevent_initial_call=True
)
def _run_phase(n_clicks, min_gap, cooldown, cache, theme):
    import numpy as np
    from urllib.parse import quote as urlparse
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not cache:
        return html.Div("撠??頛????葫蝯??", style={"color": "#ffb703"})

    # 敺?翰????????
    trade_df = df_from_pack(cache.get("trade_data"))
    daily_state = df_from_pack(cache.get("daily_state"))
    
    if trade_df is None or trade_df.empty:
        return "?曆??唬漱?????
    
    if daily_state is None or daily_state.empty:
        return "?曆???daily_state嚗???亥???甈??嚗?
    
    if "equity" not in daily_state.columns:
        return "daily_state 蝻箏?甈??甈?? 'equity'"

    equity = daily_state["equity"]

    # ?澆?雿?歇撖怠末????????
    try:
        from SSS_EnsembleTab import trade_contribution_by_phase
        table = trade_contribution_by_phase(trade_df, equity, min_gap, cooldown).copy()
    except Exception as e:
        return f"鈭斗?鞎Ｙ???圾?瑁?憭望?: {e}"

    if table.empty:
        return "?∟???

    # ?詨?甈??頧??
    num_cols = ["鈭斗?蝑??","鞈???梢?蝮賢?(%)","??挾?冶DD(%)","??挾瘛刻甜??%)"]
    for c in num_cols:
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce")

    # ====== 蝮賡? KPI ======
    avg_net = table["??挾瘛刻甜??%)"].mean() if "??挾瘛刻甜??%)" in table else np.nan
    avg_mdd = table["??挾?冶DD(%)"].mean() if "??挾?冶DD(%)" in table else np.nan
    succ_all = (table["??挾瘛刻甜??%)"] > 0).mean() if "??挾瘛刻甜??%)" in table else np.nan
    succ_acc = np.nan
    if "??挾" in table.columns and "??挾瘛刻甜??%)" in table.columns:
        mask_acc = table["??挾"].astype(str).str.contains("??Ⅳ", na=False)
        if mask_acc.any():
            succ_acc = (table.loc[mask_acc, "??挾瘛刻甜??%)"] > 0).mean()
    risk_eff = np.nan
    if pd.notna(avg_net) and pd.notna(avg_mdd) and avg_mdd != 0:
        risk_eff = avg_net / abs(avg_mdd)

    # ====== CSV ???嚗?策銴?ˊ?剁?DataTable ?行??批遣銝??嚗?=====
    csv_text = table.to_csv(index=False)
    csv_data_url = "data:text/csv;charset=utf-8," + urlparse(csv_text)

    # ====== 銝駁?璅??嚗?????摨??摮??======
    theme = theme or "theme-dark"
    if theme == "theme-dark":
        table_bg = "#1a1a1a"; cell_color = "#ffffff"
        header_bg = "#2a2a2a"; header_color = "#ffffff"; border = "#444444"
        accent_bg = "#243447"; accent_color = "#ffffff"
    elif theme == "theme-light":
        table_bg = "#ffffff"; cell_color = "#111111"
        header_bg = "#f2f2f2"; header_color = "#111111"; border = "#cccccc"
        accent_bg = "#eef2ff"; accent_color = "#111111"
    else:  # theme-blue
        table_bg = "#0b1e3a"; cell_color = "#ffe066"
        header_bg = "#12345b"; header_color = "#ffe066"; border = "#335577"
        accent_bg = "#12345b"; accent_color = "#ffe066"

    style_table = {
        "overflowX": "auto",
        "overflowY": "auto",
        "maxHeight": "70vh",
        "fontSize": "12px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": table_bg,
        "border": f"1px solid {border}",
        # ??迂?詨????銴?ˊ
        "userSelect": "text", "-webkit-user-select": "text",
        "-moz-user-select": "text", "-ms-user-select": "text",
    }
    style_cell = {
        "textAlign": "center",
        "padding": "8px",
        "minWidth": "80px",
        "backgroundColor": table_bg,
        "color": cell_color,
        "border": f"1px solid {border}",
        "whiteSpace": "normal",
        "height": "auto",
    }
    style_header = {
        "backgroundColor": header_bg,
        "color": header_color,
        "fontWeight": "bold",
        "textAlign": "center",
        "borderBottom": f"2px solid {border}",
    }

    # ====== 摰??銵冽? ======
    # 瘜冽?嚗?ull_table 撠?? ordered 霈??摰?儔敺???啣?蝢?

    # ====== ??????KPI + Top3 / Worst3嚗?=====
    def kpi(label, value):
        return html.Div([
            html.Div(label, style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "bold"})
        ], style={
            "backgroundColor": accent_bg, "color": accent_color,
            "padding": "10px 14px", "borderRadius": "12px", "minWidth": "160px"
        })

    kpi_bar = html.Div([
        kpi("撟喳?瘥?挾瘛刻甜??%)", f"{avg_net:.2f}" if pd.notna(avg_net) else "??),
        kpi("撟喳?瘥?挾 MDD(%)", f"{avg_mdd:.2f}" if pd.notna(avg_mdd) else "??),
        kpi("??????券?)", f"{succ_all*100:.1f}%" if pd.notna(succ_all) else "??),
        kpi("???????Ⅳ)", f"{succ_acc*100:.1f}%" if pd.notna(succ_acc) else "??),
        kpi("憸券????", f"{risk_eff:.3f}" if pd.notna(risk_eff) else "??),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"10px"})

    # ====== ??? KPI嚗??蝣?vs 皜?Ⅳ ======
    def _group_metrics(mask):
        if {"??挾瘛刻甜??%)","??挾?冶DD(%)"}.issubset(table.columns):
            sub = table.loc[mask]
            if sub.empty:
                return None
            a_net = sub["??挾瘛刻甜??%)"].mean()
            a_mdd = sub["??挾?冶DD(%)"].mean()
            succ  = (sub["??挾瘛刻甜??%)"] > 0).mean()
            eff   = (a_net / abs(a_mdd)) if pd.notna(a_net) and pd.notna(a_mdd) and a_mdd != 0 else np.nan
            return {"count": int(len(sub)), "avg_net": a_net, "avg_mdd": a_mdd, "succ": succ, "eff": eff}
        return None

    def _fmt(val, pct=False, dec=2):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "??
        return f"{val*100:.1f}%" if pct else f"{val:.{dec}f}"

    def group_row(title, m):
        return html.Div([
            html.Div(title, style={"fontWeight":"bold","marginRight":"12px","minWidth":"72px","alignSelf":"center"}),
            kpi("畾菜?", f"{m['count']}" if m else "??),
            kpi("撟喳?瘛刻甜??%)", _fmt(m['avg_net']) if m else "??),
            kpi("撟喳?MDD(%)",   _fmt(m['avg_mdd']) if m else "??),
            kpi("?????,        _fmt(m['succ'], pct=True) if m else "??),
            kpi("憸券????",      _fmt(m['eff'],  dec=3) if m else "??),
        ], style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"8px"})

    acc_metrics = dis_metrics = None
    if "??挾" in table.columns:
        mask_acc = table["??挾"].astype(str).str.contains("??Ⅳ", na=False)
        mask_dis = table["??挾"].astype(str).str.contains("皜?Ⅳ", na=False)
        acc_metrics = _group_metrics(mask_acc)
        dis_metrics = _group_metrics(mask_dis)

    group_section = html.Div([
        html.H6("??? KPI嚗??蝣?vs 皜?Ⅳ嚗?, style={"margin":"8px 0 6px 0"}),
        group_row("??Ⅳ畾?, acc_metrics),
        group_row("皜?Ⅳ畾?, dis_metrics),
    ], style={"marginTop":"4px"})

    # ====== Top/Worst 靘?????嚗????/ ?芸?蝣?/ ?芣?蝣潘? ======
    source_selector = html.Div([
        html.Div("Top/Worst 靘??", style={"marginRight":"8px", "alignSelf":"center"}),
        dcc.RadioItems(
            id="phase-source",
            options=[
                {"label": "?券?",   "value": "all"},
                {"label": "??Ⅳ畾?, "value": "acc"},
                {"label": "皜?Ⅳ畾?, "value": "dis"},
            ],
            value="all",
            inline=True,
            inputStyle={"marginRight":"4px"},
            labelStyle={"marginRight":"12px"}
        )
    ], style={"display":"flex","gap":"6px","alignItems":"center","margin":"6px 0 8px 0"})

    # 甈?????嚗???渲” & Top/Worst ?梁?嚗?
    ordered = [c for c in ["??挾","????交?","蝯???交?","鈭斗?蝑??",
                           "??挾瘛刻甜??%)","鞈???梢?蝮賢?(%)","??挾?冶DD(%)","?臬????"] if c in table.columns]
    basis_col = "??挾瘛刻甜??%)" if "??挾瘛刻甜??%)" in table.columns else "鞈???梢?蝮賢?(%)"

    # ====== 摰??銵冽? ======
    full_table = dash_table.DataTable(
        id="phase-datatable",
        columns=[{"name": c, "id": c, "type": ("numeric" if c in num_cols else "text")} for c in ordered],
        data=table[ordered].to_dict("records"),
        # ???
        page_action="native",
        page_current=0,
        page_size=100,            # ??身瘥?? 100嚗??閬???臬???ㄐ
        # 鈭??
        sort_action="native",
        filter_action="native",
        # 銝??
        export_format="csv",
        export_headers="display",
        # 銴?ˊ
        cell_selectable=True,
        virtualization=False,     # ??????????踹?銴?ˊ???銴???航??
        fixed_rows={"headers": True},
        style_table=style_table,
        style_cell=style_cell,
        style_header=style_header,
        css=[{
            "selector": ".dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner *",
            "rule": "user-select: text; -webkit-user-select: text; -moz-user-select: text; -ms-user-select: text;"
        }],
    )

    # ====== dcc.Store嚗??靘?Top/Worst ??? callback 雿輻? ======
    store = dcc.Store(id="phase-table-store", data={
        "records": table[ordered].to_dict("records"),
        "ordered": ordered,
        "basis": basis_col,
        "has_stage": "??挾" in table.columns
    })

    # ??身嚗???其?皞?????銝甈∴??踹?蝛箇???
    def _subset(src):
        df = table
        if "??挾" not in df.columns:
            return df
        if src == "acc":
            return df[df["??挾"].astype(str).str.contains("??Ⅳ", na=False)]
        if src == "dis":
            return df[df["??挾"].astype(str).str.contains("皜?Ⅳ", na=False)]
        return df
    base = _subset("all")
    top3   = base.nlargest(3, basis_col) if basis_col in base else base.head(3)
    worst3 = base.nsmallest(3, basis_col) if basis_col in base else base.tail(3)

    def simple_table(df, tbl_id):
        return dash_table.DataTable(
            id=tbl_id,
            columns=[{"name": c, "id": c} for c in ordered],
            data=df[ordered].to_dict("records"),
            page_action="none",
            style_table=style_table, style_cell=style_cell, style_header=style_header
        )

    top3_table = simple_table(top3, "phase-top-table")
    worst3_table = simple_table(worst3, "phase-worst-table")

    # ====== Copy / Download 撌亙???======
    tools = html.Div([
        html.Button("銴?ˊ?券?嚗?SV嚗?, id="phase-copy-btn",
                    style={"padding": "6px 10px", "borderRadius": "8px", "cursor": "pointer"}),
        dcc.Clipboard(target_id="phase-csv-text", title="Copy", style={"marginLeft": "6px"}),
        html.A("銝?? CSV", href=csv_data_url, download="trade_contribution.csv",
               style={"marginLeft": "12px", "textDecoration": "none"})
    ], style={"display": "flex", "alignItems": "center", "gap": "4px", "marginBottom": "8px"})

    # ?梯???CSV ???靘??嚗?策 Clipboard ?剁?
    csv_hidden = html.Pre(id="phase-csv-text", children=csv_text, style={"display": "none"})

    # ====== Tabs嚗??霈??/ 摰??銵冽? ======
    tabs = dcc.Tabs(id="phase-tabs", value="summary", children=[
        dcc.Tab(label="?????, value="summary", children=[
            kpi_bar,
            group_section,
            source_selector,
            html.H6("?鞈箇? 3 畾蛛?靘??皞?????甈??", style={"marginTop":"8px"}),
            top3_table,
            html.H6("??抒? 3 畾蛛?靘??皞?????甈??", style={"marginTop":"16px"}),
            worst3_table
        ]),
        dcc.Tab(label="摰??銵冽?", value="full", children=[full_table]),
    ])

    return html.Div([tools, csv_hidden, store, tabs], style={"marginTop": "8px"})

# --------- ?寥?皜祈岫???蝭?? Callback ---------
@app.callback(
    Output("batch-phase-results", "children"),
    Input("run-batch-phase", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    prevent_initial_call=True
)
def _run_batch_phase_test(n_clicks, cache):
    """?寥?皜祈岫1-24蝭?????撠??頝???瑕??????""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not cache:
        return html.Div("撠??頛????葫蝯??", style={"color": "#ffb703"})

    # 敺?翰????????
    trade_df = df_from_pack(cache.get("trade_data"))
    daily_state = df_from_pack(cache.get("daily_state"))
    
    if trade_df is None or trade_df.empty:
        return "?曆??唬漱?????
    
    if daily_state is None or daily_state.empty:
        return "?曆???daily_state嚗???亥???甈??嚗?
    
    if "equity" not in daily_state.columns:
        return "daily_state 蝻箏?甈??甈?? 'equity'"

    equity = daily_state["equity"]
    
    # 瑼Ｘ?銝行???漱??????撘?
    debug_info = []
    debug_info.append(f"???鈭斗?鞈??甈??: {list(trade_df.columns)}")
    debug_info.append(f"鈭斗?鞈??銵??: {len(trade_df)}")
    debug_info.append(f"甈??鞈??銵??: {len(equity)}")
    
    # 瑼Ｘ?敹??甈??銝阡脰?頧??
    required_mappings = {
        "date": ["date", "trade_date", "鈭斗??交?", "Date"],
        "type": ["type", "鈭斗?憿??", "action", "side", "Type"],
        "w_before": ["w_before", "鈭斗??????, "weight_before", "weight_prev"],
        "w_after": ["w_after", "鈭斗?敺????, "weight_after", "weight_next"]
    }
    
    # 撠??撠?????雿?
    found_columns = {}
    for target, possible_names in required_mappings.items():
        for name in possible_names:
            if name in trade_df.columns:
                found_columns[target] = name
                break
    
    debug_info.append(f"?曉????雿???? {found_columns}")
    
    # 憒??蝻箏?敹??甈??嚗??閰血?撱?
    if len(found_columns) < 4:
        debug_info.append("蝻箏?敹??甈??嚗??閰血?撱?..")
        
        # ??岫敺?????雿??撠?
        if "weight_change" in trade_df.columns and "w_before" not in found_columns:
            # 憒?????????????岫??遣???甈??
            trade_df = trade_df.copy()
            trade_df["w_before"] = 0.0
            trade_df["w_after"] = trade_df["weight_change"]
            found_columns["w_before"] = "w_before"
            found_columns["w_after"] = "w_after"
            debug_info.append("敺?weight_change ?萄遣 w_before ??w_after")
        
        if "price" in trade_df.columns and "type" not in found_columns:
            # 憒??????潘???身?箄眺??
            trade_df["type"] = "buy"
            found_columns["type"] = "type"
            debug_info.append("?萄遣 type 甈??嚗??閮剔? buy")
    
    # ?寥?皜祈岫???蝭?? 1-24
    results = []
    total_combinations = 24 * 24  # 576蝔桃???
    
    try:
        from SSS_EnsembleTab import trade_contribution_by_phase
        
        # ?脣漲憿舐內
        progress_div = html.Div([
            html.H6("甇???瑁??寥?皜祈岫...", style={"color": "#28a745"}),
            html.Div(f"皜祈岫蝭??嚗??撠??頝?1-24 憭抬??瑕???1-24 憭?, style={"fontSize": "12px", "color": "#666"}),
            html.Div(f"蝮賜????嚗?total_combinations}", style={"fontSize": "12px", "color": "#666"}),
            html.Div(id="batch-progress", children="???皜祈岫...")
        ])
        
        # ?瑁??寥?皜祈岫
        batch_results = []
        debug_info = []
        
        # ??葫閰虫???陛?桃?獢??
        test_min_gap, test_cooldown = 1, 1
        try:
            debug_info.append(f"???皜祈岫?桐?獢??: min_gap={test_min_gap}, cooldown={test_cooldown}")
            
            # 瑼Ｘ?鈭斗?鞈????????雿?
            if "weight_change" in trade_df.columns:
                debug_info.append(f"?曉? weight_change 甈??嚗???? {trade_df['weight_change'].min():.4f} ~ {trade_df['weight_change'].max():.4f}")
            
            # 瑼Ｘ?甈??鞈??
            if len(equity) > 0:
                debug_info.append(f"甈??鞈??蝭??: {equity.min():.2f} ~ {equity.max():.2f}")
            
            table = trade_contribution_by_phase(trade_df, equity, test_min_gap, test_cooldown)
            debug_info.append(f"?賣??瑁????嚗????”?澆之撠? {table.shape}")
            debug_info.append(f"銵冽?甈??: {list(table.columns)}")
            
            if not table.empty:
                debug_info.append(f"蝚砌?銵???? {table.iloc[0].to_dict()}")
                
                # 瑼Ｘ??臬????畾菜楊鞎Ｙ?甈??
                if "??挾瘛刻甜??%)" in table.columns:
                    debug_info.append(f"??挾瘛刻甜?餅?雿???剁???征?潭??? {table['??挾瘛刻甜??%)'].notna().sum()}")
                    debug_info.append(f"??挾瘛刻甜?餌??? {table['??挾瘛刻甜??%)'].min():.2f} ~ {table['??挾瘛刻甜??%)'].max():.2f}")
                else:
                    debug_info.append("蝻箏???挾瘛刻甜?餅?雿?)
                
                if "??挾?冶DD(%)" in table.columns:
                    debug_info.append(f"??挾?冶DD甈??摮??嚗??蝛箏潭??? {table['??挾?冶DD(%)'].notna().sum()}")
                    debug_info.append(f"??挾?冶DD蝭??: {table['??挾?冶DD(%)'].min():.2f} ~ {table['??挾?冶DD(%)'].max():.2f}")
                else:
                    debug_info.append("蝻箏???挾?冶DD甈??")
            else:
                debug_info.append("?賣?餈??蝛箄”??)
                
        except Exception as e:
            import traceback
            debug_info.append(f"?賣??瑁??航炊: {str(e)}")
            debug_info.append(f"?航炊閰單?: {traceback.format_exc()}")
        
        # 憒???桐?皜祈岫???嚗?匱蝥????葫閰?
        if not table.empty and "??挾瘛刻甜??%)" in table.columns and "??挾?冶DD(%)" in table.columns:
            debug_info.append("?桐?皜祈岫???嚗??憪????葫閰?..")
            
            for min_gap in range(1, 25):
                for cooldown in range(1, 25):
                    try:
                        table = trade_contribution_by_phase(trade_df, equity, min_gap, cooldown)
                        
                        if not table.empty:
                            # ??蕪???閬??嚗??虜???"蝯梯????"摮?見嚗?
                            data_rows = table[~table["??挾"].astype(str).str.contains("蝯梯????", na=False)]
                            
                            if len(data_rows) == 0:
                                continue
                            
                            # 閮????????
                            avg_net = data_rows["??挾瘛刻甜??%)"].mean()
                            avg_mdd = data_rows["??挾?冶DD(%)"].mean()
                            succ_rate = (data_rows["??挾瘛刻甜??%)"] > 0).mean()
                            risk_eff = avg_net / abs(avg_mdd) if avg_mdd != 0 else 0
                            
                            batch_results.append({
                                "?撠??頝?: min_gap,
                                "?瑕???: cooldown,
                                "撟喳?瘛刻甜??%)": round(avg_net, 2),
                                "撟喳?MDD(%)": round(avg_mdd, 2),
                                "?????%)": round(succ_rate * 100, 1),
                                "憸券????": round(risk_eff, 3),
                                "??挾??: len(data_rows)
                            })
                    except Exception as e:
                        # 閮???航炊雿?匱蝥??銵?
                        continue
        else:
            debug_info.append("?桐?皜祈岫憭望?嚗?歲?????葫閰?)
        
        if not batch_results:
            # 憿舐內?日?鞈??
            debug_html = html.Div([
                html.H6("?日?鞈??", style={"color": "#dc3545", "marginTop": "16px"}),
                html.Div([html.Pre(info) for info in debug_info], style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "4px", "fontSize": "11px"})
            ])
            
            return html.Div([
                html.Div("?寥?皜祈岫摰??嚗???⊥??????, style={"color": "#ffb703"}),
                html.Div("?航????嚗?, style={"marginTop": "8px", "color": "#666"}),
                html.Ul([
                    html.Li("鈭斗?鞈???澆?銝?迤蝣?),
                    html.Li("蝻箏?敹?????雿????挾瘛刻甜??%)???畾萄?MDD(%)嚗?),
                    html.Li("?????貊?????⊥??Ｙ??????挾"),
                    html.Li("?賣??瑁???????隤?)
                ], style={"fontSize": "12px", "color": "#666"}),
                debug_html
            ])
        
        # 頧???慣ataFrame銝行?摨?
        results_df = pd.DataFrame(batch_results)
        
        # ??◢?芣????摨?????嚗?
        results_df = results_df.sort_values("憸券????", ascending=False)
        
        # ???CSV銝?????
        csv_text = results_df.to_csv(index=False)
        csv_data_url = "data:text/csv;charset=utf-8," + urlparse(csv_text)
        
        # 憿舐內??0?????
        top10 = results_df.head(10)
        
        # ???蝯??銵冽?
        results_table = dash_table.DataTable(
            id="batch-results-table",
            columns=[{"name": c, "id": c} for c in results_df.columns],
            data=top10.to_dict("records"),
            page_action="none",
            style_table={"overflowX": "auto", "fontSize": "11px"},
            style_cell={"textAlign": "center", "padding": "4px", "minWidth": "60px"},
            style_header={"backgroundColor": "#28a745", "color": "white", "fontWeight": "bold"}
        )
        
        # 蝯梯????
        summary_stats = html.Div([
            html.H6("?寥?皜祈岫???", style={"marginTop": "16px", "marginBottom": "8px", "color": "#28a745"}),
            html.Div(f"???蝯???賂?{len(results_df)} / {total_combinations}", style={"fontSize": "12px"}),
            html.Div(f"?雿喲◢?芣????{results_df['憸券????'].max():.3f}", style={"fontSize": "12px"}),
            html.Div(f"?雿喳像??楊鞎Ｙ?嚗?results_df['撟喳?瘛刻甜??%)'].max():.2f}%", style={"fontSize": "12px"}),
            html.Div(f"?雿單????嚗?results_df['?????%)'].max():.1f}%", style={"fontSize": "12px"}),
            html.Div([
                html.Button("銝??摰??蝯??CSV", id="download-batch-csv", 
                           style={"backgroundColor": "#28a745", "color": "white", "border": "none", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}),
                html.A("?湔?銝??", href=csv_data_url, download="batch_phase_test_results.csv",
                       style={"marginLeft": "12px", "textDecoration": "none", "color": "#28a745"})
            ], style={"marginTop": "8px"})
        ])
        
        return html.Div([
            summary_stats,
            html.H6("??0???雿喳??貊??????◢?芣????摨??", style={"marginTop": "16px", "marginBottom": "8px"}),
            results_table
        ])
        
    except Exception as e:
        return html.Div(f"?寥?皜祈岫?瑁?憭望?: {str(e)}", style={"color": "#dc3545"})

# --- Gate analysis buttons until cache is ready ---
@app.callback(
    Output("run-rv", "disabled"),
    Output("run-phase", "disabled"),
    Output("run-batch-phase", "disabled"),
    Input("enhanced-trades-cache", "data"),
    prevent_initial_call=False
)
def _gate_analyze_buttons(cache):
    ready = bool(cache) and (
        (cache.get("trade_data") or cache.get("trade_df") or cache.get("trade_ledger") or cache.get("trade_ledger_std"))
        and (cache.get("daily_state") or cache.get("daily_state_std"))
    )
    disabled = not ready
    return disabled, disabled, disabled

# --------- 憓?撥??? Callback A嚗?? backtest-store 憛急遛蝑???詨? ---------
@app.callback(
    Output("enhanced-strategy-selector", "options"),
    Output("enhanced-strategy-selector", "value"),
    Input("backtest-store", "data"),
    prevent_initial_call=False
)
def _populate_enhanced_strategy_selector(bstore):
    """靘?backtest-store 憛急遛蝑???詨?嚗?蒂?芸??豢??雿喟???""
    if not bstore:
        return [], None
    if isinstance(bstore, dict) and bstore.get("error"):
        logger.warning(f"enhanced selector: backtest-store error={bstore.get('error')}")
        return [], None
    
    results = bstore.get("results", {})
    if not isinstance(results, dict) or not results:
        return [], None
    
    # 蝑??閰??嚗?edger_std > ledger > trade_df
    strategy_scores = []
    for strategy_name, result in results.items():
        score = 0
        if result.get("trade_ledger_std"):
            score += 100  # ?擃??嚗??皞??鈭斗?瘚?偌撣?
        elif result.get("trade_ledger"):
            score += 50   # 銝剖?嚗??憪?漱???瘞游董
        elif result.get("trade_df"):
            score += 10   # 雿??嚗?漱???蝝?
        
        # 憿?????嚗?? daily_state
        if result.get("daily_state") or result.get("daily_state_std"):
            score += 20
        
        strategy_scores.append((strategy_name, score))
    
    # ????豢?摨?
    strategy_scores.sort(key=lambda x: x[1], reverse=True)
    
    # ????詨??賊?
    options = [{"label": f"{name} (???: {score})", "value": name} 
               for name, score in strategy_scores]
    
    # ?芸??豢??擃??蝑??
    auto_select = strategy_scores[0][0] if strategy_scores else None
    
    return options, auto_select

# --------- 憓?撥??? Callback B嚗???仿?摰???亙? enhanced-trades-cache ---------
@app.callback(
    Output("enhanced-trades-cache", "data"),
    Output("enhanced-load-status", "children"),
    Input("load-enhanced-strategy", "n_clicks"),
    State("enhanced-strategy-selector", "value"),
    State("backtest-store", "data"),
    prevent_initial_call=True
)
def _load_enhanced_strategy_to_cache(n_clicks, selected_strategy, bstore):
    """頛???詨?蝑?????皜祉???? enhanced-trades-cache"""
    if not n_clicks or not selected_strategy or not bstore:
        return no_update, "隢??????乩蒂暺??頛??"
    if isinstance(bstore, dict) and bstore.get("error"):
        err = bstore.get("error")
        return no_update, f"??葫憭望?嚗?err}"
    
    results = bstore.get("results", {})
    if not isinstance(results, dict) or not results:
        return no_update, "??葫蝯???箇征嚗?????銵??皜研?
    if selected_strategy not in results:
        return no_update, f"?曆??啁??伐?{selected_strategy}"
    
    result = results[selected_strategy]
    
    # ?芸????嚗?edger_std > ledger > trade_df
    trade_data = None
    data_source = ""
    
    if result.get("trade_ledger_std"):
        trade_data = df_from_pack(result["trade_ledger_std"])
        data_source = "trade_ledger_std (璅????"
    elif result.get("trade_ledger"):
        trade_data = df_from_pack(result["trade_ledger"])
        data_source = "trade_ledger (???)"
    elif result.get("trade_df"):
        trade_data = df_from_pack(result["trade_df"])
        data_source = "trade_df (鈭斗???敦)"
    else:
        return no_update, "閰脩??亦?鈭斗?鞈??"
    
    # 璅????漱?????
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_data = norm(trade_data)
    except Exception:
        # 敺??璅?????獢?
        if trade_data is not None and len(trade_data) > 0:
            trade_data = trade_data.copy()
            trade_data.columns = [str(c).lower() for c in trade_data.columns]
            
            # 蝣箔???trade_date 甈?
            if "trade_date" not in trade_data.columns:
                if "date" in trade_data.columns:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
                elif isinstance(trade_data.index, pd.DatetimeIndex):
                    trade_data = trade_data.reset_index().rename(columns={"index": "trade_date"})
                else:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
            
            # 蝣箔???type 甈?
            if "type" not in trade_data.columns:
                if "action" in trade_data.columns:
                    trade_data["type"] = trade_data["action"].astype(str).str.lower()
                elif "side" in trade_data.columns:
                    trade_data["type"] = trade_data["side"].astype(str).str.lower()
                else:
                    trade_data["type"] = "hold"
            
            # 蝣箔???price 甈?
            if "price" not in trade_data.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_data.columns:
                        trade_data["price"] = trade_data[c]
                        break
                if "price" not in trade_data.columns:
                    trade_data["price"] = 0.0
    
    # 皞?? daily_state - ?亙歇憟???仿??????蝙?刻矽?游?鞈??
    daily_state = None
    valve_info = result.get("valve", {})
    valve_on = bool(valve_info.get("applied", False))
    
    # app_dash.py / 2025-08-22 15:30
    # ?箄??豢??亦?鞈??嚗????蝙??valve ???嚗??????其?摮??嚗???血?雿輻? baseline
    if valve_on and result.get("daily_state_valve"):
        daily_state = df_from_pack(result["daily_state_valve"])
        data_source = f"{data_source} (valve)"
    elif result.get("daily_state_std"):
        daily_state = df_from_pack(result["daily_state_std"])
        data_source = f"{data_source} (std)"
    elif result.get("daily_state"):
        daily_state = df_from_pack(result["daily_state"])
        data_source = f"{data_source} (original)"
    elif result.get("daily_state_base"):
        daily_state = df_from_pack(result["daily_state_base"])
        data_source = f"{data_source} (baseline)"
    else:
        daily_state = None
    
    # app_dash.py / 2025-08-22 16:00
    # ?詨捆?改??芸?雿輻? valve 甈???脩?嚗????????祆?雿????O2 銝?湛?
    weight_curve = None
    if result.get("weight_curve_valve"):
        weight_curve = df_from_pack(result["weight_curve_valve"])
    elif result.get("weight_curve"):
        weight_curve = df_from_pack(result["weight_curve"])
    elif result.get("weight_curve_base"):
        weight_curve = df_from_pack(result["weight_curve_base"])
    
    # ?脣??仿?????閮?
    valve_info = result.get("valve", {})  # {"applied": bool, "cap": float, "atr_ratio": float or "N/A"}
    valve_on = bool(valve_info.get("applied", False))
    
    # ?仿?????嚗??霅????垢閬?神 w_series
    if valve_on and weight_curve is not None and daily_state is not None:
        ds = daily_state.copy()
        wc = weight_curve.copy()
        # 撠?????蝝Ｗ?嚗?? ds ??'trade_date' 甈?停 merge嚗????誑蝝Ｗ?撠??
        if "trade_date" in ds.columns:
            ds["trade_date"] = pd.to_datetime(ds["trade_date"])
            wc = wc.rename("w").to_frame().reset_index().rename(columns={"index": "trade_date"})
            ds = ds.merge(wc, on="trade_date", how="left")
        else:
            # 隞亦揣撘??朣?
            ds.index = pd.to_datetime(ds.index)
            wc.index = pd.to_datetime(wc.index)
            # 靽格迤嚗?Ⅱ靽?wc ??Series 銝虫?甇?Ⅱ撠??
            if isinstance(wc, pd.DataFrame):
                if "w" in wc.columns:
                    wc_series = wc["w"]
                else:
                    wc_series = wc.iloc[:, 0]  # ??洵銝??
            else:
                wc_series = wc
            ds["w"] = wc_series.reindex(ds.index).ffill().bfill()
        daily_state = ds
    
    # 皞?? df_raw
    df_raw = df_from_pack(bstore.get("df_raw")) if bstore.get("df_raw") else pd.DataFrame()
    
    # ---- pack valve flags into cache ----
    cache_data = {
        "strategy": selected_strategy,
        "trade_data": pack_df(trade_data) if trade_data is not None else None,
        "daily_state": pack_df(daily_state) if daily_state is not None else None,
        "weight_curve": pack_df(weight_curve) if weight_curve is not None else None,
        "df_raw": pack_df(df_raw) if df_raw is not None else None,
        "valve": valve_info,
        "valve_applied": valve_on,
        "ensemble_params": result.get("ensemble_params", {}),
        "data_source": data_source,
        "timestamp": datetime.now().isoformat(),
        # ???啣?嚗?aseline ??valve ???銝雿菜??脣翰??
        "daily_state_base": result.get("daily_state_base"),
        "weight_curve_base": result.get("weight_curve_base"),
        "trade_ledger_base": result.get("trade_ledger_base"),
        # ???啣?嚗?alve ???銝雿菜??脣翰??
        "daily_state_valve": result.get("daily_state_valve"),
        "weight_curve_valve": result.get("weight_curve_valve"),
        "trade_ledger_valve": result.get("trade_ledger_valve"),
        "equity_curve_valve": result.get("equity_curve_valve"),
    }
    
    status_msg = f"??撌脰???{selected_strategy} ({data_source})"
    if daily_state is not None:
        status_msg += f"嚗????{len(daily_state)} 蝑??蝺????
    if trade_data is not None:
        status_msg += f"嚗????{len(trade_data)} 蝑?漱??
    
    return cache_data, status_msg

# --------- 憓?撥??? Callback C嚗????翰???雿喟???---------
@app.callback(
    Output("enhanced-trades-cache", "data", allow_duplicate=True),
    Output("enhanced-load-status", "children", allow_duplicate=True),
    Input("backtest-store", "data"),
    State("enhanced-strategy-selector", "value"),
    prevent_initial_call='initial_duplicate'
)
def _auto_cache_best_strategy(bstore, current_selection):
    """??葫摰??敺????翰???雿喟???""
    if not bstore:
        return no_update, no_update
    if isinstance(bstore, dict) and bstore.get("error"):
        err = bstore.get("error")
        logger.warning(f"enhanced auto-cache skipped due backtest error: {err}")
        return no_update, no_update
    
    results = bstore.get("results", {})
    if not isinstance(results, dict) or not results:
        return no_update, no_update
    
    # 憒??撌脩??????????銝????
    if current_selection:
        return no_update, no_update
    
    # 蝑??閰??嚗?edger_std > ledger > trade_df
    strategy_scores = []
    for strategy_name, result in results.items():
        score = 0
        if result.get("trade_ledger_std"):
            score += 100  # ?擃??嚗??皞??鈭斗?瘚?偌撣?
        elif result.get("trade_ledger"):
            score += 50   # 銝剖?嚗??憪?漱???瘞游董
        elif result.get("trade_df"):
            score += 10   # 雿??嚗?漱???蝝?
        
        # 憿?????嚗?? daily_state
        if result.get("daily_state") or result.get("daily_state_std"):
            score += 20
        
        strategy_scores.append((strategy_name, score))
    
    # ????豢?摨???豢??雿喟???
    if not strategy_scores:
        return no_update, no_update
    
    strategy_scores.sort(key=lambda x: x[1], reverse=True)
    best_strategy = strategy_scores[0][0]
    best_result = results[best_strategy]
    
    # 皞??鈭斗?鞈??嚗?????摨??ledger_std > ledger > trade_df嚗?
    trade_data = None
    data_source = ""
    
    if best_result.get("trade_ledger_std"):
        trade_data = df_from_pack(best_result["trade_ledger_std"])
        data_source = "trade_ledger_std (璅????"
    elif best_result.get("trade_ledger"):
        trade_data = df_from_pack(best_result["trade_ledger"])
        data_source = "trade_ledger (???)"
    elif best_result.get("trade_df"):
        trade_data = df_from_pack(best_result["trade_df"])
        data_source = "trade_df (鈭斗???敦)"
    else:
        return no_update, no_update
    
    # 璅????漱?????
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_data = norm(trade_data)
    except Exception:
        # 敺??璅?????獢?
        if trade_data is not None and len(trade_data) > 0:
            trade_data = trade_data.copy()
            trade_data.columns = [str(c).lower() for c in trade_data.columns]
            
            # 蝣箔???trade_date 甈?
            if "trade_date" not in trade_data.columns:
                if "date" in trade_data.columns:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
                elif isinstance(trade_data.index, pd.DatetimeIndex):
                    trade_data = trade_data.reset_index().rename(columns={"index": "trade_date"})
                else:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
            
            # 蝣箔???type 甈?
            if "type" not in trade_data.columns:
                if "action" in trade_data.columns:
                    trade_data["type"] = trade_data["action"].astype(str).str.lower()
                elif "side" in trade_data.columns:
                    trade_data["type"] = trade_data["side"].astype(str).str.lower()
                else:
                    trade_data["type"] = "hold"
            
            # 蝣箔???price 甈?
            if "price" not in trade_data.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_data.columns:
                        trade_data["price"] = trade_data[c]
                        break
                if "price" not in trade_data.columns:
                    trade_data["price"] = 0.0
    
    # ---- choose daily_state consistently ----
    valve_info = best_result.get("valve", {}) or {}
    valve_on = bool(valve_info.get("applied", False))
    
    daily_state = None
    # app_dash.py / 2025-08-22 15:30
    # ?箄??豢??亦?鞈??嚗????蝙??valve ???嚗??????其?摮??嚗???血?雿輻? baseline
    if valve_on and best_result.get("daily_state_valve"):
        daily_state = df_from_pack(best_result["daily_state_valve"])
        data_source = f"{data_source} (valve)"
    elif best_result.get("daily_state_std"):
        daily_state = df_from_pack(best_result["daily_state_std"])
        data_source = f"{data_source} (std)"
    elif best_result.get("daily_state"):
        daily_state = df_from_pack(best_result["daily_state"])
        data_source = f"{data_source} (original)"
    elif best_result.get("daily_state_base"):
        daily_state = df_from_pack(best_result["daily_state_base"])
        data_source = f"{data_source} (baseline)"
    
    # app_dash.py / 2025-08-22 16:00
    # ?詨捆?改??芸?雿輻? valve 甈???脩?嚗????????祆?雿????O2 銝?湛?
    weight_curve = None
    if best_result.get("weight_curve_valve"):
        weight_curve = df_from_pack(best_result["weight_curve_valve"])
    elif best_result.get("weight_curve"):
        weight_curve = df_from_pack(best_result["weight_curve"])
    elif best_result.get("weight_curve_base"):
        weight_curve = df_from_pack(best_result["weight_curve_base"])
    
    # ?仿?????嚗??霅????垢閬?神 w_series
    if valve_on and weight_curve is not None and daily_state is not None:
        ds = daily_state.copy()
        wc = weight_curve.copy()
        # 撠?????蝝Ｗ?嚗?? ds ??'trade_date' 甈?停 merge嚗????誑蝝Ｗ?撠??
        if "trade_date" in ds.columns:
            ds["trade_date"] = pd.to_datetime(ds["trade_date"])
            wc = wc.rename("w").to_frame().reset_index().rename(columns={"index": "trade_date"})
            ds = ds.merge(wc, on="trade_date", how="left")
        else:
            # 隞亦揣撘??朣?
            ds.index = pd.to_datetime(ds.index)
            wc.index = pd.to_datetime(wc.index)
            # 靽格迤嚗?Ⅱ靽?wc ??Series 銝虫?甇?Ⅱ撠??
            if isinstance(wc, pd.DataFrame):
                if "w" in wc.columns:
                    wc_series = wc["w"]
                else:
                    wc_series = wc.iloc[:, 0]  # ??洵銝??
            else:
                wc_series = wc
            ds["w"] = wc_series.reindex(ds.index).ffill().bfill()
        daily_state = ds
    
    # 皞?? df_raw
    df_raw = df_from_pack(bstore.get("df_raw")) if bstore.get("df_raw") else pd.DataFrame()
    
    # ---- pack valve flags into cache ----
    cache_data = {
        "strategy": best_strategy,
        "trade_data": pack_df(trade_data) if trade_data is not None else None,
        "daily_state": pack_df(daily_state) if daily_state is not None else None,
        "weight_curve": pack_df(weight_curve) if weight_curve is not None else None,
        "df_raw": pack_df(df_raw) if df_raw is not None else None,
        "valve": valve_info,
        "valve_applied": valve_on,
        "ensemble_params": best_result.get("ensemble_params", {}),
        "data_source": data_source,
        "timestamp": datetime.now().isoformat(),
        "auto_cached": True,
        # ???啣?嚗?aseline ???銝雿菜??脣翰??
        "daily_state_base": best_result.get("daily_state_base"),
        "weight_curve_base": best_result.get("weight_curve_base"),
        "trade_ledger_base": best_result.get("trade_ledger_base"),
        # ???啣?嚗?alve ???銝雿菜??脣翰??
        "daily_state_valve": best_result.get("daily_state_valve"),
        "weight_curve_valve": best_result.get("weight_curve_valve"),
        "trade_ledger_valve": best_result.get("trade_ledger_valve"),
        "equity_curve_valve": best_result.get("equity_curve_valve"),
    }
    
    status_msg = f"?? ?芸?敹怠??雿喟??伐?{best_strategy} ({data_source})"
    if daily_state is not None:
        status_msg += f"嚗????{len(daily_state)} 蝑??蝺????
    if trade_data is not None:
        status_msg += f"嚗????{len(trade_data)} 蝑?漱??
    
    return cache_data, status_msg

# --------- ?啣?嚗?◢???梢??啣?嚗?areto Map嚗?allback ---------
@app.callback(
    Output("pareto-map-graph", "figure"),
    Output("pareto-map-status", "children"),
    Input("generate-pareto-map", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    State("backtest-store", "data"),
    State("rv-mode", "value"),
    State("risk-cap-input", "value"),
    State("atr-ratio-threshold", "value"),
    prevent_initial_call=True
)
def generate_pareto_map(n_clicks, cache, backtest_data, rv_mode, risk_cap_value, atr_ratio_value):
    """???憸券?-?梢??啣?嚗?areto Map嚗????? cap ??ATR(20)/ATR(60) 瘥?澆?蝯??"""
    logger.debug(f"=== Pareto Map ?????? ===")
    logger.debug(f"n_clicks: {n_clicks}")
    logger.debug(f"cache 摮??: {cache is not None}")
    logger.debug(f"backtest_data 摮??: {backtest_data is not None}")
    
    if not n_clicks:
        logger.warning("瘝??暺??鈭?辣")
        return go.Figure(), "??隢??????????
    
    # ?芸?雿輻? enhanced-trades-cache嚗??????????岫敺?backtest-store ???
    if cache:
        logger.debug("雿輻? enhanced-trades-cache 鞈??")
        df_raw = df_from_pack(cache.get("df_raw"))
        daily_state = df_from_pack(cache.get("daily_state"))
        data_source = "enhanced-trades-cache"
        logger.debug(f"df_raw 敶Ｙ?: {df_raw.shape if df_raw is not None else 'None'}")
        logger.debug(f"daily_state 敶Ｙ?: {daily_state.shape if daily_state is not None else 'None'}")
    elif backtest_data and backtest_data.get("results"):
        logger.debug("雿輻? backtest-store 鞈??")
        results = backtest_data["results"]
        logger.debug(f"?舐?蝑??: {list(results.keys())}")
        
        # 敺?backtest-store ?豢?蝚砌???? daily_state ?????
        selected_strategy = None
        for strategy_name, result in results.items():
            logger.debug(f"瑼Ｘ?蝑?? {strategy_name}: daily_state={result.get('daily_state') is not None}, daily_state_std={result.get('daily_state_std') is not None}")
            if result.get("daily_state") or result.get("daily_state_std"):
                selected_strategy = strategy_name
                logger.debug(f"?豢?蝑??: {selected_strategy}")
                break
        
        if not selected_strategy:
            logger.error("瘝???曉???? daily_state ?????)
            return go.Figure(), "????葫蝯??銝剜?????啣???daily_state ?????
        
        result = results[selected_strategy]
        daily_state = df_from_pack(result.get("daily_state") or result.get("daily_state_std"))
        df_raw = df_from_pack(backtest_data.get("df_raw"))
        data_source = f"backtest-store ({selected_strategy})"
        logger.debug(f"df_raw 敶Ｙ?: {df_raw.shape if df_raw is not None else 'None'}")
        logger.debug(f"daily_state 敶Ｙ?: {daily_state.shape if daily_state is not None else 'None'}")
    else:
        logger.error("瘝???舐???????皞?)
        return go.Figure(), "??隢???瑁???葫嚗???潦???敺??皜祉?????乓???亦???

    # 鞈??撽??
    logger.debug("=== 鞈??撽?? ===")
    if df_raw is None or df_raw.empty:
        logger.error("df_raw ?箇征")
        return go.Figure(), "???曆??啗??寡???(df_raw)"
    if daily_state is None or daily_state.empty:
        logger.error("daily_state ?箇征")
        return go.Figure(), "???曆???daily_state嚗???亥???甈??嚗?
    
    # 鞈??銝?雲???銵??撠??
    if len(daily_state) < 60:
        logger.warning("鞈??銝?雲嚗?60憭抬?嚗?歇?仿????")
        return go.Figure(), "??? 鞈??銝?雲嚗?60憭抬?嚗?歇?仿????"
    
    logger.debug(f"df_raw 甈??: {list(df_raw.columns)}")
    logger.debug(f"daily_state 甈??: {list(daily_state.columns)}")

    # 甈??撠??
    c_open = "open" if "open" in df_raw.columns else _first_col(df_raw, ["Open","?????])
    c_close = "close" if "close" in df_raw.columns else _first_col(df_raw, ["Close","?嗥???])
    c_high  = "high" if "high" in df_raw.columns else _first_col(df_raw, ["High","?擃??"])
    c_low   = "low"  if "low"  in df_raw.columns else _first_col(df_raw, ["Low","?雿??"])
    
    logger.debug(f"甈??撠??蝯??: open={c_open}, close={c_close}, high={c_high}, low={c_low}")
    
    if c_open is None or c_close is None:
        logger.error("蝻箏?敹??????潭?雿?)
        return go.Figure(), "???∪?鞈??蝻箏? open/close 甈??"

    # 皞??頛詨?摨??
    open_px = pd.to_numeric(df_raw[c_open], errors="coerce").dropna()
    open_px.index = pd.to_datetime(df_raw.index)
    
    # ??open_px 敺??皞?? w嚗?aseline ?芸?嚗?
    ds_base = df_from_pack(cache.get("daily_state_base")) if cache else None
    wc_base = series_from_pack(cache.get("weight_curve_base")) if cache else None
    
    # 敺?backtest-store 靘?????
    if ds_base is None and (not cache) and backtest_data and "results" in backtest_data:
        ds_base = df_from_pack(result.get("daily_state_base"))
        # 瘜冽?嚗?eight_curve_base 銋???賢??冽? result
        try:
            wc_base = series_from_pack(result.get("weight_curve_base"))
        except Exception:
            wc_base = None
    
    # 隞?baseline w ?箏????瘝???????銵?daily_state['w']
    if ds_base is not None and (not ds_base.empty) and ("w" in ds_base.columns):
        w = pd.to_numeric(ds_base["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
    elif wc_base is not None and (not wc_base.empty):
        w = pd.to_numeric(wc_base, errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
    else:
        # 敺??嚗?窒?函?銵?daily_state嚗???賢歇鋡恍??憯??嚗?
        if "w" not in daily_state.columns:
            return go.Figure(), "??daily_state 蝻箏?甈??甈?? 'w'"
        w = pd.to_numeric(daily_state["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)

    bench = pd.DataFrame({
        "?嗥???: pd.to_numeric(df_raw[c_close], errors="coerce"),
    }, index=pd.to_datetime(df_raw.index))
    if c_high and c_low:
        bench["?擃??"] = pd.to_numeric(df_raw[c_high], errors="coerce")
        bench["?雿??"] = pd.to_numeric(df_raw[c_low], errors="coerce")

    # ATR 璅??瑼Ｘ?嚗???????蹂??湛?
    logger.debug("=== ATR 璅??瑼Ｘ? ===")
    a20, a60 = calculate_atr(df_raw, 20), calculate_atr(df_raw, 60)
    if a20 is None or a60 is None or a20.dropna().size < 60 or a60.dropna().size < 60:
        logger.warning("ATR 璅??銝?雲嚗???唾郎蝷?)
        return go.Figure(), "?? ATR 璅??銝?雲嚗????????????冽??瑁????"
    
    # ???????潮? - ???撅?瑼餌蔭?交?暺?
    logger.debug("=== ??????????潮? ===")
    import numpy as np
    
    # 霈?????身摰?
    cap_now = float(risk_cap_value) if risk_cap_value else 0.8
    atr_now = float(atr_ratio_value) if atr_ratio_value else 1.2
    
    # ?箸??潮?
    caps = np.round(np.linspace(0.10, 1.00, 19), 2)
    atr_mults = np.round(np.linspace(1.00, 2.00, 21), 2)
    
    # 撠??撅閮剖?璊???潮?嚗????◤?扳?敹賜?嚗?
    if risk_cap_value is not None:
        caps = np.unique(np.r_[caps, float(risk_cap_value)])
    if atr_ratio_value is not None:
        atr_mults = np.unique(np.r_[atr_mults, float(atr_ratio_value)])
    
    logger.debug(f"?嗅?閮剖?: cap={cap_now:.2f}, atr={atr_now:.2f}")
    logger.debug(f"cap 蝭??: {len(caps)} ??潘?敺?{caps[0]} ??{caps[-1]}")
    logger.debug(f"ATR 瘥?潛??? {len(atr_mults)} ??潘?敺?{atr_mults[0]} ??{atr_mults[-1]}")
    logger.debug(f"蝮賜????: {len(caps) * len(atr_mults)}")

    pareto_rows = []
    tried = 0
    succeeded = 0
    
    # 瑼Ｘ??臬??臭誑?臬? risk_valve_backtest
    try:
        from SSS_EnsembleTab import risk_valve_backtest
        logger.debug("????臬? risk_valve_backtest")
    except Exception as e:
        logger.error(f"?臬? risk_valve_backtest 憭望?: {e}")
        return go.Figure(), f"???⊥??臬? risk_valve_backtest: {e}"
    
    logger.debug("????瑁???????...")
    for cap_level in caps:
        for atr_mult in atr_mults:
            tried += 1
            if tried % 50 == 0:  # 瘥?0甈∟????甈⊿脣漲
                logger.debug(f"?脣漲: {tried}/{len(caps) * len(atr_mults)} (cap={cap_level:.2f}, atr={atr_mult:.2f})")
            
            try:
                out = risk_valve_backtest(
                    open_px=open_px, w=w, cost=None, benchmark_df=bench,
                    mode=(rv_mode or "cap"), cap_level=float(cap_level),
                    slope20_thresh=0.0, slope60_thresh=0.0,
                    atr_win=20, atr_ref_win=60, atr_ratio_mult=float(atr_mult),
                    use_slopes=True, slope_method="polyfit", atr_cmp="gt"
                )
                
                if not isinstance(out, dict) or "metrics" not in out:
                    logger.warning(f"cap={cap_level:.2f}, atr={atr_mult:.2f}: ????澆??啣虜")
                    continue
                
                m = out["metrics"]
                sig = out["signals"]["risk_trigger"]
                trigger_days = int(sig.fillna(False).sum())

                # ???????????砌??箸迨蝯?????雿?
                pf = float(m.get("pf_valve", np.nan))
                mdd = float(m.get("mdd_valve", np.nan))
                rt_sum_valve = float(m.get("right_tail_sum_valve", np.nan))
                rt_sum_orig = float(m.get("right_tail_sum_orig", np.nan)) if m.get("right_tail_sum_orig") is not None else np.nan
                rt_reduction = float(m.get("right_tail_reduction", np.nan)) if m.get("right_tail_reduction") is not None else (rt_sum_orig - rt_sum_valve if np.isfinite(rt_sum_orig) and np.isfinite(rt_sum_valve) else np.nan)

                # ?園?銝蝑??鞈??
                pareto_rows.append({
                    "cap": cap_level,
                    "atr": atr_mult,
                    "pf": pf,
                    "max_drawdown": abs(mdd) if pd.notna(mdd) else np.nan,
                    "right_tail_sum_valve": rt_sum_valve,
                    "right_tail_sum_orig": rt_sum_orig,
                    "right_tail_reduction": rt_reduction,
                    "risk_trigger_days": trigger_days,
                    "label": f"cap={cap_level:.2f}, atr={atr_mult:.2f}"
                })
                succeeded += 1
                
                if succeeded % 20 == 0:  # 瘥?0甈⊥???????甈?
                    logger.debug(f"???: {succeeded} 蝯?(cap={cap_level:.2f}, atr={atr_mult:.2f})")
                    
            except Exception as e:
                logger.warning(f"cap={cap_level:.2f}, atr={atr_mult:.2f} ?瑁?憭望?: {e}")
                continue

    logger.debug(f"=== ???摰?? ===")
    logger.debug(f"??岫: {tried} 蝯?????: {succeeded} 蝯?)
    
    if not pareto_rows:
        logger.error("瘝????????隞颱?鞈??暺?)
        return go.Figure(), "???⊥?敺?◢?芷????葫????貊???葉???鞈??"

    # ??reduction ?園??莎?頞?之=???憭??撠撾?頞??嚗??蝚血?????脰?蝝????云憭??撠整?
    logger.debug("??????蝯??鞈??...")
    dfp = pd.DataFrame(pareto_rows).dropna(subset=["pf","max_drawdown","right_tail_reduction"]).reset_index(drop=True)
    logger.debug(f"???敺??????? {len(dfp)}")
    logger.debug(f"dfp 甈??: {list(dfp.columns)}")
    
    if dfp.empty:
        logger.error("???敺?????蝛?)
        return go.Figure(), "??鞈?????敺??蝛綽?隢?炎?亙?憪????
    
    logger.debug("???蝜芾ˊ??”...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfp['max_drawdown'],
        y=dfp['pf'],
        mode='markers',
        marker=dict(
            size=np.clip(dfp['risk_trigger_days'] / 5.0, 6, 30),
            color=dfp['right_tail_reduction'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="?喳偏???撟?漲")
        ),
        text=dfp['label'],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "MDD: %{x:.2%}<br>" +
            "PF: %{y:.2f}<br>" +
            "?喳偏蝮賢?(?仿?): %{customdata[0]:.2f}<br>" +
            "?喳偏蝮賢?(???): %{customdata[1]:.2f}<br>" +
            "?喳偏???: %{marker.color:.2f}<br>" +
            "憸券?閫貊?憭拇?: %{marker.size:.0f} 憭?br>" +
            "<extra></extra>"
        ),
        customdata=dfp[["right_tail_sum_valve","right_tail_sum_orig"]].values,
        name="cap-atr grid"
    ))
    
    # ?????urrent???閮??嚗?????撅閮剖?嚗?
    if cap_now in caps and atr_now in atr_mults:
        # ?曉??嗅?閮剖?撠?????雿?
        current_point = dfp[(dfp['cap'] == cap_now) & (dfp['atr'] == atr_now)]
        if not current_point.empty:
            fig.add_trace(go.Scatter(
                x=current_point['max_drawdown'],
                y=current_point['pf'],
                mode='markers',
                marker=dict(
                    size=20,
                    symbol='star',
                    color='gold',
                    line=dict(color='black', width=2)
                ),
                text=f"Current: cap={cap_now:.2f}, atr={atr_now:.2f}",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "MDD: %{x:.2%}<br>" +
                    "PF: %{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                name="Current Settings"
            ))
    
    # ?????lobal???閮??嚗??撅?瑼餉身摰??
    if risk_cap_value is not None and atr_ratio_value is not None:
        # ??岫?曉?撠???????????雿?
        global_cap = float(risk_cap_value)
        global_atr = float(atr_ratio_value)
        global_point = dfp[(dfp['cap'] == global_cap) & (dfp['atr'] == global_atr)]
        
        if not global_point.empty:
            fig.add_trace(go.Scatter(
                x=global_point['max_drawdown'],
                y=global_point['pf'],
                mode='markers',
                marker=dict(
                    size=25,
                    symbol='diamond',
                    color='blue',
                    line=dict(color='white', width=2)
                ),
                text=f"Global: cap={global_cap:.2f}, atr={global_atr:.2f}",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "MDD: %{x:.2%}<br>" +
                    "PF: %{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                name="Global Setting"
            ))

    fig.update_layout(
        title={
            'text': f'憸券?-?梢??啣?嚗?areto Map嚗? {succeeded}/{tried} 蝯?,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="?憭批??歹???椰??末嚗?,
        yaxis_title="PF ?脣????嚗??銝??憟踝?",
        xaxis=dict(tickformat=".1%", gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(248,249,250,0.75)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            itemsizing="constant",
        ),
        hovermode="closest",
        margin=dict(t=86, r=240, l=64, b=56)
    )

    status_msg = f"????????嚗????cap?ATR 瘥??{succeeded}/{tried} 蝯??????喳偏隤踵?撟?漲嚗??=???嚗??=?曉之嚗??憭批?=憸券?閫貊?憭拇???????撅閮剖?嚗?ap={cap_now:.2f}, atr={atr_now:.2f}??????皞??{data_source}"
    return fig, status_msg

# --------- 2025/12/23 09:53 ?啣?:瘥??閮??-?瑁???葫?單? ---------
def run_prediction_script(
    n_clicks,
    ticker: str = "00631L.TW",
    warehouse_file: str = "strategy_warehouse.json",
):
    """
    Execute predict_tomorrow.py safely and return status + refresh signal.
    """
    if not n_clicks:
        return "", no_update

    result = run_prediction_job(
        ticker=ticker,
        warehouse_file=warehouse_file,
        timeout=120,
        run_func=subprocess.run,
    )

    if result.timed_out:
        timeout_val = int(result.timeout_seconds or 120)
        return (
            html.Div(
                f"??Prediction timed out after {timeout_val}s.",
                style={"color": "#dc3545", "marginTop": "10px"},
            ),
            no_update,
        )
    if result.error:
        return (
            html.Div(
                f"??蝟餌絞?航炊: {result.error}",
                style={"color": "#dc3545", "marginTop": "10px"},
            ),
            no_update,
        )

    if result.returncode == 0:
        return (
            html.Div(
                "??閮??摰??嚗???????惜????唳???誑?亦???啁????,
                style={"color": "#28a745", "marginTop": "10px", "fontWeight": "bold"},
            ),
            datetime.now().isoformat(),
        )

    err_text = (result.stderr or result.stdout or "Unknown error").strip()
    return (
        html.Div(
            [
                html.Div("???瑁?憭望?", style={"fontWeight": "bold"}),
                html.Pre(err_text, style={"fontSize": "12px", "whiteSpace": "pre-wrap"}),
            ],
            style={"color": "#dc3545", "marginTop": "10px"},
        ),
        no_update,
    )


register_process_callbacks(app, run_prediction_script_func=run_prediction_script)


if __name__ == '__main__':
    # ??????隤?頂蝯梧??芸?撖阡???? app ???
    _initialize_app_logging()
    
    # ??????瑼Ｘ??湔??∪?嚗??頛舫?銝剖? sss_core.logic嚗?
    try:
        refresh_market_data(TICKER_LIST)
    except Exception as e:
        logger.warning(f"???????寞??啣仃?? {e}嚗?匱蝥???????)
    
    # 閮剔蔭?游??函?????券?蝵?
    app.run_server(
        debug=True, 
        host='127.0.0.1', 
        port=8050,
        threaded=True,
        use_reloader=False  # ?踹?????券?????蝔??憿?
    )     

