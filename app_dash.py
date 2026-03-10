import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from dash.dependencies import ALL
import shutil
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from analysis import config as cfg
from analysis.daily_param_signals import build_daily_signal_context, run_daily_param_signals
import yfinance as yf
import logging
import numpy as np
import re
import sqlite3
from urllib.parse import quote as urlparse
from typing import Any, Dict, Optional
try:
    from analysis.strategy_manager import manager
except ImportError:
    manager = None # 防呆
# 配置 logger - 使用統一日誌系統（按需初始化）
from analysis.logging_config import get_logger, init_logging
import os

# 設定環境變數（但不立即初始化）
os.environ["SSS_CREATE_LOGS"] = "1"

# 獲取日誌器（懶加載）
logger = get_logger("SSS.App")
ENABLE_CASH_ONLY_XRAY_SUBPLOTS = os.environ.get("SSS_DASH_CASH_ONLY_XRAY", "0") == "1"
DISABLE_SMAA_RAW_HIGHLIGHT = os.environ.get("SSS_DASH_DISABLE_SMAA_HIGHLIGHT", "0") == "1"

def normalize_daily_state_columns(ds: pd.DataFrame) -> pd.DataFrame:
    """將不同來源的 daily_state 欄位語意統一：
    - 若 equity 實為倉位市值，改名為 position_value
    - 建立 portfolio_value = position_value + cash
    - 保證有 invested_pct / cash_pct
    """
    if ds is None or ds.empty:
        return ds
    ds = ds.copy()

    # 若已經有 position_value 與 cash，直接建立 portfolio_value
    if {'position_value','cash'}.issubset(ds.columns):
        ds['portfolio_value'] = ds['position_value'] + ds['cash']

    # 僅有 equity + cash 的情況 -> 判斷 equity 是總資產還是倉位
    elif {'equity','cash'}.issubset(ds.columns):
        # 判斷規則：若 equity/(equity+cash) 的中位數顯著 < 0.9，較像「倉位市值」
        ratio = (ds['equity'] / (ds['equity'] + ds['cash'])).replace([np.inf, -np.inf], np.nan).clip(0,1)
        if ratio.median(skipna=True) < 0.9:
            # 把 equity 當成倉位市值
            ds = ds.rename(columns={'equity':'position_value'})
            ds['portfolio_value'] = ds['position_value'] + ds['cash']
        else:
            # equity 已是總資產，反推倉位（若沒有 position_value）
            if 'position_value' not in ds.columns:
                ds['position_value'] = (ds['equity'] - ds['cash']).fillna(0.0)
            ds['portfolio_value'] = ds['equity']

    # 百分比欄位統一
    if 'portfolio_value' in ds.columns:
        pv = ds['portfolio_value'].replace(0, np.nan)
        if 'invested_pct' not in ds.columns and 'position_value' in ds.columns:
            ds['invested_pct'] = (ds['position_value'] / pv).fillna(0.0).clip(0,1)
        if 'cash_pct' not in ds.columns and 'cash' in ds.columns:
            ds['cash_pct'] = (ds['cash'] / pv).fillna(0.0).clip(0,1)

    # 為了向下相容：保留 equity = portfolio_value（供舊繪圖函式使用）
    if 'portfolio_value' in ds.columns:
        ds['equity'] = ds['portfolio_value']

    return ds

def _initialize_app_logging():
    """初始化應用程式日誌系統"""
    # 只在實際需要時才初始化檔案日誌
    init_logging(enable_file=True)
    logger.setLevel(logging.DEBUG)
    logger.info("=== App Dash 啟動 - 統一日誌系統 ===")
    logger.info("已啟用詳細調試模式 - 調試資訊將寫入日誌檔案")
    logger.info(f"日誌目錄: {os.path.abspath('analysis/log')}")
    return logger

# --- Smart Leverage 輔助計算函式 ---
def calculate_smart_leverage_equity(daily_state, df_target, safe_ticker="0050.TW"):
    """
    動態計算 Smart Leverage 權益曲線
    將 daily_state 中的 Cash 部位模擬為持有 safe_ticker

    Args:
        daily_state: 原始的每日狀態 DataFrame (必須含 'w' 權重欄位)
        df_target: 攻擊性資產的價格數據 (必須含 'close')
        safe_ticker: 防守性資產的代碼 (預設 0050.TW)

    Returns:
        修改後的 daily_state (包含重新計算的 equity, cash, position_value)
    """
    try:
        if daily_state is None or daily_state.empty:
            return daily_state
        if "w" not in daily_state.columns:
            logger.warning("Smart Leverage: daily_state 缺少 'w' 欄位，無法計算")
            return daily_state

        safe_path = Path(f"data/{safe_ticker.replace(':', '_')}_data_raw.csv")

        def _download_safe_history(start: str, end: Optional[str] = None) -> Optional[pd.DataFrame]:
            logger.info("下載 %s 用於 Smart Leverage... (%s ~ %s)", safe_ticker, start, end or "latest")
            df_safe_dl = yf.download(safe_ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df_safe_dl is None or df_safe_dl.empty:
                return None
            if isinstance(df_safe_dl.columns, pd.MultiIndex):
                df_safe_dl.columns = [str(c[0]).strip().lower() for c in df_safe_dl.columns]
            else:
                df_safe_dl.columns = [str(c).strip().lower() for c in df_safe_dl.columns]
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            df_safe_dl.to_csv(safe_path, encoding="utf-8")
            return df_safe_dl

        if not safe_path.exists():
            df_safe_dl = _download_safe_history("2010-01-01")
            if df_safe_dl is None:
                logger.warning("Smart Leverage: 無法下載 %s，維持原始資料", safe_ticker)
                return daily_state

        try:
            df_safe = pd.read_csv(safe_path, index_col=0, parse_dates=False, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df_safe = pd.read_csv(safe_path, index_col=0, parse_dates=False, encoding="cp950")
            except UnicodeDecodeError:
                df_safe = pd.read_csv(safe_path, index_col=0, parse_dates=False, encoding="latin1")

        def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if isinstance(out.columns, pd.MultiIndex):
                out.columns = [str(c[0]).strip().lower() for c in out.columns]
            else:
                out.columns = [str(c).strip().lower() for c in out.columns]
            if not isinstance(out.index, pd.DatetimeIndex):
                idx_text = pd.Index(out.index).map(lambda x: str(x).strip())
                date_mask = idx_text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
                out.index = pd.to_datetime(idx_text.where(date_mask), errors="coerce")
            if out.index.tz is not None:
                out.index = out.index.tz_localize(None)
            out = out[out.index.notna()].sort_index()
            return out

        ds_raw = _normalize_idx(daily_state)
        tgt = _normalize_idx(df_target if isinstance(df_target, pd.DataFrame) else pd.DataFrame())
        safe = _normalize_idx(df_safe)

        if ds_raw.empty:
            logger.warning("Smart Leverage: daily_state 日期索引無效，維持原始資料")
            return daily_state

        safe_needs_refresh = False
        if "close" not in safe.columns or safe.empty:
            safe_needs_refresh = True
        else:
            ds_min = ds_raw.index.min()
            ds_max = ds_raw.index.max()
            safe_min = safe.index.min()
            safe_max = safe.index.max()
            if (safe_min - ds_min).days > 5 or (ds_max - safe_max).days > 5:
                safe_needs_refresh = True

        if safe_needs_refresh:
            refresh_start = min(ds_raw.index.min(), pd.Timestamp("2010-01-01")).strftime("%Y-%m-%d")
            refresh_end = (ds_raw.index.max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            safe_dl = _download_safe_history(refresh_start, refresh_end)
            if safe_dl is not None and not safe_dl.empty:
                safe = _normalize_idx(safe_dl)

        if "close" not in tgt.columns:
            logger.warning("Smart Leverage: 目標資產缺少 close 欄位，維持原始資料")
            return daily_state
        if "close" not in safe.columns:
            logger.warning("Smart Leverage: 防守資產缺少 close 欄位，維持原始資料")
            return daily_state

        tgt_close = pd.to_numeric(tgt["close"], errors="coerce")
        safe_close = pd.to_numeric(safe["close"], errors="coerce")
        w_series = pd.to_numeric(ds_raw.get("w"), errors="coerce").clip(0.0, 1.0)

        equity_col = "equity" if "equity" in ds_raw.columns else ("portfolio_value" if "portfolio_value" in ds_raw.columns else None)
        if equity_col is None:
            logger.warning("Smart Leverage: daily_state 缺少 equity/portfolio_value，維持原始資料")
            return daily_state
        equity_series = pd.to_numeric(ds_raw[equity_col], errors="coerce")

        common_idx = ds_raw.index.intersection(tgt_close.index).intersection(safe_close.index)
        common_idx = common_idx.sort_values()
        if len(common_idx) < 2:
            logger.warning("Smart Leverage: 可用交集資料不足 (%d 筆)", len(common_idx))
            return daily_state

        ds = ds_raw.reindex(common_idx).copy()
        w_aligned = w_series.reindex(common_idx).ffill().fillna(0.0).clip(0.0, 1.0)
        tgt_aligned = tgt_close.reindex(common_idx)
        safe_aligned = safe_close.reindex(common_idx)
        eq_aligned = equity_series.reindex(common_idx).ffill()

        valid_mask = tgt_aligned.notna() & safe_aligned.notna() & eq_aligned.notna() & w_aligned.notna()
        valid_idx = common_idx[valid_mask]
        if len(valid_idx) < 2:
            logger.warning("Smart Leverage: 有效資料不足 (%d 筆)", len(valid_idx))
            return daily_state

        ds = ds.loc[valid_idx].copy()
        w_aligned = w_aligned.loc[valid_idx]
        tgt_aligned = tgt_aligned.loc[valid_idx]
        safe_aligned = safe_aligned.loc[valid_idx]
        eq_aligned = eq_aligned.loc[valid_idx]

        initial_equity = float(eq_aligned.iloc[0])
        if not np.isfinite(initial_equity) or initial_equity <= 0:
            logger.warning("Smart Leverage: 初始權益異常 (%s)，維持原始資料", initial_equity)
            return daily_state

        r_target = tgt_aligned.pct_change().fillna(0.0)
        r_safe = safe_aligned.pct_change().fillna(0.0)

        fee_rate = 0.001425 * 0.3
        tax_rate = 0.001  # ETF 稅率：僅在賣出腿計入

        smart_equity = np.full(len(valid_idx), np.nan, dtype=float)
        smart_equity[0] = initial_equity

        for i in range(1, len(valid_idx)):
            prev_eq = float(smart_equity[i - 1])
            if not np.isfinite(prev_eq) or prev_eq <= 0:
                smart_equity[i] = prev_eq
                continue

            w_prev = float(w_aligned.iloc[i - 1])
            w_curr = float(w_aligned.iloc[i])
            turnover = abs(w_curr - w_prev)

            rebalance_cost = 0.0
            if turnover > 1e-9:
                trade_notional = turnover * prev_eq
                # 在攻擊/防守兩資產間換倉：一買一賣，雙邊手續費 + 賣出端交易稅。
                rebalance_cost = trade_notional * (2.0 * fee_rate + tax_rate)

            eq_after_cost = max(prev_eq - rebalance_cost, 0.0)
            combined_ret = w_curr * float(r_target.iloc[i]) + (1.0 - w_curr) * float(r_safe.iloc[i])
            smart_equity[i] = eq_after_cost * (1.0 + combined_ret)

        ds["w"] = w_aligned
        ds["equity"] = smart_equity
        if "portfolio_value" in ds.columns:
            ds["portfolio_value"] = ds["equity"]
        ds["cash"] = ds["equity"] * (1.0 - ds["w"])
        ds["position_value"] = ds["equity"] * ds["w"]
        total = ds["equity"].replace(0.0, np.nan)
        ds["invested_pct"] = (ds["position_value"] / total).fillna(0.0).clip(0.0, 1.0)
        ds["cash_pct"] = (ds["cash"] / total).fillna(0.0).clip(0.0, 1.0)

        logger.info(
            "✅ Smart Leverage 計算完成 (使用 %s，區間 %s ~ %s，最終權益: %s)",
            safe_ticker,
            valid_idx[0].date(),
            valid_idx[-1].date(),
            f"{smart_equity[-1]:,.0f}",
        )
        return ds

    except Exception:
        logger.exception("❌ Smart Leverage 計算失敗")
        return daily_state

def plot_trade_returns_bar(trades_df):
    """
    繪製單筆交易損益圖 (LIFO - 先進後出法)
    目的：反映「近期交易效率」，避免被低成本底倉掩蓋短期操作失誤。
    """
    if trades_df is None or trades_df.empty:
        return go.Figure()

    # 建立副本
    df = trades_df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    # 確保必要欄位存在
    if 'type' not in df.columns or 'price' not in df.columns:
        return go.Figure()

    # --- 如果沒有 return 欄位，啟用 LIFO 演算法計算 ---
    if 'return' not in df.columns:
        # 尋找數量欄位
        qty_col = None
        if 'shares' in df.columns: qty_col = 'shares'
        elif 'weight_change' in df.columns: qty_col = 'weight_change'
        elif 'delta_units' in df.columns: qty_col = 'delta_units'

        if qty_col:
            # === LIFO (先進後出) 計算核心 ===
            # inventory 結構: list of dict {'price': float, 'qty': float}
            inventory = []
            returns = []

            # 確保按時間正序排列
            df = df.sort_values('trade_date')

            for idx, row in df.iterrows():
                try:
                    trade_type = str(row['type']).lower()
                    price = float(row['price'])
                    qty = abs(float(row[qty_col])) # 取絕對值方便計算

                    if qty == 0:
                        returns.append(np.nan)
                        continue

                    if 'buy' in trade_type or 'add' in trade_type or 'long' in trade_type:
                        # 買入：推入堆疊 (Push)
                        inventory.append({'price': price, 'qty': qty})
                        returns.append(np.nan)

                    elif 'sell' in trade_type or 'exit' in trade_type:
                        # 賣出：從堆疊尾端開始扣 (Pop from end = LIFO)
                        remaining_sell_qty = qty
                        total_cost = 0.0
                        matched_qty = 0.0

                        # 倒著遍歷 inventory
                        while remaining_sell_qty > 0 and inventory:
                            last_batch = inventory[-1] # 看最後一筆

                            if last_batch['qty'] <= remaining_sell_qty:
                                # 這批不夠賣，全部吃掉，再找前一批
                                cost = last_batch['qty'] * last_batch['price']
                                total_cost += cost
                                matched_qty += last_batch['qty']
                                remaining_sell_qty -= last_batch['qty']
                                inventory.pop() # 移除這批
                            else:
                                # 這批夠賣，吃掉一部分，剩下的留著
                                cost = remaining_sell_qty * last_batch['price']
                                total_cost += cost
                                matched_qty += remaining_sell_qty
                                inventory[-1]['qty'] -= remaining_sell_qty # 更新庫存數量
                                remaining_sell_qty = 0 # 賣完了

                        # 計算損益
                        if matched_qty > 0:
                            avg_buy_price = total_cost / matched_qty
                            ret = (price - avg_buy_price) / avg_buy_price
                            returns.append(ret)
                        else:
                            # 發生空庫存賣出(可能是放空策略或資料缺漏)，暫記為 0
                            returns.append(0.0)
                    else:
                        returns.append(np.nan)

                except Exception:
                    returns.append(np.nan)

            df['return'] = returns
        else:
            return go.Figure()

    # --- 以下繪圖邏輯 ---
    valid_trades = df[
        (df['type'].astype(str).str.contains('sell', case=False, na=False)) &
        (df['return'].notna())
    ].copy()

    if valid_trades.empty:
        fig = go.Figure()
        fig.update_layout(title="單筆交易報酬率分佈 (無賣出紀錄)", template='plotly_dark', height=400)
        return fig

    valid_trades['return'] = pd.to_numeric(valid_trades['return'], errors='coerce').fillna(0)
    colors = ['#00CC96' if x > 0 else '#EF553B' for x in valid_trades['return']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=valid_trades['trade_date'],
        y=valid_trades['return'],
        marker_color=colors,
        name='單筆損益 (LIFO)'
    ))

    fig.add_hline(y=0, line_color="white", line_width=1)

    fig.update_layout(
        title="近期交易效率 (LIFO 演算法)",
        xaxis_title="賣出日期",
        yaxis_title="報酬率 (vs 最近買入價)",
        yaxis_tickformat=".2%",
        template='plotly_dark',
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    fig.add_annotation(
        text="註：採先進後出法(LIFO)，反映最近一筆操作的成敗",
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


# ATR 計算函數
def calculate_atr(df, window):
    """計算 ATR (Average True Range)"""
    try:
        # app_dash.py / 2025-08-22 14:30
        # 統一 OHLC 欄位對應：收盤、最高、最低價
        high_col = None
        low_col = None
        close_col = None

        # 優先檢查英文欄位名稱（標準格式）
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_col = 'high'
            low_col = 'low'
            close_col = 'close'
        # 檢查大寫英文欄位名稱
        elif 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            high_col = 'High'
            low_col = 'Low'
            close_col = 'Close'
        # 檢查中文欄位名稱
        elif '最高價' in df.columns and '最低價' in df.columns and '收盤價' in df.columns:
            high_col = '最高價'
            low_col = '最低價'
            close_col = '收盤價'
        # 檢查其他可能的欄位名稱（降級處理）
        elif 'open' in df.columns and 'close' in df.columns:
            # 如果沒有高低價，用開盤價和收盤價近似
            high_col = 'open'
            low_col = 'close'
            close_col = 'close'

        if high_col and low_col and close_col:
            # 有高低價時，計算 True Range
            high = df[high_col]
            low = df[low_col]
            close = df[close_col]

            # 確保數據為數值型
            high = pd.to_numeric(high, errors='coerce')
            low = pd.to_numeric(low, errors='coerce')
            close = pd.to_numeric(close, errors='coerce')

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
        else:
            # 只有收盤價時，用價格變化近似
            if close_col:
                close = pd.to_numeric(df[close_col], errors='coerce')
            elif 'close' in df.columns:
                close = pd.to_numeric(df['close'], errors='coerce')
            elif 'Close' in df.columns:
                close = pd.to_numeric(df['Close'], errors='coerce')
            else:
                # 只記一次警告，避免重複刷屏
                if not hasattr(calculate_atr, '_warning_logged'):
                    logger.warning("找不到可用的價格欄位來計算 ATR，降級為 ATR-only 模式")
                    calculate_atr._warning_logged = True
                return pd.Series(index=df.index, dtype=float)

            price_change = close.diff().abs()
            atr = price_change.rolling(window=window).mean()

        # 檢查計算結果
        if atr is None or atr.empty or atr.isna().all():
            if not hasattr(calculate_atr, '_warning_logged'):
                logger.warning(f"ATR 計算結果無效，window={window}，降級為 ATR-only 模式")
                calculate_atr._warning_logged = True
            return pd.Series(index=df.index, dtype=float)

        return atr
    except Exception as e:
        if not hasattr(calculate_atr, '_warning_logged'):
            logger.warning(f"ATR 計算失敗: {e}，降級為 ATR-only 模式")
            calculate_atr._warning_logged = True
        return pd.Series(index=df.index, dtype=float)


def _build_benchmark_df(df_raw):
    """建立基準資料 DataFrame，統一處理欄位名稱和數據轉換"""
    # app_dash.py / 2025-08-22 14:30
    # 統一 OHLC 欄位對應：收盤、最高、最低價
    bench = pd.DataFrame(index=pd.to_datetime(df_raw.index))

    # 收盤價欄位 - 優先使用英文欄位，回退到中文欄位
    if 'close' in df_raw.columns:
        bench["收盤價"] = pd.to_numeric(df_raw["close"], errors="coerce")
    elif 'Close' in df_raw.columns:
        bench["收盤價"] = pd.to_numeric(df_raw["Close"], errors="coerce")
    elif '收盤價' in df_raw.columns:
        bench["收盤價"] = pd.to_numeric(df_raw["收盤價"], errors="coerce")

    # 最高價和最低價欄位 - 優先使用英文欄位，回退到中文欄位
    if 'high' in df_raw.columns and 'low' in df_raw.columns:
        bench["最高價"] = pd.to_numeric(df_raw["high"], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw["low"], errors="coerce")
    elif 'High' in df_raw.columns and 'Low' in df_raw.columns:
        bench["最高價"] = pd.to_numeric(df_raw["High"], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw["Low"], errors="coerce")
    elif '最高價' in df_raw.columns and '最低價' in df_raw.columns:
        bench["最高價"] = pd.to_numeric(df_raw["最高價"], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw["最低價"], errors="coerce")

    return bench


def _calc_nav_relative_ratio(daily_state_df: pd.DataFrame, df_raw: pd.DataFrame) -> float:
    """
    計算 NAV 相對淨值比（BH=1 基準）：
    ratio = (strategy_nav_end / strategy_nav_start) / (benchmark_close_end / benchmark_close_start)
    """
    try:
        if daily_state_df is None or df_raw is None:
            return np.nan
        if not isinstance(daily_state_df, pd.DataFrame) or not isinstance(df_raw, pd.DataFrame):
            return np.nan
        if daily_state_df.empty or df_raw.empty or "equity" not in daily_state_df.columns:
            return np.nan

        close_col = None
        for c in ("close", "Close", "收盤價"):
            if c in df_raw.columns:
                close_col = c
                break
        if close_col is None:
            return np.nan

        eq = pd.to_numeric(daily_state_df["equity"], errors="coerce").dropna()
        close_s = pd.to_numeric(df_raw[close_col], errors="coerce").dropna()
        if eq.empty or close_s.empty:
            return np.nan

        eq.index = pd.to_datetime(eq.index, errors="coerce")
        close_s.index = pd.to_datetime(close_s.index, errors="coerce")
        eq = eq[~eq.index.isna()]
        close_s = close_s[~close_s.index.isna()]
        if isinstance(eq.index, pd.DatetimeIndex) and eq.index.tz is not None:
            eq.index = eq.index.tz_localize(None)
        if isinstance(close_s.index, pd.DatetimeIndex) and close_s.index.tz is not None:
            close_s.index = close_s.index.tz_localize(None)

        idx = eq.index.intersection(close_s.index)
        if len(idx) < 2:
            return np.nan

        eq = eq.loc[idx]
        close_s = close_s.loc[idx]
        eq_start = float(eq.iloc[0])
        eq_end = float(eq.iloc[-1])
        px_start = float(close_s.iloc[0])
        px_end = float(close_s.iloc[-1])
        if eq_start <= 0 or px_start <= 0 or not np.isfinite(eq_end) or not np.isfinite(px_end):
            return np.nan

        strategy_nav = eq_end / eq_start
        benchmark_nav = px_end / px_start
        if benchmark_nav <= 0 or not np.isfinite(strategy_nav) or not np.isfinite(benchmark_nav):
            return np.nan
        return float(strategy_nav / benchmark_nav)
    except Exception:
        return np.nan


def _compute_ssma_prominence_threshold(
    smaa: pd.Series,
    min_dist: int,
    quantile_win: int,
    prom_factor: float,
) -> pd.Series:
    """重建 ssma_turn 每日 prominence 門檻序列（僅供 UI 繪圖）。"""
    if smaa is None or len(smaa) == 0:
        return pd.Series(dtype=float)

    smaa_num = pd.to_numeric(smaa, errors="coerce")
    series_clean = smaa_num.dropna()
    if series_clean.empty:
        return pd.Series(index=smaa_num.index, dtype=float)

    min_dist_i = max(int(min_dist or 1), 1)
    quantile_win_i = max(int(quantile_win or (min_dist_i + 1)), min_dist_i + 1)
    q = float(prom_factor or 50.0) / 100.0
    q = min(max(q, 0.0), 1.0)

    prom = series_clean.rolling(
        window=min_dist_i + 1,
        min_periods=min_dist_i + 1,
    ).apply(lambda x: np.ptp(x), raw=True)

    prom_valid = prom.dropna()
    if prom_valid.empty:
        return pd.Series(index=smaa_num.index, dtype=float)

    initial_threshold = float(prom_valid.quantile(q))
    threshold_series = (
        prom.rolling(window=quantile_win_i, min_periods=quantile_win_i)
        .quantile(q)
        .shift(1)
        .ffill()
        .fillna(initial_threshold)
    )
    return threshold_series.reindex(smaa_num.index)


def _build_indicator_daily_frame(
    df_ind: pd.DataFrame,
    strategy_type: str,
    params: dict,
    buy_dates: Optional[list] = None,
    sell_dates: Optional[list] = None,
) -> pd.DataFrame:
    """產生每日 SMAA 與閥值資料，供 Dash 顯示。"""
    if df_ind is None or df_ind.empty or "smaa" not in df_ind.columns:
        return pd.DataFrame()

    ind = df_ind.copy()
    ind.index = pd.to_datetime(ind.index, errors="coerce")
    ind = ind[ind.index.notna()].sort_index()
    if ind.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=ind.index)
    out["smaa"] = pd.to_numeric(ind["smaa"], errors="coerce")

    stype = str(strategy_type or "").strip().lower()
    if stype in {"single", "dual", "rma"} and {"base", "sd"}.issubset(ind.columns):
        base = pd.to_numeric(ind["base"], errors="coerce")
        sd = pd.to_numeric(ind["sd"], errors="coerce")
        buy_mult = float(params.get("buy_mult", 1.0))
        sell_mult = float(params.get("sell_mult", 1.0))
        pine_parity_mode = bool(params.get("pine_parity_mode", False))

        out["base"] = base
        out["buy_threshold"] = base - sd * buy_mult if pine_parity_mode else base + sd * buy_mult
        out["sell_threshold"] = base + sd * sell_mult
    elif stype == "ssma_turn":
        out["prom_threshold"] = _compute_ssma_prominence_threshold(
            out["smaa"],
            min_dist=int(params.get("min_dist", 5)),
            quantile_win=int(params.get("quantile_win", 100)),
            prom_factor=float(params.get("prom_factor", 50.0)),
        )
        buy_set = set(pd.to_datetime(buy_dates or [], errors="coerce"))
        sell_set = set(pd.to_datetime(sell_dates or [], errors="coerce"))
        out["is_buy_signal"] = out.index.isin(buy_set)
        out["is_sell_signal"] = out.index.isin(sell_set)

    return out.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _extract_latest_t_signal(
    strategy_type: str,
    indicator_daily: Optional[pd.DataFrame] = None,
    daily_state: Optional[pd.DataFrame] = None,
    trade_df: Optional[pd.DataFrame] = None,
    trade_cooldown_bars: int = 0,
) -> Dict[str, Any]:
    """計算策略訊號快照：最後可用日可執行訊號 + 最近一次已發出訊號。"""
    stype = str(strategy_type or "").strip().lower()
    out: Dict[str, Any] = {
        "signal": "HOLD",
        "data_date": "",
        "reason": "",
        "last_signal": "HOLD",
        "last_signal_date": "",
        "last_signal_reason": "",
    }

    # 交易紀錄：用來對齊「上次已發出訊號」與買入冷卻條件
    last_trade_date: Optional[pd.Timestamp] = None
    last_exec_side: Optional[str] = None
    trades = trade_df.copy() if isinstance(trade_df, pd.DataFrame) else pd.DataFrame()
    if not trades.empty:
        trades.columns = [str(c).lower() for c in trades.columns]
        if "trade_date" in trades.columns:
            trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce")
        if "signal_date" in trades.columns:
            trades["signal_date"] = pd.to_datetime(trades["signal_date"], errors="coerce")
        if "type" not in trades.columns and "action" in trades.columns:
            trades["type"] = trades["action"].astype(str)
        if "type" in trades.columns:
            t = trades["type"].astype(str).str.lower()
            side = pd.Series("OTHER", index=trades.index, dtype=object)
            side[t.str.contains("buy|add|long", na=False)] = "BUY"
            side[t.str.contains("sell|exit", na=False)] = "SELL"
            trades["side_norm"] = side
            exec_rows = trades[trades["side_norm"].isin(["BUY", "SELL"])].copy()
            if not exec_rows.empty:
                exec_rows = exec_rows.sort_values("trade_date")
                last_row = exec_rows.iloc[-1]
                last_exec_side = str(last_row["side_norm"])
                sig_dt = last_row.get("signal_date")
                trd_dt = last_row.get("trade_date")
                if pd.notna(sig_dt):
                    out["last_signal_date"] = pd.Timestamp(sig_dt).strftime("%Y-%m-%d")
                elif pd.notna(trd_dt):
                    out["last_signal_date"] = pd.Timestamp(trd_dt).strftime("%Y-%m-%d")
                out["last_signal"] = last_exec_side
                out["last_signal_reason"] = "已執行的最後訊號"
                if pd.notna(trd_dt):
                    last_trade_date = pd.Timestamp(trd_dt)

    # 推估目前是否持倉（以每日狀態優先）
    in_position = False
    ds = daily_state.copy() if isinstance(daily_state, pd.DataFrame) else pd.DataFrame()
    if not ds.empty:
        ds.index = pd.to_datetime(ds.index, errors="coerce")
        ds = ds[ds.index.notna()].sort_index()
        if "w" in ds.columns:
            w_s = pd.to_numeric(ds["w"], errors="coerce").dropna()
            if not w_s.empty:
                in_position = bool(float(w_s.iloc[-1]) > 1e-6)
        elif "shares" in ds.columns:
            shares_s = pd.to_numeric(ds["shares"], errors="coerce").dropna()
            if not shares_s.empty:
                in_position = bool(float(shares_s.iloc[-1]) > 0)
        elif "position_value" in ds.columns:
            pos_s = pd.to_numeric(ds["position_value"], errors="coerce").dropna()
            if not pos_s.empty:
                in_position = bool(float(pos_s.iloc[-1]) > 0)
    elif last_exec_side is not None:
        in_position = bool(last_exec_side == "BUY")

    def _bars_since_last_trade(ref_index: pd.DatetimeIndex) -> Optional[int]:
        if last_trade_date is None or ref_index.empty:
            return None
        idx = pd.DatetimeIndex(pd.to_datetime(ref_index, errors="coerce"))
        idx = idx[~idx.isna()]
        if idx.empty:
            return None
        pos_now = len(idx) - 1
        pos_last = idx.searchsorted(last_trade_date, side="right") - 1
        if pos_last < 0:
            return None
        return int(pos_now - pos_last)

    def _apply_actionable_filter(raw_signal: str, raw_reason: str, ref_index: pd.DatetimeIndex) -> tuple[str, str]:
        sig = str(raw_signal).upper()
        reason = str(raw_reason)
        if sig == "BUY":
            if in_position:
                return "HOLD", "已有持倉不再買"
            bars = _bars_since_last_trade(ref_index)
            if bars is not None and int(trade_cooldown_bars or 0) > 0 and bars <= int(trade_cooldown_bars):
                return "HOLD", "交易冷卻中"
            return "BUY", reason
        if sig == "SELL":
            if not in_position:
                return "HOLD", "無持倉可賣"
            return "SELL", reason
        return "HOLD", reason

    if stype == "ensemble":
        if ds.empty or "w" not in ds.columns:
            return out
        w = pd.to_numeric(ds["w"], errors="coerce").dropna()
        if w.empty:
            return out
        out["data_date"] = w.index[-1].strftime("%Y-%m-%d")
        if len(w) >= 2:
            dw_s = w.diff().fillna(0.0)
            sig_s = pd.Series("HOLD", index=dw_s.index, dtype=object)
            sig_s[dw_s > 1e-12] = "BUY"
            sig_s[dw_s < -1e-12] = "SELL"
            out["signal"] = str(sig_s.iloc[-1])
            out["reason"] = ""
            non_hold = sig_s[sig_s != "HOLD"]
            if not non_hold.empty:
                if not out["last_signal_date"]:
                    out["last_signal"] = str(non_hold.iloc[-1])
                    out["last_signal_date"] = non_hold.index[-1].strftime("%Y-%m-%d")
                    out["last_signal_reason"] = "w_t - w_(t-1)"
        else:
            out["reason"] = "單一資料點無法判斷訊號變化"
        return out

    ind = indicator_daily.copy() if isinstance(indicator_daily, pd.DataFrame) else pd.DataFrame()
    if ind.empty or "smaa" not in ind.columns:
        return out
    ind.index = pd.to_datetime(ind.index, errors="coerce")
    ind = ind[ind.index.notna()].sort_index()
    if ind.empty:
        return out

    row = ind.iloc[-1]
    smaa = pd.to_numeric(pd.Series([row.get("smaa")]), errors="coerce").iloc[0]
    if pd.isna(smaa):
        return out

    out["data_date"] = ind.index[-1].strftime("%Y-%m-%d")
    if stype in {"single", "dual", "rma"}:
        buy_th = pd.to_numeric(pd.Series([row.get("buy_threshold")]), errors="coerce").iloc[0]
        sell_th = pd.to_numeric(pd.Series([row.get("sell_threshold")]), errors="coerce").iloc[0]
        sig_s = pd.Series("HOLD", index=ind.index, dtype=object)
        if "buy_threshold" in ind.columns:
            buy_mask = pd.to_numeric(ind["smaa"], errors="coerce") < pd.to_numeric(ind["buy_threshold"], errors="coerce")
            sig_s[buy_mask.fillna(False)] = "BUY"
        if "sell_threshold" in ind.columns:
            sell_mask = pd.to_numeric(ind["smaa"], errors="coerce") > pd.to_numeric(ind["sell_threshold"], errors="coerce")
            sig_s[sell_mask.fillna(False)] = "SELL"

        raw_now = "HOLD"
        raw_reason = "在買賣閥值之間"
        if pd.notna(buy_th) and smaa < buy_th:
            raw_now = "BUY"
            raw_reason = "smaa小於買入閥值"
        elif pd.notna(sell_th) and smaa > sell_th:
            raw_now = "SELL"
            raw_reason = "smaa大於賣出閥值"
        out["signal"], out["reason"] = _apply_actionable_filter(raw_now, raw_reason, ind.index)

        non_hold = sig_s[sig_s != "HOLD"]
        if not non_hold.empty and not out["last_signal_date"]:
            out["last_signal"] = str(non_hold.iloc[-1])
            out["last_signal_date"] = non_hold.index[-1].strftime("%Y-%m-%d")
            ls = out["last_signal"]
            out["last_signal_reason"] = (
                "smaa小於買入閥值" if ls == "BUY" else "smaa大於賣出閥值"
            )
        return out

    if stype == "ssma_turn":
        is_buy_series = pd.Series(ind.get("is_buy_signal", False), index=ind.index).astype(bool)
        is_sell_series = pd.Series(ind.get("is_sell_signal", False), index=ind.index).astype(bool)
        sig_s = pd.Series("HOLD", index=ind.index, dtype=object)
        sig_s[is_buy_series] = "BUY"
        sig_s[is_sell_series] = "SELL"

        raw_now = "HOLD"
        raw_reason = "無轉折訊號"
        is_buy = bool(is_buy_series.iloc[-1])
        is_sell = bool(is_sell_series.iloc[-1])
        if is_buy and not is_sell:
            raw_now = "BUY"
            raw_reason = "轉折向上訊號"
        elif is_sell and not is_buy:
            raw_now = "SELL"
            raw_reason = "轉折向下訊號"
        out["signal"], out["reason"] = _apply_actionable_filter(raw_now, raw_reason, ind.index)

        non_hold = sig_s[sig_s != "HOLD"]
        if not non_hold.empty and not out["last_signal_date"]:
            out["last_signal"] = str(non_hold.iloc[-1])
            out["last_signal_date"] = non_hold.index[-1].strftime("%Y-%m-%d")
            out["last_signal_reason"] = (
                "轉折向上訊號" if out["last_signal"] == "BUY" else "轉折向下訊號"
            )
        return out

    return out


def _create_daily_smaa_threshold_figure(
    indicator_df: pd.DataFrame,
    strategy_name: str,
    strategy_type: str,
    theme: str,
) -> go.Figure:
    """建立每日 SMAA 與閥值圖（非 Ensemble 策略）。"""
    fig = go.Figure()
    if indicator_df is None or indicator_df.empty:
        fig.update_layout(
            title=f"{strategy_name} 每日 SMAA / 閥值（無資料）",
            template="plotly_white",
            height=320,
        )
        return fig

    ind = indicator_df.copy()
    ind.index = pd.to_datetime(ind.index, errors="coerce")
    ind = ind[ind.index.notna()].sort_index()
    if ind.empty:
        fig.update_layout(
            title=f"{strategy_name} 每日 SMAA / 閥值（無資料）",
            template="plotly_white",
            height=320,
        )
        return fig

    if theme == "theme-light":
        template = "plotly_white"
        paper_bg = "#ffffff"
        plot_bg = "#ffffff"
        font_color = "#212529"
        grid_color = "rgba(70, 70, 70, 0.16)"
    elif theme == "theme-blue":
        template = "plotly_dark"
        paper_bg = "#001a33"
        plot_bg = "#001a33"
        font_color = "#ffe066"
        grid_color = "rgba(255, 224, 102, 0.18)"
    else:
        template = "plotly_dark"
        paper_bg = "#121212"
        plot_bg = "#121212"
        font_color = "#e0e0e0"
        grid_color = "rgba(224, 224, 224, 0.14)"

    fig.add_trace(
        go.Scatter(
            x=ind.index,
            y=pd.to_numeric(ind.get("smaa"), errors="coerce"),
            name="SMAA",
            mode="lines",
            line=dict(color="#4dabf7", width=1.8),
        )
    )

    stype = str(strategy_type or "").strip().lower()
    if stype in {"single", "dual", "rma"}:
        if "base" in ind.columns:
            fig.add_trace(
                go.Scatter(
                    x=ind.index,
                    y=pd.to_numeric(ind["base"], errors="coerce"),
                    name="基準線",
                    mode="lines",
                    line=dict(color="#f59f00", width=1.3),
                )
            )
        if "buy_threshold" in ind.columns:
            fig.add_trace(
                go.Scatter(
                    x=ind.index,
                    y=pd.to_numeric(ind["buy_threshold"], errors="coerce"),
                    name="買入閥值",
                    mode="lines",
                    line=dict(color="#40c057", width=1.2, dash="dot"),
                )
            )
        if "sell_threshold" in ind.columns:
            fig.add_trace(
                go.Scatter(
                    x=ind.index,
                    y=pd.to_numeric(ind["sell_threshold"], errors="coerce"),
                    name="賣出閥值",
                    mode="lines",
                    line=dict(color="#fa5252", width=1.2, dash="dash"),
                )
            )
    elif stype == "ssma_turn" and "prom_threshold" in ind.columns:
        fig.add_trace(
            go.Scatter(
                x=ind.index,
                y=pd.to_numeric(ind["prom_threshold"], errors="coerce"),
                name="Prominence 閥值",
                mode="lines",
                yaxis="y2",
                line=dict(color="#ffd43b", width=1.2, dash="dash"),
            )
        )

    layout_kwargs = {}
    if stype == "ssma_turn" and "prom_threshold" in ind.columns:
        layout_kwargs["yaxis2"] = dict(
            title="Prominence 閥值",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        )

    fig.update_layout(
        title=f"{strategy_name} 每日 SMAA / 閥值",
        template=template,
        height=360,
        hovermode="x unified",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=40, b=30),
        **layout_kwargs,
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color)
    fig.update_yaxes(title="SMAA", showgrid=True, gridcolor=grid_color)
    return fig

def calculate_equity_curve(open_px, w, cap, atr_ratio):
    """計算權益曲線"""
    try:
        # 簡化的權益曲線計算
        # 這裡使用開盤價和權重的乘積來模擬權益變化
        equity = (open_px * w * cap).cumsum()
        return equity
    except Exception as e:
        logger.warning(f"權益曲線計算失敗: {e}")
        return None

def calculate_trades_from_equity(equity_curve, open_px, w, cap, atr_ratio):
    """從權益曲線計算交易記錄"""
    try:
        if equity_curve is None or equity_curve.empty:
            return None

        # 簡化的交易記錄生成
        # 這裡根據權重變化來識別交易
        weight_changes = w.diff().abs()
        trade_dates = weight_changes[weight_changes > 0.01].index

        trades = []
        for date in trade_dates:
            trades.append({
                'trade_date': date,
                'return': 0.0  # 簡化，實際應該計算報酬率
            })

        if trades:
            return pd.DataFrame(trades)
        else:
            return pd.DataFrame(columns=['trade_date', 'return'])

    except Exception as e:
        logger.warning(f"交易記錄計算失敗: {e}")
        return None

# 解包器函數：支援 pack_df/pack_series 和傳統 JSON 字串兩種格式
def df_from_pack(data):
    """從 pack_df 結果或 JSON 字串解包 DataFrame"""
    import io, json
    import pandas as pd

    # 如果已經是 DataFrame，直接返回
    if isinstance(data, pd.DataFrame):
        return data

    # 檢查是否為 None 或空字串
    if data is None:
        return pd.DataFrame()

    # 如果是字串，進行額外檢查
    if isinstance(data, str):
        if data == "" or data == "[]":
            return pd.DataFrame()
        # 先嘗試 split → 再退回預設
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
    """從 pack_series 結果或 JSON 字串解包 Series"""
    import io
    import pandas as pd

    # 如果已經是 Series，直接返回
    if isinstance(data, pd.Series):
        return data

    # 檢查是否為 None 或空字串
    if data is None:
        return pd.Series(dtype=float)

    # 如果是字串，進行額外檢查
    if isinstance(data, str):
        if data == "" or data == "[]":
            return pd.Series(dtype=float)
        # Series 也先試 split
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

try:
    from SSSv096 import (
        param_presets, load_data, compute_single, compute_dual, compute_RMA,
        compute_ssma_turn_combined, backtest_unified, plot_stock_price, plot_equity_cash, plot_weight_series,
        calculate_holding_periods, get_param_preset_names
    )
except ImportError:
    from SSSv096 import (
        param_presets, load_data, compute_single, compute_dual, compute_RMA,
        compute_ssma_turn_combined, backtest_unified, plot_stock_price, plot_equity_cash, plot_weight_series,
        calculate_holding_periods
    )

    def get_param_preset_names(include_hidden: bool = False):
        names = list(param_presets.keys())
        if include_hidden:
            return names
        return [
            name
            for name, cfg in param_presets.items()
            if not bool(cfg.get("hidden", False)) and not bool(cfg.get("ui_hidden", False))
        ]

# 🔥 使用新的健壯序列化工具（Pickle + Gzip + Base64）
try:
    from sss_core.data_utils import pack_df_robust as pack_df, pack_series_robust as pack_series
    from sss_core.data_utils import unpack_df_robust as df_from_pack, unpack_series_robust as series_from_pack
except Exception:
    # Fallback 到舊版 (若 data_utils 不存在)
    try:
        from sss_core.schemas import pack_df, pack_series
    except Exception:
        from schemas import pack_df, pack_series

# 匯入權重欄位確保函式
try:
    from sss_core.normalize import _ensure_weight_columns
except Exception:
    # 如果無法匯入，定義一個空的函式作為 fallback
    def _ensure_weight_columns(df):
        return df

# 假設你有 get_version_history_html
try:
    from version_history import get_version_history_html
except ImportError:
    def get_version_history_html() -> str:
        return "<b>無法載入版本歷史記錄</b>"

# --- 保證放進 Store 的都是 JSON-safe ---
def _pack_any(x):
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return pack_df(x)          # orient="split" + date_format="iso"
    if isinstance(x, pd.Series):
        return pack_series(x)      # orient="split" + date_format="iso"
    return x

def _pack_result_for_store(result: dict) -> dict:
    # 統一把所有 pandas 物件轉成字串（JSON）
    keys = [
        'trade_df', 'trades_df', 'signals_df',
        'equity_curve', 'cash_curve', 'price_series',
        'daily_state', 'trade_ledger',
        'daily_state_std', 'trade_ledger_std',
        'weight_curve',
        'indicator_daily',
        'crash_debug_df', 'crash_state',
        # ➊ 新增：保存未套閥門 baseline
        'daily_state_base', 'trade_ledger_base', 'weight_curve_base',
        # ➋ 新增：保存 valve 版本
        'daily_state_valve', 'trade_ledger_valve', 'weight_curve_valve', 'equity_curve_valve'
    ]
    out = dict(result)
    for k in keys:
        if k in out:
            out[k] = _pack_any(out[k])
    # 另外把 datetime tuple 的 trades 轉可序列化（你原本也有做）
    if 'trades' in out and isinstance(out['trades'], list):
        out['trades'] = [
            (str(t[0]), t[1], str(t[2])) if isinstance(t, tuple) and len(t) == 3 else t
            for t in out['trades']
        ]
    return out


def _first_non_empty_result_df(result: dict, keys: list[str]) -> pd.DataFrame:
    """依序挑選第一個可用且非空的 DataFrame。"""
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
    """依序挑選第一個可用且非空的 Series。"""
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
    """補齊在場比例與年化換手率。"""
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


def _parse_count_and_limit(raw: Any) -> tuple[float, float]:
    if raw is None:
        return np.nan, np.nan
    nums = re.findall(r"[-+]?\d[\d,]*", str(raw))
    if not nums:
        return np.nan, np.nan
    count = float(nums[0].replace(",", ""))
    limit_count = float(nums[1].replace(",", "")) if len(nums) >= 2 else np.nan
    return count, limit_count


def _extract_twse_breadth_counts(raw_json: Any) -> tuple[float, float, float, float]:
    """Extract up/down and limit-up/limit-down counts from mi_index payload."""
    if not isinstance(raw_json, str) or not raw_json:
        return np.nan, np.nan, np.nan, np.nan
    try:
        obj = json.loads(raw_json)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    for table in obj.get("tables", []):
        if not isinstance(table, dict):
            continue
        rows = table.get("data", [])
        fields = table.get("fields", [])
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        if not isinstance(fields, list) or len(fields) < 2:
            continue
        row0 = rows[0] if isinstance(rows[0], list) else []
        row1 = rows[1] if isinstance(rows[1], list) else []
        if len(row0) < 2 or len(row1) < 2:
            continue
        if "(" not in str(row0[1]) or "(" not in str(row1[1]):
            continue

        up, up_limit = _parse_count_and_limit(row0[1])
        down, down_limit = _parse_count_and_limit(row1[1])
        if np.isfinite(up) and np.isfinite(down):
            return up, down, up_limit, down_limit
    return np.nan, np.nan, np.nan, np.nan


def _build_twse_market_features(db_path: str = "twse_data.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        fmt = pd.read_sql_query("SELECT date, trade_value, taiex, change FROM fmtqik", conn)
        mi = pd.read_sql_query("SELECT date, has_data, raw_json FROM mi_index WHERE has_data=1", conn)
    finally:
        conn.close()

    for col in ["trade_value", "taiex", "change"]:
        fmt[col] = pd.to_numeric(fmt[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    fmt["date"] = pd.to_datetime(fmt["date"], errors="coerce")
    fmt = fmt.dropna(subset=["date"]).set_index("date").sort_index()

    rows: list[tuple[pd.Timestamp, float, float, float, float]] = []
    for _, rec in mi.iterrows():
        dt = pd.to_datetime(rec.get("date"), errors="coerce")
        if pd.isna(dt):
            continue
        up, down, up_limit, down_limit = _extract_twse_breadth_counts(rec.get("raw_json"))
        rows.append((dt, up, down, up_limit, down_limit))

    breadth = pd.DataFrame(
        rows,
        columns=["date", "up", "down", "up_limit", "down_limit"],
    ).set_index("date").sort_index()
    breadth["breadth"] = (breadth["up"] - breadth["down"]) / (breadth["up"] + breadth["down"])
    breadth["limit_net"] = (breadth["up_limit"] - breadth["down_limit"]) / (
        breadth["up_limit"] + breadth["down_limit"]
    )

    mk = fmt.join(breadth[["breadth", "limit_net"]], how="left")
    mk["breadth"] = mk["breadth"].ffill()
    mk["limit_net"] = mk["limit_net"].ffill()
    prev = mk["taiex"] - mk["change"]
    mk["chg_pct"] = mk["change"] / prev.replace(0.0, np.nan)

    for n in [20, 40, 60, 120]:
        mk[f"ma{n}"] = mk["taiex"].rolling(n, min_periods=max(10, n // 2)).mean()
    for n in [10, 20, 40, 60]:
        mk[f"tv{n}"] = mk["trade_value"].rolling(n, min_periods=max(5, n // 2)).mean()
    return mk


def _build_crash_overlay_state(
    index: pd.Index,
    market_df: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame]:
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    mk = market_df.copy()
    mk.index = pd.to_datetime(mk.index, errors="coerce")
    mk = mk[mk.index.notna()].sort_index()
    if mk.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    ma_win = int(params.get("ma", 20))
    dd_win = int(params.get("dd_win", 250))
    tv_win = int(params.get("tv_win", 10))
    dd_th = float(params.get("dd_th", -0.10))
    day_drop_th = float(params.get("day_drop_th", -0.03))
    breadth_th = float(params.get("breadth_th", -0.30))
    limit_net_th = float(params.get("limit_net_th", -0.20))
    tv_ratio_th = float(params.get("tv_ratio_th", 1.2))
    min_hits = int(params.get("min_hits", 3))
    cap = float(params.get("cap", 0.1))
    cooldown = int(params.get("cooldown", 5))
    reentry_mode_raw = str(params.get("reentry_mode", "time_only") or "time_only").strip().lower()
    reentry_mode = reentry_mode_raw if reentry_mode_raw in {"time_only", "crash_off_2d", "trend_or_crash_off2"} else "time_only"
    reentry_min_hold = max(int(params.get("reentry_min_hold", 1) or 1), 1)

    taiex = pd.to_numeric(mk.get("taiex"), errors="coerce").reindex(idx).ffill()
    breadth = pd.to_numeric(mk.get("breadth"), errors="coerce").reindex(idx).ffill()
    limit_net = pd.to_numeric(mk.get("limit_net"), errors="coerce").reindex(idx).ffill()
    chg_pct = pd.to_numeric(mk.get("chg_pct"), errors="coerce").reindex(idx).ffill()
    ma = pd.to_numeric(mk.get(f"ma{ma_win}"), errors="coerce").reindex(idx).ffill()
    tv = pd.to_numeric(mk.get("trade_value"), errors="coerce").reindex(idx).ffill()
    tv_ref = pd.to_numeric(mk.get(f"tv{tv_win}"), errors="coerce").reindex(idx).ffill()
    tv_ratio = tv / tv_ref.replace(0.0, np.nan)

    rolling_peak = taiex.rolling(dd_win, min_periods=max(20, dd_win // 3)).max()
    drawdown = taiex / rolling_peak - 1.0

    trend_down = taiex < ma
    panic_day = chg_pct < day_drop_th
    panic_breadth = breadth < breadth_th
    panic_limit = limit_net < limit_net_th
    panic_volume = tv_ratio > tv_ratio_th
    panic_hits = (
        panic_day.fillna(False).astype(int)
        + panic_breadth.fillna(False).astype(int)
        + panic_limit.fillna(False).astype(int)
        + panic_volume.fillna(False).astype(int)
    )
    crash = (trend_down & (drawdown < dd_th) & (panic_hits >= min_hits)).fillna(False)
    crash_off_2d = (~crash).rolling(2, min_periods=2).sum().ge(2).fillna(False)

    state_pre_shift = pd.Series(1.0, index=idx, dtype=float)
    remain = 0
    held = 0
    reentry_release = pd.Series(False, index=idx, dtype=bool)
    for i in range(len(idx)):
        if bool(crash.iloc[i]):
            remain = cooldown
            held = 0
        if remain > 0:
            held += 1
            release = False
            if reentry_mode != "time_only" and held >= reentry_min_hold:
                is_trend_up = not bool(trend_down.iloc[i])
                is_crash_off2 = bool(crash_off_2d.iloc[i])
                if reentry_mode == "crash_off_2d":
                    release = is_crash_off2
                elif reentry_mode == "trend_or_crash_off2":
                    release = is_trend_up or is_crash_off2

            if release:
                state_pre_shift.iloc[i] = 1.0
                reentry_release.iloc[i] = True
                remain = 0
                held = 0
            else:
                state_pre_shift.iloc[i] = cap
                remain -= 1
        else:
            state_pre_shift.iloc[i] = 1.0

    # Next-bar execution to avoid look-ahead.
    state = state_pre_shift.shift(1).ffill().fillna(1.0).clip(lower=min(cap, 1.0), upper=1.0)

    debug_df = pd.DataFrame(
        {
            "state": state,
            "state_pre_shift": state_pre_shift,
            "taiex": taiex,
            "ma": ma,
            "drawdown": drawdown,
            "trend_down": trend_down.fillna(False),
            "breadth": breadth,
            "limit_net": limit_net,
            "chg_pct": chg_pct,
            "tv_ratio": tv_ratio,
            "panic_day": panic_day.fillna(False),
            "panic_breadth": panic_breadth.fillna(False),
            "panic_limit": panic_limit.fillna(False),
            "panic_volume": panic_volume.fillna(False),
            "panic_hits": panic_hits,
            "crash_core": (trend_down & (drawdown < dd_th)).fillna(False),
            "crash": crash,
            "crash_off_2d": crash_off_2d,
            "reentry_release": reentry_release,
            "reentry_mode": reentry_mode,
            "reentry_min_hold": int(reentry_min_hold),
            "day_drop_th": float(day_drop_th),
            "breadth_th": float(breadth_th),
            "limit_net_th": float(limit_net_th),
            "tv_ratio_th": float(tv_ratio_th),
            "min_hits": int(min_hits),
            "dd_th": float(dd_th),
        },
        index=idx,
    )
    return state, debug_df


def _recompute_metrics_from_equity_and_trades(
    equity: pd.Series,
    trades_df: pd.DataFrame,
    base_metrics: Optional[dict] = None,
) -> dict:
    metrics = dict(base_metrics or {})
    if equity is None or len(equity) < 2:
        return metrics

    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return metrics

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1.0 / 252.0)
    ann_ret = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if (1.0 + total_ret) > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min()) if not dd.empty else np.nan

    daily_ret = eq.pct_change().dropna()
    vol = float(daily_ret.std() * np.sqrt(252.0)) if len(daily_ret) > 1 else np.nan
    sharpe = float((daily_ret.mean() * np.sqrt(252.0)) / daily_ret.std()) if len(daily_ret) > 1 and daily_ret.std() > 0 else np.nan
    downside = daily_ret[daily_ret < 0]
    sortino = float((daily_ret.mean() * np.sqrt(252.0)) / downside.std()) if len(downside) > 1 and downside.std() > 0 else np.nan
    calmar = float(ann_ret / abs(mdd)) if np.isfinite(ann_ret) and np.isfinite(mdd) and mdd < 0 else np.nan

    sell_rets = pd.Series(dtype=float)
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        t = trades_df.copy()
        t.columns = [str(c).lower() for c in t.columns]
        if "return" in t.columns:
            ret_s = pd.to_numeric(t["return"], errors="coerce")
        else:
            ret_s = pd.Series(np.nan, index=t.index)
        if "type" in t.columns:
            sell_mask = t["type"].astype(str).str.lower().str.contains("sell|exit", na=False)
            sell_rets = ret_s[sell_mask].dropna()
        else:
            sell_rets = ret_s.dropna()

    num_trades = int(len(sell_rets))
    win_rate = float((sell_rets > 0).mean()) if num_trades > 0 else np.nan
    avg_win = float(sell_rets[sell_rets > 0].mean()) if (sell_rets > 0).any() else np.nan
    avg_loss = float(sell_rets[sell_rets < 0].mean()) if (sell_rets < 0).any() else np.nan
    payoff = float(abs(avg_win / avg_loss)) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0 else np.nan
    profit_factor = float(sell_rets[sell_rets > 0].sum() / abs(sell_rets[sell_rets < 0].sum())) if (sell_rets < 0).any() else np.nan

    metrics.update(
        {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "annualized_volatility": vol,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff,
            "profit_factor": profit_factor,
        }
    )
    return metrics


def _apply_crash_overlay_to_result(
    result: dict,
    df_raw: pd.DataFrame,
    strat_params: dict,
    ticker: str,
    preset_name: str,
    overlay_params: dict[str, Any],
    market_df: Optional[pd.DataFrame] = None,
) -> dict:
    if not isinstance(result, dict):
        return result
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return result

    ds = _first_non_empty_result_df(
        result,
        ["daily_state_valve", "daily_state_std", "daily_state", "daily_state_base"],
    )
    if ds.empty or "w" not in ds.columns:
        logger.warning("[CrashOverlay] 缺少 daily_state/w，略過")
        return result

    ds = ds.copy()
    ds.index = pd.to_datetime(ds.index, errors="coerce")
    ds = ds[ds.index.notna()].sort_index()
    if ds.empty:
        return result

    mk = market_df
    if mk is None:
        db_path = str(overlay_params.get("db_path", "twse_data.db"))
        if not os.path.exists(db_path):
            logger.warning("[CrashOverlay] 找不到資料庫 %s，略過", db_path)
            return result
        try:
            mk = _build_twse_market_features(db_path=db_path)
        except Exception as exc:
            logger.warning("[CrashOverlay] 讀取市場資料失敗: %s", exc)
            return result
    if mk is None or mk.empty:
        return result

    state, crash_debug_df = _build_crash_overlay_state(ds.index, mk, overlay_params)
    if state.empty:
        return result

    common_idx = ds.index.intersection(state.index)
    if len(common_idx) < 3:
        return result

    w_old = pd.to_numeric(ds.loc[common_idx, "w"], errors="coerce").fillna(0.0)
    w_new = (w_old * state.reindex(common_idx).fillna(1.0)).clip(0.0, 1.0)
    ds_valve = ds.loc[common_idx].copy()
    ds_valve["w"] = w_new

    if "daily_state_base" not in result and result.get("daily_state") is not None:
        result["daily_state_base"] = result["daily_state"]
    if "trade_ledger_base" not in result and result.get("trade_ledger") is not None:
        result["trade_ledger_base"] = result["trade_ledger"]
    if "weight_curve_base" not in result and result.get("weight_curve") is not None:
        result["weight_curve_base"] = result["weight_curve"]

    px_col = "open" if "open" in df_raw.columns else ("close" if "close" in df_raw.columns else None)
    if px_col is None:
        logger.warning("[CrashOverlay] df_raw 缺少 open/close，略過")
        return result

    open_px = pd.to_numeric(df_raw[px_col], errors="coerce")
    open_px.index = pd.to_datetime(df_raw.index, errors="coerce")
    open_px = open_px[open_px.index.notna()].sort_index()
    open_px = open_px.reindex(common_idx).dropna()
    if open_px.empty:
        return result
    w_eval = w_new.reindex(open_px.index).ffill().fillna(0.0)

    try:
        from SSS_EnsembleTab import CostParams, calculate_performance
    except Exception as exc:
        logger.warning("[CrashOverlay] 無法匯入回測函式: %s", exc)
        return result

    trade_cost = strat_params.get("trade_cost", {}) if isinstance(strat_params, dict) else {}
    ticker_core = str(ticker or "").split(".")[0]
    default_tax_bp = 10.0 if ticker_core.startswith("00") else 30.0
    cost = CostParams(
        buy_fee_bp=float(trade_cost.get("buy_fee_bp", 4.27)),
        sell_fee_bp=float(trade_cost.get("sell_fee_bp", 4.27)),
        sell_tax_bp=float(trade_cost.get("sell_tax_bp", default_tax_bp)),
    )

    eq, trades_df, ledger_df, _ = calculate_performance(open_px, w_eval, cost)
    if ledger_df is None or ledger_df.empty:
        return result

    crash_debug_df = crash_debug_df.reindex(common_idx).copy() if isinstance(crash_debug_df, pd.DataFrame) else pd.DataFrame()
    if not crash_debug_df.empty:
        result["crash_debug_df"] = pack_df(crash_debug_df)
    result["crash_state"] = pack_series(state.reindex(common_idx).fillna(1.0))

    trades_out = trades_df.copy() if isinstance(trades_df, pd.DataFrame) else pd.DataFrame()
    if not trades_out.empty:
        trades_out.columns = [str(c).lower() for c in trades_out.columns]
        if "trade_date" in trades_out.columns:
            trades_out["trade_date"] = pd.to_datetime(trades_out["trade_date"], errors="coerce")
        else:
            trades_out["trade_date"] = pd.NaT
        if "reason" not in trades_out.columns:
            trades_out["reason"] = ""
        trades_out["base_reason"] = trades_out["reason"].fillna("").astype(str)

        w_base_eval = w_old.reindex(open_px.index).ffill().fillna(0.0)
        delta_overlay = (w_eval - w_base_eval).fillna(0.0)
        step_overlay = w_eval.diff().fillna(w_eval)
        step_base = w_base_eval.diff().fillna(w_base_eval)
        step_impact = (step_overlay - step_base).fillna(0.0)

        step_map = {pd.Timestamp(k): float(v) for k, v in step_impact.items()}
        level_map = {pd.Timestamp(k): float(v) for k, v in delta_overlay.items()}
        trade_step = trades_out["trade_date"].map(step_map).fillna(0.0)
        trade_level = trades_out["trade_date"].map(level_map).fillna(0.0)
        crash_triggered = (trade_step.abs() > 1e-6) | (trade_level.abs() > 1e-6)
        side = trades_out.get("type", pd.Series("", index=trades_out.index)).astype(str).str.lower()
        crash_reason = pd.Series("", index=trades_out.index, dtype=object)
        crash_reason.loc[crash_triggered & side.str.contains("buy|add|long", na=False)] = "crash_overlay_buy"
        crash_reason.loc[crash_triggered & side.str.contains("sell|exit", na=False)] = "crash_overlay_sell"
        crash_reason.loc[crash_triggered & (crash_reason == "")] = "crash_overlay_adjust"

        trades_out["crash_triggered"] = crash_triggered
        trades_out["crash_reason"] = crash_reason
        overwrite_mask = crash_triggered & (crash_reason != "")
        trades_out.loc[overwrite_mask, "reason"] = crash_reason.loc[overwrite_mask]
        trades_df = trades_out

    daily_state_valve_df = ledger_df.copy()
    if isinstance(ds_valve, pd.DataFrame) and not ds_valve.empty:
        ds_valve_aligned = ds_valve.reindex(daily_state_valve_df.index)
        for col in ds_valve_aligned.columns:
            if col not in daily_state_valve_df.columns:
                daily_state_valve_df[col] = ds_valve_aligned[col]
        if "w" in ds_valve_aligned.columns:
            daily_state_valve_df["w"] = pd.to_numeric(ds_valve_aligned["w"], errors="coerce").ffill().fillna(0.0)

    packed_daily_state_valve = pack_df(daily_state_valve_df)
    packed_trade_ledger_valve = pack_df(trades_df if isinstance(trades_df, pd.DataFrame) else pd.DataFrame())

    result["daily_state_valve"] = packed_daily_state_valve
    result["trade_ledger_valve"] = packed_trade_ledger_valve
    result["weight_curve_valve"] = pack_series(w_eval)
    result["equity_curve_valve"] = pack_series(eq)
    result["daily_state_std"] = packed_daily_state_valve
    result["trade_ledger_std"] = packed_trade_ledger_valve

    existing_metrics = result.get("metrics", {})
    result["metrics"] = _recompute_metrics_from_equity_and_trades(eq, trades_df, existing_metrics)
    result["_risk_valve_applied"] = True

    trigger_days = int((state.reindex(common_idx).fillna(1.0) < 0.999).sum())
    trigger_rate = float(trigger_days / max(len(common_idx), 1))
    result["valve"] = {
        "applied": True,
        "mode": "crash_only",
        "preset": preset_name,
        "cap": float(overlay_params.get("cap", np.nan)),
        "reentry_mode": str(overlay_params.get("reentry_mode", "time_only")),
        "reentry_min_hold": int(overlay_params.get("reentry_min_hold", 1) or 1),
        "trigger_days": trigger_days,
        "trigger_rate": trigger_rate,
        "atr_ratio": "crash_only",
    }
    result["crash_overlay"] = {
        "preset": preset_name,
        "params": dict(overlay_params),
        "trigger_days": trigger_days,
        "trigger_rate": trigger_rate,
    }
    return result


def _calc_equity_stats_from_series(eq: pd.Series) -> dict[str, float]:
    if eq is None or len(eq) < 2:
        return {"total_return": np.nan, "annual_return": np.nan, "max_drawdown": np.nan}
    s = pd.to_numeric(eq, errors="coerce").dropna()
    if len(s) < 2:
        return {"total_return": np.nan, "annual_return": np.nan, "max_drawdown": np.nan}
    total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1.0 / 252.0)
    ann = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if (1.0 + total_ret) > 0 else np.nan
    mdd = float((s / s.cummax() - 1.0).min())
    return {"total_return": total_ret, "annual_return": ann, "max_drawdown": mdd}


def _compute_crash_overlay_preset_comparison(
    ds_base: pd.DataFrame,
    df_raw: pd.DataFrame,
    ticker: str,
    strat_params: Optional[dict] = None,
    market_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (normalized_equity_df, summary_df) for baseline + all crash presets."""
    if ds_base is None or ds_base.empty or "w" not in ds_base.columns:
        return pd.DataFrame(), pd.DataFrame()
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if market_df is None or market_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ds = ds_base.copy()
    ds.index = pd.to_datetime(ds.index, errors="coerce")
    ds = ds[ds.index.notna()].sort_index()
    if ds.empty:
        return pd.DataFrame(), pd.DataFrame()

    px_col = "open" if "open" in df_raw.columns else ("close" if "close" in df_raw.columns else None)
    if px_col is None:
        return pd.DataFrame(), pd.DataFrame()

    open_px = pd.to_numeric(df_raw[px_col], errors="coerce")
    open_px.index = pd.to_datetime(df_raw.index, errors="coerce")
    open_px = open_px[open_px.index.notna()].sort_index()
    idx = open_px.index.intersection(ds.index)
    if len(idx) < 20:
        return pd.DataFrame(), pd.DataFrame()

    open_px = open_px.reindex(idx).dropna()
    if open_px.empty:
        return pd.DataFrame(), pd.DataFrame()

    w_base = pd.to_numeric(ds["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0).clip(0.0, 1.0)

    try:
        from SSS_EnsembleTab import CostParams, calculate_performance
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    trade_cost = strat_params.get("trade_cost", {}) if isinstance(strat_params, dict) else {}
    ticker_core = str(ticker or "").split(".")[0]
    default_tax_bp = 10.0 if ticker_core.startswith("00") else 30.0
    cost = CostParams(
        buy_fee_bp=float(trade_cost.get("buy_fee_bp", 4.27)),
        sell_fee_bp=float(trade_cost.get("sell_fee_bp", 4.27)),
        sell_tax_bp=float(trade_cost.get("sell_tax_bp", default_tax_bp)),
    )

    eq_base, trades_base, _, _ = calculate_performance(open_px, w_base, cost)
    if eq_base is None or len(eq_base) < 2:
        return pd.DataFrame(), pd.DataFrame()

    eq_map: dict[str, pd.Series] = {"原始": eq_base}
    rows: list[dict[str, Any]] = []

    base_stats = _calc_equity_stats_from_series(eq_base)
    rows.append(
        {
            "方案": "原始",
            "總回報率": base_stats["total_return"],
            "年化回報率": base_stats["annual_return"],
            "最大回撤": base_stats["max_drawdown"],
            "交易次數": int(len(trades_base)) if isinstance(trades_base, pd.DataFrame) else 0,
            "觸發天數": 0,
            "觸發率": 0.0,
        }
    )

    label_map = {
        "sparse_vturn_cap010_v2": "Sparse VTurn v2 (cap=0.10)",
        "sparse_vturn_cap015_v2": "Sparse VTurn v2 (cap=0.15)",
        "sparse_vturn_cap020_v2": "Sparse VTurn v2 (cap=0.20)",
        "best_00631l_v1": "Best 00631L v1",
        "balanced_00631l_v1": "Balanced 00631L v1",
        "mild_00631l_v1": "Mild 00631L v1",
    }

    for preset_key, preset in CRASH_OVERLAY_PRESETS.items():
        try:
            state, _ = _build_crash_overlay_state(open_px.index, market_df, preset)
            if state.empty:
                continue
            w_overlay = (w_base * state.reindex(open_px.index).fillna(1.0)).clip(0.0, 1.0)
            eq, trades_df, _, _ = calculate_performance(open_px, w_overlay, cost)
            if eq is None or len(eq) < 2:
                continue
            label = label_map.get(preset_key, preset_key)
            eq_map[label] = eq
            st = _calc_equity_stats_from_series(eq)
            trigger_days = int((state.reindex(open_px.index).fillna(1.0) < 0.999).sum())
            trigger_rate = float(trigger_days / max(len(open_px), 1))
            rows.append(
                {
                    "方案": label,
                    "總回報率": st["total_return"],
                    "年化回報率": st["annual_return"],
                    "最大回撤": st["max_drawdown"],
                    "交易次數": int(len(trades_df)) if isinstance(trades_df, pd.DataFrame) else 0,
                    "觸發天數": trigger_days,
                    "觸發率": trigger_rate,
                }
            )
        except Exception:
            continue

    if len(eq_map) < 2:
        return pd.DataFrame(), pd.DataFrame()

    eq_norm = pd.DataFrame({k: (v / v.iloc[0]) for k, v in eq_map.items() if len(v) > 1})
    summary = pd.DataFrame(rows)
    return eq_norm, summary


def _create_crash_xray_figure(
    crash_debug_df: pd.DataFrame,
    *,
    strategy_name: str,
    theme: str,
) -> go.Figure:
    if crash_debug_df is None or crash_debug_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{strategy_name} Crash X-Ray（無資料）", height=340)
        return fig

    df = crash_debug_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{strategy_name} Crash X-Ray（無資料）", height=340)
        return fig

    if theme == "theme-light":
        template = "plotly_white"
    else:
        template = "plotly_dark"

    min_hits = float(pd.to_numeric(df.get("min_hits"), errors="coerce").dropna().iloc[-1]) if "min_hits" in df.columns and pd.to_numeric(df.get("min_hits"), errors="coerce").dropna().size else 3.0
    day_drop_th = float(pd.to_numeric(df.get("day_drop_th"), errors="coerce").dropna().iloc[-1]) if "day_drop_th" in df.columns and pd.to_numeric(df.get("day_drop_th"), errors="coerce").dropna().size else np.nan
    breadth_th = float(pd.to_numeric(df.get("breadth_th"), errors="coerce").dropna().iloc[-1]) if "breadth_th" in df.columns and pd.to_numeric(df.get("breadth_th"), errors="coerce").dropna().size else np.nan
    limit_net_th = float(pd.to_numeric(df.get("limit_net_th"), errors="coerce").dropna().iloc[-1]) if "limit_net_th" in df.columns and pd.to_numeric(df.get("limit_net_th"), errors="coerce").dropna().size else np.nan
    tv_ratio_th = float(pd.to_numeric(df.get("tv_ratio_th"), errors="coerce").dropna().iloc[-1]) if "tv_ratio_th" in df.columns and pd.to_numeric(df.get("tv_ratio_th"), errors="coerce").dropna().size else np.nan

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            "panic_hits（含 min_hits）",
            "大跌（chg_pct）",
            "跌家暴增（breadth）",
            "跌停潮（limit_net）",
            "爆量（tv_ratio）",
        ),
        row_heights=[0.22, 0.19, 0.19, 0.19, 0.21],
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=pd.to_numeric(df.get("panic_hits"), errors="coerce"),
            name="panic_hits",
            marker_color="#ff922b",
            opacity=0.8,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=np.full(len(df), min_hits, dtype=float),
            name="min_hits",
            line=dict(color="#ff6b6b", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=pd.to_numeric(df.get("chg_pct"), errors="coerce"),
            name="chg_pct",
            line=dict(color="#4dabf7", width=1.4),
        ),
        row=2,
        col=1,
    )
    if np.isfinite(day_drop_th):
        fig.add_hline(y=day_drop_th, line_color="#fa5252", line_width=1, line_dash="dash", row=2, col=1)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=pd.to_numeric(df.get("panic_day"), errors="coerce").fillna(0.0).astype(float),
            name="panic_day",
            marker_color="rgba(250,82,82,0.35)",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=pd.to_numeric(df.get("breadth"), errors="coerce"),
            name="breadth",
            line=dict(color="#69db7c", width=1.4),
        ),
        row=3,
        col=1,
    )
    if np.isfinite(breadth_th):
        fig.add_hline(y=breadth_th, line_color="#fa5252", line_width=1, line_dash="dash", row=3, col=1)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=pd.to_numeric(df.get("panic_breadth"), errors="coerce").fillna(0.0).astype(float),
            name="panic_breadth",
            marker_color="rgba(250,82,82,0.35)",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=pd.to_numeric(df.get("limit_net"), errors="coerce"),
            name="limit_net",
            line=dict(color="#74c0fc", width=1.4),
        ),
        row=4,
        col=1,
    )
    if np.isfinite(limit_net_th):
        fig.add_hline(y=limit_net_th, line_color="#fa5252", line_width=1, line_dash="dash", row=4, col=1)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=pd.to_numeric(df.get("panic_limit"), errors="coerce").fillna(0.0).astype(float),
            name="panic_limit",
            marker_color="rgba(250,82,82,0.35)",
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=pd.to_numeric(df.get("tv_ratio"), errors="coerce"),
            name="tv_ratio",
            line=dict(color="#ffd43b", width=1.4),
        ),
        row=5,
        col=1,
    )
    if np.isfinite(tv_ratio_th):
        fig.add_hline(y=tv_ratio_th, line_color="#fa5252", line_width=1, line_dash="dash", row=5, col=1)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=pd.to_numeric(df.get("panic_volume"), errors="coerce").fillna(0.0).astype(float),
            name="panic_volume",
            marker_color="rgba(250,82,82,0.35)",
        ),
        row=5,
        col=1,
    )

    fig.update_layout(
        template=template,
        title=f"{strategy_name} Crash Overlay X-Ray",
        height=1280,
        hovermode="x unified",
        margin=dict(l=55, r=20, t=55, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="hits", row=1, col=1)
    fig.update_yaxes(title_text="chg_pct", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="breadth", row=3, col=1)
    fig.update_yaxes(title_text="limit_net", row=4, col=1)
    fig.update_yaxes(title_text="tv_ratio", row=5, col=1)
    fig.update_xaxes(showgrid=True, row=5, col=1)
    return fig

default_tickers = ["00631L.TW", "2330.TW", "00663L.TW", "00663L.TW", "00675L.TW", "00685L.TW"]


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

# Crash-only overlay presets.
# 2026-03-10 sensitivity map suggests the robust core region is:
#   min_hits=3, cooldown=5, with cap around 0.10~0.20.
CRASH_OVERLAY_PRESETS = {
    "sparse_vturn_cap010_v2": {
        "ma": 20,
        "dd_win": 250,
        "dd_th": -0.0852797990720294,
        "day_drop_th": -0.0240517748364169,
        "breadth_th": -0.3227538840287519,
        "limit_net_th": -0.3401478408310758,
        "tv_win": 10,
        "tv_ratio_th": 1.756825442586568,
        "min_hits": 3,
        "cap": 0.10,
        "cooldown": 5,
        "db_path": "twse_data.db",
    },
    "sparse_vturn_cap015_v2": {
        "ma": 20,
        "dd_win": 250,
        "dd_th": -0.0852797990720294,
        "day_drop_th": -0.0240517748364169,
        "breadth_th": -0.3227538840287519,
        "limit_net_th": -0.3401478408310758,
        "tv_win": 10,
        "tv_ratio_th": 1.756825442586568,
        "min_hits": 3,
        "cap": 0.15,
        "cooldown": 5,
        "db_path": "twse_data.db",
    },
    "sparse_vturn_cap020_v2": {
        "ma": 20,
        "dd_win": 250,
        "dd_th": -0.0852797990720294,
        "day_drop_th": -0.0240517748364169,
        "breadth_th": -0.3227538840287519,
        "limit_net_th": -0.3401478408310758,
        "tv_win": 10,
        "tv_ratio_th": 1.756825442586568,
        "min_hits": 3,
        "cap": 0.20,
        "cooldown": 5,
        "db_path": "twse_data.db",
    },
    "best_00631l_v1": {
        "ma": 20,
        "dd_win": 250,
        "dd_th": -0.0852797990720294,
        "day_drop_th": -0.0240517748364169,
        "breadth_th": -0.3227538840287519,
        "limit_net_th": -0.3401478408310758,
        "tv_win": 10,
        "tv_ratio_th": 1.756825442586568,
        "min_hits": 3,
        "cap": 0.05,
        "cooldown": 5,
        "db_path": "twse_data.db",
    },
    "balanced_00631l_v1": {
        "ma": 60,
        "dd_win": 250,
        "dd_th": -0.1418599704645447,
        "day_drop_th": -0.0342399324425173,
        "breadth_th": -0.3914386391806469,
        "limit_net_th": -0.0937363569245414,
        "tv_win": 20,
        "tv_ratio_th": 1.5549949091094372,
        "min_hits": 3,
        "cap": 0.15,
        "cooldown": 21,
        "db_path": "twse_data.db",
    },
    "mild_00631l_v1": {
        "ma": 40,
        "dd_win": 90,
        "dd_th": -0.0763065377812218,
        "day_drop_th": -0.0392352640774065,
        "breadth_th": -0.4268922350909967,
        "limit_net_th": -0.1630301044112239,
        "tv_win": 40,
        "tv_ratio_th": 1.4251039919392948,
        "min_hits": 2,
        "cap": 0.20,
        "cooldown": 5,
        "db_path": "twse_data.db",
    },
}
DEFAULT_CRASH_OVERLAY_PRESET = "sparse_vturn_cap015_v2"
CRASH_OVERLAY_PRESET_OPTIONS = [
    {"label": "Sparse VTurn v2 (cap=0.15) [Recommended]", "value": "sparse_vturn_cap015_v2"},
    {"label": "Sparse VTurn v2 (cap=0.10)", "value": "sparse_vturn_cap010_v2"},
    {"label": "Sparse VTurn v2 (cap=0.20)", "value": "sparse_vturn_cap020_v2"},
    {"label": "Best 00631L v1", "value": "best_00631l_v1"},
    {"label": "Balanced 00631L v1", "value": "balanced_00631l_v1"},
    {"label": "Mild 00631L v1", "value": "mild_00631l_v1"},
]

# 初始化倉庫下拉選單選項
warehouse_options = []
if manager:
    try:
        files = manager.list_warehouses()
        warehouse_options = [
            {'label': '🟢 現役 (Active)' if f == "strategy_warehouse.json" else f, 'value': f}
            for f in files
        ]
    except Exception:
        warehouse_options = [{'label': '🟢 現役 (Active)', 'value': 'strategy_warehouse.json'}]

theme_list = ['theme-dark', 'theme-light', 'theme-blue']

def get_theme_label(theme):
    if theme == 'theme-dark':
        return '🌑 深色主題'
    elif theme == 'theme-light':
        return '🌕 淺色主題'
    else:
        return '💙 藍黃主題'

def get_column_display_name(column_name):
    """將交易欄位名稱轉為中文顯示。"""
    column_mapping = {
        "trade_date": "交易日期",
        "signal_date": "訊號日期",
        "type": "交易類型",
        "price": "價格",
        "weight_change": "權重變化",
        "w_before": "交易前權重",
        "w_after": "交易後權重",
        "return": "報酬",
        "shares": "股數",
        "reason": "原因",
        "fee": "手續費",
        "net_amount": "淨額",
        "leverage_ratio": "槓桿比",
        "strategy_version": "策略版本",
        "indicator_smaa": "指標SMAA",
        "indicator_base": "指標BASE",
        "indicator_sd": "指標SD",
        "delta_units": "股數變化",
        "exec_notional": "成交金額",
        "equity_after": "交易後權益",
        "equity_pct": "權益%",
        "invested_pct": "在場比例",
        "position_value": "部位市值",
        "crash_triggered": "Crash觸發",
        "crash_reason": "Crash原因",
        "base_reason": "原始原因",
        "comment": "備註",
    }
    return column_mapping.get(column_name, column_name)


DISPLAY_NAME = {
    "trade_date": "交易日期",
    "signal_date": "訊號日期",
    "type": "交易類型",
    "price": "價格",
    "weight_change": "權重變化",
    "w_before": "交易前權重",
    "w_after": "交易後權重",
    "return": "報酬",
    "shares": "股數",
    "reason": "原因",
    "fee": "手續費",
    "net_amount": "淨額",
    "leverage_ratio": "槓桿比",
    "strategy_version": "策略版本",
    "indicator_smaa": "指標SMAA",
    "indicator_base": "指標BASE",
    "indicator_sd": "指標SD",
    "delta_units": "股數變化",
    "exec_notional": "成交金額",
    "equity_after": "交易後權益",
    "equity_pct": "權益%",
    "invested_pct": "在場比例",
    "position_value": "部位市值",
    "crash_triggered": "Crash觸發",
    "crash_reason": "Crash原因",
    "base_reason": "原始原因",
    "comment": "備註",
}


HIDE_COLS = {
    "shares_before",
    "shares_after",
    "fee_buy",
    "fee_sell",
    "sell_tax",
    "tax",
    "date",
    "open",
    "equity_open_after_trade",
    "cash_after",
    "cash_pct",
}


PREFER_ORDER = [
    "trade_date",
    "signal_date",
    "type",
    "price",
    "weight_change",
    "w_before",
    "w_after",
    "return",
    "shares",
    "reason",
    "base_reason",
    "crash_triggered",
    "crash_reason",
    "fee",
    "net_amount",
    "leverage_ratio",
    "strategy_version",
    "indicator_smaa",
    "indicator_base",
    "indicator_sd",
    "delta_units",
    "exec_notional",
    "equity_after",
    "equity_pct",
    "invested_pct",
    "position_value",
    "comment",
]


UNIFIED_MIN_COLUMNS = [
    "trade_date",
    "signal_date",
    "type",
    "price",
    "weight_change",
    "w_before",
    "w_after",
    "return",
    "shares",
    "reason",
    "fee",
    "net_amount",
    "leverage_ratio",
    "strategy_version",
    "indicator_smaa",
    "indicator_base",
    "indicator_sd",
]


def format_trade_like_df_for_display(df):
    """統一交易明細欄位與格式，並轉為中文顯示。"""
    import pandas as pd

    if df is None or len(df) == 0:
        return df

    d = df.copy()

    hide = [c for c in HIDE_COLS if c in d.columns]
    if hide:
        d = d.drop(columns=hide, errors="ignore")

    for col in UNIFIED_MIN_COLUMNS:
        if col not in d.columns:
            d[col] = pd.NA

    if {"equity_after", "cash_after"}.issubset(d.columns):
        tot = d["equity_after"] + d["cash_after"]
        if "equity_pct" not in d.columns:
            d["equity_pct"] = d.apply(
                lambda r: ""
                if pd.isna(r["equity_after"]) or pd.isna(tot.loc[r.name]) or tot.loc[r.name] <= 0
                else f"{(r['equity_after'] / tot.loc[r.name]):.2%}",
                axis=1,
            )

    def _fmt_percent_cell(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        if s == "":
            return ""
        if s.endswith("%"):
            v = pd.to_numeric(s[:-1], errors="coerce")
            return f"{v:.2f}%" if pd.notna(v) else s
        v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
        if pd.isna(v):
            return s
        # 比例欄位通常是 0~1；若來源已是 0~100 則保留為百分點。
        return f"{v:.2%}" if abs(v) <= 1.0 else f"{v:.2f}%"

    def _fmt_numeric_cell(x, decimals=2, as_int=False, fixed_decimals=False):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        if s == "":
            return ""
        v = pd.to_numeric(pd.Series([s.replace(",", "")]), errors="coerce").iloc[0]
        if pd.isna(v):
            return s
        if as_int:
            return f"{int(round(v)):,}"
        txt = f"{float(v):,.{decimals}f}"
        if fixed_decimals:
            return txt
        return txt.rstrip("0").rstrip(".")

    def _fmt_date_col(s):
        dt = pd.to_datetime(s, errors="coerce")
        return dt.dt.strftime("%Y-%m-%d").fillna("")

    for col in ["trade_date", "signal_date"]:
        if col in d.columns:
            d[col] = _fmt_date_col(d[col])

    if "type" in d.columns:
        type_map = {
            "buy": "買入",
            "add": "加碼",
            "long": "買入",
            "sell": "賣出",
            "sell_forced": "強制賣出",
            "forced_sell": "強制賣出",
            "exit": "賣出",
            "hold": "持有",
        }
        type_series = d["type"].fillna("").astype(str).str.lower()
        d["type"] = type_series.map(type_map).fillna(type_series)

    if "reason" in d.columns:
        reason_map = {
            "signal_entry": "訊號進場",
            "signal_exit": "訊號出場",
            "stop_loss": "停損",
            "force_liquidate": "強制平倉",
            "forced_sell": "強制賣出",
            "sell_forced": "強制賣出",
            "loan_repayment": "還款賣出",
            "end_of_period": "期末平倉",
            "ensemble_rebalance_buy": "再平衡買入",
            "ensemble_rebalance_sell": "再平衡賣出",
            "crash_overlay_buy": "Crash Overlay 買入",
            "crash_overlay_sell": "Crash Overlay 賣出",
            "crash_overlay_adjust": "Crash Overlay 調整",
        }
        reason_series = d["reason"].fillna("").astype(str).str.strip().str.lower()
        d["reason"] = reason_series.map(reason_map).fillna(reason_series)
        for extra_reason_col in ["crash_reason", "base_reason"]:
            if extra_reason_col in d.columns:
                extra_series = d[extra_reason_col].fillna("").astype(str).str.strip().str.lower()
                d[extra_reason_col] = extra_series.map(reason_map).fillna(extra_series)

    if "crash_triggered" in d.columns:
        d["crash_triggered"] = d["crash_triggered"].apply(
            lambda x: "是" if str(x).strip().lower() in {"1", "true", "t", "yes", "y", "是"} else ("否" if str(x).strip() != "" else "")
        )

    if "price" in d.columns:
        d["price"] = d["price"].apply(lambda x: _fmt_numeric_cell(x, decimals=1, as_int=False, fixed_decimals=True))
    for col in ["weight_change", "w_before", "w_after"]:
        if col in d.columns:
            d[col] = d[col].apply(_fmt_percent_cell)
    if "return" in d.columns:
        d["return"] = d["return"].apply(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
    for col in ["equity_pct", "invested_pct"]:
        if col in d.columns:
            d[col] = d[col].apply(_fmt_percent_cell)
    if "shares" in d.columns:
        d["shares"] = d["shares"].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    for col in ["fee", "net_amount"]:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: _fmt_numeric_cell(x, as_int=True))
    if "leverage_ratio" in d.columns:
        d["leverage_ratio"] = d["leverage_ratio"].apply(lambda x: _fmt_numeric_cell(x, decimals=1, as_int=False, fixed_decimals=True))
    for col in ["indicator_smaa", "indicator_base", "indicator_sd"]:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: _fmt_numeric_cell(x, decimals=1, as_int=False, fixed_decimals=True))

    for col in ["exec_notional", "equity_after", "position_value"]:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    if "delta_units" in d.columns:
        d["delta_units"] = d["delta_units"].apply(lambda x: f"{int(round(x)):,}" if pd.notnull(x) else "")
    if "strategy_version" in d.columns:
        d["strategy_version"] = d["strategy_version"].fillna("").astype(str)

    exist = [c for c in PREFER_ORDER if c in d.columns]
    others = [c for c in d.columns if c not in exist]
    d = d[exist + others]

    d = d.rename(columns={k: DISPLAY_NAME.get(k, k) for k in d.columns})
    return d

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# --------- Dash Layout ---------
app.layout = html.Div([
    dcc.Store(id='theme-store', data='theme-dark'),

    # === Header Controls ===
    html.Div([
        html.Button(id='theme-toggle', n_clicks=0, children='🌑 深色主題', className='btn btn-secondary main-header-bar'),
        html.Button(id='history-btn', n_clicks=0, children='📚 版本沿革', className='btn btn-secondary main-header-bar ms-2'),
    ], className='header-controls'),

    # 版本沿革模態框
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("各版本沿革紀錄")),
        dbc.ModalBody([
            dcc.Markdown(get_version_history_html(), dangerously_allow_html=True)
        ], className='version-history-modal-body'),
        dbc.ModalFooter(
            dbc.Button("關閉", id="history-close", className="ms-auto", n_clicks=0)
        ),
    ], id="history-modal", size="lg", is_open=False),

    # === Top Control Bar (代替側邊欄) ===
    dbc.Container([
        # Row 1: 基本回測參數
        dbc.Row(id='ctrl-row-basic', children=[
            dbc.Col([
                html.Label("股票代號", className="small mb-1"),
                dcc.Dropdown(id='ticker-dropdown', options=[{'label': t, 'value': t} for t in default_tickers],
                            value=default_tickers[0]),
            ], width=2),
            dbc.Col([
                html.Label("起始日期", className="small mb-1"),
                dcc.Input(id='start-date', type='text', value='2010-01-01'),
            ], width=1),
            dbc.Col([
                html.Label("結束日期", className="small mb-1"),
                dcc.Input(id='end-date', type='text', value='', placeholder='留空=最新'),
            ], width=1),
            dbc.Col([
                html.Label("券商折數", className="small mb-1"),
                dcc.Slider(id='discount-slider', min=0.1, max=0.7, step=0.01, value=0.3,
                          marks={0.1:'0.1',0.3:'0.3',0.5:'0.5',0.7:'0.7'},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], width=3),
            dbc.Col([
                html.Label("冷卻期 (bars)", className="small mb-1"),
                dcc.Input(id='cooldown-bars', type='number', min=0, max=20, value=3, style={'width': '70px'}),
            ], width=1),
            dbc.Col([
                html.Div([
                    dbc.Checkbox(id='bad-holding', value=False, label="賣出報酬率<-20%,等待下次賣點", className="small"),
                    dbc.Checkbox(id='auto-run', value=True, label="自動運算（參數變動即回測）", className="small mt-1"),
                    html.Label("勾選要隱藏的策略", className="small mt-2 mb-1"),
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

        # === 展開/收合按鈕 ===
        html.Div([
            dbc.Button(
                "⚙️ 顯示/隱藏 進階參數 (Crash Overlay、策略倉庫、Ensemble)",
                id="collapse-button",
                className="mb-2 w-100",
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
        ]),

        # === Collapsible Area: 包裹 Row 2 & Row 3 ===
        dbc.Collapse(
            id="collapse-settings",
            is_open=False,  # 預設關閉
            children=[
                # Row 2: 風險控制（保留 Crash Overlay / Smart Leverage）
                dbc.Row(id='ctrl-row-risk', children=[
                    dbc.Col([
                        html.Label("🔧 風險控制", id='label-risk-title', className="small mb-1 fw-bold"),
                        html.Label("測試選項", className="small mb-1"),
                        dbc.Checkbox(id='smart-leverage-switch', value=False, label="Smart Leverage (0050代替現金)", className="small text-success fw-bold"),
                        dbc.Checkbox(id='crash-overlay-switch', value=False, label="Crash-only Overlay", className="small text-warning fw-bold mt-1"),
                        dcc.Dropdown(
                            id='crash-overlay-preset',
                            options=CRASH_OVERLAY_PRESET_OPTIONS,
                            value=DEFAULT_CRASH_OVERLAY_PRESET,
                            clearable=False,
                            style={"marginTop": "4px", "fontSize": "12px"},
                        ),
                    ], width=12),
                ], className='p-2 mb-2', style={'borderRadius': '4px'}),

                # Row 3: 策略倉庫 & 兩組 Ensemble 參數 + 執行按鈕
                dbc.Row(id='ctrl-row-ensemble', children=[
                    # 倉庫選擇
                    dbc.Col([
                        html.Label("📂 策略倉庫版本", className="small mb-1 fw-bold"),
                        dcc.Dropdown(id='warehouse-dropdown', options=warehouse_options, value='strategy_warehouse.json'),
                    ], width=2),

                    # Ensemble_Majority 參數 (藍色)
                    dbc.Col([
                        html.Label("📊 Ensemble Majority", id='label-maj-title', className="small mb-1 fw-bold"),
                        html.Div([
                            html.Span("floor:", className="small me-1"),
                            dcc.Input(id='majority-floor', type='number', min=0.0, max=1.0, step=0.05, value=0.2,
                                     style={'width': '55px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("ema:", className="small me-1"),
                            dcc.Input(id='majority-ema-span', type='number', min=1, max=20, step=1, value=3,
                                     style={'width': '50px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("Δcap:", className="small me-1"),
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

                    # Ensemble_Proportional 參數 (綠色)
                    dbc.Col([
                        html.Label("📈 Ensemble Proportional", id='label-prop-title', className="small mb-1 fw-bold"),
                        html.Div([
                            html.Span("floor:", className="small me-1"),
                            dcc.Input(id='prop-floor', type='number', min=0.0, max=1.0, step=0.05, value=0.2,
                                     style={'width': '55px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("ema:", className="small me-1"),
                            dcc.Input(id='prop-ema-span', type='number', min=1, max=20, step=1, value=3,
                                     style={'width': '50px', 'display': 'inline-block', 'marginRight': '8px'}),
                            html.Span("Δcap:", className="small me-1"),
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

                    # 執行按鈕
                    dbc.Col([
                        html.Label("\u00A0", className="small mb-1"),
                        html.Button("🚀 一鍵執行所有回測", id='run-btn', n_clicks=0, className="btn btn-primary w-100"),
                    ], width=2),
                ], className='p-2 mb-2', style={'borderRadius': '4px'}),
            ]  # 結束 Collapse children
        ),  # 結束 Collapse

    ], fluid=True, className='mb-3'),

    # === Main Content Area ===
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Tabs(
                    id='main-tabs',
                    value='backtest',
                    children=[
                        dcc.Tab(label="策略回測", value="backtest"),
                        dcc.Tab(label="所有策略買賣點比較", value="compare"),
                        dcc.Tab(label="🔍 增強分析", value="enhanced"),
                        dcc.Tab(label="⚙️ 每日訊號戰情室", value="daily_signal"),
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

# --------- 進階參數區塊 展開/收合控制 ---------
@app.callback(
    Output("collapse-settings", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse-settings", "is_open")],
)
def toggle_advanced_settings(n, is_open):
    """控制進階參數區塊的展開/收合"""
    if n:
        return not is_open
    return is_open

# --------- 倉庫列表更新 ---------
@app.callback(
    Output('warehouse-dropdown', 'options'),
    Output('warehouse-dropdown', 'value'),
    Input('run-btn', 'n_clicks')
)
def update_warehouse_list(n):
    """填充倉庫下拉選單"""
    if manager:
        files = manager.list_warehouses()
        options = [
            {'label': '🟢 現役 (Active)' if f == "strategy_warehouse.json" else f, 'value': f}
            for f in files
        ]
        return options, "strategy_warehouse.json"
    return [], "strategy_warehouse.json"

# --------- 執行回測並存到 Store ---------
@app.callback(
    Output('backtest-store', 'data'),
    [
        Input('run-btn', 'n_clicks'),
        Input('auto-run', 'value'),
        Input('hide-strategy-presets', 'value'),
        Input('ticker-dropdown', 'value'),
        Input('start-date', 'value'),
        Input('end-date', 'value'),
        Input('discount-slider', 'value'),
        Input('cooldown-bars', 'value'),
        Input('bad-holding', 'value'),
        Input('crash-overlay-switch', 'value'),
        Input('crash-overlay-preset', 'value'),
        Input('warehouse-dropdown', 'value'),
        # Ensemble_Majority 參數
        Input('majority-floor', 'value'),
        Input('majority-ema-span', 'value'),
        Input('majority-delta-cap', 'value'),
        Input('majority-min-cooldown', 'value'),
        Input('majority-min-trade-dw', 'value'),
        # Ensemble_Proportional 參數
        Input('prop-floor', 'value'),
        Input('prop-ema-span', 'value'),
        Input('prop-delta-cap', 'value'),
        Input('prop-min-cooldown', 'value'),
        Input('prop-min-trade-dw', 'value'),
    ],
    State('backtest-store', 'data')
)
def run_backtest(n_clicks, auto_run, hidden_strategy_presets, ticker, start_date, end_date, discount, cooldown, bad_holding,
                crash_overlay_on, crash_overlay_preset, warehouse_file,
                maj_floor, maj_ema, maj_delta, maj_cooldown, maj_dw,
                prop_floor, prop_ema, prop_delta, prop_cooldown, prop_dw,
                stored_data):
    # === 調試日誌（僅在 DEBUG 級別時顯示）===
    logger.debug(f"run_backtest 被調用 - n_clicks: {n_clicks}, auto_run: {auto_run}, trigger: {ctx.triggered_id}")

    # 移除自動快取清理，避免多用户衝突
    # 讓 joblib.Memory 自動管理快取，只在需要時手動清理
    if n_clicks is None and not auto_run:
        logger.debug(f"早期返回：n_clicks={n_clicks}, auto_run={auto_run}")
        return stored_data

    # 載入數據
    df_raw, df_factor = load_data(ticker, start_date, end_date, "Self")
    if df_raw.empty:
        logger.warning(f"無法載入 {ticker} 的數據")
        return {"error": f"無法載入 {ticker} 的數據"}

    ctx_trigger = ctx.triggered_id

    # 只在 auto-run 為 True 或按鈕被點擊時運算
    if not auto_run and ctx_trigger != 'run-btn':
        logger.debug(f"跳過回測：auto_run={auto_run}, ctx_trigger={ctx_trigger}")
        return stored_data

    hidden_set = set(hidden_strategy_presets or [])
    active_strategy_names = [name for name in get_strategy_names(include_legacy=True) if name not in hidden_set]
    if not active_strategy_names:
        logger.warning("無可用策略：目前篩選後策略清單為空。")
        return {"error": "無可用策略，請取消部分隱藏項目或檢查 param_presets 設定。"}

    logger.info(
        f"開始執行回測 - ticker: {ticker}, 策略數: {len(active_strategy_names)}, 隱藏數: {len(hidden_set)}"
    )
    results = {}
    crash_overlay_key = (
        crash_overlay_preset
        if crash_overlay_preset in CRASH_OVERLAY_PRESETS
        else DEFAULT_CRASH_OVERLAY_PRESET
    )
    crash_overlay_params = CRASH_OVERLAY_PRESETS.get(crash_overlay_key, {})
    crash_market_df = None
    if crash_overlay_on:
        crash_db_path = str(crash_overlay_params.get("db_path", "twse_data.db"))
        if not os.path.exists(crash_db_path):
            logger.warning("[CrashOverlay] 資料庫不存在: %s，將跳過 crash-only 套用", crash_db_path)
            crash_overlay_on = False
        else:
            try:
                crash_market_df = _build_twse_market_features(db_path=crash_db_path)
                logger.info("[CrashOverlay] 已載入市場特徵: preset=%s, rows=%d", crash_overlay_key, len(crash_market_df))
            except Exception as exc:
                logger.warning("[CrashOverlay] 載入市場特徵失敗: %s", exc)
                crash_overlay_on = False

    for strat in active_strategy_names:
        # 只使用 param_presets 中的參數
        strat_params = param_presets[strat].copy()
        strat_type = strat_params["strategy_type"]
        smaa_src = strat_params.get("smaa_source", "Self")
        data_provider = strat_params.get("data_provider", "yfinance")
        pine_parity_mode = bool(strat_params.get("pine_parity_mode", False))

        # 為每個策略載入對應的數據
        df_raw, df_factor = load_data(
            ticker,
            start_date,
            end_date if end_date else None,
            smaa_source=smaa_src,
            data_provider=data_provider,
            pine_parity_mode=pine_parity_mode,
        )

        if strat_type == 'ssma_turn':
            calc_keys = [
                'linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win',
                'signal_filter_mode', 'volume_target_pass_rate', 'volume_target_lookback'
            ]
            ssma_params = {k: v for k, v in strat_params.items() if k in calc_keys}
            backtest_params = ssma_params.copy()
            backtest_params['stop_loss'] = strat_params.get('stop_loss', 0.0)

            # 重新計算策略信號（使用策略參數）
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_raw, df_factor, **ssma_params, smaa_source=smaa_src)
            if df_ind.empty:
                continue
            result = backtest_unified(df_ind, strat_type, backtest_params, buy_dates, sell_dates, discount=discount, trade_cooldown_bars=cooldown, bad_holding=bad_holding)
            indicator_daily_df = _build_indicator_daily_frame(
                df_ind,
                strat_type,
                strat_params,
                buy_dates=buy_dates,
                sell_dates=sell_dates,
            )
            if indicator_daily_df is not None and not indicator_daily_df.empty:
                result["indicator_daily"] = indicator_daily_df
            result["latest_t_signal"] = _extract_latest_t_signal(
                strategy_type=strat_type,
                indicator_daily=indicator_daily_df,
            )

            # 全局風險閥門已移除
            result['valve'] = {
                "applied": False,
                "cap": "N/A",
                "atr_ratio": "N/A"
            }
        elif strat_type == 'ensemble':# 使用新的 ensemble_runner 避免循環依賴
            result = {}
            if manager:
                # 🔥 使用選定的 warehouse_file
                active_strats = manager.load_strategies(warehouse_file)
                # 提取策略名稱 (去除 .csv 副檔名)
                strat_list = [s['name'].replace('.csv', '') for s in active_strats]

                # 🔥🔥🔥 增強型檔案搜尋 - 支援從 archive 找回遺失的 IS/OOS 檔案 🔥🔥🔥
                TRADES_DIR = Path("sss_backtest_outputs")

                # 定義搜尋路徑：優先找當前目錄，找不到再找 archive
                search_paths = [TRADES_DIR]
                archive_dir = Path("archive")
                if archive_dir.exists():
                    # 找出所有備份目錄中的 sss_backtest_outputs
                    archive_subdirs = list(archive_dir.glob("*/sss_backtest_outputs"))
                    search_paths.extend(archive_subdirs)
                    logger.info(f"📦 偵測到 {len(archive_subdirs)} 個備份目錄，將納入搜尋範圍")

                file_map = {}
                missing_strategies = []

                for s_name in strat_list:
                    found_file = None
                    # 遍歷所有可能的路徑
                    for search_dir in search_paths:
                        if not search_dir.exists():
                            continue
                        # 搜尋邏輯（寬鬆匹配）
                        # s_name 可能是 "trades_from_results_RMA_trial337" 或 "RMA_trial337"
                        candidates = list(search_dir.glob(f"*{s_name}*.csv"))
                        if candidates:
                            # 找到最新的那個（如果有重複）
                            found_file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                            location = "當前目錄" if search_dir == TRADES_DIR else f"備份 ({search_dir.parent.name})"
                            logger.info(f"   ✅ [{s_name}] 映射成功 -> {found_file.name} (來源: {location})")
                            break  # 找到了就停止搜尋其他目錄

                    if found_file:
                        file_map[s_name] = found_file
                    else:
                        missing_strategies.append(s_name)
                        logger.warning(f"   ⚠️ [{s_name}] 徹底遺失，無法在當前或封存目錄找到 CSV")

                # 報告搜尋結果
                logger.info(f"📊 [{strat}] 檔案映射統計: 成功 {len(file_map)}/{len(strat_list)}，遺失 {len(missing_strategies)}")

                if not file_map:
                    logger.error(f"❌ [{strat}] 無法找到任何有效的策略檔案，Ensemble 終止")
                    # 🔥 回傳錯誤結果到 UI，而不是靜默跳過
                    results[strat] = {
                        'error': f'找不到任何策略檔案 (共需要 {len(strat_list)} 個)',
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
                        'error_detail': f"倉庫 {warehouse_file} 中的策略檔案已被封存或遺失。請檢查 archive/ 目錄。"
                    }
                    continue

                if missing_strategies:
                    logger.warning(f"⚠️ [{strat}] 部分策略檔案遺失 ({len(missing_strategies)}/{len(strat_list)})，Ensemble 將使用可用的 {len(file_map)} 個策略繼續執行")
                    logger.warning(f"   遺失清單: {', '.join(missing_strategies[:5])}{'...' if len(missing_strategies) > 5 else ''}")

                if strat_list:
                    strat_params['strategies'] = strat_list
                    strat_params['file_map'] = file_map  # 🔥 傳遞檔案對應表
                    logger.info(f"[{strat}] 從 {warehouse_file} 注入策略: {strat_list}")
                else:
                    logger.warning(f"[{strat}] 倉庫 {warehouse_file} 是空的！Ensemble 可能無法運行")
            else:
                logger.error("無法載入 Strategy Manager")

            flat_params = {}
            try:
                from runners.ensemble_runner import run_ensemble_backtest
                from SSS_EnsembleTab import EnsembleParams, CostParams, RunConfig

                # 把 SSSv096 的巢狀參數攤平
                flat_params.update(strat_params.get('params', {}))
                flat_params.update(strat_params.get('trade_cost', {}))
                flat_params['method'] = strat_params.get('method', 'majority')
                flat_params['ticker'] = ticker

                # 使用比例門檻避免 N 變動時失真
                if 'majority_k' in flat_params and flat_params.get('method') == 'majority':
                    flat_params['majority_k_pct'] = 0.55
                    flat_params.pop('majority_k', None)
                    logger.info(f"[Ensemble] 使用比例門檻 majority_k_pct={flat_params['majority_k_pct']}")

                # 創建配置 - 根據策略類型使用對應的 UI 參數
                # Ensemble_Majority 使用藍色參數，Ensemble_Proportional 使用綠色參數
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
                    # 其他 Ensemble 策略使用預設值
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
                logger.info(f"[{strat}] UI參數: floor={ui_floor}, ema_span={ui_ema}, delta_cap={ui_delta}, cooldown={ui_cooldown}, dw={ui_dw}")

                cost_params = CostParams(
                    buy_fee_bp=flat_params.get("buy_fee_bp", 4.27),
                    sell_fee_bp=flat_params.get("sell_fee_bp", 4.27),
                    sell_tax_bp=flat_params.get("sell_tax_bp", 30.0)
                )

                cfg = RunConfig(
                    ticker=ticker,
                    method=flat_params.get("method", "majority"),
                    strategies=list(file_map.keys()),  # 🔥 只傳入有找到檔案的策略名
                    file_map=file_map,  # 🔥 傳入路徑對照表
                    params=ensemble_params,
                    cost=cost_params
                )

                # 傳遞比例門檻參數
                if flat_params.get("majority_k_pct"):
                    cfg.majority_k_pct = flat_params.get("majority_k_pct")
                else:
                    cfg.majority_k_pct = 0.55
                    logger.info(f"[Ensemble] 強制設定 majority_k_pct=0.55")

                logger.info(f"[Ensemble] 執行配置: ticker={ticker}, method={flat_params.get('method')}, majority_k_pct={flat_params.get('majority_k_pct', 'N/A')}")

                # 使用新的 ensemble_runner 執行
                backtest_result = run_ensemble_backtest(cfg)

                # 🔥🔥🔥 補強：計算完整績效指標（Ensemble 原生指標較陽春）🔥🔥🔥
                metrics = backtest_result.stats.copy() if backtest_result.stats else {}
                equity = backtest_result.equity_curve

                if equity is not None and not equity.empty and len(equity) > 1:
                    # 1. 基礎回報指標
                    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
                    days = (equity.index[-1] - equity.index[0]).days
                    years = max(days / 365.25, 0.1)
                    ann_ret = (1 + total_ret) ** (1 / years) - 1

                    # 2. 風險指標
                    roll_max = equity.cummax()
                    dd = (equity / roll_max) - 1
                    mdd = dd.min()

                    # 3. 夏普比率
                    daily_ret = equity.pct_change().dropna()
                    if len(daily_ret) > 0 and daily_ret.std() != 0:
                        sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5)
                    else:
                        sharpe = 0

                    # 4. 卡瑪比率
                    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

                    # 5. 索提諾比率
                    downside = daily_ret[daily_ret < 0]
                    if len(downside) > 0 and downside.std() != 0:
                        sortino = daily_ret.mean() / downside.std() * (252**0.5)
                    else:
                        sortino = 0

                    # 6. 波動率
                    if len(daily_ret) > 0:
                        annualized_volatility = daily_ret.std() * (252**0.5)
                    else:
                        annualized_volatility = 0

                    # 7. 更新 metrics（補上缺少的 key，解決 UI NoneType 錯誤）
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

                    logger.info(f"[Ensemble] 📊 補強後指標: 年化回報={ann_ret:.2%}, MDD={mdd:.2%}, Sharpe={sharpe:.2f}, Calmar={calmar:.2f}")
                else:
                    logger.warning(f"[Ensemble] ⚠️ 權益曲線不足，無法計算完整指標")

                # 轉換為舊格式以保持相容性
                result = {
                    'trades': [],
                    'trade_df': pack_df(backtest_result.trades),
                    'trades_df': pack_df(backtest_result.trades),
                    'signals_df': pack_df(backtest_result.trades[['trade_date', 'type', 'price']].rename(columns={'type': 'action'}) if not backtest_result.trades.empty else pd.DataFrame(columns=['trade_date', 'action', 'price'])),
                    'metrics': metrics,  # 🔥 使用補強後的 metrics
                    'equity_curve': pack_series(backtest_result.equity_curve),
                    'cash_curve': pack_series(backtest_result.cash_curve) if backtest_result.cash_curve is not None else "",
                    'weight_curve': pack_series(backtest_result.weight_curve) if backtest_result.weight_curve is not None else pack_series(pd.Series(0.0, index=backtest_result.equity_curve.index)),
                    'price_series': pack_series(backtest_result.price_series) if backtest_result.price_series is not None else pack_series(pd.Series(1.0, index=backtest_result.equity_curve.index)),
                    'daily_state': pack_df(backtest_result.daily_state),
                    'trade_ledger': pack_df(backtest_result.ledger),
                    'daily_state_std': pack_df(backtest_result.daily_state),
                    'trade_ledger_std': pack_df(backtest_result.ledger),
                    'valve': {
                        "applied": False,
                        "cap": "N/A",
                        "atr_ratio": "N/A"
                    },
                    'latest_t_signal': _extract_latest_t_signal(
                        strategy_type='ensemble',
                        daily_state=backtest_result.daily_state,
                    ),
                }

                logger.info(f"[Ensemble] 執行成功: 權益曲線長度={len(backtest_result.equity_curve)}, 交易數={len(backtest_result.ledger) if backtest_result.ledger is not None and not backtest_result.ledger.empty else 0}")

                # 🔥 驗證資料完整性
                if backtest_result.daily_state is not None and not backtest_result.daily_state.empty:
                    logger.info(f"[Ensemble] daily_state 欄位: {list(backtest_result.daily_state.columns)}, 前3列:\n{backtest_result.daily_state.head(3)}")
                else:
                    logger.warning(f"[Ensemble] ⚠️ daily_state 是空的或 None")

            except Exception as e:
                logger.error(f"Ensemble 策略執行失敗: {e}")
                # 創建空的結果
                result = {
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'trades_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0},
                    'equity_curve': pd.Series(1.0, index=df_raw.index),
                    'latest_t_signal': {'signal': 'HOLD', 'data_date': '', 'reason': 'ensemble_error'},
                }

            # === 修復 3：添加調試日誌，核對子策略集合是否一致 ===
            logger.info(f"[Ensemble] 執行完成，ticker={ticker}, method={flat_params.get('method')}")
            if 'equity_curve' in result and hasattr(result['equity_curve'], 'shape'):
                logger.info(f"[Ensemble] 權益曲線長度: {len(result['equity_curve'])}")
            if 'trade_df' in result and hasattr(result['trade_df'], 'shape'):
                logger.info(f"[Ensemble] 交易記錄數量: {len(result['trade_df'])}")
        else:
            if strat_type == 'single':
                df_ind = compute_single(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["devwin"], smaa_source=smaa_src)
            elif strat_type == 'dual':
                df_ind = compute_dual(df_raw, df_factor, strat_params["linlen"], strat_params["factor"], strat_params["smaalen"], strat_params["short_win"], strat_params["long_win"], smaa_source=smaa_src)
            elif strat_type == 'RMA':
                df_ind = compute_RMA(
                    df_raw,
                    df_factor,
                    strat_params["linlen"],
                    strat_params["factor"],
                    strat_params["smaalen"],
                    strat_params["rma_len"],
                    strat_params["dev_len"],
                    smaa_source=smaa_src,
                    pine_parity_mode=pine_parity_mode,
                )
            if df_ind.empty:
                continue
            result = backtest_unified(df_ind, strat_type, strat_params, discount=discount, trade_cooldown_bars=cooldown, bad_holding=bad_holding)
            indicator_daily_df = _build_indicator_daily_frame(df_ind, strat_type, strat_params)
            if indicator_daily_df is not None and not indicator_daily_df.empty:
                result["indicator_daily"] = indicator_daily_df
            result["latest_t_signal"] = _extract_latest_t_signal(
                strategy_type=strat_type,
                indicator_daily=indicator_daily_df,
            )

            result['valve'] = {
                "applied": False,
                "cap": "N/A",
                "atr_ratio": "N/A"
            }
        result["trade_cooldown_bars_used"] = int(cooldown or 0)
        # 統一使用 orient="split" 打包，避免重複序列化
        # 注意：Ensemble 策略已經在 pack_df/pack_series 中處理過，這裡只處理單策略
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

        # << 新增：一律做最後保險打包，補上 daily_state / weight_curve 等 >>
        result = _pack_result_for_store(result)

        # === Crash-only Overlay（可選）：在 baseline/valve 後再覆蓋一次 ===
        if crash_overlay_on:
            try:
                result = _apply_crash_overlay_to_result(
                    result=result,
                    df_raw=df_raw,
                    strat_params=strat_params if isinstance(strat_params, dict) else {},
                    ticker=ticker,
                    preset_name=crash_overlay_key,
                    overlay_params=crash_overlay_params,
                    market_df=crash_market_df,
                )
                valve_meta = result.get("valve", {}) if isinstance(result, dict) else {}
                if valve_meta.get("mode") == "crash_only" and valve_meta.get("applied"):
                    logger.info(
                        "[%s] Crash-only overlay 已套用: preset=%s, trigger_days=%s, trigger_rate=%.2f%%",
                        strat,
                        crash_overlay_key,
                        valve_meta.get("trigger_days", "N/A"),
                        float(valve_meta.get("trigger_rate", 0.0)) * 100.0,
                    )
            except Exception as exc:
                logger.warning("[%s] Crash-only overlay 套用失敗: %s", strat, exc)

        results[strat] = result

    # 使用第一個策略的數據作為主要顯示數據
    first_strat = list(results.keys())[0] if results else active_strategy_names[0]
    first_smaa_src = param_presets[first_strat].get("smaa_source", "Self")
    first_provider = param_presets[first_strat].get("data_provider", "yfinance")
    first_pine_parity = bool(param_presets[first_strat].get("pine_parity_mode", False))
    df_raw_main, _ = load_data(
        ticker,
        start_date,
        end_date if end_date else None,
        smaa_source=first_smaa_src,
        data_provider=first_provider,
        pine_parity_mode=first_pine_parity,
    )

    # 統一使用 orient="split" 序列化，確保一致性
    payload = {
        'results': results,
        'df_raw': df_raw_main.to_json(date_format='iso', orient='split'),
        'ticker': ticker
    }

    # 防守性檢查：如還有漏網的非序列化物件就能提早看出
    try:
        json.dumps(payload)
    except Exception as e:
        logger.exception("[BUG] backtest-store payload 仍含不可序列化物件：%s", e)
        # 如果要強制不噴，可做 fallback：json.dumps(..., default=str) 但通常不建議吞掉

    # === 回測完成日誌 ===
    logger.info(f"回測完成 - 策略數: {len(results)}, ticker: {ticker}, 數據行數: {len(df_raw_main)}")
    logger.debug(f"策略列表: {list(results.keys())}")

    return payload

# --------- 主頁籤內容顯示 ---------
@app.callback(
    Output('tab-content', 'children'),
    Input('backtest-store', 'data'),
    Input('main-tabs', 'value'),
    Input('theme-store', 'data'),
    Input('smart-leverage-switch', 'value'),  # 🔥 新增 Smart Leverage 開關
    Input('hide-strategy-presets', 'value'),
    Input('ticker-dropdown', 'value'),
)
def update_tab(data, tab, theme, smart_leverage_on, hidden_strategy_presets, ticker_for_daily):
    # 確保 pandas 可用
    import pandas as pd

    # === 調試日誌（僅在 DEBUG 級別時顯示）===
    logger.debug(f"update_tab 被調用 - tab: {tab}")
    # === 根據主題決定顏色變數 ===
    if theme == 'theme-light':
        # 淺色模式配色
        plotly_template = 'plotly_white'
        bg_color = '#ffffff'
        font_color = '#212529'

        # 卡片與表格配色
        card_bg = '#f8f9fa'       # 淺灰卡片
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
        # 藍色模式配色
        plotly_template = 'plotly_dark'
        bg_color = '#001a33'
        font_color = '#ffe066'

        # 卡片與表格配色
        card_bg = '#002b4d'       # 深藍卡片
        card_border = '#004080'
        card_text = '#ffe066'

        table_header_bg = '#003366'
        table_cell_bg = '#001a33'
        table_text = '#ffe066'
        table_border = '#004080'

        legend_bgcolor = 'rgba(0, 26, 51, 0.8)'
        legend_bordercolor = '#ffe066'
        legend_font_color = '#ffe066'

    else:  # theme-dark (預設)
        plotly_template = 'plotly_dark'
        bg_color = '#121212'
        font_color = '#e0e0e0'

        # 卡片與表格配色
        card_bg = '#1e1e1e'       # 深灰卡片
        card_border = '#333'
        card_text = '#fff'

        table_header_bg = '#2d2d2d'
        table_cell_bg = '#1e1e1e'
        table_text = '#e0e0e0'
        table_border = '#444'

        legend_bgcolor = 'rgba(30,30,30,0.8)'
        legend_bordercolor = '#fff'
        legend_font_color = '#fff'

    if tab == "daily_signal":
        daily_ticker = ticker_for_daily
        if not daily_ticker and isinstance(data, dict):
            daily_ticker = data.get("ticker")
        if not daily_ticker:
            daily_ticker = default_tickers[0]

        context = build_daily_signal_context(
            ticker=daily_ticker,
            hidden_strategy_presets=hidden_strategy_presets,
        )
        req_df = context.requirements_df.copy()
        display_cols = [
            "strategy_name",
            "strategy_type",
            "symbol",
            "field",
            "min_bars",
            "available_bars",
            "ready",
            "notes",
        ]
        for col in display_cols:
            if col not in req_df.columns:
                req_df[col] = ""
        req_df = req_df[display_cols]
        req_df["ready"] = req_df["ready"].apply(lambda x: "可用" if bool(x) else "不足")

        req_col_name = {
            "strategy_name": "策略名稱",
            "strategy_type": "策略類型",
            "symbol": "代號",
            "field": "欄位",
            "min_bars": "最少資料根數",
            "available_bars": "目前可用根數",
            "ready": "狀態",
            "notes": "備註",
        }

        requirements_table = dash_table.DataTable(
            columns=[{"name": req_col_name.get(c, c), "id": c} for c in req_df.columns],
            data=req_df.to_dict("records"),
            style_table={"overflowX": "auto", "backgroundColor": bg_color},
            style_header={
                "backgroundColor": table_header_bg,
                "color": table_text,
                "border": f"1px solid {table_border}",
                "fontWeight": "bold",
            },
            style_cell={
                "backgroundColor": table_cell_bg,
                "color": table_text,
                "textAlign": "center",
                "border": f"1px solid {table_border}",
                "padding": "6px",
            },
            style_data_conditional=[
                {"if": {"filter_query": "{ready} = 不足"}, "color": "#ff6b6b", "fontWeight": "bold"}
            ],
            page_size=15,
            id="daily-requirements-table",
        )

        estimate_inputs = []
        for item in context.input_schema:
            ready = bool(item.get("ready"))
            latest_value = item.get("latest_value")
            required_by = ", ".join(item.get("required_by", []))
            estimate_inputs.append(
                html.Div(
                    [
                        html.Div(
                            f"{item['symbol']} {item['field']}（最少 {item['min_bars']} 根，目前 {item['available_bars']} 根）",
                            style={
                                "fontSize": "12px",
                                "fontWeight": "bold",
                                "color": "#28a745" if ready else "#ff6b6b",
                            },
                        ),
                        dcc.Input(
                            id={"type": "daily-estimate-input", "key": item["key"]},
                            type="number",
                            value=latest_value,
                            debounce=True,
                            style={"width": "180px", "marginTop": "4px"},
                        ),
                        html.Div(
                            f"使用策略：{required_by}",
                            style={"fontSize": "11px", "color": "#888", "marginTop": "2px"},
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "width": "320px",
                        "marginRight": "12px",
                        "marginBottom": "10px",
                        "verticalAlign": "top",
                    },
                )
            )

        return html.Div(
            [
                html.H4("📡 每日訊號戰情室（param_presets 真實/實驗）"),
                html.Div(
                    f"計算日期（台北時區）：{context.run_date.strftime('%Y-%m-%d')} | 標的：{daily_ticker}",
                    style={"color": "#888", "marginBottom": "10px"},
                ),
                html.H5("策略需求矩陣"),
                requirements_table,
                html.Hr(),
                html.H5("估算輸入（實驗模式）"),
                html.Div(estimate_inputs if estimate_inputs else "目前沒有需要輸入的估算欄位。"),
                html.Button(
                    "🚀 計算今日訊號（真實 + 實驗）",
                    id="btn-run-prediction",
                    n_clicks=0,
                    className="btn btn-danger",
                    style={"marginTop": "8px", "marginBottom": "12px"},
                ),
                html.Div(id="prediction-status-msg"),
                html.Div(id="daily-signal-results", style={"marginTop": "12px"}),
            ],
            style={"padding": "20px"},
        )

    if not data:
        logger.warning("沒有回測數據，顯示提示訊息")
        return html.Div("請先執行回測")

    # data 現在已經是 dict，不需要 json.loads
    results = data['results']
    df_raw = df_from_pack(data['df_raw'])  # 使用 df_from_pack 統一解包
    ticker = data['ticker']
    strategy_names = list(results.keys())

    logger.debug(f"數據解析完成 - 策略數: {len(strategy_names)}, ticker: {ticker}, 數據行數: {len(df_raw) if df_raw is not None else 0}")

    if tab == "backtest":
        # 創建策略回測的子頁籤
        strategy_tabs = []

        for strategy in strategy_names:
            result = results.get(strategy)
            if not result:
                continue

            # 🔥 檢查是否有錯誤訊息（例如 Ensemble 找不到檔案）
            if 'error' in result:
                error_msg = result.get('error', '未知錯誤')
                error_detail = result.get('error_detail', '')
                strategy_tabs.append(
                    dcc.Tab(
                        label=f"❌ {strategy}",
                        value=strategy,
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        children=[
                            html.Div([
                                dbc.Alert([
                                    html.H4("⚠️ 回測執行失敗", className="alert-heading"),
                                    html.P(error_msg, style={"font-weight": "bold", "font-size": "16px"}),
                                    html.Hr(),
                                    html.P(error_detail, className="mb-0"),
                                    html.Br(),
                                    html.P([
                                        "💡 可能原因：",
                                        html.Ul([
                                            html.Li("策略檔案已被封存到 archive/ 目錄"),
                                            html.Li("倉庫版本與當前資料不匹配"),
                                            html.Li("策略尚未執行回測，缺少必要的 CSV 檔案"),
                                        ])
                                    ]),
                                    html.P([
                                        "🔧 建議操作：",
                                        html.Ul([
                                            html.Li("檢查 archive/ 目錄中是否有相關備份"),
                                            html.Li("切換到其他倉庫版本試試"),
                                            html.Li("或重新執行 run_full_pipeline 產生新的策略檔案"),
                                        ])
                                    ])
                                ], color="warning", style={"margin": "20px"})
                            ], style={"padding": "20px"})
                        ]
                    )
                )
                continue  # 跳過正常的渲染流程

            # === 統一入口：讀取交易表、日狀態、權益曲線 ===
            # 讀交易表的統一入口：先用標準鍵，再 fallback
            trade_df = None
            candidates = [
                result.get('trades'),      # 全局覆寫後標準鍵
                result.get('trades_ui'),   # 舊格式（若還存在）
                result.get('trade_df'),    # 某些策略自帶
            ]

            for cand in candidates:
                if cand is None:
                    continue
                # cand 可能已是 DataFrame 或打包字串
                df = df_from_pack(cand) if isinstance(cand, str) else cand
                if df is not None and getattr(df, 'empty', True) is False:
                    trade_df = df.copy()
                    break

            if trade_df is None:
                # 建立空表避免後續崩
                trade_df = pd.DataFrame(columns=['trade_date','type','price','shares','return'])

            # app_dash.py / 2025-08-22 16:00
            # 取用 daily_state：優先使用套閥版本，其次原始，最後 baseline（與 O2 一致）
            daily_state_std = None

            # 🔥 診斷：顯示可用的 daily_state 來源
            available_sources = [k for k in ['daily_state_valve', 'daily_state_std', 'daily_state', 'daily_state_base'] if result.get(k)]
            logger.info(f"[{strategy}] 可用的 daily_state 來源: {available_sources}")

            if result.get('daily_state_valve'):
                daily_state_std = df_from_pack(result['daily_state_valve'])
                logger.info(f"[{strategy}] 使用 daily_state_valve，解包後形狀: {daily_state_std.shape if daily_state_std is not None else 'None'}")
            elif result.get('daily_state_std'):
                daily_state_std = df_from_pack(result['daily_state_std'])
                logger.info(f"[{strategy}] 使用 daily_state_std，解包後形狀: {daily_state_std.shape if daily_state_std is not None else 'None'}")
            elif result.get('daily_state'):
                daily_state_std = df_from_pack(result['daily_state'])
                logger.info(f"[{strategy}] 使用 daily_state，解包後形狀: {daily_state_std.shape if daily_state_std is not None else 'None'}")
            elif result.get('daily_state_base'):
                daily_state_std = df_from_pack(result['daily_state_base'])
                logger.info(f"[{strategy}] 使用 daily_state_base，解包後形狀: {daily_state_std.shape if daily_state_std is not None else 'None'}")
            else:
                daily_state_std = pd.DataFrame()
                logger.warning(f"[{strategy}] ⚠️ 沒有找到任何 daily_state 來源，使用空 DataFrame")

            # app_dash.py / 2025-08-22 16:00
            # 取用 trade_ledger：優先使用套閥版本，其次原始，最後 baseline（與 O2 一致）
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

            # 記錄來源選擇結果
            logger.info(f"[UI] {strategy} trades 來源優先序：trades -> trades_ui -> trade_df；實際使用={'trades' if 'trades' in result else ('trades_ui' if 'trades_ui' in result else 'trade_df')}")
            logger.info(f"[UI] {strategy} 讀取後前 3 列 w: {daily_state_std['w'].head(3).tolist() if 'w' in daily_state_std.columns else 'N/A'}")

            # 標準化交易資料，確保有統一的 trade_date/type/price 欄位
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
                logger.info(f"標準化後 trades_ui 欄位: {list(trade_df.columns)}")
            except Exception as e:
                logger.warning(f"無法使用 sss_core 標準化，使用後備方案: {e}")
                # 後備標準化方案
                if trade_df is not None and len(trade_df) > 0:
                    trade_df = trade_df.copy()
                    trade_df.columns = [str(c).lower() for c in trade_df.columns]

                    # 確保有 trade_date 欄
                    if "trade_date" not in trade_df.columns:
                        if "date" in trade_df.columns:
                            trade_df["trade_date"] = pd.to_datetime(trade_df["date"], errors="coerce")
                        elif isinstance(trade_df.index, pd.DatetimeIndex):
                            trade_df = trade_df.reset_index().rename(columns={"index": "trade_date"})
                        else:
                            trade_df["trade_date"] = pd.NaT
                    else:
                        trade_df["trade_date"] = pd.to_datetime(trade_df["trade_date"], errors="coerce")

                    # 確保有 type 欄
                    if "type" not in trade_df.columns:
                        if "action" in trade_df.columns:
                            trade_df["type"] = trade_df["action"].astype(str).str.lower()
                        elif "side" in trade_df.columns:
                            trade_df["type"] = trade_df["side"].astype(str).str.lower()
                        else:
                            trade_df["type"] = "hold"

                    # 確保有 price 欄
                    if "price" not in trade_df.columns:
                        for c in ["open", "price_open", "exec_price", "px", "close"]:
                            if c in trade_df.columns:
                                trade_df["price"] = trade_df[c]
                                break
                        if "price" not in trade_df.columns:
                            trade_df["price"] = 0.0

            # 型別對齊：保證 trade_date 為 Timestamp，price/shares 為 float
            if 'trade_date' in trade_df.columns:
                trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            if 'signal_date' in trade_df.columns:
                trade_df['signal_date'] = pd.to_datetime(trade_df['signal_date'])
            if 'price' in trade_df.columns:
                trade_df['price'] = pd.to_numeric(trade_df['price'], errors='coerce')
            if 'shares' in trade_df.columns:
                trade_df['shares'] = pd.to_numeric(trade_df['shares'], errors='coerce')

            # === 新：若有 trade_ledger，優先顯示更完整的欄位 ===
            ledger_df = df_from_pack(result.get('trade_ledger'))
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                ledger_ui = norm(ledger_df) if ledger_df is not None and len(ledger_df)>0 else pd.DataFrame()
            except Exception:
                ledger_ui = ledger_df if ledger_df is not None else pd.DataFrame()

            # === 修正：優先使用標準化後的 trade_ledger_std ===
            # 使用 utils_payload 標準化後的結果，確保欄位齊全
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

            # 記錄原始欄位（偵錯用）
            logger.info("[UI] trade_df 原始欄位：%s", list(base_df.columns) if base_df is not None else None)
            logger.info("[UI] trade_ledger_std 原始欄位：%s", list(trade_ledger_std.columns) if trade_ledger_std is not None else None)

            # 為了 100% 保證 weight_change 出現，先確保權重欄位
            base_df = _ensure_weight_columns(base_df)
            # 使用新的統一格式化函式
            display_df = format_trade_like_df_for_display(base_df)

            # === 交易流水帳(ledger)表格：先準備顯示版 ===
            ledger_src = trade_ledger_std if (trade_ledger_std is not None and not trade_ledger_std.empty) else \
                         (ledger_ui if (ledger_ui is not None and not ledger_ui.empty) else pd.DataFrame())

            if ledger_src is not None and not ledger_src.empty:
                # 為了 100% 保證 weight_change 出現，先確保權重欄位
                ledger_src = _ensure_weight_columns(ledger_src)
                # 使用新的統一格式化函式
                ledger_display = format_trade_like_df_for_display(ledger_src)
                ledger_columns = [{"name": i, "id": i} for i in ledger_display.columns]
                ledger_data = ledger_display.to_dict('records')
            else:
                ledger_columns = []
                ledger_data = []

            # ==============================================================================
            # 🔥 關鍵修正：先處理 Smart Leverage 並更新 Metrics，再生成 KPI 卡片
            # ==============================================================================

            # 先決定使用哪個 daily_state
            daily_state = df_from_pack(
                result.get('daily_state_valve') or result.get('daily_state')
            )

            # 優先使用標準化後的資料
            if daily_state_std is not None and not daily_state_std.empty:
                ds_for_calc = daily_state_std
                logger.info(f"[UI] 使用標準化後的 daily_state_std 進行 Smart Leverage 計算")
            else:
                ds_for_calc = daily_state
                logger.info(f"[UI] 使用原始 daily_state 進行 Smart Leverage 計算")

            # 如果勾選了 Smart Leverage，立即計算並更新 Metrics
            if smart_leverage_on and ds_for_calc is not None and not ds_for_calc.empty and 'w' in ds_for_calc.columns:
                logger.info(f"[{strategy}] 啟用 Smart Leverage 計算 (更新指標與圖表)...")

                # A. 計算新淨值 (0050 替代現金)
                smart_ds = calculate_smart_leverage_equity(ds_for_calc, df_raw, safe_ticker="0050.TW")

                # B. 替換掉原本的變數，後面的畫圖會用到
                if daily_state_std is not None and not daily_state_std.empty:
                    daily_state_std = smart_ds
                else:
                    daily_state = smart_ds

                ds_for_calc = smart_ds

                # C. 重算 Metrics (讓 KPI 卡片數字變更)
                try:
                    eq = smart_ds['equity']

                    # 總報酬
                    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1

                    # 年化報酬
                    days = (eq.index[-1] - eq.index[0]).days
                    years = max(days / 365.25, 0.1)
                    ann_ret = (1 + total_ret) ** (1 / years) - 1

                    # MDD
                    roll_max = eq.cummax()
                    dd = (eq / roll_max) - 1
                    mdd = dd.min()

                    # Sharpe (簡易版)
                    daily_ret = eq.pct_change().dropna()
                    sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if daily_ret.std() != 0 else 0

                    # 強制更新 result['metrics']
                    if 'metrics' not in result:
                        result['metrics'] = {}

                    result['metrics']['total_return'] = total_ret
                    result['metrics']['annual_return'] = ann_ret
                    result['metrics']['max_drawdown'] = mdd
                    result['metrics']['sharpe_ratio'] = sharpe

                    logger.info(f"[{strategy}] Metrics 重算完成: Ret={total_ret:.2%}, Ann={ann_ret:.2%}, MDD={mdd:.2%}, Sharpe={sharpe:.2f}")

                except Exception as e:
                    logger.error(f"[{strategy}] 重算 Metrics 失敗: {e}")

            # ==============================================================================
            # 🔥 現在 result['metrics'] 已經是最新的了，可以生成 KPI 卡片
            # ==============================================================================

            metrics = result.get('metrics', {})
            tooltip = f"{strategy} 策略説明"
            strategy_cfg = param_presets.get(strategy, {})
            strategy_type = str(strategy_cfg.get("strategy_type", "")).strip().lower()
            param_display = {k: v for k, v in strategy_cfg.items() if k != "strategy_type"}

            def _compact_param_value(v: Any) -> str:
                if isinstance(v, dict):
                    return "{" + ", ".join(f"{kk}={_compact_param_value(vv)}" for kk, vv in v.items()) + "}"
                if isinstance(v, list):
                    return "[" + ", ".join(_compact_param_value(x) for x in v) + "]"
                if isinstance(v, float):
                    if pd.isna(v) or np.isinf(v):
                        return "N/A"
                    return f"{v:.6g}"
                return str(v)

            param_str = " | ".join(f"{k}={_compact_param_value(v)}" for k, v in param_display.items())
            param_line_text = f"參數設定: {param_str}"
            avg_holding = calculate_holding_periods(trade_df)
            metrics['avg_holding_period'] = avg_holding
            daily_state_hint = daily_state_std if (daily_state_std is not None and not daily_state_std.empty) else daily_state
            _ensure_exposure_metrics(metrics, result, daily_state_hint=daily_state_hint)
            nav_rel_ratio = _calc_nav_relative_ratio(ds_for_calc, df_raw)
            metrics["nav_relative_ratio"] = float(nav_rel_ratio) if pd.notna(nav_rel_ratio) else np.nan

            label_map = {
                "total_return": "總回報率",
                "annual_return": "年化回報率",
                "win_rate": "勝率",
                "max_drawdown": "最大回撤",
                "max_drawdown_duration": "回撤持續",
                "calmar_ratio": "卡瑪比率",
                "sharpe_ratio": "夏普比率",
                "sortino_ratio": "索提諾比率",
                "payoff_ratio": "盈虧比",
                "profit_factor": "盈虧因子",
                "time_in_market": "在場比例",
                "turnover_py": "年化換手率",
                "num_trades": "交易次數",
                "avg_holding_period": "平均持倉天數",
                "nav_relative_ratio": "相對淨值比",
                "annualized_volatility": "年化波動率",
                "max_consecutive_wins": "最大連續盈利",
                "max_consecutive_losses": "最大連續虧損",
                "avg_win": "平均盈利",
                "avg_loss": "平均虧損",
            }

            # 🔥 安全格式化輔助函式 - 處理 None/NaN/Inf 值
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

            metric_display_order = [
                "total_return",
                "annual_return",
                "max_drawdown",
                "calmar_ratio",
                "sharpe_ratio",
                "sortino_ratio",
                "annualized_volatility",
                "win_rate",
                "profit_factor",
                "payoff_ratio",
                "num_trades",
                "avg_holding_period",
                "time_in_market",
                "turnover_py",
                "max_drawdown_duration",
                "nav_relative_ratio",
                "max_consecutive_wins",
                "max_consecutive_losses",
                "avg_win",
                "avg_loss",
            ]
            ordered_metric_keys = [k for k in metric_display_order if k in metrics]
            ordered_metric_keys.extend(sorted([k for k in metrics.keys() if k not in ordered_metric_keys]))

            metric_cards = []
            for k in ordered_metric_keys:
                v = metrics.get(k)
                if k == "nav_relative_ratio":
                    txt = safe_fmt(v, is_pct=False, suffix="x")
                elif k in ["total_return", "annual_return", "win_rate", "max_drawdown", "annualized_volatility", "avg_win", "avg_loss", "time_in_market", "turnover_py"]:
                    txt = safe_fmt(v, is_pct=True)
                elif k in ["calmar_ratio", "sharpe_ratio", "sortino_ratio", "payoff_ratio", "profit_factor"]:
                    txt = safe_fmt(v, is_pct=False)
                elif k in ["max_drawdown_duration", "avg_holding_period"]:
                    txt = safe_fmt(v, is_pct=False, suffix=" 天")
                elif k in ["num_trades", "max_consecutive_wins", "max_consecutive_losses"]:
                    txt = safe_fmt(v, is_int=True)
                else:
                    txt = safe_fmt(v)
                metric_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(label_map.get(k, k), className="card-title-label", style={"color": card_text}),
                                html.Div(txt, style={"font-weight": "bold", "font-size": "20px", "color": card_text})
                            ])
                        ], style={"background": card_bg, "border": f"1px solid {card_border}", "border-radius": "8px", "margin-bottom": "6px"})
                    ], xs=12, sm=6, md=4, lg=2, xl=2, style={"minWidth": "100px", "margin-bottom": "6px", "maxWidth": "12.5%"})
                )

            # 🎯 使用統一圖表系統（三圖同步縮放）
            from sss_core.plotting_unified import create_unified_dashboard

            # 優先使用標準化後的資料（已在前面更新過），確保欄位完整
            if daily_state_std is not None and not daily_state_std.empty:
                daily_state_display = daily_state_std
                logger.info(f"[UI] 繪圖使用 daily_state_std，欄位: {list(daily_state_std.columns)}")
            else:
                daily_state_display = daily_state
                logger.info(f"[UI] 繪圖使用原始 daily_state，欄位: {list(daily_state.columns) if daily_state is not None else None}")

            # 檢查點（快速自查）
            logger.info(f"[UI] trade_df cols={list(trade_df.columns)} head=\n{trade_df.head(3)}")

            # ✅ 欄位語意統一
            daily_state_display = normalize_daily_state_columns(daily_state_display)

            # 🔧 修正 log 檢查（原本錯用 daily_state.columns）
            logger.info(f"[UI] daily_state_display cols={list(daily_state_display.columns) if daily_state_display is not None else None}")
            if daily_state_display is not None:
                has_cols = {"equity"}.issubset(daily_state_display.columns)
                logger.info(f"[UI] daily_state_display head=\n{daily_state_display[['equity']].head(3) if has_cols else 'Missing equity columns'}")

            vote_series = None
            vote_threshold = None
            vote_diagnostics = _new_vote_diagnostics()
            vote_warning_component = html.Div()
            indicator_daily = df_from_pack(result.get("indicator_daily"))
            crash_state_series = series_from_pack(result.get("crash_state"))
            crash_debug_df = df_from_pack(result.get("crash_debug_df"))

            # === 🎯 建立統一圖表（三圖同步） ===
            if daily_state_display is not None and not daily_state_display.empty and {"equity"}.issubset(daily_state_display.columns):
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

                # 使用統一圖表系統
                unified_fig = create_unified_dashboard(
                    df_raw=df_raw,
                    daily_state=daily_state_display,
                    trade_df=base_df,
                    ticker=ticker,
                    theme='dark' if theme == 'theme-dark' else 'light',
                    votes_series=vote_series,
                    votes_threshold=vote_threshold,
                    indicator_df=indicator_daily,
                    show_raw_signal_highlight=not DISABLE_SMAA_RAW_HIGHLIGHT,
                    crash_state=crash_state_series if ENABLE_CASH_ONLY_XRAY_SUBPLOTS else None,
                    crash_trade_df=base_df,
                )

                # 原始的分離圖表（已註解）
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

                # === 新增：資金權重表格 ===
                # 使用標準化後的 daily_state_display（已經標準化過了）
                # 準備資金權重表格數據
                if not daily_state_display.empty:
                    # 選擇要顯示的欄位（與 Streamlit 一致）
                    display_cols = ["portfolio_value", "position_value", "invested_pct", "w"]
                    available_cols = [col for col in display_cols if col in daily_state_display.columns]

                    if available_cols:
                        # 格式化數據用於顯示
                        display_daily_state = daily_state_display[available_cols].copy()
                        display_daily_state.index = display_daily_state.index.strftime('%Y-%m-%d')

                        # 格式化數值
                        for col in ["portfolio_value", "position_value"]:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(
                                    lambda x: f"{int(round(x)):,}" if pd.notnull(x) and not pd.isna(x) else ""
                                )

                        for col in ["invested_pct"]:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(
                                    lambda x: f"{x:.2%}" if pd.notnull(x) else ""
                                )

                        for col in ['w']:
                            if col in display_daily_state.columns:
                                display_daily_state[col] = display_daily_state[col].apply(
                                    lambda x: f"{x:.2%}" if pd.notnull(x) else ""
                                )

                        # 創建資金權重表格
                        daily_state_table = html.Div([
                            html.H5("總資產配置", style={"marginTop": "16px", "color": font_color}),
                            html.Div("每日資產配置狀態，包含總資產、倉位市值、投資比例等",
                                     style={"fontSize": "14px", "color": font_color, "marginBottom": "8px"}),
                            dash_table.DataTable(
                                columns=[{"name": i, "id": i} for i in display_daily_state.columns],
                                data=display_daily_state.head(20).to_dict('records'),  # 只顯示前20筆
                                style_table={'overflowX': 'auto', 'backgroundColor': bg_color},
                                style_cell={'textAlign': 'center', 'backgroundColor': table_cell_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                                style_header={'backgroundColor': table_header_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                                id={'type': 'daily-state-table', 'strategy': strategy}
                            ),
                            html.Div(f"顯示前20筆記錄，共{len(display_daily_state)}筆",
                                     style={"fontSize": "12px", "color": "#888", "textAlign": "center", "marginTop": "8px"})
                        ])
                    else:
                        daily_state_table = html.Div("資金權重資料不足", style={"color": "#888", "fontStyle": "italic"})
                else:
                    daily_state_table = html.Div("資金權重資料為空", style={"color": "#888", "fontStyle": "italic"})
            else:
                # 回退：沒有 daily_state，使用空白統一圖表
                logger.info("[UI] 使用 fallback：daily_state 為空，建立空白圖表")
                unified_fig = go.Figure()
                unified_fig.add_annotation(
                    text="資料不足，無法繪製圖表",
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

                # 原始的 fallback 邏輯（已註解）
                # fig2 = plot_equity_cash(trade_df, df_raw)
                # if daily_state_display is not None and not daily_state_display.empty and 'w' in daily_state_display.columns:
                #     fig_w = plot_weight_series(daily_state_display['w'], title="持有權重變化")
                #     fig_w.update_layout(...)
                # else:
                #     fig_w = go.Figure()

                daily_state_table = html.Div("使用交易表重建的權益曲線", style={"color": "#888", "fontStyle": "italic"})

            # 原始的 fig2 layout 更新（已不需要）
            # fig2.update_layout(
            #     template=plotly_template, font_color=font_color, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            #     legend_font_color=legend_font_color,
            #     legend=dict(bgcolor=legend_bgcolor, bordercolor=legend_bordercolor, font=dict(color=legend_font_color))
            # )

            crash_xray_section = html.Div()
            if ENABLE_CASH_ONLY_XRAY_SUBPLOTS:
                if isinstance(crash_debug_df, pd.DataFrame) and not crash_debug_df.empty:
                    crash_xray_fig = _create_crash_xray_figure(
                        crash_debug_df=crash_debug_df,
                        strategy_name=strategy,
                        theme=theme,
                    )
                    crash_xray_section = html.Div(
                        [
                            html.H5("Crash Overlay X-Ray", style={"marginTop": "12px", "color": font_color}),
                            html.Div(
                                "顯示 panic_hits 與 4 類觸發條件（大跌/跌家暴增/跌停潮/爆量）",
                                style={"fontSize": "12px", "color": "#888", "marginBottom": "6px"},
                            ),
                            dcc.Graph(
                                figure=crash_xray_fig,
                                config={"displayModeBar": True, "scrollZoom": True},
                                className="crash-xray-graph",
                            ),
                        ],
                        style={"marginTop": "6px"},
                    )
                else:
                    crash_xray_section = html.Div(
                        "Crash Overlay X-Ray 無資料（尚未套用 crash-only 或判定資料缺失）。",
                        style={"fontSize": "12px", "color": "#888", "marginTop": "10px"},
                    )

            # === 計算風險閥門徽章內容 ===
            valve = results.get(strategy, {}).get('valve', {}) or {}
            valve_badge_text = ("已套用" if valve.get("applied") else "未套用")
            valve_badge_extra = []
            if isinstance(valve.get("cap"), (int, float)):
                valve_badge_extra.append(f"CAP={valve['cap']:.2f}")
            if isinstance(valve.get("atr_ratio"), (int, float)):
                valve_badge_extra.append(f"ATR比值={valve['atr_ratio']:.2f}")
            elif valve.get("atr_ratio") == "forced":
                valve_badge_extra.append("強制觸發")
            if str(valve.get("mode")) == "crash_only":
                if valve.get("preset"):
                    valve_badge_extra.append(f"Crash={valve.get('preset')}")
                if isinstance(valve.get("trigger_rate"), (int, float)):
                    valve_badge_extra.append(f"觸發率={float(valve['trigger_rate']):.1%}")
                if isinstance(valve.get("trigger_days"), (int, float)):
                    valve_badge_extra.append(f"天數={int(valve['trigger_days'])}")

            valve_badge = html.Span(
                "🛡️ 風險閥門：" + valve_badge_text + ((" | " + " | ".join(valve_badge_extra)) if valve_badge_extra else ""),
                style={
                    "marginLeft": "8px",
                    "color": ("#dc3545" if valve.get("applied") else "#6c757d"),
                    "fontWeight": "bold"
                }
            ) if valve else html.Span("")

            unified_fig_height = getattr(unified_fig.layout, "height", None)
            if unified_fig_height is None:
                unified_fig_height = 900
            unified_graph_style = {'height': f'{int(unified_fig_height)}px'}

            cooldown_used = int(result.get("trade_cooldown_bars_used", 0) or 0)
            latest_t_signal = _extract_latest_t_signal(
                strategy_type=strategy_type,
                indicator_daily=indicator_daily,
                daily_state=daily_state_display,
                trade_df=trade_df,
                trade_cooldown_bars=cooldown_used,
            )

            sig_code = str(latest_t_signal.get("signal", "HOLD")).upper()
            sig_date = str(latest_t_signal.get("data_date") or "N/A")
            sig_reason = str(latest_t_signal.get("reason") or "")
            last_sig_code = str(latest_t_signal.get("last_signal", "HOLD")).upper()
            last_sig_date = str(latest_t_signal.get("last_signal_date") or "N/A")
            sig_text_map = {"BUY": "買入", "SELL": "賣出", "HOLD": "觀望"}
            sig_color_map = {"BUY": "#2f9e44", "SELL": "#e03131", "HOLD": "#868e96"}
            sig_label = sig_text_map.get(sig_code, sig_code)
            sig_color = sig_color_map.get(sig_code, "#868e96")
            last_sig_label = sig_text_map.get(last_sig_code, last_sig_code)
            last_sig_color = sig_color_map.get(last_sig_code, "#868e96")
            last_sig_reason = str(latest_t_signal.get("last_signal_reason") or "")

            def _friendly_signal_reason(raw_reason: str, signal_code: str, *, is_last: bool = False) -> str:
                reason = str(raw_reason or "").strip()
                reason_map = {
                    "smaa小於買入閥值": "閥值偏向低檔",
                    "smaa大於賣出閥值": "閥值偏向高檔",
                    "在買賣閥值之間": "閥值在區間",
                    "交易冷卻中": "冷卻期內先不動作",
                    "已有持倉不再買": "已持有部位",
                    "無持倉可賣": "無持倉可賣出",
                    "無轉折訊號": "無明確轉折",
                    "轉折向上訊號": "出現轉強訊號",
                    "轉折向下訊號": "出現轉弱訊號",
                    "已執行的最後訊號": "最近一次實際訊號",
                    "單一資料點無法判斷訊號變化": "資料不足",
                }
                if reason in reason_map:
                    return reason_map[reason]
                if reason:
                    return reason
                if is_last:
                    return "沒有可回溯上次訊號說明"
                if signal_code == "BUY":
                    return "訊號為買入"
                if signal_code == "SELL":
                    return "訊號為賣出"
                return "無明確買賣條件，先觀望"

            sig_reason_text = _friendly_signal_reason(sig_reason, sig_code, is_last=False)
            last_sig_reason_text = _friendly_signal_reason(last_sig_reason, last_sig_code, is_last=True)

            actionable_signal_card = dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div("最新訊號", className="card-title-label", style={"color": card_text}),
                                    html.Div([
                                        html.Span(
                                            sig_label,
                                            style={
                                                "fontWeight": "bold",
                                                "fontSize": "20px",
                                                "color": sig_color,
                                                "lineHeight": "1.2",
                                            }
                                        ),
                                        html.Span(
                                            f"({sig_date})",
                                            style={
                                                "fontSize": "12px",
                                                "color": card_text
                                            }
                                        ),
                                    ]),

                                    html.Div(sig_reason_text, style={"fontSize": "11px", "color": "#888"}),
                                ]
                            )
                        ],
                        style={
                            "background": card_bg,
                            "border": f"1px solid {card_border}",
                            "borderLeft": f"6px solid {sig_color}",
                            "borderRadius": "8px",
                            "margin-bottom": "6px",
                        },
                    )
                ],
                xs=12, sm=6, md=4, lg=2, xl=2,
                style={"minWidth": "100px", "margin-bottom": "6px", "maxWidth": "12.5%"},
            )

            latest_emitted_signal_card = dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div("上次發出訊號", className="card-title-label", style={"color": card_text}),
                                    html.Div([
                                        html.Span(
                                        last_sig_label,
                                        style={"fontWeight": "bold", "fontSize": "20px", "color": last_sig_color, "lineHeight": "1.2"},
                                    ),
                                        html.Span(f"({last_sig_date})", style={"fontSize": "12px", "color": card_text})
                                    ]),
                                    html.Div(last_sig_reason_text, style={"fontSize": "11px", "color": "#888"}),
                                ]
                            )
                        ],
                        style={
                            "background": card_bg,
                            "border": f"1px solid {card_border}",
                            "borderLeft": f"6px solid {last_sig_color}",
                            "borderRadius": "8px",
                            "margin-bottom": "6px",
                        },
                    )
                ],
                xs=12, sm=6, md=4, lg=2, xl=2,
                style={"minWidth": "100px", "margin-bottom": "6px", "maxWidth": "12.5%"},
            )

            summary_cards = [actionable_signal_card, latest_emitted_signal_card] + metric_cards
            trade_table_style_conditional = []
            if isinstance(display_df, pd.DataFrame) and "Crash觸發" in display_df.columns:
                trade_table_style_conditional = [
                    {
                        "if": {"filter_query": "{Crash觸發} = 是"},
                        "backgroundColor": "rgba(255, 107, 107, 0.20)",
                        "fontWeight": "bold",
                    }
                ]

            strategy_content = html.Div([
                html.H4([
                    f"回測策略: {strategy} ",
                    html.Span("ⓘ", title=tooltip, style={"cursor": "help", "color": "#888"}),
                    valve_badge
                ]),
                html.Div(
                    param_line_text,
                    title=param_line_text,
                    style={
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "maxWidth": "100%",
                        "fontSize": "13px",
                    },
                ),
                dbc.Row(summary_cards, style={"flex-wrap": "wrap"}, className='metrics-cards-row'),
                html.Br(),
                # 🎯 統一圖表（三圖同步縮放）
                dcc.Graph(
                    figure=unified_fig,
                    config={'displayModeBar': True, 'scrollZoom': True},
                    className='unified-dashboard-graph',
                    style=unified_graph_style
                ),
                crash_xray_section,

                # 原始的三張分離圖表（已註解）
                # dcc.Graph(figure=fig1, config={'displayModeBar': True}, className='main-metrics-graph'),
                # dcc.Graph(figure=fig2, config={'displayModeBar': True}, className='main-cash-graph'),
                # dcc.Graph(figure=fig_w, config={'displayModeBar': True}, className='main-weight-graph'),
                # 將交易明細標題與説明合併為同一行
                html.Div([
                    html.H5("交易明細", style={"marginBottom": 0, "marginRight": "12px", "color": font_color}),
                    html.Div("實際下單日為信號日的隔天（S+1），修改代碼會影響很多層面，暫不修改",
                             style={"fontWeight": "bold", "fontSize": "16px", "color": font_color})
                ], style={"display": "flex", "alignItems": "center", "marginTop": "16px"}),

                dash_table.DataTable(
                    columns=[{"name": get_column_display_name(i), "id": i} for i in display_df.columns],
                    data=display_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'backgroundColor': bg_color},
                    style_cell={'textAlign': 'center', 'backgroundColor': table_cell_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                    style_header={'backgroundColor': table_header_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
                    style_data_conditional=trade_table_style_conditional,
                    id={'type': 'strategy-table', 'strategy': strategy}
                ),

                # === 新增：交易明細 CSV 下載按鈕 ===
                html.Div([
                    html.Button(
                        "📥 下載交易明細 CSV",
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

        # 創建策略回測的子頁籤容器
        return html.Div([
            dcc.Tabs(
                id='strategy-tabs',
                value=f"strategy_{strategy_names[0]}" if strategy_names else "no_strategy",
                children=strategy_tabs,
                className='strategy-tabs-bar'
            )
        ])

    elif tab == "compare":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['close'], name='Close Price', line=dict(color='dodgerblue')))
        colors = ['green', 'limegreen', 'red', 'orange', 'purple', 'blue', 'pink', 'cyan']

        # 定義策略到顏色的映射
        strategy_colors = {strategy: colors[i % len(colors)] for i, strategy in enumerate(strategy_names)}

        # 為圖表添加買賣點
        for i, strategy in enumerate(strategy_names):
            result = results.get(strategy)
            if not result:
                continue
            # 使用解包器函數，支援 pack_df 和傳統 JSON 字串兩種格式
            trade_df = df_from_pack(result.get('trade_df'))

            # 標準化交易資料
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
            except Exception:
                # 後備標準化方案
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
            if trade_df.empty:
                continue
            buys = trade_df[trade_df['type'] == 'buy']
            sells = trade_df[trade_df['type'] == 'sell']
            sells_forced = trade_df[trade_df['type'] == 'sell_forced']

            fig.add_trace(go.Scatter(x=buys['trade_date'], y=buys['price'], mode='markers', name=f'{strategy} Buy',
                                     marker=dict(symbol='cross', size=8, color=colors[i % len(colors)])))
            fig.add_trace(go.Scatter(x=sells['trade_date'], y=sells['price'], mode='markers', name=f'{strategy} Sell',
                                     marker=dict(symbol='x', size=8, color=colors[i % len(colors)])))
            if not sells_forced.empty:
                fig.add_trace(go.Scatter(x=sells_forced['trade_date'], y=sells_forced['price'], mode='markers',
                                         name=f'{strategy} Forced',
                                         marker=dict(symbol='square', size=8, color='gray')))

        # 更新圖表佈局
        fig.update_layout(
            title=f'{ticker} 所有策略買賣點比較',
            xaxis_title='Date', yaxis_title='股價', template=plotly_template,
            font_color=font_color, plot_bgcolor=bg_color, paper_bgcolor=bg_color, legend_font_color=legend_font_color,
            legend=dict(
                x=1.05, y=1, xanchor='left', yanchor='top',
                bordercolor=legend_bordercolor, borderwidth=1, bgcolor=legend_bgcolor,
                itemsizing='constant', orientation='v', font=dict(color=legend_font_color)
            )
        )

        # 準備比較表格數據
        comparison_data = []
        for strategy in strategy_names:
            result = results.get(strategy)
            if not result:
                continue

            # 讀取交易數據
            trade_df = df_from_pack(result.get('trade_df'))

            # 標準化交易資料
            try:
                from sss_core.normalize import normalize_trades_for_ui as norm
                trade_df = norm(trade_df)
            except Exception:
                # 後備標準化方案
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

            # 計算詳細統計信息
            detailed_stats = calculate_strategy_detailed_stats(trade_df, df_raw)

            metrics = result['metrics']
            ds_for_nav = df_from_pack(
                result.get("daily_state_valve")
                or result.get("daily_state_std")
                or result.get("daily_state")
                or result.get("daily_state_base")
            )
            nav_rel_ratio = _calc_nav_relative_ratio(ds_for_nav, df_raw)
            metrics["nav_relative_ratio"] = float(nav_rel_ratio) if pd.notna(nav_rel_ratio) else np.nan

            # 安全取值函數：處理 None -> 0 的轉換，避免格式化錯誤
            def safe_get(key, default=0.0):
                v = metrics.get(key)
                return v if v is not None else default

            nav_rel_val = safe_get("nav_relative_ratio", np.nan)
            nav_rel_text = f"{nav_rel_val:.2f}x" if pd.notna(nav_rel_val) else "N/A"

            comparison_data.append({
                '策略': strategy,
                '總回報率': f"{safe_get('total_return'):.2%}",
                '年化回報率': f"{safe_get('annual_return'):.2%}",
                '最大回撤': f"{safe_get('max_drawdown'):.2%}",
                '卡瑪比率': f"{safe_get('calmar_ratio'):.2f}",
                'NAV相對淨值比': nav_rel_text,
                '交易次數': int(safe_get('num_trades', 0)),
                '勝率': f"{safe_get('win_rate'):.2%}",
                '盈虧比': f"{safe_get('payoff_ratio'):.2f}",
                '平均持有天數': f"{detailed_stats['avg_holding_days']:.1f}",
                '賣後買平均天數': f"{detailed_stats['avg_sell_to_buy_days']:.1f}",
                '目前狀態': detailed_stats['current_status'],
                '距離上次操作天數': f"{detailed_stats['days_since_last_action']}"
            })

        # 定義顏色調整函數
        def adjust_color_for_theme(color, theme):
            # 預定義顏色到 RGB 的映射
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

            rgb = color_to_rgb.get(color, '128, 128, 128')  # 默認灰色

            if theme == 'theme-dark':
                return f'rgba({rgb}, 0.2)'  # 透明度 0.2
            elif theme == 'theme-light':
                return f'rgba({rgb}, 1)'    # 透明度 1
            else:  # theme-blue
                return f'rgba({rgb}, 0.5)'  # 透明度 0.5

        # 創建比較表格並應用條件樣式
        compare_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in comparison_data[0].keys()] if comparison_data else [],
            data=comparison_data,
            style_table={'overflowX': 'auto', 'backgroundColor': bg_color},
            style_cell={'textAlign': 'center', 'backgroundColor': table_cell_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
            style_header={'backgroundColor': table_header_bg, 'color': table_text, 'border': f'1px solid {table_border}'},
            style_data_conditional=[
                {
                    'if': {'row_index': i},
                    'backgroundColor': adjust_color_for_theme(strategy_colors[row['策略']], theme),
                    'border': f'1px solid {strategy_colors[row['策略']]}'
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
        # === 增強分析頁面 ===
        enhanced_controls = html.Div([
            html.H4("🔍 增強分析"),

            # === 新增：從回測結果載入區塊 ===
            html.Details([
                html.Summary("🧠 從回測結果載入"),
                html.Div([
                    html.Div("選擇策略（自動評分：ledger_std > ledger > trade_df）",
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    dcc.Dropdown(
                        id="enhanced-strategy-selector",
                        placeholder="請先執行回測...",
                        style={"width":"100%","marginBottom":"8px"}
                    ),
                    html.Button("載入選定策略", id="load-enhanced-strategy", n_clicks=0,
                               style={"width":"100%","marginBottom":"8px"}),
                    html.Div(id="enhanced-load-status", style={"fontSize":"12px","color":"#888"}),
                    html.Div("💡 回測完成後會自動快取最佳策略",
                             style={"fontSize":"11px","color":"#666","fontStyle":"italic","marginTop":"4px"})
                ])
            ], style={"marginBottom":"16px"}),

            # === 隱藏的 cache store ===
            dcc.Store(id="enhanced-trades-cache"),

            html.Details([
                html.Summary("風險閥門回測"),
                html.Div([
                    dcc.Dropdown(
                        id="rv-mode", options=[
                            {"label":"降低上限 (cap)","value":"cap"},
                            {"label":"禁止加碼 (ban_add)","value":"ban_add"},
                        ], value="cap", clearable=False, style={"width":"240px"}
                    ),
                    dcc.Slider(id="rv-cap", min=0.1, max=1.0, step=0.05, value=0.5,
                               tooltip={"placement":"bottom","always_visible":True}),
                    html.Div("ATR(20)/ATR(60) 比值門檻", style={"marginTop":"8px"}),
                    dcc.Slider(id="rv-atr-mult", min=1.0, max=2.0, step=0.05, value=1.3,
                               tooltip={"placement":"bottom","always_visible":True}),
                    html.Button("執行風險閥門回測", id="run-rv", n_clicks=0, style={"marginTop":"8px"})
                ])
            ]),

            html.Div(id="rv-summary", style={"marginTop":"12px"}),
            dcc.Graph(id="rv-equity-chart"),
            dcc.Graph(id="rv-dd-chart"),

            # === 新增：數據比對功能 ===
            html.Details([
                html.Summary("🔍 數據比對與診斷"),
                html.Div([
                    html.Div("直接輸出實際數據進行比對，診斷增強分析資料與參數一致性問題",
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Button("輸出數據比對報告", id="export-data-comparison", n_clicks=0,
                               style={"width":"100%","marginBottom":"8px","backgroundColor":"#17a2b8","color":"white"}),
                    html.Div(id="data-comparison-output", style={"fontSize":"12px","color":"#666","marginTop":"8px"}),
                    dcc.Download(id="data-comparison-csv")
                ])
            ], style={"border":"1px solid #17a2b8","borderRadius":"8px","padding":"12px","marginTop":"12px"}),

            # === 新增：風險-報酬地圖（Pareto Map）區塊 ===
            html.Details([
                html.Summary("📊 風險-報酬地圖（Pareto Map）"),
                html.Div([
                    html.Div("生成策略的風險-報酬分析圖表",
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Button("生成 Pareto Map", id="generate-pareto-map", n_clicks=0,
                               style={"width":"100%","marginBottom":"8px"}),
                    html.Div(id="pareto-map-status", style={"fontSize":"12px","color":"#888","marginBottom":"8px"}),
                    dcc.Graph(id="pareto-map-graph", style={"height":"600px"}),
                    html.Div([
                        html.Button("📥 下載 Pareto Map 數據 (CSV)", id="download-pareto-csv", n_clicks=0,
                                   style={"width":"100%","marginBottom":"8px"}),
                        dcc.Download(id="pareto-csv-download"),
                        html.H6("圖表說明：", style={"marginTop":"16px","marginBottom":"8px"}),
                        html.Ul([
                            html.Li("橫軸：最大回撤（愈左愈好）"),
                            html.Li("縱軸：PF 獲利因子（愈上愈好）"),
                            html.Li("顏色：右尾調整幅度（紅色=削減右尾，藍色=放大右尾，0為中線）"),
                            html.Li("點大小：風險觸發天數（越大＝管得越勤）"),
                            html.Li("理想區域：綠色虛線框內（又上又左、顏色接近中線、點不要大到誇張）")
                        ], style={"fontSize":"12px","color":"#666"})
                    ])
                ])
            ], style={"border":"1px solid #333","borderRadius":"8px","padding":"12px","marginTop":"12px"}),
            # === 交易貢獻拆解區塊 ===
            html.Details([
                html.Summary("🔍 交易貢獻拆解"),
                html.Div([
                    html.Div("拆解交易貢獻，分析不同加碼/減碼階段的績效表現",
                             style={"marginBottom":"8px","fontSize":"14px","color":"#666"}),
                    html.Div([
                        html.Div([
                            html.Label("最小間距 (天)", style={"fontSize":"12px","color":"#888"}),
                            dcc.Input(id="phase-min-gap", type="number", value=5, min=0, max=30, step=1,
                                     style={"width":"80px","marginRight":"16px"})
                        ], style={"display":"inline-block","marginRight":"16px"}),
                                            html.Div([
                        html.Label("冷卻期 (天)", style={"fontSize":"12px","color":"#888"}),
                        dcc.Input(id="phase-cooldown", type="number", value=10, min=0, max=30, step=1,
                                 style={"width":"80px"})
                    ], style={"display":"inline-block"})
                ], style={"marginBottom":"8px"}),
                html.Div([
                    html.Button("執行交易貢獻拆解", id="run-phase", n_clicks=0,
                               style={"width":"48%","marginBottom":"8px","marginRight":"2%"}),
                    html.Button("批量測試參數範圍", id="run-batch-phase", n_clicks=0,
                               style={"width":"48%","marginBottom":"8px","marginLeft":"2%","backgroundColor":"#28a745","color":"white"})
                ], style={"display":"flex","justifyContent":"space-between"}),
                    html.Div([
                        html.H6("參數說明：", style={"marginTop":"16px","marginBottom":"8px"}),
                        html.Ul([
                            html.Li("最小間距：兩次加碼至少要間隔幾天，才算獨立訊號（過濾短期噪音）"),
                            html.Li("冷卻期：每次加碼後，必須過多久才允許下一筆加碼（避免過度曝險）"),
                            html.Li("用途：讓拆解聚焦在比較有意義的加碼波段，避免被短期小單稀釋")
                        ], style={"fontSize":"12px","color":"#666","marginBottom":"16px"}),
                        html.Div(id="phase-table"),
                        html.Div([
                            html.H6("批量測試結果", style={"marginTop":"16px","marginBottom":"8px","color":"#28a745"}),
                            html.Div(id="batch-phase-results", style={"fontSize":"12px"})
                        ])
                    ])
                ])
            ], style={"border":"1px solid #333","borderRadius":"8px","padding":"12px","marginTop":"12px"})
        ])

        return enhanced_controls
# --------- 版本沿革模態框控制和主題切換 ---------
@app.callback(
    Output("history-modal", "is_open"),
    Output('main-bg', 'className'),
    Output('theme-toggle', 'children'),
    Output('theme-store', 'data'),
    # [新增] 控制上方參數列的樣式
    Output('ctrl-row-basic', 'style'),
    Output('ctrl-row-risk', 'style'),
    Output('ctrl-row-ensemble', 'style'),
    # [新增] 控制特殊標題的顏色
    Output('label-risk-title', 'style'),
    Output('label-maj-title', 'style'),
    Output('label-prop-title', 'style'),
    [Input("history-btn", "n_clicks"), Input("history-close", "n_clicks"), Input('theme-toggle', 'n_clicks')],
    [State("history-modal", "is_open"), State('theme-store', 'data')],
    # 移除 prevent_initial_call=True，這樣初始化時才會正確渲染顏色
)
def toggle_history_modal_and_theme(history_btn, history_close, theme_btn, is_open, current_theme):
    ctx_trigger = ctx.triggered_id

    # 決定下一個主題
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

    # === 定義樣式設定 (Style Presets) ===
    # 預設邊框圓角
    base_style = {'borderRadius': '4px', 'transition': 'background-color 0.3s'}

    if next_theme == 'theme-light':
        # 淺色模式 (原版配色)
        style_basic = {**base_style, 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6', 'color': '#212529'}
        style_risk  = {**base_style, 'backgroundColor': '#fff3cd', 'border': '1px solid #ffeeba', 'color': '#856404'}
        style_ens   = {**base_style, 'backgroundColor': '#d1ecf1', 'border': '1px solid #bee5eb', 'color': '#0c5460'}

        # 標題顏色
        c_risk = {'color': '#856404'} # 深黃
        c_maj  = {'color': '#0066cc'} # 深藍
        c_prop = {'color': '#28a745'} # 深綠

        btn_label = '🌕 淺色主題'

    elif next_theme == 'theme-blue':
        # 藍色模式
        style_basic = {**base_style, 'backgroundColor': '#0b1e3a', 'border': '1px solid #446688', 'color': '#ffe066'}
        style_risk  = {**base_style, 'backgroundColor': '#3a2e05', 'border': '1px solid #886600', 'color': '#ffcc00'}
        style_ens   = {**base_style, 'backgroundColor': '#0f2b33', 'border': '1px solid #005566', 'color': '#00ccff'}

        c_risk = {'color': '#ffcc00'}
        c_maj  = {'color': '#3399ff'}
        c_prop = {'color': '#66ff66'}

        btn_label = '💙 藍黃主題'

    else: # theme-dark (預設)
        # 深色模式 (調整為深灰/深褐/深青，文字反白)
        style_basic = {**base_style, 'backgroundColor': '#2b2b2b', 'border': '1px solid #444', 'color': '#e0e0e0'}

        # Risk: 深褐色背景 + 金黃色邊框
        style_risk  = {**base_style, 'backgroundColor': '#2c2505', 'border': '1px solid #665200', 'color': '#e0e0e0'}

        # Ensemble: 深青色背景 + 青色邊框
        style_ens   = {**base_style, 'backgroundColor': '#0c282e', 'border': '1px solid #0f4c5c', 'color': '#e0e0e0'}

        # 標題顏色：在深色背景上需要亮一點
        c_risk = {'color': '#ffc107'} # 亮黃 Warning
        c_maj  = {'color': '#66b2ff'} # 亮藍
        c_prop = {'color': '#75df8a'} # 亮綠

        btn_label = '🌑 深色主題'

    return (
        is_open,
        next_theme,
        btn_label,
        next_theme,
        # 回傳樣式
        style_basic, style_risk, style_ens,
        c_risk, c_maj, c_prop
    )

# --------- 下載交易紀錄 ---------
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

    # 從觸發的按鈕ID中提取策略名稱
    strategy = ctx_trigger['strategy']

    # 從backtest_data中獲取對應策略的交易數據
    # backtest_data 現在已經是 dict，不需要 json.loads
    results = backtest_data['results']
    result = results.get(strategy)

    if not result:
        return [None] * len(n_clicks)

    # 使用解包器函數，支援 pack_df 和傳統 JSON 字串兩種格式
    trade_df = df_from_pack(result.get('trade_df'))

    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_df = norm(trade_df)
    except Exception:
        # 後備標準化方案
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

    # 創建下載數據
    def to_xlsx(bytes_io):
        with pd.ExcelWriter(bytes_io, engine='openpyxl') as writer:
            trade_df.to_excel(writer, sheet_name='交易紀錄', index=False)

    return [dcc.send_bytes(to_xlsx, f"{strategy}_交易紀錄.xlsx") if i and i > 0 else None for i in n_clicks]

# --------- 下載交易明細 CSV ---------
@app.callback(
    Output({'type': 'download-trade-details-data', 'strategy': ALL}, 'data'),
    Input({'type': 'download-trade-details-csv', 'strategy': ALL}, 'n_clicks'),
    State({'type': 'strategy-table', 'strategy': ALL}, 'data'),
    State('backtest-store', 'data'),
    prevent_initial_call=True
)
def download_trade_details_csv(n_clicks, table_data, backtest_data):
    """下載交易明細為 CSV 格式"""
    ctx_trigger = ctx.triggered_id
    if not ctx_trigger or not backtest_data:
        return [None] * len(n_clicks)

    # 從觸發的按鈕ID中提取策略名稱
    strategy = ctx_trigger['strategy']

    # 從backtest_data中獲取對應策略的交易數據
    results = backtest_data['results']
    result = results.get(strategy)

    if not result:
        return [None] * len(n_clicks)

    # 使用解包器函數，支援 pack_df 和傳統 JSON 字串兩種格式
    trade_df = df_from_pack(result.get('trade_df'))

    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_df = norm(trade_df)
    except Exception:
        # 後備標準化方案
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

    # 創建 CSV 下載數據
    def to_csv(bytes_io):
        # 使用 UTF-8 BOM 確保 Excel 能正確顯示中文
        bytes_io.write('\ufeff'.encode('utf-8'))
        trade_df.to_csv(bytes_io, index=False, encoding='utf-8-sig')

    return [dcc.send_bytes(to_csv, f"{strategy}_交易明細.csv") if i and i > 0 else None for i in n_clicks]

def calculate_strategy_detailed_stats(trade_df, df_raw):
    """計算交易統計與目前持有狀態。"""
    if trade_df is None or trade_df.empty:
        return {
            "avg_holding_days": 0,
            "avg_sell_to_buy_days": 0,
            "current_status": "未持有",
            "days_since_last_action": 0,
        }

    def _normalize_type_value(v: object) -> str:
        s = str(v).strip().lower()
        type_map = {
            "buy": "buy",
            "add": "buy",
            "long": "buy",
            "entry": "buy",
            "買入": "buy",
            "加碼": "buy",
            "sell": "sell",
            "exit": "sell",
            "賣出": "sell",
            "sell_forced": "sell_forced",
            "forced_sell": "sell_forced",
            "force_liquidate": "sell_forced",
            "強制賣出": "sell_forced",
            "強制平倉": "sell_forced",
        }
        return type_map.get(s, s)

    def _normalize_reason_value(v: object) -> str:
        s = str(v).strip().lower()
        reason_map = {
            "end_of_period": "end_of_period",
            "end_of_backtest": "end_of_period",
            "backtest_end": "end_of_period",
            "期末平倉": "end_of_period",
        }
        return reason_map.get(s, s)

    df = trade_df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    if "type" in df.columns:
        df["type_norm"] = df["type"].apply(_normalize_type_value)
    else:
        df["type_norm"] = ""
    if "reason" in df.columns:
        df["reason_norm"] = df["reason"].apply(_normalize_reason_value)
    else:
        df["reason_norm"] = ""

    # 統計時忽略「期末強制平倉」，避免汙染持有狀態。
    effective_df = df.copy()
    while not effective_df.empty:
        last_row = effective_df.iloc[-1]
        if (
            str(last_row.get("type_norm", "")) == "sell_forced"
            and str(last_row.get("reason_norm", "")) == "end_of_period"
        ):
            effective_df = effective_df.iloc[:-1].reset_index(drop=True)
            continue
        break
    if effective_df.empty:
        effective_df = df.copy()

    holding_periods = []
    for i in range(len(effective_df) - 1):
        current_type = str(effective_df.iloc[i].get("type_norm", ""))
        next_type = str(effective_df.iloc[i + 1].get("type_norm", ""))
        if current_type == "buy" and next_type in ["sell", "sell_forced"]:
            buy_date = effective_df.iloc[i].get("trade_date")
            sell_date = effective_df.iloc[i + 1].get("trade_date")
            if pd.notna(buy_date) and pd.notna(sell_date):
                holding_periods.append((sell_date - buy_date).days)
    avg_holding_days = sum(holding_periods) / len(holding_periods) if holding_periods else 0

    sell_to_buy_periods = []
    for i in range(len(effective_df) - 1):
        current_type = str(effective_df.iloc[i].get("type_norm", ""))
        next_type = str(effective_df.iloc[i + 1].get("type_norm", ""))
        if current_type in ["sell", "sell_forced"] and next_type == "buy":
            sell_date = effective_df.iloc[i].get("trade_date")
            buy_date = effective_df.iloc[i + 1].get("trade_date")
            if pd.notna(sell_date) and pd.notna(buy_date):
                sell_to_buy_periods.append((buy_date - sell_date).days)
    avg_sell_to_buy_days = sum(sell_to_buy_periods) / len(sell_to_buy_periods) if sell_to_buy_periods else 0

    current_date = pd.Timestamp(datetime.now())
    if df_raw is not None and not df_raw.empty:
        current_date = pd.to_datetime(df_raw.index[-1], errors="coerce")

    if effective_df.empty:
        current_status = "未持有"
        days_since_last_action = 0
    else:
        last_trade = effective_df.iloc[-1]
        last_date = pd.to_datetime(last_trade.get("trade_date"), errors="coerce")
        w_after = pd.to_numeric(last_trade.get("w_after", pd.NA), errors="coerce")

        if pd.notna(w_after):
            current_status = "持有" if float(w_after) > 1e-6 else "未持有"
        else:
            last_type = str(last_trade.get("type_norm", ""))
            current_status = "持有" if last_type == "buy" else "未持有"

        if pd.notna(last_date) and pd.notna(current_date):
            days_since_last_action = int((current_date - last_date).days)
        else:
            days_since_last_action = 0

    return {
        "avg_holding_days": round(avg_holding_days, 1),
        "avg_sell_to_buy_days": round(avg_sell_to_buy_days, 1),
        "current_status": current_status,
        "days_since_last_action": days_since_last_action,
    }

def is_price_data_up_to_date(csv_path):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
        else:
            last_date = pd.to_datetime(df.iloc[-1, 0])

        # 獲取台北時間的今天
        today = pd.Timestamp.now(tz='Asia/Taipei').normalize()

        # 檢查是否為工作日（週一到週五）
        if today.weekday() >= 5:  # 週六(5)或週日(6)
            # 如果是週末，檢查最後數據是否為上個工作日
            last_weekday = today - pd.Timedelta(days=today.weekday() - 4)  # 上個週五
            return last_date >= last_weekday
        else:
            # 如果是工作日，檢查是否為今天或昨天（考慮數據延遲）
            yesterday = today - pd.Timedelta(days=1)
            return last_date >= yesterday
    except Exception:
        return False

def fetch_yf_data(ticker: str, filename: Path, start_date: str = "2000-01-01", end_date: str | None = None):
    now_taipei = pd.Timestamp.now(tz='Asia/Taipei')
    try:
        end_date_str = end_date if end_date is not None else now_taipei.strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date_str, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("下載的數據為空")
        df.to_csv(filename)
        print(f"成功下載 '{ticker}' 數據到 '{filename}'.")
    except Exception as e:
        print(f"警告: '{ticker}' 下載失敗: {e}")

def ensure_all_price_data_up_to_date(ticker_list, data_dir):
    """智能檢查並更新股價數據，只在必要時下載"""
    for ticker in ticker_list:
        filename = Path(data_dir) / f"{ticker.replace(':','_')}_data_raw.csv"
        if not is_price_data_up_to_date(filename):
            print(f"{ticker} 股價資料需要更新，開始下載...")
            fetch_yf_data(ticker, filename)
        else:
            print(f"{ticker} 股價資料已是最新，跳過下載。")

# 簡化的股價數據下載剎車機制
def should_download_price_data():
    """檢查是否需要下載股價數據的剎車機制"""
    try:
        # 檢查是否為交易時間（避免在交易時間頻繁下載）
        now = pd.Timestamp.now(tz='Asia/Taipei')
        if now.weekday() < 5:  # 工作日
            hour = now.hour
            if 9 <= hour <= 13:  # 交易時間
                print("當前為交易時間，跳過股價數據下載以避免幹擾")
                return False

        # 檢查數據文件是否存在且較新（避免重複下載）
        data_files_exist = all(
            os.path.exists(Path(DATA_DIR) / f"{ticker.replace(':','_')}_data_raw.csv")
            for ticker in TICKER_LIST
        )

        if data_files_exist:
            print("股價數據文件已存在，跳過初始下載")
            return False

        return True
    except Exception as e:
        print(f"剎車機制檢查失敗: {e}，允許下載")
        return True

# 在 app 啟動時呼叫（添加剎車機制）
TICKER_LIST = ['2330.TW', '2412.TW', '2414.TW', '^TWII']  # 依實際需求調整
DATA_DIR = 'data'  # 依實際路徑調整

# 安全的啟動機制
def safe_startup():
    """安全的啟動函數，避免線程衝突"""
    try:
        # 只有在剎車機制允許時才下載
        if should_download_price_data():
            ensure_all_price_data_up_to_date(TICKER_LIST, DATA_DIR)
        else:
            print("股價數據下載已由剎車機制阻止")
    except Exception as e:
        print(f"啟動時數據下載失敗: {e}，繼續啟動應用")

# --------- 增強分析 Callback：風險閥門回測（整合版） ---------
@app.callback(
    Output("rv-summary","children"),
    Output("rv-equity-chart","figure"),
    Output("rv-dd-chart","figure"),
    Input("run-rv","n_clicks"),
    State("rv-mode","value"),
    State("rv-cap","value"),
    State("rv-atr-mult","value"),
    State("enhanced-trades-cache","data"),
    State("backtest-store","data"),
    prevent_initial_call=True
)
def _run_rv(n_clicks, mode, cap_level, atr_mult, cache, backtest_data=None):
    if not n_clicks or not cache:
        return "請先載入策略資料", no_update, no_update

    # 全局風險閥門已移除：增強分析只使用頁面參數。
    effective_cap = cap_level
    effective_atr_ratio = atr_mult
    logger.info(f"增強分析使用頁面參數：CAP={effective_cap}, ATR比值門檻={effective_atr_ratio}")

    logger.info(f"=== 數據驗證 ===")

    # 以 enhanced cache 為主，必要時回退到 backtest-store。
    df_raw = df_from_pack(cache.get("df_raw"))
    daily_state = df_from_pack(cache.get("daily_state"))
    if (df_raw is None or df_raw.empty or daily_state is None or daily_state.empty) and backtest_data:
        results = backtest_data.get("results", {})
        strategy_name = cache.get("strategy") if cache else None
        if strategy_name and strategy_name in results:
            result = results[strategy_name]
            df_raw = df_from_pack(backtest_data.get("df_raw"))
            daily_state = df_from_pack(result.get("daily_state_std") or result.get("daily_state"))
            logger.info(f"回退使用 backtest-store 資料源: {strategy_name}")

    # === 原有數據驗證日誌 ===
    logger.info(f"df_raw 形狀: {df_raw.shape if df_raw is not None else 'None'}")
    logger.info(f"daily_state 形狀: {daily_state.shape if daily_state is not None else 'None'}")
    if daily_state is not None:
        logger.info(f"daily_state 欄位: {list(daily_state.columns)}")
        logger.info(f"daily_state 索引範圍: {daily_state.index.min()} 到 {daily_state.index.max()}")
        if "w" in daily_state.columns:
            logger.info(f"權重欄位統計: 最小值={daily_state['w'].min():.4f}, 最大值={daily_state['w'].max():.4f}, 平均值={daily_state['w'].mean():.4f}")

    if df_raw is None or df_raw.empty:
        return "找不到股價資料", no_update, no_update

    if daily_state is None or daily_state.empty:
        return "找不到 daily_state（每日資產/權重）", no_update, no_update

    # 欄名對齊
    c_open = "open" if "open" in df_raw.columns else _first_col(df_raw, ["Open","開盤價"])
    c_close = "close" if "close" in df_raw.columns else _first_col(df_raw, ["Close","收盤價"])
    c_high  = "high" if "high" in df_raw.columns else _first_col(df_raw, ["High","最高價"])
    c_low   = "low"  if "low"  in df_raw.columns else _first_col(df_raw, ["Low","最低價"])

    if c_open is None or c_close is None:
        return "股價資料缺少 open/close 欄位", no_update, no_update

    open_px = pd.to_numeric(df_raw[c_open], errors="coerce").dropna()
    open_px.index = pd.to_datetime(df_raw.index)

    # 權重取自 daily_state
    if "w" not in daily_state.columns:
        return "daily_state 缺少權重欄位 'w'", no_update, no_update

    w = daily_state["w"].astype(float).reindex(open_px.index).ffill().fillna(0.0)

    # 成本參數（使用 SSS_EnsembleTab 預設）
    cost = None

    # 基準：用 df_raw 當基準（即可），函式能在無高低價時回退
    bench = pd.DataFrame({
        "收盤價": pd.to_numeric(df_raw[c_close], errors="coerce"),
    }, index=pd.to_datetime(df_raw.index))
    if c_high and c_low:
        bench["最高價"] = pd.to_numeric(df_raw[c_high], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw[c_low], errors="coerce")

    # 需要用到 SSS_EnsembleTab 內新加的函式
    try:
        from SSS_EnsembleTab import risk_valve_backtest
        # === 增強分析風險閥門：確保參數一致性 (2025/08/20) ===
        enhanced_valve_params = {
            "open_px": open_px,
            "w": w,
            "cost": cost,
            "benchmark_df": bench,
            "mode": mode,
            "cap_level": float(effective_cap),  # === 修正：使用有效參數 ===
            "slope20_thresh": 0.0,
            "slope60_thresh": 0.0,
            "atr_win": 20,
            "atr_ref_win": 60,
            "atr_ratio_mult": float(effective_atr_ratio),  # === 修正：使用有效參數 ===
            "use_slopes": True,
            "slope_method": "polyfit",
            "atr_cmp": "gt"
        }

        # 記錄增強分析風險閥門配置
        logger.info(f"[Enhanced] 風險閥門配置: cap_level={enhanced_valve_params['cap_level']}, atr_ratio_mult={enhanced_valve_params['atr_ratio_mult']}")

        out = risk_valve_backtest(**enhanced_valve_params)
    except Exception as e:
        return f"風險閥門回測執行失敗: {e}", no_update, no_update

    m = out["metrics"]

    # 計算風險觸發天數
    sig = out["signals"]["risk_trigger"]
    trigger_days = int(sig.fillna(False).sum())

    # === 修正：顯示實際使用的參數 ===
    summary = html.Div([
        html.Code(f"PF: 原始 {m['pf_orig']:.2f} → 閥門 {m['pf_valve']:.2f}"), html.Br(),
        html.Code(f"MDD: 原始 {m['mdd_orig']:.2%} → 閥門 {m['mdd_valve']:.2%}"), html.Br(),
        html.Code(f"右尾總和(>P90 正報酬): 原始 {m['right_tail_sum_orig']:.2f} → 閥門 {m['right_tail_sum_valve']:.2f} (↓{m['right_tail_reduction']:.2f})"), html.Br(),
        html.Code(f"風險觸發天數：{trigger_days} 天"), html.Br(),
        html.Code(f"使用參數：CAP={effective_cap}, ATR比值門檻={effective_atr_ratio}"), html.Br(),
        html.Code("參數來源：頁面設定", style={"color": "#ffc107"})
    ])

    # 繪圖：兩版權益與回撤
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
        x=eq1.index, y=eq1, name="原始",
        mode="lines", line=dict(color=palette["orig"]["color"], width=2, dash=palette["orig"]["dash"]),
        legendgroup="equity"
    ))
    fig_eq.add_trace(go.Scatter(
        x=eq2.index, y=eq2, name="閥門",
        mode="lines", line=dict(color=palette["valve"]["color"], width=2, dash=palette["valve"]["dash"]),
        legendgroup="equity"
    ))
    fig_eq.update_layout(title="權益曲線（Open→Open）", legend_orientation="h")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd1.index, y=dd1, name="原始",
        mode="lines", line=dict(color=palette["orig"]["color"], width=2, dash=palette["orig"]["dash"]),
        legendgroup="dd"
    ))
    fig_dd.add_trace(go.Scatter(
        x=dd2.index, y=dd2, name="閥門",
        mode="lines", line=dict(color=palette["valve"]["color"], width=2, dash=palette["valve"]["dash"]),
        legendgroup="dd"
    ))
    fig_dd.update_layout(title="回撤曲線", legend_orientation="h", yaxis_tickformat=".0%")

    return summary, fig_eq, fig_dd

# --------- 增強分析 Callback：數據比對報告 ---------
@app.callback(
    Output("data-comparison-output", "children"),
    Output("data-comparison-csv", "data"),
    Input("export-data-comparison", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    State("backtest-store", "data"),
    State("rv-cap", "value"),
    State("rv-atr-mult", "value"),
    prevent_initial_call=True
)
def generate_data_comparison_report(n_clicks, cache, backtest_data, page_cap, page_atr):
    """生成數據比對報告，檢查增強分析資料完整性與當前參數。"""
    if not n_clicks:
        return "請點擊按鈕生成報告", no_update

    logger.info(f"=== 生成增強數據比對報告 ===")

    # 收集參數資訊（僅頁面參數）
    param_info = {
        "頁面風險閥門CAP": page_cap,
        "頁面ATR比值門檻": page_atr,
        "最終使用CAP": page_cap,
        "最終使用ATR比值門檻": page_atr,
    }

    # 收集數據資訊
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
            "daily_state_index_range": f"{daily_state.index.min()} 到 {daily_state.index.max()}" if daily_state is not None and not daily_state.empty else None
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

        # 選擇第一個策略進行詳細分析
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

    # 生成報告
    report_lines = []
    report_lines.append("=== 數據比對報告 ===")
    report_lines.append("")

    # 參數部分
    report_lines.append("📊 參數設定:")
    for key, value in param_info.items():
        report_lines.append(f"  {key}: {value}")
    report_lines.append("")

    # 數據部分
    report_lines.append("📈 數據狀態:")
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

    # 增強診斷建議
    report_lines.append("🔍 詳細診斷建議:")

    report_lines.append("  ℹ️  全局風險閥門已移除，增強分析僅使用頁面參數")

    # 數據完整性檢查
    enhanced_has_data = "enhanced_cache" in data_info and data_info["enhanced_cache"]["daily_state_shape"]
    backtest_has_data = "backtest_store" in data_info and data_info["backtest_store"]["results_count"] > 0

    if enhanced_has_data:
        report_lines.append("  ✅ Enhanced Cache 有數據")
        if "weight_stats" in data_info["enhanced_cache"]:
            ws = data_info["enhanced_cache"]["weight_stats"]
            report_lines.append(f"      權重範圍: {ws['min']:.4f} ~ {ws['max']:.4f}, 均值: {ws['mean']:.4f}")
    else:
        report_lines.append("  ❌ Enhanced Cache 無數據")
        report_lines.append("      → 可能需要重新執行增強分析")

    if backtest_has_data:
        report_lines.append("  ✅ Backtest Store 有結果")
    else:
        report_lines.append("  ❌ Backtest Store 無結果")
        report_lines.append("      → 可能需要重新執行回測分析")

    # 風險閥門邏輯檢查
    effective_cap = page_cap
    effective_atr = page_atr

    report_lines.append("  🔧 風險閥門配置:")
    report_lines.append(f"      有效CAP值: {effective_cap}")
    report_lines.append(f"      有效ATR門檻: {effective_atr}")

    if effective_cap and effective_cap < 0.1:
        report_lines.append("      ⚠️  CAP值過低，可能造成過度保守")
    if effective_atr and effective_atr > 3.0:
        report_lines.append("      ⚠️  ATR門檻過高，可能很少觸發")

    # 一致性檢查總結
    consistency_issues = []
    if not enhanced_has_data:
        consistency_issues.append("Enhanced Cache缺失")
    if not backtest_has_data:
        consistency_issues.append("Backtest Store缺失")

    if consistency_issues:
        report_lines.append(f"  🚨 發現一致性問題: {', '.join(consistency_issues)}")
        report_lines.append("      建議優先解決這些問題以確保分析結果一致性")
    else:
        report_lines.append("  ✅ 未發現明顯一致性問題")

    # 生成 CSV 數據
    csv_data = []
    for key, value in param_info.items():
        csv_data.append({"項目": key, "數值": str(value)})

    csv_data.append({"項目": "", "數值": ""})
    csv_data.append({"項目": "=== 數據狀態 ===", "數值": ""})

    if "enhanced_cache" in data_info:
        for key, value in data_info["enhanced_cache"].items():
            csv_data.append({"項目": f"Enhanced_{key}", "數值": str(value)})

    if "backtest_store" in data_info:
        for key, value in data_info["backtest_store"].items():
            csv_data.append({"項目": f"Backtest_{key}", "數值": str(value)})

    # 返回報告和 CSV 下載
    report_text = "\n".join(report_lines)
    csv_df = pd.DataFrame(csv_data)

    return report_text, dcc.send_data_frame(csv_df.to_csv, "data_comparison_report.csv", index=False)

def _first_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

# --------- 增強分析 Callback：交易貢獻拆解（修正版） ---------
@app.callback(
    Output("phase-table", "children"),
    Input("run-phase", "n_clicks"),
    State("phase-min-gap", "value"),
    State("phase-cooldown", "value"),
    State("enhanced-trades-cache", "data"),
    State("theme-store", "data"),   # 若沒有 theme-store，這行與下方 theme 相關可移除
    prevent_initial_call=True
)
def _run_phase(n_clicks, min_gap, cooldown, cache, theme):
    import numpy as np
    from urllib.parse import quote as urlparse
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not cache:
        return html.Div("尚未載入回測結果", style={"color": "#ffb703"})

    # 從快取還原資料
    trade_df = df_from_pack(cache.get("trade_data"))
    daily_state = df_from_pack(cache.get("daily_state"))

    if trade_df is None or trade_df.empty:
        return "找不到交易資料"

    if daily_state is None or daily_state.empty:
        return "找不到 daily_state（每日資產/權重）"

    if "equity" not in daily_state.columns:
        return "daily_state 缺少權益欄位 'equity'"

    equity = daily_state["equity"]

    # 呼叫你已寫好的分析函數
    try:
        from SSS_EnsembleTab import trade_contribution_by_phase
        table = trade_contribution_by_phase(trade_df, equity, min_gap, cooldown).copy()
    except Exception as e:
        return f"交易貢獻拆解執行失敗: {e}"

    if table.empty:
        return "無資料"

    # 數字欄位轉型
    num_cols = ["交易筆數","賣出報酬總和(%)","階段內MDD(%)","階段淨貢獻(%)"]
    for c in num_cols:
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce")

    # ====== 總體 KPI ======
    avg_net = table["階段淨貢獻(%)"].mean() if "階段淨貢獻(%)" in table else np.nan
    avg_mdd = table["階段內MDD(%)"].mean() if "階段內MDD(%)" in table else np.nan
    succ_all = (table["階段淨貢獻(%)"] > 0).mean() if "階段淨貢獻(%)" in table else np.nan
    succ_acc = np.nan
    if "階段" in table.columns and "階段淨貢獻(%)" in table.columns:
        mask_acc = table["階段"].astype(str).str.contains("加碼", na=False)
        if mask_acc.any():
            succ_acc = (table.loc[mask_acc, "階段淨貢獻(%)"] > 0).mean()
    risk_eff = np.nan
    if pd.notna(avg_net) and pd.notna(avg_mdd) and avg_mdd != 0:
        risk_eff = avg_net / abs(avg_mdd)

    # ====== CSV 文字（給複製用；DataTable 另有內建下載）======
    csv_text = table.to_csv(index=False)
    csv_data_url = "data:text/csv;charset=utf-8," + urlparse(csv_text)

    # ====== 主題樣式（避免白底白字）======
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
        # 允許選取→可複製
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

    # ====== 完整表格 ======
    # 注意：full_table 將在 ordered 變數定義後重新定義

    # ====== 易讀版（KPI + Top3 / Worst3）======
    def kpi(label, value):
        return html.Div([
            html.Div(label, style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "bold"})
        ], style={
            "backgroundColor": accent_bg, "color": accent_color,
            "padding": "10px 14px", "borderRadius": "12px", "minWidth": "160px"
        })

    kpi_bar = html.Div([
        kpi("平均每段淨貢獻(%)", f"{avg_net:.2f}" if pd.notna(avg_net) else "—"),
        kpi("平均每段 MDD(%)", f"{avg_mdd:.2f}" if pd.notna(avg_mdd) else "—"),
        kpi("成功率(全部)", f"{succ_all*100:.1f}%" if pd.notna(succ_all) else "—"),
        kpi("成功率(加碼)", f"{succ_acc*100:.1f}%" if pd.notna(succ_acc) else "—"),
        kpi("風險效率", f"{risk_eff:.3f}" if pd.notna(risk_eff) else "—"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"10px"})

    # ====== 分組 KPI：加碼 vs 減碼 ======
    def _group_metrics(mask):
        if {"階段淨貢獻(%)","階段內MDD(%)"}.issubset(table.columns):
            sub = table.loc[mask]
            if sub.empty:
                return None
            a_net = sub["階段淨貢獻(%)"].mean()
            a_mdd = sub["階段內MDD(%)"].mean()
            succ  = (sub["階段淨貢獻(%)"] > 0).mean()
            eff   = (a_net / abs(a_mdd)) if pd.notna(a_net) and pd.notna(a_mdd) and a_mdd != 0 else np.nan
            return {"count": int(len(sub)), "avg_net": a_net, "avg_mdd": a_mdd, "succ": succ, "eff": eff}
        return None

    def _fmt(val, pct=False, dec=2):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—"
        return f"{val*100:.1f}%" if pct else f"{val:.{dec}f}"

    def group_row(title, m):
        return html.Div([
            html.Div(title, style={"fontWeight":"bold","marginRight":"12px","minWidth":"72px","alignSelf":"center"}),
            kpi("段數", f"{m['count']}" if m else "—"),
            kpi("平均淨貢獻(%)", _fmt(m['avg_net']) if m else "—"),
            kpi("平均MDD(%)",   _fmt(m['avg_mdd']) if m else "—"),
            kpi("成功率",        _fmt(m['succ'], pct=True) if m else "—"),
            kpi("風險效率",      _fmt(m['eff'],  dec=3) if m else "—"),
        ], style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"8px"})

    acc_metrics = dis_metrics = None
    if "階段" in table.columns:
        mask_acc = table["階段"].astype(str).str.contains("加碼", na=False)
        mask_dis = table["階段"].astype(str).str.contains("減碼", na=False)
        acc_metrics = _group_metrics(mask_acc)
        dis_metrics = _group_metrics(mask_dis)

    group_section = html.Div([
        html.H6("分組 KPI（加碼 vs 減碼）", style={"margin":"8px 0 6px 0"}),
        group_row("加碼段", acc_metrics),
        group_row("減碼段", dis_metrics),
    ], style={"marginTop":"4px"})

    # ====== Top/Worst 來源切換（全部 / 只加碼 / 只減碼） ======
    source_selector = html.Div([
        html.Div("Top/Worst 來源", style={"marginRight":"8px", "alignSelf":"center"}),
        dcc.RadioItems(
            id="phase-source",
            options=[
                {"label": "全部",   "value": "all"},
                {"label": "加碼段", "value": "acc"},
                {"label": "減碼段", "value": "dis"},
            ],
            value="all",
            inline=True,
            inputStyle={"marginRight":"4px"},
            labelStyle={"marginRight":"12px"}
        )
    ], style={"display":"flex","gap":"6px","alignItems":"center","margin":"6px 0 8px 0"})

    # 欄位順序（完整表 & Top/Worst 共用）
    ordered = [c for c in ["階段","開始日期","結束日期","交易筆數",
                           "階段淨貢獻(%)","賣出報酬總和(%)","階段內MDD(%)","是否成功"] if c in table.columns]
    basis_col = "階段淨貢獻(%)" if "階段淨貢獻(%)" in table.columns else "賣出報酬總和(%)"

    # ====== 完整表格 ======
    full_table = dash_table.DataTable(
        id="phase-datatable",
        columns=[{"name": c, "id": c, "type": ("numeric" if c in num_cols else "text")} for c in ordered],
        data=table[ordered].to_dict("records"),
        # 分頁
        page_action="native",
        page_current=0,
        page_size=100,            # 預設每頁 100，若要改可在這裡
        # 互動
        sort_action="native",
        filter_action="native",
        # 下載
        export_format="csv",
        export_headers="display",
        # 複製
        cell_selectable=True,
        virtualization=False,     # 關閉虛擬化，避免複製時只複到可視區
        fixed_rows={"headers": True},
        style_table=style_table,
        style_cell=style_cell,
        style_header=style_header,
        css=[{
            "selector": ".dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner *",
            "rule": "user-select: text; -webkit-user-select: text; -moz-user-select: text; -ms-user-select: text;"
        }],
    )

    # ====== dcc.Store：提供 Top/Worst 動態 callback 使用 ======
    store = dcc.Store(id="phase-table-store", data={
        "records": table[ordered].to_dict("records"),
        "ordered": ordered,
        "basis": basis_col,
        "has_stage": "階段" in table.columns
    })

    # 預設（全部來源）先算一次，避免空畫面
    def _subset(src):
        df = table
        if "階段" not in df.columns:
            return df
        if src == "acc":
            return df[df["階段"].astype(str).str.contains("加碼", na=False)]
        if src == "dis":
            return df[df["階段"].astype(str).str.contains("減碼", na=False)]
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

    # ====== Copy / Download 工具列 ======
    tools = html.Div([
        html.Button("複製全部（CSV）", id="phase-copy-btn",
                    style={"padding": "6px 10px", "borderRadius": "8px", "cursor": "pointer"}),
        dcc.Clipboard(target_id="phase-csv-text", title="Copy", style={"marginLeft": "6px"}),
        html.A("下載 CSV", href=csv_data_url, download="trade_contribution.csv",
               style={"marginLeft": "12px", "textDecoration": "none"})
    ], style={"display": "flex", "alignItems": "center", "gap": "4px", "marginBottom": "8px"})

    # 隱藏的 CSV 文字來源（給 Clipboard 用）
    csv_hidden = html.Pre(id="phase-csv-text", children=csv_text, style={"display": "none"})

    # ====== Tabs：易讀版 / 完整表格 ======
    tabs = dcc.Tabs(id="phase-tabs", value="summary", children=[
        dcc.Tab(label="易讀版", value="summary", children=[
            kpi_bar,
            group_section,
            source_selector,
            html.H6("最賺的 3 段（依來源與排序欄）", style={"marginTop":"8px"}),
            top3_table,
            html.H6("最虧的 3 段（依來源與排序欄）", style={"marginTop":"16px"}),
            worst3_table
        ]),
        dcc.Tab(label="完整表格", value="full", children=[full_table]),
    ])

    return html.Div([tools, csv_hidden, store, tabs], style={"marginTop": "8px"})

# --------- 批量測試參數範圍 Callback ---------
@app.callback(
    Output("batch-phase-results", "children"),
    Input("run-batch-phase", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    prevent_initial_call=True
)
def _run_batch_phase_test(n_clicks, cache):
    """批量測試1-24範圍的最小間距和冷卻期參數"""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not cache:
        return html.Div("尚未載入回測結果", style={"color": "#ffb703"})

    # 從快取還原資料
    trade_df = df_from_pack(cache.get("trade_data"))
    daily_state = df_from_pack(cache.get("daily_state"))

    if trade_df is None or trade_df.empty:
        return "找不到交易資料"

    if daily_state is None or daily_state.empty:
        return "找不到 daily_state（每日資產/權重）"

    if "equity" not in daily_state.columns:
        return "daily_state 缺少權益欄位 'equity'"

    equity = daily_state["equity"]

    # 檢查並準備交易資料格式
    debug_info = []
    debug_info.append(f"原始交易資料欄位: {list(trade_df.columns)}")
    debug_info.append(f"交易資料行數: {len(trade_df)}")
    debug_info.append(f"權益資料行數: {len(equity)}")

    # 檢查必要欄位並進行轉換
    required_mappings = {
        "date": ["date", "trade_date", "交易日期", "Date"],
        "type": ["type", "交易類型", "action", "side", "Type"],
        "w_before": ["w_before", "交易前權重", "weight_before", "weight_prev"],
        "w_after": ["w_after", "交易後權重", "weight_after", "weight_next"]
    }

    # 尋找對應的欄位
    found_columns = {}
    for target, possible_names in required_mappings.items():
        for name in possible_names:
            if name in trade_df.columns:
                found_columns[target] = name
                break

    debug_info.append(f"找到的欄位對應: {found_columns}")

    # 如果缺少必要欄位，嘗試創建
    if len(found_columns) < 4:
        debug_info.append("缺少必要欄位，嘗試創建...")

        # 嘗試從現有欄位推導
        if "weight_change" in trade_df.columns and "w_before" not in found_columns:
            # 如果有權重變化，嘗試重建前後權重
            trade_df = trade_df.copy()
            trade_df["w_before"] = 0.0
            trade_df["w_after"] = trade_df["weight_change"]
            found_columns["w_before"] = "w_before"
            found_columns["w_after"] = "w_after"
            debug_info.append("從 weight_change 創建 w_before 和 w_after")

        if "price" in trade_df.columns and "type" not in found_columns:
            # 如果有價格，假設為買入
            trade_df["type"] = "buy"
            found_columns["type"] = "type"
            debug_info.append("創建 type 欄位，預設為 buy")

    # 批量測試參數範圍 1-24
    results = []
    total_combinations = 24 * 24  # 576種組合

    try:
        from SSS_EnsembleTab import trade_contribution_by_phase

        # 進度顯示
        progress_div = html.Div([
            html.H6("正在執行批量測試...", style={"color": "#28a745"}),
            html.Div(f"測試範圍：最小間距 1-24 天，冷卻期 1-24 天", style={"fontSize": "12px", "color": "#666"}),
            html.Div(f"總組合數：{total_combinations}", style={"fontSize": "12px", "color": "#666"}),
            html.Div(id="batch-progress", children="開始測試...")
        ])

        # 執行批量測試
        batch_results = []
        debug_info = []

        # 先測試一個簡單的案例
        test_min_gap, test_cooldown = 1, 1
        try:
            debug_info.append(f"開始測試單一案例: min_gap={test_min_gap}, cooldown={test_cooldown}")

            # 檢查交易資料的權重欄位
            if "weight_change" in trade_df.columns:
                debug_info.append(f"找到 weight_change 欄位，範圍: {trade_df['weight_change'].min():.4f} ~ {trade_df['weight_change'].max():.4f}")

            # 檢查權益資料
            if len(equity) > 0:
                debug_info.append(f"權益資料範圍: {equity.min():.2f} ~ {equity.max():.2f}")

            table = trade_contribution_by_phase(trade_df, equity, test_min_gap, test_cooldown)
            debug_info.append(f"函數執行成功，返回表格大小: {table.shape}")
            debug_info.append(f"表格欄位: {list(table.columns)}")

            if not table.empty:
                debug_info.append(f"第一行資料: {table.iloc[0].to_dict()}")

                # 檢查是否有階段淨貢獻欄位
                if "階段淨貢獻(%)" in table.columns:
                    debug_info.append(f"階段淨貢獻欄位存在，非空值數量: {table['階段淨貢獻(%)'].notna().sum()}")
                    debug_info.append(f"階段淨貢獻範圍: {table['階段淨貢獻(%)'].min():.2f} ~ {table['階段淨貢獻(%)'].max():.2f}")
                else:
                    debug_info.append("缺少階段淨貢獻欄位")

                if "階段內MDD(%)" in table.columns:
                    debug_info.append(f"階段內MDD欄位存在，非空值數量: {table['階段內MDD(%)'].notna().sum()}")
                    debug_info.append(f"階段內MDD範圍: {table['階段內MDD(%)'].min():.2f} ~ {table['階段內MDD(%)'].max():.2f}")
                else:
                    debug_info.append("缺少階段內MDD欄位")
            else:
                debug_info.append("函數返回空表格")

        except Exception as e:
            import traceback
            debug_info.append(f"函數執行錯誤: {str(e)}")
            debug_info.append(f"錯誤詳情: {traceback.format_exc()}")

        # 如果單一測試成功，繼續批量測試
        if not table.empty and "階段淨貢獻(%)" in table.columns and "階段內MDD(%)" in table.columns:
            debug_info.append("單一測試成功，開始批量測試...")

            for min_gap in range(1, 25):
                for cooldown in range(1, 25):
                    try:
                        table = trade_contribution_by_phase(trade_df, equity, min_gap, cooldown)

                        if not table.empty:
                            # 過濾掉摘要行（通常包含"統計摘要"字樣）
                            data_rows = table[~table["階段"].astype(str).str.contains("統計摘要", na=False)]

                            if len(data_rows) == 0:
                                continue

                            # 計算關鍵指標
                            avg_net = data_rows["階段淨貢獻(%)"].mean()
                            avg_mdd = data_rows["階段內MDD(%)"].mean()
                            succ_rate = (data_rows["階段淨貢獻(%)"] > 0).mean()
                            risk_eff = avg_net / abs(avg_mdd) if avg_mdd != 0 else 0

                            batch_results.append({
                                "最小間距": min_gap,
                                "冷卻期": cooldown,
                                "平均淨貢獻(%)": round(avg_net, 2),
                                "平均MDD(%)": round(avg_mdd, 2),
                                "成功率(%)": round(succ_rate * 100, 1),
                                "風險效率": round(risk_eff, 3),
                                "階段數": len(data_rows)
                            })
                    except Exception as e:
                        # 記錄錯誤但繼續執行
                        continue
        else:
            debug_info.append("單一測試失敗，跳過批量測試")

        if not batch_results:
            # 顯示除錯資訊
            debug_html = html.Div([
                html.H6("除錯資訊", style={"color": "#dc3545", "marginTop": "16px"}),
                html.Div([html.Pre(info) for info in debug_info], style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "4px", "fontSize": "11px"})
            ])

            return html.Div([
                html.Div("批量測試完成，但無有效結果", style={"color": "#ffb703"}),
                html.Div("可能原因：", style={"marginTop": "8px", "color": "#666"}),
                html.Ul([
                    html.Li("交易資料格式不正確"),
                    html.Li("缺少必要的欄位（階段淨貢獻(%)、階段內MDD(%)）"),
                    html.Li("所有參數組合都無法產生有效階段"),
                    html.Li("函數執行時發生錯誤")
                ], style={"fontSize": "12px", "color": "#666"}),
                debug_html
            ])

        # 轉換為DataFrame並排序
        results_df = pd.DataFrame(batch_results)

        # 按風險效率排序（降序）
        results_df = results_df.sort_values("風險效率", ascending=False)

        # 生成CSV下載連結
        csv_text = results_df.to_csv(index=False)
        csv_data_url = "data:text/csv;charset=utf-8," + urlparse(csv_text)

        # 顯示前10名結果
        top10 = results_df.head(10)

        # 生成結果表格
        results_table = dash_table.DataTable(
            id="batch-results-table",
            columns=[{"name": c, "id": c} for c in results_df.columns],
            data=top10.to_dict("records"),
            page_action="none",
            style_table={"overflowX": "auto", "fontSize": "11px"},
            style_cell={"textAlign": "center", "padding": "4px", "minWidth": "60px"},
            style_header={"backgroundColor": "#28a745", "color": "white", "fontWeight": "bold"}
        )

        # 統計摘要
        summary_stats = html.Div([
            html.H6("批量測試摘要", style={"marginTop": "16px", "marginBottom": "8px", "color": "#28a745"}),
            html.Div(f"有效組合數：{len(results_df)} / {total_combinations}", style={"fontSize": "12px"}),
            html.Div(f"最佳風險效率：{results_df['風險效率'].max():.3f}", style={"fontSize": "12px"}),
            html.Div(f"最佳平均淨貢獻：{results_df['平均淨貢獻(%)'].max():.2f}%", style={"fontSize": "12px"}),
            html.Div(f"最佳成功率：{results_df['成功率(%)'].max():.1f}%", style={"fontSize": "12px"}),
            html.Div([
                html.Button("下載完整結果CSV", id="download-batch-csv",
                           style={"backgroundColor": "#28a745", "color": "white", "border": "none", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}),
                html.A("直接下載", href=csv_data_url, download="batch_phase_test_results.csv",
                       style={"marginLeft": "12px", "textDecoration": "none", "color": "#28a745"})
            ], style={"marginTop": "8px"})
        ])

        return html.Div([
            summary_stats,
            html.H6("前10名最佳參數組合（按風險效率排序）", style={"marginTop": "16px", "marginBottom": "8px"}),
            results_table
        ])

    except Exception as e:
        return html.Div(f"批量測試執行失敗: {str(e)}", style={"color": "#dc3545"})

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

# --------- 增強分析 Callback A：依 backtest-store 填滿策略選單 ---------
@app.callback(
    Output("enhanced-strategy-selector", "options"),
    Output("enhanced-strategy-selector", "value"),
    Input("backtest-store", "data"),
    prevent_initial_call=False
)
def _populate_enhanced_strategy_selector(bstore):
    """依 backtest-store 填滿策略選單，並自動選擇最佳策略"""
    if not bstore:
        return [], None

    results = bstore.get("results", {})
    if not results:
        return [], None

    # 策略評分：ledger_std > ledger > trade_df
    strategy_scores = []
    for strategy_name, result in results.items():
        score = 0
        if result.get("trade_ledger_std"):
            score += 100  # 最高分：標準化交易流水帳
        elif result.get("trade_ledger"):
            score += 50   # 中分：原始交易流水帳
        elif result.get("trade_df"):
            score += 10   # 低分：交易明細

        # 額外加分：有 daily_state
        if result.get("daily_state") or result.get("daily_state_std"):
            score += 20

        strategy_scores.append((strategy_name, score))

    # 按分數排序
    strategy_scores.sort(key=lambda x: x[1], reverse=True)

    # 生成選單選項
    options = [{"label": f"{name} (分數: {score})", "value": name}
               for name, score in strategy_scores]

    # 自動選擇最高分策略
    auto_select = strategy_scores[0][0] if strategy_scores else None

    return options, auto_select

# --------- 增強分析 Callback B：載入選定策略到 enhanced-trades-cache ---------
@app.callback(
    Output("enhanced-trades-cache", "data"),
    Output("enhanced-load-status", "children"),
    Input("load-enhanced-strategy", "n_clicks"),
    State("enhanced-strategy-selector", "value"),
    State("backtest-store", "data"),
    prevent_initial_call=True
)
def _load_enhanced_strategy_to_cache(n_clicks, selected_strategy, bstore):
    """載入選定策略的回測結果到 enhanced-trades-cache"""
    if not n_clicks or not selected_strategy or not bstore:
        return no_update, "請選擇策略並點擊載入"

    results = bstore.get("results", {})
    if selected_strategy not in results:
        return no_update, f"找不到策略：{selected_strategy}"

    result = results[selected_strategy]

    # 優先順序：ledger_std > ledger > trade_df
    trade_data = None
    data_source = ""

    if result.get("trade_ledger_std"):
        trade_data = df_from_pack(result["trade_ledger_std"])
        data_source = "trade_ledger_std (標準化)"
    elif result.get("trade_ledger"):
        trade_data = df_from_pack(result["trade_ledger"])
        data_source = "trade_ledger (原始)"
    elif result.get("trade_df"):
        trade_data = df_from_pack(result["trade_df"])
        data_source = "trade_df (交易明細)"
    else:
        return no_update, "該策略無交易資料"

    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_data = norm(trade_data)
    except Exception:
        # 後備標準化方案
        if trade_data is not None and len(trade_data) > 0:
            trade_data = trade_data.copy()
            trade_data.columns = [str(c).lower() for c in trade_data.columns]

            # 確保有 trade_date 欄
            if "trade_date" not in trade_data.columns:
                if "date" in trade_data.columns:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
                elif isinstance(trade_data.index, pd.DatetimeIndex):
                    trade_data = trade_data.reset_index().rename(columns={"index": "trade_date"})
                else:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")

            # 確保有 type 欄
            if "type" not in trade_data.columns:
                if "action" in trade_data.columns:
                    trade_data["type"] = trade_data["action"].astype(str).str.lower()
                elif "side" in trade_data.columns:
                    trade_data["type"] = trade_data["side"].astype(str).str.lower()
                else:
                    trade_data["type"] = "hold"

            # 確保有 price 欄
            if "price" not in trade_data.columns:
                for c in ["open", "price_open", "exec_price", "px", "close"]:
                    if c in trade_data.columns:
                        trade_data["price"] = trade_data[c]
                        break
                if "price" not in trade_data.columns:
                    trade_data["price"] = 0.0

    # 準備 daily_state - 若已套用閥門則優先使用調整後資料
    daily_state = None
    valve_info = result.get("valve", {})
    valve_on = bool(valve_info.get("applied", False))

    # app_dash.py / 2025-08-22 15:30
    # 智能選擇日線資料：優先使用 valve 版本（如果啟用且存在），否則使用 baseline
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
    # 相容性：優先使用 valve 權重曲線，否則退回原本欄位（與 O2 一致）
    weight_curve = None
    if result.get("weight_curve_valve"):
        weight_curve = df_from_pack(result["weight_curve_valve"])
    elif result.get("weight_curve"):
        weight_curve = df_from_pack(result["weight_curve"])
    elif result.get("weight_curve_base"):
        weight_curve = df_from_pack(result["weight_curve_base"])

    # 獲取閥門狀態資訊
    valve_info = result.get("valve", {})  # {"applied": bool, "cap": float, "atr_ratio": float or "N/A"}
    valve_on = bool(valve_info.get("applied", False))

    # 若閥門生效，保證分析端覆寫 w_series
    if valve_on and weight_curve is not None and daily_state is not None:
        ds = daily_state.copy()
        wc = weight_curve.copy()
        # 對齊時間索引；若 ds 有 'trade_date' 欄就 merge，否則以索引對齊
        if "trade_date" in ds.columns:
            ds["trade_date"] = pd.to_datetime(ds["trade_date"])
            wc = wc.rename("w").to_frame().reset_index().rename(columns={"index": "trade_date"})
            ds = ds.merge(wc, on="trade_date", how="left")
        else:
            # 以索引對齊
            ds.index = pd.to_datetime(ds.index)
            wc.index = pd.to_datetime(wc.index)
            # 修正：確保 wc 是 Series 並且正確對齊
            if isinstance(wc, pd.DataFrame):
                if "w" in wc.columns:
                    wc_series = wc["w"]
                else:
                    wc_series = wc.iloc[:, 0]  # 取第一列
            else:
                wc_series = wc
            ds["w"] = wc_series.reindex(ds.index).ffill().bfill()
        daily_state = ds

    # 準備 df_raw
    df_raw = None
    if bstore.get("df_raw"):
        try:
            df_raw = df_from_pack(bstore["df_raw"])
        except Exception:
            df_raw = pd.DataFrame()

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
        # ➌ 新增：baseline 與 valve 版本一併放進快取
        "daily_state_base": result.get("daily_state_base"),
        "weight_curve_base": result.get("weight_curve_base"),
        "trade_ledger_base": result.get("trade_ledger_base"),
        # ➍ 新增：valve 版本一併放進快取
        "daily_state_valve": result.get("daily_state_valve"),
        "weight_curve_valve": result.get("weight_curve_valve"),
        "trade_ledger_valve": result.get("trade_ledger_valve"),
        "equity_curve_valve": result.get("equity_curve_valve"),
    }

    status_msg = f"✅ 已載入 {selected_strategy} ({data_source})"
    if daily_state is not None:
        status_msg += f"，包含 {len(daily_state)} 筆日線資料"
    if trade_data is not None:
        status_msg += f"，包含 {len(trade_data)} 筆交易"

    return cache_data, status_msg

# --------- 增強分析 Callback C：自動快取最佳策略 ---------
@app.callback(
    Output("enhanced-trades-cache", "data", allow_duplicate=True),
    Output("enhanced-load-status", "children", allow_duplicate=True),
    Input("backtest-store", "data"),
    State("enhanced-strategy-selector", "value"),
    prevent_initial_call='initial_duplicate'
)
def _auto_cache_best_strategy(bstore, current_selection):
    """回測完成後自動快取最佳策略"""
    if not bstore:
        return no_update, no_update

    results = bstore.get("results", {})
    if not results:
        return no_update, no_update

    # 如果已經有手動選擇，不覆蓋
    if current_selection:
        return no_update, no_update

    # 策略評分：ledger_std > ledger > trade_df
    strategy_scores = []
    for strategy_name, result in results.items():
        score = 0
        if result.get("trade_ledger_std"):
            score += 100  # 最高分：標準化交易流水帳
        elif result.get("trade_ledger"):
            score += 50   # 中分：原始交易流水帳
        elif result.get("trade_df"):
            score += 10   # 低分：交易明細

        # 額外加分：有 daily_state
        if result.get("daily_state") or result.get("daily_state_std"):
            score += 20

        strategy_scores.append((strategy_name, score))

    # 按分數排序，選擇最佳策略
    if not strategy_scores:
        return no_update, no_update

    strategy_scores.sort(key=lambda x: x[1], reverse=True)
    best_strategy = strategy_scores[0][0]
    best_result = results[best_strategy]

    # 準備交易資料（優先順序：ledger_std > ledger > trade_df）
    trade_data = None
    data_source = ""

    if best_result.get("trade_ledger_std"):
        trade_data = df_from_pack(best_result["trade_ledger_std"])
        data_source = "trade_ledger_std (標準化)"
    elif best_result.get("trade_ledger"):
        trade_data = df_from_pack(best_result["trade_ledger"])
        data_source = "trade_ledger (原始)"
    elif best_result.get("trade_df"):
        trade_data = df_from_pack(best_result["trade_df"])
        data_source = "trade_df (交易明細)"
    else:
        return no_update, no_update

    # 標準化交易資料
    try:
        from sss_core.normalize import normalize_trades_for_ui as norm
        trade_data = norm(trade_data)
    except Exception:
        # 後備標準化方案
        if trade_data is not None and len(trade_data) > 0:
            trade_data = trade_data.copy()
            trade_data.columns = [str(c).lower() for c in trade_data.columns]

            # 確保有 trade_date 欄
            if "trade_date" not in trade_data.columns:
                if "date" in trade_data.columns:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")
                elif isinstance(trade_data.index, pd.DatetimeIndex):
                    trade_data = trade_data.reset_index().rename(columns={"index": "trade_date"})
                else:
                    trade_data["trade_date"] = pd.to_datetime(trade_data["date"], errors="coerce")

            # 確保有 type 欄
            if "type" not in trade_data.columns:
                if "action" in trade_data.columns:
                    trade_data["type"] = trade_data["action"].astype(str).str.lower()
                elif "side" in trade_data.columns:
                    trade_data["type"] = trade_data["side"].astype(str).str.lower()
                else:
                    trade_data["type"] = "hold"

            # 確保有 price 欄
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
    # 智能選擇日線資料：優先使用 valve 版本（如果啟用且存在），否則使用 baseline
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
    # 相容性：優先使用 valve 權重曲線，否則退回原本欄位（與 O2 一致）
    weight_curve = None
    if best_result.get("weight_curve_valve"):
        weight_curve = df_from_pack(best_result["weight_curve_valve"])
    elif best_result.get("weight_curve"):
        weight_curve = df_from_pack(best_result["weight_curve"])
    elif best_result.get("weight_curve_base"):
        weight_curve = df_from_pack(best_result["weight_curve_base"])

    # 若閥門生效，保證分析端覆寫 w_series
    if valve_on and weight_curve is not None and daily_state is not None:
        ds = daily_state.copy()
        wc = weight_curve.copy()
        # 對齊時間索引；若 ds 有 'trade_date' 欄就 merge，否則以索引對齊
        if "trade_date" in ds.columns:
            ds["trade_date"] = pd.to_datetime(ds["trade_date"])
            wc = wc.rename("w").to_frame().reset_index().rename(columns={"index": "trade_date"})
            ds = ds.merge(wc, on="trade_date", how="left")
        else:
            # 以索引對齊
            ds.index = pd.to_datetime(ds.index)
            wc.index = pd.to_datetime(wc.index)
            # 修正：確保 wc 是 Series 並且正確對齊
            if isinstance(wc, pd.DataFrame):
                if "w" in wc.columns:
                    wc_series = wc["w"]
                else:
                    wc_series = wc.iloc[:, 0]  # 取第一列
            else:
                wc_series = wc
            ds["w"] = wc_series.reindex(ds.index).ffill().bfill()
        daily_state = ds

    # 準備 df_raw
    df_raw = None
    if bstore.get("df_raw"):
        try:
            df_raw = df_from_pack(bstore["df_raw"])
        except Exception:
            df_raw = pd.DataFrame()

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
        # ➌ 新增：baseline 版本一併放進快取
        "daily_state_base": best_result.get("daily_state_base"),
        "weight_curve_base": best_result.get("weight_curve_base"),
        "trade_ledger_base": best_result.get("trade_ledger_base"),
        # ➍ 新增：valve 版本一併放進快取
        "daily_state_valve": best_result.get("daily_state_valve"),
        "weight_curve_valve": best_result.get("weight_curve_valve"),
        "trade_ledger_valve": best_result.get("trade_ledger_valve"),
        "equity_curve_valve": best_result.get("equity_curve_valve"),
    }

    status_msg = f"🔄 自動快取最佳策略：{best_strategy} ({data_source})"
    if daily_state is not None:
        status_msg += f"，包含 {len(daily_state)} 筆日線資料"
    if trade_data is not None:
        status_msg += f"，包含 {len(trade_data)} 筆交易"

    return cache_data, status_msg

# --------- 新增：風險-報酬地圖（Pareto Map）Callback ---------
@app.callback(
    Output("pareto-map-graph", "figure"),
    Output("pareto-map-status", "children"),
    Input("generate-pareto-map", "n_clicks"),
    State("enhanced-trades-cache", "data"),
    State("backtest-store", "data"),
    State("rv-mode", "value"),
    State("rv-cap", "value"),
    State("rv-atr-mult", "value"),
    prevent_initial_call=True
)
def generate_pareto_map(n_clicks, cache, backtest_data, rv_mode, rv_cap_value, rv_atr_value):
    """生成風險-報酬地圖（Pareto Map）：掃描 cap 與 ATR(20)/ATR(60) 比值全組合"""
    logger.info(f"=== Pareto Map 生成開始 ===")
    logger.info(f"n_clicks: {n_clicks}")
    logger.info(f"cache 存在: {cache is not None}")
    logger.info(f"backtest_data 存在: {backtest_data is not None}")

    if not n_clicks:
        logger.warning("沒有點擊事件")
        return go.Figure(), "❌ 請點擊生成按鈕"

    # 優先使用 enhanced-trades-cache，如果沒有則嘗試從 backtest-store 生成
    if cache:
        logger.info("使用 enhanced-trades-cache 資料")
        df_raw = df_from_pack(cache.get("df_raw"))
        daily_state = df_from_pack(cache.get("daily_state"))
        data_source = "enhanced-trades-cache"
        logger.info(f"df_raw 形狀: {df_raw.shape if df_raw is not None else 'None'}")
        logger.info(f"daily_state 形狀: {daily_state.shape if daily_state is not None else 'None'}")
    elif backtest_data and backtest_data.get("results"):
        logger.info("使用 backtest-store 資料")
        results = backtest_data["results"]
        logger.info(f"可用策略: {list(results.keys())}")

        # 從 backtest-store 選擇第一個有 daily_state 的策略
        selected_strategy = None
        for strategy_name, result in results.items():
            logger.info(f"檢查策略 {strategy_name}: daily_state={result.get('daily_state') is not None}, daily_state_std={result.get('daily_state_std') is not None}")
            if result.get("daily_state") or result.get("daily_state_std"):
                selected_strategy = strategy_name
                logger.info(f"選擇策略: {selected_strategy}")
                break

        if not selected_strategy:
            logger.error("沒有找到包含 daily_state 的策略")
            return go.Figure(), "❌ 回測結果中沒有找到包含 daily_state 的策略"

        result = results[selected_strategy]
        daily_state = df_from_pack(result.get("daily_state") or result.get("daily_state_std"))
        df_raw = df_from_pack(backtest_data.get("df_raw"))
        data_source = f"backtest-store ({selected_strategy})"
        logger.info(f"df_raw 形狀: {df_raw.shape if df_raw is not None else 'None'}")
        logger.info(f"daily_state 形狀: {daily_state.shape if daily_state is not None else 'None'}")
    else:
        logger.error("沒有可用的資料來源")
        return go.Figure(), "❌ 請先執行回測，或於『🧠 從回測結果載入』載入策略"

    # 資料驗證
    logger.info("=== 資料驗證 ===")
    if df_raw is None or df_raw.empty:
        logger.error("df_raw 為空")
        return go.Figure(), "❌ 找不到股價資料 (df_raw)"
    if daily_state is None or daily_state.empty:
        logger.error("daily_state 為空")
        return go.Figure(), "❌ 找不到 daily_state（每日資產/權重）"

    # 資料不足時的行為對齊
    if len(daily_state) < 60:
        logger.warning("資料不足（<60天），已略過掃描")
        return go.Figure(), "⚠️ 資料不足（<60天），已略過掃描"

    logger.info(f"df_raw 欄位: {list(df_raw.columns)}")
    logger.info(f"daily_state 欄位: {list(daily_state.columns)}")

    # 欄名對齊
    c_open = "open" if "open" in df_raw.columns else _first_col(df_raw, ["Open","開盤價"])
    c_close = "close" if "close" in df_raw.columns else _first_col(df_raw, ["Close","收盤價"])
    c_high  = "high" if "high" in df_raw.columns else _first_col(df_raw, ["High","最高價"])
    c_low   = "low"  if "low"  in df_raw.columns else _first_col(df_raw, ["Low","最低價"])

    logger.info(f"欄名對齊結果: open={c_open}, close={c_close}, high={c_high}, low={c_low}")

    if c_open is None or c_close is None:
        logger.error("缺少必要的價格欄位")
        return go.Figure(), "❌ 股價資料缺少 open/close 欄位"

    # 準備輸入序列
    open_px = pd.to_numeric(df_raw[c_open], errors="coerce").dropna()
    open_px.index = pd.to_datetime(df_raw.index)

    # 取 open_px 後，準備 w（baseline 優先）
    ds_base = df_from_pack(cache.get("daily_state_base")) if cache else None
    wc_base = series_from_pack(cache.get("weight_curve_base")) if cache else None

    # 從 backtest-store 來的情況
    if ds_base is None and (not cache) and backtest_data and "results" in backtest_data:
        ds_base = df_from_pack(result.get("daily_state_base"))
        # 注意：weight_curve_base 也可能存在於 result
        try:
            wc_base = series_from_pack(result.get("weight_curve_base"))
        except Exception:
            wc_base = None

    # 以 baseline w 為優先；沒有再退回現行 daily_state['w']
    if ds_base is not None and (not ds_base.empty) and ("w" in ds_base.columns):
        w = pd.to_numeric(ds_base["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
    elif wc_base is not None and (not wc_base.empty):
        w = pd.to_numeric(wc_base, errors="coerce").reindex(open_px.index).ffill().fillna(0.0)
    else:
        # 後備：沿用現行 daily_state（可能已被閥門壓過）
        if "w" not in daily_state.columns:
            return go.Figure(), "❌ daily_state 缺少權重欄位 'w'"
        w = pd.to_numeric(daily_state["w"], errors="coerce").reindex(open_px.index).ffill().fillna(0.0)

    bench = pd.DataFrame({
        "收盤價": pd.to_numeric(df_raw[c_close], errors="coerce"),
    }, index=pd.to_datetime(df_raw.index))
    if c_high and c_low:
        bench["最高價"] = pd.to_numeric(df_raw[c_high], errors="coerce")
        bench["最低價"] = pd.to_numeric(df_raw[c_low], errors="coerce")

    # ATR 樣本檢查（與狀態面板一致）
    logger.info("=== ATR 樣本檢查 ===")
    a20, a60 = calculate_atr(df_raw, 20), calculate_atr(df_raw, 60)
    if a20 is None or a60 is None or a20.dropna().size < 60 or a60.dropna().size < 60:
        logger.warning("ATR 樣本不足，回傳警示")
        return go.Figure(), "🟡 ATR 樣本不足（請拉長期間或改用更長資料）"

    # 掃描參數格點 - 把全局門檻置入格點
    logger.info("=== 開始掃描參數格點 ===")
    import numpy as np

    # 讀取當前設定
    cap_now = float(rv_cap_value) if rv_cap_value else 0.8
    atr_now = float(rv_atr_value) if rv_atr_value else 1.2

    # 基本格點
    caps = np.round(np.linspace(0.10, 1.00, 19), 2)
    atr_mults = np.round(np.linspace(1.00, 2.00, 21), 2)

    # 將全局設定植入格點（避免被內插忽略）
    if rv_cap_value is not None:
        caps = np.unique(np.r_[caps, float(rv_cap_value)])
    if rv_atr_value is not None:
        atr_mults = np.unique(np.r_[atr_mults, float(rv_atr_value)])

    logger.info(f"當前設定: cap={cap_now:.2f}, atr={atr_now:.2f}")
    logger.info(f"cap 範圍: {len(caps)} 個值，從 {caps[0]} 到 {caps[-1]}")
    logger.info(f"ATR 比值範圍: {len(atr_mults)} 個值，從 {atr_mults[0]} 到 {atr_mults[-1]}")
    logger.info(f"總組合數: {len(caps) * len(atr_mults)}")

    pareto_rows = []
    tried = 0
    succeeded = 0

    # 檢查是否可以匯入 risk_valve_backtest
    try:
        from SSS_EnsembleTab import risk_valve_backtest
        logger.info("成功匯入 risk_valve_backtest")
    except Exception as e:
        logger.error(f"匯入 risk_valve_backtest 失敗: {e}")
        return go.Figure(), f"❌ 無法匯入 risk_valve_backtest: {e}"

    logger.info("開始執行參數掃描...")
    for cap_level in caps:
        for atr_mult in atr_mults:
            tried += 1
            if tried % 50 == 0:  # 每50次記錄一次進度
                logger.info(f"進度: {tried}/{len(caps) * len(atr_mults)} (cap={cap_level:.2f}, atr={atr_mult:.2f})")

            try:
                out = risk_valve_backtest(
                    open_px=open_px, w=w, cost=None, benchmark_df=bench,
                    mode=(rv_mode or "cap"), cap_level=float(cap_level),
                    slope20_thresh=0.0, slope60_thresh=0.0,
                    atr_win=20, atr_ref_win=60, atr_ratio_mult=float(atr_mult),
                    use_slopes=True, slope_method="polyfit", atr_cmp="gt"
                )

                if not isinstance(out, dict) or "metrics" not in out:
                    logger.warning(f"cap={cap_level:.2f}, atr={atr_mult:.2f}: 回傳格式異常")
                    continue

                m = out["metrics"]
                sig = out["signals"]["risk_trigger"]
                trigger_days = int(sig.fillna(False).sum())

                # 取用『閥門』版本作為此組合的點位
                pf = float(m.get("pf_valve", np.nan))
                mdd = float(m.get("mdd_valve", np.nan))
                rt_sum_valve = float(m.get("right_tail_sum_valve", np.nan))
                rt_sum_orig = float(m.get("right_tail_sum_orig", np.nan)) if m.get("right_tail_sum_orig") is not None else np.nan
                rt_reduction = float(m.get("right_tail_reduction", np.nan)) if m.get("right_tail_reduction") is not None else (rt_sum_orig - rt_sum_valve if np.isfinite(rt_sum_orig) and np.isfinite(rt_sum_valve) else np.nan)

                # 收集一筆點資料
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

                if succeeded % 20 == 0:  # 每20次成功記錄一次
                    logger.info(f"成功: {succeeded} 組 (cap={cap_level:.2f}, atr={atr_mult:.2f})")

            except Exception as e:
                logger.warning(f"cap={cap_level:.2f}, atr={atr_mult:.2f} 執行失敗: {e}")
                continue

    logger.info(f"=== 掃描完成 ===")
    logger.info(f"嘗試: {tried} 組，成功: {succeeded} 組")

    if not pareto_rows:
        logger.error("沒有成功生成任何資料點")
        return go.Figure(), "❌ 無法從風險閥門回測的參數組合中取得資料"

    # 用 reduction 當顏色（越大=削越多右尾→越紅），符合『顏色越紅＝削太多右尾』
    logger.info("開始處理結果資料...")
    dfp = pd.DataFrame(pareto_rows).dropna(subset=["pf","max_drawdown","right_tail_reduction"]).reset_index(drop=True)
    logger.info(f"處理後資料點數: {len(dfp)}")
    logger.info(f"dfp 欄位: {list(dfp.columns)}")

    if dfp.empty:
        logger.error("處理後資料為空")
        return go.Figure(), "❌ 資料處理後為空，請檢查原始資料"

    logger.info("開始繪製圖表...")
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
            colorbar=dict(title="右尾削減幅度")
        ),
        text=dfp['label'],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "MDD: %{x:.2%}<br>" +
            "PF: %{y:.2f}<br>" +
            "右尾總和(閥門): %{customdata[0]:.2f}<br>" +
            "右尾總和(原始): %{customdata[1]:.2f}<br>" +
            "右尾削減: %{marker.color:.2f}<br>" +
            "風險觸發天數: %{marker.size:.0f} 天<br>" +
            "<extra></extra>"
        ),
        customdata=dfp[["right_tail_sum_valve","right_tail_sum_orig"]].values,
        name="cap-atr grid"
    ))

    # 加入「Current」標記點（當前全局設定）
    if cap_now in caps and atr_now in atr_mults:
        # 找到當前設定對應的點位
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

    # 加入「Current」標記點（頁面門檻設定）
    if rv_cap_value is not None and rv_atr_value is not None:
        # 嘗試找到對應的掃描結果點位
        current_cap = float(rv_cap_value)
        current_atr = float(rv_atr_value)
        global_point = dfp[(dfp['cap'] == current_cap) & (dfp['atr'] == current_atr)]

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
                text=f"Current: cap={current_cap:.2f}, atr={current_atr:.2f}",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "MDD: %{x:.2%}<br>" +
                    "PF: %{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                name="Current Setting"
            ))

    fig.update_layout(
        title={
            'text': f'風險-報酬地圖（Pareto Map）- {succeeded}/{tried} 組',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="最大回撤（愈左愈好）",
        yaxis_title="PF 獲利因子（愈上愈好）",
        xaxis=dict(tickformat=".1%", gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(r=120)
    )

    status_msg = f"✅ 成功生成：掃描 cap×ATR 比值 {succeeded}/{tried} 組。顏色=右尾調整幅度（紅=削減，藍=放大），大小=風險觸發天數。目前頁面設定：cap={cap_now:.2f}, atr={atr_now:.2f}。資料來源：{data_source}"
    return fig, status_msg

# --------- 每日訊號：param_presets real/experiment 引擎 ---------
@app.callback(
    Output("prediction-status-msg", "children"),
    Output("daily-signal-results", "children"),
    Input("btn-run-prediction", "n_clicks"),
    State("ticker-dropdown", "value"),
    State("hide-strategy-presets", "value"),
    State({"type": "daily-estimate-input", "key": ALL}, "id"),
    State({"type": "daily-estimate-input", "key": ALL}, "value"),
    prevent_initial_call=True,
)
def run_prediction_script_callback(
    n_clicks,
    ticker,
    hidden_strategy_presets,
    estimate_ids,
    estimate_values,
):
    if not n_clicks:
        return no_update, no_update

    ticker = ticker or default_tickers[0]
    estimate_raw = {}
    for id_obj, value in zip(estimate_ids or [], estimate_values or []):
        if isinstance(id_obj, dict):
            key = id_obj.get("key")
            if key:
                estimate_raw[str(key)] = value

    try:
        result = run_daily_param_signals(
            ticker=ticker,
            hidden_strategy_presets=hidden_strategy_presets,
            estimates=estimate_raw,
            persist_experiment=False,
        )
    except Exception as exc:
        msg = html.Div(
            f"❌ 每日訊號計算失敗: {exc}",
            style={"color": "#dc3545", "fontWeight": "bold"},
        )
        return msg, html.Div()

    context = result["context"]
    real_df = result["real_df"].copy()
    exp_df = result["experiment_df"].copy()
    changed_df = result["changed_df"].copy()
    missing = result["missing_inputs"]
    signal_map = {"BUY": "買入", "SELL": "賣出", "HOLD": "觀望"}
    mode_map = {"real": "真實模式", "experiment": "實驗模式"}
    strategy_type_map = {
        "single": "單因子",
        "rma": "RMA",
        "ssma_turn": "SSMA 轉折",
        "ensemble": "集成",
    }

    def _display_df(df_in: pd.DataFrame, mode_name: str) -> pd.DataFrame:
        cols = [
            "strategy_name",
            "strategy_type",
            "mode",
            "signal",
            "price",
            "volume",
            "triggered",
            "impact_compare",
        ]
        df = df_in.copy()
        if "mode" not in df.columns:
            df["mode"] = mode_name
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        df["mode"] = df["mode"].astype(str).str.lower().map(mode_map).fillna(df["mode"])
        df["signal"] = df["signal"].astype(str).str.upper().map(signal_map).fillna(df["signal"])
        df["strategy_type"] = (
            df["strategy_type"].astype(str).str.lower().map(strategy_type_map).fillna(df["strategy_type"])
        )
        df["triggered"] = df["triggered"].apply(
            lambda x: "是" if str(x).lower() in {"true", "1", "yes"} else ("否" if str(x).lower() in {"false", "0", "no"} else x)
        )
        df["impact_compare"] = (
            df["impact_compare"]
            .astype(str)
            .replace(
                {
                    "UNCHANGED": "未變化",
                    "REAL_BASELINE": "真實基準",
                }
            )
        )
        df["impact_compare"] = (
            df["impact_compare"]
            .str.replace("BUY", "買入", regex=False)
            .str.replace("SELL", "賣出", regex=False)
            .str.replace("HOLD", "觀望", regex=False)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce").round(4)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round(0)
        df = df.fillna("")
        return df

    def _make_table(df: pd.DataFrame, table_id: str):
        col_name_map = {
            "strategy_name": "策略名稱",
            "strategy_type": "策略類型",
            "mode": "模式",
            "signal": "訊號",
            "price": "價格",
            "volume": "成交量",
            "triggered": "有觸發",
            "impact_compare": "與真實模式比較",
        }
        return dash_table.DataTable(
            id=table_id,
            columns=[{"name": col_name_map.get(c, c), "id": c} for c in df.columns],
            data=df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#2d2d2d",
                "color": "#e0e0e0",
                "border": "1px solid #444",
                "fontWeight": "bold",
            },
            style_cell={
                "backgroundColor": "#1e1e1e",
                "color": "#e0e0e0",
                "textAlign": "center",
                "border": "1px solid #444",
                "padding": "6px",
            },
            style_data_conditional=[
                {"if": {"filter_query": "{signal} = 買入"}, "color": "#48c774", "fontWeight": "bold"},
                {"if": {"filter_query": "{signal} = 賣出"}, "color": "#ff6b6b", "fontWeight": "bold"},
                {"if": {"filter_query": "{signal} = 觀望"}, "color": "#bfbfbf"},
                {"if": {"filter_query": "{impact_compare} != 未變化"}, "backgroundColor": "#3a2d1a"},
            ],
            page_size=20,
        )

    real_disp = _display_df(real_df, "real")
    exp_disp = _display_df(exp_df, "experiment")

    changed_block = html.Div("無訊號變化。")
    if not changed_df.empty:
        changed_disp = changed_df.copy()
        changed_disp = changed_disp.rename(
            columns={
                "strategy_name": "策略名稱",
                "real_signal": "真實模式訊號",
                "experiment_signal": "實驗模式訊號",
            }
        )
        for c in ["真實模式訊號", "實驗模式訊號"]:
            changed_disp[c] = changed_disp[c].astype(str).str.upper().map(signal_map).fillna(changed_disp[c])
        changed_block = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in changed_disp.columns],
            data=changed_disp.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#2d2d2d", "color": "#e0e0e0"},
            style_cell={"backgroundColor": "#1e1e1e", "color": "#e0e0e0", "textAlign": "center"},
        )

    missing_block = html.Div()
    if missing:
        missing_labels = [f"{item['symbol']} {item['field']}" for item in missing]
        missing_block = html.Div(
            [
                html.Div("實驗模式缺少必要估算欄位：", style={"fontWeight": "bold", "color": "#ff6b6b"}),
                html.Ul([html.Li(x) for x in missing_labels]),
            ],
            style={"marginTop": "8px"},
        )

    if missing:
        status_msg = html.Div(
            [
                html.Div(
                    f"⚠️ 真實模式已完成；實驗模式未執行。計算日期：{context.run_date.strftime('%Y-%m-%d')}",
                    style={"color": "#ffc107", "fontWeight": "bold"},
                ),
                missing_block,
            ]
        )
    else:
        status_msg = html.Div(
            f"✅ 計算完成（{context.run_date.strftime('%Y-%m-%d')}，不落檔）。",
            style={"color": "#28a745", "fontWeight": "bold"},
        )

    result_layout = html.Div(
        [
            html.H5("真實模式"),
            _make_table(real_disp, "daily-real-table"),
            html.Hr(),
            html.H5("實驗模式"),
            _make_table(exp_disp, "daily-exp-table"),
            html.Hr(),
            html.H5("訊號改變清單"),
            changed_block,
        ]
    )
    return status_msg, result_layout


if __name__ == '__main__':
    # 初始化日誌系統（只在實際運行 app 時）
    _initialize_app_logging()

    # 在主線程中執行啟動任務
    safe_startup()

    # 設置更安全的服務器配置
    app.run_server(
        debug=True,
        host='127.0.0.1',
        port=8050,
        threaded=True,
        use_reloader=False  # 避免重載器造成的線程問題
    )
