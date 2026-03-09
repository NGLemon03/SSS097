# -*- coding: utf-8 -*-
'SSSv096 - 股票策略回測系統核心模組'

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import warnings
import logging

# 日誌設定
from analysis.logging_config import get_logger, init_logging
import os

# 啟用日誌檔案輸出
os.environ["SSS_CREATE_LOGS"] = "1"

# 初始化核心日誌記錄器
logger = get_logger("SSS.Core")

def _initialize_core_logging():
    """初始化核心日誌系統。"""
    init_logging(enable_file=True)
    logger.info('核心日誌系統已初始化')
    return logger

# 忽略 Pandas 效能警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# 匯入策略模組與核心工具
from SSS_EnsembleTab import RunConfig, EnsembleParams, CostParams, run_ensemble
from sss_core.normalize import normalize_trades_for_ui, format_trade_df_for_display, normalize_daily_state
from sss_core.plotting import dump_equity_cash, dump_timeseries

__all__ = [
    "load_data", "compute_single", "compute_dual", "compute_RMA",
    "compute_ssma_turn_combined", "backtest_unified",
    "compute_backtest_for_periods", "calculate_metrics",
]
VERSION = "096"

# 標準庫匯入
import os
import json
import random
import hashlib
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
# 數據處理與科學計算
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import textwrap

# 圖表與資料來源
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
try:
    import streamlit as st
except ModuleNotFoundError:
    class _Dummy:
        def __getattr__(self, k): return self
        def __call__(self, *a, **k): pass
    st = _Dummy()


# 型別提示與資料類別
from typing import Dict, List, Tuple, Optional
import itertools
from dataclasses import dataclass, field

def _coerce_trade_schema(df):
    '將交易 DataFrame 欄位名稱統一為標準格式（trade_date, type, price）。'
    import pandas as pd, numpy as np
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["trade_date","type","price"])

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    # 交易日期欄位
    if "trade_date" not in out.columns:
        if "date" in out.columns:
            out["trade_date"] = pd.to_datetime(out["date"], errors="coerce")
        elif isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "trade_date"})
        else:
            out["trade_date"] = pd.NaT
    else:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")

    # 交易類型欄位
    if "type" not in out.columns:
        if "action" in out.columns:
            out["type"] = out["action"].astype(str).str.lower()
        elif "side" in out.columns:
            out["type"] = out["side"].astype(str).str.lower()
        elif "dw" in out.columns:
            out["type"] = np.where(out["dw"]>0, "buy", np.where(out["dw"]<0, "sell", "hold"))
        else:
            out["type"] = "hold"

    # 成交價格欄位
    if "price" not in out.columns:
        for c in ["open","price_open","exec_price","px","close"]:
            if c in out.columns:
                out["price"] = out[c]
                break
        if "price" not in out.columns:
            out["price"] = np.nan

    return out.sort_values("trade_date")

# 載入全域設定與路徑
from analysis import config as cfg
from analysis.logging_config import init_logging
import logging
DATA_DIR = cfg.DATA_DIR
LOG_DIR = cfg.LOG_DIR
CACHE_DIR = cfg.CACHE_DIR
# 全局費率常數
BASE_FEE_RATE = 0.001425 # 券商手續費率
TAX_RATE = 0.003 # 證券交易稅率


# 策略參數預設集
param_presets = {

"Single 2": {"linlen": 90, "factor": 40, "smaalen": 30, "devwin": 30, "buy_mult": 1.45, "sell_mult": 1.25,"stop_loss":0.2, "strategy_type": "single", "smaa_source": "Self"},
"single_1887": {"linlen": 93, "smaalen": 27, "devwin": 30, "factor": 40, "buy_mult": 1.55, "sell_mult": 0.95, "stop_loss": 0.4, "prom_factor": 0.5, "min_dist": 5, "strategy_type": "single", "smaa_source": "Self"},

"Single 3": {"linlen": 80, "factor": 10, "smaalen": 60, "devwin": 20, "buy_mult": 0.4, "sell_mult": 1.5, "strategy_type": "single", "smaa_source": "Self"},
"RMA_69": {"linlen": 151, "smaalen": 162, "rma_len": 55, "dev_len": 40, "factor": 40, "buy_mult": 1.4, "sell_mult": 3.15, "stop_loss": 0.1, "prom_factor": 0.5, "min_dist": 5, "strategy_type": "RMA", "smaa_source": "Factor (^TWII / 2414.TW)"},
"RMA_669": {"linlen": 178, "smaalen": 112, "rma_len": 95, "dev_len": 95, "factor": 40, "buy_mult": 1.7, "sell_mult": 0.9, "stop_loss": 0.4, "prom_factor": 0.5, "min_dist": 5, "strategy_type": "RMA", "smaa_source": "Self"},
"TV_RMAv2": {"linlen": 14, "smaalen": 120, "rma_len": 60, "dev_len": 20, "factor": 10, "buy_mult": 1.0, "sell_mult": 2.0, "stop_loss": 0.4, "prom_factor": 0.5, "min_dist": 5, "strategy_type": "RMA", "smaa_source": "Factor (^TWII / 2412.TW)", "trade_cooldown_bars": 5, "data_provider": "yfinance", "pine_parity_mode": True, "tv_alignment_mode": True},
"STM0": {"linlen": 25, "smaalen": 85, "factor": 80.0, "prom_factor": 9, "min_dist": 8, "buy_shift": 0, "exit_shift": 6, "vol_window": 90, "quantile_win": 65, "signal_cooldown_days": 7, "buy_mult": 0.15, "sell_mult": 0.1, "stop_loss": 0.13, "delta_cap": 0.3,
                "strategy_type": "ssma_turn", "smaa_source": "Factor (^TWII / 2414.TW)"},
"STM1": {"linlen": 15,"smaalen": 40,"factor": 40.0,"prom_factor": 70,"min_dist": 10,"buy_shift": 6,"exit_shift": 4,"vol_window": 40,"quantile_win": 65,
 "signal_cooldown_days": 10,"buy_mult": 1.55,"sell_mult": 2.1,"stop_loss": 0.15,"delta_cap": 0.3,"strategy_type": "ssma_turn","smaa_source": "Self"},
"STM3": {"linlen": 20,"smaalen": 40,"factor": 40.0,"prom_factor": 69,"min_dist": 10,"buy_shift": 6,"exit_shift": 4,"vol_window": 45,"quantile_win": 55,
    "signal_cooldown_days": 10,"buy_mult": 1.65,"sell_mult": 2.1,"stop_loss": 0.2,"delta_cap": 0.3,"strategy_type": "ssma_turn","smaa_source": "Self"},
"STM4": {"linlen": 10,"smaalen": 35,"factor": 40.0,"prom_factor": 68,"min_dist": 8,"buy_shift": 6,"exit_shift": 0,"vol_window": 40,"quantile_win": 65,
    "signal_cooldown_days": 10,"buy_mult": 1.6,"sell_mult": 2.2,"stop_loss": 0.15,"delta_cap": 0.3,"strategy_type": "ssma_turn","smaa_source": "Factor (^TWII / 2414.TW)"},
"STM_1939":{'linlen': 20, 'smaalen': 240, 'factor': 40.0, 'prom_factor': 48, 'min_dist': 14, 'buy_shift': 1, 'exit_shift': 1, 'vol_window': 80, 'quantile_win': 175, 'signal_cooldown_days': 4, 'buy_mult': 1.45, 'sell_mult': 2.6, 'stop_loss': 0.2,"delta_cap": 0.3,
                "strategy_type": "ssma_turn", "smaa_source": "Self"},

"STM_2414_273": {"linlen": 175, "smaalen": 10, "factor": 40.0, "prom_factor": 47, "min_dist": 5, "buy_shift": 0, "exit_shift": 0, "vol_window": 90, "quantile_win": 165, "signal_cooldown_days": 4, "buy_mult": 1.25, "sell_mult": 1.7, "stop_loss": 0.4, "delta_cap": 0.3, "strategy_type": "ssma_turn", "smaa_source": "Factor (^TWII / 2414.TW)"},

# 集成策略預設
"Ensemble_Majority": {
    "strategy_type": "ensemble",
    "method": "majority",
    "params": {
        "floor": 0.2,
        "ema_span": 3,
        "min_cooldown_days": 1,
        "delta_cap": 0.3,
        "min_trade_dw": 0.01,
    },
    "trade_cost": {        # 交易成本設定
        "discount_rate": 0.3,
        "buy_fee_bp": 4.27,
        "sell_fee_bp": 4.27,
        "sell_tax_bp": 30.0,
    },
    "ticker": "00631L.TW",
    "majority_k_pct": 0.55,  # 多數決門檻比例
},

"Ensemble_Proportional": {
    "strategy_type": "ensemble",
    "method": "proportional",
    "params": {
        "floor": 0.2,
        "ema_span": 3,
        "min_cooldown_days": 1,
        "delta_cap": 0.2,   # 單次最大權重變化量
        "min_trade_dw": 0.03
    },
    "trade_cost": {
        "discount_rate": 0.3,
        "buy_fee_bp": 4.27,
        "sell_fee_bp": 4.27,
        "sell_tax_bp": 30.0,
    },
    "ticker": "00631L.TW",
}

}
init_logging(enable_file=True)  # 初始化統一日誌設定
logger = logging.getLogger("SSS.System")  # 系統層級日誌記錄器

# Ensemble 模組遷移警告（僅顯示一次）
_ENSEMBLE_MOVED_WARNED = False
def _warn_ensemble_moved_once():
    global _ENSEMBLE_MOVED_WARNED
    if _ENSEMBLE_MOVED_WARNED:
        return
    logger.warning('Ensemble 功能已遷移至 SSS_EnsembleTab 模組')
    logger.info('請改用 SSS_EnsembleTab.run_ensemble()')
    _ENSEMBLE_MOVED_WARNED = True

from functools import wraps

@dataclass
class TradeSignal:
    ts: pd.Timestamp
    side: str  # "BUY", "SELL", "FORCE_SELL", "STOP_LOSS"
    reason: str

# 參數驗證工具
def validate_params(params: Dict, required_keys: set, positive_ints: set = None, positive_floats: set = None) -> bool:
    '驗證策略參數是否完整且數值合法。'
    if not all(k in params for k in required_keys):
        logger.error(f"缺少必要參數: {required_keys - set(params.keys())}")
        st.error(f"缺少必要參數: {required_keys - set(params.keys())}")
        return False
    if positive_ints:
        for k in positive_ints:
            if k in params and (not isinstance(params[k], int) or params[k] <= 0):
                logger.error(f"參數 {k} 必須為正整數")
                st.error(f"參數 {k} 必須為正整數")
                return False
    if positive_floats:
        for k in positive_floats:
            if k in params and (not isinstance(params[k], (int, float)) or params[k] <= 0):
                logger.error(f"參數 {k} 必須為正數")
                st.error(f"參數 {k} 必須為正數")
                return False
    return True

# Yahoo Finance 資料下載
def fetch_yf_data(
    ticker: str,
    filename: Path,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> None:
    '從 Yahoo Finance 下載股價資料並存為 CSV，支援快取與自動更新。'
    now_taipei = pd.Timestamp.now(tz='Asia/Taipei')
    day_start_taipei = now_taipei.normalize()
    file_exists = filename.exists()
    proceed_with_fetch = True

    try:
        pd.to_datetime(start_date, format='%Y-%m-%d')
        if end_date:
            pd.to_datetime(end_date, format='%Y-%m-%d')
    except ValueError as e:
        logger.error(f"日期格式錯誤: {e}")
        st.error(f"日期格式錯誤: {e}")
        return

    if file_exists:
        file_mod_time_taipei = pd.to_datetime(os.path.getmtime(filename), unit='s', utc=True).tz_convert('Asia/Taipei')
        logger.info(f"檔案 {filename} 最後修改時間: {file_mod_time_taipei.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        same_day = (file_mod_time_taipei.normalize() == day_start_taipei)
        tkr = str(ticker or "").upper()
        is_tw_symbol = (".TW" in tkr) or (tkr in {"^TWII", "TAIEX", "FMTQIK.TAIEX"})
        tw_close_cutoff = day_start_taipei + pd.Timedelta(hours=14, minutes=5)

        if same_day:
            # 同日資料預設視為最新；但台股收盤後若檔案仍是盤中快照，需重新抓取收盤值。
            if is_tw_symbol and now_taipei >= tw_close_cutoff and file_mod_time_taipei < tw_close_cutoff:
                logger.warning(f"{ticker} 今日資料疑似盤中快照（{file_mod_time_taipei.strftime('%H:%M:%S')}），準備更新收盤值")
                st.sidebar.warning(f"{ticker} 今日資料疑似盤中快照，準備更新收盤值")
            else:
                logger.info(f"{ticker} 股價資料已是最新")
                st.sidebar.success(f"{ticker} 股價資料已是最新")
                proceed_with_fetch = False
        else:
            logger.warning(f"{ticker} 股價資料已過期，準備更新")
            st.sidebar.warning(f"{ticker} 股價資料已過期，準備更新")
    else:
        logger.warning(f"檔案 {filename} 不存在，準備下載")
        st.sidebar.warning(f"檔案 {filename} 不存在，準備下載")

    if not proceed_with_fetch:
        return

    try:
        df = yf.download(ticker, period='max', auto_adjust=auto_adjust)
        if df.empty:
            raise ValueError("Fetched dataframe is empty")
        df.to_csv(filename)
        logger.info(f"已下載 {ticker} 股價資料至 {filename}")
        st.sidebar.success(f"已下載 {ticker} 股價資料至 {filename}")
    except Exception as e:
        logger.error(f"下載 {ticker} 失敗: {e}")
        st.sidebar.error(f"下載 {ticker} 失敗: {e}")
        if file_exists:
            logger.info('使用本地快取資料')
            st.sidebar.info('使用本地快取資料')
            return
        try:
            logger.info(f"已強制更新 {ticker} 股價數據")
            df = yf.download(ticker, period='max', auto_adjust=auto_adjust)
            if df.empty:
                raise ValueError("Fallback fetch empty")
            df.to_csv(filename)
            logger.info(f"備用下載 {ticker} 成功: {filename}")
            st.sidebar.success(f"備用下載 {ticker} 成功: {filename}")
        except Exception as e2:
            logger.error(f"備用下載 {ticker} 也失敗: {e2}")
            st.sidebar.error(f"備用下載 {ticker} 也失敗: {e2}")
            if not file_exists:
                raise RuntimeError(f"No local fallback data for {ticker}")

def is_price_data_up_to_date(csv_path):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
        else:
            last_date = pd.to_datetime(df.iloc[-1, 0])
        now_taipei = pd.Timestamp.now(tz='Asia/Taipei')
        today = now_taipei.normalize()
        last_day = pd.Timestamp(last_date).tz_localize(None).normalize()
        if last_day < today.tz_localize(None):
            return False
        if last_day > today.tz_localize(None):
            return True

        # last_day == today：同日資料通常視為最新。
        # 但台股在收盤後若檔案仍是盤中時間，視為過期，避免沿用盤中 close。
        name = Path(str(csv_path)).name.upper()
        is_tw_symbol = (".TW" in name) or ("^TWII" in name) or ("TAIEX" in name)
        if is_tw_symbol:
            close_cutoff = today + pd.Timedelta(hours=14, minutes=5)
            file_mod_time_taipei = pd.to_datetime(os.path.getmtime(csv_path), unit='s', utc=True).tz_convert('Asia/Taipei')
            if now_taipei >= close_cutoff and file_mod_time_taipei < close_cutoff:
                return False
        return True
    except Exception:
        return False

def clear_all_caches():
    """清除所有快取（Joblib / SMAA / Optuna）。"""
    try:
        # 清除 Joblib 記憶體快取
        cfg.MEMORY.clear()
        logger.info('Joblib 快取已清除')

        # 清除 SMAA 指標快取
        smaa_cache_dir = CACHE_DIR / "cache_smaa"
        if smaa_cache_dir.exists():
            import shutil
            shutil.rmtree(smaa_cache_dir)
            smaa_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info('SMAA 快取已清除')

        # 清除 Optuna 最佳化快取
        optuna_cache_dir = CACHE_DIR / "optuna16_equity"
        if optuna_cache_dir.exists():
            import shutil
            shutil.rmtree(optuna_cache_dir)
            optuna_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info('Optuna 快取已清除')

        return True
    except Exception as e:
        logger.error(f"清除快取失敗: {e}")
        return False

def force_update_price_data(ticker: str = None):
    '強制重新下載指定（或全部常用）股票的股價資料。'
    try:
        if ticker:
            # 更新指定股票
            filename = DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
            fetch_yf_data(ticker, filename, "2000-01-01")
            logger.info(f"已強制更新 {ticker} 股價數據")
        else:
            # 更新所有常用標的
            common_tickers = ["00631L.TW", "^TWII", "2414.TW", "2412.TW"]
            for t in common_tickers:
                filename = DATA_DIR / f"{t.replace(':','_')}_data_raw.csv"
                fetch_yf_data(t, filename, "2000-01-01")
            logger.info('所有常用標的股價已更新')
        return True
    except Exception as e:
        logger.error(f"強制更新股價失敗: {e}")
        return False

def _is_tradingview_provider(provider: str) -> bool:
    return str(provider or "").strip().lower() in {
        "tradingview",
        "tv",
        "pine_parity",
        "pine_parity_tv",
    }


def _is_twse_db_provider(provider: str) -> bool:
    return str(provider or "").strip().lower() in {
        "twse_db",
        "twse",
        "fmtqik",
        "fmtqik_db",
    }


def _resolve_twse_db_path() -> Path:
    candidates = [
        DATA_DIR / "twse_data.db",
        Path("twse_data.db"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


_TWSE_DB_LAST_AUTO_ATTEMPT: Dict[str, pd.Timestamp] = {}


def _fmtqik_max_date(db_path: Path) -> Optional[pd.Timestamp]:
    path = Path(db_path)
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute("SELECT MAX(date) FROM fmtqik").fetchone()
    except Exception as e:
        logger.warning(f"[twse_db] 讀取 fmtqik 最大日期失敗: {e}")
        return None
    if not row or not row[0]:
        return None
    ts = pd.to_datetime(row[0], errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _prev_weekday(day: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(day).normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _expected_latest_fmtqik_date(now_taipei: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    now_taipei = pd.Timestamp(now_taipei) if now_taipei is not None else pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
    d = _prev_weekday(now_taipei.normalize())
    close_cutoff = d + pd.Timedelta(hours=15)
    if now_taipei < close_cutoff:
        d = _prev_weekday(d - pd.Timedelta(days=1))
    return d


def _try_auto_update_twse_db(force_update: bool = False) -> bool:
    """
    依照 fmtqik 最新日期自動更新 twse_data.db。
    - 非 force 模式：若資料落後於預期交易日才更新，且每天最多嘗試一次。
    - force 模式：忽略每日節流，直接嘗試更新。
    """
    db_path = _resolve_twse_db_path()
    now_taipei = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
    today_key = now_taipei.normalize()
    key = str(db_path.resolve())

    if not force_update:
        last_attempt = _TWSE_DB_LAST_AUTO_ATTEMPT.get(key)
        if last_attempt is not None and last_attempt >= today_key:
            return False

    current_max = _fmtqik_max_date(db_path)
    expected_latest = _expected_latest_fmtqik_date(now_taipei)
    stale = force_update or (current_max is None) or (current_max < expected_latest)
    if not stale:
        return False

    _TWSE_DB_LAST_AUTO_ATTEMPT[key] = today_key

    try:
        from TWSECrawler_SQ import TWSECrawler
    except Exception as e:
        logger.warning(f"[twse_db] 無法載入 TWSECrawler_SQ，略過自動更新: {e}")
        return False

    try:
        logger.info(
            "[twse_db] 自動更新啟動: db=%s, current_max=%s, expected=%s, force=%s",
            db_path,
            current_max.strftime("%Y-%m-%d") if current_max is not None else "None",
            expected_latest.strftime("%Y-%m-%d"),
            force_update,
        )
        # 保守延遲，避免太激進；通常只需補抓當月一次請求。
        crawler = TWSECrawler(
            db_path=db_path,
            min_delay=1.0,
            max_delay=2.0,
            request_timeout=10,
            max_retries=2,
            retry_sleep_base=2,
        )
        crawler.fetch_fmtqik(start_year=2000)
        new_max = _fmtqik_max_date(db_path)
        logger.info(
            "[twse_db] 自動更新完成: old=%s, new=%s",
            current_max.strftime("%Y-%m-%d") if current_max is not None else "None",
            new_max.strftime("%Y-%m-%d") if new_max is not None else "None",
        )
        return bool(new_max is not None and (current_max is None or new_max > current_max))
    except Exception as e:
        logger.warning(f"[twse_db] 自動更新失敗: {e}")
        return False


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    out.index = pd.to_datetime(out.index, format="%Y-%m-%d", errors="coerce")
    out = out[~out.index.isna()].sort_index()

    if "close" not in out.columns:
        return pd.DataFrame()

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    for col in ["open", "high", "low"]:
        if col not in out.columns:
            out[col] = out["close"]
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["close"])
    return out[["open", "high", "low", "close", "volume"]]


def _load_fmtqik_taiex_df(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    db_path = _resolve_twse_db_path()
    if not db_path.exists():
        logger.warning(f"twse_data.db not found: {db_path}")
        return pd.DataFrame()

    query = "SELECT date, taiex FROM fmtqik WHERE date >= ?"
    params: List[str] = [start_date]
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date ASC"

    try:
        with sqlite3.connect(db_path) as conn:
            raw = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        logger.warning(f"從 {db_path} 讀取 fmtqik.taiex 失敗: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["taiex"] = pd.to_numeric(raw["taiex"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    raw = raw.dropna(subset=["date", "taiex"]).set_index("date").sort_index()
    if raw.empty:
        return pd.DataFrame()

    close = raw["taiex"].rename("close")
    out = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0.0,
        },
        index=raw.index,
    )
    if end_date:
        out = out[out.index <= pd.to_datetime(end_date)]
    return out


def _load_tv_symbol_data(
    ticker: str,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    ticker_norm = str(ticker or "").strip().lower()
    if ticker_norm in {"fmtqik.taiex", "taiex"}:
        return _load_fmtqik_taiex_df(start_date=start_date, end_date=end_date)

    filename = DATA_DIR / f"{ticker.replace(':', '_')}_data_raw.csv"
    if ticker_norm == "^twii":
        filename = DATA_DIR / "^TWII_data_raw.csv"
        if not filename.exists():
            return _load_fmtqik_taiex_df(start_date=start_date, end_date=end_date)

    if not filename.exists():
        return pd.DataFrame()

    try:
        raw = pd.read_csv(filename, parse_dates=[0], index_col=0, date_format="%Y-%m-%d")
    except Exception:
        return pd.DataFrame()

    out = _normalize_ohlcv_df(raw)
    if out.empty:
        return out
    out = out[out.index >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out.index <= pd.to_datetime(end_date)]
    return out


def _load_factor_for_smaa_source_tv(
    price_index: pd.Index,
    start_date: str,
    end_date: Optional[str],
    smaa_source: str,
) -> pd.DataFrame:
    if smaa_source not in {"Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"}:
        return pd.DataFrame()

    factor_ticker = "2412.TW" if smaa_source == "Factor (^TWII / 2412.TW)" else "2414.TW"
    df_twii = _load_tv_symbol_data("^TWII", start_date=start_date, end_date=end_date)
    df_factor_ticker = _load_tv_symbol_data(factor_ticker, start_date=start_date, end_date=end_date)
    if df_twii.empty or df_factor_ticker.empty:
        return pd.DataFrame()

    common_index = df_twii.index.intersection(df_factor_ticker.index).intersection(price_index)
    if len(common_index) < 100:
        return pd.DataFrame()

    factor_price = (df_twii.loc[common_index, "close"] / df_factor_ticker.loc[common_index, "close"]).rename("close")
    factor_volume = df_factor_ticker.loc[common_index, "volume"].rename("volume")
    return pd.DataFrame({"close": factor_price, "volume": factor_volume}).reindex(price_index).dropna()


@cfg.MEMORY.cache
def _load_data_cached(
    ticker: str,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    smaa_source: str = "Self",
    force_update: bool = False,
    data_provider: str = "yfinance",
    pine_parity_mode: bool = False,
    cache_signature: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '載入股價與因子資料（帶快取），回傳 (df_price, df_factor)。'
    provider = str(data_provider or "yfinance").strip().lower()
    if _is_tradingview_provider(provider):
        df = _load_tv_symbol_data(ticker, start_date=start_date, end_date=end_date)
        if df.empty:
            st.error(f"TradingView  {ticker} 資料載入失敗。")
            return pd.DataFrame(), pd.DataFrame()
        df.name = ticker.replace(':', '_')
        df_factor = _load_factor_for_smaa_source_tv(
            price_index=df.index,
            start_date=start_date,
            end_date=end_date,
            smaa_source=smaa_source,
        )
        df_factor.name = f"{ticker}_factor" if not df_factor.empty else None
        return df, df_factor

    use_twse_db = _is_twse_db_provider(provider)
    ticker_norm = str(ticker or "").strip().lower()
    if use_twse_db and ticker_norm in {"fmtqik.taiex", "taiex", "^twii"}:
        df = _load_fmtqik_taiex_df(start_date=start_date, end_date=end_date)
        if df.empty:
            st.error("fmtqik.taiex 資料載入失敗，且未找到 TWSE DB 中的對應資料。")
            return pd.DataFrame(), pd.DataFrame()
        df.name = ticker.replace(':', '_')
        return df, pd.DataFrame()

    use_unadjusted = bool(pine_parity_mode)
    file_suffix = "_data_raw_unadj.csv" if use_unadjusted else "_data_raw.csv"
    filename = DATA_DIR / f"{ticker.replace(':','_')}{file_suffix}"

    # 判斷是否需要下載
    if force_update:
        fetch_yf_data(ticker, filename, start_date, end_date, auto_adjust=not use_unadjusted)
    else:
        # 資料過期時自動更新
        if not is_price_data_up_to_date(filename):
            fetch_yf_data(ticker, filename, start_date, end_date, auto_adjust=not use_unadjusted)

    if not filename.exists():
        logger.error(f"資料檔案不存在: {filename}")
        st.error(f"資料檔案不存在: {filename}")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_csv(filename, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
        df.name = ticker.replace(':', '_')
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
        df = df[~df.index.isna()]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                logger.warning(f"資料缺少 {col} 欄位，以 NaN 填充")
                st.warning(f"資料缺少 {col} 欄位，以 NaN 填充")
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['close'])
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        df_factor = pd.DataFrame()  # 預設空因子數據
        if smaa_source in ["Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]:
            factor_ticker = "2412.TW" if smaa_source == "Factor (^TWII / 2412.TW)" else "2414.TW"
            factor_file = DATA_DIR / f"{factor_ticker.replace(':','_')}{file_suffix}"
            # 下載加權指數與因子股資料
            if not use_twse_db:
                twii_file = DATA_DIR / "^TWII_data_raw.csv"
                if force_update or not is_price_data_up_to_date(twii_file):
                    fetch_yf_data("^TWII", twii_file, start_date, end_date)
            if force_update or not is_price_data_up_to_date(factor_file):
                fetch_yf_data(factor_ticker, factor_file, start_date, end_date, auto_adjust=not use_unadjusted)

            if not factor_file.exists() or (not use_twse_db and not twii_file.exists()):
                logger.warning(f"因子股 {factor_ticker} 資料檔案不存在")
                st.warning(f"因子股 {factor_ticker} 資料檔案不存在")
                return df, pd.DataFrame()

            try:
                if use_twse_db:
                    df_twii = _load_fmtqik_taiex_df(start_date=start_date, end_date=end_date)
                else:
                    df_twii = pd.read_csv(twii_file, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
                df_factor_ticker = pd.read_csv(factor_file, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
                df_twii.columns = [c.lower().replace(' ', '_') for c in df_twii.columns]
                df_factor_ticker.columns = [c.lower().replace(' ', '_') for c in df_factor_ticker.columns]
                df_twii.index = pd.to_datetime(df_twii.index, format='%Y-%m-%d', errors='coerce')
                df_factor_ticker.index = pd.to_datetime(df_factor_ticker.index, format='%Y-%m-%d', errors='coerce')
                df_twii = df_twii[~df_twii.index.isna()]
                df_factor_ticker = df_factor_ticker[~df_factor_ticker.index.isna()]
                for col in ['close', 'volume']:
                    df_twii[col] = pd.to_numeric(df_twii[col], errors='coerce')
                    df_factor_ticker[col] = pd.to_numeric(df_factor_ticker[col], errors='coerce')

                common_index = df_twii.index.intersection(df_factor_ticker.index).intersection(df.index)
                if len(common_index) < 100:
                    logger.warning(f"共同交易日不足: {len(common_index)} 天（需要至少 100 天）")
                    st.warning(f"共同交易日不足: {len(common_index)} 天（需要至少 100 天）")
                    return df, pd.DataFrame()
                factor_price = (df_twii['close'].loc[common_index] / df_factor_ticker['close'].loc[common_index]).rename('close')
                factor_volume = df_factor_ticker['volume'].loc[common_index].rename('volume')
                df_factor = pd.DataFrame({'close': factor_price, 'volume': factor_volume})
                df_factor = df_factor.reindex(df.index).dropna()
                if end_date:
                    df_factor = df_factor[df_factor.index <= pd.to_datetime(end_date)]
            except Exception as e:
                logger.warning(f"因子資料處理失敗: {e}")
                st.warning(f"因子資料處理失敗: {e}")
                return df, pd.DataFrame()

        df_factor.name = f"{ticker}_factor" if not df_factor.empty else None
        return df, df_factor
    except Exception as e:
        logger.error(f"讀取 {filename} 失敗: {e}")
        st.error(f"讀取 {filename} 失敗: {e}")
        return pd.DataFrame(), pd.DataFrame()


def _file_signature(path: Path) -> str:
    p = Path(path)
    if not p.exists():
        return f"{p.name}:missing"
    stat = p.stat()
    return f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _build_load_data_cache_signature(
    ticker: str,
    smaa_source: str,
    data_provider: str,
    pine_parity_mode: bool = False,
) -> str:
    provider = str(data_provider or "yfinance").strip().lower()
    files = _required_price_files_for_load_data(
        ticker=ticker,
        smaa_source=smaa_source,
        data_provider=provider,
        pine_parity_mode=pine_parity_mode,
    )

    payload = f"{provider}|{ticker}|{smaa_source}|pine={int(bool(pine_parity_mode))}|{'|'.join(_file_signature(p) for p in files)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _required_price_files_for_load_data(
    ticker: str,
    smaa_source: str,
    data_provider: str,
    pine_parity_mode: bool = False,
) -> List[Path]:
    provider = str(data_provider or "yfinance").strip().lower()
    ticker_norm = str(ticker or "").strip().lower()
    file_suffix = "_data_raw_unadj.csv" if pine_parity_mode else "_data_raw.csv"
    files: List[Path] = []

    if _is_twse_db_provider(provider) and ticker_norm in {"fmtqik.taiex", "taiex", "^twii"}:
        files.append(_resolve_twse_db_path())
    else:
        files.append(DATA_DIR / f"{ticker.replace(':', '_')}{file_suffix}")

    if smaa_source in {"Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"}:
        factor_ticker = "2412.TW" if smaa_source == "Factor (^TWII / 2412.TW)" else "2414.TW"
        files.append(DATA_DIR / f"{factor_ticker.replace(':', '_')}{file_suffix}")
        if _is_twse_db_provider(provider):
            files.append(_resolve_twse_db_path())
        else:
            files.append(DATA_DIR / "^TWII_data_raw.csv")
    return files


def load_data(
    ticker: str,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    smaa_source: str = "Self",
    force_update: bool = False,
    data_provider: str = "yfinance",
    pine_parity_mode: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    provider_norm = str(data_provider or "yfinance").strip().lower()
    ticker_norm = str(ticker or "").strip().lower()
    uses_twse_fmtqik = _is_twse_db_provider(provider_norm) and (
        ticker_norm in {"fmtqik.taiex", "taiex", "^twii"}
        or smaa_source in {"Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"}
    )
    if uses_twse_fmtqik:
        _try_auto_update_twse_db(force_update=bool(force_update))

    cache_signature = _build_load_data_cache_signature(
        ticker=ticker,
        smaa_source=smaa_source,
        data_provider=provider_norm,
        pine_parity_mode=pine_parity_mode,
    )
    needs_refresh = bool(force_update)
    if not needs_refresh:
        for p in _required_price_files_for_load_data(
            ticker=ticker,
            smaa_source=smaa_source,
            data_provider=provider_norm,
            pine_parity_mode=pine_parity_mode,
        ):
            if p.suffix.lower() != ".csv":
                continue
            if not is_price_data_up_to_date(str(p)):
                needs_refresh = True
                break

    if needs_refresh:
        cache_signature = f"{cache_signature}:{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    return _load_data_cached(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        smaa_source=smaa_source,
        force_update=needs_refresh,
        data_provider=provider_norm,
        pine_parity_mode=pine_parity_mode,
        cache_signature=cache_signature,
    )


def load_data_wrapper(ticker: str, start_date: str = "2000-01-01",
                      end_date: str | None = None,
                      smaa_source: str = "Self",
                      data_provider: str = "yfinance",
                      pine_parity_mode: bool = False):
    'load_data 的簡化包裝，省略 force_update 參數。'
    df_price, df_factor = load_data(
        ticker,
        start_date,
        end_date,
        smaa_source,
        data_provider=data_provider,
        pine_parity_mode=pine_parity_mode,
    )
    return df_price, df_factor



# ═══ 線性回歸與技術指標計算 ═══
def linreg_last_original(series: pd.Series, length: int) -> pd.Series:
    '滾動線性回歸取末端值（原始實作，逐窗口 polyfit）。'
    if len(series) < length or series.isnull().sum() > len(series) - length:
        return pd.Series(np.nan, index=series.index)
    return series.rolling(length, min_periods=length).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x)-1) + np.polyfit(np.arange(len(x)), x, 1)[1]
        if len(x[~np.isnan(x)]) == length else np.nan, raw=True)

def linreg_last_vectorized(series: np.ndarray, length: int) -> np.ndarray:
    '滾動線性回歸取末端值（向量化版本，使用 sliding_window_view 加速）。'
    # 轉為浮點陣列
    series = np.asarray(series, dtype=float)
    if len(series) < length:
        return np.full(len(series), np.nan, dtype=float)

    # 建立滑動窗口視圖
    windows = np.lib.stride_tricks.sliding_window_view(series, length)
    valid = ~np.any(np.isnan(windows), axis=1)  # 排除含 NaN 的窗口

    # 設計矩陣 [x, 1]
    X = np.vstack([np.arange(length), np.ones(length)]).T  # shape: (length, 2)

    # 預計算 (X'X)^{-1} X'
    XtX_inv_Xt = np.linalg.inv(X.T @ X) @ X.T  # shape: (2, length)

    # 批次最小二乘法求係數
    coeffs = np.einsum('ij,kj->ki', XtX_inv_Xt, windows[valid])  # shape: (n_valid, 2)

    # 將回歸末端值填入結果
    result = np.full(len(windows), np.nan, dtype=float)
    result[valid] = coeffs[:, 0] * (length - 1) + coeffs[:, 1]

    # 對齊至原始序列長度
    output = np.full(len(series), np.nan, dtype=float)
    output[length-1:] = result
    return output

@cfg.MEMORY.cache
def calc_smaa(series: pd.Series, linlen: int, factor: float, smaalen: int) -> pd.Series:
    '計算 SMAA 指標：線性回歸去趨勢 → 放大 → 簡單移動平均。'
    # 取出原始數值
    series_values = series.values
    result = np.full(len(series), np.nan, dtype=float)

    # 檢查最低資料長度
    min_required = max(linlen, smaalen)
    if len(series) < min_required:
        logger.warning(f"資料長度不足: {len(series)}, 需要={min_required}")
        return pd.Series(result, index=series.index)

    # 計算線性回歸值
    lr = linreg_last_vectorized(series_values, linlen)

    # 去趨勢化並乘以放大因子
    detr = (series_values - lr) * factor

    # 計算簡單移動平均
    if len(detr) >= smaalen:
        sma = np.convolve(detr, np.ones(smaalen)/smaalen, mode='valid')
        result[smaalen-1:] = sma  # 對齊至正確的索引位置
    else:
        logger.warning(f"去趨勢序列長度不足: {len(detr)}, 需要={smaalen}")

    return pd.Series(result, index=series.index)

@cfg.MEMORY.cache
def compute_single(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, devwin: int, smaa_source: str = "Self") -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 避免浮點精度問題

    # 計算 SMAA 指標
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    # 計算基準線與標準差（EWM）
    base = smaa.ewm(alpha=1/devwin, adjust=False, min_periods=devwin).mean()
    sd = (smaa - base).abs().ewm(alpha=1/devwin, adjust=False, min_periods=devwin).mean()

    results_df = pd.DataFrame({
        'smaa': smaa,
        'base': base,
        'sd': sd
    }, index=df_cleaned.index)
    final_df = pd.concat([df[['open', 'high', 'low', 'close']], results_df], axis=1, join='inner')
    final_df = final_df.dropna()  # 移除不完整列
    if final_df.empty:
        logger.warning(f"compute_single 結果為空: linlen={linlen}, smaalen={smaalen}, data_len={len(df_cleaned)}, valid_smaa={len(smaa.dropna())}")
    return final_df

@cfg.MEMORY.cache
def compute_dual(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, short_win: int, long_win: int, smaa_source: str = "Self") -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 避免浮點精度問題

    # 計算 SMAA 指標
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    # 計算短期與長期基準線及標準差
    base_s = smaa.ewm(alpha=1/short_win, adjust=False, min_periods=short_win).mean()
    sd_s = (smaa - base_s).abs().ewm(alpha=1/short_win, adjust=False, min_periods=short_win).mean()
    base_l = smaa.ewm(alpha=1/long_win, adjust=False, min_periods=long_win).mean()
    sd_l = (smaa - base_l).abs().ewm(alpha=1/long_win, adjust=False, min_periods=long_win).mean()

    results_df = pd.DataFrame({
        'smaa': smaa,
        'base': base_s,
        'sd': sd_s,
        'base_long': base_l,
        'sd_long': sd_l
    }, index=df_cleaned.index)
    final_df = pd.concat([df[['open', 'high', 'low', 'close']], results_df], axis=1, join='inner')
    final_df = final_df.dropna()  # 移除不完整列
    if final_df.empty:
        logger.warning(f"compute_dual 結果為空: linlen={linlen}, smaalen={smaalen}, data_len={len(df_cleaned)}, valid_smaa={len(smaa.dropna())}")
    return final_df

@cfg.MEMORY.cache
def compute_RMA(
    df: pd.DataFrame,
    smaa_source_df: pd.DataFrame,
    linlen: int,
    factor: float,
    smaalen: int,
    rma_len: int,
    dev_len: int,
    smaa_source: str = "Self",
    pine_parity_mode: bool = False,
) -> pd.DataFrame:
    '計算 RMA 策略指標：SMAA + RMA 基準線 + 滾動標準差。'
    # 選擇資料來源
    source_df   = smaa_source_df if not smaa_source_df.empty else df
    if pine_parity_mode and not smaa_source_df.empty and "close" in source_df.columns:
        source_df = source_df.copy()
        # Pine 對齊模式：使用前一根K棒的收盤價，避免前瞻偏差
        source_df["close"] = source_df["close"].shift(1)
    df_cleaned  = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 避免浮點精度問題

    # 計算 SMAA 指標
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    # 計算 RMA 基準線與滾動標準差
    base = smaa.ewm(alpha=1/rma_len, adjust=False, min_periods=rma_len).mean()
    sd   = smaa.rolling(window=dev_len, min_periods=dev_len).std(ddof=0 if pine_parity_mode else 1)

    # 組合結果
    results = pd.DataFrame({
        'smaa': smaa,
        'base': base,
        'sd':   sd
    }, index=df_cleaned.index)
    final = pd.concat([df[['open','high','low','close']], results], axis=1, join='inner')
    final = final.dropna()  # 移除不完整列
    if final.empty:
        logger.warning(f"compute_RMA 結果為空: linlen={linlen}, smaalen={smaalen}, data_len={len(df_cleaned)}, valid_smaa={len(smaa.dropna())}")
    return final

@cfg.MEMORY.cache
def compute_ssma_turn_combined(
    df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float,
    smaalen: int, prom_factor: float, min_dist: int, buy_shift: int = 0, exit_shift: int = 0, vol_window: int = 20,
    signal_cooldown_days: int = 10, quantile_win: int = 100,
    smaa_source: str = "Self",
    signal_filter_mode: str = "volume_ma",
    volume_target_pass_rate: Optional[float] = None,
    volume_target_lookback: int = 120,
) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    logger.info(
        "compute_ssma_turn_combined: linlen=%d, factor=%.1f, smaalen=%d, prom_factor=%.1f, min_dist=%d, "
        "buy_shift=%d, exit_shift=%d, vol_window=%d, quantile_win=%d, cooldown=%d, filter=%s",
        linlen,
        factor,
        smaalen,
        prom_factor,
        min_dist,
        buy_shift,
        exit_shift,
        vol_window,
        quantile_win,
        signal_cooldown_days,
        str(signal_filter_mode),
    )

    # 參數型別轉換與驗證
    try:
        linlen = int(linlen)
        smaalen = int(smaalen)
        min_dist = int(min_dist)
        vol_window = int(vol_window)
        quantile_win = max(int(quantile_win), vol_window)
        signal_cooldown_days = int(signal_cooldown_days)
        buy_shift = int(buy_shift)
        exit_shift = int(exit_shift)
        volume_target_lookback = int(volume_target_lookback)
        if volume_target_pass_rate is not None:
            volume_target_pass_rate = float(volume_target_pass_rate)
        if min_dist < 1 or vol_window < 1 or quantile_win < 1 or signal_cooldown_days < 0:
            raise ValueError("Invalid ssma_turn parameter range")
        if volume_target_lookback < 1:
            raise ValueError("Invalid volume_target_lookback")
        if volume_target_pass_rate is not None and not (0.0 < volume_target_pass_rate <= 1.0):
            raise ValueError("volume_target_pass_rate must be in (0, 1]")
    except (ValueError, TypeError) as e:
        logger.error(f"ssma_turn 參數驗證失敗: {e}")
        st.error(f"ssma_turn 參數驗證失敗: {e}")
        return pd.DataFrame(), [], []

    filter_mode_raw = str(signal_filter_mode or "volume_ma").strip().lower()
    if filter_mode_raw in {"none", "off", "disable", "disabled", "no_volume"}:
        signal_filter_mode = "none"
    elif filter_mode_raw in {"volume_target", "adaptive_volume", "volume_quantile"}:
        signal_filter_mode = "volume_target"
    else:
        signal_filter_mode = "volume_ma"
    needs_volume_gate = signal_filter_mode in {"volume_ma", "volume_target"}

    # 選擇資料來源並檢查必要欄位
    source_df = smaa_source_df if not smaa_source_df.empty else df
    if 'close' not in source_df.columns:
        logger.error('資料來源缺少 close 欄位')
        st.error('資料來源缺少 close 欄位')
        return pd.DataFrame(), [], []
    if needs_volume_gate and 'volume' not in df.columns:
        logger.error('主標的資料缺少 volume 欄位，無法進行量能濾網')
        st.error('主標的資料缺少 volume 欄位，無法進行量能濾網')
        return pd.DataFrame(), [], []

    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)  # 避免浮點精度問題
    if df_cleaned.empty:
        logger.warning(f"清洗後資料為空，原始筆數: {len(df_cleaned)}")
        st.warning(f"清洗後資料為空，原始筆數: {len(df_cleaned)}")
        return pd.DataFrame(), [], []

    # 計算 SMAA 指標
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    series_clean = smaa.dropna()
    if series_clean.empty:
        logger.warning(f"SMAA 計算結果全為 NaN，有效筆數: {len(series_clean)}, linlen={linlen}, smaalen={smaalen}")
        st.warning(f"SMAA 計算結果全為 NaN，有效筆數: {len(series_clean)}, linlen={linlen}, smaalen={smaalen}")
        return pd.DataFrame(), [], []



    # 計算波動幅度閾值（prominence 門檻）
    prom = series_clean.rolling(window=min_dist+1, min_periods=min_dist+1).apply(lambda x: x.ptp(), raw=True)
    initial_threshold = prom.quantile(prom_factor / 100) if len(prom.dropna()) > 0 else prom.median()
    threshold_series = prom.rolling(window=quantile_win, min_periods=quantile_win).quantile(prom_factor / 100).shift(1).ffill().fillna(initial_threshold)

    # 偵測波峰與波谷
    peaks = []
    valleys = []
    last_signal_idx = -1          # 上一個訊號的索引位置
    last_signal_dt  = None        # 上一個訊號的日期

    for i in range(quantile_win, len(series_clean)):
        if (last_signal_dt is not None and
            (series_clean.index[i] - last_signal_dt).days <= signal_cooldown_days):
            continue
        window_data = series_clean.iloc[max(0, i - quantile_win):i + 1].to_numpy()
        if len(window_data) < min_dist + 1:
            continue
        current_threshold = threshold_series.iloc[i]
        window_peaks, _ = find_peaks(window_data, distance=min_dist, prominence=current_threshold)
        window_valleys, _ = find_peaks(-window_data, distance=min_dist, prominence=current_threshold)
        window_start_idx = max(0, i - quantile_win)
        if window_peaks.size > 0:
            for p_idx in window_peaks:
                peak_date = series_clean.index[window_start_idx + p_idx]
                if peak_date not in peaks:  # 去重
                    peaks.append(peak_date)
        if window_valleys.size > 0:
            for v_idx in window_valleys:
                valley_date = series_clean.index[window_start_idx + v_idx]
                if valley_date not in valleys:  # 去重
                    valleys.append(valley_date)

    # 訊號濾網：volume_ma（舊邏輯）/ volume_target（目標通過率）/ none（不做量能過濾）
    valid_peaks: List[pd.Timestamp] = []
    valid_valleys: List[pd.Timestamp] = []
    if signal_filter_mode == "none":
        valid_peaks = list(peaks)
        valid_valleys = list(valleys)
    else:
        vol_series = pd.to_numeric(df['volume'], errors='coerce')
        vol_ma = vol_series.rolling(vol_window, min_periods=vol_window).mean().shift(1)
        vol_ratio = vol_series / vol_ma.replace(0, np.nan)

        target_threshold_series = pd.Series(np.nan, index=df.index, dtype=float)
        fallback_threshold = np.nan
        if signal_filter_mode == "volume_target":
            target_pass = 0.35 if volume_target_pass_rate is None else float(volume_target_pass_rate)
            target_pass = min(max(target_pass, 0.05), 1.0)
            q = max(0.0, min(1.0, 1.0 - target_pass))
            lookback = max(int(volume_target_lookback), vol_window + 5)
            min_obs = max(10, min(lookback, vol_window + 10))
            target_threshold_series = vol_ratio.rolling(
                window=lookback,
                min_periods=min_obs,
            ).quantile(q).shift(1)
            ratio_valid = vol_ratio.dropna()
            if not ratio_valid.empty:
                fallback_threshold = float(ratio_valid.quantile(q))
            if pd.isna(fallback_threshold):
                fallback_threshold = 1.0

        def _passes_filter(ts: pd.Timestamp) -> bool:
            if ts not in df.index or ts not in vol_ma.index:
                return False
            v = vol_series.loc[ts]
            vol_avg = vol_ma.loc[ts]
            if pd.isna(v) or pd.isna(vol_avg) or vol_avg <= 0:
                return False
            if signal_filter_mode == "volume_ma":
                return bool(v > vol_avg)

            ratio = vol_ratio.loc[ts]
            if pd.isna(ratio):
                return False
            threshold = target_threshold_series.loc[ts] if ts in target_threshold_series.index else np.nan
            if pd.isna(threshold):
                threshold = fallback_threshold
            return bool(pd.notna(threshold) and ratio > threshold)

        for p in peaks:
            if _passes_filter(p):
                valid_peaks.append(p)
        for v in valleys:
            if _passes_filter(v):
                valid_valleys.append(v)

    # 套用訊號冷卻期
    def apply_cooldown(dates, cooldown_days):
        filtered = []
        last_date = pd.Timestamp('1900-01-01')
        for d in sorted(dates):
            if (d - last_date).days >= cooldown_days:
                filtered.append(d)
                last_date = d
        return filtered
    valid_peaks = apply_cooldown(valid_peaks, signal_cooldown_days)
    valid_valleys = apply_cooldown(valid_valleys, signal_cooldown_days)

    # 根據位移參數產生實際買賣日期
    buy_dates = []
    sell_dates = []
    for dt in valid_valleys:
        try:
            tgt_idx = df.index.get_loc(dt) + 1 + buy_shift
            if 0 <= tgt_idx < len(df):
                buy_dates.append(df.index[tgt_idx])
        except KeyError:
            continue
    for dt in valid_peaks:
        try:
            tgt_idx = df.index.get_loc(dt) + 1 + exit_shift
            if 0 <= tgt_idx < len(df):
                sell_dates.append(df.index[tgt_idx])
        except KeyError:
            continue

    df_ind = df[['open', 'close']].copy()
    df_ind['smaa'] = smaa.reindex(df.index)
    if df_ind.dropna().empty:
        logger.warning(f"合併後指標資料為空，linlen={linlen}, smaalen={smaalen}, valid_smaa={len(smaa.dropna())}")
    return df_ind.dropna(), buy_dates, sell_dates

def calculate_trade_mmds(trades: List[Tuple[pd.Timestamp, float, pd.Timestamp]], equity_curve: pd.Series) -> List[float]:
    '計算每筆交易期間的最大回撤（MDD）。'
    mmds = []
    for entry_date, _, exit_date in trades:
        # 擷取交易期間的權益曲線
        period_equity = equity_curve.loc[entry_date:exit_date]
        if len(period_equity) < 2:
            mmds.append(0.0)
            continue
        roll_max = period_equity.cummax()
        drawdown = period_equity / roll_max - 1
        mmds.append(drawdown.min())
    return mmds

def calculate_metrics(trades: List[Tuple[pd.Timestamp, float, pd.Timestamp]], df_ind: pd.DataFrame, equity_curve: pd.Series = None) -> Dict:
    '根據交易記錄計算回測績效指標（報酬率、回撤、夏普等）。'
    metrics = {
        'total_return': 0.0,
        'annual_return': 0.0,
        'max_drawdown': 0.0,  # 最大回撤
        'max_drawdown_duration': 0,
        'calmar_ratio': np.nan,
        'num_trades': 0,
        'win_rate': 0.0,
        'avg_win': np.nan,
        'avg_loss': np.nan,
        'payoff_ratio': np.nan,
        'sharpe_ratio': np.nan,
        'sortino_ratio': np.nan,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'avg_holding_period': np.nan,
        'annualized_volatility': np.nan,
        'profit_factor': np.nan,
    }
    if not trades:
        return metrics

    trade_metrics = pd.DataFrame(trades, columns=['entry_date', 'return', 'exit_date']).set_index('exit_date')
    trade_metrics['equity'] = (1 + trade_metrics['return']).cumprod()
    roll_max = trade_metrics['equity'].cummax()
    daily_drawdown = trade_metrics['equity'] / roll_max - 1

    # 總報酬與年化報酬
    metrics['total_return'] = trade_metrics['equity'].iloc[-1] - 1
    years = max((trade_metrics.index[-1] - trade_metrics.index[0]).days / 365.25, 1)
    metrics['annual_return'] = (1 + metrics['total_return']) ** (1 / years) - 1

    # 計算最大回撤持續期間
    dd_np = (daily_drawdown < 0).astype(bool)
    dd_series = pd.Series(dd_np, index=trade_metrics.index)
    blocks = (~dd_series).cumsum()
    dd_dur = int((dd_series.groupby(blocks).cumcount() + 1).where(dd_series).max() or 0)
    metrics['max_drawdown_duration'] = dd_dur

    # 回撤、交易次數、勝率
    metrics['max_drawdown'] = float(daily_drawdown.min())
    # metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else np.nan
    metrics['num_trades'] = len(trade_metrics)
    metrics['win_rate'] = (trade_metrics['return'] > 0).sum() / metrics['num_trades'] if metrics['num_trades'] > 0 else 0
    metrics['avg_win'] = trade_metrics[trade_metrics['return'] > 0]['return'].mean() if metrics['win_rate'] > 0 else np.nan
    metrics['avg_loss'] = trade_metrics[trade_metrics['return'] < 0]['return'].mean() if metrics['win_rate'] < 1 else np.nan
    metrics['payoff_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 and not np.isnan(metrics['avg_win']) else np.nan

    # 計算夏普比率與索提諾比率
    daily_dates = df_ind.index.intersection(pd.date_range(start=trade_metrics.index.min(), end=trade_metrics.index.max(), freq='B'))
    daily_equity = pd.Series(index=daily_dates, dtype=float)
    for date, row in trade_metrics.iterrows():
        daily_equity.loc[date] = row['equity']
    daily_equity = daily_equity.ffill()
    daily_returns = daily_equity.pct_change().dropna()

    metrics['sharpe_ratio'] = (daily_returns.mean() * np.sqrt(252)) / daily_returns.std() if daily_returns.std() != 0 else np.nan
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else np.nan
    metrics['sortino_ratio'] = (daily_returns.mean() * np.sqrt(252)) / downside_std if downside_std != 0 else np.nan

    # 最大連續勝/敗次數
    trade_metrics['win_flag'] = trade_metrics['return'] > 0
    trade_metrics['grp'] = (trade_metrics['win_flag'] != trade_metrics['win_flag'].shift(1)).cumsum()
    consec = trade_metrics.groupby(['grp', 'win_flag']).size()
    metrics['max_consecutive_wins'] = consec[consec.index.get_level_values('win_flag') == True].max() if True in consec.index.get_level_values('win_flag') else 0
    metrics['max_consecutive_losses'] = consec[consec.index.get_level_values('win_flag') == False].max() if False in consec.index.get_level_values('win_flag') else 0

    # 年化波動率
    metrics['annualized_volatility'] = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else np.nan

    # 獲利因子
    total_profits = trade_metrics[trade_metrics['return'] > 0]['return'].sum()
    total_losses = abs(trade_metrics[trade_metrics['return'] < 0]['return'].sum())
    metrics['profit_factor'] = total_profits / total_losses if total_losses != 0 else np.nan

    # 使用權益曲線計算更精確的最大回撤
    if equity_curve is not None:
        mmds = calculate_trade_mmds(trades, equity_curve)
        if mmds:
            metrics['max_drawdown'] = float(np.min(mmds)) # 取所有交易中最深回撤
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else np.nan

    return metrics

def calculate_holding_periods(trade_df: pd.DataFrame) -> float:
    '計算所有已完成交易的平均持倉天數。'
    if trade_df.empty or 'trade_date' not in trade_df.columns or 'type' not in trade_df.columns:
        return np.nan

    holding_periods = []
    entry_date = None

    for _, row in trade_df.iterrows():
        if row['type'] == 'buy':
            entry_date = row['trade_date']
        elif row['type'] == 'sell' and entry_date is not None:
            exit_date = row['trade_date']
            holding_days = (exit_date - entry_date).days
            holding_periods.append(holding_days)
            entry_date = None  # 重置

    return np.mean(holding_periods) if holding_periods else np.nan


def backtest_unified(
    df_ind: pd.DataFrame,
    strategy_type: str,
    params: Dict,
    buy_dates: Optional[List[pd.Timestamp]] = None,
    sell_dates: Optional[List[pd.Timestamp]] = None,
    discount: float = 0.30,
    trade_cooldown_bars: int = 3,
    bad_holding: bool = False,
    use_leverage: bool = False,
    lev_params: Optional[Dict] = None
) -> Dict:
    if not isinstance(df_ind, pd.DataFrame):
        logger.error(f"df_ind 型別錯誤，預期 DataFrame，實際為 {type(df_ind)}")
        return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': [], 'equity_curve': pd.Series()}

    # 集成策略分支
    elif strategy_type == "ensemble":
        import traceback
        from contextlib import nullcontext

        status_ctx = st.status("Running Ensemble backtest...", state="running") if hasattr(st, "status") else nullcontext()
        try:
            with status_ctx:
                # 讀取集成策略參數
                method = st.session_state.get('ensemble_method', 'majority')
                floor  = st.session_state.get('ensemble_floor', 0.2)
                ema    = st.session_state.get('ensemble_ema', 3)
                delta  = st.session_state.get('ensemble_delta', 0.3)

                # 交易成本參數
                buy_fee_bp  = float(st.session_state.get('buy_fee_bp',  4.27))
                sell_fee_bp = float(st.session_state.get('sell_fee_bp', 4.27))
                sell_tax_bp = float(st.session_state.get('sell_tax_bp', 30.0))
                slip_bp     = float(st.session_state.get('slippage_bp', 0.0))

                # 解析股票代碼
                ticker_name = (
                    params.get('ticker')
                    or (getattr(df_ind, 'name', None) if hasattr(df_ind, 'name') else None)
                    or 'UNKNOWN'
                )

                cost = CostParams(
                    buy_fee_bp   = buy_fee_bp,
                    sell_fee_bp  = sell_fee_bp,
                    sell_tax_bp  = sell_tax_bp,
                )
                params = EnsembleParams(
                    floor=floor,
                    ema_span=ema,
                    delta_cap=delta,
                    majority_k=6,
                    min_cooldown_days=1,
                    min_trade_dw=0.01
                )
                cfg = RunConfig(
                    ticker=ticker_name,
                    method=method,          # 投票方法
                    params=params,
                    cost=cost,
                    majority_k_pct=0.55    # 多數決門檻比例
                )

                open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg)

                trade_df_ui = normalize_trades_for_ui(trades)
                ledger_ui   = normalize_trades_for_ui(trade_ledger)

                result = {
                    'trades': trades.to_dict('records') if hasattr(trades,'to_dict') else [],
                    'trade_df': trade_df_ui,
                    'trade_ledger': ledger_ui,
                    'signals_df': pd.DataFrame(),
                    'metrics': stats if isinstance(stats, dict) else {},
                    'equity_curve': equity,
                    'daily_state': daily_state,  # 每日狀態（權益/現金/持倉）
                    'weight_curve': w,  # 權重曲線
                }

                # 顯示回測摘要
                st.info(f"[Ensemble] {method} @ {ticker_name} | 交易 {len(trades)} 筆 | 權益序列 {len(equity)} 天")

                # 完成狀態
                st.success(f"集成策略完成: {method_name}")
                st.caption(f"方法={method}, floor={floor}, ema={ema}, delta={delta}, "
                           f"手續費={buy_fee_bp}/{sell_fee_bp} bp, 稅={sell_tax_bp} bp, 滑價={slip_bp} bp")

                # 無交易時的診斷資訊
                if (trades is None) or (len(trades) == 0):
                    st.warning(
                        "集成策略未產生任何交易。"
                        "請檢查門檻參數如 majority_k_pct、delta_cap 和 min_trade_dw。"
                    )
                    # 顯示每日權重以供診斷
                    if daily_state is not None and not daily_state.empty:
                        st.info("daily_state (w) 最後 10 筆資料:")
                        st.dataframe(daily_state[['w']].tail(10) if 'w' in daily_state.columns else daily_state.tail(10))

                logger.info(f"[Ensemble] completed {method_name} with {len(trades)} trades")
                return result

        except Exception as e:
            logger.exception('Ensemble 執行失敗: %s', e)
            st.error(f"[Ensemble] 執行失敗: {e}")
            st.code(traceback.format_exc(), language="text")
            st.stop()
        finally:
            if hasattr(st, "status"):
                status_ctx.update(label='Ensemble 完成' if 'trades' in locals() else 'Ensemble 失敗',
                                  state=("complete" if 'trades' in locals() else "error"))

    tv_alignment_mode = bool(params.get("tv_alignment_mode", False))
    if tv_alignment_mode:
        BUY_FEE_RATE = 0.0
        SELL_FEE_RATE = 0.0
    else:
        BUY_FEE_RATE = BASE_FEE_RATE * discount
        SELL_FEE_RATE = BASE_FEE_RATE * discount + TAX_RATE
    ROUND_TRIP_FEE = BUY_FEE_RATE + SELL_FEE_RATE

    if use_leverage:
        from leverage import LeverageEngine
        lev = LeverageEngine(**(lev_params or {}))
    else:
        lev = None

    required_cols = ['open', 'close'] if strategy_type == 'ssma_turn' else ['open', 'close', 'smaa', 'base', 'sd']
    if df_ind.empty or not all(col in df_ind.columns for col in required_cols):
        logger.warning(f"指標資料為空或缺少必要欄位: {set(required_cols) - set(df_ind.columns)}")
        return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0}, 'equity_curve': pd.Series()}

    try:
        trade_cooldown_bars = int(trade_cooldown_bars)
        if trade_cooldown_bars < 0:
            raise ValueError("trade_cooldown_bars must be >= 0")
        params['stop_loss'] = float(params.get('stop_loss', 0.0))
        if bad_holding and params['stop_loss'] <= 0:
            raise ValueError("stop_loss must be > 0 when bad_holding is enabled")
        if strategy_type == 'ssma_turn':
            params['exit_shift'] = int(params.get('exit_shift', 0))
            if params['exit_shift'] < 0:
                raise ValueError("exit_shift must be >= 0")
        else:
            params['buy_mult'] = float(params.get('buy_mult', 0.5))
            params['sell_mult'] = float(params.get('sell_mult', 0.5))
            params['prom_factor'] = float(params.get('prom_factor', 0.5))
            params['min_dist'] = int(params.get('min_dist', 5))
            if params['buy_mult'] < 0 or params['sell_mult'] < 0:
                raise ValueError('buy_mult 和 sell_mult 不可為負數')
            if params['min_dist'] < 1:
                raise ValueError('min_dist 必須 >= 1')
    except (ValueError, TypeError) as e:
        logger.error(f"回測參數驗證失敗: {e}")
        return {'trades': [], 'trade_df': pd.DataFrame(), 'trades_df': pd.DataFrame(), 'signals_df': pd.DataFrame(), 'metrics': [], 'equity_curve': pd.Series()}

    initial_cash = 100000
    cash = initial_cash
    total_shares = 0
    trades = []
    trade_records = []
    signals = []
    in_pos = False
    entry_price = 0.0
    entry_date = None
    accum_interest = 0.0
    last_trade_idx = -trade_cooldown_bars - 1
    buy_idx = 0
    sell_idx = 0

    # 初始化權益與現金序列
    equity_curve = pd.Series(initial_cash, index=df_ind.index, dtype=float)
    cash_series = pd.Series(initial_cash, index=df_ind.index, dtype=float)
    shares_series = pd.Series(0, index=df_ind.index, dtype=float)

    signals_list = []
    if strategy_type == 'ssma_turn':
        buy_dates = sorted(buy_dates or [])
        sell_dates = sorted(sell_dates or [])
        for dt in buy_dates:
            signals_list.append(TradeSignal(ts=dt, side="BUY", reason="ssma_turn_valley"))
        for dt in sell_dates:
            signals_list.append(TradeSignal(ts=dt, side="SELL", reason="ssma_turn_peak"))
    else:
        pine_parity_mode = bool(params.get("pine_parity_mode", False))
        for i in range(len(df_ind)):
            date = df_ind.index[i]
            buy_level = (
                df_ind['base'].iloc[i] - df_ind['sd'].iloc[i] * params['buy_mult']
                if pine_parity_mode
                else df_ind['base'].iloc[i] + df_ind['sd'].iloc[i] * params['buy_mult']
            )
            if df_ind['smaa'].iloc[i] < buy_level:
                signals_list.append(TradeSignal(ts=date, side="BUY", reason=f"{strategy_type}_buy"))
            elif df_ind['smaa'].iloc[i] > df_ind['base'].iloc[i] + df_ind['sd'].iloc[i] * params['sell_mult']:
                signals_list.append(TradeSignal(ts=date, side="SELL", reason=f"{strategy_type}_sell"))

    signals_list.sort(key=lambda x: x.ts)

    n = len(df_ind)
    scheduled_buy = np.zeros(n, dtype=bool)
    scheduled_sell = np.zeros(n, dtype=bool)
    scheduled_forced = np.zeros(n, dtype=bool)
    scheduled_buy_signal_ts: List[Optional[pd.Timestamp]] = [None] * n
    scheduled_sell_signal_ts: List[Optional[pd.Timestamp]] = [None] * n
    scheduled_forced_signal_ts: List[Optional[pd.Timestamp]] = [None] * n
    scheduled_buy_reason: List[Optional[str]] = [None] * n
    scheduled_sell_reason: List[Optional[str]] = [None] * n
    scheduled_forced_reason: List[Optional[str]] = [None] * n
    idx_by_date = {date: i for i, date in enumerate(df_ind.index)}

    strategy_version = f"{strategy_type}_v{VERSION}"
    if tv_alignment_mode or bool(params.get("pine_parity_mode", False)):
        strategy_version += "_tv"
    if use_leverage:
        strategy_version += "_lev"

    sell_fee_rate_only = 0.0 if tv_alignment_mode else (BASE_FEE_RATE * discount)
    sell_tax_rate_only = 0.0 if tv_alignment_mode else TAX_RATE

    def _indicator_triplet(ts: pd.Timestamp) -> Tuple[float, float, float]:
        if ts is None:
            return (np.nan, np.nan, np.nan)
        dt = pd.Timestamp(ts)
        if dt.tzinfo is not None:
            dt = dt.tz_localize(None)
        if dt not in df_ind.index:
            return (np.nan, np.nan, np.nan)
        row = df_ind.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        smaa = pd.to_numeric(row.get('smaa', np.nan), errors='coerce')
        base = pd.to_numeric(row.get('base', np.nan), errors='coerce')
        sd = pd.to_numeric(row.get('sd', np.nan), errors='coerce')
        return (float(smaa) if pd.notna(smaa) else np.nan,
                float(base) if pd.notna(base) else np.nan,
                float(sd) if pd.notna(sd) else np.nan)

    for sig in signals_list:
        ts = pd.Timestamp(sig.ts).tz_localize(None) if sig.ts.tzinfo else sig.ts
        if ts in idx_by_date:
            i = idx_by_date[ts]
            if i + 1 < n:
                if sig.side == "BUY":
                    scheduled_buy[i + 1] = True
                    scheduled_buy_signal_ts[i + 1] = ts
                    scheduled_buy_reason[i + 1] = str(sig.reason or f"{strategy_type}_buy")
                elif sig.side in ["SELL", "STOP_LOSS", "FORCE_SELL"]:
                    scheduled_sell[i + 1] = True if sig.side == "SELL" else False
                    if sig.side == "SELL":
                        scheduled_sell_signal_ts[i + 1] = ts
                        scheduled_sell_reason[i + 1] = str(sig.reason or f"{strategy_type}_sell")
                    if sig.side in ["STOP_LOSS", "FORCE_SELL"]:
                        scheduled_forced[i + 1] = True
                        scheduled_forced_signal_ts[i + 1] = ts
                        scheduled_forced_reason[i + 1] = str(sig.reason or "force_liquidate")

    for i in range(n):
        today = df_ind.index[i]
        today_open = df_ind['open'].iloc[i]
        today_close = df_ind['close'].iloc[i]
        mkt_val = total_shares * today_close

        if use_leverage and in_pos:
            interest = lev.accrue()
            cash -= interest
            accum_interest += interest
            forced = lev.margin_call(mkt_val=mkt_val)
            if forced > 0 and i + 1 < n:
                scheduled_forced[i + 1] = True
                scheduled_forced_signal_ts[i + 1] = today
                scheduled_forced_reason[i + 1] = "force_liquidate"

        # 更新現金與持股
        cash_series.iloc[i] = cash
        shares_series.iloc[i] = total_shares
        equity_curve.iloc[i] = cash + total_shares * today_close

        if (scheduled_sell[i] or scheduled_forced[i]) and in_pos and total_shares > 0:
            if scheduled_sell[i]:
                signal_ts = scheduled_sell_signal_ts[i]
                signal_reason = str(scheduled_sell_reason[i] or f"{strategy_type}_sell")
            else:
                signal_ts = scheduled_forced_signal_ts[i]
                signal_reason = str(scheduled_forced_reason[i] or "force_liquidate")
            signal_dt = pd.to_datetime(signal_ts, errors='coerce')
            if pd.isna(signal_dt):
                signal_dt = today

            ind_smaa, ind_base, ind_sd = _indicator_triplet(signal_dt)

            exit_price = today_open
            exit_date = today
            trade_ret = (exit_price / entry_price) - 1 - ROUND_TRIP_FEE - (accum_interest / (entry_price * total_shares)) if entry_price != 0 and total_shares > 0 else 0
            if bad_holding and trade_ret < -0.20 and not scheduled_forced[i]:
                continue
            sell_shares = total_shares
            sell_notional = float(sell_shares * exit_price)
            sell_fee_amt = sell_notional * sell_fee_rate_only
            sell_tax_amt = sell_notional * sell_tax_rate_only
            sell_net_amount = sell_notional - sell_fee_amt - sell_tax_amt
            cash += sell_net_amount
            total_shares = 0
            if use_leverage and lev.loan > 0:
                repay_amt = min(cash, lev.loan)
                lev.repay(repay_amt)
                cash -= repay_amt
                trade_records.append({
                    'signal_date': signal_dt,
                    'trade_date': exit_date,
                    'type': 'repay',
                    'price': 0.0,
                    'loan_amount': repay_amt,
                    'reason': 'loan_repayment',
                    'fee': 0.0,
                    'tax': 0.0,
                    'net_amount': -float(repay_amt),
                    'leverage_ratio': 1.0,
                    'strategy_version': strategy_version,
                    'indicator_smaa': ind_smaa,
                    'indicator_base': ind_base,
                    'indicator_sd': ind_sd
                })
                signals.append({'signal_date': signal_dt, 'type': 'repay', 'price': 0.0, 'reason': 'loan_repayment'})
            trades.append((entry_date, trade_ret, exit_date))
            trade_records.append({
                'signal_date': signal_dt,
                'trade_date': exit_date,
                'type': 'sell' if scheduled_sell[i] else 'sell_forced',
                'price': exit_price,
                'shares': sell_shares,
                'return': trade_ret,
                'reason': signal_reason,
                'fee': sell_fee_amt,
                'tax': sell_tax_amt,
                'net_amount': sell_net_amount,
                'leverage_ratio': 1.0,
                'strategy_version': strategy_version,
                'indicator_smaa': ind_smaa,
                'indicator_base': ind_base,
                'indicator_sd': ind_sd
            })
            #logger.info(f"TRADE | type={'sell' if scheduled_sell[i] else 'sell_forced'} | signal_date={today} | trade_date={exit_date} | price={exit_price} | shares={sell_shares} | return={trade_ret}")
            signals.append({'signal_date': signal_dt, 'type': 'sell' if scheduled_sell[i] else 'sell_forced', 'price': today_close, 'reason': signal_reason})
            in_pos = False
            last_trade_idx = i
            accum_interest = 0.0
            if strategy_type == 'ssma_turn' and scheduled_sell[i]:
                sell_idx += 1
            continue

        if scheduled_buy[i] and not in_pos and i - last_trade_idx > trade_cooldown_bars:
            signal_dt = pd.to_datetime(scheduled_buy_signal_ts[i], errors='coerce')
            if pd.isna(signal_dt):
                signal_dt = today
            signal_reason = str(scheduled_buy_reason[i] or f"{strategy_type}_buy")
            ind_smaa, ind_base, ind_sd = _indicator_triplet(signal_dt)

            unit_total_cost = today_open * (1.0 + BUY_FEE_RATE)
            shares = int(cash // unit_total_cost) if unit_total_cost > 0 else 0
            if shares > 0:
                need_cash = shares * today_open
                buy_notional = float(need_cash)
                buy_fee_amt = buy_notional * BUY_FEE_RATE
                buy_net_amount = -(buy_notional + buy_fee_amt)
                total_buy_cost = need_cash + buy_fee_amt
                if use_leverage:
                    gap = total_buy_cost - cash
                    if gap > 0:
                        borrowable = lev.avail(mkt_val=mkt_val)
                        draw = min(gap, borrowable)
                        if draw > 0:
                            lev.borrow(draw)
                            cash += draw
                cash -= total_buy_cost
                total_shares = shares
                entry_price = today_open
                entry_date = today
                in_pos = True
                last_trade_idx = i
                accum_interest = 0.0
                trade_records.append({
                    'signal_date': signal_dt,
                    'trade_date': entry_date,
                    'type': 'buy',
                    'price': entry_price,
                    'shares': shares,
                    'reason': signal_reason,
                    'fee': buy_fee_amt,
                    'tax': 0.0,
                    'net_amount': buy_net_amount,
                    'leverage_ratio': 1.0,
                    'strategy_version': strategy_version,
                    'indicator_smaa': ind_smaa,
                    'indicator_base': ind_base,
                    'indicator_sd': ind_sd
                })
                #logger.info(f"TRADE | type=buy | signal_date={today} | trade_date={entry_date} | price={entry_price} | shares={shares}")
                signals.append({'signal_date': signal_dt, 'type': 'buy', 'price': today_close, 'reason': signal_reason})
                if strategy_type == 'ssma_turn':
                    buy_idx += 1
            continue

        if bad_holding and in_pos and entry_price > 0 and today_close / entry_price - 1 <= -params['stop_loss'] and i + 1 < n:
            scheduled_forced[i + 1] = True
            scheduled_forced_signal_ts[i + 1] = today
            scheduled_forced_reason[i + 1] = "stop_loss"

    # 期末保留未平倉部位，不產生人為的強制平倉交易

    # 更新最後一天的狀態
    cash_series.iloc[-1] = cash
    shares_series.iloc[-1] = total_shares
    equity_curve.iloc[-1] = cash + total_shares * df_ind['close'].iloc[-1]

    trade_df = pd.DataFrame(trade_records)
    trades_df = pd.DataFrame(trades, columns=['entry_date', 'ret', 'exit_date'])
    signals_df = pd.DataFrame(signals)
    metrics = calculate_metrics(trades, df_ind, equity_curve)

    # 以 equity_curve 回填核心績效，確保與每日淨值口徑一致（含費稅後現金流）
    eq_clean = pd.to_numeric(equity_curve, errors='coerce').dropna()
    if len(eq_clean) >= 2:
        total_return_eq = float(eq_clean.iloc[-1] / eq_clean.iloc[0] - 1.0)
        days_eq = max((eq_clean.index[-1] - eq_clean.index[0]).days, 1)
        years_eq = max(days_eq / 365.25, 1e-9)
        annual_return_eq = float((1 + total_return_eq) ** (1 / years_eq) - 1)
        dd_eq = eq_clean / eq_clean.cummax() - 1.0
        mdd_eq = float(dd_eq.min())

        daily_ret_eq = eq_clean.pct_change().dropna()
        sharpe_eq = np.nan
        sortino_eq = np.nan
        ann_vol_eq = np.nan
        if not daily_ret_eq.empty:
            vol = float(daily_ret_eq.std(ddof=0))
            if vol > 0:
                sharpe_eq = float(daily_ret_eq.mean() / vol * np.sqrt(252))
                ann_vol_eq = float(vol * np.sqrt(252))
            downside = daily_ret_eq[daily_ret_eq < 0]
            down_vol = float(downside.std(ddof=0)) if not downside.empty else 0.0
            if down_vol > 0:
                sortino_eq = float(daily_ret_eq.mean() / down_vol * np.sqrt(252))

        metrics['total_return'] = total_return_eq
        metrics['annual_return'] = annual_return_eq
        metrics['max_drawdown'] = mdd_eq
        metrics['calmar_ratio'] = annual_return_eq / abs(mdd_eq) if mdd_eq < 0 else np.nan
        metrics['sharpe_ratio'] = sharpe_eq
        metrics['sortino_ratio'] = sortino_eq
        metrics['annualized_volatility'] = ann_vol_eq

    # 組合每日狀態
    daily_state = pd.DataFrame({
        'equity': equity_curve,
        'cash': cash_series,
        'shares': shares_series
    })
    # 計算持倉權重（投入比例）
    daily_state['w'] = (daily_state['equity'] - daily_state['cash']) / daily_state['equity']
    daily_state['w'] = daily_state['w'].fillna(0).clip(0, 1)

    logger.info(f"{strategy_type} 回測完成 | 總報酬: {metrics.get('total_return', 0):.2%} | 交易次數: {metrics.get('num_trades', 0)}")
    return {
        'trades': trades,
        'trade_df': trade_df,
        'trades_df': trades_df,
        'signals_df': signals_df,
        'metrics': metrics,
        'equity_curve': equity_curve,
        'daily_state': daily_state,  # 每日狀態（權益/現金/持倉）
        'weight_curve': daily_state['w']  # 持倉權重曲線
    }
def compute_backtest_for_periods(ticker: str,periods: List[Tuple[str, str]],strategy_type: str,params: Dict,
    smaa_source: str = "Self",trade_cooldown_bars: int = 3,discount: float = 0.30,
    bad_holding: bool = False,df_price: Optional[pd.DataFrame] = None,df_factor: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
    """
    Run backtests for multiple date ranges.

    Notes:
    - When `df_price` / `df_factor` are provided, slicing is done by date range overlap
      (normalized to calendar day), not exact timestamp membership.
    """
    results: List[Dict] = []
    input_factor = df_factor

    for start_date, end_date in periods:
        logger.info(f"執行區間: {start_date} 至 {end_date}")
        start_ts = pd.to_datetime(start_date).normalize()
        end_ts = pd.to_datetime(end_date).normalize()

        if df_price is None or input_factor is None:
            df_raw, df_factor_slice = load_data(
                ticker,
                start_date,
                end_date,
                smaa_source,
                pine_parity_mode=bool(params.get("pine_parity_mode", False)),
            )
        else:
            df_raw = df_price.loc[(df_price.index >= start_ts) & (df_price.index <= end_ts)].copy()
            if input_factor.empty:
                df_factor_slice = pd.DataFrame()
            else:
                df_factor_slice = input_factor.loc[(input_factor.index >= start_ts) & (input_factor.index <= end_ts)].copy()

        if df_raw.empty:
            logger.warning(f"該區間無股價資料: {start_date} 至 {end_date}")
            results.append({
                'trades': [],
                'trade_df': pd.DataFrame(),
                'signals_df': pd.DataFrame(),
                'metrics': {'total_return': -np.inf, 'num_trades': 0},
                'period': {'start_date': start_date, 'end_date': end_date},
            })
            continue

        if strategy_type == 'ssma_turn':
            calc_keys = [
                'linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days',
                'signal_filter_mode', 'volume_target_pass_rate', 'volume_target_lookback',
            ]
            ssma_params = {k: v for k, v in params.items() if k in calc_keys}
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_raw, df_factor_slice, **ssma_params)
            if df_ind.empty:
                logger.warning(f"{strategy_type} indicator dataframe is empty for period {start_date} to {end_date}")
                results.append({
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': -np.inf, 'num_trades': 0},
                    'period': {'start_date': start_date, 'end_date': end_date},
                })
                continue
            result = backtest_unified(
                df_ind,
                strategy_type,
                params,
                buy_dates,
                sell_dates,
                discount=discount,
                trade_cooldown_bars=trade_cooldown_bars,
                bad_holding=bad_holding,
            )
        else:
            if strategy_type == 'single':
                df_ind = compute_single(df_raw, df_factor_slice, params['linlen'], params['factor'], params['smaalen'], params['devwin'])
            elif strategy_type == 'dual':
                df_ind = compute_dual(df_raw, df_factor_slice, params['linlen'], params['factor'], params['smaalen'], params['short_win'], params['long_win'])
            elif strategy_type == 'RMA':
                df_ind = compute_RMA(
                    df_raw,
                    df_factor_slice,
                    params['linlen'],
                    params['factor'],
                    params['smaalen'],
                    params['rma_len'],
                    params['dev_len'],
                    pine_parity_mode=bool(params.get("pine_parity_mode", False)),
                )
            else:
                logger.warning(f"不支援的策略類型: {strategy_type}")
                df_ind = pd.DataFrame()

            if df_ind.empty:
                logger.warning(f"{strategy_type} indicator dataframe is empty for period {start_date} to {end_date}")
                results.append({
                    'trades': [],
                    'trade_df': pd.DataFrame(),
                    'signals_df': pd.DataFrame(),
                    'metrics': {'total_return': -np.inf, 'num_trades': 0},
                    'period': {'start_date': start_date, 'end_date': end_date},
                })
                continue
            result = backtest_unified(
                df_ind,
                strategy_type,
                params,
                discount=discount,
                trade_cooldown_bars=trade_cooldown_bars,
                bad_holding=bad_holding,
            )

        result['period'] = {'start_date': start_date, 'end_date': end_date}
        results.append(result)

    return results


# ═══ 交易資料正規化與繪圖 ═══

def normalize_trades_for_plots(
    trades: pd.DataFrame,
    price_series: Optional[pd.Series] = None,   # 用於補全缺失價格
) -> pd.DataFrame:
    '將交易記錄正規化為繪圖所需格式（統一欄位名稱並補全價格）。'
    if trades is None or len(trades) == 0:
        return pd.DataFrame()

    trades_plot = trades.copy()
    trades_plot.columns = [str(c).lower() for c in trades_plot.columns]

    # 欄位名稱映射
    if "trade_date" not in trades_plot.columns and "date" in trades_plot.columns:
        trades_plot["trade_date"] = pd.to_datetime(trades_plot["date"], errors="coerce")
    if "type" not in trades_plot.columns and "action" in trades_plot.columns:
        trades_plot["type"] = trades_plot["action"].astype(str).str.lower()

    # 嘗試從其他欄位取得價格
    if "price" not in trades_plot.columns:
        for c in ["open", "price_open", "exec_price", "px", "close"]:
            if c in trades_plot.columns:
                trades_plot["price"] = trades_plot[c]
                break

    if "price" not in trades_plot.columns:
        trades_plot["price"] = pd.NA  # 所有候選欄位皆不存在

    if price_series is not None:
        # 用外部價格序列補全缺失值
        ps = price_series.copy()
        if not isinstance(ps.index, pd.DatetimeIndex):
            ps.index = pd.to_datetime(ps.index, errors="coerce")
        # 對齊交易日期並填入缺失價格
        if "trade_date" in trades_plot.columns:
            td = pd.to_datetime(trades_plot["trade_date"], errors="coerce")
            aligned = ps.reindex(td).reset_index(drop=True)
            # 保底欄位
            need_fill = trades_plot["price"].isna() | ~pd.to_numeric(trades_plot["price"], errors="coerce").notna()
            trades_plot.loc[need_fill, "price"] = aligned[need_fill].values

    return trades_plot

# 股價與交易信號圖
def plot_stock_price(df: pd.DataFrame, trades_df: pd.DataFrame, ticker: str) -> go.Figure:
    '繪製股價走勢與買賣信號標記。'
    # 正規化交易資料
    trades_df = normalize_trades_for_plots(trades_df, price_series=df.get("open", df["close"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='收盤價', line=dict(color='dodgerblue')))

    if not trades_df.empty:
        # 分類買賣信號
        buys = trades_df[trades_df['type'] == 'buy']
        adds = trades_df[trades_df['type'] == 'add'] if 'add' in trades_df['type'].values else pd.DataFrame()
        sells = trades_df[trades_df['type'] == 'sell']

        # 繪製買入信號
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['trade_date'],
                y=buys['price'],
                mode='markers',
                name='買入',
                marker=dict(symbol='cross', size=10, color='green')
            ))

        # 繪製加碼信號
        if not adds.empty:
            fig.add_trace(go.Scatter(
                x=adds['trade_date'],
                y=adds['price'],
                mode='markers',
                name='加碼買入',
                marker=dict(symbol='cross', size=10, color='limegreen')
            ))

        # 繪製賣出信號
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['trade_date'],
                y=sells['price'],
                mode='markers',
                name='賣出',
                marker=dict(symbol='x', size=10, color='red')
            ))

    fig.update_layout(title=f'{ticker} 股價走勢與交易信號',
                      xaxis_title='日期', yaxis_title='價格', template='plotly_white')
    return fig

def plot_indicators(df_ind: pd.DataFrame, strategy_type: str, trades_df: pd.DataFrame, params: Dict) -> go.Figure:
    fig = go.Figure()
    if strategy_type in ['single', 'dual', 'RMA']:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['smaa'], name='SMAA', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base'], name='基準線', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base'] + df_ind['sd'] * params['buy_mult'], name='買入線',
                                 line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base'] + df_ind['sd'] * params['sell_mult'], name='賣出線',
                                 line=dict(color='red', dash='dash')))
        if 'base_long' in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['base_long'], name='長期基準線', line=dict(color='purple', dash='dot')))
    else:  # ssma_turn
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['smaa'], name='SMAA', line=dict(color='blue')))

    # 正規化交易資料以疊加信號
    trades_df = normalize_trades_for_plots(trades_df, price_series=df_ind.get("smaa"))

    buys = trades_df[trades_df['type'] == 'buy']
    sells = trades_df[trades_df['type'] == 'sell']
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys['trade_date'], y=df_ind['smaa'].reindex(buys['trade_date']).values, mode='markers', name='買入信號',
                                 marker=dict(symbol='cross', size=10, color='green')))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells['trade_date'], y=df_ind['smaa'].reindex(sells['trade_date']).values, mode='markers', name='賣出信號',
                                 marker=dict(symbol='x', size=10, color='red')))

    fig.update_layout(title='技術指標與買賣信號',
                      xaxis_title='日期', yaxis_title='指標值', template='plotly_white')
    return fig

def plot_equity_cash(trades_or_ds: pd.DataFrame, df_raw: pd.DataFrame | None = None) -> go.Figure:
    '繪製權益與現金變化曲線。'
    if isinstance(trades_or_ds, pd.DataFrame) and {'equity','cash'}.issubset(trades_or_ds.columns):
        # 直接使用 daily_state 資料
        ds = trades_or_ds.copy()
        if not np.issubdtype(ds.index.dtype, np.datetime64):
            ds.index = pd.to_datetime(ds.index)
        ds = ds.sort_index()

        # 自動偵測並修正 equity 與 cash 反相問題
        if 'position_value' in ds.columns:
            try:
                corr = pd.Series(ds['equity']).corr(pd.Series(ds['cash']))
                looks_like_position_value = (corr is not None and corr < -0.95)
                if looks_like_position_value:
                    eq_fixed = ds['position_value'] + ds['cash']
                    ds = ds.copy()
                    ds['equity'] = eq_fixed  # 自動糾偏
            except Exception:
                pass

        title = 'Equity and Cash'
        if 'recon_mode' in ds.columns and not ds['recon_mode'].empty:
            mode = str(ds['recon_mode'].iloc[0])
            if mode == 'estimated_fixed_qty':
                title = f"{title} (estimated quantity)"
            elif mode == 'actual_qty':
                title = f"{title} (actual quantity)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ds.index, y=ds['equity'], name='權益', line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=ds.index, y=ds['cash'],   name='現金',   line=dict(color='gray')))
        fig.update_layout(title=title, xaxis_title='日期', yaxis_title='金額', template='plotly_white')
        return fig
    else:
        # 從交易記錄重建權益曲線
        try:
            trades_df = normalize_trades_for_plots(trades_or_ds)
            ds = reconstruct_equity_cash_from_trades(trades_df, df_raw)  # 你既有的重建邏輯

            # 檢查必要欄位
            for col in ['equity', 'cash']:
                if col not in ds.columns:
                    raise KeyError(f"重建結果缺少欄位: {col}")

            title = 'Equity and Cash'
            if 'recon_mode' in ds.columns and not ds['recon_mode'].empty:
                mode = str(ds['recon_mode'].iloc[0])
                if mode == 'estimated_fixed_qty':
                    title = f"{title} (estimated quantity)"
                elif mode == 'actual_qty':
                    title = f"{title} (actual quantity)"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ds.index, y=ds['equity'], name='權益', line=dict(color='dodgerblue')))
            fig.add_trace(go.Scatter(x=ds.index, y=ds['cash'],   name='現金',   line=dict(color='gray')))
            fig.update_layout(title=title, xaxis_title='日期', yaxis_title='金額', template='plotly_white')
            return fig
        except Exception as e:
            logger.warning(f"繪製權益/現金曲線失敗: {e}")
            return go.Figure()


def plot_weight_series(daily_state: pd.DataFrame, trades_df: pd.DataFrame = None) -> go.Figure:
    '繪製持倉權重隨時間變化的曲線，並標記買賣點。'
    ds = daily_state.copy()
    if not isinstance(ds.index, pd.DatetimeIndex):
        ds.index = pd.to_datetime(ds.index, errors='coerce')
    ds = ds.sort_index()

    # 取得權重欄位
    if 'w' in ds.columns:
        w = ds['w']
    elif 'invested_pct' in ds.columns:
        w = ds['invested_pct']
    elif 'cash_pct' in ds.columns:
        w = 1 - ds['cash_pct']
    else:
        return go.Figure()  # 無可用欄位時回傳空圖

    fig = go.Figure()

    # 主要權重曲線
    fig.add_trace(go.Scatter(
        x=w.index,
        y=w,
        name='持有權重',
        line=dict(color='dodgerblue', width=2),
        mode='lines'
    ))

    # 疊加交易信號標記
    if trades_df is not None and not trades_df.empty:
        # 確保交易資料有必要的欄位
        if 'trade_date' in trades_df.columns and 'type' in trades_df.columns:
            # 過濾有效的交易日期
            valid_trades = trades_df[
                (trades_df['trade_date'].notna()) &
                (trades_df['trade_date'].isin(w.index))
            ].copy()

            if not valid_trades.empty:
                # 確保日期格式一致
                valid_trades['trade_date'] = pd.to_datetime(valid_trades['trade_date'])

                # 為每筆交易找到對應的權重值
                trade_points = []
                for _, trade in valid_trades.iterrows():
                    trade_date = trade['trade_date']
                    if trade_date in w.index:
                        weight_value = w.loc[trade_date]
                        trade_type = trade['type']

                        # 根據交易類型決定標記樣式
                        if trade_type in ('buy', 'add'):
                            marker_symbol = 'triangle-up'
                            marker_color = 'green'
                            marker_size = 10
                        elif trade_type == 'sell':
                            marker_symbol = 'triangle-down'
                            marker_color = 'red'
                            marker_size = 10
                        else:
                            marker_symbol = 'circle'
                            marker_color = 'orange'
                            marker_size = 8

                        trade_points.append({
                            'date': trade_date,
                            'weight': weight_value,
                            'type': trade_type,
                            'symbol': marker_symbol,
                            'color': marker_color,
                            'size': marker_size
                        })

                # 添加變化點標記
                if trade_points:
                    # 買入點
                    buy_points = [p for p in trade_points if p['type'] in ('buy', 'add')]
                    if buy_points:
                        fig.add_trace(go.Scatter(
                            x=[p['date'] for p in buy_points],
                            y=[p['weight'] for p in buy_points],
                            mode='markers',
                            name='其他交易',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color='green',
                                line=dict(width=1, color='darkgreen')
                            ),
                            hovertemplate='日期: %{x}<br>權重: %{y:.2f}<extra>買入</extra>'
                        ))

                    # 賣出點
                    sell_points = [p for p in trade_points if p['type'] == 'sell']
                    if sell_points:
                        fig.add_trace(go.Scatter(
                            x=[p['date'] for p in sell_points],
                            y=[p['weight'] for p in sell_points],
                            mode='markers',
                            name='賣出',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color='red',
                                line=dict(width=1, color='darkred')
                            ),
                            hovertemplate='日期: %{x}<br>權重: %{y:.2f}<extra>賣出</extra>'
                        ))

                    # 其他類型交易點
                    other_points = [p for p in trade_points if p['type'] not in ('buy', 'add', 'sell')]
                    if other_points:
                        fig.add_trace(go.Scatter(
                            x=[p['date'] for p in other_points],
                            y=[p['weight'] for p in other_points],
                            mode='markers',
                            name='其他交易',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color='orange',
                                line=dict(width=1, color='darkorange')
                            ),
                            hovertemplate='日期: %{x}<br>權重: %{y:.2f}<extra>其他</extra>'
                        ))

    # 更新圖表佈局
    fig.update_layout(
        title='持有權重變化',
        xaxis_title='日期',
        yaxis_title='權重 (0~1)',
        template='plotly_white',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def reconstruct_equity_cash_from_trades(trades_df: pd.DataFrame, df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Reconstruct daily equity/cash from trade ledger.

    Returns columns:
    - equity
    - cash
    - recon_mode: actual_qty | estimated_fixed_qty
    """
    if trades_df is None or trades_df.empty or df_raw is None or df_raw.empty:
        return pd.DataFrame()

    # 確保有必要的欄位
    required_cols = ['trade_date', 'type', 'price']
    if not all(col in trades_df.columns for col in required_cols):
        logger.warning(f"交易紀錄缺少必要欄位: {required_cols}")
        return pd.DataFrame()

    dates = pd.to_datetime(df_raw.index)
    initial_cash = 1_000_000.0

    trades_df = trades_df.copy()
    trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'], errors='coerce')
    trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
    trades_df['type'] = trades_df['type'].astype(str).str.lower()

    has_qty = 'qty' in trades_df.columns
    if has_qty:
        trades_df['qty'] = pd.to_numeric(trades_df['qty'], errors='coerce')
    use_actual_qty = has_qty and trades_df['qty'].notna().any()
    recon_mode = 'actual_qty' if use_actual_qty else 'estimated_fixed_qty'

    by_date = {}
    for _, row in trades_df.iterrows():
        dt = row['trade_date']
        px = row['price']
        if pd.isna(dt) or pd.isna(px):
            continue
        dt = pd.Timestamp(dt).normalize()
        by_date.setdefault(dt, []).append(row)

    cash = initial_cash
    shares = 0.0
    cash_series = pd.Series(index=dates, dtype=float)
    shares_series = pd.Series(index=dates, dtype=float)

    for dt in dates:
        day = pd.Timestamp(dt).normalize()
        rows = by_date.get(day, [])
        for row in rows:
            px = float(row['price'])
            side = str(row['type'])
            if use_actual_qty:
                qty_val = float(row.get('qty', 0.0))
                trade_qty = max(0.0, qty_val)
            else:
                trade_qty = 1000.0

            if side in ('buy', 'add'):
                shares += trade_qty
                cash -= trade_qty * px
            elif side == 'sell':
                if use_actual_qty:
                    sell_qty = min(trade_qty, shares)
                else:
                    sell_qty = shares if shares > 0 else trade_qty
                shares -= sell_qty
                cash += sell_qty * px

        cash_series.loc[dt] = cash
        shares_series.loc[dt] = shares

    close_series = pd.to_numeric(df_raw['close'], errors='coerce').reindex(dates).ffill().bfill()
    equity_series = cash_series + shares_series * close_series

    result = pd.DataFrame({'equity': equity_series, 'cash': cash_series}, index=dates)
    result['recon_mode'] = recon_mode
    return result
def display_metrics_flex(metrics: dict):
    '以 flex 卡片式排版顯示績效指標。'
    # 組裝指標項目
    items = []
    for k, v in metrics.items():
        # 根據指標類型格式化數值
        if k in ["total_return", "annual_return", "win_rate", "max_drawdown", "annualized_volatility", "avg_win", "avg_loss"]:
            txt = f"{v:.2%}" if pd.notna(v) else ""
        elif k in ["calmar_ratio", "sharpe_ratio", "sortino_ratio", "payoff_ratio", "profit_factor"]:
            txt = f"{v:.2f}" if pd.notna(v) else ""
        elif k in ["max_drawdown_duration", "avg_holding_period"]:
            txt = f"{v:.1f} 天" if pd.notna(v) else ""
        elif k in ["num_trades", "max_consecutive_wins", "max_consecutive_losses"]:
            txt = str(int(v)) if pd.notna(v) else ""
        else:
            # 其他就先盡量當純文字顯示
            txt = f"{v}"
        # 指標名稱映射
        label_map = {
            "total_return": "總報酬率",
            "annual_return": "年化報酬率",
            "win_rate": "勝率",
            "max_drawdown": "最大回撤",
            "max_drawdown_duration": "最大回撤持續天數",
            "calmar_ratio": "卡瑪比率",
            "sharpe_ratio": "夏普比率",
            "sortino_ratio": "索提諾比率",
            "payoff_ratio": "盈虧比",
            "profit_factor": "獲利因子",
            "num_trades": "交易次數",
            "avg_holding_period": "平均持倉天數",
            "annualized_volatility": "年化波動率",
            "max_consecutive_wins": "最大連勝次數",
            "max_consecutive_losses": "最大連敗次數",
            "avg_win": "平均獲利",
            "avg_loss": "平均虧損",
        }
        label = label_map.get(k, k)
        items.append((label, txt))

    # 產生 HTML 卡片
    html = """
<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;">
"""
    for label, val in items:
        html += f"""
  <div style="flex:0 1 150px;border:1px solid #444;border-radius:4px;padding:8px 12px;background:#1a1a1a;">
    <div style="font-size:14px;color:#aaa;">{label}</div>
    <div style="font-size:20px;font-weight:bold;color:#fff;margin-top:4px;">{val}</div>
  </div>
"""
    html += "</div>"

    # 移除多餘縮排
    html = textwrap.dedent(html)

    st.markdown(html, unsafe_allow_html=True)

def display_strategy_summary(strategy: str, params: Dict, metrics: Dict, smaa_source: str, trade_df: pd.DataFrame):
    '顯示策略摘要：參數、持倉天數、績效指標。'
    # 參數展示
    param_display = {k: v for k, v in params.items() if k != "strategy_type"}
    st.write('策略參數: ' + ", ".join(f"{k}: {v}" for k, v in param_display.items()))

    # 計算平均持倉天數
    avg_holding_period = calculate_holding_periods(trade_df)

    # 加入平均持倉天數
    metrics['avg_holding_period'] = avg_holding_period

    # 顯示績效指標
    if metrics:
        display_metrics_flex(metrics)
    else:
        st.warning('無績效指標可顯示')
# ═══ Streamlit 主程式 ═══
def run_app():
    st.set_page_config(layout="wide")
    st.sidebar.title('SSS 策略回測系統')
    page = st.sidebar.selectbox("選擇頁面", ["Backtest", "Compare", "Version History", "Cache Status"])

    try:
        from version_history import get_version_history_html
        if page == "Version History":
            html = get_version_history_html()
            st.markdown(html, unsafe_allow_html=True)
            return
    except ImportError:
        st.warning('version_history 模組未安裝')
        if page == "Version History":
            st.markdown("")
            return

    # 快取管理頁面
    if page == "Cache Status":
        st.title('快取管理')
        st.write("管理與檢視系統快取狀態。")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("清除快取")
            if st.button("清除所有快取"):
                if clear_all_caches():
                    st.success("已清除所有快取")
                else:
                    st.error("清除快取失敗")

            st.write("**快取層級**")
            st.write("- Joblib 快取")
            st.write("- SMAA 指標快取")
            st.write("- Optuna 最佳化快取")

        with col2:
            st.subheader("強制更新股價")
            ticker_to_update = st.selectbox(
                "選擇要更新的股票代碼:",
                ["ALL", "00631L.TW", "^TWII", "2414.TW", "2412.TW"],
                index=0
            )

            if st.button("執行強制更新"):
                if ticker_to_update == "ALL":
                    success = force_update_price_data()
                else:
                    success = force_update_price_data(ticker_to_update)

                if success:
                    st.success(f"{ticker_to_update} 股價資料已更新")
                else:
                    st.error(f"{ticker_to_update} 股價更新失敗")

        st.subheader("快取詳情")
        cache_info = []

        # 收集快取資訊
        try:
            joblib_cache_size = len(list(cfg.MEMORY.store_backend.get_items()))
            cache_info.append(f"Joblib 快取項目數: {joblib_cache_size}")
        except Exception:
            cache_info.append("Joblib 快取項目數: 無法取得")

        smaa_cache_dir = CACHE_DIR / "cache_smaa"
        if smaa_cache_dir.exists():
            smaa_files = len(list(smaa_cache_dir.glob("*.joblib")))
            cache_info.append(f"SMAA 快取檔案數: {smaa_files}")
        else:
            cache_info.append("SMAA 快取檔案: 目錄不存在")

        optuna_cache_dir = CACHE_DIR / "optuna16_equity"
        if optuna_cache_dir.exists():
            optuna_files = len(list(optuna_cache_dir.glob("*.npy")))
            cache_info.append(f"Optuna 快取檔案數: {optuna_files}")
        else:
            cache_info.append("Optuna 快取檔案: 目錄不存在")

        for info in cache_info:
            st.write(f"- {info}")

        return

    # 投資組合分析
    if page == "Portfolio Analysis":
        st.title("投資組合分析")
        st.write("分析所選策略的組合績效表現。")

        # 側邊欄設定
        st.sidebar.header("投資組合設定")

        # 選擇策略
        strategy_names = list(param_presets.keys())
        selected_strategies = st.sidebar.multiselect(
            "選擇策略:",
            strategy_names,
            default=strategy_names[:5]  # 預設選前五個
        )

        if not selected_strategies:
            st.warning("請至少選擇一個策略")
            return

        # 權重分配方法
        weight_method = st.sidebar.selectbox(
            "權重分配方法:",
            ["Equal Weight", "Sharpe Weight", "Min Variance", "Return Weight"],
            index=0
        )

        # 重新平衡頻率
        rebalance_freq = st.sidebar.selectbox(
            "再平衡頻率:",
            ["Quarterly", "Yearly", "No Rebalance"],
            index=1
        )

        # 回看期間
        lookback_period = st.sidebar.slider(
            '回看期間（交易日）',
            min_value=60,
            max_value=504,  # 約兩年
            value=252,  # 約一年
            step=21
        )

        # 基本設定
        ticker = st.sidebar.selectbox('股票代碼', ["00631L.TW", "2330.TW", "AAPL", "VOO"], index=0)
        start_date = st.sidebar.text_input('起始日期', "2010-01-01")
        end_date = st.sidebar.text_input('結束日期', "")
        discount = st.sidebar.slider('手續費折扣', min_value=0.1, max_value=0.70, value=0.30, step=0.01)

        # 執行分析
        run_analysis = st.sidebar.button('開始分析')

        if run_analysis:
            with st.spinner('正在計算投資組合...'):
                # 載入價格資料
                df_raw, df_factor = load_data(ticker, start_date=start_date,
                                             end_date=end_date if end_date else None,
                                             smaa_source="Self", force_update=False)

                if df_raw.empty:
                    st.error(f"無法載入 {ticker} 的股價資料")
                    return

                # 逐策略回測
                strategy_results = {}
                equity_curves = {}

                for strategy in selected_strategies:
                    params = param_presets[strategy]
                    strategy_type = params["strategy_type"]
                    smaa_source = params.get("smaa_source", "Self")

                    # 載入該策略所需資料
                    df_raw_strategy, df_factor_strategy = load_data(
                        ticker, start_date=start_date,
                        end_date=end_date if end_date else None,
                        smaa_source=smaa_source,
                        force_update=False,
                        data_provider=params.get("data_provider", "yfinance"),
                        pine_parity_mode=bool(params.get("pine_parity_mode", False)),
                    )

                    # 執行回測
                    if strategy_type == 'ssma_turn':
                        calc_keys = [
                            'linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                            'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days',
                            'quantile_win', 'signal_filter_mode', 'volume_target_pass_rate',
                            'volume_target_lookback',
                        ]
                        ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                        backtest_params = ssma_params.copy()
                        backtest_params['stop_loss'] = params.get('stop_loss', 0.0)

                        df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                            df_raw_strategy, df_factor_strategy, **ssma_params, smaa_source=smaa_source
                        )

                        if not df_ind.empty:
                            result = backtest_unified(
                                df_ind, strategy_type, backtest_params, buy_dates, sell_dates,
                                discount=discount, trade_cooldown_bars=3, bad_holding=False
                            )
                            strategy_results[strategy] = result
                            if 'equity_curve' in result:
                                equity_curves[strategy] = result['equity_curve']
                    else:
                        if strategy_type == 'single':
                            df_ind = compute_single(
                                df_raw_strategy, df_factor_strategy, params["linlen"], params["factor"],
                                params["smaalen"], params["devwin"], smaa_source=smaa_source
                            )
                        elif strategy_type == 'dual':
                            df_ind = compute_dual(
                                df_raw_strategy, df_factor_strategy, params["linlen"], params["factor"],
                                params["smaalen"], params["short_win"], params["long_win"], smaa_source=smaa_source
                            )
                        elif strategy_type == 'RMA':
                            df_ind = compute_RMA(
                                df_raw_strategy, df_factor_strategy, params["linlen"], params["factor"],
                                params["smaalen"], params["rma_len"], params["dev_len"], smaa_source=smaa_source,
                                pine_parity_mode=bool(params.get("pine_parity_mode", False)),
                            )

                        if not df_ind.empty:
                            result = backtest_unified(
                                df_ind, strategy_type, params, discount=discount,
                                trade_cooldown_bars=3, bad_holding=False
                            )
                            strategy_results[strategy] = result
                            if 'equity_curve' in result:
                                equity_curves[strategy] = result['equity_curve']

                if not equity_curves:
                    st.error("未產生任何策略權益曲線，請檢查策略設定")
                    return

                # 計算投資組合權重與績效
                portfolio_analysis = calculate_portfolio_weights_and_performance(
                    equity_curves, weight_method, rebalance_freq, lookback_period
                )

                # 顯示分析結果
                display_portfolio_analysis(portfolio_analysis, selected_strategies, weight_method, rebalance_freq)

        return

    # 回測參數設定
    st.sidebar.header('回測設定')
    default_tickers = ["00631L.TW", "2330.TW", "AAPL", "VOO"]
    ticker = st.sidebar.selectbox('股票代碼', default_tickers, index=0)
    start_date_input = st.sidebar.text_input('起始日期', "2010-01-01")
    end_date_input = st.sidebar.text_input('結束日期', "")
    trade_cooldown_bars = st.sidebar.number_input('交易冷卻期（根數）', min_value=0, max_value=20, value=3, step=1, format="%d")
    discount = st.sidebar.slider('手續費折扣', min_value=0.1, max_value=0.70, value=0.30, step=0.01)
    st.sidebar.markdown('---')
    bad_holding = st.sidebar.checkbox('啟用停損', value=False)

    # 進階選項
    st.sidebar.header('進階選項')
    force_update = st.sidebar.checkbox('強制更新股價', value=False)
    clear_cache_before_run = st.sidebar.checkbox("執行前清除快取", value=False)

    run_backtests = st.sidebar.button("執行全部回測")

    # 載入基礎股價資料
    df_raw, df_factor = load_data(ticker, start_date=start_date_input,
                                  end_date=end_date_input if end_date_input else None,
                                  smaa_source="Self", force_update=force_update)
    if df_raw.empty:
        st.error(f"無法載入 {ticker} 的股價資料，請確認代號是否正確")
        return

    # 建立各策略標籤頁
    strategy_names = list(param_presets.keys())
    tabs = st.tabs(strategy_names + ["策略比較"])
    results = {}

    if run_backtests:
        # 執行前清除快取
        if clear_cache_before_run:
            clear_all_caches()
            st.info("已於執行前清除快取")

        with st.spinner('正在執行所有策略回測...'):
            for strategy in strategy_names:
                params = param_presets[strategy]
                strategy_type = params["strategy_type"]
                default_source = params.get("smaa_source", "Self")
                smaa_source = default_source  # 批量回測時使用預設值
                # 載入該策略所需資料
                df_raw, df_factor = load_data(
ticker,
start_date=start_date_input,
end_date=end_date_input if end_date_input else None,
smaa_source=smaa_source,
force_update=force_update,
data_provider=params.get("data_provider", "yfinance"),
pine_parity_mode=bool(params.get("pine_parity_mode", False))
                )

                if strategy_type == 'ssma_turn':
                    calc_keys = [
                        'linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                        'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win',
                        'signal_filter_mode', 'volume_target_pass_rate', 'volume_target_lookback',
                    ]
                    ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                    backtest_params = ssma_params.copy()
                    backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
                    df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                        df_raw, df_factor, **ssma_params, smaa_source=smaa_source
                    )
                    if df_ind.empty:
                        st.warning(f"{strategy} 指標計算結果為空，跳過此策略")
                        continue
                    result = backtest_unified(
                        df_ind, strategy_type, backtest_params, buy_dates, sell_dates,
                        discount=discount, trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                    )
                else:
                    if strategy_type == 'single':
                        df_ind = compute_single(
                            df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                            params["devwin"], smaa_source=smaa_source
                        )
                    elif strategy_type == 'dual':
                        df_ind = compute_dual(
                            df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                            params["short_win"], params["long_win"], smaa_source=smaa_source
                        )
                    elif strategy_type == 'RMA':
                        df_ind = compute_RMA(
                            df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                            params["rma_len"], params["dev_len"], smaa_source=smaa_source,
                            pine_parity_mode=bool(params.get("pine_parity_mode", False)),
                        )
                    if df_ind.empty:
                        st.warning(f"{strategy} 指標計算結果為空，跳過此策略")
                        continue
                    result = backtest_unified(
                        df_ind, strategy_type, params, discount=discount,
                        trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                    )

                results[strategy] = (df_ind, result['trades'], result['trade_df'],
                                     result['signals_df'], result['metrics'])

    for tab, strategy in zip(tabs[:-1], strategy_names):
        with tab:
            col1, col2 = st.columns([3, 1])
            with col1:
                # 策略說明 tooltip
                if strategy.startswith("Single"):
                    tooltip = f"{strategy}: 單通道策略"
                elif strategy == "Dual-Scale":
                    tooltip = f"{strategy}: 雙尺度策略"
                elif strategy in ["SSMA_turn_1", "SSMA_turn_2"]:
                    tooltip = f"{strategy}: SMAA 轉折策略"
                else:
                    tooltip = f'{strategy}'

                # 策略標題
                html = f"""
                <h3>
                    <span title="{tooltip}"> {strategy}
                    </span>
                </h3>
                """
                st.markdown(html, unsafe_allow_html=True)

            with col2:
                # SMAA 來源選擇
                params = param_presets[strategy]
                default_source = params.get("smaa_source", "Self")
                options = ["Self", "Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]
                smaa_source = st.selectbox(
                    'SMAA 來源',
                    options,
                    index=options.index(default_source) if default_source in options else 0,
                    key=f"smaa_source_{strategy}"
                )

            strategy_type = params["strategy_type"]
            # 載入資料並計算指標
            df_raw, df_factor = load_data(
                ticker,
                start_date=start_date_input,
                end_date=end_date_input if end_date_input else None,
                smaa_source=smaa_source,
                force_update=force_update,
                data_provider=params.get("data_provider", "yfinance"),
                pine_parity_mode=bool(params.get("pine_parity_mode", False))
            )

            # 依策略類型計算指標並回測
            if strategy_type == 'ssma_turn':
                calc_keys = [
                    'linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                    'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days',
                    'quantile_win', 'signal_filter_mode', 'volume_target_pass_rate',
                    'volume_target_lookback',
                ]
                ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                backtest_params = ssma_params.copy()
                backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
                df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                    df_raw, df_factor, **ssma_params, smaa_source=smaa_source
                )
                if df_ind.empty:
                    st.warning(f"{strategy} 指標計算結果為空，跳過此策略")
                    continue
                result = backtest_unified(
                    df_ind, strategy_type, backtest_params, buy_dates, sell_dates,
                    discount=discount, trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                )
            else:
                if strategy_type == 'single':
                    df_ind = compute_single(
                        df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                        params["devwin"], smaa_source=smaa_source
                    )
                elif strategy_type == 'dual':
                    df_ind = compute_dual(
                        df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                        params["short_win"], params["long_win"], smaa_source=smaa_source
                    )
                elif strategy_type == 'RMA':
                    df_ind = compute_RMA(
                        df_raw, df_factor, params["linlen"], params["factor"], params["smaalen"],
                        params["rma_len"], params["dev_len"], smaa_source=smaa_source,
                        pine_parity_mode=bool(params.get("pine_parity_mode", False)),
                    )
                if df_ind.empty:
                    st.warning(f"{strategy} 指標計算結果為空，跳過此策略")
                    continue
                result = backtest_unified(
                    df_ind, strategy_type, params, discount=discount,
                    trade_cooldown_bars=trade_cooldown_bars, bad_holding=bad_holding
                )

            results[strategy] = (df_ind, result['trades'], result['trade_df'],
                                 result['signals_df'], result['metrics'])

            # 顯示策略摘要
            display_strategy_summary(strategy, params, result['metrics'], smaa_source, result['trade_df'])

            # 顯示圖表與交易明細
            has_trades = ('trade_df' in result) and (result['trade_df'] is not None) and (not result['trade_df'].empty)
            if has_trades:
                st.plotly_chart(plot_stock_price(df_raw, result['trade_df'], ticker),
                                use_container_width=True, key=f"stock_price_{strategy}")

                # 取得每日狀態或交易資料
                ds_or_trades = result.get('daily_state', result.get('trade_df'))
                # 正規化每日狀態
                if ds_or_trades is not None and not ds_or_trades.empty:
                    ds_or_trades = normalize_daily_state(ds_or_trades)

                # 繪圖前的除錯資訊
                try:
                    print(f"除錯繪圖 策略={strategy}")
                    print(f"   ds_or_trades 型別: {type(ds_or_trades)}")
                    if hasattr(ds_or_trades, 'shape'):
                        print(f"   ds_or_trades 維度: {ds_or_trades.shape}")
                    if hasattr(ds_or_trades, 'columns'):
                        print(f"   ds_or_trades 欄位: {list(ds_or_trades.columns)}")

                    # 若已包含 equity/cash 欄位，直接輸出
                    if isinstance(ds_or_trades, pd.DataFrame) and {"equity", "cash"}.issubset(ds_or_trades.columns):
                        print("   偵測到 equity/cash 欄位，呼叫 dump_equity_cash")
                        dump_equity_cash(f"dash_{strategy}", ds_or_trades)
                    else:
                        print(f"   缺少 equity/cash 欄位，改用備用輸出")
                        # 輸出交易數據作為備用
                        if 'trade_df' in result and result['trade_df'] is not None:
                            dump_timeseries(f"dash_{strategy}_trades", trades=result['trade_df'])
                        # 輸出每日狀態作為備用
                        if 'daily_state' in result and result['daily_state'] is not None:
                            dump_timeseries(f"dash_{strategy}_daily_state", daily_state=result['daily_state'])

                    # 輸出權重曲線
                    if 'weight_curve' in result:
                        print(f"   weight_curve type: {type(result['weight_curve'])}")
                        if hasattr(result['weight_curve'], 'shape'):
                            print(f"   weight_curve shape: {result['weight_curve'].shape}")
                        dump_timeseries(f"dash_{strategy}_weights", weight=result['weight_curve'], price=df_raw['close'])
                    else:
                        print("   warning: missing weight_curve in result")
                        print(f"   result keys: {list(result.keys())}")
                except Exception as e:
                    print(f"plot debug failed: {e}")
                    import traceback
                    traceback.print_exc()

                st.plotly_chart(plot_equity_cash(ds_or_trades, df_raw),
                                use_container_width=True, key=f"equity_cash_{strategy}")
                st.plotly_chart(plot_indicators(df_ind, strategy_type, result['trade_df'], params),
                                use_container_width=True, key=f"indicators_{strategy}")
                st.subheader('交易明細')
                # 選擇顯示用的交易資料
                display_df = None
                if strategy_type == 'ensemble' and 'trade_ledger' in result and not result['trade_ledger'].empty:
                    display_df = result['trade_ledger']
                else:
                    display_df = result['trade_df']

                # 正規化並格式化顯示
                display_df_std = normalize_trades_for_ui(display_df)
                display_df_formatted = format_trade_df_for_display(display_df_std)

                st.dataframe(display_df_formatted)
                st.markdown('---')
            else:
                st.warning(f"{strategy}: 無交易記錄")

    # 買賣點比較標籤頁
    with tabs[-1]:
        st.subheader('所有策略買賣點比較')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['close'], name='收盤價', line=dict(color='dodgerblue')))
        colors = ['green', 'limegreen', 'red', 'orange', 'purple', 'blue', 'pink', 'cyan']
        for i, strategy in enumerate(strategy_names):
            if strategy in results and not results[strategy][2].empty:
                trade_df = results[strategy][2]
                buys = trade_df[trade_df['type'] == 'buy']
                sells = trade_df[trade_df['type'] == 'sell']
                fig.add_trace(go.Scatter(x=buys['trade_date'], y=buys['price'], mode='markers', name=f'{strategy} 買入',
                                         marker=dict(symbol='cross', size=8, color=colors[i % len(colors)])))
                fig.add_trace(go.Scatter(x=sells['trade_date'], y=sells['price'], mode='markers', name=f'{strategy} 賣出',
                                         marker=dict(symbol='x', size=8, color=colors[i % len(colors)])))
        fig.update_layout(title=f'{ticker} 所有策略買賣點比較',
                          xaxis_title='日期', yaxis_title='股價', template='plotly_white')
        fig.update_layout(
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left',
                yanchor='top',
                bordercolor="Black",
                borderwidth=1,
                bgcolor="white",
                itemsizing='constant',
                orientation='v'
            )
        )
        st.plotly_chart(fig, use_container_width=True, key="buy_sell_comparison")

        st.subheader("策略績效比較")
        comparison_data = []
        for strategy in strategy_names:
            if strategy in results and results[strategy][4]:
                metrics = results[strategy][4]
                comparison_data.append({
                    '策略': strategy,
                    '總回報率': f"{metrics.get('total_return', 0):.2%}",
                    '年化報酬率': f"{metrics.get('annual_return', 0):.2%}",
                    '最大回撤': f"{metrics.get('max_drawdown', 0):.2%}",
                    '卡瑪比率': f"{metrics.get('calmar_ratio', 0):.2f}",
                    '交易次數': metrics.get('num_trades', 0),
                    '勝率': f"{metrics.get('win_rate', 0):.2%}",
                    '盈虧比': f"{metrics.get('payoff_ratio', 0):.2f}"
                })
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data))
        else:
            st.warning('無比較數據可顯示，請先執行回測')

# 投資組合分析函數
def calculate_portfolio_weights_and_performance(equity_curves, weight_method, rebalance_freq, lookback_period):
    """計算投資組合權重與綜合權益績效。"""
    if not equity_curves:
        return {}

    # 計算重新平衡日期
    all_dates = set()
    for equity_curve in equity_curves.values():
        all_dates.update(equity_curve.index)

    common_dates = sorted(list(all_dates))
    if len(common_dates) < 2:
        return {}

    # 初始化投資組合
    portfolio_equity = pd.Series(index=common_dates, dtype=float)
    portfolio_equity.iloc[0] = 1.0

    # 權重歷史記錄
    weight_history = {strategy: [] for strategy in equity_curves.keys()}
    date_history = []

    # 執行投資組合計算
    n_strategies = len(equity_curves)
    current_weights = {strategy: 1.0 / n_strategies for strategy in equity_curves.keys()}

    # 執行投資組合計算
    if rebalance_freq == "Quarterly":
        rebalance_months = 3
    elif rebalance_freq == "Yearly":
        rebalance_months = 12
    else:
        rebalance_months = None

    # 計算重新平衡日期
    rebalance_dates = []
    if rebalance_months:
        current_date = pd.to_datetime(common_dates[0])
        end_date = pd.to_datetime(common_dates[-1])

        while current_date <= end_date:
            rebalance_dates.append(current_date)
            # 計算下一個重新平衡日期
            if current_date.month + rebalance_months > 12:
                year = current_date.year + 1
                month = (current_date.month + rebalance_months) % 12
                if month == 0:
                    month = 12
            else:
                year = current_date.year
                month = current_date.month + rebalance_months

            current_date = current_date.replace(year=year, month=month)

    # 執行投資組合計算
    for i, date in enumerate(common_dates):
        if i == 0:
            continue

        # 檢查是否需要重新平衡
        if rebalance_months and pd.to_datetime(date) in rebalance_dates:
            # 重新計算權重
            performance_metrics = {}
            for strategy_name, equity_curve in equity_curves.items():
                if date in equity_curve.index:
                    # 計算過去期間的表現
                    lookback_start = max(0, i - lookback_period)
                    if lookback_start < len(equity_curve):
                        past_equity = equity_curve.iloc[lookback_start:i+1]
                        if len(past_equity) > 1:
                            if weight_method == "Equal Weight":
                                performance_metrics[strategy_name] = 1.0
                            elif weight_method == "Sharpe Weight":
                                returns = past_equity.pct_change().dropna()
                                if len(returns) > 0 and returns.std() > 0:
                                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                                    performance_metrics[strategy_name] = max(sharpe, 0)
                                else:
                                    performance_metrics[strategy_name] = 0
                            elif weight_method == "Min Variance":
                                returns = past_equity.pct_change().dropna()
                                if len(returns) > 0 and returns.std() > 0:
                                    performance_metrics[strategy_name] = 1.0 / (returns.std() ** 2)
                                else:
                                    performance_metrics[strategy_name] = 0
                            elif weight_method == "Return Weight":
                                performance = (past_equity.iloc[-1] / past_equity.iloc[0]) - 1
                                performance_metrics[strategy_name] = max(performance, 0)
                            else:
                                performance_metrics[strategy_name] = 1.0
                        else:
                            performance_metrics[strategy_name] = 0
                    else:
                        performance_metrics[strategy_name] = 0
                else:
                    performance_metrics[strategy_name] = 0

            # 重新計算權重
            total_performance = sum(performance_metrics.values())
            if total_performance > 0:
                current_weights = {name: perf / total_performance for name, perf in performance_metrics.items()}
            else:
                # 所有策略表現為零時回退至等權重
                current_weights = {name: 1.0 / n_strategies for name in equity_curves.keys()}

        # 記錄權重
        date_history.append(date)
        for strategy in equity_curves.keys():
            weight_history[strategy].append(current_weights.get(strategy, 0))

        # 計算當日組合價值
        daily_return = 0
        for strategy_name, equity_curve in equity_curves.items():
            if date in equity_curve.index and i > 0:
                if i > 0 and common_dates[i-1] in equity_curve.index:
                    strategy_return = (equity_curve.loc[date] / equity_curve.loc[common_dates[i-1]]) - 1
                    daily_return += strategy_return * current_weights[strategy_name]

        # 更新組合權益
        portfolio_equity.loc[date] = portfolio_equity.loc[common_dates[i-1]] * (1 + daily_return)

    # 計算組合績效指標
    total_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
    years = (pd.to_datetime(common_dates[-1]) - pd.to_datetime(common_dates[0])).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 最大回撤
    cumulative_max = portfolio_equity.expanding().max()
    drawdown = (portfolio_equity - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # 夏普比率
    daily_returns = portfolio_equity.pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    return {
        'portfolio_equity': portfolio_equity,
        'weight_history': weight_history,
        'date_history': date_history,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'rebalance_freq': rebalance_freq,
        'weight_method': weight_method,
        'individual_equity_curves': equity_curves
    }

def display_portfolio_analysis(portfolio_analysis, selected_strategies, weight_method, rebalance_freq):
    '顯示投資組合分析結果：權益曲線、權重歷史、績效指標。'
    if not portfolio_analysis:
        st.error('投資組合分析失敗')
        return

    # 建立分頁
    tabs = st.tabs(['權益曲線', '權重歷史', '績效指標', '權重檢查'])

    # 權益曲線標籤頁
    with tabs[0]:
        st.subheader('投資組合權益曲線')

        fig = go.Figure()

        # 投資組合權益曲線
        portfolio_equity = portfolio_analysis['portfolio_equity']
        fig.add_trace(go.Scatter(
            x=portfolio_equity.index,
            y=portfolio_equity.values,
            name='投資組合',
            line=dict(color='red', width=3),
            mode='lines'
        ))

        # 疊加個別策略權益曲線
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, strategy in enumerate(selected_strategies):
            if strategy in portfolio_analysis['individual_equity_curves']:
                equity_curve = portfolio_analysis['individual_equity_curves'][strategy]
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    name=strategy,
                    line=dict(color=colors[i % len(colors)], width=1),
                    mode='lines'
                ))

        fig.update_layout(
            title=f"投資組合權益曲線 ({weight_method}, {rebalance_freq})",
            xaxis_title='日期',
            yaxis_title='權益值',
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # 權重歷史標籤頁
    with tabs[1]:
        st.subheader('策略權重歷史')

        # 顯示權重統計
        fig = go.Figure()

        weight_history = portfolio_analysis['weight_history']
        date_history = portfolio_analysis['date_history']

        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, strategy in enumerate(selected_strategies):
            if strategy in weight_history:
                weights = weight_history[strategy]
                fig.add_trace(go.Scatter(
                    x=date_history,
                    y=weights,
                    name=strategy,
                    line=dict(color=colors[i % len(colors)]),
                    mode='lines',
                    stackgroup='one'
                ))

        fig.update_layout(
            title=f"策略權重歷史 ({weight_method}, {rebalance_freq})",
            xaxis_title='日期',
            yaxis_title='權重',
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # 權重統計摘要
        st.subheader('權重統計')
        weight_stats = []
        for strategy in selected_strategies:
            if strategy in weight_history:
                weights = weight_history[strategy]
                weight_stats.append({
                    '策略': strategy,
                    '平均權重': f"{np.mean(weights):.3f}",
                    '最大權重': f"{np.max(weights):.3f}",
                    '最小權重': f"{np.min(weights):.3f}",
                    '權重標準差': f"{np.std(weights):.3f}"
                })

        if weight_stats:
            st.dataframe(pd.DataFrame(weight_stats))

    # 績效指標標籤頁
    with tabs[2]:
        st.subheader('績效指標總覽')

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric('總報酬率', f"{portfolio_analysis['total_return']:.2%}")

        with col2:
            st.metric("年化報酬率", f"{portfolio_analysis['annual_return']:.2%}")

        with col3:
            st.metric("最大回撤", f"{portfolio_analysis['max_drawdown']:.2%}")

        with col4:
            st.metric('夏普比率', f"{portfolio_analysis['sharpe_ratio']:.3f}")

        # 詳細績效指標
        st.subheader('詳細績效指標')

        # 計算進階績效指標
        portfolio_equity = portfolio_analysis['portfolio_equity']
        daily_returns = portfolio_equity.pct_change().dropna()

        # 索提諾比率（下行風險調整報酬）
        downside_returns = daily_returns[daily_returns < 0]
        downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = portfolio_analysis['annual_return'] / downside_risk if downside_risk > 0 else 0

        # 卡瑪比率（報酬 / 最大回撤）
        calmar_ratio = portfolio_analysis['annual_return'] / abs(portfolio_analysis['max_drawdown']) if portfolio_analysis['max_drawdown'] < 0 else 0

        # 正報酬月份比例
        monthly_returns = portfolio_equity.resample('M').last().pct_change().dropna()
        positive_months = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0

        detailed_metrics = {
            '指標': ['總報酬率', '年化報酬率', '年化波動率', '最大回撤', '夏普比率', '索提諾比率', '卡瑪比率', '正報酬月份比例'],
            '數值': [
                f"{portfolio_analysis['total_return']:.2%}",
                f"{portfolio_analysis['annual_return']:.2%}",
                f"{daily_returns.std() * np.sqrt(252):.2%}",
                f"{portfolio_analysis['max_drawdown']:.2%}",
                f"{portfolio_analysis['sharpe_ratio']:.3f}",
                f"{sortino_ratio:.3f}",
                f"{calmar_ratio:.3f}",
                f"{positive_months:.2%}"
            ]
        }

        st.dataframe(pd.DataFrame(detailed_metrics))

    # 權重檢查標籤頁
    with tabs[3]:
        st.subheader('權重檢查')

        # 日期選擇器
        if portfolio_analysis['date_history']:
            selected_date = st.selectbox(
                '選擇日期',
                portfolio_analysis['date_history'],
                index=len(portfolio_analysis['date_history']) - 1
            )

            # 創建權重圓餅圖
            st.subheader(f"Date: {selected_date} weights")

            date_index = portfolio_analysis['date_history'].index(selected_date)
            current_weights = {}

            for strategy in selected_strategies:
                if strategy in portfolio_analysis['weight_history']:
                    weight = portfolio_analysis['weight_history'][strategy][date_index]
                    current_weights[strategy] = weight

            # 創建權重圓餅圖
            fig = go.Figure(data=[go.Pie(
                labels=list(current_weights.keys()),
                values=list(current_weights.values()),
                hole=0.3
            )])

            fig.update_layout(
                title=f"投資組合權重分佈 ({selected_date})",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # 顯示權重表格
            weight_df = pd.DataFrame([
                {'策略': strategy, '權重': f"{weight:.3f}"}
                for strategy, weight in current_weights.items()
            ])

            st.dataframe(weight_df)

            # 個別策略權重趨勢
            st.subheader('權重趨勢')

            selected_strategies_for_trend = st.multiselect(
                '選擇要顯示的策略',
                selected_strategies,
                default=selected_strategies[:3]
            )

            if selected_strategies_for_trend:
                fig = go.Figure()

                colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                for i, strategy in enumerate(selected_strategies_for_trend):
                    if strategy in portfolio_analysis['weight_history']:
                        weights = portfolio_analysis['weight_history'][strategy]
                        fig.add_trace(go.Scatter(
                            x=portfolio_analysis['date_history'],
                            y=weights,
                            name=strategy,
                            line=dict(color=colors[i % len(colors)])
                        ))

                fig.update_layout(
                    title='個別策略權重趨勢',
                    xaxis_title='日期',
                    yaxis_title='權重',
                    template="plotly_white",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_app()
