# analysis/data_loader.py
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict, Optional
import SSSv096 as SSS
import os
import numpy as np
import yfinance as yf
from datetime import datetime
import logging

from analysis import config as cfg

logger = logging.getLogger("optuna_data_loader")

def load_data(ticker: str,
              start_date: str = "2000-01-01",
              end_date: str | None = None,
              smaa_source: str = "Self") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    直接調用 SSSv096.py 中的 load_data 函數，以確保數據載入邏輯完全一致。
    放棄本地的快取機制，因為它破壞了因子和價格數據的時間軸同步。
    """
    return SSS.load_data(ticker, start_date, end_date, smaa_source)

def filter_periods_by_data(df_price: pd.DataFrame, periods: List[Dict]) -> List[Dict]:
    """
    根據數據的實際日期範圍過濾分析期間。
    """
    if df_price.empty:
        return []
    first, last = df_price.index.min(), df_price.index.max()
    return [p for p in periods if pd.to_datetime(p["start"]) >= first and pd.to_datetime(p["end"]) <= last]

def load_data_for_optuna(ticker: str, start_date: str = "2000-01-01", 
                         end_date: Optional[str] = None, 
                         smaa_source: str = "Self") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    專門用於 Optuna 的數據加載函數，不會自動更新股價數據
    
    Args:
        ticker: 股票代號
        start_date: 起始日期
        end_date: 結束日期
        smaa_source: SMAA數據源
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (價格數據, 因子數據)
    """
    filename = cfg.DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
    
    if not filename.exists():
        logger.error(f"數據文件 '{filename}' 不存在.請先使用 SSS/dash 下載數據.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_csv(filename, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
        df.name = ticker.replace(':', '_')
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
        df = df[~df.index.isna()]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                logger.warning(f"警告:數據中缺少 '{col}' 欄位,將以 NaN 填充.")
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['close'])
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        df_factor = pd.DataFrame()  # 預設空因子數據
        if smaa_source in ["Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]:
            twii_file = cfg.DATA_DIR / "^TWII_data_raw.csv"
            factor_ticker = "2412.TW" if smaa_source == "Factor (^TWII / 2412.TW)" else "2414.TW"
            factor_file = cfg.DATA_DIR / f"{factor_ticker.replace(':','_')}_data_raw.csv"
            
            if not twii_file.exists() or not factor_file.exists():
                logger.warning(f"無法載入因子數據 (^TWII 或 {factor_ticker}),回退到 Self 模式.")
                return df, pd.DataFrame()
            
            try:
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
                    logger.warning(f"因子數據與價格數據的共同交易日不足 ({len(common_index)} 天),回退到 Self 模式.")
                    return df, pd.DataFrame()
                factor_price = (df_twii['close'].loc[common_index] / df_factor_ticker['close'].loc[common_index]).rename('close')
                factor_volume = df_factor_ticker['volume'].loc[common_index].rename('volume')
                df_factor = pd.DataFrame({'close': factor_price, 'volume': factor_volume})
                df_factor = df_factor.reindex(df.index).dropna()
                if end_date:
                    df_factor = df_factor[df_factor.index <= pd.to_datetime(end_date)]
            except Exception as e:
                logger.warning(f"處理因子數據時出錯: {e},回退到 Self 模式.")
                return df, pd.DataFrame()
        
        df_factor.name = f"{ticker}_factor" if not df_factor.empty else None
        return df, df_factor
    except Exception as e:
        logger.error(f"讀取或處理數據文件 '{filename}' 時出錯: {e}")
        return pd.DataFrame(), pd.DataFrame()

def check_data_availability(ticker: str, smaa_source: str = "Self") -> bool:
    """
    檢查數據是否可用
    
    Args:
        ticker: 股票代號
        smaa_source: SMAA數據源
    
    Returns:
        bool: 數據是否可用
    """
    filename = cfg.DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
    
    if not filename.exists():
        return False
    
    if smaa_source in ["Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]:
        twii_file = cfg.DATA_DIR / "^TWII_data_raw.csv"
        factor_ticker = "2412.TW" if smaa_source == "Factor (^TWII / 2412.TW)" else "2414.TW"
        factor_file = cfg.DATA_DIR / f"{factor_ticker.replace(':','_')}_data_raw.csv"
        
        if not twii_file.exists() or not factor_file.exists():
            return False
    
    return True

def get_data_info(ticker: str) -> dict:
    """
    獲取數據文件信息
    
    Args:
        ticker: 股票代號
    
    Returns:
        dict: 數據信息
    """
    filename = cfg.DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
    
    if not filename.exists():
        return {"exists": False, "message": "文件不存在"}
    
    try:
        df = pd.read_csv(filename, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
        file_size = filename.stat().st_size / 1024  # KB
        mod_time = datetime.fromtimestamp(filename.stat().st_mtime)
        
        return {
            "exists": True,
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": f"{df.index.min()} 到 {df.index.max()}",
            "file_size_kb": round(file_size, 2),
            "last_modified": mod_time.strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"exists": False, "message": f"讀取錯誤: {e}"}