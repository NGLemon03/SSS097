# -*- coding: utf-8 -*-
"""
sss_core/logic.py - SSS蝑???詨?閮????摩 (摰??靽桀儔??

?????????亥?蝞????皜研???璅??蝞???詨??賣?
1. 靽???????LeverageEngine ?舀???????蝔??頛?2. 靽桀儔 backtest_unified 銝剖??寞???NaN 撠??蝔??撏拇蔑???憿?"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
import yfinance as yf

from analysis import config as cfg
from analysis.logging_config import get_logger

logger = get_logger("SSS.Logic")

# 敹賜? pandas ??PerformanceWarning
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# 敺?config 撠??撣賊?
DATA_DIR = cfg.DATA_DIR
CACHE_DIR = cfg.CACHE_DIR
BASE_FEE_RATE = 0.001425
TAX_RATE = 0.003


# TradeSignal dataclass
@dataclass
class TradeSignal:
    ts: pd.Timestamp
    side: str
    reason: str


# --- ???撽?? ---
def validate_params(params: Dict, required_keys: set, positive_ints: set = None, positive_floats: set = None) -> bool:
    """
    撽??????臬?蝚血?閬??.
    """
    if not all(k in params for k in required_keys):
        logger.error(f"蝻箏?敹?????: {required_keys - set(params.keys())}")
        return False
    if positive_ints:
        for k in positive_ints:
            if k in params and (not isinstance(params[k], int) or params[k] <= 0):
                logger.error(f"??? {k} 敹???箸迤?湔?")
                return False
    if positive_floats:
        for k in positive_floats:
            if k in params and (not isinstance(params[k], (int, float)) or params[k] <= 0):
                logger.error(f"Invalid positive float parameter: {k}")
                return False
    return True


# --- ?豢??脣????頛?---
def fetch_yf_data(ticker: str, filename: Path, start_date: str = "2000-01-01", end_date: Optional[str] = None) -> None:
    """
    銝??銝虫?摮?Yahoo Finance ?豢?,瑼Ｘ??臬??箇??交??唳???
    """
    now_taipei = pd.Timestamp.now(tz='Asia/Taipei')
    update_midnight_taipei = now_taipei.normalize() + pd.Timedelta(days=1)
    file_exists = filename.exists()
    proceed_with_fetch = True

    try:
        pd.to_datetime(start_date, format='%Y-%m-%d')
        if end_date:
            pd.to_datetime(end_date, format='%Y-%m-%d')
    except ValueError as e:
        logger.error(f"?交??澆??航炊: {e}")
        return

    if file_exists:
        file_mod_time_taipei = pd.to_datetime(os.path.getmtime(filename), unit='s', utc=True).tz_convert('Asia/Taipei')
        # logger.info(f"?砍??豢? '{filename}' ??敺???? {file_mod_time_taipei.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if (file_mod_time_taipei.date() == now_taipei.date() and file_mod_time_taipei >= update_midnight_taipei) or \
           (now_taipei < update_midnight_taipei and file_mod_time_taipei >= (update_midnight_taipei - pd.Timedelta(days=1))):
            logger.info(f"Price data for '{ticker}' is already up to date.")
            proceed_with_fetch = False
        else:
            logger.warning(f"Price data for '{ticker}' is stale, fetching new data.")
    else:
        logger.warning(f"Price file '{filename}' not found, fetching data.")

    if not proceed_with_fetch:
        return

    try:
        df = yf.download(ticker, period='max', auto_adjust=True)
        if df.empty:
            raise ValueError("Downloaded data is empty.")
        df.to_csv(filename)
        logger.info(f"Saved data for '{ticker}' to '{filename}'.")
    except Exception as e:
        logger.error(f"霅血?: '{ticker}' ??活銝??憭望?: {e}")
        if file_exists:
            logger.info("Keeping existing local file as fallback.")
            return
        try:
            logger.info(f"??岫銝?? '{ticker}' ????湔風?脫???..")
            df = yf.download(ticker, period='max', auto_adjust=True)
            if df.empty:
                raise ValueError("Fallback fetch empty")
            df.to_csv(filename)
            logger.info(f"Fallback fetch succeeded for '{ticker}', saved to '{filename}'.")
        except Exception as e2:
            logger.error(f"?⊥?銝?? '{ticker}' ????? {e2}")
            if not file_exists:
                raise RuntimeError(f"Failed to fetch required data for {ticker}.")


def is_price_data_up_to_date(csv_path):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
        else:
            last_date = pd.to_datetime(df.iloc[-1, 0])
        today = pd.Timestamp.now(tz='Asia/Taipei').normalize()
        return last_date >= today
    except Exception:
        return False


def clear_all_caches():
    """Clear all local caches."""
    try:
        # 皜?? joblib 敹怠?
        cfg.MEMORY.clear()
        logger.info("撌脫???joblib 敹怠?")

        # 皜??敹怠??桅?
        for p in [CACHE_DIR / "cache_smaa", CACHE_DIR / "optuna16_equity"]:
            if p.exists():
                import shutil
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
        
        logger.info("撌脫???SMAA/Optuna 敹怠??桅?")
        return True
    except Exception as e:
        logger.error(f"皜??敹怠???????隤? {e}")
        return False


def force_update_price_data(ticker: str = None):
    """撘瑕??湔??∪??豢?"""
    try:
        if ticker:
            filename = DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"
            fetch_yf_data(ticker, filename, "2000-01-01")
            logger.info(f"撌脣撥?嗆???{ticker} ?∪??豢?")
        else:
            common_tickers = ["00631L.TW", "^TWII", "2414.TW", "2412.TW"]
            for t in common_tickers:
                filename = DATA_DIR / f"{t.replace(':','_')}_data_raw.csv"
                fetch_yf_data(t, filename, "2000-01-01")
            logger.info("Force update completed for common tickers.")
        return True
    except Exception as e:
        logger.error(f"撘瑕??湔??∪??豢???????隤? {e}")
        return False


@cfg.MEMORY.cache
def load_data(ticker: str, start_date: str = "2000-01-01", end_date: Optional[str] = None, smaa_source: str = "Self", force_update: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ???銝行???????舀????蝯???交????摮????
    """
    filename = DATA_DIR / f"{ticker.replace(':','_')}_data_raw.csv"

    # ?寞????瘙箏??臬??湔??∪?
    if force_update or not is_price_data_up_to_date(filename):
        fetch_yf_data(ticker, filename, start_date, end_date)

    if not filename.exists():
        logger.error(f"Price file missing and unable to fetch: '{filename}'")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_csv(filename, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
        df.name = ticker.replace(':', '_')
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
        df = df[~df.index.isna()]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['close'])
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        df_factor = pd.DataFrame()  # default for Self source
        if smaa_source in ["Factor (^TWII / 2412.TW)", "Factor (^TWII / 2414.TW)"]:
            twii_file = DATA_DIR / "^TWII_data_raw.csv"
            factor_ticker = "2412.TW" if smaa_source == "Factor (^TWII / 2412.TW)" else "2414.TW"
            factor_file = DATA_DIR / f"{factor_ticker.replace(':','_')}_data_raw.csv"
            
            if force_update or not is_price_data_up_to_date(twii_file):
                fetch_yf_data("^TWII", twii_file, start_date, end_date)
            if force_update or not is_price_data_up_to_date(factor_file):
                fetch_yf_data(factor_ticker, factor_file, start_date, end_date)

            if not twii_file.exists() or not factor_file.exists():
                logger.warning(f"Missing factor source files (^TWII or {factor_ticker}); fallback to Self source.")
                return df, pd.DataFrame()

            try:
                df_twii = pd.read_csv(twii_file, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
                df_factor_ticker = pd.read_csv(factor_file, parse_dates=[0], index_col=0, date_format='%Y-%m-%d')
                # 蝯曹??澆?
                for d in [df_twii, df_factor_ticker]:
                    d.columns = [c.lower().replace(' ', '_') for c in d.columns]
                    d.index = pd.to_datetime(d.index, format='%Y-%m-%d', errors='coerce')
                    d.dropna(subset=['close'], inplace=True)
                    d['close'] = pd.to_numeric(d['close'], errors='coerce')
                    d['volume'] = pd.to_numeric(d['volume'], errors='coerce')

                common_index = df_twii.index.intersection(df_factor_ticker.index).intersection(df.index)
                if len(common_index) < 100:
                    logger.warning(f"Insufficient overlapping factor history ({len(common_index)} rows); fallback to Self source.")
                    return df, pd.DataFrame()
                
                factor_price = (df_twii['close'].loc[common_index] / df_factor_ticker['close'].loc[common_index]).rename('close')
                factor_volume = df_factor_ticker['volume'].loc[common_index].rename('volume')
                df_factor = pd.DataFrame({'close': factor_price, 'volume': factor_volume})
                df_factor = df_factor.reindex(df.index).dropna()
                
                if end_date:
                    df_factor = df_factor[df_factor.index <= pd.to_datetime(end_date)]
            except Exception as e:
                logger.warning(f"Failed to build factor series: {e}; fallback to Self source.")
                return df, pd.DataFrame()

        df_factor.name = f"{ticker}_factor" if not df_factor.empty else None
        return df, df_factor
    except Exception as e:
        logger.error(f"霈????????豢???辣 '{filename}' ????? {e}")
        return pd.DataFrame(), pd.DataFrame()


def load_data_wrapper(ticker: str, start_date: str = "2000-01-01",
                      end_date: str | None = None,
                      smaa_source: str = "Self"):
    """?詨捆?亙?"""
    df_price, df_factor = load_data(ticker, start_date, end_date, smaa_source)
    return df_price, df_factor


# --- ???閮?????皜砍???---
def linreg_last_original(series: pd.Series, length: int) -> pd.Series:
    """?????linreg_last 撖衣?,????澆?瘥?葫閰?"""
    if len(series) < length or series.isnull().sum() > len(series) - length:
        return pd.Series(np.nan, index=series.index)
    return series.rolling(length, min_periods=length).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x)-1) + np.polyfit(np.arange(len(x)), x, 1)[1]
        if len(x[~np.isnan(x)]) == length else np.nan, raw=True)


def linreg_last_vectorized(series: np.ndarray, length: int) -> np.ndarray:
    """
    ??????蝞?遝????扯艘甇貊???敺??暺??????
    """
    # 頛詨?撽??
    series = np.asarray(series, dtype=float)
    if len(series) < length:
        return np.full(len(series), np.nan, dtype=float)

    # ???皛??蝒??
    windows = np.lib.stride_tricks.sliding_window_view(series, length)
    valid = ~np.any(np.isnan(windows), axis=1)

    # 瑽?遣 X ?拚?
    X = np.vstack([np.arange(length), np.ones(length)]).T
    
    # 閮?? (X^T X)^(-1) X^T
    try:
        XtX_inv_Xt = np.linalg.inv(X.T @ X) @ X.T
        # Compute linear-regression coefficients for each valid rolling window.
        coeffs = np.einsum('ij,kj->ki', XtX_inv_Xt, windows[valid])
        
        # Fill result for valid windows and keep NaN for invalid windows.
        result = np.full(len(windows), np.nan, dtype=float)
        result[valid] = coeffs[:, 0] * (length - 1) + coeffs[:, 1]
        
        # 憛怠???length-1 ??NaN
        output = np.full(len(series), np.nan, dtype=float)
        output[length-1:] = result
        return output
    except Exception:
        return np.full(len(series), np.nan, dtype=float)


@cfg.MEMORY.cache
def calc_smaa(series: pd.Series, linlen: int, factor: float, smaalen: int) -> pd.Series:
    """
    閮?? SMAA(?餉隅?Ｗ?敺??蝪∪?蝘餃?撟喳?).
    """
    series_values = series.values
    result = np.full(len(series), np.nan, dtype=float)

    min_required = max(linlen, smaalen)
    if len(series) < min_required:
        return pd.Series(result, index=series.index)

    # Vectorized linear-regression last value.
    lr = linreg_last_vectorized(series_values, linlen)

    # ?餉隅?Ｗ?
    detr = (series_values - lr) * factor

    # 閮?? SMA
    if len(detr) >= smaalen:
        sma = np.convolve(detr, np.ones(smaalen)/smaalen, mode='valid')
        result[smaalen-1:] = sma
    
    return pd.Series(result, index=series.index)


@cfg.MEMORY.cache
def compute_single(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, devwin: int, smaa_source: str = "Self") -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)

    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    base = smaa.ewm(alpha=1/devwin, adjust=False, min_periods=devwin).mean()
    sd = (smaa - base).abs().ewm(alpha=1/devwin, adjust=False, min_periods=devwin).mean()

    results_df = pd.DataFrame({
        'smaa': smaa,
        'base': base,
        'sd': sd
    }, index=df_cleaned.index)
    final_df = pd.concat([df[['open', 'high', 'low', 'close']], results_df], axis=1, join='inner')
    final_df = final_df.dropna()
    return final_df


@cfg.MEMORY.cache
def compute_dual(df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float, smaalen: int, short_win: int, long_win: int, smaa_source: str = "Self") -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)

    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

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
    final_df = final_df.dropna()
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
    smaa_source: str = "Self"
) -> pd.DataFrame:
    source_df = smaa_source_df if not smaa_source_df.empty else df
    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)

    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    base = smaa.ewm(alpha=1/rma_len, adjust=False, min_periods=rma_len).mean()
    sd = smaa.rolling(window=dev_len, min_periods=dev_len).std()

    results = pd.DataFrame({
        'smaa': smaa,
        'base': base,
        'sd':   sd
    }, index=df_cleaned.index)
    final = pd.concat([df[['open','high','low','close']], results], axis=1, join='inner')
    final = final.dropna()
    return final


@cfg.MEMORY.cache
def compute_ssma_turn_combined(
    df: pd.DataFrame, smaa_source_df: pd.DataFrame, linlen: int, factor: float,
    smaalen: int, prom_factor: float, min_dist: int, buy_shift: int = 0, exit_shift: int = 0, vol_window: int = 20,
    signal_cooldown_days: int = 10, quantile_win: int = 100,
    smaa_source: str = "Self"
) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    logger.info(f"閮?? ssma_turn: linlen={linlen}, smaalen={smaalen}")

    # ???撽??
    try:
        linlen = int(linlen)
        smaalen = int(smaalen)
        min_dist = int(min_dist)
        vol_window = int(vol_window)
        quantile_win = max(int(quantile_win), vol_window)
        signal_cooldown_days = int(signal_cooldown_days)
        buy_shift = max(0, int(buy_shift)) # 蝣箔????
        exit_shift = max(0, int(exit_shift))
    except (ValueError, TypeError) as e:
        logger.error(f"???憿???⊥?: {e}")
        return pd.DataFrame(), [], []

    source_df = smaa_source_df if not smaa_source_df.empty else df
    if 'close' not in source_df.columns or 'volume' not in source_df.columns:
        return pd.DataFrame(), [], []

    df_cleaned = source_df.dropna(subset=['close'])
    df_cleaned['close'] = df_cleaned['close'].round(6)
    
    smaa = calc_smaa(df_cleaned['close'], linlen, factor, smaalen)

    series_clean = smaa.dropna()
    if series_clean.empty:
        return pd.DataFrame(), [], []

    # Rolling prominence baseline for dynamic thresholding.
    prom = series_clean.rolling(window=min_dist+1, min_periods=min_dist+1).apply(lambda x: x.ptp(), raw=True)
    initial_threshold = prom.quantile(prom_factor / 100) if len(prom.dropna()) > 0 else prom.median()
    threshold_series = prom.rolling(window=quantile_win, min_periods=quantile_win).quantile(prom_factor / 100).shift(1).ffill().fillna(initial_threshold)

    # 撜航健瑼Ｘ葫
    peaks = []
    valleys = []
    last_signal_dt = None

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
                if peak_date not in peaks:
                    peaks.append(peak_date)
        if window_valleys.size > 0:
            for v_idx in window_valleys:
                valley_date = series_clean.index[window_start_idx + v_idx]
                if valley_date not in valleys:
                    valleys.append(valley_date)

    # Use prior-bar rolling volume average as filter baseline.
    vol_ma = df['volume'].rolling(vol_window, min_periods=vol_window).mean().shift(1)
    
    def filter_with_volume(dates, vol_series, price_df):
        valid = []
        for d in dates:
            if d in vol_series.index and d in price_df.index:
                if price_df.loc[d, 'volume'] > vol_series.loc[d]:
                    valid.append(d)
        return valid

    valid_peaks = filter_with_volume(peaks, vol_ma, df)
    valid_valleys = filter_with_volume(valleys, vol_ma, df)

    # Apply signal cooldown in days.
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

    # 鞎瑁都靽∟? (Shift)
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
    return df_ind.dropna(), buy_dates, sell_dates


def calculate_trade_mmds(trades: List[Tuple[pd.Timestamp, float, pd.Timestamp]], equity_curve: pd.Series) -> List[float]:
    """Calculate per-trade maximum drawdowns from equity curve."""
    mmds = []
    for entry_date, _, exit_date in trades:
        period_equity = equity_curve.loc[entry_date:exit_date]
        if len(period_equity) < 2:
            mmds.append(0.0)
            continue
        roll_max = period_equity.cummax()
        drawdown = period_equity / roll_max - 1
        mmds.append(drawdown.min())
    return mmds


def calculate_metrics(trades: List[Tuple[pd.Timestamp, float, pd.Timestamp]], df_ind: pd.DataFrame, equity_curve: pd.Series = None) -> Dict:
    """閮????葫蝮暹????"""
    metrics = {
        'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 
        'max_drawdown_duration': 0, 'calmar_ratio': np.nan,
        'num_trades': 0, 'win_rate': 0.0, 'avg_win': np.nan, 'avg_loss': np.nan,
        'payoff_ratio': np.nan, 'sharpe_ratio': np.nan, 'sortino_ratio': np.nan,
        'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
        'annualized_volatility': np.nan, 'profit_factor': np.nan,
    }
    if not trades:
        return metrics

    trade_metrics = pd.DataFrame(trades, columns=['entry_date', 'return', 'exit_date']).set_index('exit_date')
    trade_metrics['equity'] = (1 + trade_metrics['return']).cumprod()
    
    # 蝣箔? equity_curve 摮??
    if equity_curve is None:
        equity_curve = pd.Series(1, index=df_ind.index)

    # ?箸????
    metrics['total_return'] = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 1)
    metrics['annual_return'] = (1 + metrics['total_return']) ** (1 / years) - 1

    # DD
    roll_max = equity_curve.cummax()
    daily_drawdown = equity_curve / roll_max - 1
    metrics['max_drawdown'] = float(daily_drawdown.min())
    
    # DD Duration
    dd_bool = daily_drawdown < 0
    blocks = (~dd_bool).cumsum()
    dd_dur = int((dd_bool.groupby(blocks).cumcount() + 1).where(dd_bool).max() or 0)
    metrics['max_drawdown_duration'] = dd_dur

    metrics['num_trades'] = len(trade_metrics)
    metrics['win_rate'] = (trade_metrics['return'] > 0).sum() / metrics['num_trades'] if metrics['num_trades'] > 0 else 0
    
    wins = trade_metrics[trade_metrics['return'] > 0]['return']
    losses = trade_metrics[trade_metrics['return'] < 0]['return']
    metrics['avg_win'] = wins.mean() if not wins.empty else 0.0
    metrics['avg_loss'] = losses.mean() if not losses.empty else 0.0
    
    if metrics['avg_loss'] != 0:
        metrics['payoff_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
    else:
        metrics['payoff_ratio'] = np.nan

    # Sharpe & Sortino
    daily_returns = equity_curve.pct_change().dropna()
    if not daily_returns.empty and daily_returns.std() != 0:
        metrics['sharpe_ratio'] = (daily_returns.mean() * np.sqrt(252)) / daily_returns.std()
        metrics['annualized_volatility'] = daily_returns.std() * np.sqrt(252)
        
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std()
        if downside_std != 0:
            metrics['sortino_ratio'] = (daily_returns.mean() * np.sqrt(252)) / downside_std

    # Profit Factor
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else np.nan

    # MMD Override (Optional)
    mmds = calculate_trade_mmds(trades, equity_curve)
    if mmds:
        # ??ㄐ?豢?靽???亦?蝝?????憭批??歹????摰??璅??
        pass

    return metrics


def calculate_holding_periods(trade_df: pd.DataFrame) -> float:
    """Calculate average holding period in calendar days."""
    if trade_df.empty or 'trade_date' not in trade_df.columns or 'type' not in trade_df.columns:
        return np.nan

    holding_periods = []
    entry_date = None

    for _, row in trade_df.iterrows():
        if row['type'] == 'buy':
            entry_date = row['trade_date']
        elif row['type'] in ['sell', 'sell_forced'] and entry_date is not None:
            exit_date = row['trade_date']
            holding_days = (exit_date - entry_date).days
            holding_periods.append(holding_days)
            entry_date = None

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
    """
    ?瑁???葫銝西???底蝝啁??????
    靽桀儔鈭??隤斤????????澆?銝???渡????.
    銝虫????鈭??撠?NaN ?寞?????????
    """
    # Standard empty result payload for early-return branches.
    EMPTY_RESULT = {
        'trades': [],
        'trade_df': pd.DataFrame(),
        'trades_df': pd.DataFrame(),
        'signals_df': pd.DataFrame(),
        'metrics': {
            'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 
            'sharpe_ratio': 0.0, 'num_trades': 0
        },
        'equity_curve': pd.Series(dtype=float),
        'daily_state': pd.DataFrame(),
        'weight_curve': pd.Series(dtype=float)
    }

    if not isinstance(df_ind, pd.DataFrame):
        logger.error(f"df_ind 敹???臭???pandas.DataFrame嚗???喳? {type(df_ind)}")
        return EMPTY_RESULT.copy()

    # 鞎餌?閮剖?
    BUY_FEE_RATE = BASE_FEE_RATE * discount
    SELL_FEE_RATE = BASE_FEE_RATE * discount + TAX_RATE
    ROUND_TRIP_FEE = BUY_FEE_RATE + SELL_FEE_RATE

    # 瑽?▼撘??
    lev = None
    if use_leverage:
        try:
            from leverage import LeverageEngine
            lev = LeverageEngine(**(lev_params or {}))
        except ImportError:
            logger.warning("Leverage module unavailable; running without leverage.")
            use_leverage = False

    # ?豢?瑼Ｘ?
    required_cols = ['open', 'close'] if strategy_type == 'ssma_turn' else ['open', 'close', 'smaa', 'base', 'sd']
    if df_ind.empty or not all(col in df_ind.columns for col in required_cols):
        logger.warning(f"????豢?銝???湛??⊥??瑁???葫(蝻箏?甈??: {set(required_cols) - set(df_ind.columns)})")
        return EMPTY_RESULT.copy()

    # ???撽????????
    try:
        trade_cooldown_bars = int(trade_cooldown_bars)
        if trade_cooldown_bars < 0:
            raise ValueError("trade_cooldown_bars must be >= 0")
        
        params['stop_loss'] = float(params.get('stop_loss', 0.0))
        if bad_holding and params['stop_loss'] <= 0:
            raise ValueError("When bad_holding=True, stop_loss must be > 0")
        
        if strategy_type == 'ssma_turn':
            # Clamp shift parameters
            params['exit_shift'] = max(0, int(params.get('exit_shift', 0)))
            params['buy_shift'] = max(0, int(params.get('buy_shift', 0)))
        else:
            params['buy_mult'] = float(params.get('buy_mult', 0.5))
            params['sell_mult'] = float(params.get('sell_mult', 0.5))
    except (ValueError, TypeError) as e:
        logger.error(f"???撽??憭望?: {e}")
        return EMPTY_RESULT.copy()

    # --- ??葫?????---
    initial_cash = 100000.0
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
    
    # State series tracked per bar.
    equity_curve = pd.Series(initial_cash, index=df_ind.index, dtype=float)
    cash_series = pd.Series(initial_cash, index=df_ind.index, dtype=float)
    shares_series = pd.Series(0, index=df_ind.index, dtype=float)

    # ???靽∟???”
    signals_list = []
    if strategy_type == 'ssma_turn':
        buy_dates = sorted(buy_dates or [])
        sell_dates = sorted(sell_dates or [])
        for dt in buy_dates:
            signals_list.append(TradeSignal(ts=dt, side="BUY", reason="ssma_turn_valley"))
        for dt in sell_dates:
            signals_list.append(TradeSignal(ts=dt, side="SELL", reason="ssma_turn_peak"))
    else:
        # ?????縑?????(敹怠?憭?
        smaa = df_ind['smaa']
        base = df_ind['base']
        sd = df_ind['sd']
        buy_mult = params['buy_mult']
        sell_mult = params['sell_mult']
        
        buy_cond = smaa < (base + sd * buy_mult)
        sell_cond = smaa > (base + sd * sell_mult)
        
        buy_indices = df_ind.index[buy_cond]
        sell_indices = df_ind.index[sell_cond]
        
        for dt in buy_indices:
            signals_list.append(TradeSignal(ts=dt, side="BUY", reason=f"{strategy_type}_buy"))
        for dt in sell_indices:
            signals_list.append(TradeSignal(ts=dt, side="SELL", reason=f"{strategy_type}_sell"))

    signals_list.sort(key=lambda x: x.ts)

    # 撠?縑???撠???交?
    n = len(df_ind)
    scheduled_buy = np.zeros(n, dtype=bool)
    scheduled_sell = np.zeros(n, dtype=bool)
    scheduled_forced = np.zeros(n, dtype=bool)
    idx_by_date = {date: i for i, date in enumerate(df_ind.index)}

    for sig in signals_list:
        ts = pd.Timestamp(sig.ts).tz_localize(None) if sig.ts.tzinfo else sig.ts
        if ts in idx_by_date:
            i = idx_by_date[ts]
            # 靽∟???虜?冽??文??Ｙ?嚗??甈⊥??瑁?
            if i + 1 < n:
                if sig.side == "BUY":
                    scheduled_buy[i + 1] = True
                elif sig.side in ["SELL", "STOP_LOSS", "FORCE_SELL"]:
                    if sig.side == "SELL":
                        scheduled_sell[i + 1] = True
                    else:
                        scheduled_forced[i + 1] = True

    # --- ?????葫敺芰? ---
    opens = df_ind['open'].values
    closes = df_ind['close'].values
    dates = df_ind.index

    for i in range(n):
        today = dates[i]
        today_open = opens[i]
        today_close = closes[i]

        # --- FIX: guard against NaN prices ---
        if np.isnan(today_open) or np.isnan(today_close):
            cash_series.iloc[i] = cash
            shares_series.iloc[i] = total_shares
            # 瘝輻?銝???亦? Equity嚗??????????NaN
            if i > 0:
                equity_curve.iloc[i] = equity_curve.iloc[i-1]
            else:
                equity_curve.iloc[i] = initial_cash
            continue
        # --- FIX END ---

        mkt_val = total_shares * today_close

        # 瑽?▼?拇???雁???瑼Ｘ?
        if use_leverage and in_pos and lev:
            interest = lev.accrue()
            cash -= interest
            accum_interest += interest
            forced = lev.margin_call(mkt_val=mkt_val)
            if forced > 0 and i + 1 < n:
                scheduled_forced[i + 1] = True

        # Persist daily states.
        cash_series.iloc[i] = cash
        shares_series.iloc[i] = total_shares
        equity_curve.iloc[i] = cash + total_shares * today_close

        # --- 鞈????摩 ---
        if (scheduled_sell[i] or scheduled_forced[i]) and in_pos and total_shares > 0:
            exit_price = today_open # ??身???鞈??
            exit_date = today
            
            # Spread accumulated financing cost across current shares.
            interest_per_share = (accum_interest / total_shares) if total_shares > 0 else 0
            trade_ret = (exit_price / entry_price) - 1 - ROUND_TRIP_FEE - (interest_per_share / entry_price)
            
            # Bad Holding ??蕪 (憒???扳?憭芸?銝???臬撥?嗅像??????蝑???豢?蝜潛????)
            if bad_holding and trade_ret < -0.20 and not scheduled_forced[i]:
                pass # Skip sell, hold
            else:
                # ?瑁?鞈??
                cash += total_shares * exit_price
                sell_shares = total_shares
                total_shares = 0
                
                # ???瑽?▼
                if use_leverage and lev and lev.loan > 0:
                    repay_amt = min(cash, lev.loan)
                    lev.repay(repay_amt)
                    cash -= repay_amt
                    trade_records.append({
                        'signal_date': today, 'trade_date': exit_date, 'type': 'repay',
                        'price': 0.0, 'loan_amount': repay_amt, 'reason': 'repay_loan'
                    })

                trades.append((entry_date, trade_ret, exit_date))
                trade_records.append({
                    'signal_date': today, 'trade_date': exit_date,
                    'type': 'sell' if scheduled_sell[i] else 'sell_forced',
                    'price': exit_price,
                    'shares': sell_shares,
                    'return': trade_ret,
                    'reason': 'signal_sell' if scheduled_sell[i] else 'signal_forced'
                })
                signals.append({
                    'signal_date': today,
                    'type': 'sell',
                    'price': today_close,
                    'reason': 'signal_sell' if scheduled_sell[i] else 'signal_forced'
                })
                
                in_pos = False
                last_trade_idx = i
                accum_interest = 0.0
                continue

        # --- 鞎瑕???摩 ---
        if scheduled_buy[i] and not in_pos and (i - last_trade_idx > trade_cooldown_bars):
            # 閮???航眺?⊥?
            cost_per_share = today_open
            
            # FIX: 蝣箔? cost_per_share ???銝?之??0嚗?????隞?NaN ??0
            if not np.isnan(cost_per_share) and cost_per_share > 0:
                shares = int(cash // cost_per_share)
                
                if shares > 0:
                    need_cash = shares * cost_per_share
                    
                    # 瑽?▼??狡
                    if use_leverage and lev:
                        gap = need_cash - cash
                        if gap > 0:
                            borrowable = lev.avail(mkt_val=mkt_val) # ??ㄐ mkt_val ?嗅祕??0
                            draw = min(gap, borrowable)
                            if draw > 0:
                                lev.borrow(draw)
                                cash += draw
                    
                    # ?瑁?鞎瑕?
                    if cash >= need_cash:
                        cash -= need_cash
                        total_shares = shares
                        entry_price = today_open
                        entry_date = today
                        in_pos = True
                        last_trade_idx = i
                        accum_interest = 0.0
                        
                        trade_records.append({
                            'signal_date': today, 'trade_date': entry_date,
                            'type': 'buy', 'price': entry_price, 'shares': shares, 'reason': 'signal_buy'
                        })
                        signals.append({'signal_date': today, 'type': 'buy', 'price': today_close, 'reason': 'signal_buy'})
            continue

        # --- ???瑼Ｘ? (Bad Holding Stop Loss) ---
        if bad_holding and in_pos and entry_price > 0 and i + 1 < n:
            curr_ret = today_close / entry_price - 1
            if curr_ret <= -params['stop_loss']:
                scheduled_forced[i + 1] = True

    # Keep open position at period end; do not create synthetic end_of_period sell_forced.

    # --- ?渡?蝯?? ---
    trade_df = pd.DataFrame(trade_records)
    trades_df = pd.DataFrame(trades, columns=['entry_date', 'ret', 'exit_date'])
    signals_df = pd.DataFrame(signals)
    
    metrics = calculate_metrics(trades, df_ind, equity_curve)

    # ???摰??????” (蝣箔??喃蝙瘝??鈭斗?銋???豢?)
    daily_state = pd.DataFrame({
        'equity': equity_curve,
        'cash': cash_series,
        'shares': shares_series
    })
    
    # 閮?????甈??
    daily_state['w'] = (daily_state['equity'] - daily_state['cash']) / daily_state['equity']
    daily_state['w'] = daily_state['w'].fillna(0).clip(0, 1)

    logger.info(f"{strategy_type} ??葫蝯??: 蝮賢??祉? = {metrics.get('total_return', 0):.2%}, 鈭斗?甈⊥?={metrics.get('num_trades', 0)}")
    
    return {
        'trades': trades,
        'trade_df': trade_df,
        'trades_df': trades_df,
        'signals_df': signals_df,
        'metrics': metrics,
        'equity_curve': equity_curve,
        'daily_state': daily_state,  
        'weight_curve': daily_state['w']
    }
