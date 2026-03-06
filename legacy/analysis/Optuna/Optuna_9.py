'''
Optuna-based hyper-parameter optimization for 00631L strategies (Version 8, adjusted)
--------------------------------------------------
* 最佳化目標: 優先追求完整回測報酬(>300%)、交易次數(15-60次),允許輕微過擬合,其次考慮 walk_forward 期間最差報酬、夏普比率、最大回撤,最後考慮壓力測試平均報酬。
* 本腳本可直接放在 analysis/ 目錄後以 `python Optuna_9.py` 執行。
* 搜尋空間與權重在 `PARAM_SPACE` 與 `SCORE_WEIGHTS` 中設定,方便日後微調。
* 使用 sklearn 的 TimeSeriesSplit 替代 mlfinlab 的 CPCV, 自定義 PBO 函數, 確保策略在 OOS 期間穩定表現, 並分析交易時機。
v4a-v5a 修正日誌衝突：移除全局日誌處理器覆蓋，採用獨立日誌模組 (`logging_config`)。
v5a-v6 修正數據載入問題：統一預載數據，避免重複加載導致 SMAA 快取不一致。
v6-v7 修正壓力測試期間：新增交易日檢查，確保 `valid_stress_periods` 非空，防止無效計算。
v7-v71 修正回測返回值：確保 `_backtest_once` 一致返回 6 個值 (total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades)，避免解包錯誤。
v71-v9 修正 Equity Curve 缺失：新增 `equity_curve` 傳遞與驗證，修復 `compute_stress_mdd_equity` 與 `compute_excess_return_in_stress_equity` 計算。
v9 fix01 加入多散點圖, 改用 Excess Return in Stress 替代 fail_ratio, 實行 Robust Score。
v9 fix02 修正錯誤處理, 調整散點圖尺度, 改進 Stress MDD 計算。
* 命令列範例：
  python optuna_9.py --strategy RMA --n_trials 10000
  python optuna_9.py --strategy single --n_trials 10000
  python optuna_9.py --strategy dual --n_trials 10000
  python optuna_9.py --strategy ssma_turn --n_trials 10000
'''

import logging
from logging_config import setup_logging
import optuna
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import argparse
from pathlib import Path
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

setlimit = False
setminsharpe = 0.45
setmaxsharpe = 0.75
setminmdd = -0.2
setmaxmdd = -0.4
setup_logging()
logger = logging.getLogger("optuna_9")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import config as cfg
from analysis import data_loader
import SSSv096 as SSS

parser = argparse.ArgumentParser(description='Optuna 最佳化 00631L 策略')
parser.add_argument('--strategy', type=str, choices=['single', 'dual', 'RMA', 'ssma_turn', 'all'], default='all', help='指定單一策略進行最佳化 (預設: all)')
parser.add_argument('--n_trials', type=int, default=5000, help='試驗次數 (預設: 5000)')
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
results_log = []
events_log = []

def log_to_results(event_type, details, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    record = {"Timestamp": timestamp, "Event Type": event_type, "Details": details, **kwargs}


    if event_type == "試驗結果":
        results_log.append(record)
    else:
        events_log.append(record)

def load_results_to_list(csv_file: Path) -> list:
    if not csv_file.exists():
        logger.warning(f"結果檔案 {csv_file} 不存在, 無法轉回列表")
        return []
    df_results = pd.read_csv(csv_file)
    return df_results.to_dict('records')

TICKER = cfg.TICKER
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
COOLDOWN_BARS = cfg.TRADE_COOLDOWN_BARS
WF_PERIODS = [(p["test"][0], p["test"][1]) for p in cfg.WF_PERIODS]
STRESS_PERIODS = cfg.STRESS_PERIODS
STRAT_FUNC_MAP = {'single': SSS.compute_single, 'dual': SSS.compute_dual, 'RMA': SSS.compute_RMA, 'ssma_turn': SSS.compute_ssma_turn_combined}
DATA_SOURCES = cfg.SOURCES
DATA_SOURCES_WEIGHTS = {'Self': 1/3, 'Factor (^TWII / 2412.TW)': 1/3, 'Factor (^TWII / 2414.TW)': 1/3}
STRATEGY_WEIGHTS = {'single': 0.25, 'dual': 0.25, 'RMA': 0.25, 'ssma_turn': 0.25}
MIN_NUM_TRADES = 10
MAX_NUM_TRADES = 120
CPCV_NUM_SPLITS = 7
CPCV_EMBARGO_DAYS = 15
min_splits = 3

PARAM_SPACE = {
    "single": dict(linlen=(5, 240, 1), smaalen=(7, 240, 5), devwin=(5, 180, 1), factor=(20.0, 60.0, 1.0), buy_mult=(0.1, 1.2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.50, 0.05)),
    "dual": dict(linlen=(5, 240, 1), smaalen=(7, 240, 5), short_win=(10, 100, 5), long_win=(40, 240, 10), factor=(20.0, 100.0, 5.0), buy_mult=(0.2, 2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.50, 0.05)),
    "RMA": dict(linlen=(5, 60, 1), smaalen=(7, 150, 5), rma_len=(20, 80, 1), dev_len=(5, 50, 1), factor=(10.0, 60.0, 5.0), buy_mult=(0.2, 1, 0.01), sell_mult=(0.2, 2.0, 0.01), stop_loss=(0.00, 0.5, 0.05)),
    "ssma_turn": dict(linlen=(10, 120, 5), smaalen=(10, 150, 5), factor=(10.0, 80.0, 10.0), prom_factor=(5, 70, 1), min_dist=(1, 12, 1), buy_shift=(0, 7, 1), exit_shift=(0, 7, 1), vol_window=(5, 90, 5), quantile_win=(5, 180, 10), signal_cooldown_days=(1, 10, 1), buy_mult=(0.1, 2, 0.05), sell_mult=(0.1, 3, 0.1), stop_loss=(0.00, 0.50, 0.05)),
}

IND_BT_KEYS = {strat: cfg.STRATEGY_PARAMS[strat]["ind_keys"] + cfg.STRATEGY_PARAMS[strat]["bt_keys"] for strat in PARAM_SPACE}
SCORE_WEIGHTS = dict(total_return=2.5, profit_factor=0.2, wf_min_return=0.2, sharpe_ratio=0.2, max_drawdown=0.1)

def _sample_params(trial: optuna.Trial, strat: str) -> dict:
    space = PARAM_SPACE[strat]
    params = {}
    for k, v in space.items():
        if isinstance(v[0], int):
            low, high, step = v
            params[k] = trial.suggest_int(k, low, high, step=step)
        else:
            low, high, step = v
            params[k] = round(trial.suggest_float(k, low, high, step=step), 3)
    return params

def minmax(x, lo, hi, clip=True):
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1) if clip else y

def buy_and_hold_return(df_price: pd.DataFrame, start: str, end: str) -> float:
    try:
        if df_price.empty or 'close' not in df_price.columns:
            logger.error("df_price 無效")
            return 1.0
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start not in df_price.index or end not in df_price.index:
            logger.error(f"start/end 不在交易日序列: {start} → {end}")
            return 1.0
        start_p, end_p = df_price.at[start, 'close'], df_price.at[end, 'close']
        if pd.isna(start_p) or pd.isna(end_p) or start_p == 0:
            logger.warning(f"價格缺失或為 0: {start_p}, {end_p}")
            return 1.0
        return end_p / start_p
    except Exception as e:
        logger.error(f"買入並持有計算錯誤：{e}")
        return 1.0

def get_fold_period(test_blocks: list) -> tuple:
    starts = [b[0] for b in test_blocks]
    ends = [b[1] for b in test_blocks]
    return min(starts), max(ends)

def analyze_trade_timing(df_price, trades, window=20):
    buy_distances = []
    sell_distances = []
    for t in trades:
        entry, _, exit = t[0], t[1], t[2] if len(t) > 2 else (None, None, None)
        if not entry or not exit or entry not in df_price.index or exit not in df_price.index:
            logger.warning(f"無效交易日期：entry={entry}, exit={exit}")
            continue
        window_data = df_price.loc[:entry, 'close'].tail(window)
        low_idx = window_data.idxmin()
        buy_dist = (pd.Timestamp(entry) - pd.Timestamp(low_idx)).days if low_idx else 0
        buy_distances.append(buy_dist)
        window_data = df_price.loc[:exit, 'close'].tail(window)
        high_idx = window_data.idxmax()
        sell_dist = (pd.Timestamp(exit) - pd.Timestamp(high_idx)).days if high_idx else 0
        sell_distances.append(sell_dist)
    return np.mean(buy_distances) if buy_distances else 20.0, np.mean(sell_distances) if sell_distances else 20.0

def compute_simplified_sra(df_price, trades, test_blocks):
    try:
        strategy_returns = []
        for t in trades:
            if len(t) > 2 and t[0] in df_price.index and t[2] in df_price.index:
                period = df_price.loc[t[0]:t[2], 'close'].pct_change().dropna()
                strategy_returns.extend(period)
        test_start, test_end = get_fold_period(test_blocks)
        bh_returns = df_price.loc[test_start:test_end, 'close'].pct_change().dropna()
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0.0
        bh_sharpe = np.mean(bh_returns) / np.std(bh_returns) * np.sqrt(252) if np.std(bh_returns) > 0 else 0.0
        t_stat, p_value = ttest_ind(strategy_returns, bh_returns, equal_var=False)
        return strategy_sharpe, bh_sharpe, p_value
    except Exception as e:
        logger.warning(f"簡化 SRA 計算失敗: {e}")
        return 0.0, 0.0, 1.0

def compute_knn_stability(df_results: list, params: list, k: int = 5, metric: str = 'total_return') -> float:
    if len(df_results) < k + 1:
        logger.warning(f"試驗數量 {len(df_results)} 不足以計算 KNN 穩定性")
        return 0.0
    param_cols = [p for p in params if p in df_results[0]]
    if not param_cols:
        logger.warning(f"無有效參數用於 KNN 穩定性計算,參數: {params}")
        return 0.0
    X = np.array([[r[p] for p in param_cols] for r in df_results])
    y = np.array([r[metric] for r in df_results])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    stability_scores = []
    for i, idx in enumerate(indices):
        roi = y[i]
        roi_neighbors = np.mean(y[idx[1:]])
        diff = min(abs(roi - roi_neighbors), 2 * roi_neighbors if roi_neighbors > 0 else 2.0)
        stability_scores.append(diff)
    return float(np.mean(stability_scores))

def compute_pbo_score(oos_returns: list) -> float:
    if not oos_returns or len(oos_returns) < 3:
        return 0.0
    try:
        oos = np.array(oos_returns)
        mean_ret = np.mean(oos)
        median_ret = np.median(oos)
        std_ret = np.std(oos)
        skew = abs(mean_ret - median_ret) / std_ret if std_ret > 0 else 0.0
        pbo = min(1.0, skew / 2.0)
        return pbo
    except Exception as e:
        logger.warning(f"自定義 PBO 計算失敗: {e}")
        return 0.0

def compute_period_return(df_price: pd.DataFrame, trades: list, start: str, end: str) -> tuple:
    try:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start not in df_price.index or end not in df_price.index:
            logger.warning(f"期間 {start} → {end} 不在價格數據索引中")
            return 0.0, 0
        period_trades = [t for t in trades if len(t) > 2 and pd.Timestamp(t[0]) >= start and pd.Timestamp(t[2]) <= end]
        if not period_trades:
            logger.info(f"期間 {start} → {end} 無交易")
            return 0.0, 0
        returns = [t[1] for t in period_trades]
        total_return = np.prod([1 + r for r in returns]) - 1
        num_trades = len(period_trades)
        logger.info(f"期間 {start} → {end}: 報酬={total_return:.2f}, 交易數={num_trades}")
        return total_return, num_trades
    except Exception as e:
        logger.warning(f"計算期間報酬失敗: {start} → {end}, 錯誤: {e}")
        return 0.0, 0

def equity_period_return(equity: pd.Series, start, end) -> float:
    """計算指定期間的 Equity Curve 報酬率。"""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start not in equity.index or end not in equity.index:
        logger.warning(f"Equity Curve 未涵蓋 {start} → {end}")
        return np.nan
    return equity.loc[end] / equity.loc[start] - 1

def _backtest_once(strat: str, params: dict, trial_results: list, data_source: str, df_price: pd.DataFrame, df_factor: pd.DataFrame) -> tuple:
    try:
        if df_price.empty:
            logger.error(f"價格數據為空，策略: {strat}, 數據源: {data_source}")
            log_to_results("錯誤", f"價格數據為空，策略: {strat}, 數據源: {data_source}")
            return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series()
        if not hasattr(df_price, 'name') or df_price.name is None:
            df_price.name = TICKER.replace(':', '_')
        if not df_factor.empty and (not hasattr(df_factor, 'name') or df_factor.name is None):
            df_factor.name = f"{TICKER}_factor"
        compute_f = STRAT_FUNC_MAP[strat]
        ind_keys = cfg.STRATEGY_PARAMS[strat]["ind_keys"]
        ind_p = {k: params[k] for k in ind_keys}
        ind_p["smaa_source"] = data_source
        if strat == "ssma_turn":
            df_ind, buys, sells = compute_f(df_price, df_factor, **ind_p)
            if df_ind.empty:
                logger.warning(f"計算指標失敗，策略: {strat}")
                log_to_results("警告", f"計算指標失敗，策略: {strat}")
                return -np.inf, 0, 0.0, 0.0, 0.0, []
            bt = SSS.backtest_unified(df_ind=df_ind, strategy_type=strat, params=params, buy_dates=buys, sell_dates=sells, discount=COST_PER_SHARE / 100, trade_cooldown_bars=COOLDOWN_BARS)
        else:
            df_ind = compute_f(df_price, df_factor, **ind_p)
            if df_ind.empty:
                logger.warning(f"計算指標失敗，策略: {strat}")
                log_to_results("警告", f"計算指標失敗，策略: {strat}")
                return -np.inf, 0, 0.0, 0.0, 0.0, []
            bt = SSS.backtest_unified(df_ind=df_ind, strategy_type=strat, params=params, discount=COST_PER_SHARE / 100, trade_cooldown_bars=COOLDOWN_BARS)
        if not bt or "metrics" not in bt or any(pd.isna(bt["metrics"].get(k)) for k in ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor"]):
            logger.error(f"回測結果無效或指標缺失，策略: {strat}, bt: {bt}")
            return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series()
        equity_curve = bt.get("equity_curve", pd.Series(index=df_price.index, data=100000.0))  # 預設初始資金
        metrics = bt["metrics"]
        trades_df = bt.get("trades_df", [])
        if trades_df is None:
            logger.warning(f"回測未返回 trades_df，策略: {strat}")
            return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series()  # 確保返回 7 個值
        profit_factor = metrics.get("profit_factor", 0.1)
        if "profit_factor" not in metrics or pd.isna(profit_factor):
            trades = bt["trades"]
            if trades:
                gains = [t[1] for t in trades if t[1] > 0]
                losses = [abs(t[1]) for t in trades if t[1] < 0]
                profit_factor = sum(gains) / sum(losses) if losses and sum(losses) > 0 else np.inf
                profit_factor = min(profit_factor, 10.0)
        params_flat = {f"param_{k}": v for k, v in params.items()}  # 攤平參數
        trial_results.append({"total_return": metrics.get("total_return", 0.0), "num_trades": metrics.get("num_trades", 0), "sharpe_ratio": metrics.get("sharpe_ratio", 0.0), "max_drawdown": metrics.get("max_drawdown", 0.0), "profit_factor": profit_factor, "data_source": data_source, **params_flat})
        logger.info(f"回測完成，策略: {strat}, 總報酬={metrics.get('total_return', 0.0)*100:.2f}%, 交易次數={metrics.get('num_trades', 0)}")
        return (metrics.get("total_return", 0.0), metrics.get("num_trades", 0), metrics.get("sharpe_ratio", 0.0),
                metrics.get("max_drawdown", 0.0), profit_factor, bt.get("trades", []), bt.get("equity_curve", pd.Series()))  # 確保返回 7 個值
    except Exception as e:
        logger.error(f"回測失敗，策略: {strat}, 錯誤: {e}")
        return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series()

def _wf_min_return(strat: str, params: dict, data_source: str) -> float:
    try:
        df_price, df_factor = data_loader.load_data(TICKER, start_date=cfg.START_DATE, end_date="2025-06-06", smaa_source=data_source)
        if df_price.empty:
            logger.warning(f"Walk-forward 測試數據為空,策略: {strat}")
            return np.nan
        valid_dates = df_price.index
        valid_periods = []
        min_lookback = params.get('smaalen', 60) + params.get('quantile_win', 60)
        for start, end in WF_PERIODS:
            start_candidates = valid_dates[valid_dates >= pd.Timestamp(start)]
            end_candidates = valid_dates[valid_dates <= pd.Timestamp(end)]
            if start_candidates.empty or end_candidates.empty:
                logger.warning(f"無效 walk_forward 期間: {start} → {end}")
                continue
            adjusted_start = start_candidates[0]
            adjusted_end = end_candidates[-1]
            if (adjusted_end - adjusted_start).days < min_lookback * 1.5:
                logger.warning(f"Walk-forward 期間過短: {adjusted_start} → {adjusted_end}")
                continue
            valid_periods.append((adjusted_start, adjusted_end))
        if not valid_periods:
            logger.warning(f"無有效的 walk_forward 期間,策略: {strat}")
            return np.nan
        results = SSS.compute_backtest_for_periods(ticker=TICKER, periods=valid_periods, strategy_type=strat, params=params, trade_cooldown_bars=COOLDOWN_BARS, discount=COST_PER_SHARE / 100, df_price=df_price, df_factor=df_factor, smaa_source=data_source)
        valid_returns = []
        for i, r in enumerate(results):
            try:
                total_return = r["metrics"]["total_return"]
                num_trades = r["metrics"]["num_trades"]
                logger.info(f"Walk-forward 時段 {valid_periods[i][0]} 至 {valid_periods[i][1]}: 報酬={total_return:.2f}, 交易數={num_trades}")
                if np.isnan(total_return) or num_trades == 0:
                    logger.warning(f"Walk-forward 時段 {valid_periods[i][0]} 至 {valid_periods[i][1]} 無有效交易")
                    continue
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logger.warning(f"Walk-forward 期間缺少 total_return,時段: {valid_periods[i]}, 錯誤: {e}")
                continue
        return 0.0 if not valid_returns else min(valid_returns)
    except Exception as e:
        logger.error(f"Walk-forward 測試失敗,策略: {strat}, 錯誤: {e}")
        return np.nan

def _stress_avg_return(strat: str, params: dict, data_source: str, valid_stress_periods: list, df_price: pd.DataFrame, trades: list) -> float:
    if not valid_stress_periods:
        logger.warning(f"無有效的壓力測試期間,策略: {strat}")
        return np.nan
    try:
        if df_price.empty:
            logger.warning(f"壓力測試數據為空,策略: {strat}")
            return np.nan
        valid_returns = []
        for start, end in valid_stress_periods:
            total_return, num_trades = compute_period_return(df_price, trades, start, end)
            if num_trades == 0:
                logger.warning(f"壓力測試時段 {start} 至 {end} 無有效交易")
                continue
            valid_returns.append(total_return)
        return 0.0 if not valid_returns else float(np.mean(valid_returns))
    except Exception as e:
        logger.error(f"壓力測試失敗,策略: {strat}, 錯誤: {e}")
        return np.nan

def compute_stress_mdd_equity(equity: pd.Series, stress_periods):
    """計算壓力期間的最大回撤，使用完整 equity curve。"""
    mdds = []
    for start, end in stress_periods:
        if start not in equity.index or end not in equity.index:
            logger.warning(f"壓力測試期間無效: {start} → {end}")
            continue
        period_equity = equity.loc[start:end]
        drawdown = (period_equity / period_equity.cummax()) - 1
        mdds.append(drawdown.min() if not drawdown.empty else 0.0)
    return float(np.mean(mdds)) if mdds else 0.0

def compute_excess_return_in_stress_equity(equity: pd.Series, df_price: pd.DataFrame, stress_periods):
    """計算壓力期間的超額報酬，使用完整 Equity Curve。"""
    excess = []
    for start, end in stress_periods:
        strat_ret = equity_period_return(equity, start, end)
        bh_ret = buy_and_hold_return(df_price, start, end)
        if not np.isnan(strat_ret) and not np.isnan(bh_ret):
            excess.append(strat_ret - bh_ret)
    return float(np.mean(excess)) if excess else 0.0

def objective(trial: optuna.Trial):
    if args.strategy == 'all':
        strat = np.random.choice(list(STRATEGY_WEIGHTS.keys()), p=list(STRATEGY_WEIGHTS.values()))
    else:
        strat = args.strategy
    trial.set_user_attr("strategy", strat)
    data_source = np.random.choice(list(DATA_SOURCES_WEIGHTS.keys()), p=list(DATA_SOURCES_WEIGHTS.values()))
    trial.set_user_attr("data_source", data_source)
    params = _sample_params(trial, strat)
    trial_results = trial.study.user_attrs.get("trial_results", [])
    df_price, df_factor = data_loader.load_data(TICKER, start_date=cfg.START_DATE, end_date="2025-06-06", smaa_source=data_source)
    if df_price.empty:
        logger.error(f"價格數據為空，策略: {strat}")
        return -np.inf
    df_factor.to_csv(f"v9_factor_data_{TIMESTAMP}.csv")
    valid_dates = df_price.index
    valid_stress_periods = []
    warmup_days = max(params.get('linlen', 60), params.get('smaalen', 60), params.get('vol_window', 60), params.get('quantile_win', 60))
    for start, end in STRESS_PERIODS:
        start_date = pd.Timestamp(start) - timedelta(days=warmup_days)
        start_candidates = valid_dates[valid_dates >= start_date]
        end_candidates = valid_dates[valid_dates <= pd.Timestamp(end)]
        if start_candidates.empty or end_candidates.empty:
            logger.warning(f"跳過無效壓力測試期間：{start} → {end}")
            continue
        adjusted_start = start_candidates[0]
        adjusted_end = end_candidates[-1]
        if (adjusted_end - adjusted_start).days < warmup_days * 2:
            logger.warning(f"壓力測試期間過短: {adjusted_start} → {adjusted_end}")
            continue
        logger.info(f"壓力測試期間調整: {start} → {adjusted_start}, {end} → {adjusted_end}")
        valid_stress_periods.append((adjusted_start, adjusted_end))
    if not valid_stress_periods:
        logger.error(f"無有效壓力測試時段，策略: {strat}")
        log_to_results("錯誤", f"無有效壓力測試時段，策略: {strat}")
        valid_stress_periods.append((adjusted_start, adjusted_end))

    total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades, equity_curve = _backtest_once(strat, params, trial_results, data_source, df_price, df_factor)
    if total_ret == -np.inf or not (MIN_NUM_TRADES <= n_trades <= MAX_NUM_TRADES) or total_ret <= 2.0 or max_drawdown < -0.5 or profit_factor < 0.5:
        logger.info(f"試驗被剔除: 條件未滿足，策略: {strat}, 總報酬={total_ret*100:.2f}%, 交易次數={n_trades}")
        log_to_results("試驗被剔除", f"條件未滿足，策略: {strat}, 總報酬={total_ret*100:.2f}%, 交易次數={n_trades}")
        return -np.inf

    # 檢查試驗唯一性並追加結果
    params_key = tuple(sorted(params.items()))
    if not any(tuple(sorted(r.items()))[:len(params_key)] == params_key for r in trial_results):
        trial_results.append({"total_return": total_ret, "num_trades": n_trades, "sharpe_ratio": sharpe_ratio,
                             "max_drawdown": max_drawdown, "profit_factor": profit_factor, "data_source": data_source, **params})
    trial.study.set_user_attr("trial_results", trial_results)
    min_wf_ret = _wf_min_return(strat, params, data_source)
    if pd.isna(min_wf_ret):
        logger.warning(f"Walk-forward 測試無效，策略: {strat}")
        return -np.inf
    avg_stress_ret = _stress_avg_return(strat, params, data_source, valid_stress_periods, df_price, trades)
    avg_buy_dist, avg_sell_dist = analyze_trade_timing(df_price, trades)
    trial.set_user_attr("avg_buy_dist", avg_buy_dist)
    trial.set_user_attr("avg_sell_dist", avg_sell_dist)

    try:
        if df_price.empty:
            logger.error(f"CPCV 價格數據為空，策略: {strat}")
            return -np.inf
        event_times = [(t[0], t[0]) for t in trades if t[0]]
        if not event_times:
            logger.info(f"試驗被剔除: 無交易記錄，策略: {strat}")
            return -np.inf
        n_splits = min(CPCV_NUM_SPLITS, len(df_price) // (CPCV_EMBARGO_DAYS + 60))
        min_test_len = (params.get('smaalen', 60) + params.get('quantile_win', 60)) * 1.5
        if len(df_price) / n_splits < min_test_len:
            n_splits = max(3, int(len(df_price) // min_test_len))
            logger.info(f"調整 CPCV 折數至 {n_splits}")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        folds = []
        valid_dates = df_price.index
        for train_idx, test_idx in tscv.split(df_price.index):
            train_start = df_price.index[train_idx[0]]
            train_end = df_price.index[train_idx[-1]]
            test_start = df_price.index[test_idx[0]] + pd.Timedelta(days=CPCV_EMBARGO_DAYS)
            test_end = df_price.index[test_idx[-1]]
            test_start_candidates = valid_dates[valid_dates >= test_start]
            test_end_candidates = valid_dates[valid_dates <= test_end]
            if test_start_candidates.empty or test_end_candidates.empty:
                logger.warning(f"跳過無效 fold：test_start={test_start}, test_end={test_end}")
                continue
            adjusted_start = test_start_candidates[0]
            adjusted_end = test_end_candidates[-1]
            folds.append(([train_start, train_end], [adjusted_start, adjusted_end]))
        if setlimit:    
            if not (setminsharpe <= sharpe_ratio <= setmaxsharpe and setmaxmdd <= max_drawdown <= setminmdd):
                logger.info(f"試驗被剔除: Sharpe Ratio ({sharpe_ratio:.3f}) 或 MDD ({max_drawdown:.3f}) 未達標，策略: {strat}")
                log_to_results("試驗被剔除", f"Sharpe Ratio 或 MDD 未達標，策略: {strat}, Sharpe={sharpe_ratio:.3f}, MDD={max_drawdown:.3f}")
                return -np.inf
        oos_returns = []
        excess_returns = []
        sra_scores = []
        total_folds = len(folds)
        for train_block, test_block in folds:
            test_results = SSS.compute_backtest_for_periods(ticker=TICKER, periods=[(test_block[0], test_block[1])], strategy_type=strat, params=params, trade_cooldown_bars=COOLDOWN_BARS, discount=COST_PER_SHARE / 100, df_price=df_price, df_factor=df_factor, smaa_source=data_source)
            if not test_results or 'metrics' not in test_results[0]:
                logger.warning(f"無有效回測結果，策略: {strat}, 時段: {test_block}")
                continue
            metrics = test_results[0]['metrics']
            strategy_return = metrics.get('total_return', 0.0)
            test_start, test_end = test_block
            bh_return = buy_and_hold_return(df_price, test_start, test_end)
            excess_return = strategy_return - bh_return
            excess_returns.append(excess_return)
            oos_returns.append(strategy_return)
            strategy_sharpe, bh_sharpe, p_value = compute_simplified_sra(df_price, trades, [test_block])
            sra_scores.append((strategy_sharpe, p_value))

        if not excess_returns:
            logger.warning(f"無有效 excess_returns，策略: {strat}")
            return -np.inf
        excess_return_mean = np.mean(excess_returns)
        stress_mdd = compute_stress_mdd_equity(equity_curve, valid_stress_periods)
        excess_return_stress = compute_excess_return_in_stress_equity(equity_curve, df_price, valid_stress_periods)
        pbo_score = compute_pbo_score(oos_returns)
        # 儲存至 user_attrs
        trial.set_user_attr("total_return", total_ret)
        trial.set_user_attr("max_drawdown", max_drawdown)
        trial.set_user_attr("sharpe_ratio", sharpe_ratio)
        trial.set_user_attr("stress_mdd", stress_mdd)
        trial.set_user_attr("excess_return", excess_return_mean)
        trial.set_user_attr("stress_mdd", stress_mdd)
        trial.set_user_attr("excess_return_stress", excess_return_stress)
        trial.set_user_attr("pbo_score", pbo_score)
        # 同步更新至 trial_results
        if trial_results:
            trial_results[-1].update({
                "stress_mdd": stress_mdd,
                "excess_return": excess_return_mean,
                "excess_return_stress": excess_return_stress
            })
        logger.info(f"PBO 分數: {pbo_score:.3f}, 策略: {strat}")
        log_to_results("PBO 計算", f"PBO 分數: {pbo_score:.3f}, 策略: {strat}")

        # Min-Max Scaling
        tr_s = minmax(total_ret, 5, 25)
        pf_s = minmax(profit_factor, 0.5, 8)
        wf_s = minmax(min_wf_ret, 0, 2)
        sh_s = minmax(sharpe_ratio, 0.5, 0.8)
        mdd_s = 1 - minmax(abs(max_drawdown), 0, 0.3)
        excess_return_scaled = minmax(excess_return_mean, -1, 1)
        stress_mdd_scaled = minmax(abs(stress_mdd), 0, 0.5)
        pbo_score_scaled = pbo_score

        # Robust Score
        robust_score = 0.4 * excess_return_scaled + 0.3 * (1 - stress_mdd_scaled) + 0.3 * (1 - pbo_score_scaled)
        trial.set_user_attr("robust_score", robust_score)

        # 加權分數
        score = (SCORE_WEIGHTS["total_return"] * tr_s + SCORE_WEIGHTS["profit_factor"] * pf_s +
                 SCORE_WEIGHTS["sharpe_ratio"] * sh_s + SCORE_WEIGHTS["max_drawdown"] * mdd_s +
                 SCORE_WEIGHTS["wf_min_return"] * wf_s + robust_score)

        trade_penalty = 0.05 * max(0, MIN_NUM_TRADES - n_trades)
        score -= trade_penalty
        if excess_returns:
            excess_ranks = np.argsort(excess_returns)
            bottom_20_percent = int(0.2 * len(excess_returns))
            for idx in excess_ranks[:bottom_20_percent]:
                score *= 0.9
            logger.info(f"超額報酬懲罰: 倒數 20% fold 數={bottom_20_percent}, 分數調整={score:.3f}")
            log_to_results("超額報酬懲罰", f"倒數 20% fold 數={bottom_20_percent}, 分數調整={score:.3f}")
        avg_p_value = np.mean([score[1] for score in sra_scores]) if sra_scores else 1.0
        trial.set_user_attr("sra_p_value", avg_p_value)
        if pbo_score > 0.6:
            penalty = min(0.10, pbo_score / 2)
            score *= (1 - penalty)
            logger.info(f"柔性懲罰: PBO={pbo_score:.3f}, 懲罰={penalty:.3f}, 分數調整={score:.3f}")
            log_to_results("柔性懲罰", f"PBO={pbo_score:.3f}, 懲罰={penalty:.3f}, 分數調整={score:.3f}")
        stab = 0.0
        if len(trial_results) >= 6:
            stab = compute_knn_stability(trial_results, params=['linlen', 'smaalen', 'buy_mult'], k=5)
            if stab > 0.5:
                alpha = 0.2
                penalty_mult = alpha * (stab - 0.5)
                score *= (1 - min(penalty_mult, 0.10))
                logger.info(f"KNN 過擬合懲罰: 穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")
                log_to_results("KNN 過擬合懲罰", f"穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")
            trial.set_user_attr("total_return_scaled", tr_s)
            trial.set_user_attr("num_trades", n_trades)
            trial.set_user_attr("sharpe_ratio_scaled", sh_s)
            trial.set_user_attr("max_drawdown_scaled", mdd_s)
            trial.set_user_attr("profit_factor_scaled", pf_s)
            trial.set_user_attr("min_wf_return_scaled", wf_s)
            trial.set_user_attr("stability_score", float(stab))
        # 檢查試驗唯一性
        params_key = tuple(sorted(params.items()))
        if not any(tuple(sorted(r.items()))[:len(params_key)] == params_key for r in trial_results):
            trial_results.append({"total_return": total_ret, "num_trades": n_trades, "sharpe_ratio": sharpe_ratio,
                                "max_drawdown": max_drawdown, "profit_factor": profit_factor, "data_source": data_source, **params})

        # 構建格式化試驗細節
        trial_details = (
            f"試驗 {str(trial.number).zfill(5)}, 策略: {strat}, 數據源: {data_source}, 分數: {score:.3f}, "
            f"參數: {params}, 總報酬={total_ret*100:.2f}%, 交易次數={n_trades}, 夏普比率={sharpe_ratio:.3f}, "
            f"最大回撤={max_drawdown:.3f}, 盈虧因子={profit_factor:.2f}, WF最差報酬(分段)={min_wf_ret:.2f}, "
            f"壓力平均報酬(整段)={avg_stress_ret:.3f}, 穩定性得分={stab:.2f}, Robust Score={robust_score:.3f}, "
            f"Excess Return in Stress={excess_return_stress:.3f}, Stress MDD={stress_mdd:.3f}, "
            f"PBO 分數={pbo_score:.2f}, SRA p-value={avg_p_value:.3f}, 買入距離低點={avg_buy_dist:.1f}天, "
            f"賣出距離高點={avg_sell_dist:.1f}天"
        )
        logger.info(trial_details)

        # 將攤平的參數加入 log_data
        params_flat = {f"param_{k}": v for k, v in params.items()}
        log_data = {
            "trial_number": str(trial.number).zfill(5),
            "score": score,
            **params_flat,           # 加入攤平參數
            "parameters": str(params),  # 保留人類可讀字串
            "total_return": f"{total_ret*100:.2f}%",
            "num_trades": n_trades,
            "sharpe_ratio": f"{sharpe_ratio:.3f}",
            "max_drawdown": f"{max_drawdown:.3f}",
            "profit_factor": f"{profit_factor:.2f}",
            "min_wf_return": f"{min_wf_ret:.2f}",
            "avg_stress_return": f"{avg_stress_ret:.3f}",
            "stability_score": f"{stab:.2f}",
            "robust_score": f"{robust_score:.3f}",
            "excess_return_stress": f"{excess_return_stress:.3f}",
            "stress_mdd": f"{stress_mdd:.3f}",
            "pbo_score": f"{pbo_score:.2f}",
            "sra_p_value": f"{avg_p_value:.3f}",
            "avg_buy_dist": f"{avg_buy_dist:.1f}天",
            "avg_sell_dist": f"{avg_sell_dist:.1f}天"
        }
        merged_attrs = {**log_data, **{k: v for k, v in trial.user_attrs.items() if k not in log_data}}
        log_to_results("試驗結果", trial_details, **merged_attrs)

        if trial.number % 100 == 0:
            logger.debug(f"[Debug] len(stress_mdds)={len([t.user_attrs.get('stress_mdd') for t in study.trials if t.user_attrs.get('stress_mdd') is not None])}, "
                         f"len(excess_returns)={len([t.user_attrs.get('excess_return') for t in study.trials if t.user_attrs.get('excess_return') is not None])}")
            results = trial.study.user_attrs.get("trial_results", [])
            if results:
                total_returns = [r["total_return"] for r in results]
                mdds = [r.get("max_drawdown", 0.0) for r in results]
                sharpes = [r.get("sharpe_ratio", 0.0) for r in results]
            stress_mdds = [t.user_attrs.get("stress_mdd") for t in study.trials]
            excess_returns = [t.user_attrs.get("excess_return") for t in study.trials]

            # 過濾 None 和 NaN，確保有效數據
            valid_mdds = [x for x in mdds if x is not None and not np.isnan(x)]
            valid_sharpes = [x for x in sharpes if x is not None and not np.isnan(x)]
            valid_stress_mdds = [x for x in stress_mdds if x is not None and not np.isnan(x)]
            valid_excess_returns = [x for x in excess_returns if x is not None and not np.isnan(x)]

            # 記錄圖檔名稱至日誌
            plot_files = [
                f"mdd_vs_total_return_trial_{trial.number}_{TIMESTAMP}.png",
                f"sharpe_vs_total_return_trial_{trial.number}_{TIMESTAMP}.png",
                f"stress_mdd_vs_total_return_trial_{trial.number}_{TIMESTAMP}.png",
                f"excess_vs_total_return_trial_{trial.number}_{TIMESTAMP}.png"
            ]
            logger.info(f"嘗試生成圖檔: {', '.join(plot_files)}")

            # 繪製 Total Return vs MDD
            if len(valid_mdds) > 10:  # 至少 10 點數據
                x_min, x_max = min(valid_mdds) * 1.1, max(valid_mdds) * 1.1 if valid_mdds else (-0.5, 0.5)
                y_min, y_max = min(total_returns) * 0.9, max(total_returns) * 1.1 if total_returns else (0, 35)
                plt.figure(figsize=(10, 6))
                x_jitter = np.random.normal(0, 0.003, len(valid_mdds))
                plt.scatter([x + x_jitter[i] for i, x in enumerate(valid_mdds)], total_returns, alpha=0.4)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.xlabel("MDD")
                plt.ylabel("Total Return")
                plt.title(f"Total Return vs MDD - Trial {trial.number}")
                plt.text(x_min, y_max * 0.95, f'Samples: {len(valid_mdds)}', fontsize=10)
                plt.savefig(cfg.RESULT_DIR / plot_files[0])
                plt.close()

            # 繪製 Total Return vs Sharpe Ratio
            if len(valid_sharpes) > 10:
                x_min, x_max = min(valid_sharpes) * 0.9, max(valid_sharpes) * 1.1 if valid_sharpes else (0.4, 0.8)
                y_min, y_max = min(total_returns) * 0.9, max(total_returns) * 1.1 if total_returns else (0, 35)
                plt.figure(figsize=(10, 6))
                x_jitter = np.random.normal(0, 0.003, len(valid_sharpes))
                plt.scatter([x + x_jitter[i] for i, x in enumerate(valid_sharpes)], total_returns, alpha=0.4)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.xlabel("Sharpe Ratio")
                plt.ylabel("Total Return")
                plt.title(f"Total Return vs Sharpe Ratio - Trial {trial.number}")
                plt.text(x_min, y_max * 0.95, f'Samples: {len(valid_sharpes)}', fontsize=10)
                plt.savefig(cfg.RESULT_DIR / plot_files[1])
                plt.close()

            # 繪製 Total Return vs Stress MDD
            if len(valid_stress_mdds) > 10:
                x_min, x_max = min(valid_stress_mdds) * 1.1, max(valid_stress_mdds) * 1.1 if valid_stress_mdds else (-0.05, 0.05)
                y_min, y_max = min(total_returns) * 0.9, max(total_returns) * 1.1 if total_returns else (0, 35)
                plt.figure(figsize=(10, 6))
                x_jitter = np.random.normal(0, 0.003, len(valid_stress_mdds))
                plt.scatter([x + x_jitter[i] for i, x in enumerate(valid_stress_mdds)], total_returns, alpha=0.4)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.xlabel("Stress MDD")
                plt.ylabel("Total Return")
                plt.title(f"Total Return vs Stress MDD - Trial {trial.number}")
                plt.text(x_min, y_max * 0.95, f'Samples: {len(valid_stress_mdds)}', fontsize=10)
                plt.savefig(cfg.RESULT_DIR / plot_files[2])
                plt.close()

            # 繪製 Total Return vs Excess Return in Stress
            if len(valid_excess_returns) > 10:
                x_min, x_max = min(valid_excess_returns) * 1.1, max(valid_excess_returns) * 1.1 if valid_excess_returns else (-0.05, 0.05)
                y_min, y_max = min(total_returns) * 0.9, max(total_returns) * 1.1 if total_returns else (0, 35)
                plt.figure(figsize=(10, 6))
                x_jitter = np.random.normal(0, 0.003, len(valid_excess_returns))
                plt.scatter([x + x_jitter[i] for i, x in enumerate(valid_excess_returns)], total_returns, alpha=0.4)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.xlabel("Excess Return in Stress")
                plt.ylabel("Total Return")
                plt.title(f"Total Return vs Excess Return in Stress - Trial {trial.number}")
                plt.text(x_min, y_max * 0.95, f'Samples: {len(valid_excess_returns)}', fontsize=10)
                plt.savefig(cfg.RESULT_DIR / plot_files[3])
                plt.close()

    except Exception as e:
        logger.error(f"CPCV/OOS 計算失敗: {e}, 策略: {strat}")
        # 備用邏輯
        tr_s = minmax(total_ret, 5, 14)
        pf_s = minmax(profit_factor, 0.5, 8)
        wf_s = minmax(min_wf_ret, 0, 2)
        sh_s = minmax(sharpe_ratio, 0.5, 1.2)
        mdd_s = 1 - minmax(abs(max_drawdown), 0, 0.3)
        stress_mdd = compute_stress_mdd_equity(equity_curve, valid_stress_periods) if 'equity_curve' in locals() else 0.0
        excess_return_stress = compute_excess_return_in_stress_equity(equity_curve, df_price, valid_stress_periods) if 'equity_curve' in locals() else 0.0
        stress_mdd_scaled = minmax(abs(stress_mdd), 0, 0.5)
        excess_return_scaled = minmax(excess_return_mean, -1, 1)
        robust_score = 0.4 * excess_return_scaled + 0.3 * (1 - stress_mdd_scaled) + 0.3 * (1 - pbo_score_scaled)
        score = (SCORE_WEIGHTS["total_return"] * tr_s + SCORE_WEIGHTS["profit_factor"] * pf_s +
                SCORE_WEIGHTS["sharpe_ratio"] * sh_s + SCORE_WEIGHTS["max_drawdown"] * mdd_s +
                SCORE_WEIGHTS["wf_min_return"] * wf_s + robust_score)
        trade_penalty = 0.05 * max(0, MIN_NUM_TRADES - n_trades)
        score -= trade_penalty
        stab = 0.0
        if len(trial_results) >= 6:
            stab = compute_knn_stability(trial_results, params=['linlen', 'smaalen', 'buy_mult'], k=5)
            if stab > 0.5:
                alpha = 0.2
                penalty_mult = alpha * (stab - 0.5)
                score *= (1 - min(penalty_mult, 0.2))
                logger.info(f"KNN 過擬合懲罰: 穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")
                log_to_results("KNN 過擬合懲罰", f"穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")
            trial.set_user_attr("total_return_scaled", tr_s)
            trial.set_user_attr("num_trades", n_trades)
            trial.set_user_attr("sharpe_ratio_scaled", sh_s)
            trial.set_user_attr("max_drawdown_scaled", mdd_s)
            trial.set_user_attr("profit_factor_scaled", pf_s)
            trial.set_user_attr("min_wf_return_scaled", wf_s)
            trial.set_user_attr("stability_score", float(stab))
        trial_details = (f"試驗 {str(trial.number).zfill(5)}, 策略: {strat}, 數據源: {data_source}, 分數: {score:.3f}, 參數: {params}, 總報酬={total_ret*100:.2f}%, 交易次數={n_trades}, 夏普比率={sharpe_ratio:.3f}, 最大回撤={max_drawdown:.3f}, 盈虧因子={profit_factor:.2f}, WF 最差報酬={min_wf_ret:.2f}, 壓力平均報酬={avg_stress_ret:.3f}, 穩定性得分={stab:.2f}, 買入距離低點={avg_buy_dist:.1f}天, 賣出距離高點={avg_sell_dist:.1f}天")
        params_flat = {f"param_{k}": v for k, v in params.items()}
        log_data = {
            "trial_number": str(trial.number).zfill(5),
            "score": score,
            **params_flat,           # 加入攤平參數
            "parameters": str(params),  # 保留人類可讀字串
            "total_return": f"{total_ret*100:.2f}%",
            "num_trades": n_trades,
            "sharpe_ratio": f"{sharpe_ratio:.3f}",
            "max_drawdown": f"{max_drawdown:.3f}",
            "profit_factor": f"{profit_factor:.2f}",
            "min_wf_return": f"{min_wf_ret:.2f}",
            "avg_stress_return": f"{avg_stress_ret:.3f}",
            "stability_score": f"{stab:.2f}",
            "robust_score": f"{robust_score:.3f}",
            "excess_return_stress": f"{excess_return_stress:.3f}",
            "stress_mdd": f"{stress_mdd:.3f}",
            "pbo_score": f"{pbo_score:.2f}",
            "sra_p_value": f"{avg_p_value:.3f}",
            "avg_buy_dist": f"{avg_buy_dist:.1f}天",
            "avg_sell_dist": f"{avg_sell_dist:.1f}天"
        }
        merged_attrs = {**log_data, **{k: v for k, v in trial.user_attrs.items() if k not in log_data}}
        log_to_results("試驗結果（備用邏輯）", trial_details, **merged_attrs)
        return score

if __name__ == "__main__":
    import shutil
    cache_dir = Path("C:/Stock_reserach/SSS095a1/cache/price")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info(f"已清空快取目錄：{cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    optuna_sqlite = Path(cfg.RESULT_DIR) / f"optuna_9_{TIMESTAMP}.sqlite3"
    study = optuna.create_study(study_name=f"00631L_optuna_v9_{args.strategy}", direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=20), storage=f"sqlite:///{optuna_sqlite}")
    n_trials = args.n_trials
    logger.info(f"開始最佳化,策略: {args.strategy}, 共 {n_trials} 次試驗,交易次數範圍: {MIN_NUM_TRADES}-{MAX_NUM_TRADES}")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    logger.info("最佳試驗:")
    best = study.best_trial
    logger.info(f"策略: {best.user_attrs['strategy']}")
    logger.info(f"數據源: {best.user_attrs['data_source']}")
    logger.info(f"參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}")
    logger.info(f"穩健分數: {best.value:.3f}")
    logger.info(f"其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}")
    best_trial_details = (f"策略: {best.user_attrs['strategy']}, 數據源: {best.user_attrs['data_source']}, 參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}, 穩健分數: {best.value:.3f}, 其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}")
    log_to_results("最佳試驗資訊", best_trial_details)
    strategies = [args.strategy] if args.strategy != 'all' else list(PARAM_SPACE.keys())
    data_sources = DATA_SOURCES
    trial_results = [entry for entry in results_log if entry["Event Type"] in ["試驗結果", "試驗結果（備用邏輯）"]]
    for strategy in strategies:
        for data_source in data_sources:
            strategy_source_trials = [entry for entry in trial_results if entry.get("strategy") == strategy and entry.get("data_source") == data_source]
            if not strategy_source_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 無有效試驗結果")
                continue
            trial_scores = [(entry.get("trial_number"), float(entry.get("score") or -np.inf), entry["Details"]) for entry in strategy_source_trials if entry.get("score") is not None]
            trial_scores.sort(key=lambda x: x[1], reverse=True)
            top_10_trials = trial_scores[:10]
            if top_10_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 前 10 名試驗:")
                for trial_num, score, details in top_10_trials:
                    logger.info(details)
                    log_to_results(f"前 10 名 {strategy} 搭配 {data_source} 試驗", details)
    for strategy in strategies:
        for data_source in data_sources:
            strategy_source_trials = [entry for entry in trial_results if entry.get("strategy") == strategy and entry.get("data_source") == data_source]
            if not strategy_source_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 無有效試驗結果 (分組前 5 名)")
                continue
            trial_scores = [(entry.get("trial_number"), float(entry.get("score") or -np.inf), entry["Details"]) for entry in strategy_source_trials if entry.get("score") is not None]
            trial_scores.sort(key=lambda x: x[1], reverse=True)
            top_5_trials = trial_scores[:5]
            if top_5_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 分組前 5 名試驗:")
                for trial_num, score, details in top_5_trials:
                    logger.info(details)
                    log_to_results(f"前 5 名 {strategy} 搭配 {data_source} 分組試驗", details)
    results = {"best_robust_score": best.value, "best_strategy": best.user_attrs["strategy"], "best_data_source": best.user_attrs["data_source"], "best_params": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.params.items() if k not in ["strategy", "data_source"]}, "best_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.user_attrs.items()}}
    result_file = cfg.RESULT_DIR / f"optuna_9_best_params_{TICKER.replace('^','')}_{TIMESTAMP}.json"
    pd.Series(results).to_json(result_file, indent=2)
    result_csv_file = cfg.RESULT_DIR / f"optuna_9_results_{TIMESTAMP}.csv"    
    logger.info(f"最佳參數已保存至 {result_file}")
    df_results = (pd.json_normalize(results_log, sep='_')
                  .loc[lambda d: d["Event Type"].str.contains("試驗結果")]
                  .sort_values("score", ascending=False))

    df_results.to_csv(result_csv_file, index=False, encoding="utf-8-sig")
    logger.info(f"試驗結果已保存至 {result_csv_file}")
    df_events = pd.DataFrame(events_log)
    event_csv_file = cfg.RESULT_DIR / f"optuna_9_events_{TIMESTAMP}.csv"
    df_events.to_csv(event_csv_file, index=False, encoding='utf-8-sig', na_rep='0.0')
    logger.info(f"事件紀錄已保存至 {event_csv_file}")
    logger.info("前 5 筆試驗記錄:")
    for record in results_log[:5]:
        logger.info(f"[{record['Timestamp']}] - {record['Event Type']} - {record['Details']}")
    results_list = load_results_to_list(result_csv_file)
    logger.info(f"從 {result_csv_file} 載入 {len(results_list)} 筆記錄")
    rma_trials = [r for r in results_list if r.get('strategy') == 'RMA' and r['Event Type'] in ["試驗結果", "試驗結果（備用邏輯）"]]
    logger.info(f"RMA 策略試驗數量：{len(rma_trials)}")
    if rma_trials:
        logger.info(f"第一筆 RMA 試驗：{rma_trials[0]['Details']}")