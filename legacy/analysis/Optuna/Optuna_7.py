'''
Optuna-based hyper-parameter optimization for 00631L strategies (Version 6)
--------------------------------------------------
* 最佳化目標: 優先追求完整回測報酬(>300%)、交易次數(15-60次),允許輕微過擬合,其次考慮 walk_forward 期間最差報酬、夏普比率、最大回撤,最後考慮壓力測試平均報酬。
* 本腳本可直接放在 analysis/ 目錄後以 `python optuna_6.py` 執行。
* 搜尋空間與權重在 `PARAM_SPACE` 與 `SCORE_WEIGHTS` 中設定,方便日後微調。
* 使用 sklearn 的 TimeSeriesSplit 替代 mlfinlab 的 CPCV,自定義 PBO 函數,確保策略在 OOS 期間穩定跑贏買入並持有,並分析交易時機（趨吉避凶）。
* 命令列範例：
  python optuna_6.py --strategy RMA --n_trials 10000
  python optuna_6.py --strategy single --n_trials 10000
  python optuna_6.py --strategy dual --n_trials 10000
  python optuna_6.py --strategy ssma_turn --n_trials 10000

改進：
1. 數據源分流：^TWII / 2414.TW:50%, Self:30%, ^TWII / 2412.TW:20%
2. 策略分流：RMA:30%, single:30%, dual:20%, ssma_turn:20%
3. 權重調整：提高 total_return 權重(3.0),降低次要指標權重,強調報酬率。
4. Min-Max Scaling：total_return 上限 14x,profit_factor 上限 8。
5. 過擬合懲罰：KNN 穩定性門檻放寬至 0.5,懲罰係數降至 0.2,CPCV fail_ratio 放寬至 0.4,PBO 放寬至 0.6。
6. CPCV + PBO：使用 total_return 作為主要 metric,比較買入並持有,添加多指標（profit_factor、max_drawdown）與 OOS 標準差檢查。
7. 簡化 SRA：比較策略與買入並持有的 Sharpe Ratio,若 p-value > 0.05,扣 10% 分數。
8. 超額報酬懲罰：OOS 超額報酬倒數 20% 的 fold 扣 10% 分數。
9. 交易時機分析：計算買入/賣出距離高低點天數,若 > 20 天,扣 5% 分數。
10. 交易日限制：確保 CPCV 和壓力測試僅使用交易日索引，避免非交易日偏移。

依賴：
  pip install optuna tqdm scikit-learn matplotlib scipy pandas numpy
'''
import logging
from logging_config import setup_logging
import optuna
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import argparse
from pathlib import Path
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# 初始化日誌設定
setup_logging()
logger = logging.getLogger("Optuna_6")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import config as cfg
from analysis import data_loader
import SSSv096 as SSS

# ---------- 命令列參數解析 ----------
parser = argparse.ArgumentParser(description='Optuna 最佳化 00631L 策略')
parser.add_argument('--strategy', type=str, choices=['single', 'dual', 'RMA', 'ssma_turn', 'all'], default='all',
                    help='指定單一策略進行最佳化 (預設: all)')
parser.add_argument('--n_trials', type=int, default=5000,
                    help='試驗次數 (預設: 5000)')
args = parser.parse_args()

# ---------- 日誌設定 ----------
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 用於儲存重要訊息的列表
results_log = []

# 記錄重要訊息的函數
def log_to_results(event_type, details, **kwargs):
    """記錄重要訊息到 results_log 列表中。"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    record = {
        "Timestamp": timestamp,
        "Event Type": event_type,
        "Details": details,
        **kwargs
    }
    results_log.append(record)

# 從 CSV 載入結果並轉回列表
def load_results_to_list(csv_file: Path) -> list:
    """從 CSV 檔案載入結果並轉回列表格式。"""
    if not csv_file.exists():
        logger.warning(f"結果檔案 {csv_file} 不存在,無法轉回列表")
        return []
    df_results = pd.read_csv(csv_file)
    return df_results.to_dict('records')

# ---------- 全域常數 ----------
TICKER = cfg.TICKER
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
COOLDOWN_BARS = cfg.TRADE_COOLDOWN_BARS
WF_PERIODS = [(p["test"][0], p["test"][1]) for p in cfg.WF_PERIODS]
STRESS_PERIODS = cfg.STRESS_PERIODS
STRAT_FUNC_MAP = {
    'single': SSS.compute_single,
    'dual': SSS.compute_dual,
    'RMA': SSS.compute_RMA,
    'ssma_turn': SSS.compute_ssma_turn_combined
}

# 數據源選項與權重
DATA_SOURCES = cfg.SOURCES
DATA_SOURCES_WEIGHTS = {
    'Self': 1/3,
    'Factor (^TWII / 2412.TW)': 1/3,
    'Factor (^TWII / 2414.TW)': 1/3
}

# 策略選項與權重
STRATEGY_WEIGHTS = {
    'single': 0.25,
    'dual': 0.25,
    'RMA': 0.25,
    'ssma_turn': 0.25
}

# 交易次數範圍
MIN_NUM_TRADES = 5
MAX_NUM_TRADES = 120

# CPCV 設定
CPCV_NUM_SPLITS = 3
CPCV_EMBARGO_DAYS = 15  # 自定義 embargo,單位：天
min_splits = 3
# ---------- 搜尋空間（縮小激進參數範圍） ---------
PARAM_SPACE = {
    "single": dict(
        linlen=(5, 240, 1),
        smaalen=(7, 240, 5),
        devwin=(5, 180, 1),
        factor=(20.0, 60.0, 1.0),
        buy_mult=(0.1, 1.2, 0.05),
        sell_mult=(0.5, 4.0, 0.05),
        stop_loss=(0.00, 0.15, 0.01),
    ),
    "dual": dict(
        linlen=(5, 240, 1),
        smaalen=(7, 240, 5),
        short_win=(10, 100, 5),
        long_win=(40, 240, 10),
        factor=(20.0, 100.0, 5.0),
        buy_mult=(0.2, 2, 0.05),
        sell_mult=(0.5, 4.0, 0.05),
        stop_loss=(0.00, 0.15, 0.01),
    ),
    "RMA": dict(
        linlen=(5, 240, 1),
        smaalen=(7, 240, 5),
        rma_len=(20, 100, 5),
        dev_len=(10, 100, 5),
        factor=(20.0, 100.0, 5.0),
        buy_mult=(0.2, 2, 0.05),
        sell_mult=(0.5, 4.0, 0.05),
        stop_loss=(0.00, 0.15, 0.01),
    ),
    "ssma_turn": dict(
        linlen=(40, 120, 5),
        smaalen=(60, 150, 5),
        factor=(10.0, 80.0, 10.0),
        prom_factor=(5, 70, 1),
        min_dist=(5, 20, 1),
        buy_shift=(0, 7, 1),
        exit_shift=(0, 7, 1),
        vol_window=(5, 90, 5),
        quantile_win=(5, 180, 10),
        signal_cooldown_days=(1, 7, 1),
        buy_mult=(0.5, 2, 0.05),
        sell_mult=(0.2, 3, 0.1),
        stop_loss=(0.00, 0.15, 0.01),
    ),
}

# 策略層級顯式關聯 ind_keys/bt_keys
IND_BT_KEYS = {
    strat: cfg.STRATEGY_PARAMS[strat]["ind_keys"] + cfg.STRATEGY_PARAMS[strat]["bt_keys"]
    for strat in PARAM_SPACE
}

# 各部份分數權重（報酬 60%,其他 15%）
SCORE_WEIGHTS = dict(
    total_return=2.5,       # 高優先級: 完整回測報酬
    profit_factor=0.2,      # 高優先級: 盈虧因子
    wf_min_return=0.2,      # 降低次要指標權重
    sharpe_ratio=0.2,
    max_drawdown=0.1,
    success_ratio=0.5
    #stress_avg_return=0.05,
    
)

# ---------- 工具函式 ----------
def _sample_params(trial: optuna.Trial, strat: str) -> dict:
    """依據 PARAM_SPACE 產生該策略的參數組,浮點數四捨五入至 3 位小數。"""
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
    """Min-Max Scaling 工具函式。"""
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1) if clip else y

def buy_and_hold_return(df_price: pd.DataFrame, start: str, end: str) -> float:
    """計算買入並持有報酬率，嚴格基於交易日序列。"""
    try:
        # 檢查空 DataFrame
        if df_price.empty:
            logger.error("df_price 為空 DataFrame")
            return 1.0
        # 檢查 close 欄位
        if 'close' not in df_price.columns:
            logger.error(f"df_price 缺少 'close' 欄位, 欄位列表：{df_price.columns}")
            return 1.0
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        # 檢查交易日索引
        if start not in df_price.index or end not in df_price.index:
            logger.error(f"start/end 不在交易日序列: {start} → {end}")
            return 1.0
        start_p, end_p = df_price.at[start, 'close'], df_price.at[end, 'close']
        if pd.isna(start_p) or pd.isna(end_p) or start_p == 0:
            logger.warning(f"價格缺失或為 0: {start_p}, {end_p}")
            return 1.0
        return end_p / start_p
    except Exception as e:
        logger.error(f"買入並持有計算錯誤：{e}, start={start}, end={end}")
        return 1.0
def get_fold_period(test_blocks: list) -> tuple:
    """從 test_blocks 提取 OOS 期間的起止時間。"""
    starts = [b[0] for b in test_blocks]
    ends = [b[1] for b in test_blocks]
    return min(starts), max(ends)

def analyze_trade_timing(df_price, trades, window=20):
    """分析交易時機：計算買入/賣出距離近期高低點的天數。"""
    buy_distances = []
    sell_distances = []
    for t in trades:
        entry, _, exit = t[0], t[1], t[2] if len(t) > 2 else (None, None, None)
        if not entry or not exit or entry not in df_price.index or exit not in df_price.index:
            continue
        # 買入距離低點
        window_data = df_price.loc[:entry, 'close'].tail(window)
        low_idx = window_data.idxmin()
        buy_dist = (pd.Timestamp(entry) - pd.Timestamp(low_idx)).days if low_idx else 0
        buy_distances.append(buy_dist)
        # 賣出距離高點
        window_data = df_price.loc[:exit, 'close'].tail(window)
        high_idx = window_data.idxmax()
        sell_dist = (pd.Timestamp(exit) - pd.Timestamp(high_idx)).days if high_idx else 0
        sell_distances.append(sell_dist)
    
    avg_buy_dist = np.mean(buy_distances) if buy_distances else 20.0
    avg_sell_dist = np.mean(sell_distances) if sell_distances else 20.0
    return avg_buy_dist, avg_sell_dist

def compute_simplified_sra(df_price, trades, test_blocks):
    """簡化 SRA：計算策略與買入並持有的 Sharpe Ratio 差異,並進行 t-test。"""
    try:
        # 計算策略每日報酬
        strategy_returns = []
        for t in trades:
            if len(t) > 2 and t[0] in df_price.index and t[2] in df_price.index:
                period = df_price.loc[t[0]:t[2], 'close'].pct_change().dropna()
                strategy_returns.extend(period)
        
        # 計算買入並持有報酬
        test_start, test_end = get_fold_period(test_blocks)
        bh_returns = df_price.loc[test_start:test_end, 'close'].pct_change().dropna()
        
        # 計算 Sharpe Ratio
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0.0
        bh_sharpe = np.mean(bh_returns) / np.std(bh_returns) * np.sqrt(252) if np.std(bh_returns) > 0 else 0.0
        
        # t-test
        t_stat, p_value = ttest_ind(strategy_returns, bh_returns, equal_var=False)
        return strategy_sharpe, bh_sharpe, p_value
    except Exception as e:
        logger.warning(f"簡化 SRA 計算失敗: {e}")
        return 0.0, 0.0, 1.0

def compute_knn_stability(df_results: list, params: list, k: int = 5, metric: str = 'total_return') -> float:
    """計算 KNN 穩定性,標準化參數並限制差距尺度。"""
    if len(df_results) < k + 1:
        logger.warning(f"試驗數量 {len(df_results)} 不足以計算 KNN 穩定性 (需要至少 {k+1})")
        return 0.0
    param_cols = [p for p in params if p in df_results[0]]
    if not param_cols:
        logger.warning(f"無有效參數用於 KNN 穩定性計算,參數: {params}")
        return 0.0
    
    # 提取參數與回報
    X = np.array([[r[p] for p in param_cols] for r in df_results])
    y = np.array([r[metric] for r in df_results])
    
    # 標準化參數
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KNN 計算
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    stability_scores = []
    for i, idx in enumerate(indices):
        roi = y[i]
        roi_neighbors = np.mean(y[idx[1:]])  # 排除自身
        # 限制差距尺度：若差距超過 2 倍,視為 2 倍
        diff = min(abs(roi - roi_neighbors), 2 * roi_neighbors if roi_neighbors > 0 else 2.0)
        stability_scores.append(diff)
    return float(np.mean(stability_scores))

def compute_pbo_score(oos_returns: list) -> float:
    """自定義 PBO 分數：估計過擬合概率,基於 OOS 報酬分佈的偏態和方差。"""
    if not oos_returns or len(oos_returns) < 3:
        return 0.0
    try:
        oos = np.array(oos_returns)
        mean_ret = np.mean(oos)
        median_ret = np.median(oos)
        std_ret = np.std(oos)
        # 計算偏態：若均值遠高於中位數,過擬合風險較高
        skew = abs(mean_ret - median_ret) / std_ret if std_ret > 0 else 0.0
        # 過擬合分數：偏態越高,分數越高
        pbo = min(1.0, skew / 2.0)
        return pbo
    except Exception as e:
        logger.warning(f"自定義 PBO 計算失敗: {e}")
        return 0.0

def _backtest_once(strat: str, params: dict, trial_results: list, data_source: str, 
                   df_price: pd.DataFrame, df_factor: pd.DataFrame) -> tuple:
    """單段完整回測，使用預載數據以避免重複載入，確保 SMAA 快取一致性。"""
    try:
        if df_price.empty:
            logger.error(f"價格數據為空，策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("錯誤", f"價格數據為空，策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf, 0, 0.0, 0.0, 0.0, []
        
        # 確保 df_price 和 df_factor 的 name 屬性正確
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
                logger.warning(f"計算指標失敗，策略: {strat}, 數據源: {data_source}, 參數: {params}")
                log_to_results("警告", f"計算指標失敗，策略: {strat}, 數據源: {data_source}, 參數: {params}")
                return -np.inf, 0, 0.0, 0.0, 0.0, []
            bt = SSS.backtest_unified(
                df_ind=df_ind,
                strategy_type=strat,
                params=params,
                buy_dates=buys,
                sell_dates=sells,
                discount=COST_PER_SHARE / 1.0,
                trade_cooldown_bars=COOLDOWN_BARS,
            )
            metrics = bt["metrics"]
            trades_df = bt.get("trades_df")  # 使用新的 trades_df
            if trades_df is None:
                logger.warning(f"回測未返回 trades_df，策略: {strat}, 數據源: {data_source}")
                return -np.inf, 0, 0.0, 0.0, 0.0, []            
        else:
            df_ind = compute_f(df_price, df_factor, **ind_p)
            if df_ind.empty:
                logger.warning(f"計算指標失敗，策略: {strat}, 數據源: {data_source}, 參數: {params}")
                log_to_results("警告", f"計算指標失敗，策略: {strat}, 數據源: {data_source}, 參數: {params}")
                return -np.inf, 0, 0.0, 0.0, 0.0, []
            bt = SSS.backtest_unified(
                df_ind=df_ind,
                strategy_type=strat,
                params=params,
                discount=COST_PER_SHARE / 1.0,
                trade_cooldown_bars=COOLDOWN_BARS,
            )
        
            metrics = bt["metrics"]
        if len(df_ind) / len(df_price) < 0.5:
            logger.info(f"試驗被剔除: 有效數據比例過低，策略: {strat}, 數據源: {data_source}, 有效數據: {len(df_ind)}/{len(df_price)}, linlen: {params['linlen']}, smaalen: {params['smaalen']}")
            log_to_results("試驗被剔除", f"有效數據比例過低，策略: {strat}, 數據源: {data_source}, 有效數據: {len(df_ind)}/{len(df_price)}, linlen: {params['linlen']}, smaalen: {params['smaalen']}")
            return -np.inf, 0, 0.0, 0.0, 0.0, []

        profit_factor = metrics.get("profit_factor", 0.1)
        if "profit_factor" not in metrics or pd.isna(profit_factor):
            trades = bt["trades"]
            if trades:
                gains = [t[1] for t in trades if t[1] > 0]
                losses = [abs(t[1]) for t in trades if t[1] < 0]
                profit_factor = sum(gains) / sum(losses) if losses and sum(losses) > 0 else np.inf
                profit_factor = min(profit_factor, 10.0)
        
        trial_results.append({
            "total_return": metrics.get("total_return", 0.0),
            "num_trades": metrics.get("num_trades", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "profit_factor": profit_factor,
            "data_source": data_source,
            **params
        })
        logger.info(f"回測完成，策略: {strat}, 數據源: {data_source}, 總報酬={metrics.get('total_return', 0.0)*100:.2f}%, 交易次數={metrics.get('num_trades', 0)}")
        log_to_results("回測完成", f"策略: {strat}, 數據源: {data_source}, 總報酬={metrics.get('total_return', 0.0)*100:.2f}%, 交易次數={metrics.get('num_trades', 0)}")
        
        return (
            metrics.get("total_return", 0.0),
            metrics.get("num_trades", 0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("max_drawdown", 0.0),
            profit_factor,
            bt["trades"]
        )
    except Exception as e:
        logger.error(f"回測失敗，策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        log_to_results("錯誤", f"回測失敗，策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        return -np.inf, 0, 0.0, 0.0, 0.0, []

def _wf_min_return(strat: str, params: dict, data_source: str) -> float:
    try:
        df_price, df_factor = data_loader.load_data(TICKER, smaa_source=data_source)
        if df_price.empty:
            logger.warning(f"Walk-forward 測試數據為空,策略: {strat}, 數據源: {data_source}")
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
                logger.warning(f"Walk-forward 期間過短: {adjusted_start} → {adjusted_end}, 要求至少 {min_lookback * 1.5} 天")
                continue
            valid_periods.append((adjusted_start, adjusted_end))

        if not valid_periods:
            logger.warning(f"無有效的 walk_forward 期間,策略: {strat}, 數據源: {data_source}")
            return np.nan
        results = SSS.compute_backtest_for_periods(
            ticker=TICKER,
            periods=valid_periods,
            strategy_type=strat,
            params=params,
            trade_cooldown_bars=COOLDOWN_BARS,
            discount=COST_PER_SHARE / 1.0,
            df_price=df_price,
            df_factor=df_factor
        )
        valid_returns = []
        for i, r in enumerate(results):
            try:
                total_return = r["metrics"]["total_return"]
                num_trades = r["metrics"]["num_trades"]
                if np.isnan(total_return) or num_trades == 0:
                    logger.warning(f"Walk-forward 時段 {valid_periods[i][0]} 至 {valid_periods[i][1]} 無有效交易, 交易數: {num_trades}")
                    continue
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logger.warning(f"Walk-forward 期間缺少 total_return,時段: {valid_periods[i]}, 錯誤: {e}")
                continue
        if not valid_returns:
            logger.warning(f"Walk-forward 無有效報酬,策略: {strat}, 數據源: {data_source}")
            return np.nan
        return min(valid_returns)
    except Exception as e:
        logger.error(f"Walk-forward 測試失敗,策略: {strat}, 數據源: {data_source}, 錯誤: {e}")
        return np.nan

def _stress_avg_return(strat: str, params: dict, data_source: str, valid_stress_periods: list) -> float:
    """計算壓力測試平均報酬率，使用預處理的交易日期間。"""
    if not valid_stress_periods:
        logger.warning(f"無有效的壓力測試期間,策略: {strat}, 數據源: {data_source}, 參數: {params}")
        return 0.0
    try:
        df_price, df_factor = data_loader.load_data(TICKER, smaa_source=data_source)
        if df_price.empty:
            logger.warning(f"壓力測試數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("警告", f"壓力測試數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf

        results = SSS.compute_backtest_for_periods(
            ticker=TICKER,
            periods=valid_stress_periods,
            strategy_type=strat,
            params=params,
            trade_cooldown_bars=COOLDOWN_BARS,
            discount=COST_PER_SHARE / 1.0,
        )
        if not results:
            logger.warning(f"壓力測試無結果,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("警告", f"壓力測試無結果,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf

        valid_returns = []
        for r in results:
            try:
                total_return = r["metrics"]["total_return"]
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logger.warning(f"壓力測試期間缺少 total_return,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
                log_to_results("警告", f"壓力測試期間缺少 total_return,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
                continue
        return float(np.mean(valid_returns)) if valid_returns else -np.inf
    except Exception as e:
        logger.error(f"壓力測試失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        log_to_results("錯誤", f"壓力測試失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        return -np.inf

# ---------- Optuna 目標 ----------
def objective(trial: optuna.Trial):
    # 選擇策略
    if args.strategy == 'all':
        strat = np.random.choice(list(STRATEGY_WEIGHTS.keys()), p=list(STRATEGY_WEIGHTS.values()))
    else:
        strat = args.strategy
    trial.set_user_attr("strategy", strat)

    # 選擇數據源
    data_source = np.random.choice(list(DATA_SOURCES_WEIGHTS.keys()), p=list(DATA_SOURCES_WEIGHTS.values()))
    trial.set_user_attr("data_source", data_source)

    params = _sample_params(trial, strat)

    # 儲存試驗結果
    trial_results = trial.study.user_attrs.get("trial_results", [])

    # 統一載入完整數據範圍
    df_price, df_factor = data_loader.load_data(TICKER, start_date=cfg.START_DATE, smaa_source=data_source)
    if df_price.empty:
        logger.error(f"價格數據為空，策略: {strat}, 數據源: {data_source}")
        log_to_results("錯誤", f"價格數據為空，策略: {strat}, 數據源: {data_source}")
        return -np.inf

    # 0001FIX:預處理壓力測試期間為交易日
    valid_dates = df_price.index
    valid_stress_periods = []
    for start, end in STRESS_PERIODS:
        start_candidates = valid_dates[valid_dates >= pd.Timestamp(start)]
        end_candidates = valid_dates[valid_dates <= pd.Timestamp(end)]
        if start_candidates.empty or end_candidates.empty:
            logger.warning(f"跳過無效壓力測試期間：{start} → {end}, 數據範圍: {valid_dates[0]} 到 {valid_dates[-1]}")
            continue
        adjusted_start = start_candidates[0]
        adjusted_end = end_candidates[-1]
        logger.info(f"壓力測試期間調整: {start} → {adjusted_start.strftime('%Y-%m-%d')}, {end} → {adjusted_end.strftime('%Y-%m-%d')}")
        valid_stress_periods.append((adjusted_start, adjusted_end))
    if not valid_stress_periods:
        logger.error(f"無有效壓力測試時段，策略: {strat}, 數據源: {data_source}")
        log_to_results("錯誤", f"無有效壓力測試時段，策略: {strat}, 數據源: {data_source}")
        valid_stress_periods.append((adjusted_start, adjusted_end))

    total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades = _backtest_once(
        strat, params, trial_results, data_source, df_price, df_factor
    )

    # 第一層硬性門檻
    if total_ret == -np.inf:
        return -np.inf
    if not (MIN_NUM_TRADES <= n_trades <= MAX_NUM_TRADES):
        logger.info(f"試驗被剔除: 交易次數 {n_trades} 不在 {MIN_NUM_TRADES}-{MAX_NUM_TRADES} 範圍,策略: {strat}, 數據源: {data_source}")
        log_to_results("試驗被剔除", f"交易次數 {n_trades} 不在 {MIN_NUM_TRADES}-{MAX_NUM_TRADES} 範圍,策略: {strat}, 數據源: {data_source}")
        return -np.inf
    if total_ret <= 3.0:
        logger.info(f"試驗被剔除: 完整回測報酬 {total_ret*100:.2f}% <= 300%,策略: {strat}, 數據源: {data_source}")
        log_to_results("試驗被剔除", f"完整回測報酬 {total_ret*100:.2f}% <= 300%,策略: {strat}, 數據源: {data_source}")
        return -np.inf
    if max_drawdown < -0.5:
        logger.info(f"試驗被剔除: 最大回撤 {max_drawdown:.3f} < -0.5,策略: {strat}, 數據源: {data_source}")
        return -np.inf
    if profit_factor < 0.5:
        logger.info(f"試驗被剔除: 盈虧因子 {profit_factor:.2f} < 0.5,策略: {strat}, 數據源: {data_source}")
        return -np.inf

    # 更新試驗結果
    trial.study.set_user_attr("trial_results", trial_results)

    # walk_forward 最差報酬
    min_wf_ret = _wf_min_return(strat, params, data_source)
    if min_wf_ret == -np.inf:
        return -np.inf

    # 壓力測試平均報酬, 0001FIX:新增valid_stress_periods
    avg_stress_ret = _stress_avg_return(strat, params, data_source, valid_stress_periods)

    # 趨吉避凶分析
    avg_buy_dist, avg_sell_dist = analyze_trade_timing(df_price, trades)
    trial.set_user_attr("avg_buy_dist", avg_buy_dist)
    trial.set_user_attr("avg_sell_dist", avg_sell_dist)

    # CPCV 與 OOS 報酬比較
    try:
        if df_price.empty:
            logger.error(f"CPCV 價格數據為空，策略: {strat}, 數據源: {data_source}")
            return -np.inf

        # 提取交易時間
        event_times = [(t[0], t[0]) for t in trades if t[0]]
        if not event_times:
            logger.info(f"試驗被剔除: 無交易記錄，策略: {strat}, 數據源: {data_source}")
            return -np.inf

        n_splits = min(CPCV_NUM_SPLITS, len(df_price) // (CPCV_EMBARGO_DAYS + 60))
        min_test_len = (params.get('smaalen', 60) + params.get('quantile_win', 60)) * 1.5
        if len(df_price) / n_splits < min_test_len:
            n_splits = max(3, int(len(df_price) // min_test_len))
            logger.info(f"調整 CPCV 折數至 {n_splits} 以確保 test 區段長度至少 {min_test_len} 天")
        tscv = TimeSeriesSplit(n_splits=n_splits)

        folds = []
        valid_dates = df_price.index  # 交易日索引
        for train_idx, test_idx in tscv.split(df_price.index):
            train_start = df_price.index[train_idx[0]]
            train_end = df_price.index[train_idx[-1]]
            test_start = df_price.index[test_idx[0]] + pd.Timedelta(days=CPCV_EMBARGO_DAYS)
            test_end = df_price.index[test_idx[-1]]
            # 確保 test_start 和 test_end 為交易日
            test_start_candidates = valid_dates[valid_dates >= test_start]
            test_end_candidates = valid_dates[valid_dates <= test_end]
            if test_start_candidates.empty or test_end_candidates.empty:
                logger.warning(f"跳過無效 fold：test_start={test_start}, test_end={test_end}, 數據範圍={valid_dates[0]} 到 {valid_dates[-1]}")
                continue
            adjusted_start = test_start_candidates[0]
            adjusted_end = test_end_candidates[-1]
            if adjusted_start != test_start or adjusted_end != test_end:
                logger.info(f"CPCV fold 調整: test_start {test_start} → {adjusted_start}, test_end {test_end} → {adjusted_end}")
            folds.append(([train_start, train_end], [adjusted_start, adjusted_end]))

        # 計算 OOS 報酬並比較買入並持有
        oos_returns = []
        excess_returns = []
        sra_scores = []
        fail_count = 0
        total_folds = len(folds)

        for train_block, test_block in folds:
            test_results = SSS.compute_backtest_for_periods(
                ticker=TICKER,
                periods=[(test_block[0], test_block[1])],
                strategy_type=strat,
                params=params,
                trade_cooldown_bars=COOLDOWN_BARS,
                discount=COST_PER_SHARE / 1.0,
                df_price=df_price,  # 傳遞預載數據
                df_factor=df_factor
            )
            if not test_results or 'metrics' not in test_results[0]:
                logger.warning(f"無有效回測結果，策略: {strat}, 數據源: {data_source}, 時段: {test_block}")
                continue
            metrics = test_results[0]['metrics']
            strategy_return = metrics.get('total_return', 0.0)
            test_start, test_end = test_block
            bh_return = buy_and_hold_return(df_price, test_start, test_end)
            excess_return = strategy_return - bh_return
            excess_returns.append(excess_return)
            oos_returns.append(strategy_return)
            if strategy_return <= bh_return : # or metrics.get('profit_factor', 0.1) < 0.5
                fail_count += 1
                # 簡化 SRA
                strategy_sharpe, bh_sharpe, p_value = compute_simplified_sra(df_price, trades, [test_block])
                sra_scores.append((strategy_sharpe, p_value))

        fail_ratio = fail_count / total_folds if total_folds > 0 else 0.0
        trial.set_user_attr("fail_ratio", fail_ratio)

        # 檢查 OOS 報酬標準差
        oos_std = np.std(oos_returns) if oos_returns else 0.0
        trial.set_user_attr("oos_std", oos_std)

        # 自定義 PBO 計算
        pbo_score = compute_pbo_score(oos_returns)
        trial.set_user_attr("pbo_score", pbo_score)
        logger.info(f"PBO 分數: {pbo_score:.3f}, 策略: {strat}, 數據源: {data_source}")
        log_to_results("PBO 計算", f"PBO 分數: {pbo_score:.3f}, 策略: {strat}, 數據源: {data_source}")

        # 動態門檻
        fail_ratio_threshold = 0.5 if total_ret > 10.0 else 0.5
        pbo_threshold = 0.7 if total_ret > 10.0 else 0.6
        sr_s = 1 - fail_ratio  # success_ratio = 1 - fail_ratio
        # 第二層: Min-Max Scaling
        tr_s = minmax(total_ret, 5, 25)
        pf_s = minmax(profit_factor, 0.5, 8)
        wf_s = minmax(min_wf_ret, 0, 2)
        sh_s = minmax(sharpe_ratio, 0.5, 0.8)
        mdd_s = 1 - minmax(abs(max_drawdown), 0, 0.3)
        st_s = minmax(avg_stress_ret, -1, 1)

        # 加權分數
        score = (
            SCORE_WEIGHTS["total_return"] * tr_s +
            SCORE_WEIGHTS["profit_factor"] * pf_s +
            SCORE_WEIGHTS["sharpe_ratio"] * sh_s +
            SCORE_WEIGHTS["max_drawdown"] * mdd_s +
            SCORE_WEIGHTS["wf_min_return"] * wf_s +
            #SCORE_WEIGHTS["stress_avg_return"] * st_s
            SCORE_WEIGHTS["success_ratio"] * sr_s
        )

        # 交易次數懲罰
        trade_penalty = 0.05 * max(0, MIN_NUM_TRADES - n_trades)
        score -= trade_penalty

        # 交易時機懲罰
        #if avg_buy_dist > 30 or avg_sell_dist > 30:
        #    score *= 0.999  # 扣 1% 分數
        #    logger.info(f"交易時機懲罰: 買入距離低點={avg_buy_dist:.1f}天, 賣出距離高點={avg_sell_dist:.1f}天, 分數調整={score:.3f}")
        #    log_to_results("交易時機懲罰", f"買入距離低點={avg_buy_dist:.1f}天, 賣出距離高點={avg_sell_dist:.1f}天, 分數調整={score:.3f}")

        # 超額報酬排名懲罰
        if excess_returns:
            excess_ranks = np.argsort(excess_returns)
            bottom_20_percent = int(0.2 * len(excess_returns))
            for idx in excess_ranks[:bottom_20_percent]:
                score *= 0.8  # 倒數 20% 扣 10% 分數
            logger.info(f"超額報酬懲罰: 倒數 20% fold 數={bottom_20_percent}, 分數調整={score:.3f}")
            log_to_results("超額報酬懲罰", f"倒數 20% fold 數={bottom_20_percent}, 分數調整={score:.3f}")

            # SRA 懲罰
            avg_p_value = np.mean([score[1] for score in sra_scores]) if sra_scores else 1.0
            trial.set_user_attr("sra_p_value", avg_p_value)
            #if avg_p_value > 0.1:
            #    score *= 0.9  # 扣 10% 分數
            #    logger.info(f"SRA 懲罰: p_value={avg_p_value:.3f}, 分數調整={score:.3f}")
            #    log_to_results("SRA 懲罰", f"p_value={avg_p_value:.3f}, 分數調整={score:.3f}")

        # CPCV 與 PBO 柔性懲罰
        if fail_ratio > fail_ratio_threshold or pbo_score > pbo_threshold:
            penalty = min(0.10, fail_ratio + pbo_score / 2)  # 最多扣 25% 分數
            score *= (1 - penalty)
            logger.info(f"柔性懲罰: fail_ratio={fail_ratio:.3f}, PBO={pbo_score:.3f}, 懲罰={penalty:.3f}, 分數調整={score:.3f}")
            log_to_results("柔性懲罰", f"fail_ratio={fail_ratio:.3f}, PBO={pbo_score:.3f}, 懲罰={penalty:.3f}, 分數調整={score:.3f}")
        elif fail_ratio > 0.1 or pbo_score > 0.3 or oos_std > 2.0:
            penalty = min(0.1, max(fail_ratio, pbo_score / 2, oos_std / 10))
            score *= (1 - penalty)
            logger.info(f"柔性懲罰: fail_ratio={fail_ratio:.3f}, PBO={pbo_score:.3f}, OOS 標準差={oos_std:.3f}, 懲罰={penalty:.3f}, 分數調整={score:.3f}")
            log_to_results("柔性懲罰", f"fail_ratio={fail_ratio:.3f}, PBO={pbo_score:.3f}, OOS 標準差={oos_std:.3f}, 懲罰={penalty:.3f}, 分數調整={score:.3f}")

        # KNN 過擬合懲罰
        stab = 0.0
        if len(trial_results) >= 15:
            stab = compute_knn_stability(trial_results, params=['linlen', 'smaalen', 'buy_mult'], k=5)
            if stab > 0.5:
                alpha = 0.2  # 降低懲罰係數
                penalty_mult = alpha * (stab - 0.5)
                score *= (1 - min(penalty_mult, 0.10))  # 最多扣 15% 分數
                logger.info(f"KNN 過擬合懲罰: 穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")
                log_to_results("KNN 過擬合懲罰", f"穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")

            # 儲存指標
            trial.set_user_attr("total_return", float(total_ret))
            trial.set_user_attr("total_return_scaled", tr_s)
            trial.set_user_attr("num_trades", n_trades)
            trial.set_user_attr("sharpe_ratio", float(sharpe_ratio))
            trial.set_user_attr("sharpe_ratio_scaled", sh_s)
            trial.set_user_attr("max_drawdown", float(max_drawdown))
            trial.set_user_attr("max_drawdown_scaled", mdd_s)
            trial.set_user_attr("profit_factor", float(profit_factor))
            trial.set_user_attr("profit_factor_scaled", pf_s)
            trial.set_user_attr("min_wf_return", float(min_wf_ret))
            trial.set_user_attr("min_wf_return_scaled", wf_s)
            trial.set_user_attr("avg_stress_return", float(avg_stress_ret))
            trial.set_user_attr("avg_stress_return_scaled", st_s)
            trial.set_user_attr("stability_score", float(stab))
            trial.set_user_attr("data_source", data_source)

            # 記錄試驗結果
            trial_details = (
                f"試驗 {str(trial.number).zfill(5)}, "
                f"策略: {strat}, "
                f"數據源: {data_source}, "
                f"分數: {score:.3f}, "
                f"參數: {params}, "
                f"總報酬={total_ret*100:.2f}%, "
                f"交易次數={n_trades}, "
                f"夏普比率={sharpe_ratio:.3f}, "
                f"最大回撤={max_drawdown:.3f}, "
                f"盈虧因子={profit_factor:.2f}, "
                f"WF 最差報酬={min_wf_ret:.2f}, "
                f"壓力平均報酬={avg_stress_ret:.3f}, "
                f"穩定性得分={stab:.2f}, "
                f"fail_ratio={fail_ratio:.2f}, "
                f"PBO 分數={pbo_score:.2f}, "
                f"SRA p-value={avg_p_value:.3f}, "
                f"買入距離低點={avg_buy_dist:.1f}天, "
                f"賣出距離高點={avg_sell_dist:.1f}天"
            )
            logger.info(trial_details)
            log_to_results(
                "試驗結果",
                trial_details,
                trial_number=str(trial.number).zfill(5),
                strategy=strat,
                data_source=data_source,
                score=score,
                total_return=total_ret,
                num_trades=n_trades,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                profit_factor=profit_factor,
                min_wf_return=min_wf_ret,
                avg_stress_return=avg_stress_ret,
                stability_score=stab,
                fail_ratio=fail_ratio,
                pbo_score=pbo_score,
                sra_p_value=avg_p_value,
                avg_buy_dist=avg_buy_dist,
                avg_sell_dist=avg_sell_dist
            )

            if trial.number % 100 == 0:
                # 繪製總報酬 vs fail_ratio 散點圖
                results = trial.study.user_attrs.get("trial_results", [])
                if results:
                    total_returns = [r["total_return"] for r in results]
                    fail_ratios = [r.get("fail_ratio", 0.0) for r in results]
                    plt.figure(figsize=(10, 6))
                    plt.scatter(fail_ratios, total_returns, alpha=0.5)
                    plt.xlabel('Fail Ratio')
                    plt.ylabel('Total Return')
                    plt.title(f'Total Return vs Fail Ratio - Trial {trial.number}')
                    plt.savefig(cfg.RESULT_DIR / f'return_vs_fail_ratio_trial_{trial.number}_{TIMESTAMP}.png')
                    plt.close()

    except Exception as e:
        logger.error(f"CPCV/OOS 計算失敗: {e}, 策略: {strat}, 數據源: {data_source}")
        # 備用邏輯：若 CPCV 失敗,僅使用基本指標和 KNN
        tr_s = minmax(total_ret, 5, 14)
        pf_s = minmax(profit_factor, 0.5, 8)
        wf_s = minmax(min_wf_ret, 0, 2)
        sh_s = minmax(sharpe_ratio, 0.5, 1.2)
        mdd_s = 1 - minmax(abs(max_drawdown), 0, 0.3)
        st_s = minmax(avg_stress_ret, -1, 1)

        score = (
            SCORE_WEIGHTS["total_return"] * tr_s +
            SCORE_WEIGHTS["profit_factor"] * pf_s +
            SCORE_WEIGHTS["sharpe_ratio"] * sh_s +
            SCORE_WEIGHTS["max_drawdown"] * mdd_s +
            SCORE_WEIGHTS["wf_min_return"] * wf_s 
            #SCORE_WEIGHTS["stress_avg_return"] * st_s
        )

        trade_penalty = 0.05 * max(0, MIN_NUM_TRADES - n_trades)
        score -= trade_penalty

        #if avg_buy_dist > 20 or avg_sell_dist > 20:
        #    score *= 0.95
        #    logger.info(f"交易時機懲罰: 買入距離低點={avg_buy_dist:.1f}天, 賣出距離高點={avg_sell_dist:.1f}天, 分數調整={score:.3f}")
        #    log_to_results("交易時機懲罰", f"買入距離低點={avg_buy_dist:.1f}天, 賣出距離高點={avg_sell_dist:.1f}天, 分數調整={score:.3f}")

        stab = 0.0
        if len(trial_results) >= 15:
            stab = compute_knn_stability(trial_results, params=['linlen', 'smaalen', 'buy_mult'], k=5)
            if stab > 0.5:
                alpha = 0.2
                penalty_mult = alpha * (stab - 0.5)
                score *= (1 - min(penalty_mult, 0.2))
                logger.info(f"KNN 過擬合懲罰: 穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")
                log_to_results("KNN 過擬合懲罰", f"穩定性得分={stab:.3f}, 懲罰乘數={penalty_mult:.3f}, 分數調整={score:.3f}")

        trial.set_user_attr("total_return", float(total_ret))
        trial.set_user_attr("total_return_scaled", tr_s)
        trial.set_user_attr("num_trades", n_trades)
        trial.set_user_attr("sharpe_ratio", float(sharpe_ratio))
        trial.set_user_attr("sharpe_ratio_scaled", sh_s)
        trial.set_user_attr("max_drawdown", float(max_drawdown))
        trial.set_user_attr("max_drawdown_scaled", mdd_s)
        trial.set_user_attr("profit_factor", float(profit_factor))
        trial.set_user_attr("profit_factor_scaled", pf_s)
        trial.set_user_attr("min_wf_return", float(min_wf_ret))
        trial.set_user_attr("min_wf_return_scaled", wf_s)
        trial.set_user_attr("avg_stress_return", float(avg_stress_ret))
        trial.set_user_attr("avg_stress_return_scaled", st_s)
        trial.set_user_attr("stability_score", float(stab))
        trial.set_user_attr("data_source", data_source)

        trial_details = (
            f"試驗 {str(trial.number).zfill(5)}, "
            f"策略: {strat}, "
            f"數據源: {data_source}, "
            f"分數: {score:.3f}, "
            f"參數: {params}, "
            f"總報酬={total_ret*100:.2f}%, "
            f"交易次數={n_trades}, "
            f"夏普比率={sharpe_ratio:.3f}, "
            f"最大回撤={max_drawdown:.3f}, "
            f"盈虧因子={profit_factor:.2f}, "
            f"WF 最差報酬={min_wf_ret:.2f}, "
            f"壓力平均報酬={avg_stress_ret:.3f}, "
            f"穩定性得分={stab:.2f}, "
            f"買入距離低點={avg_buy_dist:.1f}天, "
            f"賣出距離高點={avg_sell_dist:.1f}天"
        )
        logger.info(trial_details)
        log_to_results(
            "試驗結果（備用邏輯）",
            trial_details,
            trial_number=str(trial.number).zfill(5),
            strategy=strat,
            data_source=data_source,
            score=score,
            total_return=total_ret,
            num_trades=n_trades,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            min_wf_return=min_wf_ret,
            avg_stress_return=avg_stress_ret,
            stability_score=stab,
            avg_buy_dist=avg_buy_dist,
            avg_sell_dist=avg_sell_dist
        )
        return score

# ---------- 執行入口 ----------
if __name__ == "__main__":
    optuna_sqlite = Path(cfg.RESULT_DIR) / f"optuna_6_{TIMESTAMP}.sqlite3"
    study = optuna.create_study(
        study_name=f"00631L_optuna_v6_{args.strategy}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        storage=f"sqlite:///{optuna_sqlite}"
    )

    n_trials = args.n_trials
    logger.info(f"開始最佳化,策略: {args.strategy}, 共 {n_trials} 次試驗,交易次數範圍: {MIN_NUM_TRADES}-{MAX_NUM_TRADES}")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # 最佳試驗資訊
    logger.info("最佳試驗:")
    best = study.best_trial
    logger.info(f"策略: {best.user_attrs['strategy']}")
    logger.info(f"數據源: {best.user_attrs['data_source']}")
    logger.info(f"參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}")
    logger.info(f"穩健分數: {best.value:.3f}")
    logger.info(f"其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}")

    best_trial_details = (
        f"策略: {best.user_attrs['strategy']}, "
        f"數據源: {best.user_attrs['data_source']}, "
        f"參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}, "
        f"穩健分數: {best.value:.3f}, "
        f"其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}"
    )
    log_to_results("最佳試驗資訊", best_trial_details)

    # 提取每個策略的前 10 名試驗,並按「策略 + 數據源」分組挑選前 10 名
    strategies = [args.strategy] if args.strategy != 'all' else list(PARAM_SPACE.keys())
    data_sources = DATA_SOURCES
    trial_results = [entry for entry in results_log if entry["Event Type"] in ["試驗結果", "試驗結果（備用邏輯）"]]

    # 1. 各策略搭配各數據源挑選前 10 名
    for strategy in strategies:
        for data_source in data_sources:
            strategy_source_trials = [
                entry for entry in trial_results 
                if entry.get("strategy") == strategy and entry.get("data_source") == data_source
            ]
            if not strategy_source_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 無有效試驗結果")
                continue

            trial_scores = []
            for entry in strategy_source_trials:
                trial_num = entry.get("trial_number")
                score = entry.get("score")
                details = entry["Details"]
                try:
                    trial_scores.append((trial_num, float(score), details))
                except (ValueError, TypeError):
                    continue

            trial_scores.sort(key=lambda x: x[1], reverse=True)
            top_10_trials = trial_scores[:10]

            if top_10_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 前 10 名試驗:")
                for trial_num, score, details in top_10_trials:
                    logger.info(details)
                    log_to_results(f"前 10 名 {strategy} 搭配 {data_source} 試驗", details)

    # 2. 各策略搭配各數據源分組挑選前 5 名
    for strategy in strategies:
        for data_source in data_sources:
            strategy_source_trials = [
                entry for entry in trial_results 
                if entry.get("strategy") == strategy and entry.get("data_source") == data_source
            ]
            if not strategy_source_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 無有效試驗結果 (分組前 5 名)")
                continue

            trial_scores = []
            for entry in strategy_source_trials:
                trial_num = entry.get("trial_number")
                score = entry.get("score")
                details = entry["Details"]
                try:
                    trial_scores.append((trial_num, float(score), details))
                except (ValueError, TypeError):
                    continue

            trial_scores.sort(key=lambda x: x[1], reverse=True)
            top_5_trials = trial_scores[:5]

            if top_5_trials:
                logger.info(f"策略 {strategy} 與數據源 {data_source} 分組前 5 名試驗:")
                for trial_num, score, details in top_5_trials:
                    logger.info(details)
                    log_to_results(f"前 5 名 {strategy} 搭配 {data_source} 分組試驗", details)

    # 保存最佳參數到 JSON
    results = {
        "best_robust_score": best.value,
        "best_strategy": best.user_attrs["strategy"],
        "best_data_source": best.user_attrs["data_source"],
        "best_params": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.params.items() if k not in ["strategy", "data_source"]},
        "best_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.user_attrs.items()}
    }
    result_file = cfg.RESULT_DIR / f"optuna_6_best_params_{TICKER.replace('^','')}_{TIMESTAMP}.json"
    pd.Series(results).to_json(result_file, indent=2)
    logger.info(f"最佳參數已保存至 {result_file}")

    # 將 results_log 保存為 CSV
    df_results = pd.DataFrame(results_log)
    result_csv_file = cfg.RESULT_DIR / f"optuna_6_results_{TIMESTAMP}.csv"
    df_results.to_csv(result_csv_file, index=False, encoding='utf-8-sig')
    logger.info(f"重要訊息已保存至 {result_csv_file}")

    # 打印前 5 筆 results_log 內容
    logger.info("前 5 筆試驗記錄:")
    for record in results_log[:5]:
        logger.info(f"[{record['Timestamp']}] - {record['Event Type']} - {record['Details']}")

    # 將結果轉回列表並範例查詢
    results_list = load_results_to_list(result_csv_file)
    logger.info(f"從 {result_csv_file} 載入 {len(results_list)} 筆記錄")
    # 範例查詢：篩選 RMA 策略的試驗
    rma_trials = [r for r in results_list if r.get('strategy') == 'RMA' and r['Event Type'] in ["試驗結果", "試驗結果（備用邏輯）"]]
    logger.info(f"RMA 策略試驗數量：{len(rma_trials)}")
    if rma_trials:
        logger.info(f"第一筆 RMA 試驗：{rma_trials[0]['Details']}")
