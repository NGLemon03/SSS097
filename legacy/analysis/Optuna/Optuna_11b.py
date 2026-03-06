# ** coding: utf-8 **
'''
Optuna-based hyper-parameter optimization for 00631L strategies (Version 10, adjusted)
--------------------------------------------------
* 最佳化目標: 優先追求完整回測報酬(>300%)、交易次數(15-60次),允許輕微過擬合,其次考慮 walk_forward 期間最差報酬、夏普比率、最大回撤,最後考慮壓力測試平均報酬.
* 本腳本可直接放在 analysis/ 目錄後以 'python {version}.py' 執行.
* 搜尋空間與權重在 'PARAM_SPACE' 與 'SCORE_WEIGHTS' 中設定,方便日後微調.
* 使用 sklearn 的 TimeSeriesSplit 替代 mlfinlab 的 CPCV, 自定義 PBO 函數, 確保策略在 OOS 期間穩定表現, 並分析交易時機.
v4  數據源,策略分流: 透過 WEIGHTS 設定抽樣比例,增加 WF、Sharpe、MDD 指標,Min-Max Scaling 使高報酬更具區分度.過擬合懲罰
v5  修正日誌衝突: 移除全局日誌處理器覆蓋, 改用獨立日誌模組 (logging_config)；確保多版並行不互相影響.
v6  修正數據載入問題: 統一預載數據, 避免重複加載導致 SMAA 快取不一致.
    CPCV + PBO: 使用 total_return 作主要 metric, 並同時檢查 profit_factor、max_drawdown、OOS 標準差.
    SRA 簡化: 比較策略與 buy-and-hold 的 Sharpe Ratio, p-value >0.05 扣 10% 分數.
    交易時機分析: 計算「買/賣距離高低點天數」, 超過門檻扣分.
    交易日限制: 確保所有時序切割只在交易日索引上執行.
v7  修正壓力測試期間: 新增交易日檢查, 確保 valid_stress_periods 非空, 避免無效或空白計算導致指標失真.
v8  修正 _backtest_once 返回結構: 確保所有分支統一回傳 6+1 欄()含 equity_curve), 避免解包錯誤.
v9  修正 Equity Curve 缺失: 新增 equity_curve 傳遞與驗證, 修復壓力測試相關函式計算.
    加入多散點圖: 分別繪製 Sharpe vs Return、MDD vs Return, 視覺化各策略群表現.
fix 改進 Stress MDD 計算: 改用 Excess Return in Stress 取代 fail_ratio
    Robust Score 計算
    錯誤處理強化: 調整 CSV/JSON 輸出邏輯, 避免空值與索引錯誤.
v10 支援單一數據源模式 (fixed)、隨機 (random)、依序遍歷 (sequential) 三種選擇.
    新增平均持倉天數指標: 回測時計算 avg_hold_days, 並可做硬篩或懲罰.
    參數‐指標相關係數分析: 對各策略試驗結果計算皮爾森相關係數, 並輸出 CSV/熱圖.
    與最佳化參數作相關性測試: 將最終 trial_results 與 best_params 做交叉檢驗.
* 命令列範例: 
# 隨機選數據源
python optuna_10.py --strategy RMA --n_trials 10000  
# 指定數據源
python optuna_10.py --strategy single --data_source "Factor (^TWII / 2412.TW)" --data_source_mode fixed --n_trials 1000  
# 依序遍歷所有數據源
python optuna_10.py --strategy ssma_turn --data_source_mode sequential --n_trials 2000  
'''
from typing import Tuple, List, Dict, Optional
import logging
from logging_config import setup_logging
import optuna
import numpy as np
import re
import sys
import pandas as pd
import shutil
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import argparse
from pathlib import Path
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from scipy.spatial.distance import cdist
import ast
from metrics import calculate_sharpe, calculate_max_drawdown, calculate_profit_factor
# 版本與紀錄
version = 'optuna_10'
setup_logging()
logger = logging.getLogger('optuna_10')
# 資料過濾
pct_threshold_self = 10
setlimit = False
setminsharpe = 0.45
setmaxsharpe = 0.75
setminmdd = -0.2
setmaxmdd = -0.4
# 設定 matplotlib 字體
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 優先使用中文支援字體
plt.rcParams['axes.unicode_minus'] = False
# 
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import config as cfg
from analysis import data_loader
import SSSv096 as SSS
from SSSv096 import load_data, compute_ssma_turn_combined, compute_single, compute_dual, compute_RMA, backtest_unified

parser = argparse.ArgumentParser(description='Optuna 最佳化 00631L 策略')
parser.add_argument('--strategy', type=str, choices=['single', 'dual', 'RMA', 'ssma_turn', 'all'], default='all', help='指定單一策略進行最佳化 (預設: all)')
parser.add_argument('--n_trials', type=int, default=5000, help='試驗次數 (預設: 5000)')
parser.add_argument('--data_source', type=str, choices=cfg.SOURCES, default=None, help='指定單一數據源, 僅在 --data_source_mode=fixed 時有效')
parser.add_argument('--data_source_mode', type=str, choices=['random', 'fixed', 'sequential'], default='random', help='數據源選擇模式: random(隨機)、fixed(指定)、sequential(依序遍歷)')
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
results_log = []
events_log = []
top_trials = []  # 儲存分數前 20 名試驗的 equity_curve

# 新增多樣性篩選函數
def pick_topN_by_diversity(trials: List[Dict], ind_keys: List[str], top_n: int = 20, pct_threshold: int = 25) -> List[Dict]:
    """
    先按 score 排序，再用歐氏距離過濾參數相似度。
    
    Args:
        trials: 試驗結果列表，每個試驗包含 score 和 parameters。
        ind_keys: 用於計算距離的參數鍵。
        top_n: 最終選取的試驗數量。
        pct_threshold: 距離門檻的百分位數。
    
    Returns:
        List[Dict]: 篩選後的試驗列表。
    """
    trials_sorted = sorted(trials, key=lambda t: -t["score"])
    vectors = []
    chosen = []
    for tr in trials_sorted:
        vec = np.array([tr["parameters"].get(k, 0) for k in ind_keys])
        if not vectors:
            vectors.append(vec)
            chosen.append(tr)
            continue
        dists = cdist([vec], vectors, metric="euclidean").ravel()
        if np.min(dists) >= np.percentile(dists, pct_threshold):
            vectors.append(vec)
            chosen.append(tr)
        if len(chosen) == top_n:
            break
    return chosen

def sanitize(data: str) -> str:
    """
    過濾檔案名稱中的特殊字符，將非字母數字字符替換為單一底線。
    Args:
        data: 要過濾的字符串。
    Returns:
        str: 安全的檔案名稱字符串。
    """
    # 將所有非字母數字字符替換為底線
    name = re.sub(r'[^0-9A-Za-z]', '_', data)
    # 移除連續底線
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def _avg_holding_days(trades):
    """
    計算交易的平均持倉天數.
    Args:
        trades: 交易列表, 每筆交易為 (entry_date, return, exit_date).
    Returns:
        float: 平均持倉天數, 若無交易則返回 0.0.
    """
    if not trades:
        return 0.0
    d = [(pd.Timestamp(t[2]) - pd.Timestamp(t[0])).days
         for t in trades if len(t) > 2]
    return float(np.mean(d)) if d else 0.0

def penalize_hold(avg_hold_days: float, target: float = 150, span: float = 90, max_penalty: float = 0.3) -> float:
    """
    計算平均持倉天數的懲罰值, 目標範圍 60 - 240 天.
    Args:
        avg_hold_days: 平均持倉天數.
        target: 目標持倉天數(中心值) = 150.
        span: 容忍範圍(±span) = 90.
        max_penalty: 最大懲罰比例 = 0.3.
    Returns:
        float: 懲罰值 = 0 ~ 0.3 (max_penalty)
    """
    penalty = max(0, abs(avg_hold_days - target) - span) / span
    return min(penalty * max_penalty, max_penalty)



def log_to_results(event_type: str, details: str, **kwargs):
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
    "single": dict(
        linlen=(5, 240, 1),
        smaalen=(7, 240, 5),
        devwin=(5, 180, 1),
        factor=(40, 40, 1),
        buy_mult=(0.1, 2.5, 0.05),
        sell_mult=(0.5, 4.0, 0.05),
        stop_loss=(0.00, 0.55, 0.2),
    ),
    "dual": dict(
        linlen=(5, 240, 1),
        smaalen=(7, 240, 5),
        short_win=(10, 100, 5),
        long_win=(40, 240, 10),
        factor=(40, 40, 1),
        buy_mult=(0.2, 2, 0.05),
        sell_mult=(0.5, 4.0, 0.05),
        stop_loss=(0.00, 0.55, 0.1),
    ),
    "RMA": dict(
        linlen=(5, 240, 1),
        smaalen=(7, 240, 5),
        rma_len=(20, 100, 5),
        dev_len=(10, 100, 5),
        factor=(40, 40, 1),
        buy_mult=(0.2, 2, 0.05),
        sell_mult=(0.5, 4.0, 0.05),
        stop_loss=(0.00, 0.55, 0.1),
    ),
    "ssma_turn": dict(
        linlen=(10, 240, 5),
        smaalen=(10, 240, 5),
        factor=(40.0, 40.0, 1),
        prom_factor=(5, 70, 1),
        min_dist=(5, 20, 1),
        buy_shift=(0, 7, 1),
        exit_shift=(0, 7, 1),
        vol_window=(5, 90, 5),
        quantile_win=(5, 180, 10),
        signal_cooldown_days=(1, 7, 1),
        buy_mult=(0.5, 2, 0.05),
        sell_mult=(0.2, 3, 0.1),
        stop_loss=(0.00, 0.55, 0.1),
    ),
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
        logger.error(f"買入並持有計算錯誤: {e}")
        return 1.0

def get_fold_period(test_blocks: list) -> tuple:
    starts = [b[0] for b in test_blocks]
    ends = [b[1] for b in test_blocks]
    return min(starts), max(ends)

def analyze_trade_timing(df_price: pd.DataFrame, trades: list, window=20) -> tuple:
    buy_distances = []
    sell_distances = []
    for t in trades:
        entry, _, exit = t[0], t[1], t[2] if len(t) > 2 else (None, None, None)
        if not entry or not exit or entry not in df_price.index or exit not in df_price.index:
            logger.warning(f"無效交易日期: entry={entry}, exit={exit}")
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

def compute_simplified_sra(df_price: pd.DataFrame, trades: list, test_blocks: list) -> tuple:
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
    if not df_results or not isinstance(df_results[0], dict):
        logger.warning("試驗結果格式無效，無法計算 KNN 穩定性")
        return 0.0
    param_cols = [f"param_{p}" for p in params if f"param_{p}" in df_results[0]]
    if not param_cols:
        logger.warning(f"無有效參數用於 KNN 穩定性計算, 參數: {params}, 可用欄位: {list(df_results[0].keys())}")
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

def equity_period_return(equity: pd.Series, start: str, end: str) -> float:
    """計算指定期間的 Equity Curve 報酬率."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start not in equity.index or end not in equity.index:
        logger.warning(f"Equity Curve 未涵蓋 {start} → {end}")
        return np.nan
    return equity.loc[end] / equity.loc[start] - 1

def compute_stress_mdd_equity(equity: pd.Series, stress_periods: list) -> float:
    """計算壓力期間的最大回撤, 使用完整 equity curve."""
    mdds = []
    for start, end in stress_periods:
        if start not in equity.index or end not in equity.index:
            logger.warning(f"壓力測試期間無效: {start} → {end}")
            continue
        period_equity = equity.loc[start:end]
        if period_equity.empty or len(period_equity) < 2:
            logger.warning(f"壓力測試期間 {start} → {end} 數據不足")
            mdds.append(np.nan)
            continue
        drawdown = (period_equity / period_equity.cummax()) - 1
        mdds.append(drawdown.min() if not drawdown.empty else np.nan)
    return np.nanmean(mdds) if mdds and not all(np.isnan(mdds)) else np.nan

def compute_stress_mdd_equity(equity: pd.Series, stress_periods: list) -> float:
    mdds = []
    for start, end in stress_periods:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start not in equity.index or end not in equity.index:
            logger.warning(f"壓力測試期間無效: {start} → {end}, 跳過")
            continue
        period_equity = equity.loc[start:end]
        if period_equity.empty or len(period_equity) < 2:
            logger.warning(f"壓力測試期間 {start} → {end} 數據不足，長度={len(period_equity)}")
            mdds.append(np.nan)
            continue
        drawdown = (period_equity / period_equity.cummax()) - 1
        mdds.append(drawdown.min() if not drawdown.empty else np.nan)
    if not mdds or all(np.isnan(mdds)):
        logger.warning("無有效壓力期間 MDD 數據")
        return np.nan
    return np.nanmean(mdds)

def compute_excess_return_in_stress_equity(equity: pd.Series, df_price: pd.DataFrame, stress_periods: list) -> float:
    excess = []
    for start, end in stress_periods:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start not in equity.index or end not in equity.index or start not in df_price.index or end not in df_price.index:
            logger.warning(f"壓力測試期間無效: {start} → {end}, 跳過")
            continue
        strat_ret = equity_period_return(equity, start, end)
        bh_ret = buy_and_hold_return(df_price, start, end)
        if not np.isnan(strat_ret) and not np.isnan(bh_ret):
            excess.append(strat_ret - bh_ret)
        else:
            logger.warning(f"壓力測試期間 {start} → {end} 報酬計算無效")
            excess.append(np.nan)
    if not excess or all(np.isnan(excess)):
        logger.warning("無有效壓力期間超額報酬數據")
        return np.nan
    return np.nanmean(excess)

def _backtest_once(strat: str, params: dict, trial_results: list, data_source: str, df_price: pd.DataFrame, df_factor: pd.DataFrame) -> Tuple[float, int, float, float, float, List, pd.Series, float]:
    """
    執行單次回測, 返回 8 項結果: total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades, equity_curve, avg_hold_days.
    ⚠ 所有 return 語句必須固定 8 項, 避免解包錯誤.
    """
    try:
        if df_price.empty:
            logger.error(f"價格數據為空, 策略: {strat}, 數據源: {data_source}")
            log_to_results("錯誤", f"價格數據為空, 策略: {strat}, 數據源: {data_source}")
            params_flat = {f"param_{k}": v for k, v in params.items()}
            trial_results.append({
                "total_return": -np.inf,
                "num_trades": 0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "data_source": data_source,
                "avg_hold_days": 0.0,
                "stress_mdd": None,
                "excess_return_stress": None,
                "parameters": params,
                "score": -np.inf,
                **params_flat
            })
            return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0
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
                logger.warning(f"計算指標失敗, 策略: {strat}")
                log_to_results("警告", f"計算指標失敗, 策略: {strat}")
                params_flat = {f"param_{k}": v for k, v in params.items()}
                trial_results.append({
                    "total_return": -np.inf,
                    "num_trades": 0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                    "data_source": data_source,
                    "avg_hold_days": 0.0,
                    "stress_mdd": None,
                    "excess_return_stress": None,
                    "parameters": params,
                    "score": -np.inf,
                    **params_flat
                })
                return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0
            bt = SSS.backtest_unified(df_ind=df_ind, strategy_type=strat, params=params, buy_dates=buys, sell_dates=sells, discount=COST_PER_SHARE / 100, trade_cooldown_bars=COOLDOWN_BARS)
        else:
            df_ind = compute_f(df_price, df_factor, **ind_p)
            if df_ind.empty:
                logger.warning(f"計算指標失敗, 策略: {strat}")
                log_to_results("警告", f"計算指標失敗, 策略: {strat}")
                params_flat = {f"param_{k}": v for k, v in params.items()}
                trial_results.append({
                    "total_return": -np.inf,
                    "num_trades": 0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                    "data_source": data_source,
                    "avg_hold_days": 0.0,
                    "stress_mdd": None,
                    "excess_return_stress": None,
                    "parameters": params,
                    "score": -np.inf,
                    **params_flat
                })
                return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0
            bt = SSS.backtest_unified(df_ind=df_ind, strategy_type=strat, params=params, discount=COST_PER_SHARE / 100, trade_cooldown_bars=COOLDOWN_BARS)
        if not bt or "metrics" not in bt or any(pd.isna(bt["metrics"].get(k)) for k in ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor"]):
            logger.error(f"回測結果無效或指標缺失, 策略: {strat}, bt: {bt}")
            params_flat = {f"param_{k}": v for k, v in params.items()}
            trial_results.append({
                "total_return": -np.inf,
                "num_trades": 0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "data_source": data_source,
                "avg_hold_days": 0.0,
                "stress_mdd": None,
                "excess_return_stress": None,
                "parameters": params,
                "score": -np.inf,
                **params_flat
            })
            return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0
        equity_curve = bt.get("equity_curve", pd.Series(index=df_price.index, data=100000.0))
        metrics = bt["metrics"]
        trades_df = bt.get("trades_df", [])
        if trades_df is None:
            logger.warning(f"回測未返回 trades_df, 策略: {strat}")
            params_flat = {f"param_{k}": v for k, v in params.items()}
            trial_results.append({
                "total_return": -np.inf,
                "num_trades": 0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "data_source": data_source,
                "avg_hold_days": 0.0,
                "stress_mdd": None,
                "excess_return_stress": None,
                "parameters": params,
                "score": -np.inf,
                **params_flat
            })
            return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0

        sharpe_ratio = calculate_sharpe(equity_curve.pct_change())
        max_drawdown = calculate_max_drawdown(equity_curve)
        profit_factor = calculate_profit_factor(bt.get("trades", []))
        avg_hold_days = _avg_holding_days(bt.get("trades", []))
        params_flat = {f"param_{k}": v for k, v in params.items()}
        trial_results.append({
            "total_return": metrics.get("total_return", 0.0),
            "num_trades": metrics.get("num_trades", 0),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "data_source": data_source,
            "avg_hold_days": avg_hold_days,
            "stress_mdd": None,
            "excess_return_stress": None,
            "parameters": params,
            "score": -np.inf,
            **params_flat
        })
        logger.info(f"回測完成, 策略: {strat}, 總報酬={metrics.get('total_return', 0.0)*100:.2f}%, 交易次數={metrics.get('num_trades', 0)}, 平均持倉天數={avg_hold_days:.1f}天")
        return (metrics.get("total_return", 0.0), metrics.get("num_trades", 0), sharpe_ratio,
                max_drawdown, profit_factor, bt.get("trades", []),
                bt.get("equity_curve", pd.Series()), avg_hold_days)
    except Exception as e:
        logger.error(f"回測失敗, 策略: {strat}, 錯誤: {e}")
        params_flat = {f"param_{k}": v for k, v in params.items()}
        trial_results.append({
            "total_return": -np.inf,
            "num_trades": 0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "data_source": data_source,
            "avg_hold_days": 0.0,
            "stress_mdd": None,
            "excess_return_stress": None,
            "parameters": params,
            "score": -np.inf,
            **params_flat
        })
        return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0

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



def objective(trial: optuna.Trial):
    global top_trials
    import heapq
    import numpy as np
    import pandas as pd
    from datetime import timedelta

    # 選擇策略
    if args.strategy == 'all':
        strat = np.random.choice(list(STRATEGY_WEIGHTS.keys()), p=list(STRATEGY_WEIGHTS.values()))
    else:
        strat = args.strategy
    trial.set_user_attr("strategy", strat)

    # 選擇數據源
    if args.data_source_mode == 'sequential':
        data_source = trial.study.user_attrs["data_source"]
    else:
        data_source = trial.study.user_attrs.get("data_source")
        if not data_source:
            if args.data_source_mode == 'fixed':
                if not args.data_source:
                    logger.error("固定數據源模式下必須指定 --data_source")
                    raise ValueError("缺少 --data_source 參數")
                data_source = args.data_source
            else:
                data_source = np.random.choice(list(DATA_SOURCES_WEIGHTS.keys()), p=list(DATA_SOURCES_WEIGHTS.values()))
    trial.set_user_attr("data_source", data_source)

    # 採樣參數
    params = _sample_params(trial, strat)
    trial_results = trial.study.user_attrs.get("trial_results", [])
    if len(trial_results) > 1000:
        trial_results = trial_results[-500:]  # 限制記憶體使用
        trial.study.set_user_attr("trial_results", trial_results)

    # 載入數據
    df_price, df_factor = data_loader.load_data(TICKER, start_date=cfg.START_DATE, end_date="2025-06-06", smaa_source=data_source)
    if df_price.empty:
        logger.error(f"價格數據為空, 策略: {strat}")
        return -np.inf
    if trial.number == 0:
        safe_ds = sanitize(data_source)
        out_csv = cfg.RESULT_DIR / f"optuna_factor_data_{strat}_{TIMESTAMP}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_factor.to_csv(out_csv)

    # 驗證壓力測試期間
    valid_dates = df_price.index
    valid_stress_periods = []
    warmup_days = max(params.get('linlen', 60), params.get('smaalen', 60), params.get('vol_window', 60), params.get('quantile_win', 60))
    for start, end in STRESS_PERIODS:
        start_date = pd.Timestamp(start) - timedelta(days=warmup_days)
        end_date = pd.Timestamp(end)
        if start_date < df_price.index[0] or end_date > df_price.index[-1]:
            logger.warning(f"壓力測試期間超出數據範圍: {start} → {end}, 跳過")
            continue
        start_candidates = valid_dates[valid_dates >= start_date]
        end_candidates = valid_dates[valid_dates <= end_date]
        if start_candidates.empty or end_candidates.empty:
            logger.warning(f"跳過無效壓力測試期間: {start} → {end}")
            continue
        adjusted_start = start_candidates[0]
        adjusted_end = end_candidates[-1]
        if (adjusted_end - adjusted_start).days < warmup_days * 2:
            logger.warning(f"壓力測試期間過短: {adjusted_start} → {adjusted_end}")
            continue
        logger.info(f"有效壓力測試期間: {adjusted_start} → {adjusted_end}")
        valid_stress_periods.append((adjusted_start, adjusted_end))
    if not valid_stress_periods:
        logger.error(f"無有效壓力測試時段, 策略: {strat}")
        log_to_results("錯誤", f"無有效壓力測試時段, 策略: {strat}")
        return -np.inf

    # 執行回測
    (total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades,
     equity_curve, avg_hold_days) = _backtest_once(strat, params, trial_results, data_source, df_price, df_factor)

    trial.set_user_attr("avg_hold_days", avg_hold_days)
    trial.set_user_attr("equity_curve_json", equity_curve.to_json())

    if total_ret == -np.inf or not (MIN_NUM_TRADES <= n_trades <= MAX_NUM_TRADES) or total_ret <= 2.0 or max_drawdown < -0.5 or profit_factor < 0.5:
        logger.info(f"試驗被剔除: 條件未滿足, 策略: {strat}, 總報酬={total_ret*100:.2f}%, 交易次數={n_trades}")
        log_to_results("試驗被剔除", f"條件未滿足, 策略: {strat}, 總報酬={total_ret*100:.2f}%, 交易次數={n_trades}")
        trial_results.append({
            "trial_number": str(trial.number).zfill(5),
            "parameters": params,
            "score": -np.inf,
            "equity_curve_json": equity_curve.to_json(),
            "strategy": strat,
            "data_source": data_source
        })
        trial.study.set_user_attr("trial_results", trial_results)
        return -np.inf

    # 檢查試驗唯一性並追加結果
    params_flat = {f"param_{k}": v for k, v in params.items()}
    min_wf_ret = _wf_min_return(strat, params, data_source)
    if pd.isna(min_wf_ret):
        logger.warning(f"Walk-forward 測試無效, 策略: {strat}")
        trial_results.append({
            "trial_number": str(trial.number).zfill(5),
            "parameters": params,
            "score": -np.inf,
            "equity_curve_json": equity_curve.to_json(),
            "strategy": strat,
            "data_source": data_source
        })
        trial.study.set_user_attr("trial_results", trial_results)
        return -np.inf
    avg_stress_ret = _stress_avg_return(strat, params, data_source, valid_stress_periods, df_price, trades)
    avg_buy_dist, avg_sell_dist = analyze_trade_timing(df_price, trades)
    trial.set_user_attr("avg_buy_dist", avg_buy_dist)
    trial.set_user_attr("avg_sell_dist", avg_sell_dist)

    try:
        if df_price.empty:
            logger.error(f"CPCV 價格數據為空, 策略: {strat}")
            return -np.inf
        event_times = [(t[0], t[0]) for t in trades if t[0]]
        if not event_times:
            logger.info(f"試驗被剔除: 無交易記錄, 策略: {strat}")
            trial_results.append({
                "trial_number": str(trial.number).zfill(5),
                "parameters": params,
                "score": -np.inf,
                "equity_curve_json": equity_curve.to_json(),
                "strategy": strat,
                "data_source": data_source
            })
            trial.study.set_user_attr("trial_results", trial_results)
            return -np.inf
        n_splits = min(CPCV_NUM_SPLITS, len(df_price) // (CPCV_EMBARGO_DAYS + 60))
        min_test_len = (params.get('smaalen', 60) + params.get('quantile_win', 60)) * 1.5
        if len(df_price) / n_splits < min_test_len:
            n_splits = max(3, int(len(df_price) // min_test_len))
            logger.info(f"調整 CPCV 分割數至 {n_splits}")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        folds = []
        valid_dates = df_price.index
        for train_idx, test_idx in tscv.split(df_price):
            train_start = df_price.index[train_idx[0]]
            train_end = df_price.index[train_idx[-1]]
            test_start = df_price.index[test_idx[0]] + pd.Timedelta(days=CPCV_EMBARGO_DAYS)
            test_end = df_price.index[test_idx[-1]]
            test_start_candidates = valid_dates[valid_dates >= test_start]
            test_end_candidates = valid_dates[valid_dates <= test_end]
            if test_start_candidates.empty or test_end_candidates.empty:
                logger.warning(f"跳過無效 fold: test_start={test_start}, test_end={test_end}")
                continue
            adjusted_start = test_start_candidates[0]
            adjusted_end = test_end_candidates[-1]
            folds.append(([train_start, train_end], [adjusted_start, adjusted_end]))
        if setlimit:
            if not (setminsharpe <= sharpe_ratio <= setmaxsharpe and setmaxmdd <= max_drawdown <= setminmdd):
                logger.info(f"試驗被剔除: Sharpe Ratio ({sharpe_ratio:.3f}) 或 MDD ({max_drawdown:.3f}) 未達標, 策略: {strat}")
                log_to_results("試驗被剔除", f"Sharpe Ratio 或 MDD 未達標, 策略: {strat}, Sharpe={sharpe_ratio:.3f}, MDD={max_drawdown:.3f}")
                trial_results.append({
                    "trial_number": str(trial.number).zfill(5),
                    "parameters": params,
                    "score": -np.inf,
                    "equity_curve_json": equity_curve.to_json(),
                    "strategy": strat,
                    "data_source": data_source
                })
                trial.study.set_user_attr("trial_results", trial_results)
                return -np.inf
        oos_returns = []
        excess_returns = []
        sra_scores = []
        for train_block, test_block in folds:
            test_results = SSS.compute_backtest_for_periods(ticker=TICKER, periods=[(test_block[0], test_block[1])], strategy_type=strat, params=params, trade_cooldown_bars=COOLDOWN_BARS, discount=COST_PER_SHARE / 100, df_price=df_price, df_factor=df_factor, smaa_source=data_source)
            if not test_results or 'metrics' not in test_results[0]:
                logger.warning(f"無有效回測結果, 策略: {strat}, 時段: {test_block}")
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
            logger.warning(f"無有效 excess_returns, 策略: {strat}")
            trial_results.append({
                "trial_number": str(trial.number).zfill(5),
                "parameters": params,
                "score": -np.inf,
                "equity_curve_json": equity_curve.to_json(),
                "strategy": strat,
                "data_source": data_source
            })
            trial.study.set_user_attr("trial_results", trial_results)
            return -np.inf
        excess_return_mean = np.mean(excess_returns)
        stress_mdd = compute_stress_mdd_equity(equity_curve, valid_stress_periods)
        excess_return_stress = compute_excess_return_in_stress_equity(equity_curve, df_price, valid_stress_periods)
        pbo_score = compute_pbo_score(oos_returns)
        trial.set_user_attr("total_return", total_ret)
        trial.set_user_attr("max_drawdown", max_drawdown)
        trial.set_user_attr("sharpe_ratio", sharpe_ratio)
        trial.set_user_attr("stress_mdd", stress_mdd)
        trial.set_user_attr("excess_return", excess_return_mean)
        trial.set_user_attr("excess_return_stress", excess_return_stress)
        trial.set_user_attr("pbo_score", pbo_score)

        # 更新 trial_results
        trial_results.append({
            "trial_number": str(trial.number).zfill(5),
            "parameters": params,
            "score": -np.inf,  # 稍後更新
            "equity_curve_json": equity_curve.to_json(),
            "strategy": strat,
            "data_source": data_source,
            "total_return": total_ret,
            "num_trades": n_trades,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "avg_hold_days": avg_hold_days,
            "stress_mdd": stress_mdd,
            "excess_return_stress": excess_return_stress
        })
        trial.study.set_user_attr("trial_results", trial_results)

        logger.info(f"PBO 分數: {pbo_score:.3f}, 策略: {strat}")
        log_to_results("PBO 計算", f"PBO 分數: {pbo_score:.3f}, 策略: {strat}")

        # Min-Max Scaling
        tr_s = minmax(total_ret, 5, 25)
        pf_s = minmax(profit_factor, 0.5, 8)
        wf_s = minmax(min_wf_ret, 0, 2)
        sh_s = minmax(sharpe_ratio, 0.5, 0.8)
        mdd_s = 1 - minmax(abs(max_drawdown), 0, 0.3)
        excess_return_scaled = minmax(excess_return_mean, -1, 1)
        stress_mdd_scaled = minmax(abs(stress_mdd), 0, 0.5) if not np.isnan(stress_mdd) else 0.0
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

        # 平均持倉懲罰
        penalty = penalize_hold(avg_hold_days)
        score *= (1 - penalty)
        logger.info(f"持倉懲罰: 平均持倉={avg_hold_days:.1f}天, 懲罰={penalty:.3f}, 分數調整={score:.3f}")
        log_to_results("持倉懲罰", f"平均持倉={avg_hold_days:.1f}天, 懲罰={penalty:.3f}, 分數調整={score:.3f}")

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

        # 更新 top_trials
        trial_entry = {
            "score": score,
            "equity_curve_json": equity_curve.to_json(),
            "trial_number": str(trial.number).zfill(5),
            "strategy": strat,
            "data_source": data_source
        }
        if len(top_trials) < 20:
            heapq.heappush(top_trials, (score, trial.number, trial_entry))
        else:
            heapq.heappushpop(top_trials, (score, trial.number, trial_entry))
        top_trials.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # 更新 trial_results 的 score
        trial_results[-1]["score"] = score
        trial.study.set_user_attr("trial_results", trial_results)

        # 構建試驗細節
        trial_details = (
            f"試驗 {str(trial.number).zfill(5)}, 策略: {strat}, 數據源: {data_source}, 分數: {score:.3f}, "
            f"參數: {params}, 總報酬={total_ret*100:.2f}%, 交易次數={n_trades}, 夏普比率={sharpe_ratio:.3f}, "
            f"最大回撤={max_drawdown:.3f}, 盈虧因子={profit_factor:.2f}, WF最差報酬(分段)={min_wf_ret:.2f}, "
            f"壓力平均報酬(整段)={avg_stress_ret:.3f}, 穩定性得分={stab:.2f}, Robust Score={robust_score:.3f}, "
            f"Excess Return in Stress={excess_return_stress:.3f}, Stress MDD={stress_mdd:.3f}, "
            f"PBO 分數={pbo_score:.2f}, SRA p-value={avg_p_value:.3f}, 買入距離低點={avg_buy_dist:.1f}天, "
            f"賣出距離高點={avg_sell_dist:.1f}天, 平均持倉天數={avg_hold_days:.1f}天"
        )
        logger.info(trial_details)

        # 記錄 log_data
        log_data = {
            "trial_number": str(trial.number).zfill(5),
            "score": score,
            **params_flat,
            "parameters": str(params),
            "equity_curve_json": equity_curve.to_json(),
            "total_return": f"{total_ret*100:.2f}%",
            "raw_total_return": total_ret,
            "num_trades": n_trades,
            "sharpe_ratio": f"{sharpe_ratio:.3f}",
            "raw_sharpe_ratio": sharpe_ratio,
            "max_drawdown": f"{max_drawdown:.3f}",
            "raw_max_drawdown": max_drawdown,
            "profit_factor": f"{profit_factor:.2f}",
            "raw_profit_factor": profit_factor,
            "min_wf_return": f"{min_wf_ret:.2f}",
            "raw_min_wf_return": min_wf_ret,
            "avg_stress_return": f"{avg_stress_ret:.3f}",
            "raw_avg_stress_return": avg_stress_ret,
            "stability_score": f"{stab:.2f}",
            "raw_stability_score": stab,
            "robust_score": f"{robust_score:.3f}",
            "raw_robust_score": robust_score,
            "excess_return_stress": f"{excess_return_stress:.3f}",
            "raw_excess_return_stress": excess_return_stress,
            "stress_mdd": f"{stress_mdd:.3f}",
            "raw_stress_mdd": stress_mdd,
            "pbo_score": f"{pbo_score:.2f}",
            "raw_pbo_score": pbo_score,
            "sra_p_value": f"{avg_p_value:.3f}",
            "raw_sra_p_value": avg_p_value,
            "avg_buy_dist": f"{avg_buy_dist:.1f}天",
            "raw_avg_buy_dist": avg_buy_dist,
            "avg_sell_dist": f"{avg_sell_dist:.1f}天",
            "raw_avg_sell_dist": avg_sell_dist,
            "avg_hold_days": f"{avg_hold_days:.1f}天",
            "raw_avg_hold_days": avg_hold_days,
            "strategy": strat
        }
        merged_attrs = {**log_data, **{k: v for k, v in trial.user_attrs.items() if k not in log_data}}
        log_to_results("試驗結果", trial_details, **merged_attrs)
        return score
    except Exception as e:
        logger.error(f"CPCV/OOS 計算失敗: {e}, 策略: {strat}")
        trial_results.append({
            "trial_number": str(trial.number).zfill(5),
            "parameters": params,
            "score": -np.inf,
            "equity_curve_json": equity_curve.to_json(),
            "strategy": strat,
            "data_source": data_source
        })
        trial.study.set_user_attr("trial_results", trial_results)
        return -np.inf
def plot_all_scatter(study, timestamp: str, data_source: str):
    results = study.user_attrs.get("trial_results", [])
    if not results:
        logger.warning("無試驗結果，無法生成散點圖")
        return
    
    # 轉換為數值，處理字串格式
    def convert_to_numeric(value, default=np.nan):
        try:
            if isinstance(value, str) and '%' in value:
                return float(value.strip('%')) / 100
            return pd.to_numeric(value, errors='coerce')
        except:
            return default

    total_returns = [convert_to_numeric(r.get("total_return", 0.0)) for r in results]
    mdds = [convert_to_numeric(r.get("max_drawdown", 0.0)) for r in results]
    sharpes = [convert_to_numeric(r.get("sharpe_ratio", 0.0)) for r in results]
    stress_mdds = [convert_to_numeric(r.get("stress_mdd", np.nan)) for r in results]
    excess_returns = [convert_to_numeric(r.get("excess_return_stress", np.nan)) for r in results]
    
    # 移除 NaN 值並記錄有效數據量
    valid_mdds = [x for x in mdds if not np.isnan(x)]
    valid_sharpes = [x for x in sharpes if not np.isnan(x)]
    valid_stress_mdds = [x for x in stress_mdds if not np.isnan(x)]
    valid_excess_returns = [x for x in excess_returns if not np.isnan(x)]
    
    logger.info(f"有效數據量: MDD={len(valid_mdds)}, Sharpe={len(valid_sharpes)}, Stress MDD={len(valid_stress_mdds)}, Excess Return={len(valid_excess_returns)}")
    
    # 檢查數據分佈
    if valid_stress_mdds:
        logger.info(f"Stress MDD 範圍: {min(valid_stress_mdds):.3f} ~ {max(valid_stress_mdds):.3f}")
    else:
        logger.warning("無有效 Stress MDD 數據，可能是壓力期間與數據範圍不匹配")
    if valid_excess_returns:
        logger.info(f"Excess Return Stress 範圍: {min(valid_excess_returns):.3f} ~ {max(valid_excess_returns):.3f}")
    else:
        logger.warning("無有效 Excess Return Stress 數據，可能是壓力期間無交易")
    
    safe_ds = sanitize(data_source)
    plot_files = [
        f"{safe_ds}_mdd_vs_total_return_{timestamp}.png",
        f"{safe_ds}_sharpe_vs_total_return_{timestamp}.png",
        f"{safe_ds}_stress_mdd_vs_total_return_{timestamp}.png",
        f"{safe_ds}_excess_vs_total_return_{timestamp}.png"
    ]
    
    if len(valid_mdds) > 3:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_mdds, total_returns[:len(valid_mdds)], alpha=0.4)
        plt.xlabel("MDD")
        plt.ylabel("Total Return")
        plt.title(f"數據源：{data_source}\nTotal Return vs MDD")
        plt.savefig(cfg.RESULT_DIR / plot_files[0])
        plt.close()
        logger.info(f"生成散點圖：{plot_files[0]}")
    
    if len(valid_sharpes) > 3:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_sharpes, total_returns[:len(valid_sharpes)], alpha=0.4)
        plt.xlabel("Sharpe Ratio")
        plt.ylabel("Total Return")
        plt.title(f"數據源：{data_source}\nTotal Return vs Sharpe Ratio")
        plt.savefig(cfg.RESULT_DIR / plot_files[1])
        plt.close()
        logger.info(f"生成散點圖：{plot_files[1]}")
    
    if len(valid_stress_mdds) > 3:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_stress_mdds, total_returns[:len(valid_stress_mdds)], alpha=0.4)
        plt.xlabel("Stress MDD")
        plt.ylabel("Total Return")
        plt.title(f"數據源：{data_source}\nTotal Return vs Stress MDD")
        plt.savefig(cfg.RESULT_DIR / plot_files[2])
        plt.close()
        logger.info(f"生成散點圖：{plot_files[2]}")
    else:
        logger.warning(f"Stress MDD 數據量不足 ({len(valid_stress_mdds)})，跳過散點圖生成")
    
    if len(valid_excess_returns) > 3:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_excess_returns, total_returns[:len(valid_excess_returns)], alpha=0.4)
        plt.xlabel("Excess Return in Stress")
        plt.ylabel("Total Return")
        plt.title(f"數據源：{data_source}\nTotal Return vs Excess Return in Stress")
        plt.savefig(cfg.RESULT_DIR / plot_files[3])
        plt.close()
        logger.info(f"生成散點圖：{plot_files[3]}")
    else:
        logger.warning(f"Excess Return Stress 數據量不足 ({len(valid_excess_returns)})，跳過散點圖生成")

def generate_preset_equity_curve(ticker: str, start_date: str, end_date: str, preset_params: Dict, cache_dir: str = str(cfg.SMAA_CACHE_DIR)) -> Optional[pd.Series]:
    """
    為指定的 param_presets 生成 equity curve。
    
    Args:
        ticker: 股票代號。
        start_date: 數據起始日期。
        end_date: 數據結束日期。
        preset_params: 來自 SSSv096.py 的 param_presets 單一參數組。
        cache_dir: SMAA 快取目錄。
    
    Returns:
        pd.Series: equity curve，若生成失敗則返回 None。
    """
    try:
        strategy_type = preset_params.get('strategy_type')
        smaa_source = preset_params.get('smaa_source', 'Self')
        logger.info(f"生成 {strategy_type} 策略的 equity curve，數據源: {smaa_source}")

        # 載入數據
        df_price, df_factor = load_data(ticker, start_date=start_date, end_date=end_date, smaa_source=smaa_source)
        if df_price.empty:
            logger.error(f"價格數據為空，無法生成 equity curve，策略: {strategy_type}")
            return None

        # 根據策略生成 df_ind 與買賣信號
        if strategy_type == 'ssma_turn':
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 
                         'vol_window', 'quantile_win', 'signal_cooldown_days']
            ssma_params = {k: v for k, v in preset_params.items() if k in calc_keys}
            backtest_params = ssma_params.copy()
            backtest_params['stop_loss'] = preset_params.get('stop_loss', 0.0)
            backtest_params['buy_mult'] = preset_params.get('buy_mult', 0.5)
            backtest_params['sell_mult'] = preset_params.get('sell_mult', 0.5)
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                df_price, df_factor, **ssma_params, smaa_source=smaa_source, cache_dir=cache_dir
            )
            if df_ind.empty:
                logger.warning(f"{strategy_type} 策略計算失敗，數據不足")
                return None
            result = backtest_unified(
                df_ind, strategy_type, backtest_params, buy_dates, sell_dates,
                discount=0.30, trade_cooldown_bars=7, bad_holding=False
            )
        else:
            if strategy_type == 'single':
                df_ind = compute_single(
                    df_price, df_factor, preset_params['linlen'], preset_params['factor'], 
                    preset_params['smaalen'], preset_params['devwin'], smaa_source=smaa_source, cache_dir=cache_dir
                )
            elif strategy_type == 'dual':
                df_ind = compute_dual(
                    df_price, df_factor, preset_params['linlen'], preset_params['factor'], 
                    preset_params['smaalen'], preset_params['short_win'], preset_params['long_win'], 
                    smaa_source=smaa_source, cache_dir=cache_dir
                )
            elif strategy_type == 'RMA':
                df_ind = compute_RMA(
                    df_price, df_factor, preset_params['linlen'], preset_params['factor'], 
                    preset_params['smaalen'], preset_params['rma_len'], preset_params['dev_len'], 
                    smaa_source=smaa_source, cache_dir=cache_dir
                )
            if df_ind.empty:
                logger.warning(f"{strategy_type} 策略計算失敗，數據不足")
                return None
            result = backtest_unified(
                df_ind, strategy_type, preset_params, discount=0.30, trade_cooldown_bars=7, bad_holding=False
            )

        equity_curve = result.get('equity_curve')
        if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
            logger.error(f"生成的 equity curve 無效，策略: {strategy_type}")
            return None
        
        return equity_curve.rename(f"preset_{preset_params.get('name', strategy_type)}")

    except Exception as e:
        logger.error(f"生成 equity curve 失敗，策略: {strategy_type}，錯誤: {e}")
        return None

def compute_equity_correlations_with_presets(
    trial_results: List[Dict],
    param_presets: Dict,
    top_n: int = 20,
    ind_keys: Optional[List[str]] = None,
    ticker: str = '00631L.TW',
    start_date: str = '2010-01-01',
    end_date: str = '2025-06-06',
    output_dir: Path = Path('results'),
    data_source: str = 'Self',
    TIMESTAMP: Optional[str] = None
) -> pd.DataFrame:
    # --- 以下為唯一定義版本 ---
    global top_trials
    TIMESTAMP = TIMESTAMP or datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if ind_keys is None:
        ind_keys = cfg.STRATEGY_PARAMS['single']['ind_keys'] + cfg.STRATEGY_PARAMS['single']['bt_keys']
    
    # 轉換 parameters 並進行多樣性篩選
    ready = []
    for tr in trial_results:
        if "parameters" not in tr or "score" not in tr:
            logger.warning(f"試驗 {tr.get('trial_number', 'unknown')} 缺少 parameters 或 score 鍵，跳過")
            continue
        try:
            params = ast.literal_eval(tr["parameters"]) if isinstance(tr["parameters"], str) else tr["parameters"]
            if not isinstance(params, dict):
                logger.warning(f"試驗 {tr.get('trial_number', 'unknown')} 的 parameters 格式無效，跳過")
                continue
            ready.append(dict(tr, parameters=params))
        except (ValueError, SyntaxError) as e:
            logger.warning(f"試驗 {tr.get('trial_number', 'unknown')} 的參數解析失敗：{e}，跳過")
            continue
    if not ready:
        logger.error("無有效試驗數據，無法計算相關性")
        raise ValueError("無有效試驗數據")
    
    top_diverse = pick_topN_by_diversity(ready, ind_keys, top_n, pct_threshold=pct_threshold_self)
    logger.info(f"從 {len(ready)} 筆試驗中篩選出 {len(top_diverse)} 筆多樣性試驗")
    if len(top_diverse) < top_n:
        logger.warning(f"篩選試驗數量 ({len(top_diverse)}) 小於預期 top_n ({top_n})，建議降低 pct_threshold 或增加 n_trials")

    # 收集試驗的 equity curve
    df_list = []
    names = []
    for trial in top_diverse:
        eq_json = trial.get('equity_curve_json')  # 只認這個鍵
        trial_num = trial.get('trial_number', 'unknown')
        if eq_json is None:
            logger.warning(f"試驗 {trial_num} 的 equity curve 缺失，跳過")
            continue
        try:
            eq = pd.read_json(eq_json, typ='series')
            if not isinstance(eq, pd.Series) or eq.empty:
                logger.warning(f"試驗 {trial_num} 的 equity curve 無效，跳過")
                continue
            # 確保索引為時間戳
            eq.index = pd.to_datetime(eq.index, unit='ms')
            ser = eq.pct_change().rename(f"trial_{trial_num}")
            df_list.append(ser)
            names.append(f"trial_{trial_num}")
        except ValueError as e:
            logger.warning(f"試驗 {trial_num} 的 equity curve 反序列化失敗: {e}，跳過")
            continue

    # 生成 param_presets 的 equity curve
    for preset_name, preset_params in param_presets.items():
        preset_params['name'] = preset_name
        eq = generate_preset_equity_curve(ticker, start_date, end_date, preset_params)
        if eq is not None:
            ser = eq.pct_change().rename(f"preset_{preset_name}")
            df_list.append(ser)
            names.append(f"preset_{preset_name}")

    if not df_list:
        logger.warning("無有效試驗 equity curve，僅輸出 presets 或返回空相關性矩陣")
        return pd.DataFrame()

    eq_df = pd.concat(df_list, axis=1).fillna(0)
    logger.info(f"合併後的 equity curve DataFrame 形狀: {eq_df.shape}")

    corr_matrix = eq_df.corr()
    
    corr_file = output_dir / f"equity_corr_top_{top_n}_{sanitize(ticker)}_{sanitize(data_source)}_{TIMESTAMP}.csv"
    corr_matrix.to_csv(corr_file, encoding='utf-8-sig')
    logger.info(f"相關性矩陣已儲存至 {corr_file}")

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title(f"Top {top_n} Trials and Param Presets Equity Curve Daily Return Correlation")
    plt.tight_layout()
    heatmap_file = output_dir / f"equity_corr_top_{top_n}_{sanitize(ticker)}_{sanitize(data_source)}_{TIMESTAMP}.png"
    plt.savefig(heatmap_file)
    plt.close()
    logger.info(f"熱圖已儲存至 {heatmap_file}")

    return corr_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna 最佳化 00631L 策略')
    parser.add_argument('--strategy', type=str, choices=['single', 'dual', 'RMA', 'ssma_turn', 'all'], default='all')
    parser.add_argument('--n_trials', type=int, default=5000)
    parser.add_argument('--data_source', type=str, choices=cfg.SOURCES, default=None)
    parser.add_argument('--data_source_mode', type=str, choices=['random', 'fixed', 'sequential'], default='random')
    args = parser.parse_args()

    # 清空快取目錄
    cache_dir = Path("C:/Stock_reserach/SSS095a1/cache/price")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info(f"已清空快取目錄: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.data_source_mode == 'sequential':
        for data_source in cfg.SOURCES:
            logger.info(f"開始針對數據源 {data_source} 進行最佳化, 策略: {args.strategy}")
            safe_ds = sanitize(data_source)
            optuna_sqlite = cfg.RESULT_DIR / f"optuna_{args.strategy}_{safe_ds}_{RUN_TS}.sqlite3"

            results_log.clear()
            events_log.clear()
            top_trials = []

            study = optuna.create_study(
                study_name=f"00631L_optuna_{args.strategy}_{safe_ds}",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
                storage=f"sqlite:///{optuna_sqlite}"
            )
            study.set_user_attr("data_source", data_source)
            study.optimize(objective, n_trials=args.n_trials, n_jobs=4, show_progress_bar=True)

            plot_all_scatter(study, RUN_TS, data_source)
            trial_results_all = study.user_attrs.get("trial_results", [])
            if len(trial_results_all) > 1000:
                trial_results_all = trial_results_all[-500:]
                study.set_user_attr("trial_results", trial_results_all)

            try:
                ind_keys = cfg.STRATEGY_PARAMS[args.strategy]['ind_keys'] + cfg.STRATEGY_PARAMS[args.strategy]['bt_keys']
                corr_matrix = compute_equity_correlations_with_presets(
                    trial_results=trial_results_all,
                    param_presets=SSS.param_presets,
                    top_n=20,
                    ind_keys=ind_keys,
                    ticker=cfg.TICKER,
                    start_date=cfg.START_DATE,
                    end_date="2025-06-06",
                    output_dir=cfg.RESULT_DIR,
                    data_source=data_source,
                    TIMESTAMP=RUN_TS
                )
            except (ValueError, AttributeError) as e:
                logger.error(f"相關性計算失敗，數據源: {data_source}，錯誤: {e}")
                continue

            # 現有代碼
            # 標準化並篩選結果
            df_results = pd.json_normalize(results_log, sep='_').loc[lambda d: d["Event Type"] == "試驗結果"].sort_values("score", ascending=False)

            # 為每個試驗結果添加對應的 equity_curve_json
            equity_mapping = {t["trial_number"]: t.get("equity_curve_json", "") for t in results_log if t["Event Type"] == "試驗結果"}
            df_results["equity_curve_json"] = df_results["trial_number"].map(equity_mapping).fillna("")

            result_csv_file = cfg.RESULT_DIR / f"optuna_results_{args.strategy}_{safe_ds}_{RUN_TS}.csv"
            df_results.to_csv(result_csv_file, index=False, encoding="utf-8-sig")
            logger.info(f"試驗結果已保存至 {result_csv_file}")
            best = study.best_trial
            logger.info(f"最佳試驗(數據源: {data_source}): ")
            logger.info(f"策略: {best.user_attrs['strategy']}")
            logger.info(f"數據源: {best.user_attrs['data_source']}")
            logger.info(f"參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}")
            logger.info(f"穩健分數: {best.value:.3f}")
            logger.info(f"其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}")
            best_trial_details = (
                f"策略: {best.user_attrs['strategy']}, 數據源: {best.user_attrs['data_source']}, "
                f"參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}, "
                f"穩健分數: {best.value:.3f}, 其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}"
            )
            log_to_results("最佳試驗資訊", best_trial_details)

            results = {
                "best_robust_score": best.value,
                "best_strategy": best.user_attrs["strategy"],
                "best_data_source": best.user_attrs["data_source"],
                "best_params": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.params.items() if k not in ["strategy", "data_source"]},
                "best_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.user_attrs.items()},
                "best_avg_hold_days": round(best.user_attrs.get("avg_hold_days", 0.0), 1)
            }
            result_file = cfg.RESULT_DIR / f"optuna_best_params_{args.strategy}_{safe_ds}_{RUN_TS}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            pd.Series(results).to_json(result_file, indent=2)
            result_csv_file = cfg.RESULT_DIR / f"optuna_results_{args.strategy}_{safe_ds}_{RUN_TS}.csv"
            df_results.to_csv(result_csv_file, index=False, encoding="utf-8-sig")
            logger.info(f"最佳參數已保存至 {result_file}")
            logger.info(f"試驗結果已保存至 {result_csv_file}")
            event_csv_file = cfg.RESULT_DIR / f"optuna_events_{args.strategy}_{safe_ds}_{RUN_TS}.csv"
            df_events = pd.DataFrame(events_log)
            df_events.to_csv(event_csv_file, index=False, encoding='utf-8-sig', na_rep='0.0')
            logger.info(f"事件紀錄已保存至 {event_csv_file}")

            logger.info("前 5 筆試驗記錄:")
            for record in results_log[:5]:
                logger.info(f"[{record['Timestamp']}] - {record['Event Type']} - {record['Details']}")
            results_list = load_results_to_list(result_csv_file)
            logger.info(f"從 {result_csv_file} 載入 {len(results_list)} 筆記錄")

            results_log.clear()
            events_log.clear()

    else:
        # random 或 fixed 模式
        top_trials = []
        optuna_sqlite = cfg.RESULT_DIR / f"optuna_{args.strategy}_{TIMESTAMP}.sqlite3"
        optuna_sqlite.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=f"00631L_optuna_{args.strategy}_{args.data_source_mode}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
            storage=f"sqlite:///{optuna_sqlite}"
        )
        study.optimize(objective, n_trials=args.n_trials, n_jobs=4, show_progress_bar=True)
        plot_all_scatter(study, TIMESTAMP, args.data_source if args.data_source else "unknown")

        # 獲取試驗結果
        trial_results_all = study.user_attrs.get("trial_results", [])
        if len(trial_results_all) > 1000:
            trial_results_all = trial_results_all[-500:]  # 限制記憶體
            study.set_user_attr("trial_results", trial_results_all)

        # 計算 equity curve 相關性
        try:
            corr_matrix = compute_equity_correlations_with_presets(
                trial_results=trial_results_all,
                param_presets=SSS.param_presets,
                top_n=20,
                ticker=TICKER,
                start_date=cfg.START_DATE,
                end_date="2025-06-06",
                output_dir=cfg.RESULT_DIR
            )
        except (ValueError, AttributeError) as e:
            logger.error(f"相關性計算失敗，錯誤: {e}")

        # 處理試驗結果
        df_results = pd.json_normalize(results_log, sep='_').loc[lambda d: d["Event Type"].str.contains("試驗結果")].sort_values("score", ascending=False)
        df_results["equity_curve_json"] = [t.get("equity_curve_json", "") for t in results_log if t["Event Type"].str.contains("試驗結果")]

        # 記錄最佳試驗
        best = study.best_trial
        logger.info("最佳試驗:")
        logger.info(f"策略: {best.user_attrs['strategy']}")
        logger.info(f"數據源: {best.user_attrs['data_source']}")
        logger.info(f"參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}")
        logger.info(f"穩健分數: {best.value:.3f}")
        logger.info(f"其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}")
        best_trial_details = (
            f"策略: {best.user_attrs['strategy']}, 數據源: {best.user_attrs['data_source']}, "
            f"參數: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.params.items() if k not in ['strategy', 'data_source'])} }}, "
            f"穩健分數: {best.value:.3f}, 其他指標: {{ {', '.join(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' for k, v in best.user_attrs.items())} }}"
        )
        log_to_results("最佳試驗資訊", best_trial_details)

        # 記錄前 10 名與前 5 名試驗
        strategies = [args.strategy] if args.strategy != 'all' else list(PARAM_SPACE.keys())
        data_sources = DATA_SOURCES
        trial_results = [entry for entry in results_log if entry["Event Type"] in ["試驗結果", "試驗結果(備用邏輯)"]]
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

        # 儲存結果
        results = {
            "best_robust_score": best.value,
            "best_strategy": best.user_attrs["strategy"],
            "best_data_source": best.user_attrs["data_source"],
            "best_params": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.params.items() if k not in ["strategy", "data_source"]},
            "best_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.user_attrs.items()},
            "best_avg_hold_days": round(best.user_attrs.get("avg_hold_days", 0.0), 1)
        }
        result_file = cfg.RESULT_DIR / f"optuna_best_params_{args.strategy}_{TIMESTAMP}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(results).to_json(result_file, indent=2)
        result_csv_file = cfg.RESULT_DIR / f"optuna_results_{args.strategy}_{TIMESTAMP}.csv"
        df_results.to_csv(result_csv_file, index=False, encoding="utf-8-sig")
        logger.info(f"最佳參數已保存至 {result_file}")
        logger.info(f"試驗結果已保存至 {result_csv_file}")
        event_csv_file = cfg.RESULT_DIR / f"optuna_events_{args.strategy}_{TIMESTAMP}.csv"
        df_events = pd.DataFrame(events_log)
        df_events.to_csv(event_csv_file, index=False, encoding='utf-8-sig', na_rep='0.0')
        logger.info(f"事件紀錄已保存至 {event_csv_file}")

        # 記錄前 5 筆試驗
        logger.info("前 5 筆試驗記錄:")
        for record in results_log[:5]:
            logger.info(f"[{record['Timestamp']}] - {record['Event Type']} - {record['Details']}")
        results_list = load_results_to_list(result_csv_file)
        logger.info(f"從 {result_csv_file} 載入 {len(results_list)} 筆記錄")