'''
Optuna-based hyper-parameter optimization for 00631L strategies (Version 4a)
--------------------------------------------------
* 最佳化目標:優先追求完整回測報酬(>300%)、交易次數(15-60次)、避免過擬合,其次考慮走查期間最差報酬、夏普比率、最大回撤,最後考慮壓力測試平均報酬。
* 本腳本可直接放在 analysis/ 目錄後以 `python optuna_4a.py` 執行。
* 搜尋空間與權重在 `PARAM_SPACE` 與 `SCORE_WEIGHTS` 中設定,方便日後微調。
python optuna_4a.py --strategy RMA --n_trials 10000
python optuna_4a.py --strategy single --n_trials 10000
python optuna_4a.py --strategy dual --n_trials 10000
python optuna_4a.py --strategy ssma_turn --n_trials 10000
1. 數據源分流
 DATA_SOURCES_WEIGHTS,設置抽樣比例:^TWII / 2414.:50%,Self:30%,^TWII / 2412.TW:20%
2. 策略抽樣分流
STRATEGY_WEIGHTS,設置抽樣比例:RMA:30%,single:30%,dual:20%,ssma_turn:20%
3.支援命令列參數 --strategy,允許單獨優化某一策略,若未指定策略,使用加權抽樣(第2項)
python optuna_4a.py --strategy RMA


4.權重調整:
提高 wf_min_return, sharpe_ratio, max_drawdown 的權重(各 0.3),降低 stress_avg_return 權重(0.1),強調穩定性和風險控制。
5.Min-Max Scaling:
total_return 上限設為 14×(約 95% 分位),使 10-14× 的試驗分數更具區分度。
profit_factor 上限設為 8,突出高盈虧因子的試驗。
5.過擬合懲罰:
啟動門檻降至 15 次試驗。
懲罰係數增至 0.2 * (stab - 0.05),對 stability_score > 1 的試驗施加更強懲罰。
依賴:
  pip install optuna tqdm scikit-learn
'''

import logging
from pathlib import Path
import optuna
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import argparse

# 清除全局日誌處理器,防止其他模組干擾
logging.getLogger().handlers = []

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- 專案模組 ---
from analysis import config as cfg
from analysis import data_loader
import SSSv095a1 as SSS  # 核心計算與回測

# ---------- 命令列參數解析 ----------
parser = argparse.ArgumentParser(description='Optuna optimization for 00631L strategies')
parser.add_argument('--strategy', type=str, choices=['single', 'dual', 'RMA', 'ssma_turn', 'all'], default='all',
                    help='Specify a single strategy to optimize (default: all)')
parser.add_argument('--n_trials', type=int, default=10000,
                    help='Number of trials to run (default: 10000)')
args = parser.parse_args()

# ---------- 日誌設定 ----------
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = cfg.LOG_DIR / f'optuna_4a_{TIMESTAMP}.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 使用獨立日誌文件
        logging.StreamHandler()
    ]
)

# 用於儲存重要訊息的列表
results_log = []

# 記錄重要訊息的函數
def log_to_results(event_type, details, **kwargs):
    """記錄重要訊息到 results_log 列表中。"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]  # 精確到毫秒
    record = {
        "Timestamp": timestamp,
        "Event Type": event_type,
        "Details": details,
        **kwargs
    }
    results_log.append(record)

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
DATA_SOURCES = cfg.SOURCES  # ['Self', 'Factor (^TWII / 2412.TW)', 'Factor (^TWII / 2414.TW)']
DATA_SOURCES_WEIGHTS = {
    'Self': 0.3,
    'Factor (^TWII / 2412.TW)': 0.2,
    'Factor (^TWII / 2414.TW)': 0.5
}

# 策略選項與權重
STRATEGY_WEIGHTS = {
    'single': 0.3,
    'dual': 0.2,
    'RMA': 0.3,
    'ssma_turn': 0.2
}

# 交易次數範圍
MIN_NUM_TRADES = 15
MAX_NUM_TRADES = 60

# ---------- 搜尋空間 ---------
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
        linlen=(5, 240, 1),
        smaalen=(7, 240, 5),
        factor=(10.0, 80.0, 10.0),
        prom_factor=(10, 70, 1),
        min_dist=(5, 20, 1),
        buy_shift=(0, 7, 1),
        exit_shift=(0, 7, 1),
        vol_window=(5, 90, 5),
        quantile_win=(5, 180, 10),
        signal_cooldown_days=(1, 7, 1),
        buy_mult=(0.1, 2, 0.05),
        sell_mult=(0.2, 4, 0.1),
        stop_loss=(0.00, 0.15, 0.01),
    ),
}

# 策略層級顯式關聯 ind_keys/bt_keys
IND_BT_KEYS = {
    strat: cfg.STRATEGY_PARAMS[strat]["ind_keys"] + cfg.STRATEGY_PARAMS[strat]["bt_keys"]
    for strat in PARAM_SPACE
}

# 各部份分數權重
SCORE_WEIGHTS = dict(
    total_return=2,        # 高優先級:完整回測報酬
    profit_factor=1.0,     # 高優先級:盈虧因子
    wf_min_return=0.3,     # 提高走查報酬權重
    sharpe_ratio=0.3,      # 提高夏普比率權重
    max_drawdown=0.3,      # 提高最大回撤權重
    stress_avg_return=0.1, # 降低壓力測試報酬權重
)

# ---------- 工具函式 ----------

def _sample_params(trial: optuna.Trial, strat: str) -> dict:
    """依據 PARAM_SPACE 產生該策略的參數組。"""
    space = PARAM_SPACE[strat]
    params = {}
    for k, v in space.items():
        if isinstance(v[0], int):
            low, high, step = v
            params[k] = trial.suggest_int(k, low, high, step=step)
        else:
            low, high, step = v
            params[k] = trial.suggest_float(k, low, high, step=step)
    return params

def minmax(x, lo, hi, clip=True):
    """Min-Max Scaling 工具函式。"""
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1) if clip else y

def compute_knn_stability(df_results: list, params: list, k: int = 5, metric: str = 'total_return') -> float:
    """計算 KNN 穩定性,評估過擬合風險。"""
    if len(df_results) < k + 1:
        logging.warning(f"試驗數量 {len(df_results)} 不足以計算 KNN 穩定性 (需要至少 {k+1})")
        log_to_results("Warning", f"試驗數量 {len(df_results)} 不足以計算 KNN 穩定性 (需要至少 {k+1})")
        return 0.0
    param_cols = [p for p in params if p in df_results[0]]
    if not param_cols:
        logging.warning(f"無有效參數用於 KNN 穩定性計算,參數: {params}")
        log_to_results("Warning", f"無有效參數用於 KNN 穩定性計算,參數: {params}")
        return 0.0
    X = np.array([[r[p] for p in param_cols] for r in df_results])
    y = np.array([r[metric] for r in df_results])
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    stability_scores = []
    for i, idx in enumerate(indices):
        roi = y[i]
        roi_neighbors = np.mean(y[idx[1:]])  # 排除自身
        stability_scores.append(abs(roi - roi_neighbors))
    return float(np.mean(stability_scores))

def _backtest_once(strat: str, params: dict, trial_results: list, data_source: str) -> tuple[float, int, float, float, float]:
    """單段完整回測,回傳 (total_return, num_trades, sharpe_ratio, max_drawdown, profit_factor)。"""
    try:
        df_price, df_factor = data_loader.load_data(TICKER, smaa_source=data_source)
        if df_price.empty:
            logging.error(f"價格數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("Error", f"價格數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf, 0, 0.0, 0.0, 0.0
    except Exception as e:
        logging.error(f"數據加載失敗,策略: {strat}, 數據源: {data_source}, 錯誤: {e}")
        log_to_results("Error", f"數據加載失敗,策略: {strat}, 數據源: {data_source}, 錯誤: {e}")
        return -np.inf, 0, 0.0, 0.0, 0.0

    compute_f = STRAT_FUNC_MAP[strat]
    ind_keys = cfg.STRATEGY_PARAMS[strat]["ind_keys"]
    ind_p = {k: params[k] for k in ind_keys}
    try:
        if strat == "ssma_turn":
            df_ind, buys, sells = compute_f(df_price, df_factor, **ind_p)
            bt = SSS.backtest_unified(
                df_ind=df_ind,
                strategy_type=strat,
                params=params,
                buy_dates=buys,
                sell_dates=sells,
                discount=COST_PER_SHARE / 1.0,
                trade_cooldown_bars=COOLDOWN_BARS,
            )
        else:
            df_ind = compute_f(df_price, df_factor, **ind_p)
            bt = SSS.backtest_unified(
                df_ind=df_ind,
                strategy_type=strat,
                params=params,
                discount=COST_PER_SHARE / 1.0,
                trade_cooldown_bars=COOLDOWN_BARS,
            )
        metrics = bt["metrics"]
        profit_factor = metrics.get("profit_factor", 1.0)
        if "profit_factor" not in metrics:
            trades = bt["trades"]
            if trades:
                gains = [t[1] for t in trades if t[1] > 0]
                losses = [abs(t[1]) for t in trades if t[1] < 0]
                profit_factor = sum(gains) / sum(losses) if losses and sum(losses) > 0 else np.inf
                profit_factor = min(profit_factor, 10.0)  # 限制極端值
        trial_results.append({
            "total_return": metrics["total_return"],
            "num_trades": metrics.get("num_trades", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "profit_factor": profit_factor,
            "data_source": data_source,
            **params
        })
        logging.info(f"Backtest completed for {strat} with data_source {data_source}: total_return={metrics['total_return']*100:.2f}%, num_trades={metrics.get('num_trades', 0)}")
        log_to_results("Backtest Completed", f"Strategy: {strat}, data_source: {data_source}, total_return={metrics['total_return']*100:.2f}%, num_trades={metrics.get('num_trades', 0)}")
        return (
            metrics["total_return"],
            metrics.get("num_trades", 0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("max_drawdown", 0.0),
            profit_factor
        )
    except Exception as e:
        logging.error(f"回測失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        log_to_results("Error", f"回測失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        return -np.inf, 0, 0.0, 0.0, 0.0

def _wf_min_return(strat: str, params: dict, data_source: str) -> float:
    """計算走查期間最差報酬。"""
    try:
        df_price, df_factor = data_loader.load_data(TICKER, smaa_source=data_source)
        if df_price.empty:
            logging.warning(f"走查測試數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("Warning", f"走查測試數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf
        valid_periods = [
            (start, end) for start, end in WF_PERIODS
            if start in df_price.index and end in df_price.index
        ]
        if not valid_periods:
            logging.warning(f"無有效的走查期間,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("Warning", f"無有效的走查期間,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf

        results = SSS.compute_backtest_for_periods(
            ticker=TICKER,
            periods=valid_periods,
            strategy_type=strat,
            params=params,
            trade_cooldown_bars=COOLDOWN_BARS,
            discount=COST_PER_SHARE / 1.0,
        )
        if not results:
            logging.warning(f"走查測試無結果,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("Warning", f"走查測試無結果,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf

        valid_returns = []
        for r in results:
            try:
                total_return = r["metrics"]["total_return"]
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logging.warning(f"走查期間缺少 total_return,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
                log_to_results("Warning", f"走查期間缺少 total_return,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
                continue
        return min(valid_returns) if valid_returns else -np.inf
    except Exception as e:
        logging.error(f"走查測試失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        log_to_results("Error", f"走查測試失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        return -np.inf

def _stress_avg_return(strat: str, params: dict, data_source: str) -> float:
    """計算壓力區平均報酬。"""
    if not STRESS_PERIODS:
        return 0.0
    try:
        df_price, df_factor = data_loader.load_data(TICKER, smaa_source=data_source)
        if df_price.empty:
            logging.warning(f"壓力測試數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("Warning", f"壓力測試數據為空,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf
        valid_periods = [
            (start, end) for start, end in STRESS_PERIODS
            if start in df_price.index and end in df_price.index
        ]
        if not valid_periods:
            logging.warning(f"無有效的壓力測試期間,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("Warning", f"無有效的壓力測試期間,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf

        results = SSS.compute_backtest_for_periods(
            ticker=TICKER,
            periods=valid_periods,
            strategy_type=strat,
            params=params,
            trade_cooldown_bars=COOLDOWN_BARS,
            discount=COST_PER_SHARE / 1.0,
        )
        if not results:
            logging.warning(f"壓力測試無結果,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            log_to_results("Warning", f"壓力測試無結果,策略: {strat}, 數據源: {data_source}, 參數: {params}")
            return -np.inf

        valid_returns = []
        for r in results:
            try:
                total_return = r["metrics"]["total_return"]
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logging.warning(f"壓力測試期間缺少 total_return,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
                log_to_results("Warning", f"壓力測試期間缺少 total_return,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
                continue
        return float(np.mean(valid_returns)) if valid_returns else -np.inf
    except Exception as e:
        logging.error(f"壓力測試失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        log_to_results("Error", f"壓力測試失敗,策略: {strat}, 數據源: {data_source}, 參數: {params}, 錯誤: {e}")
        return -np.inf

# ---------- Optuna 目標 ----------
def objective(trial: optuna.Trial):
    # 選擇策略
    if args.strategy == 'all':
        strat = np.random.choice(
            list(STRATEGY_WEIGHTS.keys()),
            p=list(STRATEGY_WEIGHTS.values())
        )
    else:
        strat = args.strategy
    trial.set_user_attr("strategy", strat)

    # 選擇數據源
    data_source = np.random.choice(
        list(DATA_SOURCES_WEIGHTS.keys()),
        p=list(DATA_SOURCES_WEIGHTS.values())
    )
    trial.set_user_attr("data_source", data_source)

    params = _sample_params(trial, strat)

    # 儲存試驗結果
    trial_results = trial.study.user_attrs.get("trial_results", [])
    
    # 完整回測
    total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor = _backtest_once(strat, params, trial_results, data_source)
    
    # 第一層:硬性門檻
    if total_ret == -np.inf:
        return -np.inf
    if not (MIN_NUM_TRADES <= n_trades <= MAX_NUM_TRADES):
        logging.info(f"試驗被剔除:交易次數 {n_trades} 不在 {MIN_NUM_TRADES}-{MAX_NUM_TRADES} 範圍,策略: {strat}, 數據源: {data_source}")
        log_to_results("Trial Excluded", f"交易次數 {n_trades} 不在 {MIN_NUM_TRADES}-{MAX_NUM_TRADES} 範圍,策略: {strat}, 數據源: {data_source}")
        return -np.inf
    if total_ret <= 3.0:
        logging.info(f"試驗被剔除:完整回測報酬 {total_ret:.2%} <= 300%,策略: {strat}, 數據源: {data_source}")
        log_to_results("Trial Excluded", f"完整回測報酬 {total_ret:.2%} <= 300%,策略: {strat}, 數據源: {data_source}")
        return -np.inf

    # 更新 trial_results
    trial.study.set_user_attr("trial_results", trial_results)

    # 走查最差報酬
    min_wf_ret = _wf_min_return(strat, params, data_source)
    if min_wf_ret == -np.inf:
        return -np.inf

    # 壓力測試平均報酬
    avg_stress_ret = _stress_avg_return(strat, params, data_source)

    # 第二層:Min-Max Scaling
    tr_s = minmax(total_ret, 5, 14)  # ROI Clip @ 14x (95% 分位)
    pf_s = minmax(profit_factor, 0.5, 8)  # 提高盈虧因子上限
    wf_s = minmax(min_wf_ret, 0, 2)
    sh_s = minmax(sharpe_ratio, 0, 0.8)
    mdd_s = 1 - minmax(abs(max_drawdown), 0, 0.3)
    st_s = minmax(avg_stress_ret, -1, 1)

    # 加權分數
    score = (
        SCORE_WEIGHTS["total_return"] * tr_s +
        SCORE_WEIGHTS["profit_factor"] * pf_s +
        SCORE_WEIGHTS["sharpe_ratio"] * sh_s +
        SCORE_WEIGHTS["max_drawdown"] * mdd_s +
        SCORE_WEIGHTS["wf_min_return"] * wf_s +
        SCORE_WEIGHTS["stress_avg_return"] * st_s
    )

    # 交易次數懲罰
    trade_penalty = 0.05 * max(0, 15 - n_trades)  # 懲罰交易次數 < 15
    score -= trade_penalty

    # 第三層:過擬合懲罰
    stab = 0.0
    if len(trial_results) >= 15:  # 降低啟動門檻
        stab = compute_knn_stability(trial_results, params=['linlen', 'smaalen', 'buy_mult'], k=5)
        if stab > 0.05:
            penalty = 0.2 * (stab - 0.05)  # 提高懲罰力度
            score *= max(0.0, 1 - penalty)
            logging.info(f"過擬合懲罰:穩定性得分={stab:.4f}, 懲罰值={penalty:.4f}")
            log_to_results("Overfit Penalty", f"穩定性得分={stab:.4f}, 懲罰值={penalty:.4f}")

    # 儲存指標
    trial.set_user_attr("total_return", total_ret)
    trial.set_user_attr("total_return_scaled", tr_s)
    trial.set_user_attr("num_trades", n_trades)
    trial.set_user_attr("sharpe_ratio", sharpe_ratio)
    trial.set_user_attr("sharpe_ratio_scaled", sh_s)
    trial.set_user_attr("max_drawdown", max_drawdown)
    trial.set_user_attr("max_drawdown_scaled", mdd_s)
    trial.set_user_attr("profit_factor", profit_factor)
    trial.set_user_attr("profit_factor_scaled", pf_s)
    trial.set_user_attr("min_wf_return", min_wf_ret)
    trial.set_user_attr("min_wf_return_scaled", wf_s)
    trial.set_user_attr("avg_stress_return", avg_stress_ret)
    trial.set_user_attr("avg_stress_return_scaled", st_s)
    trial.set_user_attr("stability_score", stab)
    trial.set_user_attr("data_source", data_source)

    # 記錄試驗結果,添加結構化欄位
    trial_details = (
        f"Trial {trial.number}, "
        f"strategy: {strat}, "
        f"data_source: {data_source}, "
        f"value: {score}, "
        f"parameters: {params}, "
        f"total_return={total_ret*100:.2f}%, "
        f"num_trades={n_trades}, "
        f"sharpe_ratio={sharpe_ratio:.4f}, "
        f"max_drawdown={max_drawdown:.4f}, "
        f"profit_factor={profit_factor:.4f}, "
        f"min_wf_return={min_wf_ret:.4f}, "
        f"avg_stress_return={avg_stress_ret:.4f}, "
        f"stability_score={stab:.4f}"
    )
    log_to_results(
        "Trial Result",
        trial_details,
        trial_number=trial.number,
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
        stability_score=stab
    )

    return score

# ---------- 執行入口 ----------
if __name__ == "__main__":
    optuna_sqlite = Path(cfg.RESULT_DIR) / f"optuna_4a_{TIMESTAMP}.sqlite3"
    study = optuna.create_study(
        study_name=f"00631L_optuna_v4a_{args.strategy}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        storage=f"sqlite:///{optuna_sqlite}"
    )

    n_trials = args.n_trials
    logging.info(f"開始最佳化,策略: {args.strategy},共 {n_trials} trials,交易次數範圍:{MIN_NUM_TRADES}-{MAX_NUM_TRADES} …")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # 最佳試驗資訊
    logging.info("最佳 trial:")
    best = study.best_trial
    logging.info(f"Strategy        : {best.user_attrs['strategy']}")
    logging.info(f"Data Source     : {best.user_attrs['data_source']}")
    logging.info(f"Params          : { {k:v for k,v in best.params.items() if k not in ['strategy', 'data_source']} }")
    logging.info(f"Robust Score    : {best.value:.4f}")
    logging.info(f"Other metrics   : {best.user_attrs}")

    # 記錄最佳試驗資訊
    best_trial_details = (
        f"Strategy: {best.user_attrs['strategy']}, "
        f"Data Source: {best.user_attrs['data_source']}, "
        f"Params: { {k:v for k,v in best.params.items() if k not in ['strategy', 'data_source']} }, "
        f"Robust Score: {best.value:.4f}, "
        f"Other metrics: {best.user_attrs}"
    )
    log_to_results("Best Trial Info", best_trial_details)

    # 提取每個策略的前 10 名試驗,並按「策略 + 數據源」分組挑選前 10 名
    strategies = [args.strategy] if args.strategy != 'all' else list(PARAM_SPACE.keys())
    data_sources = DATA_SOURCES
    trial_results = [entry for entry in results_log if entry["Event Type"] == "Trial Result"]

    # 1. 各策略搭配各數據源挑選前 10 名
    for strategy in strategies:
        for data_source in data_sources:
            strategy_source_trials = [
                entry for entry in trial_results 
                if entry.get("strategy") == strategy and entry.get("data_source") == data_source
            ]
            if not strategy_source_trials:
                logging.info(f"策略 {strategy} 與數據源 {data_source} 無有效試驗結果")
                continue

            trial_scores = []
            for entry in strategy_source_trials:
                trial_num = entry.get("trial_number")
                score = entry.get("score")
                details = entry["Details"]
                try:
                    trial_scores.append((trial_num, score, details))
                except ValueError:
                    continue

            trial_scores.sort(key=lambda x: x[1], reverse=True)
            top_10_trials = trial_scores[:10]

            if top_10_trials:
                logging.info(f"策略 {strategy} 與數據源 {data_source} 前 10 名試驗:")
                for trial_num, score, details in top_10_trials:
                    logging.info(details)
                    log_to_results(f"Top 10 {strategy} with {data_source} Trial", details)

    # 2. 各策略搭配各數據源分組挑選前 5 名
    for strategy in strategies:
        for data_source in data_sources:
            strategy_source_trials = [
                entry for entry in trial_results 
                if entry.get("strategy") == strategy and entry.get("data_source") == data_source
            ]
            if not strategy_source_trials:
                logging.info(f"策略 {strategy} 與數據源 {data_source} 無有效試驗結果(分組前 5 名)")
                continue

            trial_scores = []
            for entry in strategy_source_trials:
                trial_num = entry.get("trial_number")
                score = entry.get("score")
                details = entry["Details"]
                try:
                    trial_scores.append((trial_num, score, details))
                except ValueError:
                    continue

            trial_scores.sort(key=lambda x: x[1], reverse=True)
            top_5_trials = trial_scores[:5]

            if top_5_trials:
                logging.info(f"策略 {strategy} 與數據源 {data_source} 分組前 5 名試驗:")
                for trial_num, score, details in top_5_trials:
                    logging.info(details)
                    log_to_results(f"Top 5 {strategy} with {data_source} Grouped Trial", details)

    # 保存最佳參數到 JSON
    results = {
        "best_robust_score": best.value,
        "best_strategy": best.user_attrs["strategy"],
        "best_data_source": best.user_attrs["data_source"],
        "best_params": {k: v for k, v in best.params.items() if k not in ["strategy", "data_source"]},
        "best_metrics": best.user_attrs
    }
    result_file = cfg.RESULT_DIR / f"optuna_4a_best_params_{TICKER.replace('^','')}_{TIMESTAMP}.json"
    pd.Series(results).to_json(result_file, indent=2)
    logging.info(f"最佳參數已保存至 {result_file}")

    # 將 results_log 保存為 CSV
    df_results = pd.DataFrame(results_log)
    result_csv_file = cfg.RESULT_DIR / f"optuna_4a_results_{TIMESTAMP}.csv"
    df_results.to_csv(result_csv_file, index=False)
    logging.info(f"重要訊息已保存至 {result_csv_file}")