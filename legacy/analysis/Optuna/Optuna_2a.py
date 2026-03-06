'''
 Optuna-based hyper-parameter optimization for 00631L strategies
 --------------------------------------------------------------
 * 最佳化目標：優先追求完整回測報酬(>500%)、交易次數(15-60次)、避免過擬合，其次考慮走查期間最差報酬、夏普值、最大回撤，最後考慮壓力測試平均報酬。
 * 本腳本可直接放在 analysis/ 目錄後以 `python optuna_search_v1.py` 執行。
 * 依 config.py 的 N_JOBS 自動使用 CPU；可自行透過 CLI 參數或環境變數調整 n_trials、direction 等。
 * 搜尋空間與權重在 `PARAM_SPACE` 與 `SCORE_WEIGHTS` 中設定，方便日後微調。

 依賴：
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- 專案模組 ---
from analysis import config as cfg
from analysis import data_loader
import SSSv095a1 as SSS  # 核心計算與回測

# ---------- 日誌設定 ----------
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(cfg.LOG_DIR / f'optuna_1a_{TIMESTAMP}.log'),
        logging.StreamHandler()
    ]
)

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

# ---------- 搜尋空間 ----------
PARAM_SPACE = {
    "single": dict(
        linlen=(20, 120, 5),
        smaalen=(10, 150, 5),
        devwin=(10, 100, 5),
        factor=(20.0, 60.0, 1.0),
        buy_mult=(0.1, 1.0, 0.05),
        sell_mult=(0.5, 2.0, 0.05),
        stop_loss=(0.0, 0.2, 0.01),
    ),
    "dual": dict(
        linlen=(20, 120, 5),
        smaalen=(10, 150, 5),
        short_win=(20, 80, 5),
        long_win=(100, 200, 10),
        factor=(20.0, 60.0, 1.0),
        buy_mult=(0.1, 1.0, 0.05),
        sell_mult=(0.5, 2.0, 0.05),
        stop_loss=(0.0, 0.2, 0.01),
    ),
    "RMA": dict(
        linlen=(20, 120, 5),
        smaalen=(10, 150, 5),
        rma_len=(20, 120, 5),
        dev_len=(10, 60, 5),
        factor=(20.0, 60.0, 1.0),
        buy_mult=(0.1, 1.0, 0.05),
        sell_mult=(0.5, 2.0, 0.05),
        stop_loss=(0.0, 0.2, 0.01),
    ),
    "ssma_turn": dict(
        linlen=(10, 50, 5),
        smaalen=(20, 120, 10),
        factor=(20.0, 80.0, 10.0),
        prom_factor=(30, 80, 5),
        min_dist=(5, 15, 1),
        buy_shift=(0, 5, 1),
        exit_shift=(0, 7, 1),
        vol_window=(10, 30, 5),
        quantile_win=(50, 150, 10),
        signal_cooldown_days=(1, 7, 1),
        buy_mult=(0.2, 0.8, 0.1),
        sell_mult=(0.2, 0.8, 0.1),
        stop_loss=(0.05, 0.15, 0.01),
    ),
}

# 策略層級顯式關聯 ind_keys/bt_keys
IND_BT_KEYS = {
    strat: cfg.STRATEGY_PARAMS[strat]["ind_keys"] + cfg.STRATEGY_PARAMS[strat]["bt_keys"]
    for strat in PARAM_SPACE
}

# 各部份分數權重
SCORE_WEIGHTS = dict(
    total_return=2.0,        # 高優先級：完整回測報酬
    profit_factor=1.2,       # 高優先級：盈虧因子
    wf_min_return=1.0,       # 次高優先級：走查最差報酬
    sharpe_ratio=0.6,        # 次低優先級：夏普值
    max_drawdown=0.6,        # 次低優先級：最大回撤
    stress_avg_return=0.3,   # 最低優先級：壓力測試平均報酬
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

def compute_knn_stability(df_results: list, params: list, k: int = 5, metric: str = 'total_return') -> float:
    """計算 KNN 穩定性，評估過擬合風險。"""
    if len(df_results) < k + 1:
        logging.warning(f"試驗數量 {len(df_results)} 不足以計算 KNN 穩定性 (需要至少 {k+1})")
        return 0.0
    param_cols = [p for p in params if p in df_results[0]]
    if not param_cols:
        logging.warning(f"無有效參數用於 KNN 穩定性計算，參數: {params}")
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

def _backtest_once(strat: str, params: dict, trial_results: list) -> tuple[float, int, float, float, float]:
    """單段完整回測，回傳 (total_return, num_trades, sharpe_ratio, max_drawdown, profit_factor)。"""
    try:
        df_price, df_factor = data_loader.load_data(TICKER)
        if df_price.empty:
            logging.error(f"價格數據為空，策略: {strat}, 參數: {params}")
            return -np.inf, 0, 0.0, 0.0, 0.0
    except Exception as e:
        logging.error(f"數據加載失敗，策略: {strat}, 錯誤: {e}")
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
        # 計算 profit_factor（若未提供，則從交易記錄計算）
        profit_factor = metrics.get("profit_factor", 1.0)
        if "profit_factor" not in metrics:
            trades = bt["trades"]
            if trades:
                gains = [t[1] for t in trades if t[1] > 0]
                losses = [abs(t[1]) for t in trades if t[1] < 0]
                profit_factor = sum(gains) / sum(losses) if losses and sum(losses) > 0 else np.inf
                profit_factor = min(profit_factor, 10.0)  # 限制極端值
        # 將結果添加到 trial_results
        trial_results.append({
            "total_return": metrics["total_return"],
            "num_trades": metrics.get("num_trades", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "profit_factor": profit_factor,
            **params
        })
        return (
            metrics["total_return"],
            metrics.get("num_trades", 0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("max_drawdown", 0.0),
            profit_factor
        )
    except Exception as e:
        logging.error(f"回測失敗，策略: {strat}, 參數: {params}, 錯誤: {e}")
        return -np.inf, 0, 0.0, 0.0, 0.0

def _wf_min_return(strat: str, params: dict) -> float:
    """計算走查期間最差回報。"""
    try:
        df_price, df_factor = data_loader.load_data(TICKER)
        if df_price.empty:
            logging.warning(f"走查測試數據為空，策略: {strat}, 參數: {params}")
            return -np.inf
        valid_periods = [
            (start, end) for start, end in WF_PERIODS
            if start in df_price.index and end in df_price.index
        ]
        if not valid_periods:
            logging.warning(f"無有效的走查期間，策略: {strat}, 參數: {params}")
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
            logging.warning(f"走查測試無結果，策略: {strat}, 參數: {params}")
            return -np.inf

        valid_returns = []
        for r in results:
            try:
                total_return = r["metrics"]["total_return"]
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logging.warning(f"走查期間缺少 total_return，策略: {strat}, 參數: {params}, 錯誤: {e}")
                continue
        return min(valid_returns) if valid_returns else -np.inf
    except Exception as e:
        logging.error(f"走查測試失敗，策略: {strat}, 參數: {params}, 錯誤: {e}")
        return -np.inf

def _stress_avg_return(strat: str, params: dict) -> float:
    """計算壓力區平均回報。"""
    if not STRESS_PERIODS:
        return 0.0
    try:
        df_price, df_factor = data_loader.load_data(TICKER)
        if df_price.empty:
            logging.warning(f"壓力測試數據為空，策略: {strat}, 參數: {params}")
            return -np.inf
        valid_periods = [
            (start, end) for start, end in STRESS_PERIODS
            if start in df_price.index and end in df_price.index
        ]
        if not valid_periods:
            logging.warning(f"無有效的壓力測試期間，策略: {strat}, 參數: {params}")
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
            logging.warning(f"壓力測試無結果，策略: {strat}, 參數: {params}")
            return -np.inf

        valid_returns = []
        for r in results:
            try:
                total_return = r["metrics"]["total_return"]
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logging.warning(f"壓力測試期間缺少 total_return，策略: {strat}, 參數: {params}, 錯誤: {e}")
                continue
        return float(np.mean(valid_returns)) if valid_returns else -np.inf
    except Exception as e:
        logging.error(f"壓力測試失敗，策略: {strat}, 參數: {params}, 錯誤: {e}")
        return -np.inf

# ---------- Optuna 目標 ----------

def objective(trial: optuna.Trial):
    strat = trial.suggest_categorical("strategy", list(PARAM_SPACE.keys()))
    params = _sample_params(trial, strat)

    # 儲存試驗結果以計算穩定性
    trial_results = trial.study.user_attrs.get("trial_results", [])
    
    # 完整回測
    total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor = _backtest_once(strat, params, trial_results)
    
    # 第一層：硬性門檻
    if total_ret == -np.inf:
        return -np.inf
    if not (15 <= n_trades <= 60):
        logging.info(f"試驗被剔除：交易次數 {n_trades} 不在 15-60 範圍，策略: {strat}")
        return -np.inf
    if total_ret <= 5.0:  # 500% = 5.0
        logging.info(f"試驗被剔除：完整回測報酬 {total_ret:.2%} <= 500%，策略: {strat}")
        return -np.inf

    # 更新 trial_results
    trial.study.set_user_attr("trial_results", trial_results)

    # 走查最差報酬
    min_wf_ret = _wf_min_return(strat, params)
    if min_wf_ret == -np.inf:
        return -np.inf

    # 壓力測試平均報酬
    avg_stress_ret = _stress_avg_return(strat, params)

    # 第二層：主要排序指標（Z-score 標準化）
    metrics = {
        "total_return": total_ret,
        "profit_factor": np.log(max(profit_factor, 1.0)),  # 對數平滑
        "wf_min_return": min_wf_ret,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": -abs(max_drawdown),  # 負值加權
        "stress_avg_return": avg_stress_ret
    }
    # 簡單標準化（基於當前試驗的歷史數據）
    z_scores = {}
    for key in metrics:
        historical = [r[key] for r in trial_results if key in r]
        if historical:
            mean = np.mean(historical)
            std = np.std(historical)
            z_scores[key] = (metrics[key] - mean) / std if std > 0 else metrics[key]
        else:
            z_scores[key] = metrics[key]

    # 計算加權分數
    score = (
        SCORE_WEIGHTS["total_return"] * z_scores["total_return"]
        + SCORE_WEIGHTS["profit_factor"] * z_scores["profit_factor"]
        + SCORE_WEIGHTS["wf_min_return"] * z_scores["wf_min_return"]
        + SCORE_WEIGHTS["sharpe_ratio"] * z_scores["sharpe_ratio"]
        + SCORE_WEIGHTS["max_drawdown"] * z_scores["max_drawdown"]
        + SCORE_WEIGHTS["stress_avg_return"] * z_scores["stress_avg_return"]
    )

    # 第三層：過擬合懲罰
    stab = 0.0
    if len(trial_results) >= 50:  # 僅在試驗數 ≥ 50 時計算
        stab = compute_knn_stability(trial_results, params=['linlen', 'smaalen', 'buy_mult'], k=5)
        if stab > 0.05:
            penalty = 0.2 * (stab - 0.05) / 0.05
            score *= (1 - penalty)
            logging.info(f"過擬合懲罰：穩定性得分={stab:.4f}, 懲罰值={penalty:.4f}")

    # 儲存指標
    trial.set_user_attr("total_return", total_ret)
    trial.set_user_attr("num_trades", n_trades)
    trial.set_user_attr("sharpe_ratio", sharpe_ratio)
    trial.set_user_attr("max_drawdown", max_drawdown)
    trial.set_user_attr("profit_factor", profit_factor)
    trial.set_user_attr("min_wf_return", min_wf_ret)
    trial.set_user_attr("avg_stress_return", avg_stress_ret)
    trial.set_user_attr("stability_score", stab)

    return score

# ---------- 執行入口 ----------

if __name__ == "__main__":
    optuna_sqlite = Path(cfg.RESULT_DIR) / f"optuna_00631L_{TIMESTAMP}.sqlite3"
    study = optuna.create_study(
        study_name="00631L_optuna_v1",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        storage=f"sqlite:///{optuna_sqlite}"
    )

    n_trials = int(os.getenv("OPTUNA_TRIALS", 200))
    logging.info(f"開始最佳化，共 {n_trials} trials …")
    study.optimize(objective, n_trials=n_trials, n_jobs=cfg.N_JOBS, show_progress_bar=True)

    logging.info("最佳 trial:")
    best = study.best_trial
    logging.info(f"Strategy        : {best.params['strategy']}")
    logging.info(f"Params          : { {k:v for k,v in best.params.items() if k!='strategy'} }")
    logging.info(f"Robust Score    : {best.value:.4f}")
    logging.info(f"Other metrics   : {best.user_attrs}")

    # 保存最佳參數
    results = {
        "best_robust_score": best.value,
        "best_strategy": best.params["strategy"],
        "best_params": {k: v for k, v in best.params.items() if k != "strategy"},
        "best_metrics": best.user_attrs
    }
    result_file = cfg.RESULT_DIR / f"optuna_best_params_{TICKER.replace('^','')}_{TIMESTAMP}.json"
    pd.Series(results).to_json(result_file, indent=2)
    logging.info(f"最佳參數已保存至 {result_file}")