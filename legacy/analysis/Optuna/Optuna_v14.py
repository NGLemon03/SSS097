import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import optuna
import ast
import shutil
from datetime import datetime
from analysis.data_loader import load_data
from SSSv096 import backtest_unified, compute_single, compute_dual, compute_RMA, compute_ssma_turn_combined
from analysis.metrics import calculate_max_drawdown
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# 設定訓練/OOS區間
TRAIN_START = "2014-10-23"
TRAIN_END = "2021-12-31"
OOS_START = "2022-01-01"
OOS_END = "2025-06-15"
TICKER = "00631L.TW"

# 使用獨立 cache 目錄，避免與 v13 衝突
CACHE_DIR = Path("../cache_v14")
# 移除自動清理快取目錄，避免與其他程式衝突
# if CACHE_DIR.exists():
#     shutil.rmtree(CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 交易次數篩選
MIN_NUM_TRADES = 10  # 最小交易次數
MAX_NUM_TRADES = 240  # 最大交易次數

# 參數空間（可依需求擴充）
PARAM_SPACE = {
    "single": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), devwin=(5, 180, 1),
        factor=(40, 40, 1), buy_mult=(0.1, 2.5, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.2)),
    "dual": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), short_win=(10, 100, 5), long_win=(40, 240, 10),
        factor=(40, 40, 1), buy_mult=(0.2, 2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.1)),
    "RMA": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), rma_len=(20, 100, 5), dev_len=(10, 100, 5),
        factor=(40, 40, 1), buy_mult=(0.2, 2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.1)),
    "ssma_turn": dict(
        linlen=(10, 240, 5), smaalen=(10, 240, 5), factor=(40.0, 40.0, 1), prom_factor=(5, 70, 1),
        min_dist=(5, 20, 1), buy_shift=(0, 7, 1), exit_shift=(0, 7, 1), vol_window=(5, 90, 5), quantile_win=(5, 180, 10),
        signal_cooldown_days=(1, 7, 1), buy_mult=(0.5, 2, 0.05), sell_mult=(0.2, 3, 0.1), stop_loss=(0.00, 0.55, 0.1)),
}

# v13 的 SCORE_WEIGHTS
SCORE_WEIGHTS = dict(
    total_return=2.5,
    sharpe_ratio=0.2,
    max_drawdown=0.1
)

# v14 的權重設計（修正版 - 移除OOS_return避免資料洩漏）
WEIGHTS = {
    'PBO': 0.25,        # PBO 分數（穩健性）- 提高權重
    'train_score': 0.50, # 訓練集綜合分數 - 主要權重
    'knn_stability': 0.25  # KNN 穩定性分數 - 提高權重
}

# 新增過擬合檢測門檻
OVERFITTING_THRESHOLDS = {
    'min_knn_stability': 0.3,  # KNN穩定性最小門檻
    'max_sharpe_degradation': 0.5,  # 最大夏普比率退化
    'max_return_degradation': 0.2,  # 最大報酬率退化
}

STRAT_FUNC_MAP = {
    'single': compute_single,
    'dual': compute_dual,
    'RMA': compute_RMA,
    'ssma_turn': compute_ssma_turn_combined
}

# 全局變數用於 KNN 穩定性計算
trial_results = []

def compute_knn_stability(df_results: list, params: list, k: int = 5, metric: str = 'total_return') -> float:
    """計算 KNN 穩定性分數"""
    if len(df_results) < k + 1:
        return 0.0
    if not df_results or not isinstance(df_results[0], dict):
        return 0.0
    
    # 找出可用的參數欄位
    param_cols = [f"param_{p}" for p in params if f"param_{p}" in df_results[0]]
    if not param_cols:
        return 0.0
    
    try:
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
    except Exception as e:
        print(f"KNN 穩定性計算失敗: {e}")
        return 0.0

# 參數採樣（參數名稱唯一化）
def sample_params(trial, strat):
    space = PARAM_SPACE[strat]
    params = {}
    for k, v in space.items():
        param_name = f"{strat}_{k}"
        if isinstance(v[0], int):
            low, high, step = int(v[0]), int(v[1]), int(v[2])
            params[k] = trial.suggest_int(param_name, low, high, step=step)
        else:
            low, high, step = v
            params[k] = round(trial.suggest_float(param_name, low, high, step=step), 3)
    return params

def run_backtest(strat, params, df_price, df_factor):
    if strat == "ssma_turn":
        calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
        ssma_params = {k: v for k, v in params.items() if k in calc_keys}
        buy_mult = params.get('buy_mult', 0.5)
        sell_mult = params.get('sell_mult', 0.5)
        stop_loss = params.get('stop_loss', 0.0)
        df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_price, df_factor, **ssma_params)
        if df_ind.empty:
            return np.nan, np.nan, np.nan, 0
        result = backtest_unified(df_ind, strat, params, buy_dates, sell_dates, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
    else:
        if strat == 'single':
            df_ind = compute_single(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['devwin'])
        elif strat == 'dual':
            df_ind = compute_dual(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['short_win'], params['long_win'])
        elif strat == 'RMA':
            df_ind = compute_RMA(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'])
        else:
            return np.nan, np.nan, np.nan, 0
        if df_ind.empty:
            return np.nan, np.nan, np.nan, 0
        result = backtest_unified(df_ind, strat, params, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
    
    eq = result.get('equity_curve')
    trades_df = result.get('trades_df', pd.DataFrame())
    n_trades = len(trades_df) if not trades_df.empty else 0
    
    if eq is None or eq.empty:
        return np.nan, np.nan, np.nan, n_trades
    
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    daily_returns = eq.pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    mdd = calculate_max_drawdown(eq)
    return total_return, sharpe, mdd, n_trades

def calculate_pbo_score(equity_curve, trades_df):
    """計算 PBO 分數（簡化版）"""
    if equity_curve is None or equity_curve.empty:
        return 0.0
    
    # 簡化的 PBO 計算：基於權益曲線的穩定性
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) < 30:
        return 0.0
    
    # 計算正報酬期間比例
    positive_periods = (daily_returns > 0).sum() / len(daily_returns)
    
    # 計算最大回撤的穩定性（回撤越小越好）
    mdd = calculate_max_drawdown(equity_curve)
    mdd_score = max(0, 1 - abs(mdd))
    
    # 綜合 PBO 分數
    pbo_score = 0.6 * positive_periods + 0.4 * mdd_score
    return pbo_score

def objective(trial):
    global trial_results
    
    strat = trial.suggest_categorical('strategy', list(PARAM_SPACE.keys()))
    params = sample_params(trial, strat)
    
    # 載入訓練集
    df_price_train, df_factor_train = load_data(TICKER, start_date=TRAIN_START, end_date=TRAIN_END, smaa_source='Self')
    train_ret, train_sharpe, train_mdd, train_trades = run_backtest(strat, params, df_price_train, df_factor_train)
    
    # 載入 OOS
    df_price_oos, df_factor_oos = load_data(TICKER, start_date=OOS_START, end_date=OOS_END, smaa_source='Self')
    oos_ret, oos_sharpe, oos_mdd, oos_trades = run_backtest(strat, params, df_price_oos, df_factor_oos)
    
    # 交易次數篩選
    if np.isnan(train_ret) or np.isnan(oos_ret):
        score = -np.inf
    elif not (MIN_NUM_TRADES <= train_trades <= MAX_NUM_TRADES):
        score = -np.inf
    elif train_ret <= 0.2 or train_mdd < -0.8:  # 基本篩選條件
        score = -np.inf
    else:
        # 1. 訓練集綜合分數（v13 算式）
        train_score = (SCORE_WEIGHTS["total_return"] * train_ret +
                       SCORE_WEIGHTS["sharpe_ratio"] * train_sharpe + 
                       SCORE_WEIGHTS["max_drawdown"] * (1 - abs(train_mdd)))
        
        # 2. OOS 報酬率
        oos_return_score = oos_ret
        
        # 3. PBO 分數（基於訓練集權益曲線）
        # 重新執行訓練集回測以取得完整結果
        if strat == "ssma_turn":
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
            ssma_params = {k: v for k, v in params.items() if k in calc_keys}
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_price_train, df_factor_train, **ssma_params)
            if not df_ind.empty:
                train_result = backtest_unified(df_ind, strat, params, buy_dates, sell_dates, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
            else:
                train_result = None
        else:
            if strat == 'single':
                df_ind = compute_single(df_price_train, df_factor_train, params['linlen'], params['factor'], params['smaalen'], params['devwin'])
            elif strat == 'dual':
                df_ind = compute_dual(df_price_train, df_factor_train, params['linlen'], params['factor'], params['smaalen'], params['short_win'], params['long_win'])
            elif strat == 'RMA':
                df_ind = compute_RMA(df_price_train, df_factor_train, params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'])
            else:
                train_result = None
            
            if df_ind is not None and not df_ind.empty:
                train_result = backtest_unified(df_ind, strat, params, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
            else:
                train_result = None
        
        if train_result:
            pbo_score = calculate_pbo_score(train_result.get('equity_curve'), train_result.get('trades_df', pd.DataFrame()))
        else:
            pbo_score = 0.0
        
        # 4. KNN 穩定性分數
        # 將當前試驗結果加入 trial_results 用於 KNN 計算
        current_result = {
            'total_return': train_ret,
            'parameters': params,
            **{f"param_{k}": v for k, v in params.items()}
        }
        trial_results.append(current_result)
        
        # 計算 KNN 穩定性（基於訓練集報酬率）
        knn_stability = compute_knn_stability(trial_results, list(params.keys()), k=5, metric='total_return')
        
        # 5. 過擬合檢測和懲罰
        overfitting_penalty = 0.0
        
        # KNN穩定性檢查
        if knn_stability < OVERFITTING_THRESHOLDS['min_knn_stability']:
            overfitting_penalty += 0.3 * (OVERFITTING_THRESHOLDS['min_knn_stability'] - knn_stability)
        
        # 如果試驗數量足夠，計算樣本內外退化
        if len(trial_results) >= 10:
            # 簡單的過擬合檢測：基於最近試驗的穩定性
            recent_trials = trial_results[-10:]
            recent_returns = [t['total_return'] for t in recent_trials]
            return_std = np.std(recent_returns)
            return_mean = np.mean(recent_returns)
            
            # 如果報酬率變異過大，可能是過擬合
            if return_std > 0.1 and return_mean > 0.5:
                overfitting_penalty += 0.2
        
        # 6. 最終分數（應用過擬合懲罰）
        base_score = (WEIGHTS['PBO'] * pbo_score +
                     WEIGHTS['train_score'] * train_score +
                     WEIGHTS['knn_stability'] * knn_stability)
        
        score = base_score * (1 - overfitting_penalty)
    
    # 記錄所有指標
    trial.set_user_attr('train_return', train_ret)
    trial.set_user_attr('train_sharpe', train_sharpe)
    trial.set_user_attr('train_mdd', train_mdd)
    trial.set_user_attr('train_trades', train_trades)
    trial.set_user_attr('oos_return', oos_ret)
    trial.set_user_attr('oos_sharpe', oos_sharpe)
    trial.set_user_attr('oos_mdd', oos_mdd)
    trial.set_user_attr('oos_trades', oos_trades)
    trial.set_user_attr('pbo_score', pbo_score if 'pbo_score' in locals() else 0.0)
    trial.set_user_attr('train_score', train_score if 'train_score' in locals() else 0.0)
    trial.set_user_attr('knn_stability', knn_stability if 'knn_stability' in locals() else 0.0)
    trial.set_user_attr('overfitting_penalty', overfitting_penalty if 'overfitting_penalty' in locals() else 0.0)
    
    # 記錄原始參數（不帶前綴）
    for k, v in params.items():
        trial.set_user_attr(k, v)
    
    return score

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=24000, show_progress_bar=True)
    
    # 匯出結果
    results = []
    for t in study.trials:
        row = dict()
        row['strategy'] = t.params.get('strategy', None)
        
        # 輸出原始參數名（不帶前綴）
        strat = row['strategy']
        if strat is not None:
            for k in PARAM_SPACE[strat].keys():
                row[k] = t.user_attrs.get(k, np.nan)
        
        # 輸出分數與指標
        row['score'] = t.value
        row['train_return'] = t.user_attrs.get('train_return', np.nan)
        row['train_sharpe'] = t.user_attrs.get('train_sharpe', np.nan)
        row['train_mdd'] = t.user_attrs.get('train_mdd', np.nan)
        row['train_trades'] = t.user_attrs.get('train_trades', np.nan)
        row['oos_return'] = t.user_attrs.get('oos_return', np.nan)
        row['oos_sharpe'] = t.user_attrs.get('oos_sharpe', np.nan)
        row['oos_mdd'] = t.user_attrs.get('oos_mdd', np.nan)
        row['oos_trades'] = t.user_attrs.get('oos_trades', np.nan)
        row['pbo_score'] = t.user_attrs.get('pbo_score', np.nan)
        row['train_score'] = t.user_attrs.get('train_score', np.nan)
        row['knn_stability'] = t.user_attrs.get('knn_stability', np.nan)
        row['overfitting_penalty'] = t.user_attrs.get('overfitting_penalty', np.nan)
        
        results.append(row)
    
    df = pd.DataFrame(results)
    out_csv = Path("../results/optuna_v14_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"已輸出 {len(df)} 筆結果到 {out_csv}")

if __name__ == "__main__":
    main() 