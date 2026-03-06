# -*- coding: utf-8 -*-
# Optuna V15 - 整合多樣性篩選和多數據源支持
'''
* 命令列範例: 
# 隨機選數據源
python Optuna_15.py --strategy RMA --n_trials 10000  
# 指定數據源
python Optuna_15.py --strategy single --data_source "Factor (^TWII / 2412.TW)" --data_source_mode fixed --n_trials 1000  
# 依序遍歷所有數據源
python Optuna_15.py --strategy ssma_turn --data_source_mode sequential --n_trials 2000  
'''
import os
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)) 
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import optuna

from analysis import config as cfg
from analysis.logging_config import setup_logging

# 命令列參數解析
parser = argparse.ArgumentParser(description='Optuna V15 最佳化 00631L 策略')
parser.add_argument('--strategy', type=str, choices=['single', 'ssma_turn', 'RMA', 'all'], default='all', help='指定單一策略進行最佳化 (預設: all)')
parser.add_argument('--n_trials', type=int, default=5000, help='試驗次數 (預設: 5000)')
parser.add_argument('--data_source', type=str, choices=['Self', 'Factor (^TWII / 2412.TW)', 'Factor (^TWII / 2414.TW)'], default=None, help='指定單一數據源, 僅在 --data_source_mode=fixed 時有效')
parser.add_argument('--data_source_mode', type=str, choices=['random', 'fixed', 'sequential'], default='random', help='數據源選擇模式: random(隨機)、fixed(指定)、sequential(依序遍歷)')
args = parser.parse_args()

# 配置常數
TICKER = "00631L.TW"
START_DATE = "2010-01-01"
END_DATE = "2025-06-17"
MAX_NUM_TRADES = 200
MIN_NUM_TRADES = 5
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 壓力測試期間
STRESS_PERIODS = [
    ("2020-02-20", "2020-03-23"),  # COVID-19 崩盤
    ("2022-01-03", "2022-06-16"),  # 2022 上半年熊市
    ("2023-07-31", "2023-10-27"),  # 2023 下半年調整
]

# 參數空間
PARAM_SPACE = {
    "single": dict(
        linlen=(5, 240, 1),smaalen=(7, 240, 5),devwin=(5, 180, 1),
        factor=(40, 40, 1),buy_mult=(0.1, 2.5, 0.05),sell_mult=(0.5, 4.0, 0.05),stop_loss=(0.00, 0.55, 0.2),),
    "dual": dict(
        linlen=(5, 240, 1),smaalen=(7, 240, 5),short_win=(10, 100, 5),long_win=(40, 240, 10),
        factor=(40, 40, 1),buy_mult=(0.2, 2, 0.05),sell_mult=(0.5, 4.0, 0.05),stop_loss=(0.00, 0.55, 0.1),),
    "RMA": dict(
        linlen=(5, 240, 1),smaalen=(7, 240, 5),rma_len=(20, 100, 5),dev_len=(10, 100, 5),
        factor=(40, 40, 1),buy_mult=(0.2, 2, 0.05),sell_mult=(0.5, 4.0, 0.05),stop_loss=(0.00, 0.55, 0.1),),
    "ssma_turn": dict(
        linlen=(10, 240, 5),smaalen=(10, 240, 5),factor=(40.0, 40.0, 1),prom_factor=(5, 70, 1),
        min_dist=(5, 20, 1),buy_shift=(0, 7, 1),exit_shift=(0, 7, 1),vol_window=(5, 90, 5),quantile_win=(5, 180, 10),
        signal_cooldown_days=(1, 7, 1),buy_mult=(0.5, 2, 0.05),sell_mult=(0.2, 3, 0.1),stop_loss=(0.00, 0.55, 0.1),),
}
# 策略權重
STRATEGY_WEIGHTS = {
    "single": 0.4,
    "ssma_turn": 0.4,
    "RMA": 0.2
}

# 數據源配置
DATA_SOURCES = {
    "Self": "Self",
    "Factor (^TWII / 2412.TW)": "Factor (^TWII / 2412.TW)",
    "Factor (^TWII / 2414.TW)": "Factor (^TWII / 2414.TW)"
}

DATA_SOURCES_WEIGHTS = {
    "Self": 0.33,
    "Factor (^TWII / 2412.TW)": 0.33,
    "Factor (^TWII / 2414.TW)": 0.34
}

# 結果記錄
results_log = []
events_log = []

setup_logging()
logger = logging.getLogger("Optuna_v15")

def log_to_results(event_type: str, details: str, **kwargs):
    """記錄試驗結果"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record = {
        "Timestamp": timestamp,
        "Event Type": event_type,
        "Details": details,
        **kwargs
    }
    results_log.append(record)

def log_to_events(event_type: str, details: str, **kwargs):
    """記錄事件"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record = {
        "Timestamp": timestamp,
        "Event Type": event_type,
        "Details": details,
        **kwargs
    }
    events_log.append(record)

def sanitize_filename(filename: str) -> str:
    """清理檔案名中的特殊字符"""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()

def pick_topN_by_diversity(trials: List[Dict], metric_keys: List[str], top_n: int = 5) -> List[Dict]:
    """
    基於性能指標的多樣性 top N 試驗選擇，參考 OSv3 的實現
    """
    logger.info(f"開始多樣性篩選: {len(trials)} 個試驗, 目標選取 {top_n} 個")
    
    # 轉換為 DataFrame
    df = pd.DataFrame(trials)
    
    # 使用關鍵指標進行分組
    key_metrics = ['num_trades', 'excess_return_stress', 'avg_hold_days']
    logger.info(f"使用關鍵指標: {key_metrics}")
    
    # 檢查必要指標是否存在
    missing_metrics = [metric for metric in key_metrics if metric not in df.columns]
    if missing_metrics:
        logger.warning(f"缺少指標: {missing_metrics}，將使用可用的指標")
        key_metrics = [metric for metric in key_metrics if metric in df.columns]
    
    if not key_metrics:
        logger.error("沒有可用的關鍵指標，無法進行篩選")
        return []
    
    # 針對不同指標特性的分組處理
    for metric in key_metrics:
        if metric not in df.columns:
            continue
            
        if metric == 'num_trades':
            # num_trades: 分級處理，每5次為一組，避免過於細碎
            df[f'grouped_{metric}'] = (df[metric] // 5) * 5
            logger.info(f"指標 {metric}: 原始值範圍 [{df[metric].min()}, {df[metric].max()}], 分級後範圍 [{df[f'grouped_{metric}'].min()}, {df[f'grouped_{metric}'].max()}]")
            
        elif metric == 'excess_return_stress':
            # excess_return_stress: 四捨五入到小數點後一位
            df[f'grouped_{metric}'] = df[metric].round(1)
            logger.info(f"指標 {metric}: 原始值範圍 [{df[metric].min():.3f}, {df[metric].max():.3f}], 四捨五入後範圍 [{df[f'grouped_{metric}'].min():.1f}, {df[f'grouped_{metric}'].max():.1f}]")
            
        elif metric == 'avg_hold_days':
            # avg_hold_days: 四捨五入到小數點後一位
            df[f'grouped_{metric}'] = df[metric].round(1)
            logger.info(f"指標 {metric}: 原始值範圍 [{df[metric].min():.3f}, {df[metric].max():.3f}], 四捨五入後範圍 [{df[f'grouped_{metric}'].min():.1f}, {df[f'grouped_{metric}'].max():.1f}]")
    
    # 按分組後的指標創建組別標識
    group_cols = [f'grouped_{metric}' for metric in key_metrics]
    df['group'] = df[group_cols].astype(str).agg('_'.join, axis=1)
    
    # 統計分組情況
    group_counts = df['group'].value_counts()
    logger.info(f"分組統計: 共 {len(group_counts)} 個不同組別")
    
    # 按分數排序並選擇每個分組中分數最高的試驗
    df_sorted = df.sort_values(by='score', ascending=False)
    chosen_trials = []
    seen_groups = set()
    
    logger.info("開始選取試驗...")
    
    for idx, row in df_sorted.iterrows():
        group = row['group']
        trial_num = row.get('trial_number', idx)
        score = row['score']
        
        if group not in seen_groups:
            chosen_trials.append(row.to_dict())
            seen_groups.add(group)
            logger.info(f"選取試驗 {trial_num}: score={score:.3f}, 組別={group[:50]}...")
        else:
            logger.debug(f"跳過試驗 {trial_num}: score={score:.3f}, 組別已存在")
        
        if len(chosen_trials) >= top_n:
            logger.info(f"已選取 {len(chosen_trials)} 個試驗，達到目標數量")
            break
    
    if len(chosen_trials) < top_n:
        logger.warning(f"只選取了 {len(chosen_trials)} 個試驗，少於目標 {top_n} 個")
    
    return chosen_trials

def sample_params(trial: optuna.Trial, strat: str) -> Dict:
    """採樣參數"""
    space = PARAM_SPACE[strat]
    params = {}
    for k, v in space.items():
        if isinstance(v[0], int):
            low, high, step = int(v[0]), int(v[1]), int(v[2])
            params[k] = trial.suggest_int(k, low, high, step=step)
        else:
            low, high, step = v
            params[k] = round(trial.suggest_float(k, low, high, step=step), 3)
    return params

def run_backtest(strat: str, params: dict, df_price: pd.DataFrame, df_factor: pd.DataFrame) -> Tuple[float, int, float, float, float, List, pd.Series, float]:
    """執行回測，返回績效指標"""
    try:
        from SSSv096 import (
            compute_single, compute_ssma_turn_combined, compute_RMA,
            backtest_unified
        )
        
        # 根據策略類型計算指標
        if strat == 'single':
            df_ind = compute_single(df_price, df_factor, 
                                  params['linlen'], params['factor'], 
                                  params['smaalen'], params['devwin'])
        elif strat == 'ssma_turn':
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 
                        'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
            ssma_params = {k: v for k, v in params.items() if k in calc_keys}
            df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                df_price, df_factor, **ssma_params
            )
        elif strat == 'RMA':
            df_ind = compute_RMA(df_price, df_factor,
                               params['linlen'], params['factor'], 
                               params['smaalen'], params['rma_len'], params['dev_len'])
        else:
            raise ValueError(f"不支持的策略類型: {strat}")
        
        if df_ind.empty:
            logger.warning(f"{strat} 策略計算失敗，返回默認值")
            return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0
        
        # 執行回測
        if strat == 'ssma_turn':
            result = backtest_unified(df_ind, strat, params, buy_dates, sell_dates)
        else:
            result = backtest_unified(df_ind, strat, params)
        
        trades = result['trades']
        trade_df = result['trade_df']
        metrics = result['metrics']
        equity_curve = result['equity_curve']
        
        # 計算平均持倉天數
        if not trade_df.empty:
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
            avg_hold_days = float(np.mean(holding_periods)) if holding_periods else 0.0
        else:
            avg_hold_days = 0.0
        
        return (metrics['total_return'], metrics['num_trades'], 
                metrics['sharpe_ratio'], metrics['max_drawdown'], 
                metrics['profit_factor'], trades, equity_curve, avg_hold_days)
        
    except Exception as e:
        logger.error(f"回測執行失敗: {e}")
        return -np.inf, 0, 0.0, 0.0, 0.0, [], pd.Series(), 0.0

def calculate_pbo_score(equity_curve: pd.Series, trades_df: pd.DataFrame) -> float:
    """計算 PBO 分數"""
    if equity_curve.empty or len(equity_curve) < 100:
        return 0.0
    
    try:
        # 計算每日回報率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 計算滾動夏普比率
        window = min(252, len(daily_returns) // 4)
        rolling_sharpe = daily_returns.rolling(window).mean() / daily_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()
        
        if len(rolling_sharpe) < 10:
            return 0.0
        
        # 計算穩定性指標
        sharpe_std = rolling_sharpe.std()
        sharpe_mean = rolling_sharpe.mean()
        
        if sharpe_std == 0:
            return 0.0
        
        # PBO 分數：基於夏普比率的變異係數
        pbo_score = min(1.0, sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else 1.0)
        
        return float(pbo_score)
        
    except Exception as e:
        logger.warning(f"PBO 計算失敗: {e}")
        return 0.0

def select_data_source(trial: optuna.Trial, data_source_mode: str = 'random', fixed_data_source: Optional[str] = None) -> str:
    """
    選擇數據源，支持三種模式：
    - random: 隨機選擇
    - fixed: 固定數據源
    - sequential: 順序選擇
    """
    if data_source_mode == 'sequential':
        # 順序模式：根據試驗編號循環選擇
        data_sources = list(DATA_SOURCES.keys())
        data_source = data_sources[trial.number % len(data_sources)]
        logger.info(f"順序模式選擇數據源: {data_source}")
    elif data_source_mode == 'fixed':
        # 固定模式：使用指定的數據源
        if not fixed_data_source:
            raise ValueError("固定數據源模式下必須指定 fixed_data_source")
        data_source = fixed_data_source
        logger.info(f"固定模式使用數據源: {data_source}")
    else:
        # 隨機模式：根據權重隨機選擇
        data_source = np.random.choice(list(DATA_SOURCES_WEIGHTS.keys()), 
                                      p=list(DATA_SOURCES_WEIGHTS.values()))
        logger.info(f"隨機模式選擇數據源: {data_source}")
    
    trial.set_user_attr("data_source", data_source)
    return data_source

def calculate_stress_metrics(equity_curve: pd.Series, df_price: pd.DataFrame) -> dict:
    """計算壓力測試指標"""
    stress_metrics = {}
    
    for start, end in STRESS_PERIODS:
        try:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            
            if start_ts not in equity_curve.index or end_ts not in equity_curve.index:
                logger.warning(f"壓力測試期間 {start} → {end} 不在數據範圍內")
                continue
            
            # 計算期間回報率
            period_return = equity_curve.loc[end_ts] / equity_curve.loc[start_ts] - 1
            
            # 計算期間最大回撤
            period_equity = equity_curve.loc[start_ts:end_ts]
            if len(period_equity) > 1:
                roll_max = period_equity.cummax()
                drawdown = period_equity / roll_max - 1
                period_mdd = drawdown.min()
            else:
                period_mdd = 0.0
            
            # 計算買入持有回報率
            if start_ts in df_price.index and end_ts in df_price.index:
                bh_return = df_price.loc[end_ts, 'close'] / df_price.loc[start_ts, 'close'] - 1
                excess_return = period_return - bh_return
            else:
                excess_return = period_return
            
            stress_metrics[(start, end)] = {
                'return': period_return,
                'mdd': period_mdd,
                'excess_return': excess_return
            }
            
        except Exception as e:
            logger.warning(f"壓力測試期間 {start} → {end} 計算失敗: {e}")
            continue
    
    return stress_metrics

def append_trial_result(trial_results: List[Dict], trial: optuna.Trial, params: Dict, score: float,
                        equity_curve: pd.Series, strategy: str, data_source: str, extra_metrics: Optional[Dict] = None):
    """
    追加試驗結果至 trial_results 並更新 study.user_attrs
    """
    params_flat = {f"param_{k}": v for k, v in params.items()}
    record = {
        "trial_number": str(trial.number).zfill(5),
        "parameters": params,
        "score": score,
        "strategy": strategy,
        "data_source": data_source,
        **params_flat
    }
    if extra_metrics:
        record.update(extra_metrics)
    
    logger.debug(f"追加試驗記錄: trial_number={record['trial_number']}, 鍵={list(record.keys())}")
    trial_results.append(record)
    trial.study.set_user_attr("trial_results", trial_results)

def objective(trial: optuna.Trial, data_source_mode: str = 'random', fixed_data_source: Optional[str] = None, strategy_mode: str = 'all') -> float:
    """主要目標函數"""
    try:
        # 選擇策略
        if strategy_mode == 'all':
            strat = np.random.choice(list(STRATEGY_WEIGHTS.keys()), p=list(STRATEGY_WEIGHTS.values()))
        else:
            strat = strategy_mode
        trial.set_user_attr("strategy", strat)
        
        # 選擇數據源
        data_source = select_data_source(trial, data_source_mode, fixed_data_source)
        
        # 採樣參數
        params = sample_params(trial, strat)
        
        logger.info(f"試驗 {trial.number} 開始，策略: {strat}, 數據源: {data_source}")
        
        # 載入數據
        from SSSv096 import load_data
        df_price, df_factor = load_data(TICKER, START_DATE, END_DATE, data_source)
        
        if df_price.empty:
            logger.error(f"價格數據為空, 策略: {strat}")
            # 記錄失敗的試驗
            trial_results = trial.study.user_attrs.get("trial_results", [])
            append_trial_result(trial_results, trial, params, -np.inf, 
                              pd.Series(), strat, data_source, 
                              {"total_return": -np.inf, "num_trades": 0, "sharpe_ratio": 0.0, 
                               "max_drawdown": 0.0, "profit_factor": 0.0, "avg_hold_days": 0.0})
            log_to_results("試驗被剔除", f"試驗 {trial.number} 價格數據為空", 
                          trial_number=trial.number, strategy=strat, data_source=data_source, score=-np.inf)
            return -np.inf
        
        # 執行回測
        (total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, 
         trades, equity_curve, avg_hold_days) = run_backtest(strat, params, df_price, df_factor)
        
        # 基本篩選條件
        if (total_ret == -np.inf or 
            not (MIN_NUM_TRADES <= n_trades <= MAX_NUM_TRADES) or 
            total_ret <= 0.1 or 
            max_drawdown < -0.8 or 
            profit_factor < 0.1):
            logger.info(f"試驗 {trial.number} 未通過基本篩選")
            # 記錄失敗的試驗
            trial_results = trial.study.user_attrs.get("trial_results", [])
            append_trial_result(trial_results, trial, params, -np.inf, 
                              equity_curve, strat, data_source, 
                              {"total_return": total_ret, "num_trades": n_trades, "sharpe_ratio": sharpe_ratio, 
                               "max_drawdown": max_drawdown, "profit_factor": profit_factor, "avg_hold_days": avg_hold_days})
            log_to_results("試驗被剔除", f"試驗 {trial.number} 未通過基本篩選", 
                          trial_number=trial.number, strategy=strat, data_source=data_source, score=-np.inf)
            return -np.inf
        
        # 計算額外指標
        pbo_score = calculate_pbo_score(equity_curve, pd.DataFrame())
        stress_metrics = calculate_stress_metrics(equity_curve, df_price)
        
        # 計算壓力測試平均超額回報
        excess_returns = [m['excess_return'] for m in stress_metrics.values()]
        avg_excess_return = np.mean(excess_returns) if excess_returns else 0.0
        
        # 計算綜合分數
        base_score = total_ret
        risk_penalty = max(0, abs(max_drawdown) * 0.5)
        stability_bonus = (1 - pbo_score) * 0.1
        stress_bonus = max(0, avg_excess_return) * 0.2
        trade_frequency_bonus = 0.05 if 10 <= n_trades <= 50 else 0.0
        
        final_score = float(base_score - risk_penalty + stability_bonus + 
                      stress_bonus + trade_frequency_bonus)
        
        # 設置試驗屬性
        trial.set_user_attr("total_return", total_ret)
        trial.set_user_attr("num_trades", n_trades)
        trial.set_user_attr("sharpe_ratio", sharpe_ratio)
        trial.set_user_attr("max_drawdown", max_drawdown)
        trial.set_user_attr("profit_factor", profit_factor)
        trial.set_user_attr("avg_hold_days", avg_hold_days)
        trial.set_user_attr("pbo_score", pbo_score)
        trial.set_user_attr("avg_excess_return", avg_excess_return)
        trial.set_user_attr("excess_return_stress", avg_excess_return)
        
        # 記錄成功的試驗
        trial_results = trial.study.user_attrs.get("trial_results", [])
        extra_metrics = {
            "total_return": total_ret,
            "num_trades": n_trades,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "avg_hold_days": avg_hold_days,
            "pbo_score": pbo_score,
            "avg_excess_return": avg_excess_return
        }
        append_trial_result(trial_results, trial, params, final_score, 
                          equity_curve, strat, data_source, extra_metrics)
        
        # 記錄試驗結果
        details = (f"策略: {strat}, 數據源: {data_source}, 總報酬: {total_ret:.3f}, "
                  f"交易次數: {n_trades}, 夏普比率: {sharpe_ratio:.3f}, "
                  f"最大回撤: {max_drawdown:.3f}, 獲利因子: {profit_factor:.3f}, "
                  f"平均持倉天數: {avg_hold_days:.1f}, 最終分數: {final_score:.3f}")
        
        log_to_results("試驗結果", details, 
                      trial_number=trial.number, strategy=strat, data_source=data_source, 
                      score=final_score, total_return=total_ret, num_trades=n_trades,
                      sharpe_ratio=sharpe_ratio, max_drawdown=max_drawdown, 
                      profit_factor=profit_factor, avg_hold_days=avg_hold_days)
        
        logger.info(f"試驗 {trial.number} 完成，最終分數: {final_score:.4f}")
        
        return final_score
        
    except Exception as e:
        logger.error(f"試驗 {trial.number} 執行失敗: {e}")
        # 記錄錯誤的試驗
        trial_results = trial.study.user_attrs.get("trial_results", [])
        append_trial_result(trial_results, trial, params if 'params' in locals() else {}, -np.inf, 
                          pd.Series(), strat if 'strat' in locals() else 'unknown', 
                          data_source if 'data_source' in locals() else 'unknown', 
                          {"total_return": -np.inf, "num_trades": 0, "sharpe_ratio": 0.0, 
                           "max_drawdown": 0.0, "profit_factor": 0.0, "avg_hold_days": 0.0})
        log_to_events("試驗錯誤", f"試驗 {trial.number} 執行失敗: {e}", 
                     trial_number=trial.number, strategy=strat if 'strat' in locals() else 'unknown')
        return -np.inf

def main():
    """主函數，支持命令列參數和輸出檔案"""
    logger.info(f"開始 Optuna V15 測試 - 策略: {args.strategy}, 數據源模式: {args.data_source_mode}, 試驗次數: {args.n_trials}")
    
    # 確保結果目錄存在
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)
    
    if args.data_source_mode == 'sequential':
        # 依序遍歷所有數據源
        data_sources = list(DATA_SOURCES.keys())
        for data_source in data_sources:
            logger.info(f"開始處理數據源: {data_source}")
            
            # 創建 study
            safe_ds = sanitize_filename(data_source)
            optuna_sqlite = result_dir / f"optuna_{args.strategy}_{safe_ds}_{TIMESTAMP}.sqlite3"
            study = optuna.create_study(
                study_name=f"00631L_optuna_{args.strategy}_{safe_ds}",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
                storage=f"sqlite:///{optuna_sqlite}"
            )
            
            # 運行試驗
            study.optimize(
                lambda trial: objective(trial, 'fixed', data_source, args.strategy), 
                n_trials=args.n_trials, 
                n_jobs=1, 
                show_progress_bar=True
            )
            
            # 輸出結果
            output_results(study, args.strategy, data_source, result_dir)
            
            # 清空記錄
            results_log.clear()
            events_log.clear()
            
    elif args.data_source_mode == 'fixed':
        # 固定數據源模式
        if not args.data_source:
            logger.error("固定數據源模式下必須指定 --data_source")
            return
        
        logger.info(f"使用固定數據源: {args.data_source}")
        
        # 創建 study
        safe_ds = sanitize_filename(args.data_source)
        optuna_sqlite = result_dir / f"optuna_{args.strategy}_{safe_ds}_{TIMESTAMP}.sqlite3"
        study = optuna.create_study(
            study_name=f"00631L_optuna_{args.strategy}_{safe_ds}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
            storage=f"sqlite:///{optuna_sqlite}"
        )
        
        # 運行試驗
        study.optimize(
            lambda trial: objective(trial, 'fixed', args.data_source, args.strategy), 
            n_trials=args.n_trials, 
            n_jobs=1, 
            show_progress_bar=True
        )
        
        # 輸出結果
        output_results(study, args.strategy, args.data_source, result_dir)
        
    else:
        # 隨機數據源模式
        logger.info("使用隨機數據源模式")
        
        # 創建 study
        optuna_sqlite = result_dir / f"optuna_{args.strategy}_random_{TIMESTAMP}.sqlite3"
        study = optuna.create_study(
            study_name=f"00631L_optuna_{args.strategy}_random",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
            storage=f"sqlite:///{optuna_sqlite}"
        )
        
        # 運行試驗
        study.optimize(
            lambda trial: objective(trial, 'random', None, args.strategy), 
            n_trials=args.n_trials, 
            n_jobs=1, 
            show_progress_bar=True
        )
        
        # 輸出結果
        output_results(study, args.strategy, "random", result_dir)

def output_results(study, strategy: str, data_source: str, result_dir: Path):
    """輸出試驗結果到檔案"""
    safe_ds = sanitize_filename(data_source)
    
    # 從 study.user_attrs 獲取試驗結果
    trial_results = study.user_attrs.get("trial_results", [])
    
    if not trial_results:
        logger.warning(f"trial_results 為空，數據源: {data_source}，跳過結果輸出")
        return
    
    # 轉換為 DataFrame
    df_results = pd.json_normalize(trial_results, sep='_')
    logger.info(f"df_results 形狀: {df_results.shape}, 欄位: {list(df_results.columns)}")
    
    if df_results.empty:
        logger.warning(f"df_results 為空，數據源: {data_source}，可能所有試驗被剔除")
        return
    
    # 檢查必要欄位
    required_cols = {"trial_number", "score", "strategy", "data_source"}
    missing_cols = required_cols - set(df_results.columns)
    if missing_cols:
        logger.error(f"df_results 缺少必要欄位: {missing_cols}，數據源: {data_source}")
        return
    
    df_results = df_results.sort_values("score", ascending=False)
    
    # 保存試驗結果 CSV
    result_csv_file = result_dir / f"optuna_results_{strategy}_{safe_ds}_{TIMESTAMP}.csv"
    df_results.to_csv(result_csv_file, index=False, encoding="utf-8-sig")
    logger.info(f"試驗結果已保存至 {result_csv_file}")
    
    # 保存事件記錄 CSV
    event_csv_file = result_dir / f"optuna_events_{strategy}_{safe_ds}_{TIMESTAMP}.csv"
    df_events = pd.DataFrame(events_log)
    df_events.to_csv(event_csv_file, index=False, encoding='utf-8-sig', na_rep='0.0')
    logger.info(f"事件紀錄已保存至 {event_csv_file}")
    
    # 保存最佳參數 JSON
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
    
    result_file = result_dir / f"optuna_best_params_{strategy}_{safe_ds}_{TIMESTAMP}.json"
    pd.Series(results).to_json(result_file, indent=2)
    logger.info(f"最佳參數已保存至 {result_file}")
    
    # 顯示前 5 筆試驗記錄
    logger.info("前 5 筆試驗記錄:")
    for i, record in enumerate(trial_results[:5]):
        trial_num = record.get('trial_number', 'unknown')
        score = record.get('score', -np.inf)
        strategy_name = record.get('strategy', 'unknown')
        data_source_name = record.get('data_source', 'unknown')
        total_return = record.get('total_return', 0.0)
        num_trades = record.get('num_trades', 0)
        logger.info(f"試驗 {trial_num}: 策略={strategy_name}, 數據源={data_source_name}, 分數={score:.3f}, 總報酬={total_return:.3f}, 交易次數={num_trades}")
        log_to_results(f"前 5 名試驗 {i+1}", f"試驗 {trial_num}: 策略={strategy_name}, 數據源={data_source_name}, 分數={score:.3f}, 總報酬={total_return:.3f}, 交易次數={num_trades}")

if __name__ == "__main__":
    main()
