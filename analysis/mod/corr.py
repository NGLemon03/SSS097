# ** coding: utf-8 **

from typing import Tuple, List, Dict  
import logging
from logging_config import setup_logging
import optuna
import numpy as np
import re
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
from scipy.stats import pearsonr
import seaborn as sns
# 版本與紀錄
version = 'optuna_10'
setup_logging()
logger = logging.getLogger('optuna_10')
# 資料過濾
setlimit = False
setminsharpe = 0.45
setmaxsharpe = 0.75
setminmdd = -0.2
setmaxmdd = -0.4
# 設定 matplotlib 字體
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import config as cfg
from analysis import data_loader
import SSSv096 as SSS

def compute_correlations(results: list, param_keys: list, metric_keys: List[str]) -> pd.DataFrame:
    """
    計算參數與指標之間的皮爾森相關係數.
    Args:
        results: 試驗結果列表, 每筆為字典, 包含參數與指標.
        param_keys: 要分析的參數名稱列表.
        metric_keys: 要分析的指標名稱列表.
    Returns:
        pd.DataFrame: 相關係數矩陣, 行=參數, 列=指標.
    """
    correlations = {}
    for param in param_keys:
        correlations[param] = {}
        param_values = [pd.to_numeric(r.get(f"param_{param}", np.nan), errors="coerce") for r in results]
        for metric in metric_keys:
            metric_values = [pd.to_numeric(r.get(metric, np.nan), errors="coerce") for r in results]
            valid_pairs = [(p, m) for p, m in zip(param_values, metric_values) if not np.isnan(p) and not np.isnan(m)]
            if len(valid_pairs) < 3:
                logger.warning(f"參數 {param} 與指標 {metric} 的有效樣本數 {len(valid_pairs)} < 3, 相關係數設為 NaN")
                correlations[param][metric] = np.nan
            else:
                p_values, m_values = zip(*valid_pairs)
                corr, _ = pearsonr(p_values, m_values)
                correlations[param][metric] = corr
    return pd.DataFrame(correlations).T



def plot_correlation_heatmap(corr_df: pd.DataFrame, strategy_name: str, timestamp: str) -> Path:
    """
    繪製相關係數熱圖.
    Args:
        corr_df: 相關係數 DataFrame.
        strategy_name: 策略名稱.
        timestamp: 時間戳記.
    Returns:
        Path: 熱圖檔案路徑, 若未生成則返回 None.
    """
    if corr_df.notna().sum().sum() == 0:
        logger.warning(f"策略 {strategy_name} 的相關係數全為 NaN, 跳過熱圖繪製")
        return None
    plot_file = cfg.RESULT_DIR / f"correlation_heatmap_{strategy_name}_{timestamp}.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title(f"Correlation Matrix for {strategy_name}")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"相關係數熱圖已保存至 {plot_file}")
    return plot_file

def analyze_param_correlations(ticker: str, start_date: str, end_date: str, trial_results_all: list) -> dict:
    """
    分析 Optuna 試驗結果中參數與指標的相關係數.
    Args:
        ticker: 股票代號.
        start_date: 數據起始日期.
        end_date: 數據結束日期.
        trial_results_all: 所有試驗結果列表.
    Returns:
        dict: 每個策略的相關係數 DataFrame.
    """
    correlation_dfs = {}
    for strategy in ["single", "dual", "RMA", "ssma_turn"]:
        strategy_results = [r for r in trial_results_all if r.get("strategy") == strategy]
        if not strategy_results:
            logger.info(f"策略 {strategy} 無 Optuna 試驗結果, 跳過相關係數計算")
            continue
        param_keys = [k for k in PARAM_SPACE[strategy].keys()]
        metric_keys = ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor", "num_trades", "stress_mdd", "excess_return_stress"]
        corr_df = compute_correlations(strategy_results, param_keys, metric_keys)
        correlation_dfs[strategy] = corr_df
        logger.info(f"策略 {strategy} 相關係數(行=參數, 列=指標): \n{corr_df}")
        corr_file = cfg.RESULT_DIR / f"param_correlations_{strategy}_{TIMESTAMP}.csv"
        corr_file.parent.mkdir(parents=True, exist_ok=True)
        corr_df.to_csv(corr_file, encoding="utf-8-sig")
        logger.info(f"相關係數已保存至 {corr_file}")
        plot_file = plot_correlation_heatmap(corr_df, strategy, TIMESTAMP)
        if plot_file:
            log_to_results("相關係數熱圖", f"策略 {strategy} 熱圖生成", correlation_heatmap=str(plot_file))
    return correlation_dfs

def merge_correlations_simplified(df_results: pd.DataFrame, correlation_dfs: dict, strategy: str, timestamp: str) -> pd.DataFrame:
    """
    簡化版：將相關係數合併至試驗結果 DataFrame。
    Args:
        df_results: 試驗結果 DataFrame
        correlation_dfs: 相關係數 DataFrame 字典
        strategy: 策略名稱
        timestamp: 時間戳記
    Returns:
        pd.DataFrame: 合併後的 DataFrame
    """
    if strategy not in correlation_dfs:
        logger.warning(f"策略 {strategy} 無相關係數數據，跳過合併")
        return df_results
    
    # 讀取相關係數
    corr_df = correlation_dfs[strategy]
    
    # 轉為長格式
    corr_long = corr_df.reset_index().melt(id_vars='index', var_name='metric', value_name='corr')
    corr_long['index'] = 'param_' + corr_long['index']
    
    # 複製原始 DataFrame
    merged_results = df_results.copy()
    
    # 逐一新增相關係數欄位
    for _, row in corr_long.iterrows():
        param = row['index']
        metric = row['metric']
        corr_value = row['corr']
        if pd.isna(corr_value):
            logger.warning(f"參數 {param} 與指標 {metric} 的相關係數為 NaN，跳過")
            continue
        col_name = f"corr_{param[6:]}_{metric}"
        merged_results[col_name] = pd.Series([corr_value] + [np.nan] * (len(merged_results) - 1), index=merged_results.index)
    
    # 儲存結果
    out_file = cfg.RESULT_DIR / f"optuna_results_with_corr_{strategy}_{timestamp}.csv"
    merged_results.to_csv(out_file, index=False, encoding='utf-8-sig')
    logger.info(f"已將相關係數合併至 {out_file}")
    log_to_results("相關係數合併", f"已將相關係數合併至 {out_file}", merged_file=str(out_file))
    
    return merged_results