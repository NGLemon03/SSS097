import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import ast
import shutil
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from analysis.data_loader import load_data
from SSSv096 import backtest_unified, param_presets, compute_single, compute_RMA, compute_ssma_turn_combined, compute_dual

# 新增 Ensemble 策略支援
try:
    from ensemble_wrapper import EnsembleStrategyWrapper
    ENSEMBLE_AVAILABLE = True
except ImportError:
    print("警告：無法導入 ensemble_wrapper，Ensemble 策略將不可用")
    ENSEMBLE_AVAILABLE = False
from analysis.config import RESULT_DIR, WF_PERIODS, STRESS_PERIODS, CACHE_DIR
from analysis.metrics import calculate_max_drawdown
from analysis.logging_config import setup_logging
import logging


# 初始化快取目錄
# 移除自動清理快取目錄，避免與其他程式衝突
# shutil.rmtree(CACHE_DIR, ignore_errors=True)
(CACHE_DIR / "price").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "smaa").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "factor").mkdir(parents=True, exist_ok=True)

# 初始化日誌
setup_logging()
logger = logging.getLogger("OSv3")

# === 說明氣泡函數 ===
def get_column_tooltips():
    """
    返回各欄位的說明氣泡文字
    """
    tooltips = {
        # 基本績效指標
        'total_return': '總報酬率：整個回測期間的累積報酬率',
        'annual_return': '年化報酬率：將總報酬率轉換為年化標準',
        'sharpe_ratio': '夏普值：超額報酬率與波動率的比值，衡量風險調整後報酬',
        'max_drawdown': '最大回撤：權益曲線從峰值到谷值的最大跌幅',
        'calmar_ratio': '卡瑪值：年化報酬率與最大回撤的比值',
        'num_trades': '交易次數：整個回測期間的總交易次數',
        'win_rate': '勝率：獲利交易次數佔總交易次數的比例',
        
        # 過擬合檢測指標
        'overfitting_score': '過擬合分數：0-100，分數越高表示過擬合風險越大',
        'parameter_sensitivity': '參數敏感性：策略對參數變化的敏感程度，越高越容易過擬合',
        'consistency_score': '一致性分數：樣本內外表現的一致性，越高越穩定',
        'stability_score': '穩定性分數：策略在不同時間段的表現穩定性',
        'overfitting_risk': '過擬合風險：綜合過擬合風險評分，0-100',
        'pbo_score': 'PBO分數：Probability of Backtest Overfitting，回測過擬合機率',
        
        # 樣本內外比較
        'train_sharpe': '樣本內夏普值：訓練期間的夏普值',
        'test_sharpe': '樣本外夏普值：測試期間的夏普值',
        'sharpe_degradation': '夏普值退化：樣本內外夏普值的差異',
        'train_return': '樣本內報酬率：訓練期間的年化報酬率',
        'test_return': '樣本外報酬率：測試期間的年化報酬率',
        'return_degradation': '報酬率退化：樣本內外報酬率的差異',
        
        # 穩定性指標
        'mean_return': '平均報酬率：各期間的平均報酬率',
        'std_return': '報酬率標準差：各期間報酬率的變異程度',
        'cv': '變異係數：標準差與平均值的比值，衡量相對變異性',
        'positive_periods_ratio': '正報酬期間比例：產生正報酬的期間佔總期間的比例',
        'rank_stability': '排名穩定性：策略在不同期間的排名一致性',
        
        # 其他指標
        'avg_hold_days': '平均持倉天數：平均每次交易的持倉時間',
        'profit_factor': '獲利因子：總獲利與總虧損的比值',
        'cpcv_oos_mean': 'CPCV樣本外平均：交叉驗證樣本外的平均報酬率',
        'cpcv_oos_min': 'CPCV樣本外最小值：交叉驗證樣本外的最低報酬率',
        'sharpe_var': '夏普值變異數：滾動夏普值的變異程度',
        
        # 風險分布
        'low_risk': '低風險比例：過擬合風險≤30的策略比例',
        'medium_risk': '中風險比例：過擬合風險31-60的策略比例',
        'high_risk': '高風險比例：過擬合風險>60的策略比例',
        'risk_return_correlation': '風險報酬相關性：過擬合風險與總報酬率的相關係數'
    }
    return tooltips

def create_tooltip_text(column_name):
    """
    為指定欄位創建說明氣泡文字
    """
    tooltips = get_column_tooltips()
    return tooltips.get(column_name, f"欄位：{column_name}")

def display_dataframe_with_tooltips(df, title="", key=""):
    """
    顯示帶有說明氣泡的DataFrame
    """
    if df.empty:
        st.warning("沒有數據可顯示")
        return
    
    # 顯示標題
    if title:
        st.subheader(title)
    
    # 創建說明文字
    tooltip_text = ""
    for col in df.columns:
        if col in get_column_tooltips():
            tooltip_text += f"**{col}**: {get_column_tooltips()[col]}\n\n"
    
    # 顯示說明氣泡
    if tooltip_text:
        with st.expander("📖 欄位說明", expanded=False):
            st.markdown(tooltip_text)
    
    # 顯示DataFrame
    st.dataframe(df, use_container_width=True, key=key)

# === 參數相關性分析與多樣性過濾函數（移到檔案前面） ===
def compute_param_correlations(optuna_results_df, strategy, data_source):
    """
    計算指定策略和數據源的參數與績效指標的相關性
    
    Args:
        optuna_results_df: Optuna結果DataFrame
        strategy: 策略名稱
        data_source: 數據源名稱
    
    Returns:
        pd.DataFrame: 參數相關性矩陣
    """
    # 篩選指定策略和數據源的試驗
    mask = (optuna_results_df['strategy'] == strategy) & (optuna_results_df['data_source'] == data_source)
    strategy_trials = optuna_results_df[mask].copy()
    
    if len(strategy_trials) < 10:
        logger.warning(f"策略 {strategy} 數據源 {data_source} 的試驗數量不足 ({len(strategy_trials)})，跳過相關性分析")
        return pd.DataFrame()
    
    # 解析參數
    strategy_trials['parameters'] = strategy_trials['parameters'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # 提取參數到獨立欄位
    param_keys = set()
    for params in strategy_trials['parameters']:
        if isinstance(params, dict):
            param_keys.update(params.keys())
    
    for param in param_keys:
        strategy_trials[f'param_{param}'] = strategy_trials['parameters'].apply(
            lambda x: x.get(param, np.nan) if isinstance(x, dict) else np.nan
        )
    
    # 計算相關性
    correlations = {}
    metric_keys = ["total_return", "sharpe_ratio"]
    
    for param in param_keys:
        param_col = f'param_{param}'
        if param_col not in strategy_trials.columns:
            continue
            
        correlations[param] = {}
        param_values = pd.to_numeric(strategy_trials[param_col], errors='coerce')
        
        for metric in metric_keys:
            if metric not in strategy_trials.columns:
                continue
                
            metric_values = pd.to_numeric(strategy_trials[metric], errors='coerce')
            
            # 確保是pandas Series並移除NaN值
            if not isinstance(param_values, pd.Series):
                param_values = pd.Series(param_values)
            if not isinstance(metric_values, pd.Series):
                metric_values = pd.Series(metric_values)
            
            # 移除NaN值
            valid_mask = param_values.notna() & metric_values.notna()
            if valid_mask.sum() < 3:
                correlations[param][metric] = np.nan
                continue
            
            valid_params = param_values[valid_mask]
            valid_metrics = metric_values[valid_mask]
            
            try:
                corr, p_value = pearsonr(valid_params, valid_metrics)
                correlations[param][metric] = corr
            except:
                correlations[param][metric] = np.nan
    
    return pd.DataFrame(correlations).T

def pick_topN_by_diversity(trials, metric_keys, top_n=5):
    """
    基於 fine_grained_cluster 的多樣性 top N 試驗選擇，支援欄位自動適應
    
    Args:
        trials: 試驗列表，每個試驗包含 score 和 fine_grained_cluster
        metric_keys: 用於分組的指標鍵（支援 num_trades、excess_return_stress、avg_hold_days）
        top_n: 最終選取的試驗數量
    
    Returns:
        List: 篩選後的試驗列表
    """
    logger.info(f"開始多樣性篩選: {len(trials)} 個試驗, 目標選取 {top_n} 個")
    
    # 轉換為 DataFrame
    df = pd.DataFrame(trials)
    
    # 檢查 DataFrame 是否為空
    if df.empty:
        logger.error("試驗數據為空，無法進行篩選")
        return []
    
    # 檢查必要欄位
    if 'score' not in df.columns:
        logger.error("缺少必要欄位 'score'，無法進行篩選")
        return []
    
    # 優先使用 fine_grained_cluster 進行分組
    if 'fine_grained_cluster' in df.columns:
        logger.info("使用 fine_grained_cluster 進行分組")
        
        # 檢查 fine_grained_cluster 欄位是否有有效數據
        valid_clusters = df['fine_grained_cluster'].notna() & (df['fine_grained_cluster'] >= 0)
        if valid_clusters.sum() > 0:
            # 使用 fine_grained_cluster 進行分組
            df['group'] = df['fine_grained_cluster'].astype(str)
            logger.info(f"fine_grained_cluster 分組: 共 {df['fine_grained_cluster'].nunique()} 個群組")
            
            # 統計各群組的試驗數量
            cluster_counts = df['fine_grained_cluster'].value_counts().sort_index()
            logger.info("各群組試驗數量:")
            for cluster_id, count in cluster_counts.items():
                logger.info(f"  群組 {cluster_id}: {count} 個試驗")
        else:
            logger.warning("fine_grained_cluster 欄位無有效數據，使用備用分組方法")
            df['group'] = 'default'
    else:
        logger.warning("缺少 fine_grained_cluster 欄位，使用備用分組方法")
        
        # 定義關鍵指標（按優先級排序）
        key_metrics = ['num_trades', 'excess_return_stress', 'avg_hold_days']
        logger.info(f"期望的關鍵指標: {key_metrics}")
        
        # 檢查哪些指標實際存在
        available_metrics = []
        for metric in key_metrics:
            if metric in df.columns:
                # 檢查欄位是否有有效數據
                valid_data = df[metric].notna() & (df[metric] != np.inf) & (df[metric] != -np.inf)
                if valid_data.sum() > 0:
                    available_metrics.append(metric)
                    logger.info(f"✓ 指標 {metric} 可用，有效數據: {valid_data.sum()}/{len(df)}")
                else:
                    logger.warning(f"✗ 指標 {metric} 存在但無有效數據")
            else:
                logger.warning(f"✗ 指標 {metric} 不存在")
        
        # 如果沒有可用的關鍵指標，使用基本篩選
        if not available_metrics:
            logger.warning("沒有可用的關鍵指標，使用基本分數排序篩選")
            df_sorted = df.sort_values(by='score', ascending=False)
            chosen_trials = df_sorted.head(top_n).to_dict('records')
            logger.info(f"基本篩選完成，選取 {len(chosen_trials)} 個試驗")
            return chosen_trials
        
        logger.info(f"使用可用指標: {available_metrics}")
        
        # 針對不同指標特性的分組處理
        for metric in available_metrics:
            if metric not in df.columns:
                continue
                
            # 處理 NaN 值
            df[metric] = df[metric].fillna(df[metric].median())
                
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
        group_cols = [f'grouped_{metric}' for metric in available_metrics]
        df['group'] = df[group_cols].astype(str).agg('_'.join, axis=1)
    
    # 統計分組情況
    group_counts = df['group'].value_counts()
    logger.info(f"分組統計: 共 {len(group_counts)} 個不同組別")
    
    # 顯示前幾個組別的詳細信息
    for i, (group, count) in enumerate(group_counts.head(5).items()):
        group_str = str(group)[:50] if group else "空組別"
        logger.info(f"組別 {i+1}: {count} 個試驗, 組別標識: {group_str}...")
    
    # 按分數排序並選擇每個分組中分數最高的試驗
    df_sorted = df.sort_values(by='score', ascending=False)
    chosen_trials = []
    seen_groups = set()
    
    logger.info("開始選取試驗...")
    
    for idx, row in df_sorted.iterrows():
        group = row['group']
        trial_num = row.get('trial_number', f'trial_{idx}')
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
        # 如果分組篩選不足，補充高分試驗
        remaining_needed = top_n - len(chosen_trials)
        remaining_trials = df_sorted[~df_sorted.index.isin([t.get('index', i) for i, t in enumerate(chosen_trials)])]
        if len(remaining_trials) > 0:
            # 使用更安全的方式轉換 DataFrame 到字典列表
            additional_trials = []
            for _, row in remaining_trials.head(remaining_needed).iterrows():
                additional_trials.append(row.to_dict())
            chosen_trials.extend(additional_trials)
            logger.info(f"補充 {len(additional_trials)} 個高分試驗")
    
    # 顯示最終選取的試驗信息
    logger.info("最終選取的試驗:")
    for i, trial in enumerate(chosen_trials):
        trial_num = trial.get('trial_number', f'trial_{i}')
        score = trial.get('score', -np.inf)
        logger.info(f"  {i+1}. 試驗 {trial_num}: score={score:.3f}")
        # 顯示關鍵指標（安全處理）
        if 'fine_grained_cluster' in df.columns:
            cluster_id = trial.get('fine_grained_cluster', 'N/A')
            logger.info(f"     群組: {cluster_id}")
        else:
            # 只有在使用備用分組方法時才顯示指標
            try:
                key_metrics_values = {k: trial.get(k, 'N/A') for k in available_metrics}
                logger.info(f"     關鍵指標: {key_metrics_values}")
            except NameError:
                logger.info("     使用 fine_grained_cluster 分組")
    
    return chosen_trials

# Streamlit UI 配置
st.set_page_config(layout="wide", page_title="00631L 策略回測與走查分析")

# 自訂 multiselect 標籤顏色為水藍色
st.markdown(
    '''
    <style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #00BFFF !important;
        color: #222 !important;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

st.title("00631L 策略分析與回測")

# 動態生成走查區間
def generate_walk_forward_periods(data_index, n_splits, min_days=30):
    """根據數據日期範圍生成平分的走查區間"""
    if data_index.empty:
        st.error("數據索引為空，無法生成走查區間")
        logger.error("數據索引為空，無法生成走查區間")
        return []
    start_date = data_index.min()
    end_date = data_index.max()
    total_days = (end_date - start_date).days
    if total_days < min_days:
        st.error(f"數據範圍過短（{total_days} 天），無法生成走查區間")
        logger.error(f"數據範圍過短（{total_days} 天），無法生成走查區間")
        return []
    if total_days < min_days * n_splits:
        n_splits = max(1, total_days // min_days)
        logger.warning(f"數據範圍過短，調整分段數為 {n_splits}")
    days_per_split = total_days // n_splits
    periods = []
    current_start = start_date
    for i in range(n_splits):
        if i == n_splits - 1:
            current_end = end_date
        else:
            current_end = current_start + pd.Timedelta(days=days_per_split - 1)
            end_candidates = data_index[data_index <= current_end]
            if end_candidates.empty:
                break
            current_end = end_candidates[-1]
        if (current_end - current_start).days >= min_days:
            periods.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)
        start_candidates = data_index[data_index >= current_start]
        if start_candidates.empty:
            break
        current_start = start_candidates[0]
    return periods

# 計算 PR 值
def calculate_pr_values(series, is_mdd, is_initial_period, hedge_mask):
    """
    報酬率量化分數（Relative Strength Score）或 MDD 量化分數（Risk-Adjusted MDD Score）
    :param series: 要計算分數的系列（報酬率或 MDD）
    :param is_mdd: 是否為 MDD（True）或報酬率（False）
    :param is_initial_period: 是否為初始時段
    :param hedge_mask: 避險掩碼，True 表示該策略在該時段避險
    :return: 分數系列
    """
    score = pd.Series(index=series.index, dtype=float)
    if is_initial_period:
        # 初始時段：排除避險策略
        score[hedge_mask] = np.nan
        valid = ~hedge_mask
        valid_series = series[valid]
        if is_mdd:
            # MDD 修正：MDD越小（越接近0）分數越高
            # MDD通常是負值，我們希望它越接近0越好
            benchmark = abs(valid_series.mean())
            if benchmark == 0 or np.isnan(benchmark):
                score[valid] = 0
            else:
                # 修正公式：MDD越小分數越高
                # 使用 (1 + valid_series/benchmark) 而不是 (1 - valid_series/benchmark)
                # 因為valid_series是負值，所以加號會讓MDD越小分數越高
                score[valid] = 100 * (1 + valid_series / benchmark)
        else:
            # 報酬率：報酬率越高分數越高
            benchmark = valid_series.mean()
            std = valid_series.std()
            if std == 0 or np.isnan(std):
                score[valid] = 0
            else:
                score[valid] = 100 * (valid_series - benchmark) / std
    else:
        # 非初始時段：避險策略給 100 分
        score[hedge_mask] = 100
        valid = ~hedge_mask
        valid_series = series[valid]
        if is_mdd:
            # MDD 修正：MDD越小（越接近0）分數越高
            benchmark = abs(valid_series.mean())
            if benchmark == 0 or np.isnan(benchmark):
                score[valid] = 0
            else:
                # 修正公式：MDD越小分數越高
                score[valid] = 100 * (1 + valid_series / benchmark)
        else:
            # 報酬率：報酬率越高分數越高
            benchmark = valid_series.mean()
            std = valid_series.std()
            if std == 0 or np.isnan(std):
                score[valid] = 0
            else:
                score[valid] = 100 * (valid_series - benchmark) / std
    return score

# 在文件開頭添加新的過擬合檢測函數
def calculate_overfitting_metrics(train_returns, test_returns, strategy_name):
    """
    計算過擬合指標 - 增強版
    """
    if len(train_returns) == 0 or len(test_returns) == 0:
        return {}
    
    # 計算樣本內外表現差異
    train_sharpe = train_returns.mean() / train_returns.std() if train_returns.std() > 0 else 0
    test_sharpe = test_returns.mean() / test_returns.std() if test_returns.std() > 0 else 0
    
    # 樣本內外夏普值差異
    sharpe_degradation = train_sharpe - test_sharpe
    
    # 樣本內外報酬率差異
    return_degradation = train_returns.mean() - test_returns.mean()
    
    # 穩定性指標（變異係數）
    train_cv = train_returns.std() / abs(train_returns.mean()) if train_returns.mean() != 0 else float('inf')
    test_cv = test_returns.std() / abs(test_returns.mean()) if test_returns.mean() != 0 else float('inf')
    
    # 過擬合分數（0-100，越高越過擬合）
    # 修正計算方式：考慮夏普值和報酬率的相對重要性
    sharpe_weight = 0.6
    return_weight = 0.4
    overfitting_score = min(100, max(0, 
        abs(sharpe_degradation) * 50 * sharpe_weight + 
        abs(return_degradation) * 200 * return_weight
    ))
    
    return {
        'train_sharpe': train_sharpe,
        'test_sharpe': test_sharpe,
        'train_return': train_returns.mean(),
        'test_return': test_returns.mean(),
        'sharpe_degradation': sharpe_degradation,
        'return_degradation': return_degradation,
        'overfitting_score': overfitting_score
    }

def calculate_enhanced_overfitting_analysis(optuna_results_df, strategy_name, data_source_name):
    """
    計算增強的過擬合分析 - 基於Optuna結果中的新指標
    """
    # 篩選指定策略和數據源的試驗
    mask = (optuna_results_df['strategy'] == strategy_name) & (optuna_results_df['data_source'] == data_source_name)
    strategy_trials = optuna_results_df[mask].copy()
    
    if len(strategy_trials) < 10:
        return {}
    
    # 檢查是否有新的過擬合指標
    new_metrics = ['parameter_sensitivity', 'consistency_score', 'stability_score', 'overfitting_risk']
    available_metrics = [metric for metric in new_metrics if metric in strategy_trials.columns]
    
    if not available_metrics:
        return {}
    
    # 計算統計摘要
    analysis_results = {}
    
    for metric in available_metrics:
        values = pd.to_numeric(strategy_trials[metric], errors='coerce')
        # 使用pandas方法處理NaN值
        valid_values = values.dropna()
        
        if len(valid_values) > 0:
            analysis_results[f'{metric}_mean'] = np.mean(valid_values)
            analysis_results[f'{metric}_std'] = np.std(valid_values)
            analysis_results[f'{metric}_min'] = np.min(valid_values)
            analysis_results[f'{metric}_max'] = np.max(valid_values)
            analysis_results[f'{metric}_median'] = np.median(valid_values)
            
            # 計算風險等級
            if metric == 'overfitting_risk':
                low_risk = np.sum(valid_values <= 30) / len(valid_values)
                medium_risk = np.sum((valid_values > 30) & (valid_values <= 60)) / len(valid_values)
                high_risk = np.sum(valid_values > 60) / len(valid_values)
                analysis_results[f'{metric}_risk_distribution'] = {
                    'low_risk': low_risk,
                    'medium_risk': medium_risk,
                    'high_risk': high_risk
                }
    
    # 計算相關性分析
    if 'overfitting_risk' in available_metrics and 'total_return' in strategy_trials.columns:
        risk_values = pd.to_numeric(strategy_trials['overfitting_risk'], errors='coerce')
        return_values = pd.to_numeric(strategy_trials['total_return'], errors='coerce')
        
        valid_mask = risk_values.notna() & return_values.notna()
        if valid_mask.sum() > 5:
            correlation = risk_values[valid_mask].corr(return_values[valid_mask])
            analysis_results['risk_return_correlation'] = correlation
    
    return analysis_results

def calculate_strategy_stability(period_returns_dict):
    """
    計算策略穩定性指標
    """
    if not period_returns_dict:
        return {}
    
    # 確保所有策略的期間報酬率都有相同的索引
    all_periods = set()
    for returns in period_returns_dict.values():
        all_periods.update(returns.index)
    
    # 創建統一的DataFrame
    returns_df = pd.DataFrame(index=sorted(all_periods))
    for name, returns in period_returns_dict.items():
        returns_df[name] = returns
    
    stability_metrics = {}
    
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        if len(returns) < 2:
            continue
            
        # 計算各期間表現的一致性（注意：period_returns已經是百分比）
        mean_return = returns.mean() / 100  # 轉換為小數
        std_return = returns.std() / 100    # 轉換為小數
        
        # 修正變異係數計算，避免無限值
        if abs(mean_return) > 1e-10:  # 避免除以接近零的值
            cv = std_return / abs(mean_return)
        else:
            cv = float('inf') if std_return > 0 else 0.0
        
        # 計算正報酬期間比例
        positive_periods = (returns > 0).sum() / len(returns)
        
        # 計算表現排名穩定性（如果有多個策略）
        if len(returns_df.columns) > 1:
            # 計算該策略在不同期間的排名穩定性
            # 對每個期間計算該策略的排名，然後計算排名的一致性
            strategy_rankings = []
            
            # 對每個期間計算該策略的排名
            for period in returns_df.index:
                period_returns = returns_df.loc[period].dropna()
                if len(period_returns) > 1 and col in period_returns.index:
                    # 計算排名（1為最好，n為最差）
                    rankings = period_returns.rank(ascending=False)
                    # 獲取該策略的排名
                    strategy_rank = rankings[col]
                    strategy_rankings.append(strategy_rank)
            
            if len(strategy_rankings) > 1:
                # 計算該策略排名的變異係數（標準差/平均值）
                strategy_rankings_series = pd.Series(strategy_rankings)
                rank_mean = strategy_rankings_series.mean()
                rank_std = strategy_rankings_series.std()
                
                if rank_mean > 0:
                    # 排名穩定性 = 1 - (排名標準差 / 排名平均值)
                    # 這樣排名越穩定，值越接近1
                    rank_stability = max(0, 1 - (rank_std / rank_mean))
                else:
                    rank_stability = 1.0
                
                logger.info(f"策略 {col} 排名穩定性計算: {len(strategy_rankings)} 個期間, 平均排名: {rank_mean:.2f}, 排名標準差: {rank_std:.2f}, 穩定性: {rank_stability:.3f}")
            else:
                rank_stability = 1.0
                logger.info(f"策略 {col} 排名穩定性: 只有一個期間，設為 1.0")
        else:
            rank_stability = 1.0
            logger.info(f"策略 {col} 排名穩定性: 只有一個策略，設為 1.0")
            
        stability_metrics[col] = {
            'mean_return': mean_return,
            'std_return': std_return,
            'cv': cv,
            'positive_periods_ratio': positive_periods,
            'rank_stability': rank_stability
        }
    
    return stability_metrics

def calculate_risk_adjusted_metrics(equity_curve, strategy_name):
    """
    計算風險調整後的報酬率指標
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {}
    
    # 計算日報酬率
    daily_returns = equity_curve.pct_change().dropna()
    
    # 基本指標
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    annual_return = total_return * (252 / len(daily_returns))
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 最大回撤
    max_drawdown = calculate_max_drawdown(equity_curve)
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # 索提諾比率（只考慮下行風險）
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
    
    # 卡瑪值（年化報酬率/最大回撤）
    kama_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # 一致性評分（正報酬月份比例）
    monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
    positive_months = (monthly_returns > 0).sum() / len(monthly_returns)
    
    # 綜合評分（結合報酬率和風險）
    composite_score = (sharpe_ratio * 0.3 + sortino_ratio * 0.3 + positive_months * 0.2 + (1 - max_drawdown) * 0.2)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'kama_ratio': kama_ratio,
        'max_drawdown': max_drawdown,
        'positive_months_ratio': positive_months,
        'composite_score': composite_score
    }


def run_ensemble_strategy(params, ticker="00631L.TW"):
    """運行 Ensemble 策略"""
    if not ENSEMBLE_AVAILABLE:
        st.error("Ensemble 策略不可用，請檢查 ensemble_wrapper 是否正確安裝")
        # 返回空结果避免UI误报
        empty_equity = pd.Series(1.0, index=pd.date_range('2020-01-01', periods=100, freq='D'))
        empty_trades = pd.DataFrame(columns=['date', 'action', 'weight', 'price'])
        empty_stats = {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0}
        return empty_equity, empty_trades, empty_stats
    
    try:
        # 創建 Ensemble 策略包裝器
        wrapper = EnsembleStrategyWrapper()
        
        # 運行策略
        equity_curve, trades, stats, method_name = wrapper.ensemble_strategy(
            method=params['method'],
            params=params,
            ticker=ticker
        )
        
        # 計算額外的指標
        additional_metrics = calculate_risk_adjusted_metrics(equity_curve, method_name)
        
        # 合併指標
        combined_stats = {**stats, **additional_metrics}
        
        return equity_curve, trades, combined_stats
        
    except Exception as e:
        st.error(f"運行 Ensemble 策略時發生錯誤: {str(e)}")
        logger.error(f"Ensemble 策略錯誤: {e}")
        # 返回空结果避免UI误报
        empty_equity = pd.Series(1.0, index=pd.date_range('2020-01-01', periods=100, freq='D'))
        empty_trades = pd.DataFrame(columns=['date', 'action', 'weight', 'price'])
        empty_stats = {'total_return': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'calmar_ratio': 0.0, 'num_trades': 0}
        return empty_equity, empty_trades, empty_stats


def get_ensemble_strategy_info():
    """獲取 Ensemble 策略信息"""
    if not ENSEMBLE_AVAILABLE:
        return None
    
    try:
        wrapper = EnsembleStrategyWrapper()
        return wrapper.get_strategy_info()
    except Exception as e:
        logger.error(f"獲取 Ensemble 策略信息時發生錯誤: {e}")
        return None

# 解析文件名以提取策略和數據源信息
def parse_optuna_filename(filename):
    """解析optuna結果文件名，提取策略和數據源信息"""
    name = Path(filename).stem  # 移除.csv後綴
    
    # 檢查是否是fine-grained processed文件
    if name.endswith('_fine_grained_processed'):
        # 新格式：{strategy}_{data_source}_fine_grained_processed
        name = name.replace('_fine_grained_processed', '')
        
        # 查找策略名稱
        strategy = None
        if name.startswith('ssma_turn_'):
            strategy = 'ssma_turn'
            name = name.replace('ssma_turn_', '')
        elif name.startswith('single_'):
            strategy = 'single'
            name = name.replace('single_', '')
        elif name.startswith('dual_'):
            strategy = 'dual'
            name = name.replace('dual_', '')
        elif name.startswith('RMA_'):
            strategy = 'RMA'
            name = name.replace('RMA_', '')
        
        if not strategy:
            return None
        
        # 查找數據源 - 支援新的格式
        data_source = None
        
        # 新格式：Factor_TWII__2412.TW 或 Factor_TWII__2414.TW
        if name.startswith('Factor_TWII__2412.TW'):
            data_source = 'Factor (^TWII / 2412.TW)'
        elif name.startswith('Factor_TWII__2414.TW'):
            data_source = 'Factor (^TWII / 2414.TW)'
        elif name == 'Self':
            data_source = 'Self'
        
        if not data_source:
            return None
        
        return {
            'strategy': strategy,
            'data_source': data_source,
            'timestamp': 'fine_grained_processed',  # 標記為新格式
            'filename': filename
        }
    
    # 檢查是否是optuna_results文件（舊格式）
    if not name.startswith('optuna_results_'):
        return None
    
    # 移除前綴
    name = name.replace('optuna_results_', '')
    
    # 查找策略名稱
    strategy = None
    if name.startswith('ssma_turn_'):
        strategy = 'ssma_turn'
        name = name.replace('ssma_turn_', '')
    elif name.startswith('single_'):
        strategy = 'single'
        name = name.replace('single_', '')
    elif name.startswith('dual_'):
        strategy = 'dual'
        name = name.replace('dual_', '')
    elif name.startswith('RMA_'):
        strategy = 'RMA'
        name = name.replace('RMA_', '')
    
    if not strategy:
        return None
    
    # 查找數據源
    data_source = None
    if name.startswith('Self_'):
        data_source = 'Self'
        name = name.replace('Self_', '')
    elif name.startswith('Factor_TWII__2412.TW_'):
        data_source = 'Factor (^TWII / 2412.TW)'
        name = name.replace('Factor_TWII__2412.TW_', '')
    elif name.startswith('Factor_TWII__2414.TW_'):
        data_source = 'Factor (^TWII / 2414.TW)'
        name = name.replace('Factor_TWII__2414.TW_', '')
    elif name.startswith('Mixed_'):
        data_source = 'Mixed'
        name = name.replace('Mixed_', '')
    
    if not data_source:
        return None
    
    # 提取時間戳
    timestamp = name
    
    return {
        'strategy': strategy,
        'data_source': data_source,
        'timestamp': timestamp,
        'filename': filename
    }

# === 新增：UI 選擇 Optuna 版本和資料夾 ===
st.sidebar.header("🔧 資料來源設定")

# 定義可能的結果資料夾
result_folders = {
    "Fine-grained Processed (新)": Path("../results/fine_grained_processed"),
    "Optuna 13 (預設)": Path("../results"),
    "Optuna 15": Path("../results_op15"),
    "Optuna 13 (備用)": Path("../results_op13"),
}

# 檢查哪些資料夾存在且有檔案
available_folders = {}
for folder_name, folder_path in result_folders.items():
    if folder_path.exists():
        csv_files = list(folder_path.glob("*.csv"))
        if csv_files:
            available_folders[folder_name] = {
                'path': folder_path,
                'file_count': len(csv_files)
            }

if not available_folders:
    st.error("找不到任何包含 optuna 結果的資料夾！")
    st.stop()

# UI 選擇資料夾
folder_options = [f"{name} ({info['file_count']} 個檔案)" for name, info in available_folders.items()]
selected_folder_name = st.sidebar.selectbox(
    "選擇 Optuna 結果來源",
    options=list(available_folders.keys()),
    index=0,
    help="選擇包含 optuna_results_*.csv 檔案的資料夾"
)

# 獲取選中的資料夾路徑
RESULT_DIR = available_folders[selected_folder_name]['path']
st.sidebar.success(f"已選擇: {selected_folder_name}")

# 顯示資料夾資訊
with st.sidebar.expander("📁 資料夾資訊", expanded=False):
    st.info(f"資料夾路徑: {RESULT_DIR}")
    st.info(f"CSV 檔案數量: {available_folders[selected_folder_name]['file_count']}")
    
    # 顯示資料夾中的檔案列表
    csv_files = list(RESULT_DIR.glob("*.csv"))
    st.subheader("檔案列表:")
    for file_path in csv_files[:10]:  # 只顯示前10個
        st.text(f"• {file_path.name}")
    if len(csv_files) > 10:
        st.text(f"... 還有 {len(csv_files) - 10} 個檔案")

# === 載入所有optuna結果文件（加入欄位自動適應） ===
optuna_files = list(RESULT_DIR.glob("*.csv"))

# 解析文件名並顯示
file_info_list = []
for file_path in optuna_files:
    try:
        info = parse_optuna_filename(file_path.name)
        if info:
            file_info_list.append(info)
    except Exception as e:
        logger.warning(f"無法解析文件名 {file_path.name}: {e}")

# 載入所有optuna結果（加入欄位自動適應處理）
all_optuna_results = []
for file_info in file_info_list:
    file_path = RESULT_DIR / file_info['filename']
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            
            # 欄位自動適應：處理參數欄位
            # 檢查是否是 fine-grained processed 文件
            if file_info['timestamp'] == 'fine_grained_processed':
                # 新格式：已經有 fine_grained_cluster 欄位，參數欄位處理
                logger.info(f"檔案 {file_info['filename']} 是 fine-grained processed 格式")
                
                # 檢查是否有 parameters 欄位（JSON 格式）
                if 'parameters' in df.columns:
                    # 如果有 parameters 欄位，嘗試解析 JSON
                    try:
                        import json
                        df['parameters'] = df['parameters'].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip() else {})
                        logger.info(f"檔案 {file_info['filename']} 成功解析 parameters 欄位")
                    except Exception as e:
                        logger.warning(f"無法解析 parameters 欄位: {e}")
                        df['parameters'] = df['parameters'].apply(lambda x: {} if pd.isna(x) else {})
                else:
                    # 如果沒有 parameters 欄位，創建一個空的
                    logger.warning(f"檔案 {file_info['filename']} 沒有找到 parameters 欄位，使用預設值")
                    df['parameters'] = [{}] * len(df)
                
                # 確保有 fine_grained_cluster 欄位
                if 'fine_grained_cluster' not in df.columns:
                    logger.warning(f"檔案 {file_info['filename']} 缺少 fine_grained_cluster 欄位")
                    df['fine_grained_cluster'] = 0
                
            else:
                # 舊格式：Optuna 13/15 格式
                # 檢查是否有 param_* 欄位（Optuna 13/15 格式）
                param_columns = [col for col in df.columns if col.startswith('param_')]
                
                if param_columns:
                    # 如果有 param_* 欄位，從這些欄位構建 parameters 字典
                    logger.info(f"檔案 {file_info['filename']} 使用 param_* 欄位格式，找到 {len(param_columns)} 個參數欄位")
                    
                    # 構建 parameters 字典
                    parameters_list = []
                    for _, row in df.iterrows():
                        params_dict = {}
                        for col in param_columns:
                            param_name = col.replace('param_', '')
                            param_value = row[col]
                            # 嘗試轉換為適當的數據類型
                            if param_value is not None and str(param_value) != 'nan':
                                try:
                                    # 嘗試轉換為數字
                                    if isinstance(param_value, str) and '.' in param_value:
                                        params_dict[param_name] = float(param_value)
                                    else:
                                        params_dict[param_name] = int(param_value)
                                except:
                                    params_dict[param_name] = param_value
                            else:
                                params_dict[param_name] = None
                        parameters_list.append(params_dict)
                    
                    df['parameters'] = parameters_list
                    
                elif 'parameters' in df.columns:
                    # 如果有 parameters 欄位（JSON 格式），嘗試解析
                    try:
                        df['parameters'] = df['parameters'].apply(ast.literal_eval)
                        logger.info(f"檔案 {file_info['filename']} 使用 JSON 格式 parameters 欄位")
                    except:
                        logger.warning(f"無法解析 parameters 欄位，使用預設值: {file_info['filename']}")
                        df['parameters'] = df['parameters'].apply(lambda x: {} if pd.isna(x) else {})
                else:
                    # 如果沒有參數欄位，創建一個空的
                    logger.warning(f"檔案 {file_info['filename']} 沒有找到參數欄位，使用預設值")
                    df['parameters'] = [{}] * len(df)
            
            # 添加文件信息
            df['source_file'] = file_info['filename']
            df['strategy'] = file_info['strategy']
            df['data_source'] = file_info['data_source']
            
            # 欄位自動適應：確保必要欄位存在
            required_fields = ['trial_number', 'score']
            for field in required_fields:
                if field not in df.columns:
                    logger.warning(f"檔案 {file_info['filename']} 缺少欄位 {field}，使用預設值")
                    if field == 'trial_number':
                        df[field] = range(len(df))
                    elif field == 'score':
                        df[field] = -np.inf
            
            all_optuna_results.append(df)
        except Exception as e:
            st.sidebar.error(f"載入失敗 {file_info['filename']}: {str(e)}")

if not all_optuna_results:
    st.error("沒有成功載入任何optuna結果文件")
    st.stop()

# 合併所有結果（確保所有欄位都存在）
all_columns = set()
for df in all_optuna_results:
    all_columns.update(df.columns)

# 為每個 DataFrame 添加缺失的欄位
for df in all_optuna_results:
    for col in all_columns:
        if col not in df.columns:
            if col in ['num_trades', 'sharpe_ratio', 'max_drawdown', 'profit_factor']:
                df[col] = np.nan
            elif col in ['avg_hold_days', 'excess_return_stress']:
                df[col] = np.nan
            else:
                df[col] = None

optuna_results = pd.concat(all_optuna_results, ignore_index=True)

# 按策略和數據源分組，每個組合選取top5
selected_trials = []

# 新增參數相關性分析結果存儲
param_correlations = {}

for strategy in optuna_results['strategy'].unique():
    for data_source in optuna_results['data_source'].unique():
        # 篩選該策略+數據源的試驗
        mask = (optuna_results['strategy'] == strategy) & (optuna_results['data_source'] == data_source)
        strategy_trials_raw = optuna_results[mask]
        
        if len(strategy_trials_raw) == 0:
            continue
            
        logger.info(f"處理策略 {strategy} + 數據源 {data_source}: {len(strategy_trials_raw)} 個試驗")
        
        # 1. 先進行參數相關性分析
        corr_df = compute_param_correlations(optuna_results, strategy, data_source)
        if not corr_df.empty:
            param_correlations[f"{strategy}_{data_source}"] = corr_df
            
            # 根據相關性強度選擇主要參數（絕對值>0.1的參數）
            important_params = []
            for param in corr_df.index:
                max_corr = corr_df.loc[param].abs().max()
                if max_corr > 0.1:  # 相關性閾值
                    important_params.append(param)
            
            # 如果沒有重要參數，使用預設參數
            if not important_params:
                if strategy == 'single':
                    important_params = ['linlen', 'smaalen', 'buy_mult']
                elif strategy == 'dual':
                    important_params = ['linlen', 'smaalen', 'short_win', 'long_win']
                elif strategy == 'RMA':
                    important_params = ['linlen', 'smaalen', 'rma_len', 'dev_len']
                elif strategy == 'ssma_turn':
                    important_params = ['linlen', 'smaalen', 'prom_factor', 'buy_mult']
        else:
            # 使用預設參數
            if strategy == 'single':
                important_params = ['linlen', 'smaalen', 'buy_mult']
            elif strategy == 'dual':
                important_params = ['linlen', 'smaalen', 'short_win', 'long_win']
            elif strategy == 'RMA':
                important_params = ['linlen', 'smaalen', 'rma_len', 'dev_len']
            elif strategy == 'ssma_turn':
                important_params = ['linlen', 'smaalen', 'prom_factor', 'buy_mult']
        
        # 定義性能指標用於多樣性篩選（欄位自動適應）
        metric_keys = ['num_trades']  # 基本指標
        
        # 檢查是否有 Optuna 15 的新欄位
        if 'excess_return_stress' in strategy_trials_raw.columns:
            metric_keys.append('excess_return_stress')
        if 'avg_hold_days' in strategy_trials_raw.columns:
            metric_keys.append('avg_hold_days')
        
        logger.info(f"策略 {strategy} 使用性能指標: {metric_keys}")
        
        # 2. 轉換為字典格式
        trials_dict = []
        for _, trial in strategy_trials_raw.iterrows():
            trial_dict = trial.to_dict()
            trials_dict.append(trial_dict)
        
        # 3. 使用多樣性過濾選擇top5
        logger.info(f"開始為策略 {strategy} + {data_source} 進行多樣性篩選...")
        diverse_trials = pick_topN_by_diversity(
            trials_dict, 
            metric_keys, 
            top_n=5
        )
        
        logger.info(f"策略 {strategy} + {data_source} 篩選完成，選取 {len(diverse_trials)} 個試驗")
        
        # 4. 添加策略和數據源信息
        for trial in diverse_trials:
            # 生成簡短名稱（使用更安全的方式）
            if '2412' in data_source:
                short_name = f"{strategy}_2412_{trial['trial_number']}"
            elif '2414' in data_source:
                short_name = f"{strategy}_2414_{trial['trial_number']}"
            elif 'Self' in data_source:
                short_name = f"{strategy}_Self_{trial['trial_number']}"
            else:
                # 使用更安全的方式處理data_source
                clean_source = data_source.replace('(', '').replace(')', '').replace('/', '_').replace('^', '').strip()
                short_name = f"{strategy}_{clean_source}_{trial['trial_number']}"
            
            trial['short_name'] = short_name
            selected_trials.append(trial)
            logger.info(f"添加試驗: {short_name} (score: {trial['score']:.3f})")

logger.info(f"所有策略篩選完成，總共選取 {len(selected_trials)} 個試驗")

# 建立策略清單
optuna_strategies = [
    {
        'name': trial['short_name'],  # 使用簡短名稱
        'strategy_type': trial['strategy'],
        'params': trial['parameters'],
        'smaa_source': trial['data_source']
    } for trial in selected_trials
]
preset_strategies = [
    {
        'name': key,
        'strategy_type': value['strategy_type'],
        'params': {k: v for k, v in value.items() if k not in ['strategy_type', 'smaa_source']},
        'smaa_source': value['smaa_source']
    } for key, value in param_presets.items()
]
all_strategies = optuna_strategies + preset_strategies

# 側邊欄只做訊息呈現
st.sidebar.header("📊 系統狀態")
st.sidebar.info(f"已載入 {len(selected_trials)} 個試驗")
st.sidebar.info(f"可用策略: {len(all_strategies)} 個")

# 策略統計
strategy_counts = {}
for trial in selected_trials:
    strategy = trial['strategy']
    data_source = trial['data_source']
    key = f"{strategy}_{data_source}"
    strategy_counts[key] = strategy_counts.get(key, 0) + 1

st.sidebar.subheader("策略分布:")
for key, count in strategy_counts.items():
    st.sidebar.text(f"• {key}: {count}")

# 載入信息移到導航欄
with st.sidebar.expander("📊 載入信息", expanded=False):
    st.info(f"找到 {len(optuna_files)} 個optuna results文件")
    
    # 顯示所有文件信息
    st.subheader("可用的文件:")
    for file_info in file_info_list:
        st.text(f"• {file_info['strategy']} - {file_info['data_source']} ({file_info['timestamp']})")
    
    # 顯示欄位資訊
    st.subheader("📋 欄位資訊:")
    if not optuna_results.empty:
        all_columns = list(optuna_results.columns)
        st.text(f"總欄位數: {len(all_columns)}")
        st.text("主要欄位:")
        for col in ['trial_number', 'score', 'strategy', 'data_source', 'parameters', 'num_trades', 'sharpe_ratio', 'max_drawdown']:
            if col in all_columns:
                st.text(f"✓ {col}")
            else:
                st.text(f"✗ {col} (缺失)")
        
        # 檢查 Optuna 15 特有欄位
        optuna15_fields = ['avg_hold_days', 'excess_return_stress', 'pbo_score']
        st.text("Optuna 15 特有欄位:")
        for col in optuna15_fields:
            if col in all_columns:
                st.text(f"✓ {col}")
            else:
                st.text(f"✗ {col} (缺失)")
    
    # 顯示參數相關性分析結果
    if param_correlations:
        st.subheader("📈 參數相關性分析")
        st.write("計算每個參數對 total_return 和 sharpe_ratio 的皮爾森相關係數")
        for key, corr_df in param_correlations.items():
            if not corr_df.empty:
                st.write(f"**{key}**")
                # 格式化顯示相關性矩陣
                formatted_corr = corr_df.style.format("{:.3f}")
                st.dataframe(formatted_corr, use_container_width=True)
                
                # 顯示重要參數
                important_params = []
                for param in corr_df.index:
                    max_corr = corr_df.loc[param].abs().max()
                    if max_corr > 0.1:
                        important_params.append(f"{param} ({max_corr:.3f})")
                
                if important_params:
                    st.write(f"**重要參數 (|相關性| > 0.1):** {', '.join(important_params)}")
                st.divider()

# 策略統計
strategy_counts = {}
for trial in selected_trials:
    strategy = trial['strategy']
    data_source = trial['data_source']
    key = f"{strategy}_{data_source}"
    strategy_counts[key] = strategy_counts.get(key, 0) + 1

st.sidebar.subheader("策略分布:")
for key, count in strategy_counts.items():
    st.sidebar.text(f"• {key}: {count}")

# 載入信息移到導航欄
with st.sidebar.expander("📊 載入信息", expanded=False):
    st.info(f"找到 {len(optuna_files)} 個optuna results文件")
    
    # 顯示所有文件信息
    st.subheader("可用的文件:")
    for file_info in file_info_list:
        st.text(f"• {file_info['strategy']} - {file_info['data_source']} ({file_info['timestamp']})")



# 走查設定移到主頁面
col1, col2 = st.columns(2)
with col1:
    walk_forward_mode = st.selectbox("走查模式", ["動態平分區間", "固定 WF_PERIODS"])
with col2:
    if walk_forward_mode == "動態平分區間":
        n_splits = st.number_input("分段數", min_value=1, max_value=10, value=6, step=1)

# 策略選擇 - 擴大選擇區塊

# 添加 Ensemble 策略即时调参面板
if ENSEMBLE_AVAILABLE:
    st.subheader("🎯 Ensemble 策略即时调参")
    
    # 显示建议参数组合
    with st.expander("📋 小型网格扫描优化后的建议参数", expanded=False):
        st.markdown("""
        **基于小型网格扫描的优化结果：**
        
        **Majority策略建议：**
        - `delta_cap`: 0.10（权重变化上限）
        - `min_cooldown_days`: 5（冷却天数）
        - `min_trade_dw`: 0.02（最小权重变化阈值）
        
        **Proportional策略建议：**
        - `delta_cap`: 0.30（权重变化上限）
        - `min_cooldown_days`: 1（冷却天数）
        - `min_trade_dw`: 0.02（最小权重变化阈值）
        
        **优化效果：**
        - 调整次数：9 → 2（减少78%）
        - Turnover：0.80 → 0.60（减少25%）
        - 报酬率：维持≥98%基线水平
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ensemble_method = st.selectbox(
            "集成方法",
            ["majority", "proportional"],
            help="Majority: K-of-N 多数决, Proportional: 按多头比例分配",
            key="ensemble_method"
        )
        
        # 根据选择的ensemble方法自动调整参数默认值
        if ensemble_method == "majority":
            default_delta_cap = 0.10
            default_cooldown = 5
            default_min_trade_dw = 0.02
        else:  # proportional
            default_delta_cap = 0.30
            default_cooldown = 1
            default_min_trade_dw = 0.02
        
        ensemble_floor = st.slider(
            "底仓比例",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="最小持仓比例，避免完全空仓",
            key="ensemble_floor"
        )
        
        ensemble_ema_span = st.slider(
            "EMA 平滑天数",
            min_value=1,
            max_value=30,
            value=3,
            help="指数移动平均平滑天数，减少权重波动",
            key="ensemble_ema_span"
        )
        
        ensemble_delta_cap = st.slider(
            "权重变化上限",
            min_value=0.05,
            max_value=0.50,
            value=default_delta_cap,  # 根据ensemble方法自动调整
            step=0.01,
            help=f"每日权重变化的最大幅度（{ensemble_method.title()}策略建议{default_delta_cap}）",
            key="ensemble_delta_cap"
        )
    
    with col2:
        if ensemble_method == "majority":
            ensemble_majority_k = st.slider(
                "多数决门槛 (K)",
                min_value=3,
                max_value=11,
                value=6,
                help="K-of-N 中的 K 值，需要至少 K 个策略看多",
                key="ensemble_majority_k"
            )
        else:
            ensemble_majority_k = 6  # 默认值，不显示
        
        ensemble_min_cooldown_days = st.slider(
            "最小冷却天数",
            min_value=1,
            max_value=10,
            value=default_cooldown,  # 根据ensemble方法自动调整
            step=1,
            help=f"避免频繁调整权重的最小间隔天数（{ensemble_method.title()}策略建议{default_cooldown}）",
            key="ensemble_min_cooldown_days"
        )
        
        ensemble_min_trade_dw = st.slider(
            "最小权重变化阈值",
            min_value=0.00,
            max_value=0.10,
            value=default_min_trade_dw,  # 根据ensemble方法自动调整
            step=0.01,
            help=f"忽略微小的权重变化，减少交易成本（{ensemble_method.title()}策略建议{default_min_trade_dw}）",
            key="ensemble_min_trade_dw"
        )
    
    # Ensemble 策略运行按钮
    if st.button("🚀 运行 Ensemble 策略", type="secondary", key="run_ensemble"):
        with st.spinner("正在运行 Ensemble 策略..."):
            try:
                # 构建参数
                ensemble_params = {
                    'method': ensemble_method,
                    'floor': ensemble_floor,
                    'ema_span': ensemble_ema_span,
                    'delta_cap': ensemble_delta_cap,
                    'majority_k': ensemble_majority_k,
                    'min_cooldown_days': ensemble_min_cooldown_days,
                    'min_trade_dw': ensemble_min_trade_dw
                }
                
                # 运行策略
                equity_curve, trades, stats = run_ensemble_strategy(ensemble_params)
                
                if equity_curve is not None and not equity_curve.empty:
                    st.success("✅ Ensemble 策略运行成功！")
                    
                    # 显示绩效指标
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总报酬率", f"{stats.get('total_return', 0):.2%}")
                    with col2:
                        st.metric("年化报酬率", f"{stats.get('annual_return', 0):.2%}")
                    with col3:
                        st.metric("最大回撤", f"{stats.get('max_drawdown', 0):.2%}")
                    with col4:
                        st.metric("夏普比率", f"{stats.get('sharpe_ratio', 0):.3f}")
                    
                    # 显示权益曲线
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve.values,
                        name='Ensemble 策略',
                        line=dict(color='red', width=2)
                    ))
                    fig.update_layout(
                        title="Ensemble 策略权益曲线",
                        xaxis_title="日期",
                        yaxis_title="权益",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示交易记录
                    if not trades.empty:
                        st.subheader("交易记录")
                        st.dataframe(trades, use_container_width=True)
                else:
                    st.warning("⚠️ Ensemble 策略运行完成，但未返回有效结果")
                    
            except Exception as e:
                st.error(f"❌ Ensemble 策略运行失败: {str(e)}")
    
    st.divider()

selected_strategies = st.multiselect(
    "選擇要分析的策略", 
    options=[s['name'] for s in all_strategies], 
    default=[s['name'] for s in all_strategies],  # 預設全選
    key="main_strategy_selector"
)

# 快速選擇按鈕
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("全選"):
        st.session_state.main_strategies = [s['name'] for s in all_strategies]
        st.rerun()
with col2:
    if st.button("清除"):
        st.session_state.main_strategies = []
        st.rerun()
with col3:
    if st.button("選擇Optuna策略"):
        optuna_names = [s['name'] for s in optuna_strategies]
        st.session_state.main_strategies = optuna_names
        st.rerun()

if not selected_strategies:
    st.warning("請至少選擇一個策略進行分析")
    st.stop()

st.info(f"已選擇 {len(selected_strategies)} 個策略進行分析")

# 相關性矩陣設定移到主頁面
corr_range = st.slider("相關性值範圍", min_value=-1.0, max_value=1.0, value=(0.8, 1.0), step=0.1, help="設定相關性矩陣熱圖的顏色範圍")

# 執行回測按鈕
if st.button("🚀 執行回測與分析", type="primary"):
    # 過濾選中的策略
    strategies_to_run = [s for s in all_strategies if s['name'] in selected_strategies]

    with st.spinner("正在執行回測..."):
        # 載入數據
        smaa_sources = set(strategy['smaa_source'] for strategy in strategies_to_run)
        df_price_dict = {}
        df_factor_dict = {}
        all_indices = []
        
        # 將數據載入訊息收集到expander中
        with st.expander("📊 載入信息", expanded=False):
            st.subheader("數據載入狀態:")
            for source in smaa_sources:
                df_price, df_factor = load_data(ticker="00631L.TW", smaa_source=source)
                if not df_price.empty:
                    all_indices.append(df_price.index)
                df_price_dict[source] = df_price
                df_factor_dict[source] = df_factor
                st.text(f"• {source}: 已載入 {len(df_price)} 筆數據")

        # 合併所有時間軸，創建一個全域時間軸
        if all_indices:
            global_index = pd.Index([])
            for index in all_indices:
                global_index = global_index.union(index)
        else:
            global_index = pd.Index([])

        # 執行回測
        results = {}
        initial_equity = 100000.0
        for strategy in strategies_to_run:
            name = strategy['name']
            strategy_type = strategy['strategy_type']
            params = strategy['params']
            smaa_source = strategy['smaa_source']
            df_price = df_price_dict[smaa_source]
            df_factor = df_factor_dict[smaa_source]
            
            logger.info(f"處理策略 {name}，參數: {params}")
            
            # 修正參數處理：支援多種格式
            # 檢查是否有 parameters 欄位（JSON 格式）
            if isinstance(params, dict) and 'parameters' in params:
                # 如果有 parameters 欄位，使用它
                actual_params = params['parameters']
                logger.info(f"使用 JSON 格式參數: {actual_params}")
            elif isinstance(params, str):
                # 如果是字符串，嘗試解析為 JSON
                try:
                    import json
                    actual_params = json.loads(params)
                    logger.info(f"解析 JSON 字符串參數: {actual_params}")
                except json.JSONDecodeError:
                    logger.error(f"無法解析 JSON 參數: {params}")
                    continue
            elif isinstance(params, dict) and len(params) == 0:
                # 如果是空字典，嘗試從 trial 中獲取 parameters
                logger.warning(f"策略 {name} 的參數為空字典，嘗試從 trial 中獲取")
                # 這裡需要從 selected_trials 中找到對應的 trial
                for trial in selected_trials:
                    if trial['short_name'] == name:
                        if isinstance(trial['parameters'], str):
                            try:
                                import json
                                actual_params = json.loads(trial['parameters'])
                                logger.info(f"從 trial 解析 JSON 字符串參數: {actual_params}")
                            except json.JSONDecodeError:
                                logger.error(f"無法解析 trial 的 JSON 參數: {trial['parameters']}")
                                continue
                        else:
                            actual_params = trial['parameters']
                            logger.info(f"從 trial 獲取參數: {actual_params}")
                        break
                else:
                    logger.error(f"找不到策略 {name} 對應的 trial")
                    continue
            else:
                # 否則直接使用 params（已經是分離的欄位）
                actual_params = params
                logger.info(f"使用分離欄位參數: {actual_params}")
            
            if strategy_type == 'single':
                df_ind = compute_single(df_price, df_factor, actual_params['linlen'], actual_params['factor'], actual_params['smaalen'], actual_params['devwin'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'dual':
                df_ind = compute_dual(df_price, df_factor, actual_params['linlen'], actual_params['factor'], actual_params['smaalen'], actual_params['short_win'], actual_params['long_win'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'RMA':
                df_ind = compute_RMA(df_price, df_factor, actual_params['linlen'], actual_params['factor'], actual_params['smaalen'], actual_params['rma_len'], actual_params['dev_len'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'ssma_turn':
                calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
                ssma_params = {k: v for k, v in actual_params.items() if k in calc_keys}
                backtest_params = ssma_params.copy()
                backtest_params['buy_mult'] = actual_params.get('buy_mult', 0.5)
                backtest_params['sell_mult'] = actual_params.get('sell_mult', 0.5)
                backtest_params['stop_loss'] = actual_params.get('stop_loss', 0.0)
                logger.info(f"SSMA_Turn 參數: {ssma_params}")
                df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_price, df_factor, **ssma_params, smaa_source=smaa_source)
                logger.info(f"策略 {name} 生成的買入信號數: {len(buy_dates)}, 賣出信號數: {len(sell_dates)}")
                logger.info(f"買入信號: {buy_dates[:5] if buy_dates else '無'}")
                logger.info(f"賣出信號: {sell_dates[:5] if sell_dates else '無'}")
                if df_ind.empty:
                    logger.error(f"策略 {name} 的 df_ind 為空，跳過回測")
                    st.error(f"策略 {name} 的 df_ind 為空，跳過回測")
                    continue
                # 使用與SSSv096一致的預設參數設定
                result = backtest_unified(df_ind, strategy_type, actual_params, buy_dates, sell_dates, 
                                         discount=0.30, trade_cooldown_bars=3, bad_holding=False)
                results[name] = result
                continue
            
            required_cols = ['open', 'close', 'smaa', 'base', 'sd']
            if df_ind.empty or not all(col in df_ind.columns for col in required_cols):
                logger.error(f"策略 {name} 的 df_ind 缺少必要欄位: {set(required_cols) - set(df_ind.columns)}，跳過回測")
                st.error(f"策略 {name} 的 df_ind 缺少必要欄位: {set(required_cols) - set(df_ind.columns)}，跳過回測")
                continue
            
            # 對於 single 和 RMA，目前的回測參數是固定的
            # 注意：如果這些策略也需要不同的 discount 或 cooldown，這裡需要修改
            if df_ind.empty:
                logger.error(f"策略 {name} 的 df_ind 為空，跳過計算")
                continue
            
            result = backtest_unified(df_ind, strategy_type, actual_params, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
            results[name] = result

        # 提取權益曲線
        equity_curves = pd.DataFrame({name: result['equity_curve'].reindex(global_index, fill_value=initial_equity) for name, result in results.items()})
        
        # 計算並儲存指標
        for name, result in results.items():
            if 'equity_curve' in result and not result['equity_curve'].empty:
                # 計算基本指標
                equity_curve = result['equity_curve']
                total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
                annual_return = total_return * (252 / len(equity_curve))
                daily_returns = equity_curve.pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                max_drawdown = calculate_max_drawdown(equity_curve)
                calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
                
                # 計算風險調整指標
                risk_adjusted_metrics = calculate_risk_adjusted_metrics(equity_curve, name)
                
                # 修正勝率計算 - 移到metrics字典創建之前
                trades_df = result.get('trades_df', pd.DataFrame())
                logger.info(f"策略 {name} 的 trades_df 形狀: {trades_df.shape}")
                logger.info(f"策略 {name} 的 trades_df 欄位: {list(trades_df.columns)}")

                # 添加更詳細的調試信息
                if not trades_df.empty:
                    logger.info(f"策略 {name} 的 trades_df 前5行:")
                    logger.info(trades_df.head())
                    logger.info(f"策略 {name} 的 trades_df 數據類型:")
                    logger.info(trades_df.dtypes)
                    if 'ret' in trades_df.columns:
                        logger.info(f"策略 {name} 的 'ret' 欄位統計:")
                        logger.info(f"  非空值數量: {trades_df['ret'].notna().sum()}")
                        logger.info(f"  空值數量: {trades_df['ret'].isna().sum()}")
                        logger.info(f"  唯一值: {trades_df['ret'].unique()}")
                        logger.info(f"  大於0的數量: {(trades_df['ret'] > 0).sum()}")
                        logger.info(f"  等於0的數量: {(trades_df['ret'] == 0).sum()}")
                        logger.info(f"  小於0的數量: {(trades_df['ret'] < 0).sum()}")

                # 計算勝率
                win_rate = 0
                if not trades_df.empty and 'ret' in trades_df.columns:
                    winning_trades = trades_df[trades_df['ret'] > 0]
                    total_trades = len(trades_df)
                    winning_count = len(winning_trades)
                    win_rate = winning_count / total_trades if total_trades > 0 else 0
                    logger.info(f"策略 {name} 勝率計算: 總交易數={total_trades}, 獲利交易數={winning_count}, 勝率={win_rate:.2%}")
                    
                    # 添加詳細的交易記錄日誌
                    if total_trades > 0:
                        logger.info(f"策略 {name} 交易報酬率統計:")
                        logger.info(f"  最小報酬率: {trades_df['ret'].min():.4f}")
                        logger.info(f"  最大報酬率: {trades_df['ret'].max():.4f}")
                        logger.info(f"  平均報酬率: {trades_df['ret'].mean():.4f}")
                        logger.info(f"  正報酬交易: {winning_count} 筆")
                        logger.info(f"  負報酬交易: {total_trades - winning_count} 筆")
                else:
                    logger.warning(f"策略 {name} 的 trades_df 為空或缺少 'ret' 欄位")
                    if not trades_df.empty:
                        logger.warning(f"策略 {name} 的 trades_df 實際欄位: {list(trades_df.columns)}")

                # 合併所有指標
                result['metrics'] = {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'calmar_ratio': calmar_ratio,
                    'max_drawdown': max_drawdown,
                    'num_trades': len(result.get('trades_df', pd.DataFrame())),
                    'win_rate': win_rate,  # 使用計算出的勝率
                    **risk_adjusted_metrics  # 添加風險調整指標
                }
                
                # 修正卡瑪值計算
                if result['metrics']['max_drawdown'] != 0:
                    result['metrics']['calmar_ratio'] = result['metrics']['annual_return'] / abs(result['metrics']['max_drawdown'])
                else:
                    result['metrics']['calmar_ratio'] = 0
        
        if equity_curves.empty:
            logger.error("未生成任何權益曲線，請檢查回測邏輯或數據")
            st.error("未生成任何權益曲線，請檢查回測邏輯或數據")
        else:
            logger.info(f"權益曲線形狀: {equity_curves.shape}")
            logger.info(equity_curves.head())

            # 使用標籤頁呈現結果
            tabs = st.tabs(["相關性矩陣熱圖", "報酬率相關", "最大回撤相關", "總結", "過擬合檢測", "Top策略參數"])

            # 標籤頁 1: 相關性矩陣熱圖
            with tabs[0]:
                st.subheader("策略相關性矩陣熱圖")
                
                # 顯示說明氣泡
                with st.expander("📖 相關性分析說明", expanded=False):
                    st.markdown("""
                    **相關性矩陣**: 顯示不同策略之間的相關性
                    
                    **相關係數範圍**: -1 到 +1
                    - +1: 完全正相關（兩個策略完全同步）
                    - 0: 無相關（兩個策略獨立）
                    - -1: 完全負相關（兩個策略完全相反）
                    
                    **投資組合意義**:
                    - 低相關性（接近0）: 適合組合投資，分散風險
                    - 高相關性（接近±1）: 組合效果有限，風險集中
                    """)
                
                # 直接使用已計算的結果
                if equity_curves.empty:
                    st.warning("沒有權益曲線數據")
                else:
                    # 計算相關性矩陣
                    corr_matrix = equity_curves.corr()
                    corr_matrix.to_csv(RESULT_DIR / 'correlation_matrix.csv')
                    
                    # 使用主頁面設定的相關性範圍
                    zmin, zmax = corr_range
                    
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu_r',
                        zmin=zmin,
                        zmax=zmax,
                        text_auto=True,
                        title="策略相關性矩陣"
                    )
                    fig.update_layout(
                        width=2200,
                        height=1500,
                        xaxis_title="策略",
                        yaxis_title="策略",
                        coloraxis_colorbar_title="相關性"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 添加相關性統計信息
                    st.subheader("相關性統計")
                    # 使用numpy的triu_indices來獲取上三角矩陣的值
                    if corr_matrix is not None:
                        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                        corr_stats = {
                            '平均相關性': upper_triangle.mean(),
                            '最大相關性': upper_triangle.max(),
                            '最小相關性': upper_triangle.min(),
                            '相關性標準差': upper_triangle.std()
                        }
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("平均相關性", f"{corr_stats['平均相關性']:.3f}")
                        with col2:
                            st.metric("最大相關性", f"{corr_stats['最大相關性']:.3f}")
                        with col3:
                            st.metric("最小相關性", f"{corr_stats['最小相關性']:.3f}")
                        with col4:
                            st.metric("相關性標準差", f"{corr_stats['相關性標準差']:.3f}")

            # 標籤頁 2: 報酬率相關
            with tabs[1]:
                st.subheader("走查時段報酬率分數 (%)")
                period_returns = {}
                period_mdd = {}
                period_pr = {}
                period_mdd_pr = {}
                period_hedge_counts = {}
                periods = WF_PERIODS if walk_forward_mode == "固定 WF_PERIODS" else generate_walk_forward_periods(equity_curves.index, n_splits)

                for i, period in enumerate([(p['test'] if isinstance(p, dict) else p) for p in periods]):
                    start = pd.to_datetime(period[0])
                    end = pd.to_datetime(period[1])
                    valid_dates = pd.to_datetime(equity_curves.index)
                    if start < valid_dates[0]:
                        adjusted_start = valid_dates[0]
                        logger.warning(f"起始日期 {start.strftime('%Y-%m-%d')} 早於數據開始，調整為 {adjusted_start.strftime('%Y-%m-%d')}")
                    else:
                        adjusted_start = valid_dates[valid_dates >= start][0]
                    if end > valid_dates[-1]:
                        adjusted_end = valid_dates[-1]
                        logger.warning(f"結束日期 {end.strftime('%Y-%m-%d')} 晚於數據結束，調整為 {adjusted_end.strftime('%Y-%m-%d')}")
                    else:
                        adjusted_end = valid_dates[valid_dates <= end][-1]
                    if (adjusted_end - adjusted_start).days < 30:
                        logger.warning(f"走查區間 {adjusted_start.strftime('%Y-%m-%d')} 至 {adjusted_end.strftime('%Y-%m-%d')} 過短，跳過")
                        continue
                    period_equity = equity_curves.loc[adjusted_start:adjusted_end]
                    for col in period_equity.columns:
                        if pd.isna(period_equity[col].iloc[0]):
                            period_equity[col].iloc[0] = initial_equity
                    # 計算報酬率和 MDD
                    period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                    period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                    # 計算避險掩碼
                    hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                    # 記錄避險次數
                    if i > 0:  # 非初始時段
                        period_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                    else:  # 初始時段
                        period_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                    # 計算報酬率 PR 值
                    pr_values_ret = calculate_pr_values(period_return, is_mdd=False, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                    period_pr[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pr_values_ret
                    # 儲存報酬率
                    period_returns[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = period_return

                period_returns_df = pd.DataFrame(period_returns).T
                period_returns_df.to_csv(RESULT_DIR / 'period_returns.csv')
                st.dataframe(period_returns_df.style.format("{:.2f}%"))
                period_pr_df = pd.DataFrame(period_pr).T
                period_pr_df.to_csv(RESULT_DIR / 'period_pr.csv')
                st.subheader("走查時段報酬率分數 (%)")
                st.dataframe(period_pr_df.style.format("{:.2f}"))
                period_hedge_counts_df = pd.DataFrame(period_hedge_counts).T
                period_hedge_counts_df.to_csv(RESULT_DIR / 'period_hedge_counts.csv')
                st.subheader("走查時段避險次數")
                st.dataframe(period_hedge_counts_df)
                stress_returns = {}
                stress_pr = {}
                stress_hedge_counts = {}
                for i, (start, end) in enumerate(STRESS_PERIODS):
                    start = pd.to_datetime(start)
                    end = pd.to_datetime(end)
                    if start in equity_curves.index and end in equity_curves.index:
                        period_equity = equity_curves.loc[start:end]
                        for col in period_equity.columns:
                            if pd.isna(period_equity[col].iloc[0]):
                                period_equity[col].iloc[0] = initial_equity
                        # 計算報酬率和 MDD
                        period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                        period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                        # 計算避險掩碼
                        hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                        # 記錄避險次數（壓力時段第一個為初始時段）
                        if i > 0:  # 非初始時段
                            stress_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                        else:  # 初始時段
                            stress_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                        # 計算報酬率 PR 值
                        pr_values_ret = calculate_pr_values(period_return, is_mdd=False, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                        stress_pr[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pr_values_ret
                        # 儲存報酬率
                        stress_returns[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = period_return

                stress_returns_df = pd.DataFrame(stress_returns).T
                stress_returns_df.to_csv(RESULT_DIR / 'stress_returns.csv')
                st.subheader("壓力時段報酬率 (%)")
                st.dataframe(stress_returns_df.style.format("{:.2f}%"))
                stress_pr_df = pd.DataFrame(stress_pr).T
                stress_pr_df.to_csv(RESULT_DIR / 'stress_pr.csv')
                st.subheader("壓力時段報酬率分數 (%)")
                st.dataframe(stress_pr_df.style.format("{:.2f}"))
                stress_hedge_counts_df = pd.DataFrame(stress_hedge_counts).T
                stress_hedge_counts_df.to_csv(RESULT_DIR / 'stress_hedge_counts.csv')
                st.subheader("壓力時段避險次數")
                st.dataframe(stress_hedge_counts_df)

            # 標籤頁 3: 最大回撤相關
            with tabs[2]:
                st.subheader("走查時段最大回撤分數 (%)")
                period_mdd = {}
                period_mdd_pr = {}
                period_mdd_hedge_counts = {}
                periods = WF_PERIODS if walk_forward_mode == "固定 WF_PERIODS" else generate_walk_forward_periods(equity_curves.index, n_splits)
                for i, period in enumerate([(p['test'] if isinstance(p, dict) else p) for p in periods]):
                    start = pd.to_datetime(period[0])
                    end = pd.to_datetime(period[1])
                    valid_dates = pd.to_datetime(equity_curves.index)
                    if start < valid_dates[0]:
                        adjusted_start = valid_dates[0]
                    else:
                        adjusted_start = valid_dates[valid_dates >= start][0]
                    if end > valid_dates[-1]:
                        adjusted_end = valid_dates[-1]
                    else:
                        adjusted_end = valid_dates[valid_dates <= end][-1]
                    if (adjusted_end - adjusted_start).days < 30:
                        continue
                    period_equity = equity_curves.loc[adjusted_start:adjusted_end]
                    for col in period_equity.columns:
                        if pd.isna(period_equity[col].iloc[0]):
                            period_equity[col].iloc[0] = initial_equity
                    # 計算報酬率（用於避險掩碼）
                    period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                    # 計算 MDD
                    period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                    # 計算避險掩碼
                    hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                    # 記錄避險次數
                    if i > 0:  # 非初始時段
                        period_mdd_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                    else:  # 初始時段
                        period_mdd_hedge_counts[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                    # 計算 MDD 分數
                    pr_values_mdd = calculate_pr_values(period_mdd_value, is_mdd=True, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                    period_mdd_pr[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = pr_values_mdd
                    # 儲存 MDD
                    period_mdd[(adjusted_start.strftime('%Y-%m-%d'), adjusted_end.strftime('%Y-%m-%d'))] = period_mdd_value

                period_mdd_df = pd.DataFrame(period_mdd).T
                period_mdd_df.to_csv(RESULT_DIR / 'period_mdd.csv')
                st.dataframe(period_mdd_df.style.format("{:.2f}%"))
                period_mdd_pr_df = pd.DataFrame(period_mdd_pr).T
                period_mdd_pr_df.to_csv(RESULT_DIR / 'period_mdd_pr.csv')
                st.subheader("走查時段最大回撤分數 (%)")
                st.dataframe(period_mdd_pr_df.style.format("{:.2f}"))
                period_mdd_hedge_counts_df = pd.DataFrame(period_mdd_hedge_counts).T
                period_mdd_hedge_counts_df.to_csv(RESULT_DIR / 'period_mdd_hedge_counts.csv')
                st.subheader("走查時段最大回撤避險次數")
                st.dataframe(period_mdd_hedge_counts_df)


                stress_mdd = {}
                stress_mdd_pr = {}
                stress_mdd_hedge_counts = {}
                for i, (start, end) in enumerate(STRESS_PERIODS):
                    start = pd.to_datetime(start)
                    end = pd.to_datetime(end)
                    if start in equity_curves.index and end in equity_curves.index:
                        period_equity = equity_curves.loc[start:end]
                        for col in period_equity.columns:
                            if pd.isna(period_equity[col].iloc[0]):
                                period_equity[col].iloc[0] = initial_equity
                        # 計算報酬率和 MDD
                        period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) * 100
                        period_mdd_value = period_equity.apply(calculate_max_drawdown) * 100
                        # 計算避險掩碼
                        hedge_mask = (period_return == 0) & (period_mdd_value == 0)
                        # 記錄避險次數
                        if i > 0:  # 非初始時段
                            stress_mdd_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = hedge_mask.astype(int)
                        else:  # 初始時段
                            stress_mdd_hedge_counts[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pd.Series(0, index=hedge_mask.index)
                        # 計算 MDD 分數
                        pr_values_mdd = calculate_pr_values(period_mdd_value, is_mdd=True, is_initial_period=(i == 0), hedge_mask=hedge_mask)
                        stress_mdd_pr[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = pr_values_mdd
                        # 儲存 MDD
                        stress_mdd[(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))] = period_mdd_value

                stress_mdd_df = pd.DataFrame(stress_mdd).T
                stress_mdd_df.to_csv(RESULT_DIR / 'stress_mdd.csv')
                st.subheader("壓力時段最大回撤 (%)")
                st.dataframe(stress_mdd_df.style.format("{:.2f}%"))
                stress_mdd_pr_df = pd.DataFrame(stress_mdd_pr).T
                stress_mdd_pr_df.to_csv(RESULT_DIR / 'stress_mdd_pr.csv')
                st.subheader("壓力時段最大回撤分數 (%)")
                st.dataframe(stress_mdd_pr_df.style.format("{:.2f}"))
                stress_mdd_hedge_counts_df = pd.DataFrame(stress_mdd_hedge_counts).T
                stress_mdd_hedge_counts_df.to_csv(RESULT_DIR / 'stress_mdd_hedge_counts.csv')
                st.subheader("壓力時段最大回撤避險次數")
                st.dataframe(stress_mdd_hedge_counts_df)

                mdd = {}
                for name, equity in equity_curves.items():
                    mdd[name] = calculate_max_drawdown(equity) * 100
                mdd_df = pd.Series(mdd)
                mdd_df.to_csv(RESULT_DIR / 'mdd.csv')

            # 標籤頁 4: 總結
            with tabs[3]:
                st.subheader("策略匯總分析")
                
                # 顯示說明氣泡
                with st.expander("📖 匯總分析說明", expanded=False):
                    st.markdown("""
                    **策略匯總分析**: 綜合評估各策略的表現                
                    - 風險調整報酬: 夏普比*總報酬率
                    - 綜合評分: sqrt(總報酬率)*10*0.3+夏普值*0.25+最大回撤*0.2+勝率*0.15+風險調整報酬*0.1-過擬合分數*0.5
                    

                    """)
                
                # 創建策略匯總表
                summary_data = []
                for name, result in results.items():
                    if 'metrics' in result and result['metrics']:
                        metrics = result['metrics']
                        
                        # 計算風險調整後報酬率
                        risk_adjusted_return = metrics.get('sharpe_ratio', 0) * metrics.get('total_return', 0)
                        
                        summary_data.append({
                            "策略": name,
                            "總報酬率": metrics.get('total_return', 0),
                            "年化報酬率": metrics.get('annual_return', 0),
                            "最大回撤": metrics.get('max_drawdown', 0),
                            "夏普值": metrics.get('sharpe_ratio', 0),
                            "卡瑪值": metrics.get('calmar_ratio', 0),
                            "交易次數": metrics.get('num_trades', 0),
                            "勝率": metrics.get('win_rate', 0),
                            "風險調整報酬": risk_adjusted_return
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data).set_index("策略")
                    
                    # 計算綜合評分（加入過擬合懲罰）
                    # 首先計算過擬合分數
                    overfitting_scores = {}
                    for name, result in results.items():
                        if 'equity_curve' in result and not result['equity_curve'].empty:
                            equity_curve = result['equity_curve']
                            
                            # 分割訓練和測試期間
                            split_point = int(len(equity_curve) * 0.7)
                            train_equity = equity_curve.iloc[:split_point]
                            test_equity = equity_curve.iloc[split_point:]
                            
                            if len(train_equity) > 30 and len(test_equity) > 30:
                                train_returns = train_equity.pct_change().dropna()
                                test_returns = test_equity.pct_change().dropna()
                                
                                # 計算年化報酬率
                                train_annual_return = train_returns.mean() * 252
                                test_annual_return = test_returns.mean() * 252
                                
                                # 計算夏普值
                                train_sharpe = train_annual_return / (train_returns.std() * np.sqrt(252)) if train_returns.std() > 0 else 0
                                test_sharpe = test_annual_return / (test_returns.std() * np.sqrt(252)) if test_returns.std() > 0 else 0
                                
                                # 計算過擬合指標
                                sharpe_degradation = train_sharpe - test_sharpe
                                return_degradation = train_annual_return - test_annual_return
                                
                                # 過擬合分數（0-100，越高越過擬合）
                                sharpe_weight = 0.6
                                return_weight = 0.4
                                overfitting_score = min(100, max(0, 
                                    abs(sharpe_degradation) * 50 * sharpe_weight + 
                                    abs(return_degradation) * 200 * return_weight
                                ))
                                overfitting_scores[name] = overfitting_score
                            else:
                                overfitting_scores[name] = 50  # 預設中等風險
                        else:
                            overfitting_scores[name] = 50  # 預設中等風險
                    
                    # 計算綜合評分（扣掉過擬合分數*0.5）
                    summary_df['綜合評分'] = (
                        np.sqrt(summary_df['總報酬率']) * 10 * 0.3 +
                        summary_df['夏普值'] * 0.25 +
                        (1 + summary_df['最大回撤']) * 0.2 +
                        summary_df['勝率'] * 0.1 +
                        np.sqrt(summary_df['風險調整報酬']) * 0.05
                    )
                    print(f"總報酬率分數: {np.sqrt(summary_df['總報酬率']) * 10 * 0.3},夏普值分數: {summary_df['夏普值'] * 0.25},最大回撤分數: {(1 + summary_df['最大回撤']) * 0.2},勝率分數: {summary_df['勝率'] * 0.1},風險調整報酬分數: {np.sqrt(summary_df['風險調整報酬']) * 0.05}")
                    # 扣掉過擬合分數懲罰
                    for name in summary_df.index:
                        if name in overfitting_scores:
                            print(f"扣掉過擬合分數: {name} {overfitting_scores[name]}")
                            summary_df.loc[name, '綜合評分'] -= overfitting_scores[name] * 0.5
                    
                    # 按綜合評分排序
                    summary_df = summary_df.sort_values('綜合評分', ascending=False)
                    
                    # 顯示匯總表
                    st.subheader("策略績效匯總表")
                    st.dataframe(summary_df.style.format({
                        '總報酬率': '{:.2%}',
                        '年化報酬率': '{:.2%}',
                        '最大回撤': '{:.2%}',
                        '夏普值': '{:.3f}',
                        '卡瑪值': '{:.3f}',
                        '交易次數': '{:d}',
                        '勝率': '{:.2%}',
                        '風險調整報酬': '{:.3f}',
                        '綜合評分': '{:.3f}'
                    }))
                    
                    # 顯示最佳策略
                    if not summary_df.empty:
                        best_strategy = summary_df.index[0]
                        st.subheader(f"🏆 最佳策略: {best_strategy}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("年化報酬率", f"{summary_df.loc[best_strategy, '年化報酬率']:.2%}")
                        with col2:
                            st.metric("夏普值", f"{summary_df.loc[best_strategy, '夏普值']:.3f}")
                        with col3:
                            st.metric("最大回撤", f"{summary_df.loc[best_strategy, '最大回撤']:.2%}")
                        
                        # 策略比較圖
                        st.subheader("策略比較")
                        
                        # 報酬率 vs 風險散點圖
                        fig_scatter = px.scatter(
                            summary_df.reset_index(),
                            x='最大回撤',
                            y='年化報酬率',
                            size='夏普值',
                            color='綜合評分',
                            hover_name='策略',
                            title="報酬率 vs 風險散點圖",
                            labels={'最大回撤': '最大回撤 (%)', '年化報酬率': '年化報酬率 (%)'}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # 策略排名圖
                        fig_ranking = px.bar(
                            summary_df.reset_index(),
                            x='策略',
                            y='綜合評分',
                            title="策略綜合評分排名",
                            color='綜合評分',
                            color_continuous_scale='viridis'
                        )
                        fig_ranking.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_ranking, use_container_width=True)
                else:
                    st.warning("沒有可用的策略數據進行匯總分析")

            # 標籤頁 5: 過擬合檢測
            with tabs[4]:             
                # 計算過擬合指標
                overfitting_metrics = {}
                for name, result in results.items():
                    if 'equity_curve' in result and not result['equity_curve'].empty:
                        equity_curve = result['equity_curve']
                        
                        # 分割訓練和測試期間
                        split_point = int(len(equity_curve) * 0.7)
                        train_equity = equity_curve.iloc[:split_point]
                        test_equity = equity_curve.iloc[split_point:]
                        
                        if len(train_equity) > 30 and len(test_equity) > 30:
                            train_returns = train_equity.pct_change().dropna()
                            test_returns = test_equity.pct_change().dropna()
                            
                            # 計算年化報酬率
                            train_annual_return = train_returns.mean() * 252
                            test_annual_return = test_returns.mean() * 252
                            
                            # 計算夏普值
                            train_sharpe = train_annual_return / (train_returns.std() * np.sqrt(252)) if train_returns.std() > 0 else 0
                            test_sharpe = test_annual_return / (test_returns.std() * np.sqrt(252)) if test_returns.std() > 0 else 0
                            
                            # 計算過擬合指標
                            sharpe_degradation = train_sharpe - test_sharpe
                            return_degradation = train_annual_return - test_annual_return
                            
                            # 過擬合分數（0-100，越高越過擬合）
                            # 修正計算方式：考慮夏普值和報酬率的相對重要性
                            sharpe_weight = 0.6
                            return_weight = 0.4
                            overfitting_score = min(100, max(0, 
                                abs(sharpe_degradation) * 50 * sharpe_weight + 
                                abs(return_degradation) * 200 * return_weight
                            ))
                            
                            overfitting_metrics[name] = {
                                'train_sharpe': train_sharpe,
                                'test_sharpe': test_sharpe,
                                'train_return': train_annual_return,
                                'test_return': test_annual_return,
                                'sharpe_degradation': sharpe_degradation,
                                'return_degradation': return_degradation,
                                'overfitting_score': overfitting_score
                            }
                
                if overfitting_metrics:
                    # 過擬合分數排名
                    overfitting_df = pd.DataFrame(overfitting_metrics).T
                    overfitting_df = overfitting_df.sort_values('overfitting_score')
                    
                    st.subheader("過擬合分數排名（越低越好）")
                    
                    # 使用說明氣泡顯示過擬合指標
                    display_df = overfitting_df[['overfitting_score', 'sharpe_degradation', 'return_degradation']].copy()
                    display_df.columns = ['過擬合分數', '夏普值退化', '報酬率退化']
                    
                    # 顯示說明氣泡
                    with st.expander("📖 過擬合指標說明", expanded=False):
                        st.markdown("""
                        **過擬合分數**: 0-100，分數越高表示過擬合風險越大
                        - 0-30: 低風險，策略較穩定
                        - 31-60: 中風險，需要關注
                        - 61-100: 高風險，可能存在過擬合問題
                        
                        **夏普值退化**: 樣本內外夏普值的差異，負值表示樣本外表現更好
                        
                        **報酬率退化**: 樣本內外年化報酬率的差異，負值表示樣本外表現更好
                        """)
                    
                    st.dataframe(display_df.style.format({
                        '過擬合分數': '{:.1f}',
                        '夏普值退化': '{:.3f}',
                        '報酬率退化': '{:.3f}'
                    }))
                    
                    # 過擬合分數分布圖
                    fig_overfitting = px.bar(
                        overfitting_df.reset_index(), 
                        x='index', 
                        y='overfitting_score',
                        title="策略過擬合分數分布",
                        labels={'index': '策略', 'overfitting_score': '過擬合分數'}
                    )
                    fig_overfitting.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="輕微過擬合")
                    fig_overfitting.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="嚴重過擬合")
                    st.plotly_chart(fig_overfitting, use_container_width=True)
                    
                    # 樣本內外表現對比
                    st.subheader("樣本內外表現對比")
                    
                    # 夏普值對比
                    comparison_sharpe_df = overfitting_df[['train_sharpe', 'test_sharpe']].reset_index()
                    comparison_sharpe_df.columns = ['策略', '樣本內夏普值', '樣本外夏普值']
                    
                    with st.expander("📖 樣本內外比較說明", expanded=False):
                        st.markdown("""
                        **樣本內夏普值**: 訓練期間（前70%數據）的夏普值
                        
                        **樣本外夏普值**: 測試期間（後30%數據）的夏普值
                        
                        理想情況下，樣本內外表現應該相近，差異過大可能表示過擬合。
                        """)
                    
                    fig_comparison_sharpe = px.bar(
                        comparison_sharpe_df,
                        x='策略',
                        y=['樣本內夏普值', '樣本外夏普值'],
                        title="樣本內外夏普值對比",
                        barmode='group'
                    )
                    st.plotly_chart(fig_comparison_sharpe, use_container_width=True)
                    
                    # 報酬率對比
                    comparison_return_df = overfitting_df[['train_return', 'test_return']].reset_index()
                    comparison_return_df.columns = ['策略', '樣本內年化報酬率', '樣本外年化報酬率']
                    
                    fig_comparison_return = px.bar(
                        comparison_return_df,
                        x='策略',
                        y=['樣本內年化報酬率', '樣本外年化報酬率'],
                        title="樣本內外年化報酬率對比",
                        barmode='group'
                    )
                    st.plotly_chart(fig_comparison_return, use_container_width=True)
                else:
                    st.warning("無法計算過擬合指標，請確保有足夠的數據進行樣本內外分割")
                
                # 策略穩定性分析
                st.subheader("策略穩定性分析")
                # 從權益曲線計算期間報酬率
                period_returns = {}
                for name, result in results.items():
                    if 'equity_curve' in result and not result['equity_curve'].empty:
                        equity_curve = result['equity_curve']
                        # 計算月度報酬率
                        monthly_returns = equity_curve.resample('M').last().pct_change().dropna() * 100
                        period_returns[name] = monthly_returns
                        logger.info(f"策略 {name} 月度報酬率: {len(monthly_returns)} 個期間")
                
                logger.info(f"開始計算穩定性指標，共 {len(period_returns)} 個策略")
                
                # 顯示期間報酬率的基本信息
                for name, returns in period_returns.items():
                    logger.info(f"策略 {name}: {len(returns)} 個期間, 範圍: {returns.index.min()} 到 {returns.index.max()}")
                
                stability_metrics = calculate_strategy_stability(period_returns)
                
                if stability_metrics:
                    stability_df = pd.DataFrame(stability_metrics).T
                    
                    # 重命名欄位為中文
                    stability_df.columns = ['平均報酬率', '報酬率標準差', '變異係數', '正報酬期間比例', '排名穩定性']
                    
                    logger.info(f"穩定性指標計算完成，共 {len(stability_df)} 個策略")
                    
                    with st.expander("📖 穩定性指標說明", expanded=False):
                        st.markdown("""
                        **平均報酬率**: 各期間的平均報酬率
                        
                        **報酬率標準差**: 各期間報酬率的變異程度，越小越穩定
                        
                        **變異係數**: 標準差與平均值的比值，衡量相對變異性，越小越穩定
                        
                        **正報酬期間比例**: 產生正報酬的期間佔總期間的比例，越高越好
                        
                        **排名穩定性**: 策略在不同期間的排名穩定性，基於排名變異係數計算，越接近1表示排名越穩定
                        """)
                    
                    # 處理無限值和異常值
                    def format_stability_value(val, format_type):
                        if pd.isna(val) or val == float('inf') or val == float('-inf'):
                            return 'N/A'
                        if format_type == 'percent':
                            return f"{val:.2%}"
                        elif format_type == 'decimal':
                            return f"{val:.2f}"
                        elif format_type == 'correlation':
                            return f"{val:.3f}"
                        else:
                            return str(val)
                    
                    # 創建格式化的DataFrame
                    formatted_df = stability_df.copy()
                    formatted_df['平均報酬率'] = formatted_df['平均報酬率'].apply(lambda x: format_stability_value(x, 'percent'))
                    formatted_df['報酬率標準差'] = formatted_df['報酬率標準差'].apply(lambda x: format_stability_value(x, 'percent'))
                    formatted_df['變異係數'] = formatted_df['變異係數'].apply(lambda x: format_stability_value(x, 'decimal'))
                    formatted_df['正報酬期間比例'] = formatted_df['正報酬期間比例'].apply(lambda x: format_stability_value(x, 'percent'))
                    formatted_df['排名穩定性'] = formatted_df['排名穩定性'].apply(lambda x: format_stability_value(x, 'correlation'))
                    
                    st.dataframe(formatted_df)
                    
                    # 穩定性熱力圖（只顯示有效的數值）
                    if len(stability_df) > 1:
                        # 過濾掉無限值和NaN值
                        heatmap_data = stability_df[['變異係數', '正報酬期間比例', '排名穩定性']].copy()
                        heatmap_data = heatmap_data.replace([float('inf'), float('-inf')], np.nan)
                        heatmap_data = heatmap_data.dropna(how='all')
                        
                        if not heatmap_data.empty:
                            fig_stability = px.imshow(
                                heatmap_data.T,
                                title="策略穩定性指標熱力圖",
                                aspect="auto"
                            )
                            st.plotly_chart(fig_stability, use_container_width=True)
                        else:
                            st.warning("無法生成穩定性熱力圖，數據包含過多無效值")
                else:
                    st.warning("無法計算穩定性指標，請確保有足夠的期間數據")

            # 顯示總體回測結果
            st.subheader("策略權益曲線")

            equity_curves_list = []
            for name, result in results.items():
                if 'equity_curve' in result and not result['equity_curve'].empty:
                    equity_curve = result['equity_curve'].rename(name)
                    equity_curves_list.append(equity_curve)

            if not equity_curves_list:
                st.warning("沒有可用的權益曲線數據進行繪製。")
                logger.warning("沒有可用的權益曲線數據進行繪製。")
            else:
                all_equity_curves = pd.concat(equity_curves_list, axis=1)
                all_equity_curves = all_equity_curves.reindex(global_index).ffill()
                all_equity_curves.ffill(inplace=True)
                all_equity_curves.fillna(initial_equity, inplace=True)

                df_plot = all_equity_curves.reset_index().melt(id_vars='index', var_name='策略', value_name='權益')
                df_plot.rename(columns={'index': '日期'}, inplace=True)
                
                fig_equity = px.line(df_plot, x='日期', y='權益', color='策略', title="策略權益曲線")
                fig_equity.update_layout(legend_title_text='variable')
                st.plotly_chart(fig_equity, use_container_width=True, key="total_equity_curve")

            # 顯示匯總指標
            st.subheader("策略匯總指標")
            summary_data = []
            for name, result in results.items():
                if 'metrics' in result and result['metrics']:
                    metrics = result['metrics']
                    
                    # 添加調試信息
                    win_rate_value = metrics.get('win_rate', 0)
                    logger.info(f"策略 {name} 的勝率值: {win_rate_value}, 類型: {type(win_rate_value)}")
                    
                    row = {
                        "策略": name,
                        "總報酬率": metrics.get('total_return', 0),
                        "年化報酬率": metrics.get('annual_return', 0),
                        "最大回撤": metrics.get('max_drawdown', 0),
                        "夏普值": metrics.get('sharpe_ratio', 0),
                        "卡瑪值": metrics.get('calmar_ratio', 0),
                        "交易次數": metrics.get('num_trades', 0),
                        "勝率": win_rate_value
                    }
                    summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data).set_index("策略")
                
                # 顯示說明氣泡
                with st.expander("📖 績效指標說明", expanded=False):
                    st.markdown("""
                    **總報酬率**: 整個回測期間的累積報酬率
                    
                    **年化報酬率**: 將總報酬率轉換為年化標準，便於比較不同期間的策略
                    
                    **最大回撤**: 權益曲線從峰值到谷值的最大跌幅，衡量下行風險
                    
                    **夏普值**: 超額報酬率與波動率的比值，衡量風險調整後報酬
                    
                    **卡瑪值**: 年化報酬率與最大回撤的比值，衡量風險調整後報酬
                    
                    **交易次數**: 整個回測期間的總交易次數
                    
                    **勝率**: 獲利交易次數佔總交易次數的比例
                    """)
                
                # 顯示表格
                st.dataframe(summary_df.style.format({
                    "總報酬率": "{:.2%}", "年化報酬率": "{:.2%}", "最大回撤": "{:.2%}",
                    "夏普值": "{:.2f}", "卡瑪值": "{:.2f}", "勝率": "{:.2%}"
                }))
            else:
                st.warning("沒有可用的匯總指標數據。")

            # 蒙地卡羅測試建議
            st.info("蒙地卡羅測試：請參考 Optuna_12.py 中的 compute_pbo_score 和 compute_simplified_sra 函數實現 PBO 分數與 SRA p 值計算")

            # 載入和執行信息
            with st.expander("📋 執行信息", expanded=False):            
                st.info(f"載入 {len(strategies_to_run)} 個策略，共 {len(results)} 個試驗")
                
                # 顯示選中的策略信息
                st.subheader("已選擇的策略:")
                for strategy in strategies_to_run:
                    st.text(f"• {strategy['name']} ({strategy['strategy_type']})")
                
                # 顯示策略統計
                strategy_counts = {}
                for strategy in strategies_to_run:
                    strategy_type = strategy['strategy_type']
                    data_source = strategy['smaa_source']
                    key = f"{strategy_type}_{data_source}"
                    strategy_counts[key] = strategy_counts.get(key, 0) + 1
                
                st.subheader("策略分布:")
                for key, count in strategy_counts.items():
                    st.text(f"• {key}: {count} 個策略")
                
                # 顯示執行信息
                st.subheader("執行信息")
                st.text(f"• 選中策略數量: {len(strategies_to_run)}")
                st.text(f"• 實際執行策略: {len(results)}")
                st.text(f"• 權益曲線數據點: {len(equity_curves) if not equity_curves.empty else 0}")

            # 標籤頁 6: Top策略參數
            with tabs[5]:
                st.subheader("Top策略參數詳情 (僅Optuna策略)")
                
                # 設定顯示的Top N策略數量
                top_n = st.slider("各策略類型顯示前N名", min_value=1, max_value=20, value=10, help="選擇每個策略類型和數據源組合要顯示的Top N策略數量")
                
                # 直接從optuna_results中獲取所有策略，按策略類型和數據源分組
                if not optuna_results.empty:
                    # 計算綜合評分（如果還沒計算的話）
                    if '綜合評分' not in optuna_results.columns:
                        # 計算風險調整後報酬率
                        optuna_results['風險調整報酬'] = optuna_results['sharpe_ratio'] * optuna_results['total_return']
                        
                        # 計算綜合評分（處理缺失欄位）
                        optuna_results['綜合評分'] = (
                            np.sqrt(optuna_results['total_return']) * 10 * 0.3 +
                            optuna_results['sharpe_ratio'] * 0.25 +
                            (1 + optuna_results['max_drawdown']) * 0.2 +
                            (optuna_results.get('win_rate', 0.5) * 0.1) +  # 如果沒有勝率，使用0.5作為預設值
                            np.sqrt(optuna_results['風險調整報酬']) * 0.05
                        )
                    
                    # 按策略類型和數據源分組並取各組合的前N名
                    top_strategies_by_group = {}
                    
                    # 直接從optuna_results生成所有可能的組合（使用tuple）
                    strategy_groups = set()
                    for _, row in optuna_results.iterrows():
                        strategy_type = row['strategy']
                        data_source = row['data_source']
                        group_key = (strategy_type, data_source)
                        strategy_groups.add(group_key)
                    
                    st.write(f"**從optuna_results生成的組合:** {list(strategy_groups)}")
                    
                    # 特別排查ssma_turn
                    st.subheader("🔍 ssma_turn 詳細排查")
                    ssma_turn_data = optuna_results[optuna_results['strategy'] == 'ssma_turn']
                    st.write(f"**ssma_turn總數據量:** {len(ssma_turn_data)}")
                    
                    if not ssma_turn_data.empty:
                        st.write("**ssma_turn數據源分布:**")
                        ssma_sources = ssma_turn_data['data_source'].value_counts()
                        for source, count in ssma_sources.items():
                            st.write(f"• {source}: {count} 個策略")
                        
                        # 檢查ssma_turn的欄位
                        st.write("**ssma_turn欄位檢查:**")
                        required_fields = ['total_return', 'sharpe_ratio', 'max_drawdown', '綜合評分']
                        for field in required_fields:
                            if field in ssma_turn_data.columns:
                                valid_count = ssma_turn_data[field].notna().sum()
                                st.write(f"• {field}: {valid_count}/{len(ssma_turn_data)} 有效值")
                            else:
                                st.write(f"• {field}: 欄位不存在")
                        
                        # 檢查ssma_turn的綜合評分
                        if '綜合評分' in ssma_turn_data.columns:
                            st.write("**ssma_turn綜合評分統計:**")
                            st.write(f"• 最小值: {ssma_turn_data['綜合評分'].min():.3f}")
                            st.write(f"• 最大值: {ssma_turn_data['綜合評分'].max():.3f}")
                            st.write(f"• 平均值: {ssma_turn_data['綜合評分'].mean():.3f}")
                    
                    # 為每個策略類型和數據源組合找到對應的策略
                    for group_key in strategy_groups:
                        strategy_type, data_source = group_key
                        
                        # 直接從optuna_results中篩選該組合的所有策略
                        mask = (optuna_results['strategy'] == strategy_type) & (optuna_results['data_source'] == data_source)
                        group_df = optuna_results[mask].copy()
                        
                        st.write(f"**調試 {strategy_type}_{data_source}:** 找到 {len(group_df)} 個策略")
                        
                        # 特別調試ssma_turn的篩選
                        if 'ssma_turn' in strategy_type:
                            st.write(f"  **ssma_turn調試:**")
                            st.write(f"  - 篩選條件: strategy == '{strategy_type}' AND data_source == '{data_source}'")
                            
                            # 檢查strategy欄位的唯一值
                            unique_strategies = optuna_results['strategy'].unique()
                            st.write(f"  - optuna_results中strategy欄位的唯一值: {list(unique_strategies)}")
                            
                            # 檢查data_source欄位的唯一值
                            unique_sources = optuna_results['data_source'].unique()
                            st.write(f"  - optuna_results中data_source欄位的唯一值: {list(unique_sources)}")
                            
                            # 分別檢查兩個條件
                            strategy_mask = optuna_results['strategy'] == strategy_type
                            source_mask = optuna_results['data_source'] == data_source
                            st.write(f"  - strategy == '{strategy_type}' 的結果: {strategy_mask.sum()} 個")
                            st.write(f"  - data_source == '{data_source}' 的結果: {source_mask.sum()} 個")
                            st.write(f"  - 兩個條件AND的結果: {(strategy_mask & source_mask).sum()} 個")
                        
                        if not group_df.empty:
                            # 檢查是否有fine_grained_cluster欄位
                            if 'fine_grained_cluster' in group_df.columns:
                                st.write(f"  **使用fine_grained_cluster進行多樣性分組**")
                                
                                # 轉換為字典格式供pick_topN_by_diversity使用
                                trials_dict = group_df.to_dict('records')
                                
                                # 使用pick_topN_by_diversity進行多樣性篩選
                                diverse_trials = pick_topN_by_diversity(
                                    trials_dict, 
                                    metric_keys=['num_trades'],  # 使用基本指標
                                    top_n=top_n
                                )
                                
                                # 轉換回DataFrame
                                if diverse_trials:
                                    top_group_strategies = pd.DataFrame(diverse_trials)
                                    st.write(f"  → 使用fine_grained_cluster選取前 {len(top_group_strategies)} 名")
                                else:
                                    # 如果多樣性篩選失敗，使用基本排序
                                    top_group_strategies = group_df.sort_values('綜合評分', ascending=False).head(top_n)
                                    st.write(f"  → 多樣性篩選失敗，使用基本排序選取前 {len(top_group_strategies)} 名")
                            else:
                                # 沒有fine_grained_cluster欄位，使用基本排序
                                top_group_strategies = group_df.sort_values('綜合評分', ascending=False).head(top_n)
                                st.write(f"  → 無fine_grained_cluster欄位，使用基本排序選取前 {len(top_group_strategies)} 名")
                            
                            top_strategies_by_group[group_key] = top_group_strategies
                        else:
                            st.write(f"  → 沒有找到策略")

                    
                    st.info(f"顯示各策略類型和數據源組合前 {top_n} 名的Optuna策略參數")
                    
                    # 顯示策略分布調試信息
                    st.subheader("📊 策略分布調試信息")
                    strategy_counts = optuna_results['strategy'].value_counts()
                    st.write("**策略類型分布:**")
                    for strategy, count in strategy_counts.items():
                        st.write(f"• {strategy}: {count} 個策略")
                    
                    data_source_counts = optuna_results['data_source'].value_counts()
                    st.write("**數據源分布:**")
                    for data_source, count in data_source_counts.items():
                        st.write(f"• {data_source}: {count} 個策略")
                    
                    # 顯示策略+數據源組合分布
                    st.write("**策略+數據源組合分布:**")
                    group_counts = optuna_results.groupby(['strategy', 'data_source']).size()
                    for (strategy, data_source), count in group_counts.items():
                        st.write(f"• {strategy}_{data_source}: {count} 個策略")
                    
                    # 顯示各策略類型和數據源組合的績效摘要
                    for group_key, top_group_strategies in top_strategies_by_group.items():
                        strategy_type, data_source = group_key
                        st.subheader(f"{strategy_type} - {data_source} 策略績效摘要 (前{top_n}名)")
                        
                        # 準備績效摘要數據
                        performance_summary = top_group_strategies[['total_return', 'sharpe_ratio', 'max_drawdown', '綜合評分']].copy()
                        performance_summary.columns = ['總報酬率', '夏普值', '最大回撤', '綜合評分']
                        
                        # 添加其他績效指標（如果存在）
                        if 'num_trades' in top_group_strategies.columns:
                            performance_summary['交易次數'] = top_group_strategies['num_trades']
                        if 'win_rate' in top_group_strategies.columns:
                            performance_summary['勝率'] = top_group_strategies['win_rate']
                        if 'profit_factor' in top_group_strategies.columns:
                            performance_summary['獲利因子'] = top_group_strategies['profit_factor']
                        
                        st.dataframe(performance_summary.style.format({
                            '總報酬率': '{:.2%}',
                            '夏普值': '{:.3f}',
                            '最大回撤': '{:.2%}',
                            '綜合評分': '{:.3f}',
                            '勝率': '{:.2%}',
                            '獲利因子': '{:.3f}'
                        }))
                    
                    # 生成參數字典格式
                    st.subheader("參數字典格式（可複製到SSS的param_presets）")
                    
                    # 創建參數字典
                    param_dict = {}
                    
                    # 遍歷所有策略類型和數據源組合
                    for group_key, top_group_strategies in top_strategies_by_group.items():
                        for _, row in top_group_strategies.iterrows():
                            # 獲取策略信息
                            strategy_type = row['strategy']
                            data_source = row['data_source']
                            trial_number = row['trial_number']
                            
                            # 生成策略名稱
                            strategy_name = f"{strategy_type}_{data_source}_{trial_number}"
                            
                            # 獲取參數
                            params = row['parameters']
                            
                            # 處理參數格式
                            if isinstance(params, dict) and 'parameters' in params:
                                actual_params = params['parameters']
                            elif isinstance(params, str):
                                try:
                                    import json
                                    actual_params = json.loads(params)
                                except json.JSONDecodeError:
                                    actual_params = {}
                            else:
                                actual_params = params
                            
                            # 添加策略類型和数据源
                            if isinstance(actual_params, dict):
                                actual_params['strategy_type'] = strategy_type
                                actual_params['smaa_source'] = data_source
                                
                                # 生成策略名稱（包含策略類型和數據源信息）
                                clean_name = f"{strategy_type}_{strategy_name.replace('_', '').replace('-', '')}"
                                param_dict[clean_name] = actual_params
                    
                    # 顯示參數字典
                    if param_dict:
                        # 格式化為Python字典格式
                        param_str = "param_presets = {\n"
                        for name, params in param_dict.items():
                            param_str += f'    "{name}": {params},\n'
                        param_str += "}"
                        
                        # 顯示可複製的代碼
                        st.code(param_str, language='python')
                        
                        # 提供下載按鈕
                        st.download_button(
                            label="📥 下載參數字典",
                            data=param_str,
                            file_name=f"top_{top_n}_optuna_strategies_params.py",
                            mime="text/plain"
                        )
                        
                        # 顯示參數詳情表格（按策略類型和數據源分組）
                        st.subheader("參數詳情表格（按策略類型和數據源分組）")
                        
                        # 為每個策略類型和數據源組合創建參數詳情表格
                        for group_key, top_group_strategies in top_strategies_by_group.items():
                            strategy_type, data_source = group_key
                            st.subheader(f"{strategy_type} - {data_source} 策略參數詳情")
                            

                            
                            # 創建該組合的參數詳情DataFrame
                            param_details = []
                            for _, row in top_group_strategies.iterrows():
                                # 獲取策略信息
                                strategy_type = row['strategy']
                                data_source = row['data_source']
                                trial_number = row['trial_number']
                                strategy_name = f"{strategy_type}_{data_source}_{trial_number}"
                                
                                # 獲取參數
                                params = row['parameters']
                                
                                # 處理參數格式
                                if isinstance(params, dict) and 'parameters' in params:
                                    actual_params = params['parameters']
                                elif isinstance(params, str):
                                    try:
                                        import json
                                        actual_params = json.loads(params)
                                    except json.JSONDecodeError:
                                        actual_params = {}
                                else:
                                    actual_params = params
                                
                                if isinstance(actual_params, dict):
                                    # 創建包含績效信息的行
                                    detail_row = {
                                        '策略名稱': strategy_name,
                                        '總報酬率': f"{row['total_return']:.2%}",
                                        '夏普值': f"{row['sharpe_ratio']:.3f}",
                                        '最大回撤': f"{row['max_drawdown']:.2%}",
                                        '綜合評分': f"{row['綜合評分']:.3f}"
                                    }
                                    
                                    # 添加其他績效指標（如果存在）
                                    if 'num_trades' in row and pd.notna(row['num_trades']):
                                        detail_row['交易次數'] = row['num_trades']
                                    if 'win_rate' in row and pd.notna(row['win_rate']):
                                        detail_row['勝率'] = f"{row['win_rate']:.2%}"
                                    if 'profit_factor' in row and pd.notna(row['profit_factor']):
                                        detail_row['獲利因子'] = f"{row['profit_factor']:.3f}"
                                    
                                    # 添加策略參數
                                    detail_row.update(actual_params)
                                    param_details.append(detail_row)
                            
                            if param_details:
                                param_df = pd.DataFrame(param_details)
                                st.dataframe(param_df, use_container_width=True)

                            else:
                                st.warning(f"無法生成 {strategy_type} - {data_source} 策略的參數詳情")
                    else:
                        st.warning("無法生成參數字典，請檢查Optuna策略配置")
                else:
                    st.warning("沒有可用的Optuna策略數據來生成參數詳情")




