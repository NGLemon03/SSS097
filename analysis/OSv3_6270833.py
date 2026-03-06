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
    基於性能指標的多樣性 top N 試驗選擇，應用四捨五入規則
    
    Args:
        trials: 試驗列表，每個試驗包含 score 和指定指標
        metric_keys: 用於分組的指標鍵
        top_n: 最終選取的試驗數量
    
    Returns:
        List: 篩選後的試驗列表
    """
    logger.info(f"開始多樣性篩選: {len(trials)} 個試驗, 目標選取 {top_n} 個")
    
    # 轉換為 DataFrame
    df = pd.DataFrame(trials)
    
    # 四捨五入規則
    round_rules = {
        'min_wf_return': 1,          # 小數點後一位
        'avg_stress_return': 2,      # 小數點後三位
        'stability_score': 1,        # 小數點後二位
        'robust_score': 2,           # 小數點後三位
        'excess_return_stress': 2,   # 小數點後三位
        'stress_mdd': 2,             # 小數點後三位
        'pbo_score': 2,              # 小數點後二位
        'sra_p_value': 2,            # 小數點後三位
        'avg_hold_days': 1           # 整數位
    }
    
    logger.info(f"四捨五入規則: {round_rules}")
    
    # 應用四捨五入
    for key in metric_keys:
        if key in round_rules:
            df[f'rounded_{key}'] = df[key].round(round_rules[key])
            logger.info(f"指標 {key}: 原始值範圍 [{df[key].min():.3f}, {df[key].max():.3f}], 四捨五入後範圍 [{df[f'rounded_{key}'].min():.3f}, {df[f'rounded_{key}'].max():.3f}]")
    
    # 按四捨五入後的指標分組
    group_cols = [f'rounded_{key}' for key in metric_keys]
    df['group'] = df[group_cols].astype(str).agg('_'.join, axis=1)
    
    # 統計分組情況
    group_counts = df['group'].value_counts()
    logger.info(f"分組統計: 共 {len(group_counts)} 個不同組別")
    logger.info(f"組別大小分布: {group_counts.describe()}")
    
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
        trial_num = row['trial_number']
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
    
    # 顯示最終選取的試驗信息
    logger.info("最終選取的試驗:")
    for i, trial in enumerate(chosen_trials):
        logger.info(f"  {i+1}. 試驗 {trial['trial_number']}: score={trial['score']:.3f}")
        # 顯示關鍵指標
        key_metrics = {k: trial.get(k, 'N/A') for k in ['min_wf_return', 'avg_stress_return', 'stability_score']}
        logger.info(f"     關鍵指標: {key_metrics}")
    
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
    計算過擬合指標
    """
    if len(train_returns) == 0 or len(test_returns) == 0:
        return {}
    
    # 計算樣本內外表現差異
    train_sharpe = train_returns.mean() / train_returns.std() if train_returns.std() > 0 else 0
    test_sharpe = test_returns.mean() / test_returns.std() if test_returns.std() > 0 else 0
    
    # 樣本內外夏普比率差異
    sharpe_degradation = train_sharpe - test_sharpe
    
    # 樣本內外報酬率差異
    return_degradation = train_returns.mean() - test_returns.mean()
    
    # 穩定性指標（變異係數）
    train_cv = train_returns.std() / abs(train_returns.mean()) if train_returns.mean() != 0 else float('inf')
    test_cv = test_returns.std() / abs(test_returns.mean()) if test_returns.mean() != 0 else float('inf')
    
    # 過擬合分數（0-100，越高越過擬合）
    # 修正計算方式：考慮夏普比率和報酬率的相對重要性
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

def calculate_strategy_stability(period_returns_dict):
    """
    計算策略穩定性指標
    """
    if not period_returns_dict:
        return {}
    
    returns_df = pd.DataFrame(period_returns_dict).T
    stability_metrics = {}
    
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        if len(returns) < 2:
            continue
            
        # 計算各期間表現的一致性（注意：period_returns已經是百分比）
        mean_return = returns.mean() / 100  # 轉換為小數
        std_return = returns.std() / 100    # 轉換為小數
        cv = std_return / abs(mean_return) if mean_return != 0 else float('inf')
        
        # 計算正報酬期間比例
        positive_periods = (returns > 0).sum() / len(returns)
        
        # 計算表現排名穩定性（如果有多個策略）
        if len(returns_df.columns) > 1:
            rank_stability = returns_df.corr().loc[col].mean()
        else:
            rank_stability = 1.0
            
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
    
    # 卡瑪比率（年化報酬率/最大回撤）
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

# 解析文件名以提取策略和數據源信息
def parse_optuna_filename(filename):
    """解析optuna結果文件名，提取策略和數據源信息"""
    name = Path(filename).stem  # 移除.csv後綴
    
    # 檢查是否是optuna_results文件
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
    elif name.startswith('Factor_TWII_2412_TW_'):
        data_source = 'Factor (^TWII / 2412.TW)'
    elif name.startswith('Factor_TWII_2414_TW_'):
        data_source = 'Factor (^TWII / 2414.TW)'
    
    if not data_source:
        return None
    
    # 提取時間戳（格式：20250623_040737）
    timestamp = None
    parts = name.split('_')
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit() and i + 1 < len(parts):
            next_part = parts[i + 1]
            if len(next_part) == 6 and next_part.isdigit():
                timestamp = f"{part}_{next_part}"
                break
    
    return {
        'strategy': strategy,
        'data_source': data_source,
        'timestamp': timestamp,
        'filename': filename
    }

# 載入所有optuna結果文件
RESULT_DIR = Path("../results")
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



# 載入所有optuna結果
all_optuna_results = []
for file_info in file_info_list:
    file_path = RESULT_DIR / file_info['filename']
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            df['parameters'] = df['parameters'].apply(ast.literal_eval)
            # 添加文件信息
            df['source_file'] = file_info['filename']
            df['strategy'] = file_info['strategy']
            df['data_source'] = file_info['data_source']
            all_optuna_results.append(df)
        except Exception as e:
            st.sidebar.error(f"載入失敗 {file_info['filename']}: {str(e)}")

if not all_optuna_results:
    st.error("沒有成功載入任何optuna結果文件")
    st.stop()

# 合併所有結果
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
        
        # 定義性能指標用於多樣性篩選
        metric_keys = [
            'min_wf_return', 'avg_stress_return', 'stability_score', 'robust_score',
            'excess_return_stress', 'stress_mdd', 'pbo_score', 'sra_p_value', 'avg_hold_days'
        ]
        
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
            top_n=3
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
            
            if strategy_type == 'single':
                df_ind = compute_single(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['devwin'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'dual':
                df_ind = compute_dual(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['short_win'], params['long_win'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'RMA':
                df_ind = compute_RMA(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'], smaa_source=smaa_source)
                logger.info(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'ssma_turn':
                calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
                ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                backtest_params = ssma_params.copy()
                backtest_params['buy_mult'] = params.get('buy_mult', 0.5)
                backtest_params['sell_mult'] = params.get('sell_mult', 0.5)
                backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
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
                result = backtest_unified(df_ind, strategy_type, params, buy_dates, sell_dates, 
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
            
            result = backtest_unified(df_ind, strategy_type, params, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
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
                
                # 修正勝率計算
                trades_df = result.get('trades_df', pd.DataFrame())
                logger.info(f"策略 {name} 的 trades_df 形狀: {trades_df.shape}")
                logger.info(f"策略 {name} 的 trades_df 欄位: {list(trades_df.columns)}")

                if not trades_df.empty and 'ret' in trades_df.columns:
                    winning_trades = trades_df[trades_df['ret'] > 0]
                    total_trades = len(trades_df)
                    winning_count = len(winning_trades)
                    win_rate = winning_count / total_trades if total_trades > 0 else 0
                    result['metrics']['win_rate'] = win_rate
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
                    result['metrics']['win_rate'] = 0
                    logger.warning(f"策略 {name} 的 trades_df 為空或缺少 'ret' 欄位")
                
                # 修正卡瑪比率計算
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
            tabs = st.tabs(["相關性矩陣熱圖", "報酬率相關", "最大回撤相關", "總結", "過擬合檢測"])

            # 標籤頁 1: 相關性矩陣熱圖
            with tabs[0]:

                
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
                st.subheader("整體最大回撤 (%)")
                st.dataframe(mdd_df.to_frame().style.format("{:.2f}%"))

            # 標籤頁 4: 總結
            with tabs[3]:
                st.subheader("各策略平均分數與避險次數")
                # 確保所有變數都是DataFrame或Series
                concat_list = []
                if not period_pr_df.empty:
                    concat_list.append(period_pr_df.mean())
                if not stress_pr_df.empty:
                    concat_list.append(stress_pr_df.mean())
                if not period_mdd_pr_df.empty:
                    concat_list.append(period_mdd_pr_df.mean())
                if not stress_mdd_pr_df.empty:
                    concat_list.append(stress_mdd_pr_df.mean())
                
                if concat_list:
                    avg_pr = pd.concat(concat_list, axis=1)
                    # 動態賦值列名
                    column_names = []
                    if not period_pr_df.empty:
                        column_names.append('走查報酬率分數')
                    if not stress_pr_df.empty:
                        column_names.append('壓力報酬率分數')
                    if not period_mdd_pr_df.empty:
                        column_names.append('走查MDD分數')
                    if not stress_mdd_pr_df.empty:
                        column_names.append('壓力MDD分數')
                    
                    avg_pr.columns = column_names
                    avg_pr['平均分數'] = avg_pr.mean(axis=1)
                else:
                    avg_pr = pd.DataFrame()
                
                wf_hedge_counts = pd.DataFrame(period_hedge_counts).T.sum()
                stress_hedge_counts = pd.DataFrame(stress_hedge_counts).T.sum()
                total_hedge_counts = wf_hedge_counts + stress_hedge_counts
                summary_df = pd.concat([avg_pr, wf_hedge_counts.to_frame('走查避險次數'), stress_hedge_counts.to_frame('壓力避險次數'), total_hedge_counts.to_frame('總避險次數')], axis=1)
                st.dataframe(summary_df.style.format("{:.2f}", subset=avg_pr.columns).format("{:d}", subset=['走查避險次數', '壓力避險次數', '總避險次數']))

            # 標籤頁 5: 過擬合檢測
            with tabs[4]:
                st.subheader("過擬合檢測分析")
                
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
                            
                            # 計算夏普比率
                            train_sharpe = train_annual_return / (train_returns.std() * np.sqrt(252)) if train_returns.std() > 0 else 0
                            test_sharpe = test_annual_return / (test_returns.std() * np.sqrt(252)) if test_returns.std() > 0 else 0
                            
                            # 計算過擬合指標
                            sharpe_degradation = train_sharpe - test_sharpe
                            return_degradation = train_annual_return - test_annual_return
                            
                            # 過擬合分數（0-100，越高越過擬合）
                            # 修正計算方式：考慮夏普比率和報酬率的相對重要性
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
                    st.dataframe(overfitting_df[['overfitting_score', 'sharpe_degradation', 'return_degradation']].style.format({
                        'overfitting_score': '{:.1f}',
                        'sharpe_degradation': '{:.3f}',
                        'return_degradation': '{:.3f}'
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
                    
                    # 夏普比率對比
                    comparison_sharpe_df = overfitting_df[['train_sharpe', 'test_sharpe']].reset_index()
                    comparison_sharpe_df.columns = ['策略', '樣本內夏普比率', '樣本外夏普比率']
                    
                    fig_comparison_sharpe = px.bar(
                        comparison_sharpe_df,
                        x='策略',
                        y=['樣本內夏普比率', '樣本外夏普比率'],
                        title="樣本內外夏普比率對比",
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
                stability_metrics = calculate_strategy_stability(period_returns)
                
                if stability_metrics:
                    stability_df = pd.DataFrame(stability_metrics).T
                    st.dataframe(stability_df.style.format({
                        'mean_return': '{:.2%}',
                        'std_return': '{:.2%}',
                        'cv': '{:.2f}',
                        'positive_periods_ratio': '{:.2%}',
                        'rank_stability': '{:.3f}'
                    }))
                    
                    # 穩定性熱力圖
                    if len(stability_df) > 1:
                        fig_stability = px.imshow(
                            stability_df[['cv', 'positive_periods_ratio', 'rank_stability']].T,
                            title="策略穩定性指標熱力圖",
                            aspect="auto"
                        )
                        st.plotly_chart(fig_stability, use_container_width=True)

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
                    
                    row = {
                        "策略": name,
                        "總報酬率": metrics.get('total_return', 0),
                        "年化報酬率": metrics.get('annual_return', 0),
                        "最大回撤": metrics.get('max_drawdown', 0),
                        "夏普比率": metrics.get('sharpe_ratio', 0),
                        "卡瑪比率": metrics.get('calmar_ratio', 0),
                        "交易次數": metrics.get('num_trades', 0),
                        "勝率": metrics.get('win_rate', 0)
                    }
                    summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data).set_index("策略")
                
                # 顯示表格
                st.dataframe(summary_df.style.format({
                    "總報酬率": "{:.2%}", "年化報酬率": "{:.2%}", "最大回撤": "{:.2%}",
                    "夏普比率": "{:.2f}", "卡瑪比率": "{:.2f}", "勝率": "{:.2%}"
                }))
            else:
                st.warning("沒有可用的匯總指標數據。")

            # 蒙地卡羅測試建議
            st.info("蒙地卡羅測試：請參考 Optuna_12.py 中的 compute_pbo_score 和 compute_simplified_sra 函數實現 PBO 分數與 SRA p 值計算")

            # 載入和執行信息
            with st.expander("📋 執行信息", expanded=False):
                st.subheader("載入信息")
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

            # 顯示日誌內容
            st.subheader("日誌內容")
            log_file = Path("logs") / "OS.log"
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    log_content = f.read()
                st.code(log_content, language="text")
            else:
                st.text("日誌檔案不存在")

