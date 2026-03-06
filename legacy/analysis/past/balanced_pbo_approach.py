"""
平衡的 PBO 方法：結合 Bailey 的嚴謹性和運算效率
目標：在運算成本和檢測準確性之間找到平衡
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

def balanced_pbo_analysis(
    trial_results: List[Dict],
    equity_curves: Dict[str, pd.Series],
    n_splits: int = 8,  # 減少分割數
    n_combinations: int = 50,  # 減少組合數
    use_tscv: bool = True  # 可選使用 TimeSeriesSplit
) -> Dict:
    """
    平衡的 PBO 分析：結合 Bailey 方法和運算效率
    
    Args:
        trial_results: 試驗結果列表
        equity_curves: 權益曲線字典
        n_splits: 時間序列分割數（減少運算量）
        n_combinations: 要測試的 IS/OOS 組合數（減少運算量）
        use_tscv: 是否使用 TimeSeriesSplit 作為備選
    
    Returns:
        Dict: PBO 分析結果
    """
    if len(trial_results) < 5:  # 降低最小試驗數要求
        logger.warning("試驗結果不足，使用簡化 PBO")
        return simplified_pbo_analysis(trial_results, equity_curves)
    
    # 1. 選擇代表性權益曲線
    sample_equity = None
    for trial_num, equity in equity_curves.items():
        if equity is not None and not equity.empty:
            sample_equity = equity
            break
    
    if sample_equity is None:
        logger.error("沒有有效的權益曲線")
        return {"pbo": 1.0, "method": "error", "n_combinations": 0}
    
    # 2. 生成時間期間
    if use_tscv and len(sample_equity) > 1000:
        # 使用 TimeSeriesSplit 生成期間（更高效）
        periods = generate_tscv_periods(sample_equity, n_splits)
        method = "tscv"
    else:
        # 使用 Bailey 方法生成期間
        periods = generate_balanced_periods(sample_equity, n_splits)
        method = "bailey"
    
    logger.info(f"使用 {method} 方法，生成 {len(periods)} 個期間")
    
    # 3. 生成 IS/OOS 組合
    n_periods = len(periods)
    n_is_periods = max(2, n_periods // 2)  # 確保至少有 2 個 IS 期間
    
    if method == "tscv":
        # TimeSeriesSplit 方式：順序性組合
        is_combinations = generate_tscv_combinations(n_periods, n_is_periods)
    else:
        # Bailey 方式：隨機組合
        is_combinations = generate_balanced_combinations(n_periods, n_is_periods, n_combinations)
    
    logger.info(f"測試 {len(is_combinations)} 個 IS/OOS 組合")
    
    # 4. 執行 PBO 分析
    rankings = []
    valid_combinations = 0
    
    for is_periods in is_combinations:
        # 確定 OOS 期間
        is_periods_set = set(is_periods)
        oos_periods = [i for i in range(n_periods) if i not in is_periods_set]
        
        # 計算每個試驗的表現
        trial_performances = calculate_trial_performances(
            trial_results, equity_curves, periods, is_periods, oos_periods
        )
        
        if len(trial_performances) < 3:  # 降低最小試驗數要求
            continue
        
        # 找到 IS 最佳參數並計算 OOS 排名
        ranking = calculate_ranking(trial_performances)
        if ranking:
            rankings.append(ranking)
            valid_combinations += 1
    
    # 5. 計算 PBO
    if not rankings:
        logger.warning("沒有有效的組合結果，使用簡化 PBO")
        return simplified_pbo_analysis(trial_results, equity_curves)
    
    pbo = calculate_final_pbo(rankings)
    
    result = {
        "pbo": pbo,
        "method": method,
        "n_combinations": valid_combinations,
        "rankings": rankings,
        "interpretation": get_pbo_interpretation(pbo)
    }
    
    logger.info(f"平衡 PBO 分析完成: PBO={pbo:.3f}, 方法={method}")
    return result

def generate_tscv_periods(equity_curve: pd.Series, n_splits: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """使用 TimeSeriesSplit 生成期間"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    periods = []
    
    for train_idx, test_idx in tscv.split(equity_curve):
        test_start = equity_curve.index[test_idx[0]]
        test_end = equity_curve.index[test_idx[-1]]
        periods.append((test_start, test_end))
    
    return periods

def generate_balanced_periods(equity_curve: pd.Series, n_splits: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """生成平衡的時間期間"""
    total_days = len(equity_curve)
    days_per_split = total_days // n_splits
    
    periods = []
    for i in range(n_splits):
        start_idx = i * days_per_split
        end_idx = (i + 1) * days_per_split if i < n_splits - 1 else total_days
        
        start_date = equity_curve.index[start_idx]
        end_date = equity_curve.index[end_idx - 1]
        periods.append((start_date, end_date))
    
    return periods

def generate_tscv_combinations(n_periods: int, n_is_periods: int) -> List[List[int]]:
    """生成 TimeSeriesSplit 風格的組合"""
    combinations_list = []
    
    # 生成順序性的 IS 組合
    for i in range(n_periods - n_is_periods + 1):
        is_periods = list(range(i, i + n_is_periods))
        combinations_list.append(is_periods)
    
    return combinations_list

def generate_balanced_combinations(n_periods: int, n_is_periods: int, n_combinations: int) -> List[List[int]]:
    """生成平衡的隨機組合"""
    all_combinations = list(combinations(range(n_periods), n_is_periods))
    
    if len(all_combinations) <= n_combinations:
        return [list(combo) for combo in all_combinations]
    
    # 隨機選擇組合
    selected_indices = np.random.choice(
        len(all_combinations), 
        n_combinations, 
        replace=False
    )
    
    return [list(all_combinations[i]) for i in selected_indices]

def calculate_trial_performances(
    trial_results: List[Dict],
    equity_curves: Dict[str, pd.Series],
    periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    is_periods: List[int],
    oos_periods: List[int]
) -> List[Dict]:
    """計算每個試驗在 IS 和 OOS 的表現"""
    trial_performances = []
    
    for trial in trial_results:
        trial_num = trial['trial_number']
        equity_curve = equity_curves.get(trial_num)
        
        if equity_curve is None or equity_curve.empty:
            continue
        
        # 計算 IS 期間的總報酬
        is_returns = []
        for period_idx in is_periods:
            start_date, end_date = periods[period_idx]
            period_return = calculate_period_return(equity_curve, start_date, end_date)
            if not np.isnan(period_return):
                is_returns.append(period_return)
        
        # 計算 OOS 期間的總報酬
        oos_returns = []
        for period_idx in oos_periods:
            start_date, end_date = periods[period_idx]
            period_return = calculate_period_return(equity_curve, start_date, end_date)
            if not np.isnan(period_return):
                oos_returns.append(period_return)
        
        if is_returns and oos_returns:
            is_total_return = np.prod([1 + r for r in is_returns]) - 1
            oos_total_return = np.prod([1 + r for r in oos_returns]) - 1
            
            trial_performances.append({
                'trial_number': trial_num,
                'is_return': is_total_return,
                'oos_return': oos_total_return
            })
    
    return trial_performances

def calculate_period_return(equity_curve: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """計算指定期間的報酬率"""
    if start_date not in equity_curve.index or end_date not in equity_curve.index:
        return np.nan
    
    start_value = equity_curve.loc[start_date]
    end_value = equity_curve.loc[end_date]
    
    if start_value == 0:
        return np.nan
    
    return (end_value / start_value) - 1

def calculate_ranking(trial_performances: List[Dict]) -> Optional[Dict]:
    """計算排名"""
    if len(trial_performances) < 2:
        return None
    
    # 找到 IS 期間表現最好的試驗
    best_trial = max(trial_performances, key=lambda x: x['is_return'])
    
    # 計算該試驗在 OOS 期間的排名
    oos_returns = [t['oos_return'] for t in trial_performances]
    oos_returns_sorted = sorted(oos_returns, reverse=True)
    
    best_oos_return = best_trial['oos_return']
    rank = oos_returns_sorted.index(best_oos_return) + 1
    percentile = rank / len(oos_returns_sorted)
    
    return {
        'best_trial': best_trial['trial_number'],
        'best_is_return': best_trial['is_return'],
        'best_oos_return': best_oos_return,
        'rank': rank,
        'percentile': percentile,
        'n_trials': len(trial_performances)
    }

def calculate_final_pbo(rankings: List[Dict]) -> float:
    """計算最終 PBO"""
    if not rankings:
        return 1.0
    
    # PBO = 排名在前 1/2 的比例
    pbo = sum(1 for r in rankings if r['percentile'] <= 0.5) / len(rankings)
    return pbo

def simplified_pbo_analysis(trial_results: List[Dict], equity_curves: Dict[str, pd.Series]) -> Dict:
    """簡化的 PBO 分析（當試驗數不足時使用）"""
    if not trial_results:
        return {"pbo": 1.0, "method": "simplified", "n_combinations": 0}
    
    # 計算所有試驗的報酬率
    returns = []
    for trial in trial_results:
        trial_num = trial['trial_number']
        equity_curve = equity_curves.get(trial_num)
        
        if equity_curve is not None and not equity_curve.empty:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            returns.append(total_return)
    
    if not returns:
        return {"pbo": 1.0, "method": "simplified", "n_combinations": 0}
    
    # 簡化的過擬合檢測：基於報酬率的變異性
    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    
    # 如果報酬率變異很大且均值很高，可能是過擬合
    if std_return > 0.1 and mean_return > 0.3:
        pbo = min(1.0, std_return / mean_return)
    else:
        pbo = 0.2  # 假設較低風險
    
    return {
        "pbo": pbo,
        "method": "simplified",
        "n_combinations": 1,
        "interpretation": get_pbo_interpretation(pbo)
    }

def get_pbo_interpretation(pbo: float) -> str:
    """根據 PBO 值提供解釋"""
    if pbo <= 0.1:
        return "優秀：過擬合風險很低，策略穩健性很高"
    elif pbo <= 0.2:
        return "良好：過擬合風險較低，策略較為穩健"
    elif pbo <= 0.3:
        return "一般：有一定過擬合風險，需要謹慎"
    elif pbo <= 0.5:
        return "較差：過擬合風險較高，建議重新設計策略"
    else:
        return "很差：過擬合風險很高，策略可能無效"

# 使用範例
if __name__ == "__main__":
    # 假設我們有試驗結果和權益曲線
    # trial_results = [...]  # 從 Optuna 結果載入
    # equity_curves = {...}  # 從快取載入
    
    # pbo_result = balanced_pbo_analysis(trial_results, equity_curves)
    # print(f"PBO: {pbo_result['pbo']:.3f}, 方法: {pbo_result['method']}")
    pass 