"""
Bailey 等人的 Probability of Backtest Overfitting (PBO) 完整實現
參考論文：Bailey, D.H., Borwein, J., Lopez de Prado, M. and Zhu, Q.J., 2016. 
The probability of backtest overfitting. Journal of Computational Finance, forthcoming.

核心思想：
1. 將時間序列切成 S 份，產生大量 IS/OOS 組合
2. 在每個 IS 中找到最佳參數
3. 計算該最佳參數在對應 OOS 中的排名
4. 統計排名在前 1/2 的比例，即為過擬合概率
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def generate_time_periods(equity_curve: pd.Series, n_splits: int = 10) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    將時間序列切成 n_splits 個等長期間
    
    Args:
        equity_curve: 權益曲線
        n_splits: 分割數
    
    Returns:
        List[Tuple]: 每個期間的 (開始時間, 結束時間)
    """
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

def calculate_period_return(equity_curve: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    計算指定期間的報酬率
    
    Args:
        equity_curve: 權益曲線
        start_date: 開始日期
        end_date: 結束日期
    
    Returns:
        float: 期間報酬率
    """
    if start_date not in equity_curve.index or end_date not in equity_curve.index:
        return np.nan
    
    start_value = equity_curve.loc[start_date]
    end_value = equity_curve.loc[end_date]
    
    if start_value == 0:
        return np.nan
    
    return (end_value / start_value) - 1

def bailey_pbo_analysis(
    trial_results: List[Dict],
    equity_curves: Dict[str, pd.Series],
    n_splits: int = 10,
    n_combinations: int = 100
) -> Dict:
    """
    執行 Bailey 等人的 PBO 分析
    
    Args:
        trial_results: 試驗結果列表，每個元素包含 'trial_number', 'parameters', 'score'
        equity_curves: 權益曲線字典，key 為 trial_number
        n_splits: 時間序列分割數
        n_combinations: 要測試的 IS/OOS 組合數
    
    Returns:
        Dict: PBO 分析結果
    """
    if len(trial_results) < 2:
        logger.warning("試驗結果不足，無法進行 PBO 分析")
        return {"pbo": 1.0, "n_combinations": 0, "rankings": []}
    
    # 1. 選擇一個代表性的權益曲線來生成時間期間
    # 使用第一個有效的權益曲線
    sample_equity = None
    for trial_num, equity in equity_curves.items():
        if equity is not None and not equity.empty:
            sample_equity = equity
            break
    
    if sample_equity is None:
        logger.error("沒有有效的權益曲線")
        return {"pbo": 1.0, "n_combinations": 0, "rankings": []}
    
    # 2. 生成時間期間
    periods = generate_time_periods(sample_equity, n_splits)
    logger.info(f"將時間序列分割為 {len(periods)} 個期間")
    
    # 3. 生成 IS/OOS 組合
    n_periods = len(periods)
    n_is_periods = n_periods // 2
    
    # 生成所有可能的 IS 組合
    all_is_combinations = list(combinations(range(n_periods), n_is_periods))
    
    # 隨機選擇 n_combinations 個組合
    if len(all_is_combinations) > n_combinations:
        selected_combinations = np.random.choice(
            len(all_is_combinations), 
            n_combinations, 
            replace=False
        )
        is_combinations = [all_is_combinations[i] for i in selected_combinations]
    else:
        is_combinations = all_is_combinations
    
    logger.info(f"測試 {len(is_combinations)} 個 IS/OOS 組合")
    
    # 4. 對每個組合進行分析
    rankings = []
    valid_combinations = 0
    
    for is_periods in is_combinations:
        # 確定 IS 和 OOS 期間
        is_periods_set = set(is_periods)
        oos_periods = [i for i in range(n_periods) if i not in is_periods_set]
        
        # 計算每個試驗在 IS 和 OOS 的表現
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
        
        if len(trial_performances) < 2:
            continue
        
        # 找到 IS 期間表現最好的試驗
        best_trial = max(trial_performances, key=lambda x: x['is_return'])
        
        # 計算該試驗在 OOS 期間的排名
        oos_returns = [t['oos_return'] for t in trial_performances]
        oos_returns_sorted = sorted(oos_returns, reverse=True)
        
        best_oos_return = best_trial['oos_return']
        rank = oos_returns_sorted.index(best_oos_return) + 1
        percentile = rank / len(oos_returns_sorted)
        
        rankings.append({
            'is_periods': is_periods,
            'oos_periods': oos_periods,
            'best_trial': best_trial['trial_number'],
            'best_is_return': best_trial['is_return'],
            'best_oos_return': best_oos_return,
            'rank': rank,
            'percentile': percentile,
            'n_trials': len(trial_performances)
        })
        
        valid_combinations += 1
    
    # 5. 計算 PBO
    if not rankings:
        logger.warning("沒有有效的組合結果")
        return {"pbo": 1.0, "n_combinations": 0, "rankings": []}
    
    # PBO = 排名在前 1/2 的比例
    pbo = sum(1 for r in rankings if r['percentile'] <= 0.5) / len(rankings)
    
    # 6. 統計分析
    percentiles = [r['percentile'] for r in rankings]
    mean_percentile = np.mean(percentiles)
    std_percentile = np.std(percentiles)
    
    # 計算排名分佈
    rank_distribution = {}
    for r in rankings:
        rank_bin = f"{(r['rank']-1)//5*5+1}-{((r['rank']-1)//5+1)*5}"
        rank_distribution[rank_bin] = rank_distribution.get(rank_bin, 0) + 1
    
    result = {
        "pbo": pbo,
        "n_combinations": valid_combinations,
        "mean_percentile": mean_percentile,
        "std_percentile": std_percentile,
        "rank_distribution": rank_distribution,
        "rankings": rankings,
        "interpretation": get_pbo_interpretation(pbo)
    }
    
    logger.info(f"PBO 分析完成: PBO={pbo:.3f}, 平均百分位={mean_percentile:.3f}")
    return result

def get_pbo_interpretation(pbo: float) -> str:
    """
    根據 PBO 值提供解釋
    
    Args:
        pbo: PBO 值 (0-1)
    
    Returns:
        str: 解釋文字
    """
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

def plot_pbo_results(pbo_result: Dict, save_path: Optional[str] = None):
    """
    繪製 PBO 分析結果
    
    Args:
        pbo_result: PBO 分析結果
        save_path: 保存路徑（可選）
    """
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 排名分佈直方圖
    rankings = pbo_result['rankings']
    percentiles = [r['percentile'] for r in rankings]
    
    ax1.hist(percentiles, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(0.5, color='red', linestyle='--', label='中位數')
    ax1.set_xlabel('OOS 排名百分位')
    ax1.set_ylabel('頻次')
    ax1.set_title(f'PBO 排名分佈 (PBO = {pbo_result["pbo"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. IS vs OOS 報酬散點圖
    is_returns = [r['best_is_return'] for r in rankings]
    oos_returns = [r['best_oos_return'] for r in rankings]
    
    ax2.scatter(is_returns, oos_returns, alpha=0.6, color='green')
    ax2.plot([min(is_returns), max(is_returns)], [min(is_returns), max(is_returns)], 
             'r--', alpha=0.5, label='y=x')
    ax2.set_xlabel('IS 期間最佳報酬')
    ax2.set_ylabel('OOS 期間報酬')
    ax2.set_title('IS vs OOS 報酬關係')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 排名分佈條形圖
    rank_dist = pbo_result['rank_distribution']
    ranks = sorted(rank_dist.keys(), key=lambda x: int(x.split('-')[0]))
    counts = [rank_dist[r] for r in ranks]
    
    ax3.bar(ranks, counts, alpha=0.7, color='orange')
    ax3.set_xlabel('排名區間')
    ax3.set_ylabel('頻次')
    ax3.set_title('OOS 排名分佈')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 統計摘要
    ax4.axis('off')
    summary_text = f"""
    PBO 分析摘要
    
    PBO 值: {pbo_result['pbo']:.3f}
    有效組合數: {pbo_result['n_combinations']}
    平均百分位: {pbo_result['mean_percentile']:.3f}
    百分位標準差: {pbo_result['std_percentile']:.3f}
    
    解釋: {pbo_result['interpretation']}
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PBO 分析圖表已保存至 {save_path}")
    
    plt.show()

# 使用範例
if __name__ == "__main__":
    # 假設我們有試驗結果和權益曲線
    # trial_results = [...]  # 從 Optuna 結果載入
    # equity_curves = {...}  # 從快取載入
    
    # pbo_result = bailey_pbo_analysis(trial_results, equity_curves)
    # plot_pbo_results(pbo_result, "pbo_analysis.png")
    pass 