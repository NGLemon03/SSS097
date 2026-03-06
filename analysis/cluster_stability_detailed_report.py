import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 載入比較結果
comparison_df = pd.read_csv('results/hierarchical_cluster_analysis/cluster_quality_comparison.csv')

def analyze_strategy_stability():
    """分析不同策略的叢集穩定性"""
    print("=== 策略叢集穩定性分析 ===")
    
    # 按策略分組分析
    strategy_stats = comparison_df.groupby('strategy').agg({
        'silhouette_score': ['mean', 'std', 'min', 'max'],
        'calinski_harabasz_score': ['mean', 'std', 'min', 'max'],
        'davies_bouldin_score': ['mean', 'std', 'min', 'max'],
        'best_k': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print("\n策略叢集品質統計:")
    print(strategy_stats)
    
    # 策略間差異檢定
    strategies = comparison_df['strategy'].unique()
    
    print("\n策略間差異統計檢定:")
    for metric in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']:
        groups = [comparison_df[comparison_df['strategy'] == s][metric].values for s in strategies]
        
        # ANOVA檢定
        try:
            f_stat, p_value = f_oneway(*groups)
            print(f"{metric} - ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
        except:
            print(f"{metric} - ANOVA: 無法計算")
        
        # Kruskal-Wallis檢定
        try:
            h_stat, h_p_value = kruskal(*groups)
            print(f"{metric} - Kruskal-Wallis: H={h_stat:.3f}, p={h_p_value:.3f}")
        except:
            print(f"{metric} - Kruskal-Wallis: 無法計算")
    
    return strategy_stats

def analyze_datasource_stability():
    """分析不同數據源的叢集穩定性"""
    print("\n=== 數據源叢集穩定性分析 ===")
    
    # 按數據源分組分析
    datasource_stats = comparison_df.groupby('datasource').agg({
        'silhouette_score': ['mean', 'std', 'min', 'max'],
        'calinski_harabasz_score': ['mean', 'std', 'min', 'max'],
        'davies_bouldin_score': ['mean', 'std', 'min', 'max'],
        'best_k': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print("\n數據源叢集品質統計:")
    print(datasource_stats)
    
    # 數據源間差異檢定
    datasources = comparison_df['datasource'].unique()
    
    print("\n數據源間差異統計檢定:")
    for metric in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']:
        groups = [comparison_df[comparison_df['datasource'] == d][metric].values for d in datasources]
        
        # ANOVA檢定
        try:
            f_stat, p_value = f_oneway(*groups)
            print(f"{metric} - ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
        except:
            print(f"{metric} - ANOVA: 無法計算")
        
        # Kruskal-Wallis檢定
        try:
            h_stat, h_p_value = kruskal(*groups)
            print(f"{metric} - Kruskal-Wallis: H={h_stat:.3f}, p={h_p_value:.3f}")
        except:
            print(f"{metric} - Kruskal-Wallis: 無法計算")
    
    return datasource_stats

def create_stability_visualizations():
    """創建穩定性分析可視化"""
    OUTPUT_DIR = 'results/hierarchical_cluster_analysis'
    
    # 1. 策略穩定性比較
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Silhouette Score by Strategy
    ax1 = axes[0, 0]
    sns.boxplot(data=comparison_df, x='strategy', y='silhouette_score', ax=ax1)
    ax1.set_title('Silhouette Score by Strategy', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylabel('Silhouette Score')
    
    # Calinski-Harabasz Score by Strategy
    ax2 = axes[0, 1]
    sns.boxplot(data=comparison_df, x='strategy', y='calinski_harabasz_score', ax=ax2)
    ax2.set_title('Calinski-Harabasz Score by Strategy', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylabel('Calinski-Harabasz Score')
    
    # Davies-Bouldin Score by Strategy
    ax3 = axes[0, 2]
    sns.boxplot(data=comparison_df, x='strategy', y='davies_bouldin_score', ax=ax3)
    ax3.set_title('Davies-Bouldin Score by Strategy', fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_ylabel('Davies-Bouldin Score')
    
    # Silhouette Score by Datasource
    ax4 = axes[1, 0]
    sns.boxplot(data=comparison_df, x='datasource', y='silhouette_score', ax=ax4)
    ax4.set_title('Silhouette Score by Datasource', fontsize=12)
    ax4.set_ylabel('Silhouette Score')
    
    # Calinski-Harabasz Score by Datasource
    ax5 = axes[1, 1]
    sns.boxplot(data=comparison_df, x='datasource', y='calinski_harabasz_score', ax=ax5)
    ax5.set_title('Calinski-Harabasz Score by Datasource', fontsize=12)
    ax5.set_ylabel('Calinski-Harabasz Score')
    
    # Davies-Bouldin Score by Datasource
    ax6 = axes[1, 2]
    sns.boxplot(data=comparison_df, x='datasource', y='davies_bouldin_score', ax=ax6)
    ax6.set_title('Davies-Bouldin Score by Datasource', fontsize=12)
    ax6.set_ylabel('Davies-Bouldin Score')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stability_comparison_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 熱力圖顯示策略-數據源組合的穩定性
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Silhouette Score 熱力圖
    pivot_sil = comparison_df.pivot(index='strategy', columns='datasource', values='silhouette_score')
    sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0])
    axes[0].set_title('Silhouette Score Heatmap', fontsize=12)
    
    # Calinski-Harabasz Score 熱力圖
    pivot_cal = comparison_df.pivot(index='strategy', columns='datasource', values='calinski_harabasz_score')
    sns.heatmap(pivot_cal, annot=True, fmt='.0f', cmap='RdYlBu_r', ax=axes[1])
    axes[1].set_title('Calinski-Harabasz Score Heatmap', fontsize=12)
    
    # Davies-Bouldin Score 熱力圖
    pivot_db = comparison_df.pivot(index='strategy', columns='datasource', values='davies_bouldin_score')
    sns.heatmap(pivot_db, annot=True, fmt='.3f', cmap='RdYlBu', ax=axes[2])  # 反轉顏色映射
    axes[2].set_title('Davies-Bouldin Score Heatmap', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stability_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 聚類方法穩定性分析
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 按聚類方法分組
    method_stats = comparison_df.groupby('best_method').agg({
        'silhouette_score': 'mean',
        'calinski_harabasz_score': 'mean',
        'davies_bouldin_score': 'mean'
    })
    
    # Silhouette Score by Method
    ax1 = axes[0]
    method_stats['silhouette_score'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Average Silhouette Score by Clustering Method')
    ax1.set_ylabel('Silhouette Score')
    ax1.tick_params(axis='x', rotation=0)
    
    # Calinski-Harabasz Score by Method
    ax2 = axes[1]
    method_stats['calinski_harabasz_score'].plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Average Calinski-Harabasz Score by Clustering Method')
    ax2.set_ylabel('Calinski-Harabasz Score')
    ax2.tick_params(axis='x', rotation=0)
    
    # Davies-Bouldin Score by Method
    ax3 = axes[2]
    method_stats['davies_bouldin_score'].plot(kind='bar', ax=ax3, color='salmon')
    ax3.set_title('Average Davies-Bouldin Score by Clustering Method')
    ax3.set_ylabel('Davies-Bouldin Score')
    ax3.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/clustering_method_stability.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_stability_ranking():
    """創建穩定性排名分析"""
    print("\n=== 叢集穩定性排名 ===")
    
    # 計算綜合穩定性分數 (標準化後加權平均)
    scaler = StandardScaler()
    
    # 標準化各指標
    sil_scaled = scaler.fit_transform(comparison_df[['silhouette_score']].values.reshape(-1, 1)).flatten()
    cal_scaled = scaler.fit_transform(comparison_df[['calinski_harabasz_score']].values.reshape(-1, 1)).flatten()
    db_scaled = scaler.fit_transform(comparison_df[['davies_bouldin_score']].values.reshape(-1, 1)).flatten()
    
    # Davies-Bouldin 分數需要反轉（越低越好）
    db_scaled = -db_scaled
    
    # 計算綜合分數 (等權重)
    stability_score = (sil_scaled + cal_scaled + db_scaled) / 3
    
    # 添加到DataFrame
    comparison_df['stability_score'] = stability_score
    
    # 排名
    ranking_df = comparison_df[['strategy', 'datasource', 'stability_score', 'silhouette_score', 
                               'calinski_harabasz_score', 'davies_bouldin_score', 'best_k', 'best_method']].copy()
    ranking_df = ranking_df.sort_values('stability_score', ascending=False)
    
    print("\n叢集穩定性排名 (由高到低):")
    for i, (_, row) in enumerate(ranking_df.iterrows(), 1):
        print(f"{i:2d}. {row['strategy']:25s} - {row['datasource']:8s} | "
              f"Stability: {row['stability_score']:6.3f} | "
              f"Sil: {row['silhouette_score']:5.3f} | "
              f"Cal: {row['calinski_harabasz_score']:6.0f} | "
              f"DB: {row['davies_bouldin_score']:5.3f} | "
              f"k={row['best_k']} ({row['best_method']})")
    
    # 保存排名結果
    ranking_df.to_csv('results/hierarchical_cluster_analysis/stability_ranking.csv', index=False)
    
    return ranking_df

def generate_stability_report():
    """生成完整的穩定性分析報告"""
    print("開始生成叢集穩定性詳細分析報告...")
    
    # 執行各項分析
    strategy_stats = analyze_strategy_stability()
    datasource_stats = analyze_datasource_stability()
    ranking_df = create_stability_ranking()
    
    # 創建可視化
    create_stability_visualizations()
    
    # 生成文字報告
    report = []
    report.append("=== Hierarchical 叢集穩定性分析報告 ===\n")
    
    report.append("1. 整體統計:")
    report.append(f"   - 總分析組合數: {len(comparison_df)}")
    report.append(f"   - 平均 Silhouette Score: {comparison_df['silhouette_score'].mean():.3f}")
    report.append(f"   - 平均 Calinski-Harabasz Score: {comparison_df['calinski_harabasz_score'].mean():.1f}")
    report.append(f"   - 平均 Davies-Bouldin Score: {comparison_df['davies_bouldin_score'].mean():.3f}")
    report.append(f"   - 最佳聚類數分布: {dict(comparison_df['best_k'].value_counts())}")
    report.append(f"   - 聚類方法分布: {dict(comparison_df['best_method'].value_counts())}\n")
    
    report.append("2. 策略穩定性排名 (前5名):")
    for i, (_, row) in enumerate(ranking_df.head().iterrows(), 1):
        report.append(f"   {i}. {row['strategy']} - {row['datasource']} (穩定性分數: {row['stability_score']:.3f})")
    
    report.append("\n3. 數據源穩定性比較:")
    for datasource in comparison_df['datasource'].unique():
        ds_data = comparison_df[comparison_df['datasource'] == datasource]
        report.append(f"   {datasource}: 平均 Silhouette={ds_data['silhouette_score'].mean():.3f}, "
                     f"平均 Calinski-Harabasz={ds_data['calinski_harabasz_score'].mean():.1f}")
    
    report.append("\n4. 聚類方法穩定性:")
    for method in comparison_df['best_method'].unique():
        method_data = comparison_df[comparison_df['best_method'] == method]
        report.append(f"   {method}: 平均 Silhouette={method_data['silhouette_score'].mean():.3f}, "
                     f"使用次數={len(method_data)}")
    
    # 保存報告
    with open('results/hierarchical_cluster_analysis/stability_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("報告已生成完成！")
    print('\n'.join(report))

if __name__ == "__main__":
    generate_stability_report() 