import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_param_cluster_distributions():
    """分析 param_cluster 的分布圖，包含堆疊效果"""
    
    print("=== Param Cluster 分布分析 ===")
    
    # 讀取整合分析結果
    df = pd.read_csv('analysis/results/op16/integrated_analysis_results.csv')
    
    # 只保留有 param_cluster 的資料
    df = df[df['param_cluster'].notna()]
    df['param_cluster'] = df['param_cluster'].astype(int)
    
    print(f"總共 {len(df)} 個 trials")
    print(f"Param clusters: {sorted(df['param_cluster'].unique())}")
    
    # 創建輸出目錄
    output_dir = 'analysis/results/op16/param_cluster_distributions'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. total_return 分布（堆疊直方圖）
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='total_return', hue='param_cluster', 
                element='step', stat='count', common_norm=False, 
                bins=30, multiple='stack', alpha=0.7)
    plt.title('Total Return 分布（依 Param Cluster 堆疊）', fontsize=14)
    plt.xlabel('Total Return', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Param Cluster')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_return_distribution_stacked.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. score 分布（堆疊直方圖）
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='score', hue='param_cluster', 
                element='step', stat='count', common_norm=False, 
                bins=30, multiple='stack', alpha=0.7)
    plt.title('Score 分布（依 Param Cluster 堆疊）', fontsize=14)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Param Cluster')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distribution_stacked.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. 各參數分布（堆疊直方圖）
    param_cols = [col for col in df.columns if col.startswith('param_') and col != 'param_cluster']
    
    print(f"找到 {len(param_cols)} 個參數: {param_cols}")
    
    for param in param_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=param, hue='param_cluster', 
                    element='step', stat='count', common_norm=False, 
                    bins=20, multiple='stack', alpha=0.7)
        plt.title(f'{param} 分布（依 Param Cluster 堆疊）', fontsize=14)
        plt.xlabel(param, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Param Cluster')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{param}_distribution_stacked.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 4. Boxplot 比較（非堆疊，但可看出分布差異）
    # total_return boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='param_cluster', y='total_return')
    plt.title('Total Return 分布（Boxplot, 依 Param Cluster 分組）', fontsize=14)
    plt.xlabel('Param Cluster', fontsize=12)
    plt.ylabel('Total Return', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_return_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # score boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='param_cluster', y='score')
    plt.title('Score 分布（Boxplot, 依 Param Cluster 分組）', fontsize=14)
    plt.xlabel('Param Cluster', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. 各參數 boxplot
    for param in param_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='param_cluster', y=param)
        plt.title(f'{param} 分布（Boxplot, 依 Param Cluster 分組）', fontsize=14)
        plt.xlabel('Param Cluster', fontsize=12)
        plt.ylabel(param, fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{param}_boxplot.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 6. Violin plot（更詳細的分布形狀）
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='param_cluster', y='total_return')
    plt.title('Total Return 分布（Violin Plot, 依 Param Cluster 分組）', fontsize=14)
    plt.xlabel('Param Cluster', fontsize=12)
    plt.ylabel('Total Return', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_return_violin.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 7. 統計摘要
    print("\n=== 各 Param Cluster 統計摘要 ===")
    
    # total_return 統計
    print("\nTotal Return 統計:")
    return_stats = df.groupby('param_cluster')['total_return'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    print(return_stats)
    
    # score 統計
    print("\nScore 統計:")
    score_stats = df.groupby('param_cluster')['score'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    print(score_stats)
    
    # 各參數統計
    for param in param_cols:
        print(f"\n{param} 統計:")
        param_stats = df.groupby('param_cluster')[param].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
        print(param_stats)
    
    # 8. 輸出統計摘要到 CSV
    stats_summary = []
    for param in ['total_return', 'score'] + param_cols:
        stats = df.groupby('param_cluster')[param].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
        stats['parameter'] = param
        stats = stats.reset_index()
        stats_summary.append(stats)
    
    stats_df = pd.concat(stats_summary, ignore_index=True)
    stats_df.to_csv(f'{output_dir}/param_cluster_statistics.csv', index=False)
    print(f"\n統計摘要已輸出到: {output_dir}/param_cluster_statistics.csv")
    
    print(f"\n所有圖表已保存到: {output_dir}/")
    
    return df

if __name__ == "__main__":
    analyze_param_cluster_distributions() 