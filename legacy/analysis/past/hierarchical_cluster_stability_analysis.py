import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, f_oneway, kruskal
import os
import glob
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import json
from collections import defaultdict

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 設定路徑
HIERARCHICAL_DIR = 'results/op16'
OUTPUT_DIR = 'results/hierarchical_cluster_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_datasource_from_filename(filename):
    """從檔案名提取數據源"""
    fname = filename.upper().replace(' ', '').replace('(', '').replace(')', '').replace('^', '').replace('/', '').replace('.', '').replace('_', '')
    if 'SELF' in fname:
        return 'Self'
    if '2412' in fname:
        return '2412'
    if '2414' in fname:
        return '2414'
    return None

def extract_strategy_from_filename(filename):
    """從檔案名提取策略名稱"""
    file_pattern = re.compile(r'optuna_results_([a-zA-Z0-9_]+)_([^_]+)')
    m = file_pattern.search(filename)
    if m:
        return m.group(1)
    return None

def load_and_preprocess_data():
    """載入並預處理所有數據"""
    all_files = glob.glob(f'{HIERARCHICAL_DIR}/*.csv')
    data_dict = {}
    
    for f in all_files:
        fname = os.path.basename(f)
        strategy = extract_strategy_from_filename(fname)
        datasource = extract_datasource_from_filename(fname)
        
        if strategy and datasource:
            df = pd.read_csv(f)
            
            # 自動展開 parameters 欄位
            if 'parameters' in df.columns:
                param_df = df['parameters'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
                param_df = pd.json_normalize(param_df.tolist())
                param_df = param_df.add_prefix('param_')
                df = pd.concat([df, param_df], axis=1)
            
            key = (strategy, datasource)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(df)
    
    # 合併每個策略-數據源組合的所有檔案
    merged_data = {}
    for key, dfs in data_dict.items():
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_data[key] = merged_df
    
    return merged_data

def perform_clustering_analysis(df, strategy, datasource):
    """對單個數據集進行聚類分析"""
    param_cols = [col for col in df.columns if col.startswith('param_')]
    
    if len(param_cols) < 2 or len(df) < 10:
        return None
    
    # 準備數據
    X = df[param_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降維
    pca = PCA(n_components=min(2, len(param_cols)))
    X_pca = pca.fit_transform(X_scaled)
    
    # 自動決定最佳分群數
    best_k = 2
    best_score = -1
    best_method = 'kmeans'
    
    # 測試不同的聚類方法
    methods = {
        'kmeans': KMeans,
        'hierarchical': AgglomerativeClustering
    }
    
    for method_name, method_class in methods.items():
        for k in range(2, min(6, len(df))):
            try:
                if method_name == 'kmeans':
                    clusterer = method_class(n_clusters=k, random_state=42)
                else:
                    clusterer = method_class(n_clusters=k)
                
                labels = clusterer.fit_predict(X_pca)
                
                # 計算聚類品質指標
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(X_pca, labels)
                    if sil_score > best_score:
                        best_score = sil_score
                        best_k = k
                        best_method = method_name
            except:
                continue
    
    # 使用最佳參數進行最終聚類
    if best_method == 'kmeans':
        final_clusterer = KMeans(n_clusters=best_k, random_state=42)
    else:
        final_clusterer = AgglomerativeClustering(n_clusters=best_k)
    
    final_labels = final_clusterer.fit_predict(X_pca)
    
    # 計算聚類品質指標
    sil_score = silhouette_score(X_pca, final_labels)
    cal_score = calinski_harabasz_score(X_pca, final_labels)
    db_score = davies_bouldin_score(X_pca, final_labels)
    
    return {
        'labels': final_labels,
        'X_pca': X_pca,
        'best_k': best_k,
        'best_method': best_method,
        'silhouette_score': sil_score,
        'calinski_harabasz_score': cal_score,
        'davies_bouldin_score': db_score,
        'param_cols': param_cols
    }

def analyze_cluster_performance(df, cluster_labels, strategy, datasource):
    """分析各聚類的性能差異"""
    performance_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    results = {}
    
    for metric in performance_metrics:
        if metric not in df.columns:
            continue
        
        # 過濾有效數據
        mask = (~df[metric].isna()) & np.isfinite(df[metric])
        if mask.sum() < 10:
            continue
        
        metric_data = df.loc[mask, metric]
        cluster_data = cluster_labels[mask]
        
        # 計算各聚類的統計量
        cluster_stats = {}
        for cluster_id in np.unique(cluster_data):
            cluster_mask = cluster_data == cluster_id
            if cluster_mask.sum() > 0:
                cluster_stats[cluster_id] = {
                    'mean': metric_data[cluster_mask].mean(),
                    'std': metric_data[cluster_mask].std(),
                    'count': cluster_mask.sum(),
                    'median': metric_data[cluster_mask].median()
                }
        
        # 進行統計檢定
        cluster_groups = [metric_data[cluster_data == c] for c in np.unique(cluster_data) if (cluster_data == c).sum() > 0]
        
        if len(cluster_groups) > 1:
            try:
                f_stat, p_value = f_oneway(*cluster_groups)
                h_stat, h_p_value = kruskal(*cluster_groups)
            except:
                f_stat, p_value = np.nan, np.nan
                h_stat, h_p_value = np.nan, np.nan
        else:
            f_stat, p_value = np.nan, np.nan
            h_stat, h_p_value = np.nan, np.nan
        
        results[metric] = {
            'cluster_stats': cluster_stats,
            'anova_f': f_stat,
            'anova_p': p_value,
            'kruskal_h': h_stat,
            'kruskal_p': h_p_value
        }
    
    return results

def create_cluster_visualizations(df, cluster_result, strategy, datasource, performance_results):
    """創建聚類可視化圖表"""
    if cluster_result is None:
        return
    
    # 1. 聚類散點圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA散點圖
    ax1 = axes[0, 0]
    for cluster_id in np.unique(cluster_result['labels']):
        mask = cluster_result['labels'] == cluster_id
        ax1.scatter(cluster_result['X_pca'][mask, 0], cluster_result['X_pca'][mask, 1], 
                   label=f'Cluster {cluster_id}', alpha=0.7)
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_title(f'{strategy} - {datasource}\nClustering Results (k={cluster_result["best_k"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 各聚類的性能箱線圖
    ax2 = axes[0, 1]
    performance_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    available_metrics = [m for m in performance_metrics if m in df.columns]
    
    if available_metrics:
        metric = available_metrics[0]  # 使用第一個可用指標
        mask = (~df[metric].isna()) & np.isfinite(df[metric])
        if mask.sum() > 0:
            metric_data = df.loc[mask, metric]
            cluster_data = cluster_result['labels'][mask]
            
            cluster_groups = []
            cluster_labels = []
            for cluster_id in np.unique(cluster_data):
                cluster_mask = cluster_data == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_groups.append(metric_data[cluster_mask])
                    cluster_labels.append(f'Cluster {cluster_id}')
            
            if cluster_groups:
                ax2.boxplot(cluster_groups, labels=cluster_labels)
                ax2.set_ylabel(metric)
                ax2.set_title(f'{metric} by Cluster')
                ax2.grid(True, alpha=0.3)
    
    # 3. 聚類品質指標
    ax3 = axes[1, 0]
    quality_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
    quality_values = [cluster_result[m] for m in quality_metrics]
    quality_labels = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
    
    bars = ax3.bar(quality_labels, quality_values)
    ax3.set_ylabel('Score')
    ax3.set_title('Clustering Quality Metrics')
    ax3.grid(True, alpha=0.3)
    
    # 為 Davies-Bouldin 分數著色（越低越好）
    bars[2].set_color('red')
    
    # 4. 聚類大小分布
    ax4 = axes[1, 1]
    cluster_counts = np.bincount(cluster_result['labels'])
    cluster_labels = [f'Cluster {i}' for i in range(len(cluster_counts))]
    
    ax4.pie(cluster_counts, labels=cluster_labels, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_cluster_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_analysis(all_results):
    """創建跨策略和數據源的比較分析"""
    # 收集所有聚類品質指標
    comparison_data = []
    
    for (strategy, datasource), result in all_results.items():
        if result and 'cluster_result' in result:
            cluster_result = result['cluster_result']
            comparison_data.append({
                'strategy': strategy,
                'datasource': datasource,
                'silhouette_score': cluster_result['silhouette_score'],
                'calinski_harabasz_score': cluster_result['calinski_harabasz_score'],
                'davies_bouldin_score': cluster_result['davies_bouldin_score'],
                'best_k': cluster_result['best_k'],
                'best_method': cluster_result['best_method']
            })
    
    if not comparison_data:
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 1. 聚類品質比較
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Silhouette Score 比較
    ax1 = axes[0, 0]
    sns.boxplot(data=comparison_df, x='strategy', y='silhouette_score', ax=ax1)
    ax1.set_title('Silhouette Score by Strategy')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # Calinski-Harabasz Score 比較
    ax2 = axes[0, 1]
    sns.boxplot(data=comparison_df, x='strategy', y='calinski_harabasz_score', ax=ax2)
    ax2.set_title('Calinski-Harabasz Score by Strategy')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # Davies-Bouldin Score 比較
    ax3 = axes[1, 0]
    sns.boxplot(data=comparison_df, x='strategy', y='davies_bouldin_score', ax=ax3)
    ax3.set_title('Davies-Bouldin Score by Strategy')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    
    # 最佳聚類數比較
    ax4 = axes[1, 1]
    sns.countplot(data=comparison_df, x='best_k', ax=ax4)
    ax4.set_title('Distribution of Optimal Cluster Numbers')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cluster_quality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 數據源比較
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Silhouette Score by Datasource
    ax1 = axes[0, 0]
    sns.boxplot(data=comparison_df, x='datasource', y='silhouette_score', ax=ax1)
    ax1.set_title('Silhouette Score by Datasource')
    
    # Calinski-Harabasz Score by Datasource
    ax2 = axes[0, 1]
    sns.boxplot(data=comparison_df, x='datasource', y='calinski_harabasz_score', ax=ax2)
    ax2.set_title('Calinski-Harabasz Score by Datasource')
    
    # Davies-Bouldin Score by Datasource
    ax3 = axes[1, 0]
    sns.boxplot(data=comparison_df, x='datasource', y='davies_bouldin_score', ax=ax3)
    ax3.set_title('Davies-Bouldin Score by Datasource')
    
    # 聚類方法分布
    ax4 = axes[1, 1]
    sns.countplot(data=comparison_df, x='best_method', ax=ax4)
    ax4.set_title('Distribution of Best Clustering Methods')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cluster_datasource_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存比較結果
    comparison_df.to_csv(f'{OUTPUT_DIR}/cluster_quality_comparison.csv', index=False)
    
    return comparison_df

def main():
    """主函數"""
    print("開始 hierarchical 聚類穩定性分析...")
    
    # 載入數據
    merged_data = load_and_preprocess_data()
    print(f"載入了 {len(merged_data)} 個策略-數據源組合")
    
    all_results = {}
    
    # 對每個策略-數據源組合進行分析
    for (strategy, datasource), df in merged_data.items():
        print(f"分析 {strategy} - {datasource}...")
        
        # 進行聚類分析
        cluster_result = perform_clustering_analysis(df, strategy, datasource)
        
        if cluster_result is not None:
            # 分析聚類性能
            performance_results = analyze_cluster_performance(df, cluster_result['labels'], strategy, datasource)
            
            # 創建可視化
            create_cluster_visualizations(df, cluster_result, strategy, datasource, performance_results)
            
            # 保存結果
            all_results[(strategy, datasource)] = {
                'cluster_result': cluster_result,
                'performance_results': performance_results,
                'data_size': len(df)
            }
            
            print(f"  - 最佳聚類數: {cluster_result['best_k']}")
            print(f"  - 最佳方法: {cluster_result['best_method']}")
            print(f"  - Silhouette Score: {cluster_result['silhouette_score']:.3f}")
        else:
            print(f"  - 跳過（數據不足）")
    
    # 創建比較分析
    if all_results:
        comparison_df = create_comparison_analysis(all_results)
        
        # 輸出總結報告
        print("\n=== 聚類穩定性分析總結 ===")
        print(f"總共分析了 {len(all_results)} 個策略-數據源組合")
        
        if comparison_df is not None:
            print(f"\n聚類品質統計:")
            print(f"Silhouette Score 平均: {comparison_df['silhouette_score'].mean():.3f}")
            print(f"Calinski-Harabasz Score 平均: {comparison_df['calinski_harabasz_score'].mean():.3f}")
            print(f"Davies-Bouldin Score 平均: {comparison_df['davies_bouldin_score'].mean():.3f}")
            
            print(f"\n最佳聚類數分布:")
            print(comparison_df['best_k'].value_counts().sort_index())
            
            print(f"\n最佳聚類方法分布:")
            print(comparison_df['best_method'].value_counts())
    
    print(f"\n分析結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 