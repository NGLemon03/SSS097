# v8: 快速自適應分群，優化速度與效果平衡
# 主要功能：快速特徵選擇、簡化方法比較、高效分群評估
# 在保持效果的前提下大幅提升執行速度

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import json
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 設定路徑
INPUT_DIR = 'results/risk_enhanced_clustering'
OUTPUT_DIR = 'results/fast_adaptive_clustering'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_risk_clustered_data():
    """載入風險分群後的數據"""
    all_files = glob.glob(f'{INPUT_DIR}/*_risk_clustered.csv')
    data_dict = {}
    
    for f in all_files:
        fname = os.path.basename(f)
        print(f"處理文件: {fname}")
        
        # 提取策略和數據源
        if 'RMA_Self' in fname:
            strategy = 'RMA_Self'
            datasource = 'Self'
        elif 'single_Self' in fname:
            strategy = 'single_Self'
            datasource = 'Self'
        elif 'ssma_turn_Self' in fname:
            strategy = 'ssma_turn_Self'
            datasource = 'Self'
        else:
            continue
        
        df = pd.read_csv(f)
        print(f"載入 {strategy} - {datasource}: {len(df)} 行")
        data_dict[(strategy, datasource)] = df
    
    return data_dict

def create_simple_enhanced_features(df):
    """創建簡單的增強特徵"""
    enhanced_df = df.copy()
    
    # 只選擇最重要的參數進行增強
    important_params = ['param_1', 'param_2', 'param_3']
    available_params = [p for p in important_params if p in df.columns]
    
    # 簡單的參數增強
    for col in available_params:
        if df[col].min() > 0:
            enhanced_df[f'{col}_log'] = np.log(df[col])
        enhanced_df[f'{col}_squared'] = df[col] ** 2
    
    # 簡單的風險調整
    if 'annual_volatility' in df.columns and 'total_return' in df.columns:
        enhanced_df['vol_return_ratio'] = df['annual_volatility'] / (df['total_return'] + 1e-8)
    
    return enhanced_df

def fast_clustering_comparison(X_scaled, max_clusters=8):
    """快速分群方法比較"""
    methods = {}
    
    # 1. Hierarchical Clustering (快速)
    print("測試階層式分群...")
    Z = linkage(X_scaled, method='ward')
    best_hier_k = 2
    best_hier_score = -1
    
    # 減少測試的群數範圍
    for k in range(2, min(max_clusters + 1, len(X_scaled) // 5)):
        try:
            labels = fcluster(Z, t=k, criterion='maxclust')
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_hier_score:
                    best_hier_score = score
                    best_hier_k = k
        except:
            continue
    
    methods['hierarchical'] = {
        'method': 'hierarchical',
        'best_k': best_hier_k,
        'silhouette': best_hier_score,
        'linkage_matrix': Z
    }
    
    # 2. KMeans (只測試兩種標準化方法)
    print("測試KMeans分群...")
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler()
    }
    
    best_kmeans_score = -1
    best_kmeans_k = 2
    best_kmeans_scaler = 'standard'
    
    for scaler_name, scaler in scalers.items():
        X_rescaled = scaler.fit_transform(X_scaled)
        
        for k in range(2, min(max_clusters + 1, len(X_scaled) // 5)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)  # 減少n_init
                labels = kmeans.fit_predict(X_rescaled)
                score = silhouette_score(X_rescaled, labels)
                if score > best_kmeans_score:
                    best_kmeans_score = score
                    best_kmeans_k = k
                    best_kmeans_scaler = scaler_name
            except:
                continue
    
    methods['kmeans'] = {
        'method': 'kmeans',
        'best_k': best_kmeans_k,
        'silhouette': best_kmeans_score,
        'scaler': best_kmeans_scaler
    }
    
    # 3. Agglomerative Clustering (快速替代)
    print("測試凝聚式分群...")
    best_agglo_score = -1
    best_agglo_k = 2
    
    for k in range(2, min(max_clusters + 1, len(X_scaled) // 5)):
        try:
            agglo = AgglomerativeClustering(n_clusters=k)
            labels = agglo.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_agglo_score:
                best_agglo_score = score
                best_agglo_k = k
        except:
            continue
    
    methods['agglomerative'] = {
        'method': 'agglomerative',
        'best_k': best_agglo_k,
        'silhouette': best_agglo_score
    }
    
    return methods

def apply_best_clustering(X_scaled, methods):
    """應用最佳分群方法"""
    # 選擇最佳方法
    best_method = max(methods.items(), key=lambda x: x[1]['silhouette'])
    method_name, method_config = best_method
    
    print(f"最佳分群方法: {method_name} (Silhouette: {method_config['silhouette']:.3f})")
    
    if method_name == 'hierarchical':
        labels = fcluster(method_config['linkage_matrix'], 
                         t=method_config['best_k'], criterion='maxclust')
    elif method_name == 'kmeans':
        scaler = StandardScaler() if method_config['scaler'] == 'standard' else RobustScaler()
        X_rescaled = scaler.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=method_config['best_k'], random_state=42, n_init=5)
        labels = kmeans.fit_predict(X_rescaled)
    elif method_name == 'agglomerative':
        agglo = AgglomerativeClustering(n_clusters=method_config['best_k'])
        labels = agglo.fit_predict(X_scaled)
    
    return {
        'labels': labels,
        'method': method_name,
        'config': method_config,
        'all_methods': methods
    }

def analyze_clustering_quality(df, clustering_result):
    """分析分群品質"""
    print("\n=== 分群品質分析 ===")
    
    labels = clustering_result['labels']
    method = clustering_result['method']
    config = clustering_result['config']
    
    # 基本統計
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    print(f"分群方法: {method}")
    print(f"群組數量: {n_clusters}")
    print(f"Silhouette Score: {config['silhouette']:.3f}")
    
    # 群組大小分布
    cluster_sizes = np.bincount(labels)
    print(f"群組大小分布: {cluster_sizes}")
    
    return {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'silhouette_score': config['silhouette'],
        'method': method
    }

def create_fast_visualizations(df, clustering_result, strategy, datasource):
    """創建快速可視化"""
    labels = clustering_result['labels']
    method = clustering_result['method']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. PCA散點圖
    ax1 = axes[0, 0]
    # 使用主要特徵進行PCA
    main_cols = [col for col in df.columns if col.startswith('param_') or 
                col in ['annual_volatility', 'downside_risk']]
    available_cols = [col for col in main_cols if col in df.columns][:10]  # 限制特徵數
    
    if len(available_cols) >= 2:
        X_pca = df[available_cols].fillna(0).values
        scaler = StandardScaler()
        X_pca_scaled = scaler.fit_transform(X_pca)
        
        pca = PCA(n_components=2)
        X_pca_2d = pca.fit_transform(X_pca_scaled)
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax1.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                       label=f'Cluster {cluster_id}', alpha=0.7)
        
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.set_title(f'{strategy} - {datasource}\n{method.title()} Clustering')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 群組大小分布
    ax2 = axes[0, 1]
    cluster_counts = np.bincount(labels)
    cluster_labels = [f'Cluster {i}' for i in range(len(cluster_counts))]
    
    ax2.bar(cluster_labels, cluster_counts)
    ax2.set_title('Cluster Size Distribution')
    ax2.set_ylabel('Number of Trials')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 績效指標箱線圖
    ax3 = axes[1, 0]
    if 'total_return' in df.columns:
        cluster_groups = []
        cluster_labels = []
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_groups.append(df.loc[cluster_mask, 'total_return'])
            cluster_labels.append(f'Cluster {cluster_id}')
        
        ax3.boxplot(cluster_groups, labels=cluster_labels)
        ax3.set_ylabel('Total Return')
        ax3.set_title('Total Return by Cluster')
        ax3.grid(True, alpha=0.3)
    
    # 4. 方法比較
    ax4 = axes[1, 1]
    all_methods = clustering_result['all_methods']
    method_names = list(all_methods.keys())
    silhouette_scores = [all_methods[m]['silhouette'] for m in method_names]
    
    ax4.bar(method_names, silhouette_scores)
    ax4.set_ylabel('Silhouette Score')
    ax4.set_title('Method Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_fast_clustering.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主函數"""
    print("開始快速自適應分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    all_results = {}
    
    for (strategy, datasource), df in data_dict.items():
        print(f"\n分析 {strategy} - {datasource}")
        
        # 創建簡單增強特徵
        enhanced_df = create_simple_enhanced_features(df)
        
        # 選擇特徵（參數+風險，去除績效）
        param_cols = [col for col in enhanced_df.columns if col.startswith('param_')]
        risk_cols = ['annual_volatility', 'downside_risk', 'var_95_annual', 'cvar_95_annual']
        
        # 限制特徵數量以提高速度
        all_feature_cols = param_cols + risk_cols
        available_features = [col for col in all_feature_cols if col in enhanced_df.columns][:15]  # 限制特徵數
        
        print(f"使用特徵數量: {len(available_features)}")
        
        if len(available_features) < 3:
            print("特徵不足，跳過")
            continue
        
        # 準備特徵數據
        X = enhanced_df[available_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 快速分群比較
        methods = fast_clustering_comparison(X_scaled)
        
        # 應用最佳分群
        clustering_result = apply_best_clustering(X_scaled, methods)
        
        # 分析分群品質
        quality_analysis = analyze_clustering_quality(enhanced_df, clustering_result)
        
        # 創建可視化
        create_fast_visualizations(enhanced_df, clustering_result, strategy, datasource)
        
        # 保存結果
        result_df = enhanced_df.copy()
        result_df['fast_cluster'] = clustering_result['labels']
        result_df['clustering_method'] = clustering_result['method']
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_fast_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_results[(strategy, datasource)] = {
            'quality': quality_analysis,
            'clustering_result': clustering_result,
            'data_size': len(df)
        }
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print("\n=== 快速自適應分群總結 ===")
    for (strategy, datasource), result in all_results.items():
        quality = result['quality']
        print(f"\n{strategy} - {datasource}:")
        print(f"  分群方法: {quality['method']}")
        print(f"  群組數量: {quality['n_clusters']}")
        print(f"  Silhouette Score: {quality['silhouette_score']:.3f}")
        print(f"  群組大小: {quality['cluster_sizes']}")
    
    print(f"\n所有結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 