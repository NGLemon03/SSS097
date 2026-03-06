# v7: 自適應分群分析，自動選擇最佳分群方法
# 主要功能：多方法評估、自動選擇、特徵工程、穩定性測試
# 提供無偏見的自適應分群解決方案

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
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
OUTPUT_DIR = 'results/adaptive_clustering'
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

def create_enhanced_features(df):
    """創建增強特徵，提高分群分辨力"""
    enhanced_df = df.copy()
    
    # 1. 參數特徵增強
    param_cols = [col for col in df.columns if col.startswith('param_')]
    for col in param_cols:
        if col in df.columns:
            # 添加對數變換
            if df[col].min() > 0:
                enhanced_df[f'{col}_log'] = np.log(df[col])
            # 添加平方項
            enhanced_df[f'{col}_squared'] = df[col] ** 2
    
    # 2. 風險特徵增強
    risk_cols = ['annual_volatility', 'downside_risk', 'var_95_annual', 'cvar_95_annual']
    for col in risk_cols:
        if col in df.columns:
            # 添加風險調整指標
            if 'total_return' in df.columns:
                enhanced_df[f'{col}_return_ratio'] = df[col] / (df['total_return'] + 1e-8)
    
    # 3. 參數交互項（選擇重要參數）
    important_params = ['param_1', 'param_2', 'param_3']  # 根據實際情況調整
    available_params = [p for p in important_params if p in df.columns]
    
    if len(available_params) >= 2:
        for i in range(len(available_params)):
            for j in range(i+1, len(available_params)):
                p1, p2 = available_params[i], available_params[j]
                enhanced_df[f'{p1}_{p2}_interaction'] = df[p1] * df[p2]
    
    # 4. 績效相關特徵（但不用於分群，只用於分析）
    perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor', 'score']
    for col in perf_cols:
        if col in df.columns:
            # 添加績效分位數
            enhanced_df[f'{col}_quantile'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
    
    return enhanced_df

def select_best_clustering_method(X_scaled, max_clusters=10):
    """選擇最佳分群方法和參數"""
    methods = {}
    
    # 1. Hierarchical Clustering
    print("測試階層式分群...")
    Z = linkage(X_scaled, method='ward')
    best_hier_k = 2
    best_hier_score = -1
    
    for k in range(2, min(max_clusters + 1, len(X_scaled) // 3)):
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
    
    # 2. KMeans with different scalers
    print("測試KMeans分群...")
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler()
    }
    
    best_kmeans_score = -1
    best_kmeans_k = 2
    best_kmeans_scaler = 'standard'
    
    for scaler_name, scaler in scalers.items():
        X_rescaled = scaler.fit_transform(X_scaled)
        
        for k in range(2, min(max_clusters + 1, len(X_scaled) // 3)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
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
    
    # 3. DBSCAN
    print("測試DBSCAN分群...")
    best_dbscan_score = -1
    best_dbscan_eps = 0.5
    best_dbscan_min_samples = 5
    
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for min_samples in [3, 5, 10]:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                if len(np.unique(labels)) > 1 and -1 not in labels:  # 排除噪聲點
                    score = silhouette_score(X_scaled, labels)
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_eps = eps
                        best_dbscan_min_samples = min_samples
            except:
                continue
    
    if best_dbscan_score > -1:
        methods['dbscan'] = {
            'method': 'dbscan',
            'eps': best_dbscan_eps,
            'min_samples': best_dbscan_min_samples,
            'silhouette': best_dbscan_score
        }
    
    # 4. Spectral Clustering
    print("測試譜分群...")
    best_spectral_score = -1
    best_spectral_k = 2
    
    for k in range(2, min(max_clusters + 1, len(X_scaled) // 3)):
        try:
            spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='rbf')
            labels = spectral.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_spectral_score:
                best_spectral_score = score
                best_spectral_k = k
        except:
            continue
    
    methods['spectral'] = {
        'method': 'spectral',
        'best_k': best_spectral_k,
        'silhouette': best_spectral_score
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
        scaler = StandardScaler() if method_config['scaler'] == 'standard' else \
                RobustScaler() if method_config['scaler'] == 'robust' else MinMaxScaler()
        X_rescaled = scaler.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=method_config['best_k'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_rescaled)
    elif method_name == 'dbscan':
        dbscan = DBSCAN(eps=method_config['eps'], min_samples=method_config['min_samples'])
        labels = dbscan.fit_predict(X_scaled)
    elif method_name == 'spectral':
        spectral = SpectralClustering(n_clusters=method_config['best_k'], 
                                    random_state=42, affinity='rbf')
        labels = spectral.fit_predict(X_scaled)
    
    return {
        'labels': labels,
        'method': method_name,
        'config': method_config,
        'all_methods': methods
    }

def analyze_clustering_quality(df, clustering_result, feature_names):
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
    
    # 特徵重要性分析
    if method == 'kmeans':
        # 計算每個特徵對分群的貢獻
        feature_importance = []
        for i, feature in enumerate(feature_names):
            feature_values = df[feature].values
            # 計算群組間變異 / 群組內變異
            ss_between = 0
            ss_within = 0
            
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_mean = np.mean(feature_values[cluster_mask])
                overall_mean = np.mean(feature_values)
                
                ss_between += np.sum(cluster_mask) * (cluster_mean - overall_mean) ** 2
                ss_within += np.sum((feature_values[cluster_mask] - cluster_mean) ** 2)
            
            if ss_within > 0:
                importance = ss_between / ss_within
            else:
                importance = 0
            
            feature_importance.append((feature, importance))
        
        # 排序並顯示重要特徵
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        print("\n重要特徵 (前10):")
        for feature, importance in feature_importance[:10]:
            print(f"  {feature}: {importance:.3f}")
    
    return {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'silhouette_score': config['silhouette'],
        'method': method
    }

def create_adaptive_visualizations(df, clustering_result, feature_names, strategy, datasource):
    """創建自適應分群可視化"""
    labels = clustering_result['labels']
    method = clustering_result['method']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA散點圖
    ax1 = axes[0, 0]
    # 使用原始特徵進行PCA
    param_risk_cols = [col for col in df.columns if col.startswith('param_') or 
                      col in ['annual_volatility', 'downside_risk', 'var_95_annual']]
    available_cols = [col for col in param_risk_cols if col in df.columns]
    
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
    ax3 = axes[0, 2]
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
    
    # 4. 風險指標箱線圖
    ax4 = axes[1, 0]
    if 'annual_volatility' in df.columns:
        cluster_groups = []
        cluster_labels = []
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_groups.append(df.loc[cluster_mask, 'annual_volatility'])
            cluster_labels.append(f'Cluster {cluster_id}')
        
        ax4.boxplot(cluster_groups, labels=cluster_labels)
        ax4.set_ylabel('Annual Volatility')
        ax4.set_title('Volatility by Cluster')
        ax4.grid(True, alpha=0.3)
    
    # 5. 參數分布（選擇重要參數）
    ax5 = axes[1, 1]
    important_params = ['param_1', 'param_2', 'param_3']
    available_params = [p for p in important_params if p in df.columns]
    
    if available_params:
        param = available_params[0]
        cluster_groups = []
        cluster_labels = []
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_groups.append(df.loc[cluster_mask, param])
            cluster_labels.append(f'Cluster {cluster_id}')
        
        ax5.boxplot(cluster_groups, labels=cluster_labels)
        ax5.set_ylabel(param)
        ax5.set_title(f'{param} by Cluster')
        ax5.grid(True, alpha=0.3)
    
    # 6. 方法比較
    ax6 = axes[1, 2]
    all_methods = clustering_result['all_methods']
    method_names = list(all_methods.keys())
    silhouette_scores = [all_methods[m]['silhouette'] for m in method_names]
    
    ax6.bar(method_names, silhouette_scores)
    ax6.set_ylabel('Silhouette Score')
    ax6.set_title('Method Comparison')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_adaptive_clustering.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主函數"""
    print("開始自適應分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    all_results = {}
    
    for (strategy, datasource), df in data_dict.items():
        print(f"\n分析 {strategy} - {datasource}")
        
        # 創建增強特徵
        enhanced_df = create_enhanced_features(df)
        
        # 選擇特徵（參數+風險，去除績效）
        param_cols = [col for col in enhanced_df.columns if col.startswith('param_')]
        risk_cols = ['annual_volatility', 'downside_risk', 'sortino_ratio', 'var_95_annual', 
                    'cvar_95_annual', 'skewness', 'kurtosis', 'vol_stability', 
                    'drawdown_duration', 'max_consecutive_losses']
        
        # 添加增強特徵
        enhanced_param_cols = [col for col in enhanced_df.columns if 'param_' in col]
        enhanced_risk_cols = [col for col in enhanced_df.columns if any(risk in col for risk in risk_cols)]
        
        all_feature_cols = enhanced_param_cols + enhanced_risk_cols
        available_features = [col for col in all_feature_cols if col in enhanced_df.columns]
        
        print(f"使用特徵數量: {len(available_features)}")
        
        if len(available_features) < 3:
            print("特徵不足，跳過")
            continue
        
        # 準備特徵數據
        X = enhanced_df[available_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 選擇最佳分群方法
        methods = select_best_clustering_method(X_scaled)
        
        # 應用最佳分群
        clustering_result = apply_best_clustering(X_scaled, methods)
        
        # 分析分群品質
        quality_analysis = analyze_clustering_quality(enhanced_df, clustering_result, available_features)
        
        # 創建可視化
        create_adaptive_visualizations(enhanced_df, clustering_result, available_features, strategy, datasource)
        
        # 保存結果
        result_df = enhanced_df.copy()
        result_df['adaptive_cluster'] = clustering_result['labels']
        result_df['clustering_method'] = clustering_result['method']
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_adaptive_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_results[(strategy, datasource)] = {
            'quality': quality_analysis,
            'clustering_result': clustering_result,
            'data_size': len(df)
        }
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print("\n=== 自適應分群總結 ===")
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