# v16: 自適應細粒度分群，結合自適應與細粒度需求
# 主要功能：自適應細粒度分群、多方法比較、統計分析、效果評估
# 提供智能化的細粒度分群解決方案

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats
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
OUTPUT_DIR = 'results/adaptive_fine_clustering'
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

def adaptive_fine_clustering(df, strategy, datasource):
    """自適應細粒度分群：專門針對小群組需求"""
    print(f"\n=== {strategy} - {datasource} 自適應細粒度分群 ===")
    
    # 選擇特徵
    feature_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['return', 'score', 'sharpe', 'calmar', 'sortino', 'volatility', 'risk', 'var', 'cvar', 'downside', 'drawdown']):
            feature_cols.append(col)
    
    print(f"找到特徵: {len(feature_cols)} 個")
    
    if len(feature_cols) == 0:
        print("沒有找到特徵，使用預設特徵")
        feature_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'annual_volatility']
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # 準備特徵數據
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 目標：產生約600-800個群組，每群1-5個數據
    n_samples = len(df)
    target_clusters = min(max(600, n_samples // 3), 800)  # 目標群組數
    
    print(f"目標群組數: {target_clusters}")
    print(f"預期平均群組大小: {n_samples / target_clusters:.1f}")
    
    # 方法1：使用距離閾值的階層式分群
    print("\n方法1: 距離閾值階層式分群")
    
    # 計算距離閾值
    Z = linkage(X_scaled, method='ward')
    
    # 嘗試不同的距離閾值
    distance_thresholds = np.linspace(0.1, 2.0, 20)
    best_threshold = None
    best_score = -1
    
    for threshold in distance_thresholds:
        labels = fcluster(Z, t=threshold, criterion='distance')
        n_clusters = len(np.unique(labels))
        
        if n_clusters > 1:
                    # 計算群組大小分布分數
        # 處理DBSCAN的負值標籤
        if method_name == 'dbscan':
            # 排除噪聲點
            valid_labels = labels[labels != -1]
            if len(valid_labels) == 0:
                continue
            cluster_sizes = np.bincount(valid_labels)
        else:
            cluster_sizes = np.bincount(labels)
        
        small_clusters = np.sum(cluster_sizes <= 5)
        small_ratio = small_clusters / len(cluster_sizes)
            
            # 計算與目標群組數的接近程度
            target_penalty = abs(n_clusters - target_clusters) / target_clusters
            
            # 綜合分數
            score = small_ratio - target_penalty
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    print(f"最佳距離閾值: {best_threshold:.3f}")
    
    # 使用最佳閾值進行分群
    labels_hierarchical = fcluster(Z, t=best_threshold, criterion='distance')
    
    # 方法2：使用KMeans，動態調整群組數
    print("\n方法2: 動態KMeans分群")
    
    # 從較大的群組數開始，逐步減少直到達到目標
    k_range = range(max(100, target_clusters // 2), min(target_clusters * 2, n_samples // 2), 10)
    best_k = None
    best_score_kmeans = -1
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_scaled)
        
        # 計算群組大小分布分數
        cluster_sizes = np.bincount(labels_kmeans)
        small_clusters = np.sum(cluster_sizes <= 5)
        small_ratio = small_clusters / len(cluster_sizes)
        
        # 計算與目標群組數的接近程度
        target_penalty = abs(k - target_clusters) / target_clusters
        
        # 綜合分數
        score = small_ratio - target_penalty
        
        if score > best_score_kmeans:
            best_score_kmeans = score
            best_k = k
    
    print(f"最佳K值: {best_k}")
    
    # 使用最佳K值進行分群
    if best_k is not None:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_scaled)
    else:
        # 如果沒有找到最佳K值，使用預設值
        best_k = min(100, n_samples // 2)
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_scaled)
    
    # 方法3：使用DBSCAN
    print("\n方法3: DBSCAN分群")
    
    # 嘗試不同的eps值
    eps_range = np.linspace(0.1, 1.0, 20)
    best_eps = None
    best_score_dbscan = -1
    
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels_dbscan = dbscan.fit_predict(X_scaled)
        
        # 排除噪聲點
        valid_labels = labels_dbscan[labels_dbscan != -1]
        if len(valid_labels) == 0:
            continue
            
        n_clusters = len(np.unique(valid_labels))
        
        if n_clusters > 1:
            # 計算群組大小分布分數
            cluster_sizes = np.bincount(valid_labels)
            small_clusters = np.sum(cluster_sizes <= 5)
            small_ratio = small_clusters / len(cluster_sizes)
            
            # 計算與目標群組數的接近程度
            target_penalty = abs(n_clusters - target_clusters) / target_clusters
            
            # 綜合分數
            score = small_ratio - target_penalty
            
            if score > best_score_dbscan:
                best_score_dbscan = score
                best_eps = eps
    
    print(f"最佳eps值: {best_eps:.3f}")
    
    # 使用最佳eps值進行分群
    if best_eps is not None:
        dbscan = DBSCAN(eps=best_eps, min_samples=2)
        labels_dbscan = dbscan.fit_predict(X_scaled)
    else:
        # 如果沒有找到最佳eps值，使用預設值
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        labels_dbscan = dbscan.fit_predict(X_scaled)
    
    # 比較三種方法
    methods = {
        'hierarchical': labels_hierarchical,
        'kmeans': labels_kmeans,
        'dbscan': labels_dbscan
    }
    
    best_method = None
    best_overall_score = -1
    best_labels = None
    
    for method_name, labels in methods.items():
        if labels is None or len(np.unique(labels)) <= 1:
            continue
            
        # 計算群組大小分布
        # 處理DBSCAN的負值標籤
        if method_name == 'dbscan':
            # 排除噪聲點
            valid_labels = labels[labels != -1]
            if len(valid_labels) == 0:
                continue
            cluster_sizes = np.bincount(valid_labels)
        else:
            cluster_sizes = np.bincount(labels)
        
        small_clusters = np.sum(cluster_sizes <= 5)
        medium_clusters = np.sum((cluster_sizes > 5) & (cluster_sizes <= 10))
        large_clusters = np.sum(cluster_sizes > 10)
        
        # 計算分數
        small_ratio = small_clusters / len(cluster_sizes)
        target_penalty = abs(len(cluster_sizes) - target_clusters) / target_clusters
        large_penalty = large_clusters / len(cluster_sizes)  # 懲罰大群組
        
        overall_score = small_ratio - target_penalty - large_penalty
        
        print(f"{method_name}: 群組數={len(cluster_sizes)}, 小群組比例={small_ratio:.3f}, 分數={overall_score:.3f}")
        
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_method = method_name
            best_labels = labels
    
    print(f"\n最佳方法: {best_method}")
    
    # 重新編號標籤（從1開始）
    unique_labels = np.unique(best_labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels, 1)}
    final_labels = np.array([label_mapping[label] for label in best_labels])
    
    # 統計最終結果
    final_sizes = np.bincount(final_labels)
    print(f"\n最終分群結果:")
    print(f"總群組數: {len(final_sizes)}")
    print(f"群組大小分布: {final_sizes}")
    print(f"平均群組大小: {np.mean(final_sizes):.1f}")
    print(f"最大群組大小: {np.max(final_sizes)}")
    print(f"小群組比例 (≤5): {np.sum(final_sizes <= 5) / len(final_sizes):.3f}")
    
    return {
        'final_labels': final_labels,
        'method': best_method,
        'final_sizes': final_sizes,
        'feature_cols': feature_cols,
        'target_clusters': target_clusters,
        'methods_comparison': methods
    }

def analyze_fine_clustering_quality(clustering_result, df, strategy, datasource):
    """分析細粒度分群品質"""
    print(f"\n=== {strategy} - {datasource} 分群品質分析 ===")
    
    final_labels = clustering_result['final_labels']
    
    # 基本統計
    unique_labels = np.unique(final_labels)
    final_sizes = clustering_result['final_sizes']
    
    # 計算小群組統計
    small_clusters = final_sizes[final_sizes <= 5]
    medium_clusters = final_sizes[(final_sizes > 5) & (final_sizes <= 10)]
    large_clusters = final_sizes[final_sizes > 10]
    
    print(f"小群組 (≤5): {len(small_clusters)} 個")
    print(f"中群組 (6-10): {len(medium_clusters)} 個")
    print(f"大群組 (>10): {len(large_clusters)} 個")
    
    # 特徵重要性分析
    feature_cols = clustering_result['feature_cols']
    feature_importance = {}
    
    for feature in feature_cols:
        if feature in df.columns:
            feature_values = df[feature].values
            
            # ANOVA檢定
            try:
                groups = [feature_values[final_labels == i] for i in unique_labels]
                f_stat, p_value = stats.f_oneway(*groups)
                
                feature_importance[feature] = {
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                feature_importance[feature] = {
                    'f_stat': 0,
                    'p_value': 1,
                    'significant': False
                }
    
    n_significant_features = sum(1 for f in feature_importance.values() if f['significant'])
    print(f"顯著特徵數: {n_significant_features}")
    
    return {
        'final_sizes': final_sizes,
        'small_clusters': len(small_clusters),
        'medium_clusters': len(medium_clusters),
        'large_clusters': len(large_clusters),
        'mean_size': np.mean(final_sizes),
        'max_size': np.max(final_sizes),
        'small_ratio': len(small_clusters) / len(final_sizes),
        'feature_importance': feature_importance,
        'n_significant_features': n_significant_features
    }

def create_fine_clustering_visualizations(df, clustering_result, quality_analysis, strategy, datasource):
    """創建細粒度分群可視化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    final_labels = clustering_result['final_labels']
    final_sizes = clustering_result['final_sizes']
    method = clustering_result['method']
    
    # 1. 群組大小分布
    ax1 = axes[0, 0]
    ax1.hist(final_sizes, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('群組大小')
    ax1.set_ylabel('群組數量')
    ax1.set_title('群組大小分布')
    ax1.grid(True, alpha=0.3)
    
    # 2. 群組大小統計
    ax2 = axes[0, 1]
    size_stats = {
        '小群組(≤5)': quality_analysis['small_clusters'],
        '中群組(6-10)': quality_analysis['medium_clusters'],
        '大群組(>10)': quality_analysis['large_clusters']
    }
    
    ax2.bar(size_stats.keys(), size_stats.values())
    ax2.set_ylabel('群組數量')
    ax2.set_title('群組大小統計')
    ax2.grid(True, alpha=0.3)
    
    # 3. 群組大小分布比較
    ax3 = axes[0, 2]
    small_sizes = final_sizes[final_sizes <= 5]
    medium_sizes = final_sizes[(final_sizes > 5) & (final_sizes <= 10)]
    large_sizes = final_sizes[final_sizes > 10]
    
    ax3.hist([small_sizes, medium_sizes, large_sizes], 
             label=['小群組(≤5)', '中群組(6-10)', '大群組(>10)'], 
             alpha=0.7, bins=10)
    ax3.set_xlabel('群組大小')
    ax3.set_ylabel('群組數量')
    ax3.set_title('群組大小分布比較')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 最佳分群的PCA散點圖
    ax4 = axes[1, 0]
    
    # 使用主要特徵進行PCA
    feature_cols = clustering_result['feature_cols']
    available_cols = [col for col in feature_cols if col in df.columns][:10]
    
    if len(available_cols) >= 2:
        X_pca = df[available_cols].fillna(0).values
        scaler = StandardScaler()
        X_pca_scaled = scaler.fit_transform(X_pca)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca_2d = pca.fit_transform(X_pca_scaled)
        
        # 只顯示前20個群組（避免圖表太亂）
        unique_labels = np.unique(final_labels)
        for cluster_id in unique_labels[:20]:
            mask = final_labels == cluster_id
            ax4.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                       label=f'Cluster {cluster_id}', alpha=0.7)
        
        ax4.set_xlabel('PCA Component 1')
        ax4.set_ylabel('PCA Component 2')
        ax4.set_title(f'最終分群PCA散點圖\n(顯示前20個群組)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. 特徵重要性比較
    ax5 = axes[1, 1]
    feature_importance = quality_analysis['feature_importance']
    important_features = [(f, v['f_stat']) for f, v in feature_importance.items() if v['significant']]
    important_features.sort(key=lambda x: x[1], reverse=True)
    
    if important_features:
        features, f_stats = zip(*important_features[:10])
        ax5.barh(range(len(features)), f_stats)
        ax5.set_yticks(range(len(features)))
        ax5.set_yticklabels(features)
        ax5.set_xlabel('F-statistic')
        ax5.set_title('重要特徵')
        ax5.grid(True, alpha=0.3)
    
    # 6. 最終推薦
    ax6 = axes[1, 2]
    ax6.text(0.1, 0.9, f'策略: {strategy}', fontsize=12, transform=ax6.transAxes)
    ax6.text(0.1, 0.8, f'方法: {method}', fontsize=12, transform=ax6.transAxes)
    ax6.text(0.1, 0.7, f'總群組數: {len(final_sizes)}', fontsize=12, transform=ax6.transAxes)
    ax6.text(0.1, 0.6, f'平均群組大小: {quality_analysis["mean_size"]:.1f}', fontsize=12, transform=ax6.transAxes)
    ax6.text(0.1, 0.5, f'最大群組大小: {quality_analysis["max_size"]}', fontsize=12, transform=ax6.transAxes)
    ax6.text(0.1, 0.4, f'小群組比例: {quality_analysis["small_ratio"]:.3f}', fontsize=12, transform=ax6.transAxes)
    ax6.text(0.1, 0.3, f'顯著特徵數: {quality_analysis["n_significant_features"]}', fontsize=12, transform=ax6.transAxes)
    
    # 推薦信息
    if quality_analysis['small_ratio'] >= 0.7:
        ax6.text(0.1, 0.2, '✅ 小群組比例良好', fontsize=12, transform=ax6.transAxes, color='green')
    else:
        ax6.text(0.1, 0.2, '⚠️ 小群組比例偏低', fontsize=12, transform=ax6.transAxes, color='orange')
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('最終推薦')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_adaptive_fine_clustering.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def generate_fine_clustering_recommendations(clustering_result, quality_analysis, strategy, datasource):
    """生成細粒度分群建議"""
    print(f"\n=== {strategy} - {datasource} 細粒度分群建議 ===")
    
    final_sizes = clustering_result['final_sizes']
    method = clustering_result['method']
    target_clusters = clustering_result['target_clusters']
    
    print(f"使用方法: {method}")
    print(f"目標群組數: {target_clusters}")
    print(f"實際群組數: {len(final_sizes)}")
    print(f"平均群組大小: {quality_analysis['mean_size']:.1f}")
    print(f"最大群組大小: {quality_analysis['max_size']}")
    print(f"小群組比例: {quality_analysis['small_ratio']:.3f}")
    print(f"顯著特徵數: {quality_analysis['n_significant_features']}")
    
    # 分析群組大小分布
    print(f"\n群組大小分布:")
    print(f"小群組 (≤5): {quality_analysis['small_clusters']} 個")
    print(f"中群組 (6-10): {quality_analysis['medium_clusters']} 個")
    print(f"大群組 (>10): {quality_analysis['large_clusters']} 個")
    
    # 推薦改進建議
    print(f"\n改進建議:")
    if quality_analysis['small_ratio'] < 0.5:
        print("- 小群組比例偏低，建議調整分群參數")
    if quality_analysis['max_size'] > 10:
        print("- 存在大群組，建議增加群組數或調整參數")
    if abs(len(final_sizes) - target_clusters) / target_clusters > 0.2:
        print("- 群組數與目標差距較大，建議調整參數")
    if quality_analysis['n_significant_features'] < 5:
        print("- 顯著特徵較少，建議檢查特徵選擇")
    
    return {
        'method': method,
        'total_clusters': len(final_sizes),
        'target_clusters': target_clusters,
        'mean_size': quality_analysis['mean_size'],
        'max_size': quality_analysis['max_size'],
        'small_ratio': quality_analysis['small_ratio'],
        'n_significant_features': quality_analysis['n_significant_features']
    }

def main():
    """主函數"""
    print("開始自適應細粒度分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    all_recommendations = {}
    
    for (strategy, datasource), df in data_dict.items():
        print(f"\n分析 {strategy} - {datasource}")
        
        # 自適應細粒度分群
        clustering_result = adaptive_fine_clustering(df, strategy, datasource)
        
        # 分析分群品質
        quality_analysis = analyze_fine_clustering_quality(clustering_result, df, strategy, datasource)
        
        # 創建可視化
        create_fine_clustering_visualizations(df, clustering_result, quality_analysis, strategy, datasource)
        
        # 生成建議
        recommendations = generate_fine_clustering_recommendations(clustering_result, quality_analysis, 
                                                                 strategy, datasource)
        
        # 保存結果
        result_df = df.copy()
        result_df['adaptive_fine_cluster'] = clustering_result['final_labels']
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_adaptive_fine_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_recommendations[(strategy, datasource)] = recommendations
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print(f"\n=== 自適應細粒度分群總結 ===")
    for (strategy, datasource), recs in all_recommendations.items():
        print(f"\n{strategy} - {datasource}:")
        print(f"  方法: {recs['method']}")
        print(f"  總群組數: {recs['total_clusters']}")
        print(f"  目標群組數: {recs['target_clusters']}")
        print(f"  平均群組大小: {recs['mean_size']:.1f}")
        print(f"  小群組比例: {recs['small_ratio']:.3f}")
        print(f"  顯著特徵數: {recs['n_significant_features']}")
    
    print(f"\n所有結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 