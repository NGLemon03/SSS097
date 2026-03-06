# v13: 細粒度分群，產生大量小群組
# 主要功能：細粒度分群、小群組生成、多方法比較、統計分析
# 專門針對需要大量小群組的應用場景

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
OUTPUT_DIR = 'results/fine_grained_clustering'
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

def find_fine_grained_clusters(X_scaled, n_samples, target_cluster_size=3):
    """尋找精細分群方法，目標群組大小1-5個數據"""
    print(f"尋找精細分群方法，目標群組大小: {target_cluster_size}個數據")
    
    results = {}
    
    # 計算需要的群組數
    estimated_k = max(10, n_samples // target_cluster_size)
    print(f"估計需要群組數: {estimated_k}")
    
    # 測試不同的群組數，從較大的k開始
    k_range = range(max(20, estimated_k//2), min(estimated_k*2, n_samples//2))
    
    for k in k_range:
        try:
            # 階層式分群
            Z = linkage(X_scaled, method='ward')
            hier_labels = fcluster(Z, t=k, criterion='maxclust')
            
            # KMeans分群
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            
            # 計算群組大小
            hier_sizes = np.bincount(hier_labels)
            kmeans_sizes = np.bincount(kmeans_labels)
            
            # 計算目標指標
            hier_mean_size = np.mean(hier_sizes)
            kmeans_mean_size = np.mean(kmeans_sizes)
            hier_max_size = max(hier_sizes)
            kmeans_max_size = max(kmeans_sizes)
            
            # 計算小群組比例（1-5個數據的群組）
            hier_small_ratio = np.sum(hier_sizes <= 5) / len(hier_sizes)
            kmeans_small_ratio = np.sum(kmeans_sizes <= 5) / len(kmeans_sizes)
            
            # 計算指標
            hier_silhouette = silhouette_score(X_scaled, hier_labels)
            kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
            
            # 精細分群評分 = 小群組比例 * 0.6 + Silhouette * 0.3 + 群組數 * 0.1
            hier_score = hier_small_ratio * 0.6 + hier_silhouette * 0.3 + (k / 100) * 0.1
            kmeans_score = kmeans_small_ratio * 0.6 + kmeans_silhouette * 0.3 + (k / 100) * 0.1
            
            results[f'hierarchical_k{k}'] = {
                'labels': hier_labels,
                'silhouette': hier_silhouette,
                'score': hier_score,
                'sizes': hier_sizes,
                'mean_size': hier_mean_size,
                'max_size': hier_max_size,
                'small_ratio': hier_small_ratio,
                'method': 'hierarchical',
                'k': k
            }
            
            results[f'kmeans_k{k}'] = {
                'labels': kmeans_labels,
                'silhouette': kmeans_silhouette,
                'score': kmeans_score,
                'sizes': kmeans_sizes,
                'mean_size': kmeans_mean_size,
                'max_size': kmeans_max_size,
                'small_ratio': kmeans_small_ratio,
                'method': 'kmeans',
                'k': k
            }
            
        except Exception as e:
            print(f"k={k} 時發生錯誤: {e}")
            continue
    
    # 嘗試DBSCAN，調整eps來獲得小群組
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=3)  # 減少鄰居數
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        
        # 嘗試不同的eps值
        eps_values = [np.percentile(distances[:, -1], p) for p in [70, 75, 80, 85, 90]]
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=2)  # 減少最小樣本數
            dbscan_labels = dbscan.fit_predict(X_scaled)
            
            # 檢查DBSCAN結果
            unique_labels = np.unique(dbscan_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:  # 沒有雜訊點
                dbscan_sizes = np.bincount(dbscan_labels)
                dbscan_mean_size = np.mean(dbscan_sizes)
                dbscan_max_size = max(dbscan_sizes)
                dbscan_small_ratio = np.sum(dbscan_sizes <= 5) / len(dbscan_sizes)
                dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
                dbscan_score = dbscan_small_ratio * 0.6 + dbscan_silhouette * 0.3 + (len(unique_labels) / 100) * 0.1
                
                results[f'dbscan_eps{eps:.3f}'] = {
                    'labels': dbscan_labels,
                    'silhouette': dbscan_silhouette,
                    'score': dbscan_score,
                    'sizes': dbscan_sizes,
                    'mean_size': dbscan_mean_size,
                    'max_size': dbscan_max_size,
                    'small_ratio': dbscan_small_ratio,
                    'method': 'dbscan',
                    'k': len(unique_labels),
                    'eps': eps
                }
    except Exception as e:
        print(f"DBSCAN 發生錯誤: {e}")
    
    return results

def analyze_fine_grained_quality(cluster_results, df, feature_names):
    """分析精細分群品質"""
    print("\n=== 精細分群品質分析 ===")
    
    quality_analysis = {}
    
    for method_name, result in cluster_results.items():
        labels = result['labels']
        
        # 基本統計
        unique_labels = np.unique(labels)
        cluster_sizes = result['sizes']
        
        # 計算小群組統計
        small_clusters = cluster_sizes[cluster_sizes <= 5]
        medium_clusters = cluster_sizes[(cluster_sizes > 5) & (cluster_sizes <= 10)]
        large_clusters = cluster_sizes[cluster_sizes > 10]
        
        # 特徵重要性分析
        feature_importance = {}
        for feature in feature_names:
            if feature in df.columns:
                feature_values = df[feature].values
                
                # ANOVA檢定
                try:
                    groups = [feature_values[labels == i] for i in unique_labels]
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
        
        quality_analysis[method_name] = {
            'silhouette': result['silhouette'],
            'score': result['score'],
            'cluster_sizes': cluster_sizes,
            'mean_size': result['mean_size'],
            'max_size': result['max_size'],
            'small_ratio': result['small_ratio'],
            'n_small_clusters': len(small_clusters),
            'n_medium_clusters': len(medium_clusters),
            'n_large_clusters': len(large_clusters),
            'feature_importance': feature_importance,
            'n_significant_features': sum(1 for f in feature_importance.values() if f['significant'])
        }
    
    return quality_analysis

def create_fine_grained_visualizations(df, cluster_results, quality_analysis, feature_names, strategy, datasource):
    """創建精細分群可視化"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 1. 群組數 vs 精細分群評分
    ax1 = axes[0, 0]
    method_names = list(cluster_results.keys())
    scores = [cluster_results[m]['score'] for m in method_names]
    
    # 按方法類型分組
    hier_methods = [m for m in method_names if 'hierarchical' in m]
    kmeans_methods = [m for m in method_names if 'kmeans' in m]
    dbscan_methods = [m for m in method_names if 'dbscan' in m]
    
    if hier_methods:
        hier_k = [int(m.split('_k')[1]) for m in hier_methods]
        hier_scores = [cluster_results[m]['score'] for m in hier_methods]
        ax1.plot(hier_k, hier_scores, 'o-', label='h', linewidth=2, markersize=6)
    
    if kmeans_methods:
        kmeans_k = [int(m.split('_k')[1]) for m in kmeans_methods]
        kmeans_scores = [cluster_results[m]['score'] for m in kmeans_methods]
        ax1.plot(kmeans_k, kmeans_scores, 's-', label='k', linewidth=2, markersize=6)
    
    if dbscan_methods:
        dbscan_scores = [cluster_results[m]['score'] for m in dbscan_methods]
        dbscan_k = [cluster_results[m]['k'] for m in dbscan_methods]
        ax1.scatter(dbscan_k, dbscan_scores, c='red', s=100, label='DBSCAN', zorder=5)
    
    ax1.set_xlabel('群組數 (k)')
    ax1.set_ylabel('精細分群評分')
    ax1.set_title('群組數 vs 精細分群評分')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 小群組比例比較
    ax2 = axes[0, 1]
    small_ratios = [cluster_results[m]['small_ratio'] for m in method_names]
    
    bars = ax2.bar(range(len(method_names)), small_ratios)
    ax2.set_xlabel('分群方法')
    ax2.set_ylabel('小群組比例 (≤5個數據)')
    ax2.set_title('各方法小群組比例')
    ax2.set_xticks(range(len(method_names)))
    
    # 簡化方法名稱標籤
    simplified_labels = []
    for m in method_names:
        if 'hierarchical' in m:
            label = 'h' + m.split('_k')[1] if '_k' in m else 'h'
        elif 'kmeans' in m:
            label = 'k' + m.split('_k')[1] if '_k' in m else 'k'
        else:
            label = m.replace('_', '\n')
        simplified_labels.append(label)
    
    ax2.set_xticklabels(simplified_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 為高比例標記顏色
    for i, ratio in enumerate(small_ratios):
        if ratio >= 0.8:
            bars[i].set_color('green')
        elif ratio >= 0.6:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    # 3. 平均群組大小比較
    ax3 = axes[0, 2]
    mean_sizes = [cluster_results[m]['mean_size'] for m in method_names]
    
    ax3.bar(range(len(method_names)), mean_sizes)
    ax3.set_xlabel('分群方法')
    ax3.set_ylabel('平均群組大小')
    ax3.set_title('各方法平均群組大小')
    ax3.set_xticks(range(len(method_names)))
    ax3.set_xticklabels(simplified_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. 最大群組大小比較
    ax4 = axes[0, 3]
    max_sizes = [cluster_results[m]['max_size'] for m in method_names]
    
    bars = ax4.bar(range(len(method_names)), max_sizes)
    ax4.set_xlabel('分群方法')
    ax4.set_ylabel('最大群組大小')
    ax4.set_title('各方法最大群組大小')
    ax4.set_xticks(range(len(method_names)))
    ax4.set_xticklabels(simplified_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 為小群組標記顏色
    for i, size in enumerate(max_sizes):
        if size <= 5:
            bars[i].set_color('green')
        elif size <= 10:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    # 5. 最佳分群的群組大小分布
    ax5 = axes[1, 0]
    best_method = max(method_names, key=lambda m: cluster_results[m]['score'])
    best_sizes = cluster_results[best_method]['sizes']
    
    # 按大小分類
    small_sizes = best_sizes[best_sizes <= 5]
    medium_sizes = best_sizes[(best_sizes > 5) & (best_sizes <= 10)]
    large_sizes = best_sizes[best_sizes > 10]
    
    ax5.hist([small_sizes, medium_sizes, large_sizes], 
             label=['小群組(≤5)', '中群組(6-10)', '大群組(>10)'], 
             alpha=0.7, bins=10)
    ax5.set_xlabel('群組大小')
    ax5.set_ylabel('群組數量')
    ax5.set_title(f'最佳方法: {best_method}\n群組大小分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 最佳分群的PCA散點圖
    ax6 = axes[1, 1]
    best_labels = cluster_results[best_method]['labels']
    
    # 使用主要特徵進行PCA
    main_cols = [col for col in df.columns if col.startswith('param_') or 
                col in ['annual_volatility', 'downside_risk']]
    available_cols = [col for col in main_cols if col in df.columns][:10]
    
    if len(available_cols) >= 2:
        X_pca = df[available_cols].fillna(0).values
        scaler = StandardScaler()
        X_pca_scaled = scaler.fit_transform(X_pca)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca_2d = pca.fit_transform(X_pca_scaled)
        
        for cluster_id in np.unique(best_labels):
            mask = best_labels == cluster_id
            ax6.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                       label=f'Cluster {cluster_id}', alpha=0.7)
        
        ax6.set_xlabel('PCA Component 1')
        ax6.set_ylabel('PCA Component 2')
        ax6.set_title(f'最佳分群: {best_method}\nPCA散點圖')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. 群組大小分布比較
    ax7 = axes[1, 2]
    # 比較前5個方法的群組大小分布
    top_methods = sorted(method_names, key=lambda m: cluster_results[m]['score'], reverse=True)[:5]
    
    for i, method in enumerate(top_methods):
        sizes = cluster_results[method]['sizes']
        # 簡化方法名稱
        if 'hierarchical' in method:
            label = 'h' + method.split('_k')[1] if '_k' in method else 'h'
        elif 'kmeans' in method:
            label = 'k' + method.split('_k')[1] if '_k' in method else 'k'
        else:
            label = method.replace('_', '\n')
        ax7.scatter([i] * len(sizes), sizes, alpha=0.6, label=label)
    
    ax7.set_xlabel('分群方法')
    ax7.set_ylabel('群組大小')
    ax7.set_title('前5名方法群組大小分布')
    ax7.set_xticks(range(len(top_methods)))
    
    top_simplified_labels = []
    for m in top_methods:
        if 'hierarchical' in m:
            label = 'h' + m.split('_k')[1] if '_k' in m else 'h'
        elif 'kmeans' in m:
            label = 'k' + m.split('_k')[1] if '_k' in m else 'k'
        else:
            label = m.replace('_', '\n')
        top_simplified_labels.append(label)
    
    ax7.set_xticklabels(top_simplified_labels, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 特徵重要性比較
    ax8 = axes[1, 3]
    if best_method in quality_analysis:
        feature_importance = quality_analysis[best_method]['feature_importance']
        important_features = [(f, v['f_stat']) for f, v in feature_importance.items() if v['significant']]
        important_features.sort(key=lambda x: x[1], reverse=True)
        
        if important_features:
            features, f_stats = zip(*important_features[:10])
            ax8.barh(range(len(features)), f_stats)
            ax8.set_yticks(range(len(features)))
            ax8.set_yticklabels(features)
            ax8.set_xlabel('F-statistic')
            ax8.set_title(f'重要特徵 ({best_method})')
            ax8.grid(True, alpha=0.3)
    
    # 9. 推薦方法比較
    ax9 = axes[2, 0]
    # 按不同標準排序
    by_score = sorted(method_names, key=lambda m: cluster_results[m]['score'], reverse=True)[:5]
    by_small_ratio = sorted(method_names, key=lambda m: cluster_results[m]['small_ratio'], reverse=True)[:5]
    by_mean_size = sorted(method_names, key=lambda m: cluster_results[m]['mean_size'])[:5]
    
    # 創建推薦表格
    recommendation_data = []
    for i, method in enumerate(by_score):
        # 簡化方法名稱
        if 'hierarchical' in method:
            method_label = 'h' + method.split('_k')[1] if '_k' in method else 'h'
        elif 'kmeans' in method:
            method_label = 'k' + method.split('_k')[1] if '_k' in method else 'k'
        else:
            method_label = method
        
        recommendation_data.append([
            method_label,
            f"{cluster_results[method]['score']:.3f}",
            f"{cluster_results[method]['small_ratio']:.3f}",
            f"{cluster_results[method]['mean_size']:.1f}",
            f"{cluster_results[method]['k']}"
        ])
    
    if recommendation_data:
        rec_df = pd.DataFrame(recommendation_data, 
                            columns=['方法', '評分', '小群組比例', '平均大小', '群組數'])
        ax9.axis('tight')
        ax9.axis('off')
        table = ax9.table(cellText=rec_df.values, colLabels=rec_df.columns, 
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax9.set_title('推薦方法 (按評分排序)')
    
    # 10. 小群組比例 vs 評分散點圖
    ax10 = axes[2, 1]
    small_ratios = [cluster_results[m]['small_ratio'] for m in method_names]
    scores = [cluster_results[m]['score'] for m in method_names]
    
    # 按方法類型標記顏色
    colors = []
    for method in method_names:
        if 'hierarchical' in method:
            colors.append('blue')
        elif 'kmeans' in method:
            colors.append('red')
        else:
            colors.append('green')
    
    ax10.scatter(small_ratios, scores, c=colors, s=100, alpha=0.7)
    ax10.set_xlabel('小群組比例')
    ax10.set_ylabel('精細分群評分')
    ax10.set_title('小群組比例 vs 評分')
    ax10.grid(True, alpha=0.3)
    
    # 添加方法標籤
    for i, method in enumerate(method_names):
        ax10.annotate(simplified_labels[i], 
                     (small_ratios[i], scores[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 11. 平均群組大小 vs 評分散點圖
    ax11 = axes[2, 2]
    mean_sizes = [cluster_results[m]['mean_size'] for m in method_names]
    
    ax11.scatter(mean_sizes, scores, c=colors, s=100, alpha=0.7)
    ax11.set_xlabel('平均群組大小')
    ax11.set_ylabel('精細分群評分')
    ax11.set_title('平均群組大小 vs 評分')
    ax11.grid(True, alpha=0.3)
    
    # 添加方法標籤
    for i, method in enumerate(method_names):
        ax11.annotate(simplified_labels[i], 
                     (mean_sizes[i], scores[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 12. 最終推薦
    ax12 = axes[2, 3]
    best_method_info = cluster_results[best_method]
    
    # 簡化最佳方法名稱
    if 'hierarchical' in best_method:
        best_method_label = 'h' + best_method.split('_k')[1] if '_k' in best_method else 'h'
    elif 'kmeans' in best_method:
        best_method_label = 'k' + best_method.split('_k')[1] if '_k' in best_method else 'k'
    else:
        best_method_label = best_method
    
    ax12.text(0.1, 0.9, f'策略: {strategy}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.8, f'最佳方法: {best_method_label}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.7, f'群組數: {best_method_info["k"]}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.6, f'平均大小: {best_method_info["mean_size"]:.1f}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.5, f'小群組比例: {best_method_info["small_ratio"]:.3f}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.4, f'精細評分: {best_method_info["score"]:.3f}', fontsize=12, transform=ax12.transAxes)
    
    # 推薦其他方法
    other_recommendations = []
    for method in method_names:
        if method != best_method:
            info = cluster_results[method]
            if info['small_ratio'] >= 0.7:  # 小群組比例高的
                # 簡化方法名稱
                if 'hierarchical' in method:
                    method_label = 'h' + method.split('_k')[1] if '_k' in method else 'h'
                elif 'kmeans' in method:
                    method_label = 'k' + method.split('_k')[1] if '_k' in method else 'k'
                else:
                    method_label = method
                other_recommendations.append(f'{method_label}: 比例{info["small_ratio"]}')
    
    if other_recommendations:
        ax12.text(0.1, 0.3, '其他推薦:', fontsize=12, transform=ax12.transAxes)
        for i, rec in enumerate(other_recommendations[:3]):
            ax12.text(0.1, 0.2 - i*0.05, rec, fontsize=10, transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    ax12.set_title('最終推薦')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_fine_grained_clustering.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def generate_fine_grained_recommendations(cluster_results, quality_analysis, strategy, datasource):
    """生成精細分群建議"""
    print(f"\n=== {strategy} - {datasource} 精細分群建議 ===")
    
    # 找出最佳分群
    best_score = -1
    best_method = None
    
    for method_name, result in cluster_results.items():
        if result['score'] > best_score:
            best_score = result['score']
            best_method = method_name
    
    print(f"最佳分群: {best_method}")
    print(f"群組數: {cluster_results[best_method]['k']}")
    print(f"平均群組大小: {cluster_results[best_method]['mean_size']:.1f}")
    print(f"最大群組大小: {cluster_results[best_method]['max_size']}")
    print(f"小群組比例: {cluster_results[best_method]['small_ratio']:.3f}")
    print(f"精細分群評分: {best_score:.3f}")
    print(f"群組大小分布: {cluster_results[best_method]['sizes']}")
    
    # 找出小群組比例高的方法
    print(f"\n小群組比例高的方法 (≥0.7):")
    high_small_ratio_methods = []
    
    for method_name, result in cluster_results.items():
        if result['small_ratio'] >= 0.7:
            high_small_ratio_methods.append({
                'method': method_name,
                'k': result['k'],
                'small_ratio': result['small_ratio'],
                'mean_size': result['mean_size'],
                'score': result['score']
            })
    
    # 按小群組比例排序
    high_small_ratio_methods.sort(key=lambda x: x['small_ratio'], reverse=True)
    
    for i, method in enumerate(high_small_ratio_methods[:5]):
        print(f"{i+1}. {method['method']}: 群組{method['k']}, 小群組比例{method['small_ratio']:.3f}, "
              f"平均大小{method['mean_size']:.1f}, 評分{method['score']:.3f}")
    
    # 找出平均群組大小小的方法
    print(f"\n平均群組大小小的方法 (≤3):")
    small_mean_size_methods = []
    
    for method_name, result in cluster_results.items():
        if result['mean_size'] <= 3:
            small_mean_size_methods.append({
                'method': method_name,
                'k': result['k'],
                'mean_size': result['mean_size'],
                'small_ratio': result['small_ratio'],
                'score': result['score']
            })
    
    # 按平均大小排序
    small_mean_size_methods.sort(key=lambda x: x['mean_size'])
    
    for i, method in enumerate(small_mean_size_methods[:5]):
        print(f"{i+1}. {method['method']}: 群組{method['k']}, 平均大小{method['mean_size']:.1f}, "
              f"小群組比例{method['small_ratio']:.3f}, 評分{method['score']:.3f}")
    
    return {
        'best_clustering': {
            'method': best_method,
            'k': cluster_results[best_method]['k'],
            'mean_size': cluster_results[best_method]['mean_size'],
            'max_size': cluster_results[best_method]['max_size'],
            'small_ratio': cluster_results[best_method]['small_ratio'],
            'score': best_score,
            'sizes': cluster_results[best_method]['sizes']
        },
        'high_small_ratio_methods': high_small_ratio_methods[:5],
        'small_mean_size_methods': small_mean_size_methods[:5]
    }

def main():
    """主函數"""
    print("開始精細分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    all_recommendations = {}
    
    for (strategy, datasource), df in data_dict.items():
        print(f"\n分析 {strategy} - {datasource}")
        
        # 選擇特徵（參數+風險，去除績效）
        param_cols = [col for col in df.columns if col.startswith('param_')]
        risk_cols = ['annual_volatility', 'downside_risk', 'var_95_annual', 'cvar_95_annual']
        
        all_feature_cols = param_cols + risk_cols
        available_features = [col for col in all_feature_cols if col in df.columns][:15]
        
        print(f"使用特徵數量: {len(available_features)}")
        
        if len(available_features) < 3:
            print("特徵不足，跳過")
            continue
        
        # 準備特徵數據
        X = df[available_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 尋找精細分群方法
        cluster_results = find_fine_grained_clusters(X_scaled, len(df), target_cluster_size=3)
        
        if not cluster_results:
            print("沒有找到合適的精細分群方法")
            continue
        
        # 分析分群品質
        quality_analysis = analyze_fine_grained_quality(cluster_results, df, available_features)
        
        # 創建可視化
        create_fine_grained_visualizations(df, cluster_results, quality_analysis, 
                                         available_features, strategy, datasource)
        
        # 生成建議
        recommendations = generate_fine_grained_recommendations(cluster_results, quality_analysis, 
                                                              strategy, datasource)
        
        # 保存最佳分群結果
        best_method = recommendations['best_clustering']['method']
        best_labels = cluster_results[best_method]['labels']
        
        result_df = df.copy()
        result_df['fine_grained_cluster'] = best_labels
        result_df['clustering_method'] = best_method
        result_df['k_value'] = cluster_results[best_method]['k']
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_fine_grained_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_recommendations[(strategy, datasource)] = recommendations
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print(f"\n=== 精細分群總結 ===")
    for (strategy, datasource), recs in all_recommendations.items():
        best = recs['best_clustering']
        print(f"\n{strategy} - {datasource}:")
        print(f"  最佳分群: {best['method']}")
        print(f"  群組數: {best['k']}")
        print(f"  平均群組大小: {best['mean_size']:.1f}")
        print(f"  小群組比例: {best['small_ratio']:.3f}")
        print(f"  精細評分: {best['score']:.3f}")
        
        if recs['high_small_ratio_methods']:
            high_ratio_rec = recs['high_small_ratio_methods'][0]
            print(f"  高小群組比例推薦: {high_ratio_rec['method']} (比例{high_ratio_rec['small_ratio']:.3f})")
    
    print(f"\n所有結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 