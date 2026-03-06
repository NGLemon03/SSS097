# v12: 實用分群分析，結合理論與實務需求
# 主要功能：實用分群策略、多維度評估、實務建議、效果驗證
# 提供可直接應用的分群解決方案

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
OUTPUT_DIR = 'results/practical_cluster_analysis'
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

def find_practical_clusters(X_scaled, n_samples):
    """尋找實用的分群方法，提供多種選擇"""
    print("尋找實用的分群方法...")
    
    results = {}
    
    # 計算合理的群組數範圍
    min_k = 3
    max_k = min(20, n_samples // 20)  # 每個群組至少20個樣本
    
    print(f"測試群組數範圍: {min_k} 到 {max_k}")
    
    for k in range(min_k, max_k + 1):
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
            
            # 計算指標
            hier_silhouette = silhouette_score(X_scaled, hier_labels)
            kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
            
            # 計算平衡性
            hier_balance = 1 / (1 + np.std(hier_sizes) / np.mean(hier_sizes))
            kmeans_balance = 1 / (1 + np.std(kmeans_sizes) / np.mean(kmeans_sizes))
            
            # 計算實用性分數 (考慮群組大小分布)
            hier_max_size = max(hier_sizes)
            kmeans_max_size = max(kmeans_sizes)
            
            # 實用性分數 = Silhouette * 0.4 + Balance * 0.4 + Size_Penalty * 0.2
            hier_size_penalty = 1 / (1 + hier_max_size / 100)  # 群組越大，懲罰越大
            kmeans_size_penalty = 1 / (1 + kmeans_max_size / 100)
            
            hier_score = hier_silhouette * 0.4 + hier_balance * 0.4 + hier_size_penalty * 0.2
            kmeans_score = kmeans_silhouette * 0.4 + kmeans_balance * 0.4 + kmeans_size_penalty * 0.2
            
            results[f'hierarchical_k{k}'] = {
                'labels': hier_labels,
                'silhouette': hier_silhouette,
                'balance': hier_balance,
                'score': hier_score,
                'sizes': hier_sizes,
                'max_size': hier_max_size,
                'method': 'hierarchical',
                'k': k,
                'size_penalty': hier_size_penalty
            }
            
            results[f'kmeans_k{k}'] = {
                'labels': kmeans_labels,
                'silhouette': kmeans_silhouette,
                'balance': kmeans_balance,
                'score': kmeans_score,
                'sizes': kmeans_sizes,
                'max_size': kmeans_max_size,
                'method': 'kmeans',
                'k': k,
                'size_penalty': kmeans_size_penalty
            }
            
        except Exception as e:
            print(f"k={k} 時發生錯誤: {e}")
            continue
    
    # 嘗試DBSCAN
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        eps = np.percentile(distances[:, -1], 85)  # 調整eps
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # 檢查DBSCAN結果
        unique_labels = np.unique(dbscan_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:  # 沒有雜訊點
            dbscan_sizes = np.bincount(dbscan_labels)
            dbscan_max_size = max(dbscan_sizes)
            dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
            dbscan_balance = 1 / (1 + np.std(dbscan_sizes) / np.mean(dbscan_sizes))
            dbscan_size_penalty = 1 / (1 + dbscan_max_size / 100)
            dbscan_score = dbscan_silhouette * 0.4 + dbscan_balance * 0.4 + dbscan_size_penalty * 0.2
            
            results['dbscan'] = {
                'labels': dbscan_labels,
                'silhouette': dbscan_silhouette,
                'balance': dbscan_balance,
                'score': dbscan_score,
                'sizes': dbscan_sizes,
                'max_size': dbscan_max_size,
                'method': 'dbscan',
                'k': len(unique_labels),
                'size_penalty': dbscan_size_penalty
            }
    except Exception as e:
        print(f"DBSCAN 發生錯誤: {e}")
    
    return results

def analyze_cluster_quality(cluster_results, df, feature_names):
    """分析分群品質"""
    print("\n=== 分群品質分析 ===")
    
    quality_analysis = {}
    
    for method_name, result in cluster_results.items():
        labels = result['labels']
        
        # 基本統計
        unique_labels = np.unique(labels)
        cluster_sizes = result['sizes']
        
        # 特徵重要性分析
        feature_importance = {}
        for feature in feature_names:
            if feature in df.columns:
                feature_values = df[feature].values
                
                # ANOVA檢定
                try:
                    groups = [feature_values[labels == i] for i in unique_labels]
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # 計算群組間變異係數
                    group_means = [np.mean(group) for group in groups]
                    cv_between = np.std(group_means) / abs(np.mean(feature_values)) if np.mean(feature_values) != 0 else 0
                    
                    feature_importance[feature] = {
                        'f_stat': f_stat,
                        'p_value': p_value,
                        'cv_between': cv_between,
                        'significant': p_value < 0.05
                    }
                except:
                    feature_importance[feature] = {
                        'f_stat': 0,
                        'p_value': 1,
                        'cv_between': 0,
                        'significant': False
                    }
        
        quality_analysis[method_name] = {
            'silhouette': result['silhouette'],
            'balance': result['balance'],
            'score': result['score'],
            'cluster_sizes': cluster_sizes,
            'max_size': result['max_size'],
            'feature_importance': feature_importance,
            'n_significant_features': sum(1 for f in feature_importance.values() if f['significant']),
            'size_penalty': result['size_penalty']
        }
    
    return quality_analysis

def create_practical_visualizations(df, cluster_results, quality_analysis, feature_names, strategy, datasource):
    """創建實用分群可視化"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 1. 群組數 vs 實用性評分
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
        dbscan_score = cluster_results[dbscan_methods[0]]['score']
        dbscan_k = cluster_results[dbscan_methods[0]]['k']
        ax1.scatter(dbscan_k, dbscan_score, c='red', s=100, label='DBSCAN', zorder=5)
    
    ax1.set_xlabel('群組數 (k)')
    ax1.set_ylabel('實用性評分')
    ax1.set_title('群組數 vs 實用性評分')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 群組大小分布
    ax2 = axes[0, 1]
    max_sizes = [cluster_results[m]['max_size'] for m in method_names]
    
    bars = ax2.bar(range(len(method_names)), max_sizes)
    ax2.set_xlabel('分群方法')
    ax2.set_ylabel('最大群組大小')
    ax2.set_title('各方法最大群組大小')
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
    
    # 為不同大小範圍標記顏色
    for i, size in enumerate(max_sizes):
        if size <= 100:
            bars[i].set_color('green')
        elif size <= 200:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    # 3. 平衡性比較
    ax3 = axes[0, 2]
    balances = [cluster_results[m]['balance'] for m in method_names]
    
    ax3.bar(range(len(method_names)), balances)
    ax3.set_xlabel('分群方法')
    ax3.set_ylabel('平衡性分數')
    ax3.set_title('各方法平衡性')
    ax3.set_xticks(range(len(method_names)))
    ax3.set_xticklabels(simplified_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. 群組數量比較
    ax4 = axes[0, 3]
    k_values = [cluster_results[m]['k'] for m in method_names]
    
    ax4.bar(range(len(method_names)), k_values)
    ax4.set_xlabel('分群方法')
    ax4.set_ylabel('群組數量')
    ax4.set_title('各方法群組數量')
    ax4.set_xticks(range(len(method_names)))
    ax4.set_xticklabels(simplified_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 5. 最佳分群的群組大小分布
    ax5 = axes[1, 0]
    best_method = max(method_names, key=lambda m: cluster_results[m]['score'])
    best_sizes = cluster_results[best_method]['sizes']
    
    ax5.bar(range(len(best_sizes)), best_sizes)
    ax5.set_xlabel('群組編號')
    ax5.set_ylabel('群組大小')
    ax5.set_title(f'最佳方法: {best_method}\n群組大小分布')
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
    
    # 簡化方法名稱標籤
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
    by_size = sorted(method_names, key=lambda m: cluster_results[m]['max_size'])[:5]
    by_balance = sorted(method_names, key=lambda m: cluster_results[m]['balance'], reverse=True)[:5]
    
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
            f"{cluster_results[method]['max_size']}",
            f"{cluster_results[method]['balance']:.3f}",
            f"{cluster_results[method]['k']}"
        ])
    
    if recommendation_data:
        rec_df = pd.DataFrame(recommendation_data, 
                            columns=['方法', '評分', '最大群組', '平衡性', '群組數'])
        ax9.axis('tight')
        ax9.axis('off')
        table = ax9.table(cellText=rec_df.values, colLabels=rec_df.columns, 
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax9.set_title('推薦方法 (按評分排序)')
    
    # 10. 群組大小 vs 評分散點圖
    ax10 = axes[2, 1]
    max_sizes = [cluster_results[m]['max_size'] for m in method_names]
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
    
    ax10.scatter(max_sizes, scores, c=colors, s=100, alpha=0.7)
    ax10.set_xlabel('最大群組大小')
    ax10.set_ylabel('實用性評分')
    ax10.set_title('群組大小 vs 評分')
    ax10.grid(True, alpha=0.3)
    
    # 添加方法標籤（簡化名稱）
    for i, method in enumerate(method_names):
        # 簡化方法名稱
        if 'hierarchical' in method:
            label = 'h' + method.split('_k')[1] if '_k' in method else 'h'
        elif 'kmeans' in method:
            label = 'k' + method.split('_k')[1] if '_k' in method else 'k'
        else:
            label = method.replace('_', '\n')
        
        ax10.annotate(label, 
                     (max_sizes[i], scores[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 11. 平衡性 vs 評分散點圖
    ax11 = axes[2, 2]
    balances = [cluster_results[m]['balance'] for m in method_names]
    
    ax11.scatter(balances, scores, c=colors, s=100, alpha=0.7)
    ax11.set_xlabel('平衡性分數')
    ax11.set_ylabel('實用性評分')
    ax11.set_title('平衡性 vs 評分')
    ax11.grid(True, alpha=0.3)
    
    # 添加方法標籤（簡化名稱）
    for i, method in enumerate(method_names):
        # 簡化方法名稱
        if 'hierarchical' in method:
            label = 'h' + method.split('_k')[1] if '_k' in method else 'h'
        elif 'kmeans' in method:
            label = 'k' + method.split('_k')[1] if '_k' in method else 'k'
        else:
            label = method.replace('_', '\n')
        
        ax11.annotate(label, 
                     (balances[i], scores[i]), 
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
    ax12.text(0.1, 0.6, f'最大群組: {best_method_info["max_size"]}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.5, f'實用性評分: {best_method_info["score"]:.3f}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.4, f'平衡性: {best_method_info["balance"]:.3f}', fontsize=12, transform=ax12.transAxes)
    
    # 推薦其他方法
    other_recommendations = []
    for method in method_names:
        if method != best_method:
            info = cluster_results[method]
            if info['max_size'] <= 150:  # 群組不太大的
                # 簡化方法名稱
                if 'hierarchical' in method:
                    method_label = 'h' + method.split('_k')[1] if '_k' in method else 'h'
                elif 'kmeans' in method:
                    method_label = 'k' + method.split('_k')[1] if '_k' in method else 'k'
                else:
                    method_label = method
                other_recommendations.append(f'{method_label}: 群組{info["k"]}, 最大{info["max_size"]}')
    
    if other_recommendations:
        ax12.text(0.1, 0.3, '其他推薦:', fontsize=12, transform=ax12.transAxes)
        for i, rec in enumerate(other_recommendations[:3]):
            ax12.text(0.1, 0.2 - i*0.05, rec, fontsize=10, transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    ax12.set_title('最終推薦')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_practical_clustering.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def generate_practical_recommendations(cluster_results, quality_analysis, strategy, datasource):
    """生成實用分群建議"""
    print(f"\n=== {strategy} - {datasource} 實用分群建議 ===")
    
    # 找出最佳分群
    best_score = -1
    best_method = None
    
    for method_name, result in cluster_results.items():
        if result['score'] > best_score:
            best_score = result['score']
            best_method = method_name
    
    print(f"最佳分群: {best_method}")
    print(f"群組數: {cluster_results[best_method]['k']}")
    print(f"最大群組大小: {cluster_results[best_method]['max_size']}")
    print(f"實用性評分: {best_score:.3f}")
    print(f"Silhouette Score: {cluster_results[best_method]['silhouette']:.3f}")
    print(f"平衡性分數: {cluster_results[best_method]['balance']:.3f}")
    print(f"群組大小: {cluster_results[best_method]['sizes']}")
    
    # 找出不同類型的推薦
    print(f"\n按評分排序的推薦方法:")
    sorted_methods = sorted(cluster_results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for i, (method, result) in enumerate(sorted_methods[:5]):
        print(f"{i+1}. {method}: 群組{result['k']}, 最大群組{result['max_size']}, "
              f"評分{result['score']:.3f}, 平衡性{result['balance']:.3f}")
    
    # 找出群組大小適中的方法
    print(f"\n群組大小適中的方法 (最大群組≤150):")
    medium_size_methods = []
    
    for method_name, result in cluster_results.items():
        if result['max_size'] <= 150:
            medium_size_methods.append({
                'method': method_name,
                'k': result['k'],
                'max_size': result['max_size'],
                'score': result['score'],
                'balance': result['balance']
            })
    
    # 按評分排序
    medium_size_methods.sort(key=lambda x: x['score'], reverse=True)
    
    for i, method in enumerate(medium_size_methods[:5]):
        print(f"{i+1}. {method['method']}: 群組{method['k']}, 最大群組{method['max_size']}, "
              f"評分{method['score']:.3f}, 平衡性{method['balance']:.3f}")
    
    # 找出平衡性好的方法
    print(f"\n平衡性好的方法 (平衡性>0.6):")
    balanced_methods = []
    
    for method_name, result in cluster_results.items():
        if result['balance'] > 0.6:
            balanced_methods.append({
                'method': method_name,
                'k': result['k'],
                'max_size': result['max_size'],
                'score': result['score'],
                'balance': result['balance']
            })
    
    # 按平衡性排序
    balanced_methods.sort(key=lambda x: x['balance'], reverse=True)
    
    for i, method in enumerate(balanced_methods[:5]):
        print(f"{i+1}. {method['method']}: 群組{method['k']}, 最大群組{method['max_size']}, "
              f"平衡性{method['balance']:.3f}, 評分{method['score']:.3f}")
    
    return {
        'best_clustering': {
            'method': best_method,
            'k': cluster_results[best_method]['k'],
            'max_size': cluster_results[best_method]['max_size'],
            'score': best_score,
            'sizes': cluster_results[best_method]['sizes']
        },
        'top_methods': sorted_methods[:5],
        'medium_size_methods': medium_size_methods[:5],
        'balanced_methods': balanced_methods[:5]
    }

def main():
    """主函數"""
    print("開始實用分群分析...")
    
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
        
        # 尋找實用分群方法
        cluster_results = find_practical_clusters(X_scaled, len(df))
        
        if not cluster_results:
            print("沒有找到合適的分群方法")
            continue
        
        # 分析分群品質
        quality_analysis = analyze_cluster_quality(cluster_results, df, available_features)
        
        # 創建可視化
        create_practical_visualizations(df, cluster_results, quality_analysis, 
                                      available_features, strategy, datasource)
        
        # 生成建議
        recommendations = generate_practical_recommendations(cluster_results, quality_analysis, 
                                                           strategy, datasource)
        
        # 保存最佳分群結果
        best_method = recommendations['best_clustering']['method']
        best_labels = cluster_results[best_method]['labels']
        
        result_df = df.copy()
        result_df['practical_cluster'] = best_labels
        result_df['clustering_method'] = best_method
        result_df['k_value'] = cluster_results[best_method]['k']
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_practical_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_recommendations[(strategy, datasource)] = recommendations
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print(f"\n=== 實用分群總結 ===")
    for (strategy, datasource), recs in all_recommendations.items():
        best = recs['best_clustering']
        print(f"\n{strategy} - {datasource}:")
        print(f"  最佳分群: {best['method']}")
        print(f"  群組數: {best['k']}")
        print(f"  最大群組: {best['max_size']}")
        print(f"  實用性評分: {best['score']:.3f}")
        print(f"  群組大小: {best['sizes']}")
        
        if recs['medium_size_methods']:
            medium_rec = recs['medium_size_methods'][0]
            print(f"  適中群組推薦: {medium_rec['method']} (群組{medium_rec['k']}, 最大群組{medium_rec['max_size']})")
    
    print(f"\n所有結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 