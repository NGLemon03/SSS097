# v11: 小群組分析，專門針對小群組需求設計
# 主要功能：小群組生成、群組大小控制、多方法比較、統計分析
# 產生大量小群組以滿足細粒度分析需求

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
OUTPUT_DIR = 'results/small_cluster_analysis'
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

def find_small_clusters(X_scaled, max_cluster_size=100, min_clusters=3):
    """尋找產生小群組的分群方法"""
    print("尋找產生小群組的分群方法...")
    
    results = {}
    n_samples = len(X_scaled)
    
    # 測試更多群組數，確保群組不會太大
    max_k = min(20, n_samples // max_cluster_size)
    
    for k in range(min_clusters, max_k + 1):
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
            
            # 檢查是否有群組太大
            hier_max_size = max(hier_sizes)
            kmeans_max_size = max(kmeans_sizes)
            
            # 只保留群組大小合適的分群
            if hier_max_size <= max_cluster_size:
                hier_silhouette = silhouette_score(X_scaled, hier_labels)
                hier_balance = 1 / (1 + np.std(hier_sizes) / np.mean(hier_sizes))
                hier_score = hier_silhouette * 0.6 + hier_balance * 0.4
                
                results[f'hierarchical_k{k}'] = {
                    'labels': hier_labels,
                    'silhouette': hier_silhouette,
                    'balance': hier_balance,
                    'score': hier_score,
                    'sizes': hier_sizes,
                    'max_size': hier_max_size,
                    'method': 'hierarchical',
                    'k': k
                }
            
            if kmeans_max_size <= max_cluster_size:
                kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
                kmeans_balance = 1 / (1 + np.std(kmeans_sizes) / np.mean(kmeans_sizes))
                kmeans_score = kmeans_silhouette * 0.6 + kmeans_balance * 0.4
                
                results[f'kmeans_k{k}'] = {
                    'labels': kmeans_labels,
                    'silhouette': kmeans_silhouette,
                    'balance': kmeans_balance,
                    'score': kmeans_score,
                    'sizes': kmeans_sizes,
                    'max_size': kmeans_max_size,
                    'method': 'kmeans',
                    'k': k
                }
            
        except Exception as e:
            print(f"k={k} 時發生錯誤: {e}")
            continue
    
    # 嘗試DBSCAN
    try:
        # 自動選擇eps
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        eps = np.percentile(distances[:, -1], 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # 檢查DBSCAN結果
        unique_labels = np.unique(dbscan_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:  # 沒有雜訊點
            dbscan_sizes = np.bincount(dbscan_labels)
            dbscan_max_size = max(dbscan_sizes)
            
            if dbscan_max_size <= max_cluster_size:
                dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
                dbscan_balance = 1 / (1 + np.std(dbscan_sizes) / np.mean(dbscan_sizes))
                dbscan_score = dbscan_silhouette * 0.6 + dbscan_balance * 0.4
                
                results['dbscan'] = {
                    'labels': dbscan_labels,
                    'silhouette': dbscan_silhouette,
                    'balance': dbscan_balance,
                    'score': dbscan_score,
                    'sizes': dbscan_sizes,
                    'max_size': dbscan_max_size,
                    'method': 'dbscan',
                    'k': len(unique_labels)
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
            'n_significant_features': sum(1 for f in feature_importance.values() if f['significant'])
        }
    
    return quality_analysis

def create_small_cluster_visualizations(df, cluster_results, quality_analysis, feature_names, strategy, datasource):
    """創建小群組分群可視化"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 1. 群組大小分布
    ax1 = axes[0, 0]
    method_names = list(cluster_results.keys())
    max_sizes = [cluster_results[m]['max_size'] for m in method_names]
    
    bars = ax1.bar(range(len(method_names)), max_sizes)
    ax1.set_xlabel('分群方法')
    ax1.set_ylabel('最大群組大小')
    ax1.set_title('各方法最大群組大小')
    ax1.set_xticks(range(len(method_names)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in method_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 為小群組標記顏色
    for i, size in enumerate(max_sizes):
        if size <= 50:
            bars[i].set_color('green')
        elif size <= 100:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    # 2. 綜合評分比較
    ax2 = axes[0, 1]
    scores = [cluster_results[m]['score'] for m in method_names]
    
    ax2.bar(range(len(method_names)), scores)
    ax2.set_xlabel('分群方法')
    ax2.set_ylabel('綜合評分')
    ax2.set_title('各方法綜合評分')
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels([m.replace('_', '\n') for m in method_names], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 群組數量比較
    ax3 = axes[0, 2]
    k_values = [cluster_results[m]['k'] for m in method_names]
    
    ax3.bar(range(len(method_names)), k_values)
    ax3.set_xlabel('分群方法')
    ax3.set_ylabel('群組數量')
    ax3.set_title('各方法群組數量')
    ax3.set_xticks(range(len(method_names)))
    ax3.set_xticklabels([m.replace('_', '\n') for m in method_names], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. 群組大小詳細分布 (最佳方法)
    ax4 = axes[0, 3]
    best_method = max(method_names, key=lambda m: cluster_results[m]['score'])
    best_sizes = cluster_results[best_method]['sizes']
    
    ax4.bar(range(len(best_sizes)), best_sizes)
    ax4.set_xlabel('群組編號')
    ax4.set_ylabel('群組大小')
    ax4.set_title(f'最佳方法: {best_method}\n群組大小分布')
    ax4.grid(True, alpha=0.3)
    
    # 5. Silhouette Score比較
    ax5 = axes[1, 0]
    silhouettes = [cluster_results[m]['silhouette'] for m in method_names]
    
    ax5.bar(range(len(method_names)), silhouettes)
    ax5.set_xlabel('分群方法')
    ax5.set_ylabel('Silhouette Score')
    ax5.set_title('各方法Silhouette Score')
    ax5.set_xticks(range(len(method_names)))
    ax5.set_xticklabels([m.replace('_', '\n') for m in method_names], rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # 6. 平衡性比較
    ax6 = axes[1, 1]
    balances = [cluster_results[m]['balance'] for m in method_names]
    
    ax6.bar(range(len(method_names)), balances)
    ax6.set_xlabel('分群方法')
    ax6.set_ylabel('平衡性分數')
    ax6.set_title('各方法平衡性')
    ax6.set_xticks(range(len(method_names)))
    ax6.set_xticklabels([m.replace('_', '\n') for m in method_names], rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # 7. 最佳分群的PCA散點圖
    ax7 = axes[1, 2]
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
            ax7.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                       label=f'Cluster {cluster_id}', alpha=0.7)
        
        ax7.set_xlabel('PCA Component 1')
        ax7.set_ylabel('PCA Component 2')
        ax7.set_title(f'最佳分群: {best_method}\nPCA散點圖')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. 群組大小分布比較
    ax8 = axes[1, 3]
    # 比較前5個方法的群組大小分布
    top_methods = sorted(method_names, key=lambda m: cluster_results[m]['score'], reverse=True)[:5]
    
    for i, method in enumerate(top_methods):
        sizes = cluster_results[method]['sizes']
        ax8.scatter([i] * len(sizes), sizes, alpha=0.6, label=method.replace('_', '\n'))
    
    ax8.set_xlabel('分群方法')
    ax8.set_ylabel('群組大小')
    ax8.set_title('前5名方法群組大小分布')
    ax8.set_xticks(range(len(top_methods)))
    ax8.set_xticklabels([m.replace('_', '\n') for m in top_methods], rotation=45, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 特徵重要性比較
    ax9 = axes[2, 0]
    if best_method in quality_analysis:
        feature_importance = quality_analysis[best_method]['feature_importance']
        important_features = [(f, v['f_stat']) for f, v in feature_importance.items() if v['significant']]
        important_features.sort(key=lambda x: x[1], reverse=True)
        
        if important_features:
            features, f_stats = zip(*important_features[:10])
            ax9.barh(range(len(features)), f_stats)
            ax9.set_yticks(range(len(features)))
            ax9.set_yticklabels(features)
            ax9.set_xlabel('F-statistic')
            ax9.set_title(f'重要特徵 ({best_method})')
            ax9.grid(True, alpha=0.3)
    
    # 10. 推薦方法比較
    ax10 = axes[2, 1]
    # 按不同標準排序
    by_score = sorted(method_names, key=lambda m: cluster_results[m]['score'], reverse=True)[:5]
    by_size = sorted(method_names, key=lambda m: cluster_results[m]['max_size'])[:5]
    by_balance = sorted(method_names, key=lambda m: cluster_results[m]['balance'], reverse=True)[:5]
    
    # 創建推薦表格
    recommendation_data = []
    for i, method in enumerate(by_score):
        recommendation_data.append([
            method,
            cluster_results[method]['score'],
            cluster_results[method]['max_size'],
            cluster_results[method]['balance'],
            cluster_results[method]['k']
        ])
    
    if recommendation_data:
        rec_df = pd.DataFrame(recommendation_data, 
                            columns=['方法', '評分', '最大群組', '平衡性', '群組數'])
        ax10.axis('tight')
        ax10.axis('off')
        table = ax10.table(cellText=rec_df.values, colLabels=rec_df.columns, 
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax10.set_title('推薦方法 (按評分排序)')
    
    # 11. 小群組方法比較
    ax11 = axes[2, 2]
    small_cluster_methods = [m for m in method_names if cluster_results[m]['max_size'] <= 50]
    
    if small_cluster_methods:
        small_scores = [cluster_results[m]['score'] for m in small_cluster_methods]
        small_sizes = [cluster_results[m]['max_size'] for m in small_cluster_methods]
        
        ax11.scatter(small_sizes, small_scores, s=100, alpha=0.7)
        for i, method in enumerate(small_cluster_methods):
            ax11.annotate(method.replace('_', '\n'), 
                         (small_sizes[i], small_scores[i]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax11.set_xlabel('最大群組大小')
        ax11.set_ylabel('綜合評分')
        ax11.set_title('小群組方法比較\n(最大群組≤50)')
        ax11.grid(True, alpha=0.3)
    
    # 12. 最終推薦
    ax12 = axes[2, 3]
    best_method_info = cluster_results[best_method]
    
    ax12.text(0.1, 0.9, f'策略: {strategy}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.8, f'最佳方法: {best_method}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.7, f'群組數: {best_method_info["k"]}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.6, f'最大群組: {best_method_info["max_size"]}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.5, f'綜合評分: {best_method_info["score"]:.3f}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.4, f'平衡性: {best_method_info["balance"]:.3f}', fontsize=12, transform=ax12.transAxes)
    
    # 推薦小群組方法
    small_methods = [m for m in method_names if cluster_results[m]['max_size'] <= 50]
    if small_methods:
        ax12.text(0.1, 0.3, '小群組推薦:', fontsize=12, transform=ax12.transAxes)
        for i, method in enumerate(small_methods[:3]):
            info = cluster_results[method]
            ax12.text(0.1, 0.2 - i*0.05, 
                     f'{method}: 群組{info["k"]}, 最大{info["max_size"]}', 
                     fontsize=10, transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    ax12.set_title('最終推薦')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_small_cluster_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def generate_small_cluster_recommendations(cluster_results, quality_analysis, strategy, datasource):
    """生成小群組分群建議"""
    print(f"\n=== {strategy} - {datasource} 小群組分群建議 ===")
    
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
    print(f"綜合評分: {best_score:.3f}")
    print(f"Silhouette Score: {cluster_results[best_method]['silhouette']:.3f}")
    print(f"平衡性分數: {cluster_results[best_method]['balance']:.3f}")
    print(f"群組大小: {cluster_results[best_method]['sizes']}")
    
    # 找出小群組方法
    print(f"\n小群組方法 (最大群組≤50):")
    small_methods = []
    
    for method_name, result in cluster_results.items():
        if result['max_size'] <= 50:
            small_methods.append({
                'method': method_name,
                'k': result['k'],
                'max_size': result['max_size'],
                'score': result['score'],
                'balance': result['balance'],
                'sizes': result['sizes']
            })
    
    # 按評分排序
    small_methods.sort(key=lambda x: x['score'], reverse=True)
    
    for i, method in enumerate(small_methods[:5]):
        print(f"{i+1}. {method['method']}: 群組{method['k']}, 最大群組{method['max_size']}, "
              f"評分{method['score']:.3f}, 平衡性{method['balance']:.3f}")
    
    # 找出平衡性好的小群組方法
    print(f"\n平衡性好的小群組方法 (平衡性>0.6):")
    balanced_small_methods = [m for m in small_methods if m['balance'] > 0.6]
    
    for i, method in enumerate(balanced_small_methods[:5]):
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
        'small_methods': small_methods[:5],
        'balanced_small_methods': balanced_small_methods[:5]
    }

def main():
    """主函數"""
    print("開始小群組分群分析...")
    
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
        
        # 尋找小群組分群方法
        cluster_results = find_small_clusters(X_scaled, max_cluster_size=100, min_clusters=3)
        
        if not cluster_results:
            print("沒有找到合適的小群組分群方法")
            continue
        
        # 分析分群品質
        quality_analysis = analyze_cluster_quality(cluster_results, df, available_features)
        
        # 創建可視化
        create_small_cluster_visualizations(df, cluster_results, quality_analysis, 
                                          available_features, strategy, datasource)
        
        # 生成建議
        recommendations = generate_small_cluster_recommendations(cluster_results, quality_analysis, 
                                                               strategy, datasource)
        
        # 保存最佳分群結果
        best_method = recommendations['best_clustering']['method']
        best_labels = cluster_results[best_method]['labels']
        
        result_df = df.copy()
        result_df['small_cluster'] = best_labels
        result_df['clustering_method'] = best_method
        result_df['k_value'] = cluster_results[best_method]['k']
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_small_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_recommendations[(strategy, datasource)] = recommendations
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print(f"\n=== 小群組分群總結 ===")
    for (strategy, datasource), recs in all_recommendations.items():
        best = recs['best_clustering']
        print(f"\n{strategy} - {datasource}:")
        print(f"  最佳分群: {best['method']}")
        print(f"  群組數: {best['k']}")
        print(f"  最大群組: {best['max_size']}")
        print(f"  綜合評分: {best['score']:.3f}")
        print(f"  群組大小: {best['sizes']}")
        
        if recs['small_methods']:
            small_rec = recs['small_methods'][0]
            print(f"  小群組推薦: {small_rec['method']} (群組{small_rec['k']}, 最大群組{small_rec['max_size']})")
    
    print(f"\n所有結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 