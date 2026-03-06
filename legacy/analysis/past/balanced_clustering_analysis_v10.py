# v10: 平衡分群分析，追求群組大小平衡與效果最佳化
# 主要功能：平衡群組大小、多方法比較、統計驗證、視覺化分析
# 提供平衡且有效的分群解決方案

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
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
OUTPUT_DIR = 'results/balanced_clustering'
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

def find_optimal_clusters(X_scaled, max_clusters=15, min_cluster_size=10):
    """尋找最佳群組數，考慮群組大小平衡"""
    print("尋找最佳群組數...")
    
    results = {}
    
    # 測試更多群組數
    for k in range(2, min(max_clusters + 1, len(X_scaled) // min_cluster_size)):
        try:
            # 階層式分群
            Z = linkage(X_scaled, method='ward')
            hier_labels = fcluster(Z, t=k, criterion='maxclust')
            
            # KMeans分群
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            
            # 計算指標
            hier_silhouette = silhouette_score(X_scaled, hier_labels)
            kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
            
            # 計算群組大小平衡性
            hier_sizes = np.bincount(hier_labels)
            kmeans_sizes = np.bincount(kmeans_labels)
            
            # 平衡性分數 (變異係數的倒數)
            hier_balance = 1 / (1 + np.std(hier_sizes) / np.mean(hier_sizes))
            kmeans_balance = 1 / (1 + np.std(kmeans_sizes) / np.mean(kmeans_sizes))
            
            # 綜合評分 (Silhouette * 0.6 + Balance * 0.4)
            hier_score = hier_silhouette * 0.6 + hier_balance * 0.4
            kmeans_score = kmeans_silhouette * 0.6 + kmeans_balance * 0.4
            
            results[k] = {
                'hierarchical': {
                    'labels': hier_labels,
                    'silhouette': hier_silhouette,
                    'balance': hier_balance,
                    'score': hier_score,
                    'sizes': hier_sizes
                },
                'kmeans': {
                    'labels': kmeans_labels,
                    'silhouette': kmeans_silhouette,
                    'balance': kmeans_balance,
                    'score': kmeans_score,
                    'sizes': kmeans_sizes
                }
            }
            
        except Exception as e:
            print(f"k={k} 時發生錯誤: {e}")
            continue
    
    return results

def analyze_cluster_quality(cluster_results, df, feature_names):
    """分析分群品質"""
    print("\n=== 分群品質分析 ===")
    
    quality_analysis = {}
    
    for k, methods in cluster_results.items():
        quality_analysis[k] = {}
        
        for method_name, result in methods.items():
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
            
            quality_analysis[k][method_name] = {
                'silhouette': result['silhouette'],
                'balance': result['balance'],
                'score': result['score'],
                'cluster_sizes': cluster_sizes,
                'feature_importance': feature_importance,
                'n_significant_features': sum(1 for f in feature_importance.values() if f['significant'])
            }
    
    return quality_analysis

def create_balanced_visualizations(df, cluster_results, quality_analysis, feature_names, strategy, datasource):
    """創建平衡分群可視化"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 1. 群組數 vs 綜合評分
    ax1 = axes[0, 0]
    k_values = list(cluster_results.keys())
    hier_scores = [cluster_results[k]['hierarchical']['score'] for k in k_values]
    kmeans_scores = [cluster_results[k]['kmeans']['score'] for k in k_values]
    
    ax1.plot(k_values, hier_scores, 'o-', label='Hierarchical', linewidth=2, markersize=6)
    ax1.plot(k_values, kmeans_scores, 's-', label='KMeans', linewidth=2, markersize=6)
    ax1.set_xlabel('群組數 (k)')
    ax1.set_ylabel('綜合評分')
    ax1.set_title('群組數 vs 綜合評分')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 群組數 vs Silhouette Score
    ax2 = axes[0, 1]
    hier_silhouettes = [cluster_results[k]['hierarchical']['silhouette'] for k in k_values]
    kmeans_silhouettes = [cluster_results[k]['kmeans']['silhouette'] for k in k_values]
    
    ax2.plot(k_values, hier_silhouettes, 'o-', label='Hierarchical', linewidth=2, markersize=6)
    ax2.plot(k_values, kmeans_silhouettes, 's-', label='KMeans', linewidth=2, markersize=6)
    ax2.set_xlabel('群組數 (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('群組數 vs Silhouette Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 群組數 vs 平衡性
    ax3 = axes[0, 2]
    hier_balances = [cluster_results[k]['hierarchical']['balance'] for k in k_values]
    kmeans_balances = [cluster_results[k]['kmeans']['balance'] for k in k_values]
    
    ax3.plot(k_values, hier_balances, 'o-', label='Hierarchical', linewidth=2, markersize=6)
    ax3.plot(k_values, kmeans_balances, 's-', label='KMeans', linewidth=2, markersize=6)
    ax3.set_xlabel('群組數 (k)')
    ax3.set_ylabel('平衡性分數')
    ax3.set_title('群組數 vs 平衡性')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 群組大小分布 (選擇最佳k)
    ax4 = axes[0, 3]
    best_k = max(k_values, key=lambda k: max(cluster_results[k]['hierarchical']['score'], 
                                            cluster_results[k]['kmeans']['score']))
    best_method = 'hierarchical' if cluster_results[best_k]['hierarchical']['score'] > cluster_results[best_k]['kmeans']['score'] else 'kmeans'
    best_sizes = cluster_results[best_k][best_method]['sizes']
    
    ax4.bar(range(len(best_sizes)), best_sizes)
    ax4.set_xlabel('群組編號')
    ax4.set_ylabel('群組大小')
    ax4.set_title(f'最佳分群: {best_method} (k={best_k})\n群組大小分布')
    ax4.grid(True, alpha=0.3)
    
    # 5. 群組數 vs 顯著特徵數
    ax5 = axes[1, 0]
    hier_sig_features = [quality_analysis[k]['hierarchical']['n_significant_features'] for k in k_values]
    kmeans_sig_features = [quality_analysis[k]['kmeans']['n_significant_features'] for k in k_values]
    
    ax5.plot(k_values, hier_sig_features, 'o-', label='Hierarchical', linewidth=2, markersize=6)
    ax5.plot(k_values, kmeans_sig_features, 's-', label='KMeans', linewidth=2, markersize=6)
    ax5.set_xlabel('群組數 (k)')
    ax5.set_ylabel('顯著特徵數')
    ax5.set_title('群組數 vs 顯著特徵數')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 群組大小熱圖
    ax6 = axes[1, 1]
    size_matrix = []
    for k in k_values[:10]:  # 只顯示前10個k值
        hier_sizes = cluster_results[k]['hierarchical']['sizes']
        kmeans_sizes = cluster_results[k]['kmeans']['sizes']
        # 標準化群組大小
        max_size = max(len(hier_sizes), len(kmeans_sizes))
        hier_norm = np.zeros(max_size)
        kmeans_norm = np.zeros(max_size)
        hier_norm[:len(hier_sizes)] = hier_sizes
        kmeans_norm[:len(kmeans_sizes)] = kmeans_sizes
        size_matrix.append([k, 'Hierarchical'] + hier_norm.tolist())
        size_matrix.append([k, 'KMeans'] + kmeans_norm.tolist())
    
    if size_matrix:
        size_df = pd.DataFrame(size_matrix, columns=['k', 'method'] + [f'cluster_{i}' for i in range(max_size)])
        size_pivot = size_df.pivot(index='k', columns='method', values='cluster_0')
        sns.heatmap(size_pivot, annot=True, fmt='.0f', ax=ax6, cmap='YlOrRd')
        ax6.set_title('群組大小熱圖 (最大群組)')
    
    # 7. 最佳分群的PCA散點圖
    ax7 = axes[1, 2]
    best_labels = cluster_results[best_k][best_method]['labels']
    
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
        ax7.set_title(f'最佳分群: {best_method} (k={best_k})\nPCA散點圖')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. 群組大小變異係數
    ax8 = axes[1, 3]
    hier_cvs = [np.std(cluster_results[k]['hierarchical']['sizes']) / np.mean(cluster_results[k]['hierarchical']['sizes']) for k in k_values]
    kmeans_cvs = [np.std(cluster_results[k]['kmeans']['sizes']) / np.mean(cluster_results[k]['kmeans']['sizes']) for k in k_values]
    
    ax8.plot(k_values, hier_cvs, 'o-', label='Hierarchical', linewidth=2, markersize=6)
    ax8.plot(k_values, kmeans_cvs, 's-', label='KMeans', linewidth=2, markersize=6)
    ax8.set_xlabel('群組數 (k)')
    ax8.set_ylabel('群組大小變異係數')
    ax8.set_title('群組數 vs 大小變異係數\n(越小越平衡)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 推薦群組數分析
    ax9 = axes[2, 0]
    # 找出推薦的k值（平衡性好的）
    balanced_k = []
    for k in k_values:
        hier_balance = cluster_results[k]['hierarchical']['balance']
        kmeans_balance = cluster_results[k]['kmeans']['balance']
        if hier_balance > 0.7 or kmeans_balance > 0.7:  # 平衡性閾值
            balanced_k.append(k)
    
    if balanced_k:
        ax9.bar(['k=' + str(k) for k in balanced_k], 
                [cluster_results[k]['hierarchical']['balance'] for k in balanced_k],
                label='Hierarchical', alpha=0.7)
        ax9.bar(['k=' + str(k) for k in balanced_k], 
                [cluster_results[k]['kmeans']['balance'] for k in balanced_k],
                label='KMeans', alpha=0.7)
        ax9.set_ylabel('平衡性分數')
        ax9.set_title('推薦群組數 (平衡性>0.7)')
        ax9.legend()
        ax9.tick_params(axis='x', rotation=45)
        ax9.grid(True, alpha=0.3)
    
    # 10. 群組大小分布比較
    ax10 = axes[2, 1]
    # 比較不同k值的群組大小分布
    k_to_compare = [3, 5, 7, 10]  # 選擇幾個k值比較
    available_k = [k for k in k_to_compare if k in k_values]
    
    if available_k:
        for i, k in enumerate(available_k):
            sizes = cluster_results[k]['hierarchical']['sizes']
            ax10.scatter([k] * len(sizes), sizes, alpha=0.6, label=f'k={k}')
        
        ax10.set_xlabel('群組數 (k)')
        ax10.set_ylabel('群組大小')
        ax10.set_title('不同k值的群組大小分布')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
    
    # 11. 特徵重要性比較
    ax11 = axes[2, 2]
    if best_k in quality_analysis:
        feature_importance = quality_analysis[best_k][best_method]['feature_importance']
        important_features = [(f, v['f_stat']) for f, v in feature_importance.items() if v['significant']]
        important_features.sort(key=lambda x: x[1], reverse=True)
        
        if important_features:
            features, f_stats = zip(*important_features[:10])
            ax11.barh(range(len(features)), f_stats)
            ax11.set_yticks(range(len(features)))
            ax11.set_yticklabels(features)
            ax11.set_xlabel('F-statistic')
            ax11.set_title(f'重要特徵 (k={best_k})')
            ax11.grid(True, alpha=0.3)
    
    # 12. 最終推薦
    ax12 = axes[2, 3]
    ax12.text(0.1, 0.8, f'策略: {strategy}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.7, f'最佳方法: {best_method}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.6, f'推薦群組數: {best_k}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.5, f'綜合評分: {cluster_results[best_k][best_method]["score"]:.3f}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.4, f'平衡性: {cluster_results[best_k][best_method]["balance"]:.3f}', fontsize=12, transform=ax12.transAxes)
    
    # 推薦其他k值
    other_recommendations = []
    for k in k_values:
        if k != best_k:
            score = cluster_results[k][best_method]['score']
            balance = cluster_results[k][best_method]['balance']
            if balance > 0.6:  # 平衡性好的
                other_recommendations.append(f'k={k} (評分:{score:.3f})')
    
    if other_recommendations:
        ax12.text(0.1, 0.3, '其他推薦:', fontsize=12, transform=ax12.transAxes)
        for i, rec in enumerate(other_recommendations[:3]):
            ax12.text(0.1, 0.2 - i*0.05, rec, fontsize=10, transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    ax12.set_title('最終推薦')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_balanced_clustering.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def generate_balanced_recommendations(cluster_results, quality_analysis, strategy, datasource):
    """生成平衡分群建議"""
    print(f"\n=== {strategy} - {datasource} 平衡分群建議 ===")
    
    # 找出最佳分群
    best_score = -1
    best_k = None
    best_method = None
    
    for k, methods in cluster_results.items():
        for method_name, result in methods.items():
            if result['score'] > best_score:
                best_score = result['score']
                best_k = k
                best_method = method_name
    
    print(f"最佳分群: {best_method} (k={best_k})")
    print(f"綜合評分: {best_score:.3f}")
    print(f"Silhouette Score: {cluster_results[best_k][best_method]['silhouette']:.3f}")
    print(f"平衡性分數: {cluster_results[best_k][best_method]['balance']:.3f}")
    print(f"群組大小: {cluster_results[best_k][best_method]['sizes']}")
    
    # 找出平衡性好的分群
    print(f"\n平衡性好的分群 (平衡性>0.6):")
    balanced_recommendations = []
    
    for k, methods in cluster_results.items():
        for method_name, result in methods.items():
            if result['balance'] > 0.6:
                balanced_recommendations.append({
                    'k': k,
                    'method': method_name,
                    'score': result['score'],
                    'balance': result['balance'],
                    'silhouette': result['silhouette'],
                    'sizes': result['sizes']
                })
    
    # 按平衡性排序
    balanced_recommendations.sort(key=lambda x: x['balance'], reverse=True)
    
    for i, rec in enumerate(balanced_recommendations[:5]):
        print(f"{i+1}. {rec['method']} (k={rec['k']}): 平衡性={rec['balance']:.3f}, "
              f"評分={rec['score']:.3f}, 群組大小={rec['sizes']}")
    
    # 找出群組大小適中的分群
    print(f"\n群組大小適中的分群 (最大群組<200):")
    size_recommendations = []
    
    for k, methods in cluster_results.items():
        for method_name, result in methods.items():
            max_size = max(result['sizes'])
            if max_size < 200:
                size_recommendations.append({
                    'k': k,
                    'method': method_name,
                    'score': result['score'],
                    'balance': result['balance'],
                    'max_size': max_size,
                    'sizes': result['sizes']
                })
    
    # 按最大群組大小排序
    size_recommendations.sort(key=lambda x: x['max_size'])
    
    for i, rec in enumerate(size_recommendations[:5]):
        print(f"{i+1}. {rec['method']} (k={rec['k']}): 最大群組={rec['max_size']}, "
              f"平衡性={rec['balance']:.3f}, 評分={rec['score']:.3f}")
    
    return {
        'best_clustering': {
            'k': best_k,
            'method': best_method,
            'score': best_score,
            'sizes': cluster_results[best_k][best_method]['sizes']
        },
        'balanced_recommendations': balanced_recommendations[:5],
        'size_recommendations': size_recommendations[:5]
    }

def main():
    """主函數"""
    print("開始平衡分群分析...")
    
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
        
        # 尋找最佳群組數
        cluster_results = find_optimal_clusters(X_scaled, max_clusters=15, min_cluster_size=10)
        
        # 分析分群品質
        quality_analysis = analyze_cluster_quality(cluster_results, df, available_features)
        
        # 創建可視化
        create_balanced_visualizations(df, cluster_results, quality_analysis, 
                                     available_features, strategy, datasource)
        
        # 生成建議
        recommendations = generate_balanced_recommendations(cluster_results, quality_analysis, 
                                                          strategy, datasource)
        
        # 保存最佳分群結果
        best_k = recommendations['best_clustering']['k']
        best_method = recommendations['best_clustering']['method']
        best_labels = cluster_results[best_k][best_method]['labels']
        
        result_df = df.copy()
        result_df['balanced_cluster'] = best_labels
        result_df['clustering_method'] = best_method
        result_df['k_value'] = best_k
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_balanced_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_recommendations[(strategy, datasource)] = recommendations
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print(f"\n=== 平衡分群總結 ===")
    for (strategy, datasource), recs in all_recommendations.items():
        best = recs['best_clustering']
        print(f"\n{strategy} - {datasource}:")
        print(f"  最佳分群: {best['method']} (k={best['k']})")
        print(f"  綜合評分: {best['score']:.3f}")
        print(f"  群組大小: {best['sizes']}")
        
        if recs['size_recommendations']:
            size_rec = recs['size_recommendations'][0]
            print(f"  小群組推薦: {size_rec['method']} (k={size_rec['k']}, 最大群組={size_rec['max_size']})")
    
    print(f"\n所有結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 