# v9: 分群比較分析，全面評估多種分群方法
# 主要功能：多方法統計比較、ANOVA分析、視覺化比較、詳細報告
# 提供科學的分群方法選擇依據

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
OUTPUT_DIR = 'results/clustering_comparison'
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

def perform_multiple_clustering(X_scaled, max_clusters=8):
    """執行多種分群方法"""
    results = {}
    
    # 1. Hierarchical Clustering
    print("執行階層式分群...")
    Z = linkage(X_scaled, method='ward')
    
    # 測試不同群數
    hier_results = {}
    for k in range(2, min(max_clusters + 1, len(X_scaled) // 5)):
        try:
            labels = fcluster(Z, t=k, criterion='maxclust')
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                hier_results[k] = {
                    'labels': labels,
                    'silhouette': score,
                    'cluster_sizes': np.bincount(labels)
                }
        except:
            continue
    
    results['hierarchical'] = hier_results
    
    # 2. KMeans Clustering
    print("執行KMeans分群...")
    kmeans_results = {}
    for k in range(2, min(max_clusters + 1, len(X_scaled) // 5)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            kmeans_results[k] = {
                'labels': labels,
                'silhouette': score,
                'cluster_sizes': np.bincount(labels)
            }
        except:
            continue
    
    results['kmeans'] = kmeans_results
    
    # 3. Agglomerative Clustering
    print("執行凝聚式分群...")
    agglo_results = {}
    for k in range(2, min(max_clusters + 1, len(X_scaled) // 5)):
        try:
            agglo = AgglomerativeClustering(n_clusters=k)
            labels = agglo.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            agglo_results[k] = {
                'labels': labels,
                'silhouette': score,
                'cluster_sizes': np.bincount(labels)
            }
        except:
            continue
    
    results['agglomerative'] = agglo_results
    
    return results

def analyze_cluster_distributions(df, clustering_results, feature_names):
    """分析群組間的分布差異"""
    print("\n=== 群組分布差異分析 ===")
    
    distribution_analysis = {}
    
    for method_name, method_results in clustering_results.items():
        print(f"\n{method_name.upper()} 分群方法:")
        distribution_analysis[method_name] = {}
        
        for k, result in method_results.items():
            labels = result['labels']
            print(f"  k={k}: Silhouette={result['silhouette']:.3f}, 群組大小={result['cluster_sizes']}")
            
            # 分析每個特徵的群組間差異
            feature_analysis = {}
            for feature in feature_names:
                if feature in df.columns:
                    feature_values = df[feature].values
                    unique_labels = np.unique(labels)
                    
                    if len(unique_labels) >= 2:
                        # 計算群組間統計差異
                        group_means = []
                        group_stds = []
                        group_values = []
                        
                        for cluster_id in unique_labels:
                            cluster_mask = labels == cluster_id
                            group_values.append(feature_values[cluster_mask])
                            group_means.append(np.mean(feature_values[cluster_mask]))
                            group_stds.append(np.std(feature_values[cluster_mask]))
                        
                        # ANOVA檢定
                        try:
                            f_stat, p_value = stats.f_oneway(*group_values)
                        except:
                            f_stat, p_value = 0, 1
                        
                        # 計算群組間變異係數
                        overall_mean = np.mean(feature_values)
                        overall_std = np.std(feature_values)
                        cv_between = np.std(group_means) / abs(overall_mean) if overall_mean != 0 else 0
                        cv_within = np.mean(group_stds) / abs(overall_mean) if overall_mean != 0 else 0
                        
                        feature_analysis[feature] = {
                            'f_stat': f_stat,
                            'p_value': p_value,
                            'cv_between': cv_between,
                            'cv_within': cv_within,
                            'group_means': group_means,
                            'group_stds': group_stds,
                            'significant': p_value < 0.05
                        }
            
            distribution_analysis[method_name][k] = feature_analysis
    
    return distribution_analysis

def create_comprehensive_visualizations(df, clustering_results, distribution_analysis, feature_names, strategy, datasource):
    """創建綜合可視化比較"""
    
    # 創建多個子圖
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 分群方法比較 (Silhouette Score)
    ax1 = plt.subplot(3, 4, 1)
    methods = list(clustering_results.keys())
    method_scores = []
    method_names = []
    
    for method in methods:
        for k, result in clustering_results[method].items():
            method_scores.append(result['silhouette'])
            method_names.append(f"{method}_k{k}")
    
    bars = ax1.bar(range(len(method_scores)), method_scores)
    ax1.set_xticks(range(len(method_scores)))
    ax1.set_xticklabels(method_names, rotation=45, ha='right')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('分群方法比較 (Silhouette Score)')
    ax1.grid(True, alpha=0.3)
    
    # 標記最高分
    max_idx = np.argmax(method_scores)
    bars[max_idx].set_color('red')
    
    # 2. 群組大小分布比較
    ax2 = plt.subplot(3, 4, 2)
    best_method = method_names[max_idx].split('_')[0]
    best_k = int(method_names[max_idx].split('_')[1][1:])
    best_result = clustering_results[best_method][best_k]
    
    cluster_sizes = best_result['cluster_sizes']
    cluster_labels = [f'Cluster {i}' for i in range(len(cluster_sizes))]
    
    ax2.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'最佳分群: {best_method} (k={best_k})')
    
    # 3-6. 重要特徵的群組分布
    important_features = ['total_return', 'annual_volatility', 'param_1', 'param_2']
    available_features = [f for f in important_features if f in df.columns]
    
    for i, feature in enumerate(available_features[:4]):
        ax = plt.subplot(3, 4, 3 + i)
        
        # 使用最佳分群結果
        labels = best_result['labels']
        unique_labels = np.unique(labels)
        
        feature_data = []
        feature_labels = []
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            feature_data.append(df.loc[cluster_mask, feature])
            feature_labels.append(f'Cluster {cluster_id}')
        
        ax.boxplot(feature_data, labels=feature_labels)
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} by Cluster')
        ax.grid(True, alpha=0.3)
    
    # 7-10. 統計檢定結果
    ax7 = plt.subplot(3, 4, 7)
    if best_method in distribution_analysis and best_k in distribution_analysis[best_method]:
        feature_stats = distribution_analysis[best_method][best_k]
        
        # 計算每個特徵的F統計量
        features = list(feature_stats.keys())[:10]  # 取前10個特徵
        f_stats = [feature_stats[f]['f_stat'] for f in features]
        p_values = [feature_stats[f]['p_value'] for f in features]
        
        # 創建F統計量條形圖
        bars = ax7.bar(range(len(features)), f_stats)
        ax7.set_xticks(range(len(features)))
        ax7.set_xticklabels(features, rotation=45, ha='right')
        ax7.set_ylabel('F-statistic')
        ax7.set_title('特徵重要性 (F-statistic)')
        ax7.grid(True, alpha=0.3)
        
        # 標記顯著特徵
        for i, p_val in enumerate(p_values):
            if p_val < 0.05:
                bars[i].set_color('red')
    
    # 8. P值分布
    ax8 = plt.subplot(3, 4, 8)
    if best_method in distribution_analysis and best_k in distribution_analysis[best_method]:
        feature_stats = distribution_analysis[best_method][best_k]
        p_values = [feature_stats[f]['p_value'] for f in feature_stats.keys()]
        
        ax8.hist(p_values, bins=20, alpha=0.7, edgecolor='black')
        ax8.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        ax8.set_xlabel('P-value')
        ax8.set_ylabel('Frequency')
        ax8.set_title('P-value Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 9. 群組間變異係數比較
    ax9 = plt.subplot(3, 4, 9)
    if best_method in distribution_analysis and best_k in distribution_analysis[best_method]:
        feature_stats = distribution_analysis[best_method][best_k]
        
        features = list(feature_stats.keys())[:10]
        cv_between = [feature_stats[f]['cv_between'] for f in features]
        cv_within = [feature_stats[f]['cv_within'] for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax9.bar(x - width/2, cv_between, width, label='Between Groups', alpha=0.8)
        ax9.bar(x + width/2, cv_within, width, label='Within Groups', alpha=0.8)
        ax9.set_xticks(x)
        ax9.set_xticklabels(features, rotation=45, ha='right')
        ax9.set_ylabel('Coefficient of Variation')
        ax9.set_title('群組間 vs 群組內變異')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    # 10. 分群方法穩定性比較
    ax10 = plt.subplot(3, 4, 10)
    stability_scores = []
    method_names_stable = []
    
    for method in methods:
        if method in clustering_results:
            scores = [clustering_results[method][k]['silhouette'] for k in clustering_results[method].keys()]
            stability_scores.append(np.std(scores))  # 標準差越小越穩定
            method_names_stable.append(method)
    
    ax10.bar(method_names_stable, stability_scores)
    ax10.set_ylabel('Silhouette Score Std')
    ax10.set_title('分群方法穩定性')
    ax10.grid(True, alpha=0.3)
    
    # 11. 群組大小平衡性
    ax11 = plt.subplot(3, 4, 11)
    balance_scores = []
    method_names_balance = []
    
    for method in methods:
        if method in clustering_results:
            for k, result in clustering_results[method].items():
                cluster_sizes = result['cluster_sizes']
                # 計算群組大小平衡性 (標準差越小越平衡)
                balance_score = np.std(cluster_sizes) / np.mean(cluster_sizes)
                balance_scores.append(balance_score)
                method_names_balance.append(f"{method}_k{k}")
    
    ax11.bar(range(len(balance_scores)), balance_scores)
    ax11.set_xticks(range(len(balance_scores)))
    ax11.set_xticklabels(method_names_balance, rotation=45, ha='right')
    ax11.set_ylabel('Balance Score (CV)')
    ax11.set_title('群組大小平衡性')
    ax11.grid(True, alpha=0.3)
    
    # 12. 綜合評分
    ax12 = plt.subplot(3, 4, 12)
    
    # 計算綜合評分 (Silhouette + 穩定性 + 平衡性)
    composite_scores = []
    composite_names = []
    
    for method in methods:
        if method in clustering_results:
            for k, result in clustering_results[method].items():
                silhouette = result['silhouette']
                cluster_sizes = result['cluster_sizes']
                balance = 1 / (1 + np.std(cluster_sizes) / np.mean(cluster_sizes))  # 平衡性分數
                
                # 綜合評分
                composite_score = silhouette * 0.6 + balance * 0.4
                composite_scores.append(composite_score)
                composite_names.append(f"{method}_k{k}")
    
    bars = ax12.bar(range(len(composite_scores)), composite_scores)
    ax12.set_xticks(range(len(composite_scores)))
    ax12.set_xticklabels(composite_names, rotation=45, ha='right')
    ax12.set_ylabel('Composite Score')
    ax12.set_title('綜合評分')
    ax12.grid(True, alpha=0.3)
    
    # 標記最高分
    max_idx = np.argmax(composite_scores)
    bars[max_idx].set_color('green')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_comprehensive_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def generate_detailed_report(clustering_results, distribution_analysis, strategy, datasource):
    """生成詳細報告"""
    print(f"\n=== {strategy} - {datasource} 詳細分析報告 ===")
    
    # 找出最佳分群方法
    best_score = -1
    best_method = None
    best_k = None
    
    for method_name, method_results in clustering_results.items():
        for k, result in method_results.items():
            if result['silhouette'] > best_score:
                best_score = result['silhouette']
                best_method = method_name
                best_k = k
    
    print(f"最佳分群方法: {best_method} (k={best_k}, Silhouette={best_score:.3f})")
    
    # 分析每個方法的優缺點
    print("\n各方法分析:")
    for method_name, method_results in clustering_results.items():
        print(f"\n{method_name.upper()}:")
        
        for k, result in method_results.items():
            cluster_sizes = result['cluster_sizes']
            balance_score = 1 / (1 + np.std(cluster_sizes) / np.mean(cluster_sizes))
            
            print(f"  k={k}: Silhouette={result['silhouette']:.3f}, "
                  f"群組大小={cluster_sizes}, 平衡性={balance_score:.3f}")
            
            # 分析特徵重要性
            if method_name in distribution_analysis and k in distribution_analysis[method_name]:
                feature_stats = distribution_analysis[method_name][k]
                significant_features = [f for f, stats in feature_stats.items() if stats['significant']]
                print(f"    顯著特徵數量: {len(significant_features)}")
                if significant_features:
                    print(f"    重要特徵: {significant_features[:5]}")  # 顯示前5個
    
    # 推薦建議
    print(f"\n推薦建議:")
    print(f"1. 如果重視分群品質: 選擇 {best_method} (k={best_k})")
    
    # 找出最平衡的分群
    best_balance_score = 0
    best_balance_method = None
    best_balance_k = None
    
    for method_name, method_results in clustering_results.items():
        for k, result in method_results.items():
            cluster_sizes = result['cluster_sizes']
            balance_score = 1 / (1 + np.std(cluster_sizes) / np.mean(cluster_sizes))
            if balance_score > best_balance_score:
                best_balance_score = balance_score
                best_balance_method = method_name
                best_balance_k = k
    
    print(f"2. 如果重視群組平衡: 選擇 {best_balance_method} (k={best_balance_k})")
    
    # 保存詳細報告
    report_data = {
        'strategy': strategy,
        'datasource': datasource,
        'best_method': best_method,
        'best_k': best_k,
        'best_silhouette': best_score,
        'clustering_results': clustering_results,
        'distribution_analysis': distribution_analysis
    }
    
    with open(f'{OUTPUT_DIR}/{strategy}_{datasource}_detailed_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

def main():
    """主函數"""
    print("開始分群方法比較分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
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
        
        # 執行多種分群方法
        clustering_results = perform_multiple_clustering(X_scaled)
        
        # 分析分布差異
        distribution_analysis = analyze_cluster_distributions(df, clustering_results, available_features)
        
        # 創建綜合可視化
        create_comprehensive_visualizations(df, clustering_results, distribution_analysis, 
                                          available_features, strategy, datasource)
        
        # 生成詳細報告
        generate_detailed_report(clustering_results, distribution_analysis, strategy, datasource)
        
        print(f"結果已保存到 {OUTPUT_DIR}")
    
    print(f"\n所有分析完成，結果保存在 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 