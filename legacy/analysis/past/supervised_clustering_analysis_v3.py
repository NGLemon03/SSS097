# v3: 監督式分群分析，針對相同分數但參數不同的案例進行深度分析
# 主要功能：特定案例相似性分析、監督式分群、穩定性分析
# 解決相同return/score但參數不同的分群問題

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
OUTPUT_DIR = 'results/supervised_clustering_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_risk_clustered_data():
    """載入風險分群後的數據"""
    all_files = glob.glob(f'{INPUT_DIR}/*_risk_clustered.csv')
    data_dict = {}
    
    for f in all_files:
        fname = os.path.basename(f)
        # 提取策略和數據源
        if 'ssma_turn_Self' in fname:
            strategy = 'ssma_turn_Self'
            datasource = 'Self'
        else:
            # 其他策略的提取邏輯
            continue
        
        df = pd.read_csv(f)
        data_dict[(strategy, datasource)] = df
    
    return data_dict

def analyze_specific_case(df, trial_numbers=[411, 612, 733]):
    """分析特定案例的相似性和差異性"""
    print(f"=== 分析特定案例: trial {trial_numbers} ===")
    
    # 篩選指定trial
    case_df = df[df['trial_number'].isin(trial_numbers)].copy()
    
    if len(case_df) == 0:
        print("未找到指定的trial")
        return None
    
    print(f"\n找到 {len(case_df)} 個trial:")
    for _, row in case_df.iterrows():
        print(f"Trial {row['trial_number']}: score={row['score']:.3f}, "
              f"total_return={row['total_return']:.3f}, "
              f"risk_cluster={row['risk_cluster']}")
    
    # 分析參數相似性
    param_cols = [col for col in df.columns if col.startswith('param_')]
    
    print(f"\n=== 參數相似性分析 ===")
    for param in param_cols:
        if param in case_df.columns:
            values = case_df[param].values
            if len(values) > 1:
                std_val = np.std(values)
                mean_val = np.mean(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                print(f"{param}: 均值={mean_val:.3f}, 標準差={std_val:.3f}, 變異係數={cv:.3f}")
    
    # 分析績效指標相似性
    perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    print(f"\n=== 績效指標相似性分析 ===")
    for perf in perf_cols:
        if perf in case_df.columns:
            values = case_df[perf].values
            if len(values) > 1:
                std_val = np.std(values)
                mean_val = np.mean(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                print(f"{perf}: 均值={mean_val:.3f}, 標準差={std_val:.3f}, 變異係數={cv:.3f}")
    
    # 分析風險指標相似性
    risk_cols = ['annual_volatility', 'downside_risk', 'var_95_annual', 'cvar_95_annual']
    print(f"\n=== 風險指標相似性分析 ===")
    for risk in risk_cols:
        if risk in case_df.columns:
            values = case_df[risk].values
            if len(values) > 1:
                std_val = np.std(values)
                mean_val = np.mean(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                print(f"{risk}: 均值={mean_val:.3f}, 標準差={std_val:.3f}, 變異係數={cv:.3f}")
    
    return case_df

def create_similarity_heatmap(df, trial_numbers, output_path):
    """創建相似性熱圖"""
    case_df = df[df['trial_number'].isin(trial_numbers)].copy()
    
    if len(case_df) < 2:
        return
    
    # 選擇要比較的特徵
    feature_cols = []
    
    # 參數特徵
    param_cols = [col for col in df.columns if col.startswith('param_')]
    feature_cols.extend(param_cols[:5])  # 取前5個參數
    
    # 績效特徵
    perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    feature_cols.extend([col for col in perf_cols if col in df.columns])
    
    # 風險特徵
    risk_cols = ['annual_volatility', 'downside_risk', 'var_95_annual']
    feature_cols.extend([col for col in risk_cols if col in df.columns])
    
    # 篩選可用特徵
    available_features = [col for col in feature_cols if col in case_df.columns]
    
    if len(available_features) < 2:
        return
    
    # 準備數據
    X = case_df[available_features].fillna(0).values
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 計算相似性矩陣
    similarity_matrix = np.corrcoef(X_scaled.T)
    
    # 創建熱圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 特徵相似性熱圖
    im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
    ax1.set_xticks(range(len(available_features)))
    ax1.set_yticks(range(len(available_features)))
    ax1.set_xticklabels(available_features, rotation=45, ha='right')
    ax1.set_yticklabels(available_features)
    ax1.set_title('Feature Similarity Matrix')
    plt.colorbar(im1, ax=ax1)
    
    # Trial間相似性
    trial_similarity = np.corrcoef(X_scaled)
    im2 = ax2.imshow(trial_similarity, cmap='RdYlBu_r', aspect='auto')
    ax2.set_xticks(range(len(trial_numbers)))
    ax2.set_yticks(range(len(trial_numbers)))
    ax2.set_xticklabels([f'Trial {t}' for t in trial_numbers])
    ax2.set_yticklabels([f'Trial {t}' for t in trial_numbers])
    ax2.set_title('Trial Similarity Matrix')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def implement_supervised_clustering(df, target_metric='score', n_clusters=3):
    """實現監督式分群"""
    print(f"\n=== 監督式分群分析 (目標: {target_metric}) ===")
    
    # 準備特徵
    feature_cols = []
    
    # 參數特徵
    param_cols = [col for col in df.columns if col.startswith('param_')]
    feature_cols.extend(param_cols)
    
    # 績效特徵 (排除目標變數)
    perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    perf_cols = [col for col in perf_cols if col in df.columns and col != target_metric]
    feature_cols.extend(perf_cols)
    
    # 風險特徵
    risk_cols = ['annual_volatility', 'downside_risk', 'var_95_annual', 'cvar_95_annual']
    risk_cols = [col for col in risk_cols if col in df.columns]
    feature_cols.extend(risk_cols)
    
    # 篩選可用特徵
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 5:
        print("可用特徵不足")
        return None
    
    # 準備數據
    X = df[available_features].fillna(0).values
    y = df[target_metric].values
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 方法1: 基於目標變數的分位數分群
    print("\n方法1: 基於目標變數分位數的監督式分群")
    quantile_labels = pd.qcut(y, q=n_clusters, labels=False, duplicates='drop')
    
    # 方法2: 基於目標變數的K-means分群
    print("方法2: 基於目標變數的K-means分群")
    y_reshaped = y.reshape(-1, 1)
    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(y_reshaped)
    
    # 方法3: 基於特徵空間的監督式分群 (使用目標變數作為權重)
    print("方法3: 特徵空間監督式分群")
    # 將目標變數標準化後加入特徵
    y_scaled = (y - np.mean(y)) / np.std(y)
    X_with_target = np.column_stack([X_scaled, y_scaled])
    
    # 使用加權距離進行分群
    feature_weights = np.ones(X_with_target.shape[1])
    feature_weights[-1] = 2.0  # 目標變數權重加倍
    
    # 計算加權距離矩陣
    from scipy.spatial.distance import pdist, squareform
    weighted_distances = pdist(X_with_target, metric='euclidean', w=feature_weights)
    distance_matrix = squareform(weighted_distances)
    
    # 使用層次分群
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(weighted_distances, method='ward')
    supervised_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    # 比較不同方法的分群結果
    results = {
        'quantile': quantile_labels,
        'kmeans_target': kmeans_labels,
        'supervised': supervised_labels,
        'original_risk': df['risk_cluster'].values if 'risk_cluster' in df.columns else None
    }
    
    # 計算分群品質
    print("\n=== 分群品質比較 ===")
    for method, labels in results.items():
        if labels is not None and len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels)
            cal_score = calinski_harabasz_score(X_scaled, labels)
            print(f"{method}: Silhouette={sil_score:.3f}, Calinski-Harabasz={cal_score:.1f}")
    
    return results

def create_supervised_clustering_visualizations(df, clustering_results, output_path):
    """創建監督式分群可視化"""
    if clustering_results is None:
        return
    
    # PCA降維
    feature_cols = [col for col in df.columns if col.startswith('param_') or 
                   col in ['total_return', 'sharpe_ratio', 'max_drawdown', 'annual_volatility']]
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 2:
        return
    
    X = df[available_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 創建子圖
    n_methods = len(clustering_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (method, labels) in enumerate(clustering_results.items()):
        if i >= 4 or labels is None:
            continue
            
        ax = axes[i]
        
        # 散點圖
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Cluster {cluster_id}', alpha=0.7)
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title(f'{method} Clustering')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_cluster_stability(df, clustering_results):
    """分析分群穩定性"""
    print("\n=== 分群穩定性分析 ===")
    
    if clustering_results is None:
        return
    
    # 計算不同方法間的一致性
    methods = list(clustering_results.keys())
    methods = [m for m in methods if clustering_results[m] is not None]
    
    if len(methods) < 2:
        return
    
    # 計算分群一致性矩陣
    consistency_matrix = np.zeros((len(methods), len(methods)))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i <= j:
                labels1 = clustering_results[method1]
                labels2 = clustering_results[method2]
                
                # 計算調整互信息 (Adjusted Mutual Information)
                from sklearn.metrics import adjusted_mutual_info_score
                ami_score = adjusted_mutual_info_score(labels1, labels2)
                consistency_matrix[i, j] = ami_score
                consistency_matrix[j, i] = ami_score
    
    # 顯示一致性矩陣
    print("\n分群方法一致性矩陣:")
    print("方法", end="\t")
    for method in methods:
        print(f"{method:15s}", end="\t")
    print()
    
    for i, method1 in enumerate(methods):
        print(f"{method1:15s}", end="\t")
        for j, method2 in enumerate(methods):
            print(f"{consistency_matrix[i, j]:.3f}", end="\t\t")
        print()
    
    return consistency_matrix

def generate_recommendations(df, clustering_results, trial_numbers):
    """生成分群調整建議"""
    print("\n=== 分群調整建議 ===")
    
    recommendations = []
    
    # 分析特定案例
    case_df = df[df['trial_number'].isin(trial_numbers)].copy()
    
    if len(case_df) == 0:
        return recommendations
    
    # 檢查不同方法的分群結果
    for method, labels in clustering_results.items():
        if labels is not None:
            case_labels = labels[df['trial_number'].isin(trial_numbers)]
            unique_labels = np.unique(case_labels)
            
            if len(unique_labels) == 1:
                recommendations.append(f"✓ {method}: 所有案例在同一群組 ({unique_labels[0]})")
            else:
                recommendations.append(f"✗ {method}: 案例分散在不同群組 {unique_labels}")
    
    # 基於相似性分析給出建議
    print("\n基於相似性分析的建議:")
    
    # 計算案例間的相似性
    feature_cols = [col for col in df.columns if col.startswith('param_')]
    available_features = [col for col in feature_cols if col in case_df.columns]
    
    if len(available_features) > 0:
        X_case = case_df[available_features].fillna(0).values
        scaler = StandardScaler()
        X_case_scaled = scaler.fit_transform(X_case)
        
        # 計算案例間距離
        from scipy.spatial.distance import pdist
        distances = pdist(X_case_scaled)
        avg_distance = np.mean(distances)
        
        if avg_distance < 0.5:
            recommendations.append("建議: 案例相似性高，可考慮合併為同一群組")
        elif avg_distance > 2.0:
            recommendations.append("建議: 案例差異性大，應分為不同群組")
        else:
            recommendations.append("建議: 案例相似性中等，可根據業務需求調整分群")
    
    return recommendations

def main():
    """主函數"""
    print("開始監督式分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    if not data_dict:
        print("未找到風險分群數據")
        return
    
    # 分析ssma_turn_Self案例
    for (strategy, datasource), df in data_dict.items():
        if 'ssma_turn_Self' in strategy:
            print(f"\n分析 {strategy} - {datasource}")
            
            # 分析特定案例
            case_df = analyze_specific_case(df, [411, 612, 733])
            
            if case_df is not None:
                # 創建相似性熱圖
                similarity_path = f'{OUTPUT_DIR}/{strategy}_{datasource}_similarity_analysis.png'
                create_similarity_heatmap(df, [411, 612, 733], similarity_path)
                
                # 實現監督式分群
                clustering_results = implement_supervised_clustering(df)
                
                if clustering_results:
                    # 創建可視化
                    viz_path = f'{OUTPUT_DIR}/{strategy}_{datasource}_supervised_clustering.png'
                    create_supervised_clustering_visualizations(df, clustering_results, viz_path)
                    
                    # 分析穩定性
                    stability_matrix = analyze_cluster_stability(df, clustering_results)
                    
                    # 生成建議
                    recommendations = generate_recommendations(df, clustering_results, [411, 612, 733])
                    
                    # 保存結果
                    results_df = df.copy()
                    for method, labels in clustering_results.items():
                        if labels is not None:
                            results_df[f'{method}_cluster'] = labels
                    
                    output_csv = f'{OUTPUT_DIR}/{strategy}_{datasource}_supervised_clustered.csv'
                    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                    
                    print(f"\n結果已保存到 {OUTPUT_DIR}")
                    print("\n".join(recommendations))

if __name__ == "__main__":
    main() 