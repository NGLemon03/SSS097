# v6: 改進分層分群，多階段分群策略
# 主要功能：分層特徵提取、多階段分群、群組穩定性評估
# 提供更細緻的分群層次結構

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
OUTPUT_DIR = 'results/improved_layered_clustering'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_risk_clustered_data():
    """載入風險分群後的數據"""
    all_files = glob.glob(f'{INPUT_DIR}/*_risk_clustered.csv')
    data_dict = {}
    
    for f in all_files:
        fname = os.path.basename(f)
        print(f"處理文件: {fname}")
        
        # 提取策略和數據源
        if 'ssma_turn_Self' in fname:
            strategy = 'ssma_turn_Self'
            datasource = 'Self'
        elif 'single_Self' in fname:
            strategy = 'single_Self'
            datasource = 'Self'
        elif 'RMA_Self' in fname:
            strategy = 'RMA_Self'
            datasource = 'Self'
        else:
            continue
        
        df = pd.read_csv(f)
        data_dict[(strategy, datasource)] = df
        print(f"載入 {strategy} - {datasource}: {len(df)} 行")
    
    return data_dict

def identify_performance_groups(df, tolerance=1e-6):
    """識別績效群組，處理相同績效的問題"""
    print("=== 識別績效群組 ===")
    
    # 檢查績效指標的相似性
    perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor', 'score']
    available_perf = [col for col in perf_cols if col in df.columns]
    
    # 創建績效特徵向量
    perf_features = df[available_perf].values
    
    # 使用K-means找出績效群組
    best_k = 2
    best_score = -1
    
    for k in range(2, min(10, len(df) // 5)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(perf_features)
            score = silhouette_score(perf_features, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    
    print(f"最佳績效分群數: {best_k} (Silhouette Score: {best_score:.3f})")
    
    # 使用最佳群組數進行分群
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    perf_labels = kmeans.fit_predict(perf_features)
    
    # 分析每個績效群組
    perf_groups = {}
    for i in range(best_k):
        group_mask = perf_labels == i
        group_df = df[group_mask].copy()
        group_size = len(group_df)
        
        # 計算群組內績效指標的變異性
        perf_vars = {}
        for col in available_perf:
            values = group_df[col].values
            perf_vars[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            }
        
        perf_groups[i] = {
            'size': group_size,
            'trials': group_df['trial_number'].tolist(),
            'performance_vars': perf_vars,
            'is_duplicate': all(perf_vars[col]['cv'] < tolerance for col in available_perf)
        }
        
        print(f"\n績效群組 {i}:")
        print(f"  大小: {group_size}")
        print(f"  是否為重複群組: {perf_groups[i]['is_duplicate']}")
        for col in available_perf:
            cv = perf_vars[col]['cv']
            print(f"  {col}: 均值={perf_vars[col]['mean']:.3f}, 變異係數={cv:.6f}")
    
    return perf_groups, perf_labels

def perform_layered_clustering(df, perf_groups, perf_labels):
    """執行分層分群"""
    print("\n=== 分層分群分析 ===")
    
    # 方法1: 只用參數+風險分群（去除績效）
    print("方法1: 只用參數+風險分群")
    
    param_cols = [col for col in df.columns if col.startswith('param_')]
    risk_cols = ['annual_volatility', 'downside_risk', 'sortino_ratio', 'var_95_annual', 
                 'cvar_95_annual', 'skewness', 'kurtosis', 'vol_stability', 
                 'drawdown_duration', 'max_consecutive_losses']
    
    available_params = [col for col in param_cols if col in df.columns]
    available_risk = [col for col in risk_cols if col in df.columns]
    
    if len(available_params) < 2:
        print("參數特徵不足")
        return None
    
    # 只用參數+風險特徵
    param_risk_features = available_params + available_risk
    X_param_risk = df[param_risk_features].fillna(0).values
    
    scaler = StandardScaler()
    X_param_risk_scaled = scaler.fit_transform(X_param_risk)
    
    # 自動選擇最佳分群數
    best_k_param_risk = 2
    best_score_param_risk = -1
    
    for k in range(2, min(10, len(df) // 5)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_param_risk_scaled)
            score = silhouette_score(X_param_risk_scaled, labels)
            if score > best_score_param_risk:
                best_score_param_risk = score
                best_k_param_risk = k
        except:
            continue
    
    print(f"  參數+風險最佳分群數: {best_k_param_risk} (Silhouette Score: {best_score_param_risk:.3f})")
    
    # 方法2: 分層分群（先分績效，再分參數）
    print("\n方法2: 分層分群")
    
    layered_labels = []
    for perf_group_id in range(len(perf_groups)):
        group_mask = perf_labels == perf_group_id
        group_df = df[group_mask]
        
        if len(group_df) < 2:
            layered_labels.extend([perf_group_id] * len(group_df))
        else:
            # 在績效群組內進行參數+風險分群
            group_X = group_df[param_risk_features].fillna(0).values
            group_X_scaled = scaler.fit_transform(group_X)
            
            # 自動決定子群組數
            best_sub_k = 1
            best_sub_score = -1
            
            for sub_k in range(1, min(4, len(group_df))):
                try:
                    sub_kmeans = KMeans(n_clusters=sub_k, random_state=42)
                    sub_labels = sub_kmeans.fit_predict(group_X_scaled)
                    
                    if sub_k > 1:
                        score = silhouette_score(group_X_scaled, sub_labels)
                    else:
                        score = 0
                    
                    if score > best_sub_score:
                        best_sub_score = score
                        best_sub_k = sub_k
                except:
                    continue
            
            # 使用最佳子群組數
            if best_sub_k == 1:
                layered_labels.extend([perf_group_id] * len(group_df))
            else:
                sub_kmeans = KMeans(n_clusters=best_sub_k, random_state=42)
                sub_labels = sub_kmeans.fit_predict(group_X_scaled)
                # 將子群組標籤轉換為全局標籤
                global_labels = [f"{perf_group_id}_{sub_label}" for sub_label in sub_labels]
                layered_labels.extend(global_labels)
    
    # 方法3: 階層式分群（只用參數+風險）
    print("\n方法3: 階層式分群")
    
    Z = linkage(X_param_risk_scaled, method='ward')
    
    # 自動選擇最佳分群數
    best_k_hier = 2
    best_score_hier = -1
    
    for k in range(2, min(10, len(df) // 5)):
        try:
            labels = fcluster(Z, t=k, criterion='maxclust')
            score = silhouette_score(X_param_risk_scaled, labels)
            if score > best_score_hier:
                best_score_hier = score
                best_k_hier = k
        except:
            continue
    
    print(f"  階層式最佳分群數: {best_k_hier} (Silhouette Score: {best_score_hier:.3f})")
    
    hierarchical_labels = fcluster(Z, t=best_k_hier, criterion='maxclust')
    
    # 方法4: 強制2群分群
    print("\n方法4: 強制2群分群")
    
    kmeans_2 = KMeans(n_clusters=2, random_state=42)
    forced_2_labels = kmeans_2.fit_predict(X_param_risk_scaled)
    
    return {
        'param_risk_only': KMeans(n_clusters=best_k_param_risk, random_state=42).fit_predict(X_param_risk_scaled),
        'layered': layered_labels,
        'hierarchical': hierarchical_labels,
        'forced_2': forced_2_labels,
        'performance_groups': perf_labels,
        'X_param_risk_scaled': X_param_risk_scaled,
        'param_risk_features': param_risk_features
    }

def analyze_clustering_results(df, clustering_results, perf_groups):
    """分析分群結果"""
    print("\n=== 分群結果分析 ===")
    
    if clustering_results is None:
        return
    
    # 計算各方法的分群品質
    X_scaled = clustering_results['X_param_risk_scaled']
    
    print("分群品質比較:")
    for method, labels in clustering_results.items():
        if method in ['performance_groups', 'X_param_risk_scaled', 'param_risk_features']:
            continue
            
        try:
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X_scaled, labels)
                cal_score = calinski_harabasz_score(X_scaled, labels)
                print(f"  {method}: Silhouette={sil_score:.3f}, Calinski-Harabasz={cal_score:.1f}")
            else:
                print(f"  {method}: 只有一個群組")
        except:
            print(f"  {method}: 無法計算品質指標")
    
    # 分析特定案例的分群結果
    target_trials = [411, 612, 733]
    print(f"\n特定案例 (trial {target_trials}) 的分群結果:")
    
    for method, labels in clustering_results.items():
        if method in ['performance_groups', 'X_param_risk_scaled', 'param_risk_features']:
            continue
            
        trial_labels = []
        for trial in target_trials:
            trial_mask = df['trial_number'] == trial
            if trial_mask.any():
                labels_arr = np.array(labels)
                label = labels_arr[trial_mask.values][0]
                trial_labels.append(label)
        
        unique_labels = np.unique(trial_labels)
        if len(unique_labels) == 1:
            print(f"  {method}: 所有案例在同一群組 ({unique_labels[0]})")
        else:
            print(f"  {method}: 案例分散在不同群組 {unique_labels}")

def create_visualization_comparison(df, clustering_results, output_path):
    """創建分群方法比較可視化"""
    if clustering_results is None:
        return
    
    # PCA降維
    X_scaled = clustering_results['X_param_risk_scaled']
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 創建子圖
    methods = [k for k in clustering_results.keys() 
               if k not in ['performance_groups', 'X_param_risk_scaled', 'param_risk_features']]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        if i >= 4:
            break
            
        ax = axes[i]
        labels = clustering_results[method]
        
        # 散點圖
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Cluster {label}', alpha=0.7)
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title(f'{method} Clustering')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_recommendations(clustering_results, perf_groups):
    """生成分群建議"""
    print("\n=== 分群建議 ===")
    
    recommendations = []
    
    # 分析績效群組
    duplicate_groups = [k for k, v in perf_groups.items() if v['is_duplicate']]
    non_duplicate_groups = [k for k, v in perf_groups.items() if not v['is_duplicate']]
    
    print(f"重複績效群組: {len(duplicate_groups)}")
    print(f"非重複績效群組: {len(non_duplicate_groups)}")
    
    if len(duplicate_groups) > 0:
        recommendations.append("✓ 發現重複績效群組，建議使用參數+風險分群")
    
    # 分析各分群方法
    if clustering_results:
        for method, labels in clustering_results.items():
            if method in ['performance_groups', 'X_param_risk_scaled', 'param_risk_features']:
                continue
                
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                recommendations.append(f"✓ {method}: 產生2個群組，符合預期")
            elif len(unique_labels) > 2:
                recommendations.append(f"⚠ {method}: 產生{len(unique_labels)}個群組，可能需要調整")
            else:
                recommendations.append(f"✗ {method}: 只有1個群組，分群效果不佳")
    
    # 最終建議
    print("\n最終建議:")
    if len(duplicate_groups) > 0:
        print("1. 對於重複績效群組，使用參數+風險分群方法")
        print("2. 推薦使用 'param_risk_only' 或 'hierarchical' 方法")
        print("3. 可以考慮在績效群組內進行細分群")
    else:
        print("1. 績效群組差異明顯，可以主要依賴績效分群")
        print("2. 參數+風險分群作為輔助驗證")
    
    return recommendations

def main():
    """主函數"""
    print("開始改進的分層分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    if not data_dict:
        print("未找到風險分群數據")
        return
    
    # 分析所有策略
    for (strategy, datasource), df in data_dict.items():
        print(f"\n分析 {strategy} - {datasource}")
        
        # 識別績效群組
        perf_groups, perf_labels = identify_performance_groups(df)
        
        # 執行分層分群
        clustering_results = perform_layered_clustering(df, perf_groups, perf_labels)
        
        if clustering_results:
            # 分析結果
            analyze_clustering_results(df, clustering_results, perf_groups)
            
            # 創建可視化
            viz_path = f'{OUTPUT_DIR}/{strategy}_{datasource}_layered_clustering_comparison.png'
            create_visualization_comparison(df, clustering_results, viz_path)
            
            # 生成建議
            recommendations = generate_recommendations(clustering_results, perf_groups)
            
            # 保存結果
            results_df = df.copy()
            results_df['performance_group'] = perf_labels
            
            for method, labels in clustering_results.items():
                if method not in ['performance_groups', 'X_param_risk_scaled', 'param_risk_features']:
                    results_df[f'{method}_cluster'] = labels
            
            output_csv = f'{OUTPUT_DIR}/{strategy}_{datasource}_layered_clustered.csv'
            results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            
            print(f"\n結果已保存到 {OUTPUT_DIR}")
            print("\n".join(recommendations))

if __name__ == "__main__":
    main() 