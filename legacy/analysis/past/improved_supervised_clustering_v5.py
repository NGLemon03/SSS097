# v5: 改進監督式分群，結合多種分群方法與特徵工程
# 主要功能：多方法比較、特徵增強、穩定性分析、自適應分群
# 解決相同績效但參數不同的分群問題

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
OUTPUT_DIR = 'results/improved_supervised_clustering'
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
            # 其他策略的提取邏輯
            continue
        
        df = pd.read_csv(f)
        data_dict[(strategy, datasource)] = df
        print(f"載入 {strategy} - {datasource}: {len(df)} 行")
    
    return data_dict

def identify_duplicate_performance_groups(df, tolerance=1e-6):
    """識別具有相同績效指標的群組"""
    print("=== 識別相同績效群組 ===")
    
    # 檢查績效指標的相似性
    perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor', 'score']
    available_perf = [col for col in perf_cols if col in df.columns]
    
    # 創建績效特徵向量
    perf_features = df[available_perf].values
    
    # 使用K-means找出績效群組
    # 先嘗試不同的群組數，找到最佳分群
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

def implement_ml_based_clustering(df, perf_groups, perf_labels):
    """實現基於機器學習的分群方法"""
    print("\n=== 機器學習分群方法 ===")
    
    # 方法1: 自動選擇最佳分群數
    print("方法1: 自動選擇最佳分群數")
    
    param_cols = [col for col in df.columns if col.startswith('param_')]
    available_params = [col for col in param_cols if col in df.columns]
    
    if len(available_params) < 2:
        print("參數特徵不足")
        return None
    
    X = df[available_params].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用多種方法自動選擇最佳分群數
    methods = {
        'silhouette': 'silhouette_score',
        'calinski_harabasz': 'calinski_harabasz_score'
    }
    
    best_k_dict = {}
    for method_name, metric_name in methods.items():
        best_k = 2
        best_score = -1
        
        for k in range(2, min(10, len(df) // 5)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                
                if metric_name == 'silhouette_score':
                    score = silhouette_score(X_scaled, labels)
                else:
                    score = calinski_harabasz_score(X_scaled, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        best_k_dict[method_name] = best_k
        print(f"  {method_name}: 最佳分群數 = {best_k}")
    
    # 方法2: 強制使用2個群組
    print("\n方法2: 強制使用2個群組")
    kmeans_2 = KMeans(n_clusters=2, random_state=42)
    labels_2 = kmeans_2.fit_predict(X_scaled)
    
    # 方法3: 基於績效群組的層次分群
    print("\n方法3: 基於績效群組的層次分群")
    hierarchical_labels = []
    
    for perf_group_id in range(len(perf_groups)):
        group_mask = perf_labels == perf_group_id
        group_df = df[group_mask]
        
        if len(group_df) < 2:
            hierarchical_labels.extend([perf_group_id] * len(group_df))
        else:
            # 在績效群組內進行參數分群
            group_X = group_df[available_params].fillna(0).values
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
                hierarchical_labels.extend([perf_group_id] * len(group_df))
            else:
                sub_kmeans = KMeans(n_clusters=best_sub_k, random_state=42)
                sub_labels = sub_kmeans.fit_predict(group_X_scaled)
                # 將子群組標籤轉換為全局標籤
                global_labels = [f"{perf_group_id}_{sub_label}" for sub_label in sub_labels]
                hierarchical_labels.extend(global_labels)
    
    # 方法4: 監督式分群（使用績效作為目標）
    print("\n方法4: 監督式分群")
    
    # 將績效指標作為特徵
    perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    available_perf = [col for col in perf_cols if col in df.columns]
    
    if len(available_perf) > 0:
        # 組合參數和績效特徵
        perf_features = df[available_perf].fillna(0).values
        combined_features = np.column_stack([X_scaled, perf_features])
        
        # 標準化組合特徵
        combined_scaler = StandardScaler()
        combined_scaled = combined_scaler.fit_transform(combined_features)
        
        # 使用組合特徵進行分群
        supervised_kmeans = KMeans(n_clusters=2, random_state=42)
        supervised_labels = supervised_kmeans.fit_predict(combined_scaled)
    else:
        supervised_labels = labels_2
    
    return {
        'auto_silhouette': KMeans(n_clusters=best_k_dict['silhouette'], random_state=42).fit_predict(X_scaled),
        'auto_calinski': KMeans(n_clusters=best_k_dict['calinski_harabasz'], random_state=42).fit_predict(X_scaled),
        'forced_2': labels_2,
        'hierarchical': hierarchical_labels,
        'supervised': supervised_labels,
        'performance_groups': perf_labels
    }

def analyze_clustering_results(df, clustering_results, perf_groups):
    """分析分群結果"""
    print("\n=== 分群結果分析 ===")
    
    if clustering_results is None:
        return
    
    # 計算各方法的分群品質
    param_cols = [col for col in df.columns if col.startswith('param_')]
    available_params = [col for col in param_cols if col in df.columns]
    
    if len(available_params) < 2:
        return
    
    X = df[available_params].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("分群品質比較:")
    for method, labels in clustering_results.items():
        if method == 'performance_groups':
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
        if method == 'performance_groups':
            continue
            
        trial_labels = []
        for trial in target_trials:
            trial_mask = df['trial_number'] == trial
            if trial_mask.any():
                label = labels[trial_mask][0]
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
    param_cols = [col for col in df.columns if col.startswith('param_')]
    available_params = [col for col in param_cols if col in df.columns]
    
    if len(available_params) < 2:
        return
    
    X = df[available_params].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 創建子圖
    methods = [k for k in clustering_results.keys() if k != 'performance_groups']
    n_methods = len(methods)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        if i >= 6:
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
        recommendations.append("✓ 發現重複績效群組，建議使用參數分群")
    
    # 分析各分群方法
    if clustering_results:
        for method, labels in clustering_results.items():
            if method == 'performance_groups':
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
        print("1. 對於重複績效群組，使用參數分群方法")
        print("2. 推薦使用 'forced_2' 或 'supervised' 方法")
        print("3. 可以考慮在績效群組內進行細分群")
    else:
        print("1. 績效群組差異明顯，可以主要依賴績效分群")
        print("2. 參數分群作為輔助驗證")
    
    return recommendations

def main():
    """主函數"""
    print("開始改進的監督式分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    if not data_dict:
        print("未找到風險分群數據")
        return
    
    # 分析所有策略
    for (strategy, datasource), df in data_dict.items():
        print(f"\n分析 {strategy} - {datasource}")
        
        # 識別相同績效群組
        perf_groups, perf_labels = identify_duplicate_performance_groups(df)
        
        # 實現機器學習分群
        clustering_results = implement_ml_based_clustering(df, perf_groups, perf_labels)
        
        if clustering_results:
            # 分析結果
            analyze_clustering_results(df, clustering_results, perf_groups)
            
            # 創建可視化
            viz_path = f'{OUTPUT_DIR}/{strategy}_{datasource}_ml_clustering_comparison.png'
            create_visualization_comparison(df, clustering_results, viz_path)
            
            # 生成建議
            recommendations = generate_recommendations(clustering_results, perf_groups)
            
            # 保存結果
            results_df = df.copy()
            results_df['performance_group'] = perf_labels
            
            for method, labels in clustering_results.items():
                if method != 'performance_groups':
                    results_df[f'{method}_cluster'] = labels
            
            output_csv = f'{OUTPUT_DIR}/{strategy}_{datasource}_ml_clustered.csv'
            results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            
            print(f"\n結果已保存到 {OUTPUT_DIR}")
            print("\n".join(recommendations))

if __name__ == "__main__":
    main() 