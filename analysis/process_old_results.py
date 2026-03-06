# 處理舊版本optuna結果文件
# 使用fine-grained clustering方法重新處理

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import glob
import sys
import optuna
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
print('ROOT:', ROOT)
print('sys.path:', sys.path)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_and_prepare_data(file_path):
    """載入並準備數據"""
    print(f"載入文件: {file_path}")
    df = pd.read_csv(file_path)
    print(f"數據量: {len(df)}")
    
    # 提取特徵
    features = ['total_return', 'num_trades', 'sharpe_ratio', 'max_drawdown', 
                'profit_factor', 'avg_hold_days', 'cpcv_oos_mean', 'cpcv_oos_min', 
                'sharpe_var', 'pbo_score']
    
    # 檢查是否有這些欄位
    available_features = [f for f in features if f in df.columns]
    print(f"可用特徵: {available_features}")
    
    if len(available_features) < 3:
        print("特徵不足，跳過此文件")
        return None, None
    
    X = df[available_features].values
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled

def fine_grained_clustering(X, df, strategy_name, data_source):
    """執行fine-grained clustering"""
    print(f"執行fine-grained clustering: {strategy_name}_{data_source}")
    
    # 計算合適的距離閾值
    from scipy.spatial.distance import pdist
    distances = pdist(X)
    distance_threshold = np.percentile(distances, 25)  # 使用25%分位數作為閾值
    
    print(f"距離閾值: {distance_threshold:.4f}")
    
    # 使用KMeans進行fine-grained clustering
    best_score = -1
    best_labels = None
    best_method = None
    
    # 根據數據量決定k，目標是產生小cluster
    n_samples = len(X)
    k_range = range(max(5, n_samples // 10), min(n_samples // 3, 100))
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels_kmeans = kmeans.fit_predict(X)
            
            if len(np.unique(labels_kmeans)) > 1:
                score_kmeans = silhouette_score(X, labels_kmeans)
                print(f"KMeans k={k}: {len(np.unique(labels_kmeans))} clusters, score: {score_kmeans:.4f}")
                
                if score_kmeans > best_score:
                    best_score = score_kmeans
                    best_labels = labels_kmeans
                    best_method = f"KMeans_k{k}"
        except Exception as e:
            print(f"KMeans k={k} 失敗: {e}")
            continue
    
    if best_labels is None:
        print("KMeans失敗，使用單一cluster")
        best_labels = np.zeros(len(X))
        best_method = "SingleCluster"
    
    print(f"最佳方法: {best_method}, 分數: {best_score:.4f}")
    
    # 添加cluster標籤
    df_result = df.copy()
    df_result['fine_grained_cluster'] = best_labels
    
    # 分析cluster分布
    cluster_sizes = df_result['fine_grained_cluster'].value_counts().sort_index()
    small_clusters = cluster_sizes[cluster_sizes <= 5]
    medium_clusters = cluster_sizes[(cluster_sizes > 5) & (cluster_sizes <= 10)]
    large_clusters = cluster_sizes[cluster_sizes > 10]
    
    print(f"Cluster分布:")
    print(f"  總cluster數: {len(cluster_sizes)}")
    print(f"  小cluster (≤5): {len(small_clusters)} 個")
    print(f"  中cluster (6-10): {len(medium_clusters)} 個")
    print(f"  大cluster (>10): {len(large_clusters)} 個")
    print(f"  小cluster比例: {len(small_clusters) / len(cluster_sizes):.3f}")
    
    return df_result

def process_file(file_path):
    """處理單個文件"""
    # 從文件名提取策略和數據源
    filename = os.path.basename(file_path)
    parts = filename.replace('.csv', '').split('_')
    
    # 解析策略名稱
    if 'RMA' in filename:
        strategy = 'RMA'
    elif 'single' in filename:
        strategy = 'single'
    elif 'ssma_turn' in filename:
        strategy = 'ssma_turn'
    else:
        print(f"無法識別策略: {filename}")
        return
    
    # 解析數據源
    if 'Self' in filename:
        data_source = 'Self'
    elif '2412.TW' in filename:
        data_source = 'Factor_TWII__2412.TW'
    elif '2414.TW' in filename:
        data_source = 'Factor_TWII__2414.TW'
    else:
        print(f"無法識別數據源: {filename}")
        return
    
    print(f"\n處理: {strategy}_{data_source}")
    
    # 載入數據
    df, X = load_and_prepare_data(file_path)
    if df is None:
        return
    
    # 執行clustering
    df_result = fine_grained_clustering(X, df, strategy, data_source)
    
    # 保存結果
    output_dir = str(ROOT / 'results/fine_grained_processed')
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{strategy}_{data_source}_fine_grained_processed.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    df_result.to_csv(output_path, index=False)
    print(f"結果已保存: {output_path}")
    
    return df_result

def main():
    """主函數"""
    print("開始處理舊版本optuna結果文件.")
    
    # 要處理的文件列表
    files_to_process = [
        str(ROOT / 'results/op16/optuna_results_RMA_Factor_TWII__2412.TW_20250702_131059_d1.0_20250704_022050.csv'),
        str(ROOT / 'results/op16/optuna_results_RMA_Factor_TWII__2414.TW_20250702_131059_d1.0_20250704_022050.csv'),
        str(ROOT / 'results/op16/optuna_results_RMA_Self_20250702_131059_d1.0_20250704_022050.csv'),
        str(ROOT / 'results/op16/optuna_results_single_Factor_TWII__2412.TW_20250702_001433_d1.0_20250704_022050.csv'),
        str(ROOT / 'results/op16/optuna_results_single_Factor_TWII__2414.TW_20250702_001433_d1.0_20250704_022050.csv'),
        str(ROOT / 'results/op16/optuna_results_single_Self_20250702_001433_d1.0_20250704_022049.csv'),
        str(ROOT / 'results/op16/optuna_results_ssma_turn_Factor_TWII__2412.TW_20250702_162008_d1.0_20250704_022049.csv'),
        str(ROOT / 'results/op16/optuna_results_ssma_turn_Factor_TWII__2414.TW_20250702_162008_d1.0_20250704_022049.csv'),
        str(ROOT / 'results/op16/optuna_results_ssma_turn_Self_20250702_162008_d1.0_20250704_022049.csv')
    ]
    
    results = {}
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            try:
                result = process_file(file_path)
                if result is not None:
                    filename = os.path.basename(file_path)
                    results[filename] = result
            except Exception as e:
                print(f"處理文件 {file_path} 時發生錯誤: {e}")
        else:
            print(f"文件不存在: {file_path}")
    
    # 生成總結報告
    print("\n" + "="*50)
    print("處理總結報告")
    print("="*50)
    
    for filename, df in results.items():
        if df is not None:
            cluster_sizes = df['fine_grained_cluster'].value_counts()
            small_clusters = len(cluster_sizes[cluster_sizes <= 5])
            total_clusters = len(cluster_sizes)
            
            print(f"{filename}:")
            print(f"  總數據量: {len(df)}")
            print(f"  總cluster數: {total_clusters}")
            print(f"  小cluster數: {small_clusters}")
            print(f"  小cluster比例: {small_clusters/total_clusters:.3f}")
            print()

if __name__ == "__main__":
    main() 