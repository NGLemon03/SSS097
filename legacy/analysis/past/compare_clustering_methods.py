# -*- coding: utf-8 -*-
'''
比較三種分群方法：
1. 機器學習DTW分群
2. 手動DTW分群 (3個分群)
3. 純參數+績效分群 (不使用DTW)
'''
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """載入並準備數據"""
    
    # 載入最新的結果檔案作為基準
    results_dir = Path("results")
    base_file = results_dir / "optuna_results_single_20250701_093224.csv"
    
    if not base_file.exists():
        print("找不到基準檔案，請先運行 optuna_16.py")
        return None
    
    df = pd.read_csv(base_file)
    print(f"載入 {len(df)} 個策略進行分析")
    return df

def method1_ml_dtw_clustering(df):
    """方法1: 機器學習DTW分群 (現有結果)"""
    print("\n=== 方法1: 機器學習DTW分群 ===")
    
    # 使用現有的分群結果
    cluster_counts = df['dtw_cluster'].value_counts().sort_index()
    print(f"分群數: {df['dtw_cluster'].nunique()}")
    print(f"各分群策略數: {dict(cluster_counts)}")
    print(f"平均每群策略數: {len(df)/df['dtw_cluster'].nunique():.1f}")
    
    # 分析分群特徵
    cluster_stats = df.groupby('dtw_cluster').agg({
        'score': ['count', 'mean', 'std'],
        'total_return': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
        'profit_factor': ['mean', 'std'],
        'avg_hold_days': ['mean', 'std']
    }).round(3)
    
    print("\n分群統計:")
    print(cluster_stats)
    
    return df['dtw_cluster'], cluster_stats

def method2_manual_dtw_clustering(df):
    """方法2: 手動DTW分群 (3個分群)"""
    print("\n=== 方法2: 手動DTW分群 (3個分群) ===")
    
    # 模擬手動分群結果 (基於分數分組)
    scores = df['score'].values
    n_clusters = 3
    
    # 使用分數分位數進行分群
    quantiles = np.percentile(scores, [33, 66])
    manual_clusters = np.ones(len(df), dtype=int)
    manual_clusters[scores > quantiles[1]] = 3
    manual_clusters[(scores > quantiles[0]) & (scores <= quantiles[1])] = 2
    
    df_manual = df.copy()
    df_manual['manual_cluster'] = manual_clusters
    
    cluster_counts = df_manual['manual_cluster'].value_counts().sort_index()
    print(f"分群數: {df_manual['manual_cluster'].nunique()}")
    print(f"各分群策略數: {dict(cluster_counts)}")
    print(f"平均每群策略數: {len(df_manual)/df_manual['manual_cluster'].nunique():.1f}")
    
    # 分析分群特徵
    cluster_stats = df_manual.groupby('manual_cluster').agg({
        'score': ['count', 'mean', 'std'],
        'total_return': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
        'profit_factor': ['mean', 'std'],
        'avg_hold_days': ['mean', 'std']
    }).round(3)
    
    print("\n分群統計:")
    print(cluster_stats)
    
    return manual_clusters, cluster_stats

def method3_param_performance_clustering(df):
    """方法3: 純參數+績效分群 (不使用DTW)"""
    print("\n=== 方法3: 純參數+績效分群 ===")
    
    # 準備特徵數據
    feature_data = []
    
    for _, row in df.iterrows():
        # 績效特徵
        performance_features = [
            row['total_return'],
            row['sharpe_ratio'],
            row['max_drawdown'],
            row['profit_factor'],
            row['avg_hold_days'],
            row['cpcv_oos_mean'],
            row['cpcv_oos_min']
        ]
        
        # 參數特徵
        params = json.loads(row['parameters'])
        param_features = [
            params['linlen'],
            params['smaalen'], 
            params['devwin'],
            params['buy_mult'],
            params['sell_mult'],
            params['stop_loss']
        ]
        
        # 合併特徵
        all_features = performance_features + param_features
        feature_data.append(all_features)
    
    # 轉換為numpy陣列
    X = np.array(feature_data)
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用PCA降維
    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans分群
    n_clusters = min(4, len(df)//5)  # 確保每群至少有5個策略
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    param_clusters = kmeans.fit_predict(X_pca)
    
    df_param = df.copy()
    df_param['param_cluster'] = param_clusters + 1  # 從1開始編號
    
    cluster_counts = df_param['param_cluster'].value_counts().sort_index()
    print(f"分群數: {df_param['param_cluster'].nunique()}")
    print(f"各分群策略數: {dict(cluster_counts)}")
    print(f"平均每群策略數: {len(df_param)/df_param['param_cluster'].nunique():.1f}")
    
    # 分析分群特徵
    cluster_stats = df_param.groupby('param_cluster').agg({
        'score': ['count', 'mean', 'std'],
        'total_return': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
        'profit_factor': ['mean', 'std'],
        'avg_hold_days': ['mean', 'std']
    }).round(3)
    
    print("\n分群統計:")
    print(cluster_stats)
    
    # 分析參數分布
    print("\n參數分布分析:")
    for cluster in sorted(df_param['param_cluster'].unique()):
        cluster_data = df_param[df_param['param_cluster'] == cluster]
        print(f"\n分群 {cluster} 參數特徵:")
        
        # 分析參數範圍
        linlens = [json.loads(row['parameters'])['linlen'] for _, row in cluster_data.iterrows()]
        smaalens = [json.loads(row['parameters'])['smaalen'] for _, row in cluster_data.iterrows()]
        buy_mults = [json.loads(row['parameters'])['buy_mult'] for _, row in cluster_data.iterrows()]
        
        print(f"  linlen: {min(linlens)}-{max(linlens)} (平均: {np.mean(linlens):.1f})")
        print(f"  smaalen: {min(smaalens)}-{max(smaalens)} (平均: {np.mean(smaalens):.1f})")
        print(f"  buy_mult: {min(buy_mults)}-{max(buy_mults)} (平均: {np.mean(buy_mults):.2f})")
    
    return param_clusters + 1, cluster_stats

def compare_methods(df):
    """比較三種方法"""
    print("="*60)
    print("🔍 三種分群方法比較")
    print("="*60)
    
    # 執行三種分群方法
    ml_clusters, ml_stats = method1_ml_dtw_clustering(df)
    manual_clusters, manual_stats = method2_manual_dtw_clustering(df)
    param_clusters, param_stats = method3_param_performance_clustering(df)
    
    # 比較分析
    print("\n" + "="*60)
    print("📊 方法比較總結")
    print("="*60)
    
    methods = [
        ("機器學習DTW", ml_clusters, ml_stats),
        ("手動DTW(3群)", manual_clusters, manual_stats),
        ("參數+績效", param_clusters, param_stats)
    ]
    
    comparison_data = []
    
    for method_name, clusters, stats in methods:
        n_clusters = len(np.unique(clusters))
        avg_cluster_size = len(df) / n_clusters
        cluster_sizes = [np.sum(clusters == i) for i in np.unique(clusters)]
        size_range = f"{min(cluster_sizes)}-{max(cluster_sizes)}"
        
        # 計算分群內變異性 (使用score的標準差)
        cluster_vars = []
        for i in np.unique(clusters):
            cluster_scores = df[clusters == i]['score']
            cluster_vars.append(cluster_scores.std())
        avg_variance = np.mean(cluster_vars)
        
        comparison_data.append({
            '方法': method_name,
            '分群數': n_clusters,
            '平均每群策略數': f"{avg_cluster_size:.1f}",
            '策略數範圍': size_range,
            '平均分群內變異性': f"{avg_variance:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 評估建議
    print("\n" + "="*60)
    print("💡 評估建議")
    print("="*60)
    
    for method_name, clusters, stats in methods:
        n_clusters = len(np.unique(clusters))
        avg_cluster_size = len(df) / n_clusters
        
        print(f"\n{method_name}:")
        if avg_cluster_size < 10:
            print("  ⚠️  每群策略數過少，可能過度分群")
        elif avg_cluster_size > 30:
            print("  ⚠️  每群策略數過多，無法有效排除過度相似")
        else:
            print("  ✅ 分群數量合理")
        
        if n_clusters < 3:
            print("  ⚠️  分群數過少，可能無法充分區分不同策略")
        elif n_clusters > 10:
            print("  ⚠️  分群數過多，管理複雜")
        else:
            print("  ✅ 分群數適中")

def main():
    """主函數"""
    print("🚀 開始比較三種分群方法")
    
    # 載入數據
    df = load_and_prepare_data()
    if df is None:
        return
    
    # 執行比較
    compare_methods(df)
    
    print("\n" + "="*60)
    print("🎯 結論")
    print("="*60)
    print("1. 機器學習DTW: 基於權益曲線行為分群，能識別相似表現模式")
    print("2. 手動DTW: 簡單但可能不夠精確")
    print("3. 參數+績效: 基於特徵分群，可能更符合參數相似性")
    print("\n建議根據實際需求選擇合適的方法！")

if __name__ == "__main__":
    main() 