# -*- coding: utf-8 -*-
'''
分析分群結果的腳本
'''
import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_clustering_results():
    """分析不同分群方法的效果"""
    
    # 讀取最新的兩個結果檔案
    results_dir = Path("results")
    
    # 檔案1: 機器學習分群結果 (20250701_093224)
    file1 = results_dir / "optuna_results_single_20250701_093224.csv"
    file1_final = results_dir / "optuna_results_single_dtw_final_20250701_093224.csv"
    
    # 檔案2: 之前的分群結果 (20250701_090755) 
    file2 = results_dir / "optuna_results_single_20250701_090755.csv"
    file2_final = results_dir / "optuna_results_single_dtw_final_20250701_090755.csv"
    
    print("=== 分析分群結果 ===\n")
    
    # 分析檔案1 (機器學習分群)
    if file1.exists() and file1_final.exists():
        print("📊 檔案1: 機器學習分群結果 (20250701_093224)")
        df1 = pd.read_csv(file1)
        df1_final = pd.read_csv(file1_final)
        
        print(f"總策略數: {len(df1)}")
        print(f"分群數: {len(df1_final)}")
        
        # 分析每個分群的策略數量和特徵
        cluster_stats = df1.groupby('dtw_cluster').agg({
            'score': ['count', 'mean', 'std'],
            'total_return': ['mean', 'std'],
            'sharpe_ratio': ['mean', 'std'],
            'max_drawdown': ['mean', 'std'],
            'profit_factor': ['mean', 'std']
        }).round(3)
        
        print("\n各分群統計:")
        print(cluster_stats)
        
        # 分析分群代表策略
        print(f"\n分群代表策略:")
        for _, row in df1_final.iterrows():
            cluster = row['dtw_cluster']
            score = row['score']
            total_ret = row['total_return']
            sharpe = row['sharpe_ratio']
            params = json.loads(row['parameters'])
            
            print(f"分群 {cluster}: 試驗{row['trial_number']} (分數: {score:.2f}, 報酬: {total_ret:.1f}%, 夏普: {sharpe:.2f})")
            print(f"  參數: linlen={params['linlen']}, smaalen={params['smaalen']}, devwin={params['devwin']}")
            print(f"  buy_mult={params['buy_mult']}, sell_mult={params['sell_mult']}, stop_loss={params['stop_loss']}")
        
        print("\n" + "="*50 + "\n")
    
    # 分析檔案2 (之前的分群)
    if file2.exists() and file2_final.exists():
        print("📊 檔案2: 之前的分群結果 (20250701_090755)")
        df2 = pd.read_csv(file2)
        df2_final = pd.read_csv(file2_final)
        
        print(f"總策略數: {len(df2)}")
        print(f"分群數: {len(df2_final)}")
        
        # 分析每個分群的策略數量和特徵
        cluster_stats2 = df2.groupby('dtw_cluster').agg({
            'score': ['count', 'mean', 'std'],
            'total_return': ['mean', 'std'],
            'sharpe_ratio': ['mean', 'std'],
            'max_drawdown': ['mean', 'std'],
            'profit_factor': ['mean', 'std']
        }).round(3)
        
        print("\n各分群統計:")
        print(cluster_stats2)
        
        # 分析分群代表策略
        print(f"\n分群代表策略:")
        for _, row in df2_final.iterrows():
            cluster = row['dtw_cluster']
            score = row['score']
            total_ret = row['total_return']
            sharpe = row['sharpe_ratio']
            params = json.loads(row['parameters'])
            
            print(f"分群 {cluster}: 試驗{row['trial_number']} (分數: {score:.2f}, 報酬: {total_ret:.1f}%, 夏普: {sharpe:.2f})")
            print(f"  參數: linlen={params['linlen']}, smaalen={params['smaalen']}, devwin={params['devwin']}")
            print(f"  buy_mult={params['buy_mult']}, sell_mult={params['sell_mult']}, stop_loss={params['stop_loss']}")
    
    # 比較分析
    print("\n" + "="*50)
    print("🔍 比較分析")
    print("="*50)
    
    if file1.exists() and file2.exists():
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        print(f"檔案1 (機器學習): {len(df1)} 策略 → {df1['dtw_cluster'].nunique()} 分群")
        print(f"檔案2 (之前): {len(df2)} 策略 → {df2['dtw_cluster'].nunique()} 分群")
        
        print(f"\n檔案1平均每群策略數: {len(df1)/df1['dtw_cluster'].nunique():.1f}")
        print(f"檔案2平均每群策略數: {len(df2)/df2['dtw_cluster'].nunique():.1f}")
        
        # 分析分群品質
        print(f"\n分群品質評估:")
        print(f"檔案1: 每群策略數範圍 {df1.groupby('dtw_cluster').size().min()}-{df1.groupby('dtw_cluster').size().max()}")
        print(f"檔案2: 每群策略數範圍 {df2.groupby('dtw_cluster').size().min()}-{df2.groupby('dtw_cluster').size().max()}")
        
        # 評估分群效果
        print(f"\n分群效果評估:")
        if len(df1)/df1['dtw_cluster'].nunique() < 20:
            print("✅ 檔案1: 分群數量合理，能有效排除過度相似")
        else:
            print("⚠️  檔案1: 每群策略數過多，可能無法有效排除過度相似")
            
        if len(df2)/df2['dtw_cluster'].nunique() < 20:
            print("✅ 檔案2: 分群數量合理，能有效排除過度相似")
        else:
            print("⚠️  檔案2: 每群策略數過多，可能無法有效排除過度相似")

if __name__ == "__main__":
    analyze_clustering_results() 