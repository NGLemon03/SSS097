import os
import glob
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 定義路徑常數
HIERARCHICAL_DIR = 'results/op16'
HSA_DIR = os.path.join(HIERARCHICAL_DIR, 'hierarchical_score_analysis')
os.makedirs(HIERARCHICAL_DIR, exist_ok=True)
os.makedirs(HSA_DIR, exist_ok=True)

# 1. 讀取所有 optuna 結果
optuna_files = glob.glob(os.path.join(HIERARCHICAL_DIR, 'optuna_results_*.csv'))
optuna_dfs = []
for f in optuna_files:
    df = pd.read_csv(f)
    df['source_file'] = os.path.basename(f)
    optuna_dfs.append(df)
optuna_all = pd.concat(optuna_dfs, ignore_index=True)

# 2. 讀取 HSA 標籤
hsa_path = os.path.join(HSA_DIR, 'param_cluster_labels.csv')
print(df.columns)
hsa_df = pd.read_csv(hsa_path)

# 3. 合併 - 修正欄位名稱
# optuna 結果使用 trial_number，HSA 結果使用 trial_id
merged = pd.merge(optuna_all, hsa_df, left_on='trial_number', right_on='trial_id', how='left')

# 4. hierarchical_cluster：各分組 top
print("=== hierarchical_cluster 各分組 top 3 trials ===")
if 'hierarchical_cluster' in merged.columns:
    for group, group_df in merged.groupby('hierarchical_cluster'):
        if 'score' in group_df.columns:
            top = group_df.nlargest(3, 'score')
            print(f"\n分組 {group} top 3:")
            print(top[['trial_number', 'score', 'source_file']])
else:
    print("找不到 hierarchical_cluster 欄位")

# 5. hierarchical_cluster/param_cluster 分組分布與參數距離
# 檢查參數欄位
param_cols = [c for c in merged.columns if c.startswith('param_')]
print(f"找到的參數欄位: {param_cols}")

if len(param_cols) >= 2:
    # PCA 2D 降維
    pca = PCA(n_components=2)
    X = merged[param_cols].fillna(0)
    X_pca = pca.fit_transform(X)
    merged['pca1'] = X_pca[:,0]
    merged['pca2'] = X_pca[:,1]

    # 畫圖：hierarchical_cluster
    if 'hierarchical_cluster' in merged.columns:
        plt.figure(figsize=(6,5))
        for group in merged['hierarchical_cluster'].dropna().unique():
            sub = merged[merged['hierarchical_cluster']==group]
            plt.scatter(sub['pca1'], sub['pca2'], label=f'Group {group}', alpha=0.6)
        plt.title('PCA by hierarchical_cluster')
        plt.legend()
        plt.show()

    # 畫圖：param_cluster
    if 'param_cluster' in merged.columns:
        plt.figure(figsize=(6,5))
        for group in merged['param_cluster'].dropna().unique():
            sub = merged[merged['param_cluster']==group]
            plt.scatter(sub['pca1'], sub['pca2'], label=f'Group {group}', alpha=0.6)
        plt.title('PCA by param_cluster')
        plt.legend()
        plt.show()
else:
    print("找不到足夠的參數欄位（需要至少 2 個 param_ 開頭的欄位）")
    print("跳過 PCA 視覺化")

# 6. 輸出合併後資料
merged.to_csv('results/op16/merged_op_hsa.csv', index=False)
print("\n已輸出合併後資料 results/op16/merged_op_hsa.csv")