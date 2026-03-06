# v18: 檢查簡化結果，驗證簡化分群效果
# 主要功能：簡化結果驗證、統計分析、效果評估、穩定性檢查
# 確保簡化分群方法的有效性

import pandas as pd
import numpy as np

# 檢查RMA_Self結果
print("=== RMA_Self 簡化細粒度分群結果 ===")
df_rma = pd.read_csv('results/simple_fine_clustering/RMA_Self_Self_simple_fine_clustered.csv')
print(f"總數據量: {len(df_rma)}")
print(f"總群組數: {df_rma['simple_fine_cluster'].nunique()}")
print(f"平均群組大小: {len(df_rma) / df_rma['simple_fine_cluster'].nunique():.1f}")

# 群組大小分布
cluster_sizes = df_rma['simple_fine_cluster'].value_counts().sort_index()
print(f"\n群組大小分布:")
print(cluster_sizes.value_counts().sort_index())

# 統計小群組
small_clusters = cluster_sizes[cluster_sizes <= 5]
medium_clusters = cluster_sizes[(cluster_sizes > 5) & (cluster_sizes <= 10)]
large_clusters = cluster_sizes[cluster_sizes > 10]

print(f"\n小群組 (≤5): {len(small_clusters)} 個")
print(f"中群組 (6-10): {len(medium_clusters)} 個")
print(f"大群組 (>10): {len(large_clusters)} 個")
print(f"小群組比例: {len(small_clusters) / len(cluster_sizes):.3f}")

# 檢查single_Self結果
print(f"\n=== single_Self 簡化細粒度分群結果 ===")
df_single = pd.read_csv('results/simple_fine_clustering/single_Self_Self_simple_fine_clustered.csv')
print(f"總數據量: {len(df_single)}")
print(f"總群組數: {df_single['simple_fine_cluster'].nunique()}")
print(f"平均群組大小: {len(df_single) / df_single['simple_fine_cluster'].nunique():.1f}")

# 群組大小分布
cluster_sizes_single = df_single['simple_fine_cluster'].value_counts().sort_index()
print(f"\n群組大小分布:")
print(cluster_sizes_single.value_counts().sort_index())

# 統計小群組
small_clusters_single = cluster_sizes_single[cluster_sizes_single <= 5]
medium_clusters_single = cluster_sizes_single[(cluster_sizes_single > 5) & (cluster_sizes_single <= 10)]
large_clusters_single = cluster_sizes_single[cluster_sizes_single > 10]

print(f"\n小群組 (≤5): {len(small_clusters_single)} 個")
print(f"中群組 (6-10): {len(medium_clusters_single)} 個")
print(f"大群組 (>10): {len(large_clusters_single)} 個")
print(f"小群組比例: {len(small_clusters_single) / len(cluster_sizes_single):.3f}")

# 檢查ssma_turn_Self結果
print(f"\n=== ssma_turn_Self 簡化細粒度分群結果 ===")
df_ssma = pd.read_csv('results/simple_fine_clustering/ssma_turn_Self_Self_simple_fine_clustered.csv')
print(f"總數據量: {len(df_ssma)}")
print(f"總群組數: {df_ssma['simple_fine_cluster'].nunique()}")
print(f"平均群組大小: {len(df_ssma) / df_ssma['simple_fine_cluster'].nunique():.1f}")

# 群組大小分布
cluster_sizes_ssma = df_ssma['simple_fine_cluster'].value_counts().sort_index()
print(f"\n群組大小分布:")
print(cluster_sizes_ssma.value_counts().sort_index())

# 統計小群組
small_clusters_ssma = cluster_sizes_ssma[cluster_sizes_ssma <= 5]
medium_clusters_ssma = cluster_sizes_ssma[(cluster_sizes_ssma > 5) & (cluster_sizes_ssma <= 10)]
large_clusters_ssma = cluster_sizes_ssma[cluster_sizes_ssma > 10]

print(f"\n小群組 (≤5): {len(small_clusters_ssma)} 個")
print(f"中群組 (6-10): {len(medium_clusters_ssma)} 個")
print(f"大群組 (>10): {len(large_clusters_ssma)} 個")
print(f"小群組比例: {len(small_clusters_ssma) / len(cluster_sizes_ssma):.3f}")

# 總結
print(f"\n=== 總結 ===")
print(f"RMA_Self: {len(cluster_sizes)} 個群組，小群組比例 {len(small_clusters) / len(cluster_sizes):.3f}")
print(f"single_Self: {len(cluster_sizes_single)} 個群組，小群組比例 {len(small_clusters_single) / len(cluster_sizes_single):.3f}")
print(f"ssma_turn_Self: {len(cluster_sizes_ssma)} 個群組，小群組比例 {len(small_clusters_ssma) / len(cluster_sizes_ssma):.3f}") 