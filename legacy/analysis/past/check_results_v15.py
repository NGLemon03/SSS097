# v15: 檢查結果，驗證分群效果與穩定性
# 主要功能：結果驗證、穩定性檢查、統計分析、效果評估
# 確保分群結果的可靠性和有效性

import pandas as pd
import numpy as np

# 檢查RMA_Self結果
print("=== RMA_Self 多階段分群結果 ===")
df_rma = pd.read_csv('results/multi_stage_clustering/RMA_Self_Self_multi_stage_clustered.csv')
print(f"總數據量: {len(df_rma)}")
print(f"總群組數: {df_rma['multi_stage_cluster'].nunique()}")
print(f"平均群組大小: {len(df_rma) / df_rma['multi_stage_cluster'].nunique():.1f}")

# 群組大小分布
cluster_sizes = df_rma['multi_stage_cluster'].value_counts().sort_index()
print(f"\n群組大小分布:")
print(cluster_sizes)

# 統計小群組
small_clusters = cluster_sizes[cluster_sizes <= 5]
medium_clusters = cluster_sizes[(cluster_sizes > 5) & (cluster_sizes <= 10)]
large_clusters = cluster_sizes[cluster_sizes > 10]

print(f"\n小群組 (≤5): {len(small_clusters)} 個")
print(f"中群組 (6-10): {len(medium_clusters)} 個")
print(f"大群組 (>10): {len(large_clusters)} 個")
print(f"小群組比例: {len(small_clusters) / len(cluster_sizes):.3f}")

# 檢查single_Self結果
print(f"\n=== single_Self 多階段分群結果 ===")
df_single = pd.read_csv('results/multi_stage_clustering/single_Self_Self_multi_stage_clustered.csv')
print(f"總數據量: {len(df_single)}")
print(f"總群組數: {df_single['multi_stage_cluster'].nunique()}")
print(f"平均群組大小: {len(df_single) / df_single['multi_stage_cluster'].nunique():.1f}")

# 群組大小分布
cluster_sizes_single = df_single['multi_stage_cluster'].value_counts().sort_index()
print(f"\n群組大小分布:")
print(cluster_sizes_single)

# 統計小群組
small_clusters_single = cluster_sizes_single[cluster_sizes_single <= 5]
medium_clusters_single = cluster_sizes_single[(cluster_sizes_single > 5) & (cluster_sizes_single <= 10)]
large_clusters_single = cluster_sizes_single[cluster_sizes_single > 10]

print(f"\n小群組 (≤5): {len(small_clusters_single)} 個")
print(f"中群組 (6-10): {len(medium_clusters_single)} 個")
print(f"大群組 (>10): {len(large_clusters_single)} 個")
print(f"小群組比例: {len(small_clusters_single) / len(cluster_sizes_single):.3f}")

# 檢查ssma_turn_Self結果
print(f"\n=== ssma_turn_Self 多階段分群結果 ===")
df_ssma = pd.read_csv('results/multi_stage_clustering/ssma_turn_Self_Self_multi_stage_clustered.csv')
print(f"總數據量: {len(df_ssma)}")
print(f"總群組數: {df_ssma['multi_stage_cluster'].nunique()}")
print(f"平均群組大小: {len(df_ssma) / df_ssma['multi_stage_cluster'].nunique():.1f}")

# 群組大小分布
cluster_sizes_ssma = df_ssma['multi_stage_cluster'].value_counts().sort_index()
print(f"\n群組大小分布:")
print(cluster_sizes_ssma)

# 統計小群組
small_clusters_ssma = cluster_sizes_ssma[cluster_sizes_ssma <= 5]
medium_clusters_ssma = cluster_sizes_ssma[(cluster_sizes_ssma > 5) & (cluster_sizes_ssma <= 10)]
large_clusters_ssma = cluster_sizes_ssma[cluster_sizes_ssma > 10]

print(f"\n小群組 (≤5): {len(small_clusters_ssma)} 個")
print(f"中群組 (6-10): {len(medium_clusters_ssma)} 個")
print(f"大群組 (>10): {len(large_clusters_ssma)} 個")
print(f"小群組比例: {len(small_clusters_ssma) / len(cluster_sizes_ssma):.3f}") 