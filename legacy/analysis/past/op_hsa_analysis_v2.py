import os
import glob
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_parameters(parameters_str):
    """從參數字串中提取參數值"""
    try:
        params = json.loads(parameters_str)
        return params
    except:
        return {}

def analyze_integrated_clustering():
    """整合 OP16 和 HSA 分群分析"""
    
    print("=== OP16 + HSA 整合分群分析 ===\n")
    
    # 1. 讀取所有 optuna 結果
    optuna_files = glob.glob('results/op16/optuna_results_*.csv')
    print(f"找到 {len(optuna_files)} 個 optuna 結果檔案")
    
    optuna_dfs = []
    for f in optuna_files:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        # 提取策略名稱和資料源
        filename = os.path.basename(f)
        if 'single' in filename:
            df['strategy'] = 'single'
        elif 'ssma' in filename:
            df['strategy'] = 'ssma'
        elif 'RMA' in filename:
            df['strategy'] = 'RMA'
        
        if 'Self' in filename:
            df['data_source'] = 'Self'
        elif '2412' in filename:
            df['data_source'] = '2412'
        elif '2414' in filename:
            df['data_source'] = '2414'
        
        optuna_dfs.append(df)
    
    optuna_all = pd.concat(optuna_dfs, ignore_index=True)
    print(f"總共 {len(optuna_all)} 個 trials")
    
    # 2. 讀取 HSA 標籤
    hsa_path = 'results/op16/hierarchical_score_analysis/param_cluster_labels.csv'
    if os.path.exists(hsa_path):
        hsa_df = pd.read_csv(hsa_path)
        print(f"找到 HSA 分群標籤，共 {len(hsa_df)} 個")
        print(f"HSA 檔案欄位: {list(hsa_df.columns)}")
    else:
        print("找不到 HSA 分群標籤檔案")
        print(f"嘗試的路徑: {hsa_path}")
        return
    
    # 3. 合併資料
    merged = pd.merge(optuna_all, hsa_df, left_on='trial_number', right_on='trial_id', how='left')
    print(f"合併後資料：{len(merged)} 筆")
    
    # 4. 分析分群分布
    print("\n=== 分群分布分析 ===")
    
    # OP16 hierarchical clustering 分布
    if 'hierarchical_cluster' in merged.columns:
        hier_counts = merged['hierarchical_cluster'].value_counts().sort_index()
        print(f"\nOP16 Hierarchical Clusters ({len(hier_counts)} 個分群):")
        for cluster, count in hier_counts.items():
            print(f"  分群 {cluster}: {count} 個 trials")
    
    # HSA param clustering 分布
    if 'param_cluster' in merged.columns:
        param_counts = merged['param_cluster'].value_counts().sort_index()
        print(f"\nHSA Param Clusters ({len(param_counts)} 個分群):")
        for cluster, count in param_counts.items():
            print(f"  分群 {cluster}: {count} 個 trials")
    
    # 5. 各分群表現分析
    print("\n=== 各分群表現分析 ===")
    
    # OP16 hierarchical clusters 表現
    if 'hierarchical_cluster' in merged.columns and 'score' in merged.columns:
        print("\nOP16 Hierarchical Clusters 表現:")
        hier_performance = merged.groupby('hierarchical_cluster')['score'].agg(['mean', 'std', 'count']).round(3)
        print(hier_performance)
        
        # 找出最佳分群
        best_hier_cluster = hier_performance['mean'].idxmax()
        print(f"\n最佳 OP16 分群: {best_hier_cluster} (平均分數: {hier_performance.loc[best_hier_cluster, 'mean']:.3f})")
    
    # HSA param clusters 表現
    if 'param_cluster' in merged.columns and 'score' in merged.columns:
        print("\nHSA Param Clusters 表現:")
        param_performance = merged.groupby('param_cluster')['score'].agg(['mean', 'std', 'count']).round(3)
        print(param_performance)
        
        # 找出最佳分群
        best_param_cluster = param_performance['mean'].idxmax()
        print(f"\n最佳 HSA 分群: {best_param_cluster} (平均分數: {param_performance.loc[best_param_cluster, 'mean']:.3f})")
    
    # 6. 參數空間分析
    print("\n=== 參數空間分析 ===")
    
    # 提取參數
    if 'parameters' in merged.columns:
        # 解析參數
        param_data = []
        for idx, row in merged.iterrows():
            params = extract_parameters(row['parameters'])
            if params:
                param_data.append(params)
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            print(f"找到 {len(param_df.columns)} 個參數: {list(param_df.columns)}")
            
            # 參數統計
            print("\n參數統計:")
            print(param_df.describe().round(3))
            
            # 參數與分群關係
            if 'param_cluster' in merged.columns and len(param_df) == len(merged):
                print("\n=== 參數與 HSA 分群關係 ===")
                param_df['param_cluster'] = merged['param_cluster'].values
                
                for param in param_df.columns[:-1]:  # 排除 param_cluster 欄位
                    if param_df[param].dtype in ['int64', 'float64']:
                        cluster_means = param_df.groupby('param_cluster')[param].mean().round(3)
                        print(f"\n{param} 在各分群的平均值:")
                        print(cluster_means)
    
    # 7. 策略和資料源分析
    print("\n=== 策略和資料源分析 ===")
    
    if 'strategy' in merged.columns and 'data_source' in merged.columns:
        strategy_source_perf = merged.groupby(['strategy', 'data_source'])['score'].agg(['mean', 'count']).round(3)
        print("各策略-資料源組合的表現:")
        print(strategy_source_perf)
    
    # 8. 輸出結果
    output_path = 'analysis/results/op16/integrated_analysis_results.csv'
    merged.to_csv(output_path, index=False)
    print(f"\n已輸出整合分析結果到: {output_path}")
    
    # 9. 生成報告
    report_path = 'analysis/results/op16/integrated_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== OP16 + HSA 整合分群分析報告 ===\n\n")
        f.write(f"分析時間: {pd.Timestamp.now()}\n")
        f.write(f"總 trials 數: {len(merged)}\n")
        f.write(f"策略數: {merged['strategy'].nunique() if 'strategy' in merged.columns else 'N/A'}\n")
        f.write(f"資料源數: {merged['data_source'].nunique() if 'data_source' in merged.columns else 'N/A'}\n")
        
        if 'hierarchical_cluster' in merged.columns:
            f.write(f"OP16 分群數: {merged['hierarchical_cluster'].nunique()}\n")
        if 'param_cluster' in merged.columns:
            f.write(f"HSA 分群數: {merged['param_cluster'].nunique()}\n")
    
    print(f"已生成分析報告: {report_path}")
    
    # 新增：分策略分數據源的 param_cluster 統計與分布圖
    analyze_param_cluster_by_strategy(merged)
    
    return merged

def analyze_param_cluster_by_strategy(merged):
    param_cols = [col for col in merged.columns if col.startswith('param_') and col != 'param_cluster']
    output_root = 'analysis/results/op16/param_cluster_distributions_by_strategy'
    os.makedirs(output_root, exist_ok=True)

    for (strategy, data_source), group in merged.groupby(['strategy', 'data_source']):
        tag = f"{strategy}_{data_source}"
        output_dir = os.path.join(output_root, tag)
        os.makedirs(output_dir, exist_ok=True)
        
        # 檢查是否有參數數據
        group_with_params = group[group[param_cols].notna().any(axis=1)]
        if len(group_with_params) == 0:
            print(f"\n=== {tag} ===")
            print(f"trials: {len(group)} - 沒有參數數據，跳過")
            continue
            
        print(f"\n=== {tag} ===")
        print(f"trials: {len(group_with_params)}")
        
        # 重新進行參數分群（按策略-數據源分組）
        if len(param_cols) >= 2 and len(group_with_params) >= 10:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # 準備參數數據
            X = group_with_params[param_cols].fillna(0).values
            
            # PCA 降維
            X_pca = PCA(n_components=2).fit_transform(X)
            
            # 自動決定最佳分群數
            best_k = 2
            best_score = -1
            for k in range(2, min(6, len(group_with_params))):
                kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
                score = silhouette_score(X_pca, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # 最終分群
            kmeans = KMeans(n_clusters=best_k, random_state=42).fit(X_pca)
            group_with_params = group_with_params.copy()
            group_with_params['local_param_cluster'] = kmeans.labels_
            
            print(f"重新分群結果: {best_k} 個分群，silhouette_score: {best_score:.3f}")
            cluster_counts = np.bincount(group_with_params['local_param_cluster'])
            print(f"各分群數量: {cluster_counts}")
            
            # 1. total_return 分布（歸一化直方圖）
            valid_return = group_with_params[group_with_params['total_return'].notna()]
            if len(valid_return) > 0:
                plt.figure(figsize=(12, 6))
                # 使用 density=True 進行歸一化，觀察比例分布
                ax = sns.histplot(data=valid_return, x='total_return', hue='local_param_cluster',
                                 element='step', stat='density', common_norm=False,
                                 bins=30, multiple='layer', alpha=0.7)
                plt.title(f'Total Return 分布（{tag}，依 Local Param Cluster 歸一化）')
                plt.xlabel('Total Return')
                plt.ylabel('Density')
                # 確保圖例正確顯示
                if ax.get_legend() is not None:
                    ax.get_legend().set_title('Local Param Cluster')
                    ax.get_legend().set_bbox_to_anchor((1.05, 1))
                    ax.get_legend().set_loc('upper left')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/total_return_distribution_normalized.png', dpi=150, bbox_inches='tight')
                plt.close()

            # 2. score 分布（歸一化直方圖）
            valid_score = group_with_params[group_with_params['score'].notna()]
            if len(valid_score) > 0:
                plt.figure(figsize=(12, 6))
                # 使用 density=True 進行歸一化，觀察比例分布
                ax = sns.histplot(data=valid_score, x='score', hue='local_param_cluster',
                                 element='step', stat='density', common_norm=False,
                                 bins=30, multiple='layer', alpha=0.7)
                plt.title(f'Score 分布（{tag}，依 Local Param Cluster 歸一化）')
                plt.xlabel('Score')
                plt.ylabel('Density')
                # 確保圖例正確顯示
                if ax.get_legend() is not None:
                    ax.get_legend().set_title('Local Param Cluster')
                    ax.get_legend().set_bbox_to_anchor((1.05, 1))
                    ax.get_legend().set_loc('upper left')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/score_distribution_normalized.png', dpi=150, bbox_inches='tight')
                plt.close()

            # 3. 各參數分布（歸一化直方圖）
            for param in param_cols:
                if param in group_with_params.columns:
                    valid_param = group_with_params[group_with_params[param].notna()]
                    if len(valid_param) > 0:
                        plt.figure(figsize=(12, 6))
                        # 使用 density=True 進行歸一化，觀察比例分布
                        ax = sns.histplot(data=valid_param, x=param, hue='local_param_cluster',
                                         element='step', stat='density', common_norm=False,
                                         bins=20, multiple='layer', alpha=0.7)
                        plt.title(f'{param} 分布（{tag}，依 Local Param Cluster 歸一化）')
                        plt.xlabel(param)
                        plt.ylabel('Density')
                        # 確保圖例正確顯示
                        if ax.get_legend() is not None:
                            ax.get_legend().set_title('Local Param Cluster')
                            ax.get_legend().set_bbox_to_anchor((1.05, 1))
                            ax.get_legend().set_loc('upper left')
                        plt.tight_layout()
                        plt.savefig(f'{output_dir}/{param}_distribution_normalized.png', dpi=150, bbox_inches='tight')
                        plt.close()

            # 4. Boxplot - 只對有數據的分群畫圖
            if len(group_with_params['local_param_cluster'].unique()) > 1:
                # 檢查 total_return 是否有有效數據
                valid_return = group_with_params[group_with_params['total_return'].notna()]
                if len(valid_return) > 0 and len(valid_return['local_param_cluster'].unique()) > 1:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=valid_return, x='local_param_cluster', y='total_return')
                    plt.title(f'Total Return 分布（Boxplot, {tag}）')
                    plt.xlabel('Local Param Cluster')
                    plt.ylabel('Total Return')
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/total_return_boxplot.png', dpi=150, bbox_inches='tight')
                    plt.close()

                # 檢查 score 是否有有效數據
                valid_score = group_with_params[group_with_params['score'].notna()]
                if len(valid_score) > 0 and len(valid_score['local_param_cluster'].unique()) > 1:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=valid_score, x='local_param_cluster', y='score')
                    plt.title(f'Score 分布（Boxplot, {tag}）')
                    plt.xlabel('Local Param Cluster')
                    plt.ylabel('Score')
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/score_boxplot.png', dpi=150, bbox_inches='tight')
                    plt.close()

                # 各參數 boxplot
                for param in param_cols:
                    if param in group_with_params.columns:
                        valid_param = group_with_params[group_with_params[param].notna()]
                        if len(valid_param) > 0 and len(valid_param['local_param_cluster'].unique()) > 1:
                            plt.figure(figsize=(10, 6))
                            sns.boxplot(data=valid_param, x='local_param_cluster', y=param)
                            plt.title(f'{param} 分布（Boxplot, {tag}）')
                            plt.xlabel('Local Param Cluster')
                            plt.ylabel(param)
                            plt.tight_layout()
                            plt.savefig(f'{output_dir}/{param}_boxplot.png', dpi=150, bbox_inches='tight')
                            plt.close()

            # 5. Violin plot
            if len(group_with_params['local_param_cluster'].unique()) > 1:
                plt.figure(figsize=(10, 6))
                sns.violinplot(data=group_with_params, x='local_param_cluster', y='total_return')
                plt.title(f'Total Return 分布（Violin Plot, {tag}）')
                plt.xlabel('Local Param Cluster')
                plt.ylabel('Total Return')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/total_return_violin.png', dpi=150, bbox_inches='tight')
                plt.close()

            # 6. 統計摘要
            stats_summary = []
            for param in ['total_return', 'score'] + param_cols:
                if param in group_with_params.columns:
                    stats = group_with_params.groupby('local_param_cluster')[param].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
                    stats['parameter'] = param
                    stats = stats.reset_index()
                    stats_summary.append(stats)
            
            if stats_summary:
                stats_df = pd.concat(stats_summary, ignore_index=True)
                stats_df.to_csv(f'{output_dir}/local_param_cluster_statistics.csv', index=False)
                print(f"統計摘要已輸出到: {output_dir}/local_param_cluster_statistics.csv")
        else:
            print(f"數據不足，無法進行分群 (參數數: {len(param_cols)}, trials: {len(group_with_params)})")

    print(f"\n所有分策略分數據源的圖表與統計已保存到: {output_root}/")

if __name__ == "__main__":
    analyze_integrated_clustering() 