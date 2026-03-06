# v1: 最早的分群與分數分析腳本，僅用於初步 score/metric 關聯與簡單聚類
# 主要功能：score/metric 散點、KMeans聚類、相關性分析
# 後續版本有更細緻的分群與多特徵分析

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import os
import glob
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

HIERARCHICAL_DIR = 'results/op16'
OUTPUT_DIR = 'results/hierarchical'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_datasource_from_filename(filename):
    # 只允許 Self、2412、2414
    fname = filename.upper().replace(' ', '').replace('(', '').replace(')', '').replace('^', '').replace('/', '').replace('.', '').replace('_', '')
    if 'SELF' in fname:
        return 'Self'
    if '2412' in fname:
        return '2412'
    if '2414' in fname:
        return '2414'
    return None

# 1. 先將所有檔案依策略、分 optuna/hier 兩組，然後每組內再依數據源分組
all_files = glob.glob(f'{HIERARCHICAL_DIR}/*.csv')
strategy_group_map = {}  # {(strategy, group): {datasource: [file, ...]}}
file_pattern = re.compile(r'optuna_results_([a-zA-Z0-9]+)_([^_]+)')
for f in all_files:
    fname = os.path.basename(f)
    m = file_pattern.search(fname)
    if m:
        strategy = m.group(1)
        datasource = extract_datasource_from_filename(fname)
        group = 'hier' if '_hierarchical_final_' in fname else 'optuna'
        if datasource is None:
            print(f'警告：無法判斷數據源，略過檔案 {f}')
            continue
        key = (strategy, group)
        if key not in strategy_group_map:
            strategy_group_map[key] = {}
        if datasource not in strategy_group_map[key]:
            strategy_group_map[key][datasource] = []
        strategy_group_map[key][datasource].append(f)

metric_targets = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor', 'cpcv_oos_mean', 'cpcv_oos_min']

for (strategy, group), ds_files in strategy_group_map.items():
    datasources = list(ds_files.keys())
    n_ds = len(datasources)
    n_metrics = len(metric_targets)
    fig, axes = plt.subplots(n_ds, n_metrics, figsize=(5 * n_metrics, 4 * n_ds), squeeze=False)
    for i, datasource in enumerate(datasources):
        files = ds_files[datasource]
        dfs = [pd.read_csv(path) for path in files]
        df = pd.concat(dfs, ignore_index=True)
        # 自動展開 parameters 欄位
        if 'parameters' in df.columns:
            param_df = df['parameters'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
            param_df = pd.json_normalize(param_df.tolist())
            param_df = param_df.add_prefix('param_')
            df = pd.concat([df, param_df], axis=1)
        if 'score' not in df.columns:
            continue
        for j, metric in enumerate(metric_targets):
            ax = axes[i, j]
            if metric not in df.columns:
                ax.set_visible(False)
                continue
            mask = (~df['score'].isna()) & np.isfinite(df['score']) & (~df[metric].isna()) & np.isfinite(df[metric])
            if mask.sum() == 0:
                ax.set_visible(False)
                continue
            # 排除極端分數（僅繪圖用）：score > (top 10% 平均值 * 2)
            scores = df.loc[mask, 'score']
            top10 = np.percentile(scores, 90)
            top10_mean = scores[scores >= top10].mean() if (scores >= top10).sum() > 0 else scores.mean()
            threshold = top10_mean * 2
            plot_mask = mask & (df['score'] <= threshold)
            if plot_mask.sum() == 0:
                ax.set_visible(False)
                continue
            ax.scatter(df.loc[plot_mask, 'score'], df.loc[plot_mask, metric], alpha=0.4, s=18)
            ax.set_xlabel('score')
            ax.set_ylabel(metric)
            ax.set_title(f'{strategy} {group} {datasource} vs {metric}')
            ax.grid(True)
        # ====== 新增：自動聚類並繪製 score vs. sharpe_ratio cluster 標色圖 ======
        if 'sharpe_ratio' in df.columns:
            param_cols = [col for col in df.columns if col.startswith('param_')]
            if len(param_cols) >= 2 and len(df) >= 10:
                X = df[param_cols].values
                X_pca = PCA(n_components=2).fit_transform(X)
                # 自動決定最佳分群數
                best_k = 2
                best_score = -1
                for k in range(2, min(6, len(df))):
                    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
                    score = silhouette_score(X_pca, kmeans.labels_)
                    if score > best_score:
                        best_score = score
                        best_k = k
                kmeans = KMeans(n_clusters=best_k, random_state=42).fit(X_pca)
                df['param_cluster'] = kmeans.labels_
                # LOG: 分群資訊
                cluster_counts = np.bincount(df['param_cluster'])
                print(f'[CLUSTER] {strategy}-{group}-{datasource}: best_k={best_k}, cluster_counts={cluster_counts}')
                # 排除極端分數（同上）
                mask = (~df['score'].isna()) & np.isfinite(df['score']) & (~df['sharpe_ratio'].isna()) & np.isfinite(df['sharpe_ratio'])
                scores = df.loc[mask, 'score']
                top10 = np.percentile(scores, 90)
                top10_mean = scores[scores >= top10].mean() if (scores >= top10).sum() > 0 else scores.mean()
                threshold = top10_mean * 2
                plot_mask = mask & (df['score'] <= threshold)
                plt.figure(figsize=(7, 5))
                for c in np.unique(df['param_cluster']):
                    c_mask = plot_mask & (df['param_cluster'] == c)
                    plt.scatter(df.loc[c_mask, 'score'], df.loc[c_mask, 'sharpe_ratio'], label=f'cluster {c}', alpha=0.5)
                plt.xlabel('score')
                plt.ylabel('sharpe_ratio')
                plt.title(f'{strategy} {group} {datasource} score vs sharpe_ratio (cluster, k={best_k})')
                plt.legend()
                plt.tight_layout()
                out_png = f'{OUTPUT_DIR}/{group}_{strategy}_{datasource}_score_vs_sharpe_cluster.png'
                plt.savefig(out_png, dpi=150)
                plt.close()
                # 輸出 cluster label
                out_csv = f'{OUTPUT_DIR}/{group}_{strategy}_{datasource}_score_vs_sharpe_cluster.csv'
                df.to_csv(out_csv, index=False)
            else:
                print(f'[CLUSTER] {strategy}-{group}-{datasource}: skip clustering (param_cols={len(param_cols)}, n_trials={len(df)})')
        plt.tight_layout()
    out_png = f'{OUTPUT_DIR}/{group}_{strategy}_score_vs_metrics.png'
    plt.savefig(out_png, dpi=150)
    plt.close()
    corrs = []
    for datasource, files in ds_files.items():
        dfs = [pd.read_csv(path) for path in files]
        df = pd.concat(dfs, ignore_index=True)
        row = {'策略': strategy, 'group': group, '資料源': datasource}
        if 'score' not in df.columns:
            continue
        for metric in metric_targets:
            if metric in df.columns:
                mask = (~df['score'].isna()) & np.isfinite(df['score']) & (~df[metric].isna()) & np.isfinite(df[metric])
                if mask.sum() > 1:
                    try:
                        c, _ = pearsonr(df.loc[mask, 'score'], df.loc[mask, metric])
                    except Exception:
                        c = np.nan
                else:
                    c = np.nan
                row[f'score vs {metric}'] = c
        corrs.append(row)
    corr_df = pd.DataFrame(corrs)
    out_csv = f'{OUTPUT_DIR}/{group}_{strategy}_score_vs_metrics_corr.csv'
    corr_df.to_csv(out_csv, index=False)
    print(f'{group}-{strategy} 各數據源 score 與多重指標相關性如下：')
    print(corr_df)

# === 自動合併所有 param_cluster 標籤 ===
SRC_DIR = OUTPUT_DIR  # 這裡 OUTPUT_DIR = 'results/hierarchical'
DST_DIR = 'results/op16/hierarchical_score_analysis'
os.makedirs(DST_DIR, exist_ok=True)

files = glob.glob(os.path.join(SRC_DIR, '*_score_vs_sharpe_cluster.csv'))

dfs = []
all_columns = set(['trial_id', 'param_cluster'])  # 收集所有可能的欄位

# 第一遍：收集所有欄位名稱
for f in files:
    df = pd.read_csv(f)
    if 'trial_number' in df.columns and 'param_cluster' in df.columns:
        param_cols = [col for col in df.columns if col.startswith('param_')]
        all_columns.update(param_cols)

# 第二遍：統一欄位並合併
for f in files:
    df = pd.read_csv(f)
    if 'trial_number' in df.columns and 'param_cluster' in df.columns:
        # 創建統一的欄位結構
        df_temp = pd.DataFrame(columns=list(all_columns))
        
        # 填充資料
        df_temp['trial_id'] = df['trial_number']
        df_temp['param_cluster'] = df['param_cluster']
        
        # 填充參數欄位
        for col in all_columns:
            if col.startswith('param_') and col in df.columns:
                df_temp[col] = df[col]
        
        dfs.append(df_temp)
    else:
        print(f'檔案 {f} 缺少必要欄位，已略過')

if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=['trial_id'])
    merged.to_csv(os.path.join(DST_DIR, 'param_cluster_labels.csv'), index=False)
    print('已自動合併並產生 param_cluster_labels.csv')
else:
    print('沒有可合併的 param_cluster 標籤檔案')