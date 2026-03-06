import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy.stats import ttest_ind
import glob

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 自動偵測所有 single/RMA/ssma_turn 檔案
single_files = sorted(glob.glob('results/op16/optuna_results_single_*.csv'))
rma_files = sorted(glob.glob('results/op16/optuna_results_RMA_*.csv'))
ssma_files = sorted(glob.glob('results/op16/optuna_results_ssma_turn_*.csv'))
all_files = single_files + rma_files + ssma_files

output_dir = 'analysis/results/single_analysis'
os.makedirs(output_dir, exist_ok=True)

for file in all_files:
    tag = Path(file).stem.replace('optuna_results_', '')
    # 解析策略與數據源
    if tag.startswith('single'):
        strat = 'single'
    elif tag.startswith('RMA'):
        strat = 'RMA'
    elif tag.startswith('ssma_turn'):
        strat = 'ssma_turn'
    else:
        strat = tag.split('_')[0]
    if 'Self' in tag:
        datasource = 'Self'
    elif '2412' in tag:
        datasource = '2412'
    elif '2414' in tag:
        datasource = '2414'
    else:
        datasource = tag
    score100_dir = f'analysis/results/single_analysis/score100_split/{tag}'
    os.makedirs(score100_dir, exist_ok=True)
    df = pd.read_csv(file)
    if 'parameters' in df.columns:
        param_df = df['parameters'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
        param_df = pd.json_normalize(param_df.tolist())
        param_df = param_df.add_prefix('param_')
        df = pd.concat([df, param_df], axis=1)
    # 1. 分組
    high = df[df['score'] > 100]
    low = df[df['score'] < 100]
    for param in ['param_stop_loss','param_buy_mult','param_sell_mult']:
        if param in df.columns:
            high_series = pd.Series(high[param]).reset_index(drop=True)
            low_series = pd.Series(low[param]).reset_index(drop=True)
            plot_df = pd.DataFrame({
                param: pd.concat([high_series, low_series], ignore_index=True),
                'score_group': ['score>100']*len(high_series) + ['score<100']*len(low_series)
            })
            # boxplot
            plt.figure(figsize=(8,5))
            sns.boxplot(x='score_group', y=param, data=plot_df, palette='Set2')
            plt.title(f'{strat}_{datasource} {param} score>100 vs <100 boxplot')
            plt.tight_layout()
            plt.savefig(f'{score100_dir}/{strat}_{datasource}_param_{param}_boxplot_score100.png', dpi=150)
            plt.close()
            # histogram
            plt.figure(figsize=(8,5))
            plt.hist(high_series.dropna(), bins=20, alpha=0.6, label='score>100')
            plt.hist(low_series.dropna(), bins=20, alpha=0.6, label='score<100')
            plt.title(f'{strat}_{datasource} {param} score>100 vs <100 histogram')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{score100_dir}/{strat}_{datasource}_param_{param}_hist_score100.png', dpi=150)
            plt.close()
    # 2. 離散化與交叉分析
    for param in ['param_stop_loss','param_buy_mult','param_sell_mult']:
        if param in df.columns:
            df[f'{param}_bin'] = pd.qcut(df[param], 4, duplicates='drop', labels=False)
    # 交叉表
    if all([f'{p}_bin' in df.columns for p in ['param_stop_loss','param_buy_mult','param_sell_mult']]):
        ctab = pd.crosstab([df['param_stop_loss_bin'],df['param_buy_mult_bin']], df['param_sell_mult_bin'])
        plt.figure(figsize=(8,6))
        sns.heatmap(ctab, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'{strat}_{datasource} stop_loss_bin x buy_mult_bin vs sell_mult_bin')
        plt.tight_layout()
        plt.savefig(f'{score100_dir}/{strat}_{datasource}_stoploss_buymult_sellmult_heatmap.png', dpi=150)
        plt.close()
        ctab.to_csv(f'{score100_dir}/{strat}_{datasource}_stoploss_buymult_sellmult_crosstab.csv')
    # 3. hierarchical_cluster 分析
    if 'hierarchical_cluster' in df.columns:
        for param in ['param_stop_loss','param_buy_mult','param_sell_mult']:
            if param in df.columns:
                plt.figure(figsize=(8,5))
                sns.boxplot(x='hierarchical_cluster', y=param, data=df, palette='Set3')
                plt.title(f'{strat}_{datasource} {param} by hierarchical_cluster')
                plt.tight_layout()
                plt.savefig(f'{score100_dir}/{strat}_{datasource}_param_{param}_boxplot_hiercluster.png', dpi=150)
                plt.close()
        # 分布表
        htab = pd.crosstab(df['hierarchical_cluster'], [df['param_stop_loss_bin'],df['param_buy_mult_bin'],df['param_sell_mult_bin']])
        htab.to_csv(f'{score100_dir}/{strat}_{datasource}_hiercluster_param_bins_crosstab.csv')
    # 4. 自動分群分析 (score, total_return)
    if 'score' in df.columns and 'total_return' in df.columns:
        X = df[['score','total_return']].replace([np.inf,-np.inf],0).fillna(0).values
        kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
        df['auto_cluster'] = kmeans.labels_
        auto_dir = f'{score100_dir}/auto_cluster'
        os.makedirs(auto_dir, exist_ok=True)
        for p in ['param_stop_loss','param_buy_mult','param_sell_mult']:
            if p in df.columns:
                plt.figure(figsize=(8,5))
                sns.boxplot(x='auto_cluster', y=p, data=df, palette='Set1')
                plt.title(f'{strat}_{datasource} {p} (score, total_return)自動分群參數分布')
                plt.tight_layout()
                plt.savefig(f'{auto_dir}/{strat}_{datasource}_param_{p}_boxplot_auto_cluster.png', dpi=150)
                plt.close()
        group_stats = df.groupby('auto_cluster')[['param_stop_loss','param_buy_mult','param_sell_mult','score','total_return','sharpe_ratio']].mean().reset_index()
        group_stats.to_csv(f'{auto_dir}/{strat}_{datasource}_auto_cluster_param_means.csv', index=False)
        ttest_results = []
        for p in ['param_stop_loss','param_buy_mult','param_sell_mult']:
            if p in df.columns:
                g0 = df.loc[df['auto_cluster']==0, p].dropna()
                g1 = df.loc[df['auto_cluster']==1, p].dropna()
                if len(g0)>2 and len(g1)>2:
                    t,pval = ttest_ind(g0, g1, equal_var=False)
                else:
                    t,pval = np.nan, np.nan
                ttest_results.append({'param':p, 't_stat':t, 'p_value':pval})
        ttest_df = pd.DataFrame(ttest_results)
        ttest_df = ttest_df.sort_values('p_value')
        ttest_df.to_csv(f'{auto_dir}/{strat}_{datasource}_auto_cluster_param_ttest.csv', index=False)
        sig_params = ttest_df[ttest_df['p_value']<0.05]['param'].tolist()
        print(f'{strat}_{datasource} 自動分群顯著參數: {sig_params}')

print(f'分析完成，所有圖表與重要結果已輸出到 {output_dir}/') 