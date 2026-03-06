# analysis/ROEA.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import config as cfg

RESULT_DIR = ROOT / 'results'
PLOT_DIR = RESULT_DIR / 'plots'
RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

SYMBOL = cfg.TICKER

def load_data():
    """載入回測與壓力測試數據
    注意：若 walk_forward_v14.py 或 grid_search_v14.py 輸出檔名變更，需同步更新以下路徑
    """
    wf_file = RESULT_DIR / f'wf_grid_ALL_{SYMBOL.replace("^","")}.csv'
    grid_file = RESULT_DIR / f'grid_ALL_{SYMBOL.replace("^","")}.csv'
    stress_file = RESULT_DIR / f'stress_test_results_{SYMBOL.replace("^","")}.csv'

    dfs = {}
    for name, file in [('walk_forward', wf_file), ('grid_search', grid_file), ('stress_test', stress_file)]:
        if file.exists():
            dfs[name] = pd.read_csv(file)
            print(f'✅ Loaded {name} data from {file}')
        else:
            print(f'⚠ {file} not found, skipping.')
            dfs[name] = pd.DataFrame()

    return dfs['walk_forward'], dfs['grid_search'], dfs['stress_test']


def analyze_performance(wf_df, grid_df, stress_df):
    summary_records = []
    from SSSv092 import calculate_metrics
    all_metric_cols = list(calculate_metrics([], pd.DataFrame()).keys())
    if not wf_df.empty:
        metric_cols = [c for c in wf_df.columns if c in all_metric_cols]
        agg_dict = {m: ['mean', 'std', 'min', 'max'] for m in metric_cols}
        wf_agg = wf_df.groupby('strategy').agg(agg_dict).reset_index()
        wf_agg.columns = ['strategy'] + [f'normal_{m}_{s}' for m in metric_cols for s in ['mean', 'std', 'min', 'max']]
        print('📊 Normal period analysis completed')
    else:
        wf_agg = pd.DataFrame()
        print('⚠ No walk-forward data for normal period analysis')

    if not stress_df.empty:
        metric_cols = [c for c in stress_df.columns if c in all_metric_cols]
        agg_dict = {m: ['mean', 'std', 'min', 'max'] for m in metric_cols}
        stress_agg = stress_df.groupby(['strategy', 'period']).agg(agg_dict).reset_index()
        stress_agg.columns = ['strategy', 'period'] + [f'stress_{m}_{s}' for m in metric_cols for s in ['mean', 'std', 'min', 'max']]

        stress_overall = stress_df.groupby('strategy').agg({m: ['mean', 'std'] for m in metric_cols}).reset_index()
        stress_overall.columns = ['strategy'] + [f'stress_{m}_{s}' for m in metric_cols for s in ['mean', 'std']]
        print('📊 Stress period analysis completed')
    else:
        stress_agg = pd.DataFrame()
        stress_overall = pd.DataFrame()
        print('⚠ No stress test data for analysis')

    if not wf_agg.empty and not stress_overall.empty:
        summary = pd.merge(wf_agg, stress_overall, on='strategy', how='outer')
        summary['return_stability'] = summary['normal_total_return_mean'] - summary['stress_total_return_mean']
        summary['robustness_score'] = summary['normal_sharpe_ratio_mean'] / (summary['stress_total_return_std'] + 1e-6)
        summary_records.append(summary)
    elif not wf_agg.empty:
        summary_records.append(wf_agg)
    elif not stress_overall.empty:
        summary_records.append(stress_overall)

    if summary_records:
        summary_df = pd.concat(summary_records, ignore_index=True)
        summary_file = RESULT_DIR / f'roea_summary_{SYMBOL.replace("^","")}.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f'📦 Summary saved to {summary_file}')
        return summary_df, stress_agg
    else:
        print('⚠ No data to summarize')
        return pd.DataFrame(), pd.DataFrame()

def plot_scatter(summary_df, wf_df, stress_df):
    if summary_df.empty or wf_df.empty or stress_df.empty:
        print('⚠ Insufficient data for scatter plot')
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=summary_df, x='normal_total_return_mean', y='stress_total_return_mean',
                    size='normal_sharpe_ratio_mean', hue='strategy', sizes=(50, 200))
    plt.axhline(0, color='red', linestyle='--', alpha=0.3)
    plt.axvline(0, color='red', linestyle='--', alpha=0.3)
    plt.xlabel('Normal Period Total Return (Mean)')
    plt.ylabel('Stress Period Total Return (Mean)')
    plt.title(f'{SYMBOL} Strategy Performance: Normal vs. Stress Periods')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_file = PLOT_DIR / f'roea_scatter_{SYMBOL.replace("^","")}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Scatter plot saved to {plot_file}')

def plot_boxplot(stress_df):
    if stress_df.empty:
        print('⚠ No stress test data for boxplot')
        return

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=stress_df, x='strategy', y='total_return', hue='period')
    plt.axhline(0, color='red', linestyle='--', alpha=0.3)
    plt.xlabel('Strategy')
    plt.ylabel('Total Return')
    plt.title(f'{SYMBOL} Total Return Distribution in Stress Periods')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_file = PLOT_DIR / f'roea_boxplot_{SYMBOL.replace("^","")}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Boxplot saved to {plot_file}')

def plot_heatmap(stress_df):
    if stress_df.empty:
        print('⚠ No stress test data for heatmap')
        return

    pivot = stress_df.pivot_table(values='total_return', index='strategy', columns='period', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
    plt.title(f'{SYMBOL} Total Return Stability Across Stress Periods')
    plt.xlabel('Stress Period')
    plt.ylabel('Strategy')
    plt.tight_layout()
    plot_file = PLOT_DIR / f'roea_heatmap_{SYMBOL.replace("^","")}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'📊 Heatmap saved to {plot_file}')

def main():
    wf_df, grid_df, stress_df = load_data()
    summary_df, stress_agg = analyze_performance(wf_df, grid_df, stress_df)
    if not summary_df.empty:
        plot_scatter(summary_df, wf_df, stress_df)
        plot_boxplot(stress_df)
        plot_heatmap(stress_df)
    if not summary_df.empty:
        best_strategy = summary_df.loc[summary_df['robustness_score'].idxmax()]
        print(f'🏆 Recommended Strategy: {best_strategy["strategy"]}')
        print(f'   Normal Return (Mean): {best_strategy["normal_total_return_mean"]:.2%}')
        print(f'   Stress Return (Mean): {best_strategy["stress_total_return_mean"]:.2%}')
        print(f'   Robustness Score: {best_strategy["robustness_score"]:.2f}')

if __name__ == '__main__':
    main()