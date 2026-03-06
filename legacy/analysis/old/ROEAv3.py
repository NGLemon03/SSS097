import sys, os
from pathlib import Path

# 1. 將專案根目錄加入模組搜尋路徑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 匯入 config
import analysis.config as cfg

ticker = cfg.TICKER
RESULT_DIR = cfg.RESULT_DIR
PLOT_DIR = cfg.PLOT_DIR


# 壓力測試期
STRESS_PERIODS = cfg.STRESS_PERIODS        
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from SSSv092 import calculate_metrics


def load_data():
    wf_csv     = RESULT_DIR / f"wf_grid_ALL_{ticker.replace('^','')}.csv"
    grid_csv   = RESULT_DIR / f"grid_ALL_{ticker.replace('^','')}.csv"
    stress_csv = RESULT_DIR / f"stress_test_results_{ticker.replace('^','')}.csv"
    return (
        pd.read_csv(wf_csv),
        pd.read_csv(grid_csv),
        pd.read_csv(stress_csv),
    )


def plot_and_stats(df, x_col, y_col, title, groupby=None):
    if len(df) < 2:
        print(f"⚠ Not enough points to plot: {title}")
        return
    if groupby:
        for g in df[groupby].unique():
            sub = df[df[groupby] == g]
            x, y = sub[x_col].astype(float), sub[y_col].astype(float)
            if len(x) < 2:
                continue
            corr, p = pearsonr(x, y)
            thr_x = x.quantile(0.70)
            thr_y = y.quantile(0.50)
            outliers = (x >= thr_x) & (y <= thr_y)
            pct = outliers.sum() / len(x) * 100
            plt.figure(figsize=(6,5))
            plt.scatter(x, y, alpha=0.4, label="All")
            plt.scatter(x[outliers], y[outliers], edgecolors='black', label=f"Outliers ({pct:.1f}%)")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{title} ({g})\nPearson r={corr:.3f}, p={p:.3g}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            (PLOT_DIR / f"roea_scatter_{ticker.replace('^','')}_{g}.png").parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(PLOT_DIR / f"roea_scatter_{ticker.replace('^','')}_{g}.png", dpi=300)


def plot_scatter(summary_df, wf_df, stress_df):
    plt.figure(figsize=(6,5))
    plt.scatter(summary_df['normal_total_return_mean'], summary_df['stress_total_return_mean'], alpha=0.6)
    plt.xlabel('Normal Mean Return')
    plt.ylabel('Stress Mean Return')
    plt.title(f'{ticker} Normal vs Stress Return')
    plt.grid(True)
    plt.tight_layout()
    (PLOT_DIR / f"roea_scatter_{ticker.replace('^','')}.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"roea_scatter_{ticker.replace('^','')}.png", dpi=300)


def plot_boxplot(stress_df):
    plt.figure(figsize=(6,5))
    sns.boxplot(x='strategy', y='total_return', data=stress_df)
    plt.title(f'{ticker} Stress Period Returns by Strategy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    (PLOT_DIR / f"roea_boxplot_{ticker.replace('^','')}.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"roea_boxplot_{ticker.replace('^','')}.png", dpi=300)


def plot_heatmap(stress_df):
    pivot = stress_df.pivot_table(index='strategy', columns='period', values='total_return')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
    plt.title(f'{ticker} Stress Returns Heatmap')
    plt.tight_layout()
    (PLOT_DIR / f"roea_heatmap_{ticker.replace('^','')}.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"roea_heatmap_{ticker.replace('^','')}.png", dpi=300)


def plot_sortino_heatmap(stress_df):
    pivot = stress_df.pivot_table(index='strategy', columns='period', values='sortino_ratio')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='magma')
    plt.title(f'{ticker} Sortino Ratio Stability Across Stress Periods')
    plt.tight_layout()
    (PLOT_DIR / f"roea_sortino_heatmap_{ticker.replace('^','')}.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"roea_sortino_heatmap_{ticker.replace('^','')}.png", dpi=300)


def analyze_performance(wf_df, grid_df, stress_df):
    summary_records = []
    all_metric_cols = list(calculate_metrics([], pd.DataFrame()).keys())

    # Normal period aggregation
    if not wf_df.empty:
        metric_cols = [c for c in wf_df.columns if c in all_metric_cols]
        agg_dict = {m: ['mean', 'std', 'min', 'max'] for m in metric_cols}
        wf_agg = wf_df.groupby('strategy').agg(agg_dict).reset_index()
        wf_agg.columns = ['strategy'] + [f'normal_{m}_{s}' for m in metric_cols for s in ['mean','std','min','max']]
        print('📊 Normal period analysis completed')
    else:
        wf_agg = pd.DataFrame()
        print('⚠ No walk-forward data for normal period analysis')

    # Stress period aggregation
    if not stress_df.empty:
        metric_cols = [c for c in stress_df.columns if c in all_metric_cols]
        stress_overall = stress_df.groupby('strategy')[metric_cols].agg(['mean','std']).reset_index()
        stress_overall.columns = ['strategy'] + [f'stress_{m}_{s}' for m in metric_cols for s in ['mean','std']]
        print('📊 Stress period analysis completed')
    else:
        stress_overall = pd.DataFrame()
        print('⚠ No stress data for stress period analysis')

    # Merge summary
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
        summary_file = RESULT_DIR / f"roea_summary_{ticker.replace('^','')}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f'📦 Summary saved to {summary_file}')
    else:
        summary_df = pd.DataFrame()

    # 繪圖
    if not summary_df.empty:
        plot_scatter(summary_df, wf_df, stress_df)
        plot_boxplot(stress_df)
        plot_heatmap(stress_df)
        plot_sortino_heatmap(stress_df)

    # 詳細散點圖
    if not wf_df.empty:
        plot_and_stats(wf_df, 'total_return', 'sharpe_ratio', 'Total Return vs Sharpe Ratio', groupby='strategy')
    if not stress_df.empty:
        for p in stress_df['period'].unique():
            sub = stress_df[stress_df['period'] == p]
            plot_and_stats(sub, 'total_return', 'sharpe_ratio', f'Total Return vs Sharpe Ratio ({p})')

    if not summary_df.empty:
        best = summary_df.loc[summary_df['robustness_score'].idxmax()]
        print(f'🏆 Recommended Strategy: {best["strategy"]}')
        print(f'   Normal Return (Mean): {best["normal_total_return_mean"]:.2%}')
        print(f'   Stress Return (Mean): {best["stress_total_return_mean"]:.2%}')
        print(f'   Robustness Score: {best["robustness_score"]:.2f}')


def main():
    wf_df, grid_df, stress_df = load_data()

    # 參數初篩：ROI > 0
    for df in (wf_df, grid_df, stress_df):
        if 'total_return' in df:
            df.query("total_return > 0", inplace=True)

    analyze_performance(wf_df, grid_df, stress_df)


if __name__ == '__main__':
    main()
