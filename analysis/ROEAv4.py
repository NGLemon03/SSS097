
# ROEAv4_fix.py
import sys
import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import prod
from scipy.stats import pearsonr, ConstantInputWarning
import warnings
from sklearn.neighbors import NearestNeighbors
import logging

# 將專案根目錄加入模組搜尋路徑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 匯入 config 與 SSSv092
from analysis import config as cfg
from SSSv095a1 import calculate_metrics, load_data_wrapper
from analysis import data_loader
# 配置參數
ticker = cfg.TICKER
RESULT_DIR = cfg.RESULT_DIR
PLOT_DIR = cfg.PLOT_DIR
STRESS_PERIODS = cfg.STRESS_PERIODS
B_H_RETURN = 12.0  # 預設 Buy-and-Hold 1200%
B_H_SHARPE = 0.5   # 預設
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
param_cols = [
    'linlen', 'factor', 'smaalen', 'devwin', 'short_win', 'long_win',
    'rma_len', 'dev_len',
    'buy_mult', 'sell_mult',
    'prom_factor', 'min_dist',
    'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days',
    'stop_loss', 'trade_cooldown_bars'
]
# 設定 matplotlib 字體
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def clean_filename(s: str) -> str:
    """清理檔案名稱，移除 Windows 檔案系統中的非法字元"""
    s = re.sub(r'[^\w\s.-]', '_', str(s))
    s = re.sub(r'\s+', '_', s.strip())
    return s
def compute_b_h_metrics(ticker, start_date=cfg.START_DATE, end_date='2025-05-15'):
    """計算 Buy-and-Hold 的總報酬與 Sharpe Ratio"""
    df_price, df_factor = data_loader.load_data(ticker, start_date, end_date)  # 改用 data_loader
    df = df_price  # 直接使用 df_price
    # 檢查日期欄位或索引
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = [col for col in df.columns if col.lower() in ['date', 'datetime', 'index']]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.set_index(date_col)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except (ValueError, TypeError):
                raise ValueError(f"No datetime index or date column found in data for {ticker}")
    # 轉換日期並過濾
    start = pd.to_datetime(start_date)
    df = df[df.index >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        df = df[df.index <= end]
    if 'close' not in df.columns:
        raise ValueError(f"'close' column not found in data for {ticker}")
    returns = df['close'].pct_change().dropna()
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0
    return {'total_return': total_return, 'sharpe_ratio': sharpe}

def load_data():
    """讀取 walk-forward, grid, stress 資料"""
    wf_csv = RESULT_DIR / f"wf_grid_ALL_{ticker.replace('^','')}.csv"
    grid_csv = RESULT_DIR / f"grid_ALL_{ticker.replace('^','')}.csv"
    stress_csv = RESULT_DIR / f"stress_test_results_{ticker.replace('^','')}.csv"
    wf_df = pd.read_csv(wf_csv) if wf_csv.exists() else pd.DataFrame()
    grid_df = pd.read_csv(grid_csv) if grid_csv.exists() else pd.DataFrame()
    stress_df = pd.read_csv(stress_csv) if stress_csv.exists() else pd.DataFrame()
    if not wf_df.empty:
        print(f"wf_df columns: {list(wf_df.columns)}")
        print(f"wf_df sample:\n{wf_df.head(3)}")
    if not stress_df.empty:
        print(f"stress_df columns: {list(stress_df.columns)}")
        print(f"stress_df sample:\n{stress_df.head(3)}")
    # 確保 data_source 欄位存在
    if 'data_source' not in wf_df.columns:
        wf_df['data_source'] = 'Self'
    else:
        wf_df['data_source'] = wf_df['data_source'].apply(clean_filename)
    if 'data_source' not in stress_df.columns:
        stress_df['data_source'] = 'Self'
    else:
        stress_df['data_source'] = stress_df['data_source'].apply(clean_filename)
    if 'ROI' in wf_df.columns:
        wf_df['ROI'] = wf_df['ROI'].astype(float) / 100
    if 'ROI_wf' in wf_df.columns:
        wf_df['ROI_wf'] = wf_df['ROI_wf'].astype(float) / 100
    # 同步 stress_df 的參數欄位


    for col in param_cols:
        if col in wf_df.columns and col not in stress_df.columns:
            stress_df[col] = pd.NA
    return wf_df, grid_df, stress_df

def plot_and_stats(df, x_col, y_col, title, groupby=None, filename=None):
    """繪製散點圖並計算相關性與異常點（來自 ROEAv1.py）"""
    if len(df) < 2:
        print(f"⚠ Not enough points to plot: {title}")
        return None, None
    
    # 確保 PLOT_DIR 存在
    try:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        if not PLOT_DIR.exists():
            raise FileNotFoundError(f"PLOT_DIR {PLOT_DIR} does not exist")
    except Exception as e:
        print(f"⚠ Failed to create PLOT_DIR {PLOT_DIR}: {e}")
        return None, None
    
    # 繪圖邏輯
    if groupby:
        stats = []
        for g in df[groupby].unique():
            sub = df[df[groupby] == g]
            if len(sub) < 2 or sub[x_col].nunique() == 1 or sub[y_col].nunique() == 1:
                print(f"⚠ Skipping correlation for {title} ({g}): insufficient unique data points")
                continue
            x, y = sub[x_col].astype(float), sub[y_col].astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                corr, p = pearsonr(x, y)
            thr_x = x.quantile(0.70)
            thr_y = y.quantile(0.50)
            outliers = (x >= thr_x) & (y <= thr_y)
            pct = outliers.sum() / len(x) * 100
            plt.figure(figsize=(6, 5))
            plt.scatter(x, y, alpha=0.4, label="All")
            plt.scatter(x[outliers], y[outliers], edgecolors='black', label=f"Outliers ({pct:.1f}%)")
            plt.axvline(B_H_RETURN, ls='--', color='red', label='Buy & Hold')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{title} ({g})\nPearson r={corr:.3f}, p={p:.3g}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            safe_g = clean_filename(g)
            fname = PLOT_DIR / f"roea_scatter_{ticker.replace('^','')}_{safe_g}_{TIMESTAMP}.png" if not filename else filename
            try:
                print(f"Attempting to save plot to {fname}")
                plt.savefig(fname, dpi=300)
                print(f"Plot saved successfully to {fname}")
            except Exception as e:
                print(f"⚠ Failed to save plot {fname}: {e}")
            plt.close()
            stats.append({'group': g, 'pearson_r': corr, 'outlier_pct': pct})
        return stats
    else:
        # 非分組繪圖邏輯
        x, y = df[x_col].astype(float), df[y_col].astype(float)
        if len(x) < 2 or x.nunique() == 1 or y.nunique() == 1:
            print(f"⚠ Skipping correlation for {title}: insufficient unique data points")
            return None, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            corr, p = pearsonr(x, y)
        thr_x = x.quantile(0.70)
        thr_y = y.quantile(0.50)
        outliers = (x >= thr_x) & (y <= thr_y)
        pct = outliers.sum() / len(x) * 100
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, alpha=0.4, label="All")
        plt.scatter(x[outliers], y[outliers], edgecolors='black', label=f"Outliers ({pct:.1f}%)")
        plt.axvline(B_H_RETURN, ls='--', color='red', label='Buy & Hold')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{title}\nPearson r={corr:.3f}, p={p:.3g}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        fname = PLOT_DIR / f"roea_scatter_{ticker.replace('^','')}_{TIMESTAMP}.png" if not filename else filename
        try:
            print(f"Attempting to save plot to {fname}")
            plt.savefig(fname, dpi=300)
            print(f"Plot saved successfully to {fname}")
        except Exception as e:
            print(f"⚠ Failed to save plot {fname}: {e}")
        plt.close()
        return corr, pct
def plot_scatter(summary_df, wf_df, stress_df):
    """正常期 vs 壓力期散點圖"""
    plt.figure(figsize=(6, 5))
    plt.scatter(summary_df['normal_total_return_mean'], summary_df['stress_total_return_mean'], alpha=0.6)
    plt.axvline(B_H_RETURN, ls='--', color='red', label='Buy & Hold')
    plt.axhline(B_H_RETURN, ls='--', color='red')
    delta = 3.0
    x = np.linspace(summary_df['normal_total_return_mean'].min(), summary_df['normal_total_return_mean'].max(), 100)
    plt.fill_between(x, x - delta, x + delta, alpha=0.2, color='green', label=f'Stable Region (±{delta})')
    plt.xlabel('Normal Mean Return')
    plt.ylabel('Stress Mean Return')
    plt.title(f'{ticker} Normal vs Stress Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"roea_scatter_normal_stress_{ticker.replace('^','')}_{TIMESTAMP}.png", dpi=300)
    plt.close()

def plot_boxplot(stress_df):
    """壓力期報酬箱型圖"""
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='strategy', y='total_return', data=stress_df)
    plt.axhline(B_H_RETURN, ls='--', color='red', label='Buy & Hold')
    plt.title(f'{ticker} Stress Period Returns by Strategy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"roea_boxplot_{ticker.replace('^','')}_{TIMESTAMP}.png", dpi=300)
    plt.close()

def plot_heatmap(stress_df):
    """壓力期報酬熱圖"""
    pivot = stress_df.pivot_table(index='strategy', columns='period', values='total_return')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
    plt.title(f'{ticker} Stress Returns Heatmap')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"roea_heatmap_{ticker.replace('^','')}_{TIMESTAMP}.png", dpi=300)
    plt.close()

def plot_sortino_heatmap(stress_df):
    """Sortino Ratio 熱圖"""
    pivot = stress_df.pivot_table(index='strategy', columns='period', values='sortino_ratio')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='magma')
    plt.title(f'{ticker} Sortino Ratio Stability Across Stress Periods')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"roea_sortino_heatmap_{ticker.replace('^','')}_{TIMESTAMP}.png", dpi=300)
    plt.close()

def plot_param_heatmap(df, x_param, y_param, score_col='robustness_score'):
    """參數熱圖"""
    pivot = df.pivot_table(index=y_param, columns=x_param, values=score_col, aggfunc='mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
    plt.title(f'{ticker} {score_col} by {x_param} vs {y_param}')
    plt.savefig(PLOT_DIR / f"roea_param_heatmap_{ticker.replace('^','')}_{x_param}_{y_param}_{TIMESTAMP}.png", dpi=300)
    plt.close()

def plot_roi_distribution(df, col='total_return'):
    """ROI 分佈圖"""
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x=col, hue='strategy', fill=True, alpha=0.3)
    plt.axvline(B_H_RETURN, ls='--', color='red', label='Buy & Hold')
    plt.title(f'{ticker} ROI Distribution by Strategy')
    plt.legend()
    plt.savefig(PLOT_DIR / f"roea_distribution_{ticker.replace('^','')}_{col}_{TIMESTAMP}.png", dpi=300)
    plt.close()

def compute_knn_stability(df, params=['linlen', 'smaalen', 'buy_mult'], k=5, metric='total_return'):
    """計算 K-NN 穩定度"""
    param_cols = [c for c in params if c in df.columns]
    if not param_cols or len(df) < k + 1:
        print(f"⚠ Skipping K-NN stability: no valid parameters ({param_cols}) or too few rows ({len(df)})")
        return [0] * len(df)
    X = df[param_cols].fillna(0).values
    k = min(k, len(df) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    stability = []
    for i, idx in enumerate(indices):
        roi = df.iloc[i][metric]
        roi_neighbors = df.iloc[idx[1:]][metric].mean()
        stability.append(abs(roi - roi_neighbors))
    return stability

def filter_candidates(df, b_h_return=B_H_RETURN, delta=3.0, min_sharpe=0.2):
    """三層漏斗過濾"""
    mask_pos = df['normal_total_return_mean'] > 0
    df = df[mask_pos].copy()
    print(f"Layer 1: {len(df)} candidates with positive normal return")
    mask_stable = (df['normal_total_return_mean'] - df['stress_total_return_mean']).abs() < delta
    df = df[mask_stable]
    print(f"Layer 2: {len(df)} candidates with stable returns (delta={delta})")
    mask_alpha = (df['normal_total_return_mean'] > b_h_return * 0.8) | (df['normal_sharpe_ratio_mean'] > min_sharpe)
    df = df[mask_alpha]
    print(f"Layer 3: {len(df)} candidates beating Buy-and-Hold or high Sharpe")
    return df

def compute_score(df, weights={'norm_return': 0.5, 'stress_return': 0.4, 'stress_std': 0.1, 'sharpe': 0.1}):
    """計算綜合得分"""
    norm_return = (df['normal_total_return_mean'] - df['normal_total_return_mean'].min()) / \
                  (df['normal_total_return_mean'].max() - df['normal_total_return_mean'].min() + 1e-6)
    stress_return = (df['stress_total_return_mean'] - df['stress_total_return_mean'].min()) / \
                    (df['stress_total_return_mean'].max() - df['stress_total_return_mean'].min() + 1e-6)
    stress_std = (df['stress_total_return_std'] - df['stress_total_return_std'].min()) / \
                 (df['stress_total_return_std'].max() - df['stress_total_return_std'].min() + 1e-6)
    sharpe = (df['normal_sharpe_ratio_mean'] - df['normal_sharpe_ratio_mean'].min()) / \
             (df['normal_sharpe_ratio_mean'].max() - df['normal_sharpe_ratio_mean'].min() + 1e-6)
    df['score'] = (weights['norm_return'] * norm_return + 
                   weights['stress_return'] * stress_return - 
                   weights['stress_std'] * stress_std + 
                   weights['sharpe'] * sharpe)
    return df

def compute_pareto(df, x_col='normal_total_return_mean', y_col='stress_total_return_mean'):
    """計算 Pareto 前緣"""
    pareto = []
    for idx, row in df.iterrows():
        dominated = ((df[x_col] >= row[x_col]) & 
                     (df[y_col] >= row[y_col]) & 
                     ((df[x_col] > row[x_col]) | (df[y_col] > row[y_col]))).any()
        if not dominated:
            pareto.append(idx)
    return df.loc[pareto]

def detect_overfit(df, params=['linlen', 'smaalen', 'buy_mult'], k=5, threshold=0.95):
    """檢測過擬合參數組"""
    df['knn_stability'] = compute_knn_stability(df, params, k)
    stability_threshold = df['knn_stability'].quantile(threshold) if df['knn_stability'].std() > 0 else 0
    return df[df['knn_stability'] <= stability_threshold]

def analyze_roi_v1(wf_df):
    """ROEAv1.py 的 ROI_wf vs ROI_total 分析"""
    if 'ROI_wf' not in wf_df.columns or 'ROI' not in wf_df.columns:
        print("⚠ Missing ROI_wf or ROI columns for ROEAv1 analysis. Skipping.")
        return pd.DataFrame(), []
    group_cols = [
        'strategy', 'data_source', 'linlen', 'smaalen', 'devwin',
        'buy_mult', 'sell_mult', 'rma_len', 'dev_len', 'prom_factor',
        'min_dist', 'signal_cooldown_days', 'buy_shift', 'exit_shift', 'vol_window'
    ]
    group_cols = [c for c in group_cols if c in wf_df.columns]
    wf_df = wf_df.copy()
    wf_df[group_cols] = wf_df[group_cols].astype(str)
    if 'period' not in wf_df.columns:
        print("⚠ Missing 'period' column. Treating data as single period.")
        wf_df['period'] = 'single'
    param_counts = wf_df.groupby(group_cols)['period'].nunique().reset_index(name='period_count')
    max_periods = wf_df['period'].nunique()
    if max_periods > 1:
        full_params = param_counts[param_counts['period_count'] == max_periods][group_cols]
        print(f"Found {len(full_params)} parameter groups in all {max_periods} periods")
    else:
        full_params = param_counts[group_cols]
        print(f"Single period detected. Using all {len(full_params)} parameter groups")
    df_full = wf_df.merge(full_params, on=group_cols, how='inner')
    agg_full = df_full.groupby(group_cols).agg(
        ROI_total=('ROI', lambda x: prod(1 + x) - 1),
        ROI_mean=('ROI_wf', 'mean')
    ).reset_index()
    if max_periods == 1:
        print("⚠ Single period detected. Skipping ROI_total vs ROI_mean plot to avoid misleading linear relationship.")
        corr, pct = None, None
    else:
        corr, pct = plot_and_stats(
            agg_full, 'ROI_mean', 'ROI_total',
            'Total ROI vs Mean ROI (Full Periods)',
            filename=PLOT_DIR / f"roea_v1_scatter_full_{ticker.replace('^','')}_{TIMESTAMP}.png"
        )
    group_keys = ['strategy']
    if 'data_source' in agg_full.columns:
        group_keys.append('data_source')
    stats_list = []
    for keys, group in agg_full.groupby(group_keys):
        if len(group) < 2:
            continue
        strat = group['strategy'].iloc[0]
        src = group['data_source'].iloc[0] if 'data_source' in group.columns else 'N/A'
        if max_periods == 1:
            print(f"⚠ Single period detected. Skipping ROI_total vs ROI_mean plot for {strat} ({src}).")
            stats = None
        else:
            safe_strat = clean_filename(strat)
            safe_src = clean_filename(src)
            stats = plot_and_stats(
                group, 'ROI_mean', 'ROI_total',
                f'{strat} ({src}): Total ROI vs Mean ROI',
                filename=PLOT_DIR / f"roea_v1_scatter_{ticker.replace('^','')}_{safe_strat}_{safe_src}_{TIMESTAMP}.png"
            )
        if stats:
            stats_list.append({
                'strategy': strat,
                'data_source': src,
                'pearson_r': stats[0],
                'outlier_pct': stats[1]
            })
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(RESULT_DIR / f"roea_v1_stats_{ticker.replace('^','')}_{TIMESTAMP}.csv", index=False)
    return agg_full, stats_list

def analyze_performance(wf_df, grid_df, stress_df):
    """性能分析與參數挑選"""
    summary_records = []
    all_metric_cols = list(calculate_metrics([], pd.DataFrame()).keys())
    if not wf_df.empty and len(wf_df) >= 2 and 'total_return' in wf_df.columns:
        agg_full, v1_stats = analyze_roi_v1(wf_df)
        if not agg_full.empty:
            print(f"📊 ROEAv1 analysis completed: {len(agg_full)} parameter groups")
    else:
        print("⚠ Skipping ROEAv1 analysis: insufficient data or missing total_return")
        agg_full, v1_stats = pd.DataFrame(), []
    if not wf_df.empty:
        metric_cols = [c for c in wf_df.columns if c in all_metric_cols]
        group_cols = ['strategy'] + [c for c in ['linlen', 'smaalen', 'buy_mult', 'sell_mult', 'rma_len', 'dev_len', 'prom_factor', 'buy_shift', 'exit_shift'] if c in wf_df.columns and c in stress_df.columns]
        agg_dict = {m: ['mean', 'std', 'min', 'max'] for m in metric_cols}
        wf_agg = wf_df.groupby(group_cols).agg(agg_dict).reset_index()
        wf_agg.columns = group_cols + [f'normal_{m}_{s}' for m in metric_cols for s in ['mean', 'std', 'min', 'max']]
        print('📊 Normal period analysis completed')
    else:
        wf_agg = pd.DataFrame()
        print('⚠ No walk-forward data for normal period analysis')
    if not stress_df.empty:
        metric_cols = [c for c in stress_df.columns if c in all_metric_cols]
        stress_overall = stress_df.groupby(group_cols)[metric_cols].agg(['mean', 'std']).reset_index()
        stress_overall.columns = group_cols + [f'stress_{m}_{s}' for m in metric_cols for s in ['mean', 'std']]
        print('📊 Stress period analysis completed')
    else:
        stress_overall = pd.DataFrame()
        print('⚠ No stress data for stress period analysis')
    if not wf_agg.empty and not stress_overall.empty:
        summary = pd.merge(wf_agg, stress_overall, on=group_cols, how='outer')
        summary['return_stability'] = summary['normal_total_return_mean'] - summary['stress_total_return_mean']
        summary['robustness_score'] = summary['normal_sharpe_ratio_mean'] / (summary['stress_total_return_std'] + 1e-6)
        if not agg_full.empty:
            summary = summary.merge(agg_full[['strategy', 'ROI_mean', 'ROI_total']], on='strategy', how='left')
        summary_records.append(summary)
    elif not wf_agg.empty:
        summary_records.append(wf_agg)
    elif not stress_overall.empty:
        summary_records.append(stress_overall)
    if summary_records:
        summary_df = pd.concat(summary_records, ignore_index=True)
        summary_file = RESULT_DIR / f"roea_summary_{ticker.replace('^','')}_{TIMESTAMP}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f'📦 Summary saved to {summary_file}')
    else:
        summary_df = pd.DataFrame()
    if not summary_df.empty:
        filtered_df = filter_candidates(summary_df)
        filtered_df = compute_score(filtered_df)
        pareto_df = compute_pareto(filtered_df)
        stable_df = detect_overfit(pareto_df)
        delta = 3.0
        stable_df = stable_df[(stable_df['normal_total_return_mean'] - stable_df['stress_total_return_mean']).abs() < delta]
        stable_df.to_csv(RESULT_DIR / f"roea_selected_{ticker.replace('^','')}_{TIMESTAMP}.csv", index=False)
        print(f'📦 Selected {len(stable_df)} parameter sets saved')
        plot_scatter(summary_df, wf_df, stress_df)
        plot_boxplot(stress_df)
        plot_heatmap(stress_df)
        plot_sortino_heatmap(stress_df)
        plot_roi_distribution(wf_df, 'total_return')
        param_pairs = [('buy_mult', 'sell_mult'), ('linlen', 'smaalen')]
        for x, y in param_pairs:
            if x in wf_df.columns and y in wf_df.columns:
                plot_param_heatmap(wf_df, x, y)
        if 'strategy' in wf_df.columns and wf_df['strategy'].str.contains('ssma_turn').any():
            wf_df['offsets_str'] = wf_df['offsets'].apply(str)
            plot_param_heatmap(wf_df[wf_df['strategy'] == 'ssma_turn'], 'offsets_str', 'prom_factor')
        plt.figure(figsize=(6, 5))
        plt.scatter(summary_df['normal_total_return_mean'], summary_df['stress_total_return_mean'], alpha=0.3, label='All')
        plt.scatter(stable_df['normal_total_return_mean'], stable_df['stress_total_return_mean'], color='red', label='Selected')
        plt.axvline(B_H_RETURN, ls='--', color='blue', label='Buy & Hold')
        plt.axhline(B_H_RETURN, ls='--', color='blue')
        x = np.linspace(summary_df['normal_total_return_mean'].min(), summary_df['normal_total_return_mean'].max(), 100)
        plt.fill_between(x, x - delta, x + delta, alpha=0.2, color='green', label=f'Stable Region (±{delta})')
        plt.xlabel('Normal Mean Return')
        plt.ylabel('Stress Mean Return')
        plt.title(f'{ticker} Parameter Selection')
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOT_DIR / f"roea_selection_{ticker.replace('^','')}_{TIMESTAMP}.png", dpi=300)
        plt.close()
    if not wf_df.empty:
        plot_and_stats(wf_df, 'total_return', 'sharpe_ratio', 'Total Return vs Sharpe Ratio', groupby='strategy')
    if not stress_df.empty:
        for p in stress_df['period'].unique():
            sub = stress_df[stress_df['period'] == p]
            plot_and_stats(sub, 'total_return', 'sharpe_ratio', f'Total Return vs Sharpe Ratio ({p})')
    if not summary_df.empty:
        best = summary_df.loc[summary_df['score'].idxmax()]
        print(f'🏆 Recommended Strategy: {best["strategy"]}')
        print(f'   Normal Return (Mean): {best["normal_total_return_mean"]:.2%}')
        print(f'   Stress Return (Mean): {best["stress_total_return_mean"]:.2%}')
        print(f'   Robustness Score: {best["robustness_score"]:.2f}')
        if 'ROI_mean' in best and 'ROI_total' in best:
            print(f'   ROI Mean (ROEAv1): {best["ROI_mean"]:.2%}')
            print(f'   ROI Total (ROEAv1): {best["ROI_total"]:.2%}')
    return summary_df

def main():
    logging.basicConfig(
        level=logging.INFO,
        filename=RESULT_DIR / f'roea_v4_{TIMESTAMP}.log',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting ROEAv4 analysis")
    
    try:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        if not PLOT_DIR.exists():
            raise FileNotFoundError(f"PLOT_DIR {PLOT_DIR} could not be created")
        logging.info(f"PLOT_DIR ensured: {PLOT_DIR}")
        logging.info(f"RESULT_DIR ensured: {RESULT_DIR}")
    except Exception as e:
        logging.error(f"Failed to create PLOT_DIR or RESULT_DIR: {e}")
        print(f"⚠ Failed to create PLOT_DIR or RESULT_DIR: {e}")
        sys.exit(1)
    
    wf_df, grid_df, stress_df = load_data()
    if 'data_source' not in wf_df.columns:
        wf_df['data_source'] = 'Self'
    if 'period' not in wf_df.columns:
        wf_df['period'] = 'single'
    if 'total_return' in stress_df:
        stress_df.query("total_return >= -0.05", inplace=True)
    analyze_performance(wf_df, grid_df, stress_df)

if __name__ == '__main__':
    b_h_metrics = compute_b_h_metrics(ticker)
    B_H_RETURN = b_h_metrics['total_return']
    B_H_SHARPE = b_h_metrics['sharpe_ratio']
    main()
