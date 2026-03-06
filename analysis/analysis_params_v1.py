'''
注意此版本的輸出沒有csv,只有圖
之後要調整
'''
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非互動式繪圖後端
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import shap
from itertools import combinations
import logging
import msvcrt
from joblib import Parallel, delayed, Memory
import pyarrow.feather as feather
import re

# 將專案根目錄加入模組搜尋路徑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis import config as cfg

# 設定 matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 載入配置
DATA_DIR = cfg.DATA_DIR
TICKER = cfg.TICKER
RESULT_DIR = cfg.RESULT_DIR
LOG_DIR = cfg.DATA_DIR
PLOT_DIR = cfg.PLOT_DIR



# 設置日誌
log_file = PLOT_DIR / "analysis_params.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 定義快取
memory = Memory(cfg.CACHE_DIR / "joblib", verbose=0)

# 定義策略參數
ind_keys = cfg.STRATEGY_PARAMS

# 定義欄位
perf_metrics = ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate']
optional_metrics = ['roi_wf']
param_cols = [
    'linlen', 'smaalen', 'devwin', 'factor', 'buy_mult', 'sell_mult',
    'stop_loss', 'prom_factor', 'min_dist', 'rma_len', 'dev_len',
    'buy_shift', 'exit_shift', 'vol_window', 'quantile_win',
    'signal_cooldown_days', 'short_win', 'long_win', 'trade_cooldown_bars'
]

# 載入資料（使用 Feather 快取）
feather_file = DATA_DIR / "grid_ALL_00631L.TW.feather"
if feather_file.exists():
    df = feather.read_feather(feather_file)
    logging.info("從 Feather 檔案載入資料")
else:
    file_path = RESULT_DIR / "grid_ALL_00631L.TW.csv"
    raw_df = pd.read_csv(file_path, engine='python')
    cols_keep = list(set(param_cols + perf_metrics + optional_metrics + ['strategy', 'data_source']) & set(raw_df.columns))
    df = raw_df[cols_keep].copy()
    feather.write_feather(df.reset_index(drop=True), feather_file)
    logging.info("已將資料儲存為 Feather 檔案")

# 修正勝率（若需要）
if 'win_rate' in df.columns:
    df['win_rate'] = df['win_rate'] * 2
    logging.info("已將 win_rate 乘以 2 進行修正")

df = df.dropna(subset=perf_metrics, how='any')

# Debug 檢查
logging.info(f"讀檔後 df.shape = {df.shape}")
logging.info(f"欄位清單：{df.columns.tolist()[:20]}...")
logging.info(f"欄位清單： {df.columns.tolist()[:20]} ...")
print("前幾行：\n", df.head())
logging.info(f"前幾行資料：\n{df.head()}")

if df.empty:
    raise ValueError("資料為空，請檢查 CSV 內容或欄位篩選條件")

# 檢查常數欄位
constant_cols = [c for c in param_cols if c in df.columns and df[c].nunique() == 1]
print("常數參數：", constant_cols)
logging.info(f"常數參數： {constant_cols}")

# 清理檔案名稱的輔助函數
def sanitize_filename(filename):
    # 移除或替換無效字元
    filename = str(filename)
    # 替換反斜線、斜線、括號等
    filename = re.sub(r'[\\/*?:"<>|()]', '_', filename)
    # 移除多餘的空格和特殊字元
    filename = re.sub(r'\s+', '_', filename.strip())
    return filename

# 快取相關性計算
@memory.cache
def compute_correlation(X_full, y, strat_params_full):
    # Filter out columns with zero variance or all NaN
    valid_cols = [i for i, col in enumerate(X_full.T) if np.std(col) > 0 and not np.all(np.isnan(col))]
    if not valid_cols:
        logging.warning(f"No valid columns for correlation with {strat_params_full}")
        return pd.Series(np.zeros(len(strat_params_full)), index=strat_params_full)
    
    X_valid = X_full[:, valid_cols]
    valid_params = [strat_params_full[i] for i in valid_cols]
    
    # Compute correlations, handling NaN results
    try:
        corr = np.corrcoef(X_valid.T, y)[0:-1, -1]
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
        return pd.Series(corr, index=valid_params).abs().reindex(strat_params_full, fill_value=0.0).sort_values(ascending=False)
    except Exception as e:
        logging.error(f"Correlation computation failed: {e}")
        return pd.Series(np.zeros(len(strat_params_full)), index=strat_params_full)

# 定義分析函數
def analyze_group(grp, group_name, strat_params, perf_metrics, plot_dir):
    # 過濾常數欄位
    strat_params = [p for p in strat_params if p in grp.columns and p not in constant_cols]
    strat_params_full = [p for p in param_cols if p in grp.columns and p not in constant_cols]

    # 檢查參數有效性
    valid_params = [p for p in strat_params_full if not grp[p].isna().all() and grp[p].nunique() > 1]
    if not valid_params:
        logging.warning(f"No valid parameters for group {group_name}, skipping correlation analysis")
        return

    # 1. 相關性分析
    imputer = SimpleImputer(strategy='median')
    try:
        X_full = imputer.fit_transform(grp[valid_params])
    except Exception as e:
        logging.error(f"Imputation failed for group {group_name}: {e}")
        return

    corr_dict = {}
    for metric in perf_metrics:
        corr_df = grp[strat_params].corrwith(grp[metric]).to_frame(f'corr_{metric}')
        corr_df[f'abs_corr_{metric}'] = corr_df[f'corr_{metric}'].abs()
        corr_df = corr_df.sort_values(by=f'abs_corr_{metric}', ascending=False)
        print(f'\n=== {group_name} 與 {metric} 的相關性分析 ===')
        logging.info(f'=== {group_name} 與 {metric} 的相關性分析 ===')
        print(corr_df)
        logging.info(f'\n{corr_df}')

        # 使用快取計算全參數相關性
        corr_all = compute_correlation(X_full, grp[metric], valid_params)
        print(f'\n=== {group_name} 全參數相關度 (填補 NaN 後) ===')
        logging.info(f'=== {group_name} 全參數相關度 (填補 NaN 後) ===')
        print(corr_all.head(10))
        logging.info(f'\n{corr_all.head(10)}')

        # 繪製熱圖
        def plot_heatmap(corr_df, metric, group_name, plot_dir):
            try:
                plt.figure(figsize=(10, 6))
                sns.heatmap(corr_df[[f'corr_{metric}']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Correlation with {metric} ({group_name})')
                plt.tight_layout()
                sanitized_group_name = sanitize_filename(group_name)
                plt.savefig(plot_dir / f'corr_{metric}_{sanitized_group_name}.png')
                plt.close()
                logging.info(f"成功生成熱圖：corr_{metric}_{sanitized_group_name}.png")
            except Exception as e:
                logging.error(f"生成熱圖失敗 (metric={metric}, group={group_name})：{e}")

        Parallel(n_jobs=cfg.N_JOBS, backend="loky")(
            delayed(plot_heatmap)(corr_df, metric, group_name, plot_dir) for _ in [1]
        )

    # 2. 分組檢視（Binning Analysis）
    def binning_analysis(param, metric, bins=10):
        try:
            grp[f'{param}_bin'] = pd.qcut(grp[param], q=bins, duplicates='drop')
            grouped = grp.groupby(f'{param}_bin', observed=True)[metric].mean().reset_index()
            print(f'\n=== {group_name}: {param} 分組 ({metric}) 平均值 ===')
            logging.info(f'=== {group_name}: {param} 分組 ({metric}) 平均值 ===')
            print(grouped)
            logging.info(f'\n{grouped}')
            return grouped[f'{param}_bin'].astype(str), grouped[metric]
        except ValueError as e:
            print(f"無法對 {param} 進行分組分析，錯誤：{e}")
            logging.info(f"無法對 {param} 進行分組分析，錯誤：{e}")
            return None, None

    # 3. 子圖繪製（散點圖與分組分析）
    top_params = corr_df.index[:3]
    metrics = perf_metrics[:3]
    fig, axes = plt.subplots(len(metrics), len(top_params), figsize=(4*len(top_params), 3*len(metrics)), sharex='col')
    axes = np.array(axes).reshape(len(metrics), len(top_params))
    for i, metric in enumerate(metrics):
        for j, param in enumerate(top_params):
            ax = axes[i, j]
            ax.scatter(grp[param], grp[metric], s=5, alpha=0.6)
            if i == 0:
                ax.set_title(param)
            if j == 0:
                ax.set_ylabel(metric)
    fig.suptitle(f'{group_name} — Top-params vs Metrics', y=1.02)
    fig.tight_layout()
    try:
        sanitized_group_name = sanitize_filename(group_name)
        fig.savefig(plot_dir / f'{sanitized_group_name}_scatter_grid.png', dpi=150)
        logging.info(f"成功儲存散點圖：{sanitized_group_name}_scatter_grid.png")
    except Exception as e:
        logging.error(f"儲存散點圖失敗 (group={group_name})：{e}")
    finally:
        plt.close(fig)

    # 分組分析子圖
    fig, axes = plt.subplots(len(metrics), len(top_params), figsize=(4*len(top_params), 3*len(metrics)), sharex='col')
    axes = np.array(axes).reshape(len(metrics), len(top_params))
    for i, metric in enumerate(metrics):
        for j, param in enumerate(top_params):
            ax = axes[i, j]
            labels, values = binning_analysis(param, metric)
            if labels is not None and values is not None:
                ax.plot(labels, values, marker='o')
                ax.tick_params(axis='x', rotation=45)
            if i == 0:
                ax.set_title(param)
            if j == 0:
                ax.set_ylabel(metric)
    fig.suptitle(f'{group_name} — Mean Metrics by Top-params Bins', y=1.02)
    fig.tight_layout()
    try:
        sanitized_group_name = sanitize_filename(group_name)
        fig.savefig(plot_dir / f'{sanitized_group_name}_binning_grid.png', dpi=150)
        logging.info(f"成功儲存分組分析圖：{sanitized_group_name}_binning_grid.png")
    except Exception as e:
        logging.error(f"儲存分組分析圖失敗 (group={group_name})：{e}")
    finally:
        plt.close(fig)

    # 所有參數散點圖（減少繪製）
    if len(valid_params) <= 8:
        n = len(valid_params)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        for k, param in enumerate(valid_params):
            ax = axes.flatten()[k]
            ax.scatter(grp[param], grp['total_return'], s=4, alpha=0.5)
            ax.set_title(param)
            ax.set_xlabel(param)
            ax.set_ylabel('total_return')
        fig.tight_layout()
        try:
            sanitized_group_name = sanitize_filename(group_name)
            fig.savefig(plot_dir / f'{sanitized_group_name}_allparams_scatter.png', dpi=150)
            logging.info(f"成功儲存全參數散點圖：{sanitized_group_name}_allparams_scatter.png")
        except Exception as e:
            logging.error(f"儲存全參數散點圖失敗 (group={group_name})：{e}")
        finally:
            plt.close(fig)

    # 4. 多元迴歸與隨機森林
    imputer = SimpleImputer(strategy='median')
    for metric in perf_metrics:
        print(f'\n=== {group_name}: {metric} 的模型分析 ===')
        logging.info(f'=== {group_name}: {metric} 的模型分析 ===')
        X = grp[valid_params]
        y = grp[metric]
        X_imp = imputer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)

        # 線性迴歸
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        r2 = r2_score(y_test, lr.predict(X_test))
        print(f'Linear Regression R² for {metric}: {r2:.4f}')
        logging.info(f'Linear Regression R² for {metric}: {r2:.4f}')

        # 隨機森林
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = pd.Series(rf.feature_importances_, index=valid_params).sort_values(ascending=False)
        print(f'\nFeature Importances for {metric}:')
        logging.info(f'Feature Importances for {metric}:')
        print(importances.to_frame('importance'))
        logging.info(f'\n{importances.to_frame("importance")}')

        # SHAP 值分析
        sample_size = min(1000, X_imp.shape[0])
        X_sample = X_imp[np.random.choice(X_imp.shape[0], sample_size, replace=False)]
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, features=X_sample, feature_names=valid_params, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance for {metric} ({group_name})')
        plt.tight_layout()
        try:
            sanitized_group_name = sanitize_filename(group_name)
            plt.savefig(plot_dir / f'shap_{metric}_{sanitized_group_name}.png')
            logging.info(f"成功儲存 SHAP 圖：shap_{metric}_{sanitized_group_name}.png")
        except Exception as e:
            logging.error(f"儲存 SHAP 圖失敗 (metric={metric}, group={group_name})：{e}")
        finally:
            plt.close()

    # 5. 參數交互作用分析
    def interaction_analysis(param1, param2, metric):
        grp['interaction'] = grp[param1] * grp[param2]
        corr = grp['interaction'].corr(grp[metric])
        print(f'\n{group_name} 交互作用: {param1} x {param2} 對 {metric} 的相關性: {corr:.4f}')
        logging.info(f'{group_name} 交互作用: {param1} x {param2} 對 {metric} 的相關性: {corr:.4f}')

    for metric in perf_metrics[:2]:
        param_pairs = list(combinations(top_params[:2], 2))
        for param1, param2 in param_pairs:
            interaction_analysis(param1, param2, metric)

    # 6. 參數無效性判斷
    @memory.cache
    def check_parameter_effectiveness_cached(param, grp, metric, perf_metrics):
        low_impact = True
        for m in perf_metrics:
            corr = abs(grp[param].corr(grp[m]))
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(grp[[param]].fillna(grp[param].median()), grp[m])
            importance = rf.feature_importances_[0]
            if corr > 0.1 or importance > 0.05:
                low_impact = False
                break
        return low_impact

    for metric in perf_metrics:
        print(f'\n=== {group_name}: {metric} 的參數無效性分析 ===')
        logging.info(f'=== {group_name}: {metric} 的參數無效性分析 ===')
        low_impact_params = []
        for param in valid_params:
            if check_parameter_effectiveness_cached(param, grp, metric, perf_metrics):
                low_impact_params.append(param)
        if low_impact_params:
            print(f'以下參數對 {metric} 及其它指標影響可能較低，可能無效：{low_impact_params}')
            logging.info(f'以下參數對 {metric} 及其它指標影響可能較低，可能無效：{low_impact_params}')
        else:
            print(f'無明顯無效參數，所有參數對 {metric} 或其他指標有一定影響。')
            logging.info(f'無明顯無效參數，所有參數對 {metric} 或其他指標有一定影響。')

    # 7. 同質結果分析
    def check_identifiability(metric, threshold=0.01):
        print(f'\n=== {group_name}: {metric} 的同質結果分析 ===')
        logging.info(f'=== {group_name}: {metric} 的同質結果分析 ===')
        grouped = grp.groupby(perf_metrics)[valid_params].nunique().reset_index()
        grouped = grouped.sort_values(metric)
        diffs = grouped[metric].diff().abs()
        similar_mask = diffs < threshold
        if similar_mask.any():
            print(f'以下列在 {metric} 差異 < {threshold}：')
            logging.info(f'以下列在 {metric} 差異 < {threshold}：')
            print(grouped.loc[similar_mask, perf_metrics + valid_params])
            logging.info(f'\n{grouped.loc[similar_mask, perf_metrics + valid_params]}')
        else:
            print('未發現同質結果。')
            logging.info('未發現同質結果。')

    for metric in perf_metrics:
        check_identifiability(metric)

    # 8. 多指標綜合圖表
    for param in top_params[:2]:
        chart_data = {
            'type': 'line',
            'data': {
                'labels': grp[param].sort_values().unique().astype(str).tolist(),
                'datasets': [
                    {
                        'label': metric,
                        'data': grp.groupby(param)[metric].mean().values.tolist(),
                        'borderColor': f'rgba({i*50}, {100+i*50}, {200-i*50}, 1)',
                        'fill': False
                    } for i, metric in enumerate(perf_metrics)
                ]
            },
            'options': {
                'title': {'display': True, 'text': f'Mean Metrics by {param} ({group_name})'},
                'scales': {
                    'x': {'title': {'display': True, 'text': param}},
                    'y': {'title': {'display': True, 'text': 'Mean Value'}}
                }
            }
        }
        print(f'\n=== {group_name}: {param} 對多指標影響的圖表 ===')
        logging.info(f'=== {group_name}: {param} 對多指標影響的圖表 ===')
        print(f'```chartjs\n{chart_data}\n```')
        logging.info(f'```chartjs\n{chart_data}\n```')

# 並行分組分析
def run_group_analysis(keys, grp, group_name, strat_params, perf_metrics, plot_dir):
    strat = keys[0] if isinstance(keys, tuple) else keys
    strat_params = ind_keys.get(strat, param_cols)
    if grp.shape[0] < 10:
        print(f"{group_name} 樣本數 {grp.shape[0]} 過少，跳過")
        logging.info(f"{group_name} 樣本數 {grp.shape[0]} 過少，跳過")
        return
    analyze_group(grp, group_name, strat_params, perf_metrics, plot_dir)

# 執行分析：四種模式
group_modes = [
    ('全統計', [None], df),
    ('分策略', ['strategy'], df),
    ('分數據源', ['data_source'], df),
    ('分策略與數據源', ['strategy', 'data_source'], df)
]

try:
    for mode_name, group_cols, data in group_modes:
        print(f'\n=== 分析模式：{mode_name} ===')
        logging.info(f'=== 分析模式：{mode_name} ===')
        if group_cols[0] is None:
            strat = data['strategy'].iloc[0] if 'strategy' in data.columns else 'single'
            strat_params = ind_keys.get(strat, param_cols)
            analyze_group(data, 'all_data', strat_params, perf_metrics, PLOT_DIR)
        else:
            group_cols = [col for col in group_cols if col in data.columns]
            if not group_cols:
                print(f"無法按 {group_cols} 分組，欄位不存在")
                logging.info(f"無法按 {group_cols} 分組，欄位不存在")
                continue
            Parallel(n_jobs=cfg.N_JOBS, backend="loky")(
                delayed(run_group_analysis)(
                    keys, grp, f"{group_cols[0]}_{sanitize_filename(keys)}" if len(group_cols) == 1 else f"strategy_{sanitize_filename(keys[0])}_source_{sanitize_filename(keys[1])}",
                    strat_params, perf_metrics, PLOT_DIR
                ) for keys, grp in data.groupby(group_cols)
            )
except Exception as e:
    logging.error(f"分析過程發生錯誤：{e}")
    raise
finally:
    # 關閉所有視窗並結束
    print("按任意鍵以關閉所有視窗並結束程式…")
    logging.info("按任意鍵以關閉所有視窗並結束程式…")
    msvcrt.getch()
    plt.close('all')