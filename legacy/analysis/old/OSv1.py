
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import ast
import shutil
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data
from SSSv096 import backtest_unified, param_presets, compute_single, compute_RMA, compute_ssma_turn_combined
from config import RESULT_DIR, WF_PERIODS, STRESS_PERIODS, CACHE_DIR
from metrics import calculate_max_drawdown

# 初始化快取目錄
shutil.rmtree(CACHE_DIR, ignore_errors=True)  # 清空快取
(CACHE_DIR / "price").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "smaa").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "factor").mkdir(parents=True, exist_ok=True)

# Streamlit UI 配置
st.set_page_config(layout="wide")
st.title("00631L 策略回測與走查分析")

# 動態生成走查區間
def generate_walk_forward_periods(data_index, n_splits, min_days=30):
    """根據數據日期範圍生成平分的走查區間"""
    start_date = data_index.min()
    end_date = data_index.max()
    total_days = (end_date - start_date).days
    if total_days < min_days * n_splits:
        n_splits = max(1, total_days // min_days)
        st.warning(f"數據範圍過短，調整分段數為 {n_splits}")
    days_per_split = total_days // n_splits
    periods = []
    current_start = start_date
    for i in range(n_splits):
        if i == n_splits - 1:
            current_end = end_date
        else:
            current_end = current_start + pd.Timedelta(days=days_per_split - 1)
            current_end = data_index[data_index <= current_end][-1]
        if (current_end - current_start).days >= min_days:
            periods.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)
        if current_start not in data_index:
            current_start = data_index[data_index >= current_start][0]
    return periods

# 選擇前 20 個多樣化試驗
def pick_topN_by_diversity(trials, ind_keys, top_n=20, pct_threshold=25):
    trials_sorted = sorted(trials, key=lambda t: -t['score'])
    selected = []
    vectors = []
    for trial in trials_sorted:
        vec = np.array([trial['parameters'][k] for k in ind_keys])
        if not vectors:
            selected.append(trial)
            vectors.append(vec)
            continue
        dists = cdist([vec], vectors, metric="euclidean")[0]
        min_dist = min(dists)
        if len(dists) > 1:
            threshold = np.percentile(dists, pct_threshold)
        else:
            threshold = 0
        if min_dist >= threshold:
            selected.append(trial)
            vectors.append(vec)
        if len(selected) == top_n:
            break
    return selected

# 載入 Optuna 結果
optuna_results_path = RESULT_DIR / "optuna_results_single_Self_20250615_060600.csv"
optuna_results = pd.read_csv(optuna_results_path)
optuna_results['parameters'] = optuna_results['parameters'].apply(ast.literal_eval)
top_trials = optuna_results.sort_values(by='score', ascending=False)

ind_keys = ['linlen', 'factor', 'smaalen', 'devwin']
selected_trials = pick_topN_by_diversity(top_trials.to_dict('records'), ind_keys)

# 建立策略清單
optuna_strategies = [
    {
        'name': f"trial_{trial['trial_number']}",
        'strategy_type': trial['strategy'],
        'params': trial['parameters'],
        'smaa_source': trial['data_source']
    } for trial in selected_trials
]
preset_strategies = [
    {
        'name': key,
        'strategy_type': value['strategy_type'],
        'params': {k: v for k, v in value.items() if k not in ['strategy_type', 'smaa_source']},
        'smaa_source': value['smaa_source']
    } for key, value in param_presets.items()
]
all_strategies = optuna_strategies + preset_strategies

# UI：選擇走查模式
st.sidebar.header("走查設定")
walk_forward_mode = st.sidebar.selectbox("走查模式", ["固定 WF_PERIODS", "動態平分區間"])
if walk_forward_mode == "動態平分區間":
    n_splits = st.sidebar.number_input("分段數", min_value=1, max_value=10, value=3, step=1)
run_backtest = st.sidebar.button("執行回測")

# 載入數據
smaa_sources = set(strategy['smaa_source'] for strategy in all_strategies)
df_price_dict = {}
for source in smaa_sources:
    df_price, df_factor = load_data(ticker="00631L.TW", smaa_source=source)
    st.write(f"數據源 {source} 的 df_price 長度: {len(df_price)}, 日期範圍: {df_price.index.min()} 到 {df_price.index.max()}")
    if not df_factor.empty:
        st.write(f"因子數據長度: {len(df_factor)}, 欄位: {df_factor.columns}")
        st.write("因子數據統計:")
        st.dataframe(df_factor.describe())
        st.write("因子數據頭部:")
        st.dataframe(df_factor.head())
    df_price_dict[source] = df_price

# 回測與分析
if run_backtest:
    with st.spinner("正在執行回測..."):
        results = {}
        for strategy in all_strategies:
            name = strategy['name']
            strategy_type = strategy['strategy_type']
            params = strategy['params']
            smaa_source = strategy['smaa_source']
            df_price = df_price_dict[smaa_source]
            
            st.write(f"處理策略 {name}，參數: {params}")
            
            if strategy_type == 'single':
                df_ind = compute_single(df_price, pd.DataFrame(), params['linlen'], params['factor'], params['smaalen'], params['devwin'], smaa_source=smaa_source)
                st.write(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'RMA':
                df_ind = compute_RMA(df_price, pd.DataFrame(), params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'], smaa_source=smaa_source)
                st.write(f"策略 {name} 的 df_ind 形狀: {df_ind.shape}, 欄位: {df_ind.columns.tolist()}")
            elif strategy_type == 'ssma_turn':
                calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
                ssma_params = {k: v for k, v in params.items() if k in calc_keys}
                backtest_params = ssma_params.copy()
                backtest_params['buy_mult'] = params.get('buy_mult', 0.5)
                backtest_params['sell_mult'] = params.get('sell_mult', 0.5)
                backtest_params['stop_loss'] = params.get('stop_loss', 0.0)
                st.write(f"SSMA_Turn 參數: {ssma_params}")
                df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_price, pd.DataFrame(), **ssma_params, smaa_source=smaa_source)
                st.write(f"策略 {name} 生成的買入信號數: {len(buy_dates)}, 賣出信號數: {len(sell_dates)}")
                st.write(f"買入信號: {buy_dates[:5] if buy_dates else '無'}")
                st.write(f"賣出信號: {sell_dates[:5] if sell_dates else '無'}")
                if df_ind.empty:
                    st.warning(f"策略 {name} 的 df_ind 為空，跳過回測")
                    continue
                result = backtest_unified(df_ind, strategy_type, backtest_params, buy_dates, sell_dates, bad_holding=True)
                results[name] = result
                continue
            
            required_cols = ['open', 'close', 'smaa', 'base', 'sd']
            if df_ind.empty or not all(col in df_ind.columns for col in required_cols):
                st.warning(f"策略 {name} 的 df_ind 缺少必要欄位: {set(required_cols) - set(df_ind.columns)}，跳過回測")
                continue
            
            result = backtest_unified(df_ind, strategy_type, params)
            results[name] = result

        # 提取權益曲線
        equity_curves = pd.DataFrame({name: result['equity_curve'] for name, result in results.items()})
        st.write(f"權益曲線形狀: {equity_curves.shape}")
        st.dataframe(equity_curves.head())

        # 計算相關性矩陣
        corr_matrix = equity_curves.corr()
        corr_matrix.to_csv(RESULT_DIR / 'correlation_matrix.csv')
        st.subheader("相關性矩陣熱圖")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        st.pyplot(plt)

        # 計算走查時段報酬率與排名
        period_returns = {}
        periods = WF_PERIODS if walk_forward_mode == "固定 WF_PERIODS" else generate_walk_forward_periods(equity_curves.index, n_splits)
        for start, end in [(p['test'] if isinstance(p, dict) else p) for p in periods]:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            valid_dates = equity_curves.index
            if start < valid_dates[0]:
                adjusted_start = valid_dates[0]
                st.warning(f"起始日期 {start} 早於數據開始，調整為 {adjusted_start}")
            else:
                adjusted_start = valid_dates[valid_dates >= start][0]
            if end > valid_dates[-1]:
                adjusted_end = valid_dates[-1]
                st.warning(f"結束日期 {end} 晚於數據結束，調整為 {adjusted_end}")
            else:
                adjusted_end = valid_dates[valid_dates <= end][-1]
            if (adjusted_end - adjusted_start).days < 30:
                st.warning(f"走查區間 {adjusted_start} 至 {adjusted_end} 過短，跳過")
                continue
            period_equity = equity_curves.loc[adjusted_start:adjusted_end]
            period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1)
            period_returns[(adjusted_start, adjusted_end)] = period_return
            st.write(f"計算走查區間 {adjusted_start} 至 {adjusted_end} 的報酬率")
        
        period_returns_df = pd.DataFrame(period_returns).T
        period_returns_df.to_csv(RESULT_DIR / 'period_returns.csv')
        st.subheader("走查時段報酬率")
        st.dataframe(period_returns_df)
        
        rankings = period_returns_df.rank(axis=1, ascending=False)
        rankings.to_csv(RESULT_DIR / 'period_rankings.csv')
        st.subheader("走查時段排名")
        st.dataframe(rankings)

        # 計算壓力時段報酬率與排名
        stress_returns = {}
        for start, end in STRESS_PERIODS:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if start in equity_curves.index and end in equity_curves.index:
                period_equity = equity_curves.loc[start:end]
                period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1)
                stress_returns[(start, end)] = period_return
        stress_returns_df = pd.DataFrame(stress_returns).T
        stress_returns_df.to_csv(RESULT_DIR / 'stress_returns.csv')
        st.subheader("壓力時段報酬率")
        st.dataframe(stress_returns_df)
        
        stress_rankings = stress_returns_df.rank(axis=1, ascending=False)
        stress_rankings.to_csv(RESULT_DIR / 'stress_rankings.csv')
        st.subheader("壓力時段排名")
        st.dataframe(stress_rankings)

        # 計算最大回撤
        mdd = {}
        for name, equity in equity_curves.items():
            mdd[name] = calculate_max_drawdown(equity)
        mdd_df = pd.Series(mdd)
        mdd_df.to_csv(RESULT_DIR / 'mdd.csv')
        st.subheader("最大回撤")
        st.dataframe(mdd_df)

        # 繪製權益曲線
        st.subheader("權益曲線")
        fig = px.line(equity_curves, title="策略權益曲線", labels={"value": "權益", "index": "日期"})
        st.plotly_chart(fig, use_container_width=True)

        # 蒙地卡羅測試建議
        st.info("蒙地卡羅測試：請參考 Optuna_12.py 中的 compute_pbo_score 和 compute_simplified_sra 函數實現 PBO 分數與 SRA p 值計算")
