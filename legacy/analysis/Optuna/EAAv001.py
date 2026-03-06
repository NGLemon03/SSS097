import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
from scipy.stats import pearsonr

# 設定 matplotlib 字體
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 設定主目錄與數據路徑
PROJECT_ROOT = Path("C:/Stock_reserach/SSS095a1")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOG_DIR = PROJECT_ROOT / "analysis" / "log" / "single"
st.set_page_config(layout="wide")
st.title("00631L 權益分析與優化平台")
st.sidebar.title("設定與導航")

# 載入 config 模組
import config
from SSSv096 import load_data, param_presets, compute_backtest_for_periods, calculate_metrics
from Optuna_11 import compute_equity_correlations_with_presets, _backtest_once

# 側邊欄輸入
ticker = st.sidebar.selectbox("股票代號", [config.TICKER], index=0)
start_date = st.sidebar.text_input("起始日期", config.START_DATE)
end_date = st.sidebar.text_input("結束日期", "2025-06-06")
trade_cooldown_bars = st.sidebar.number_input("冷卻期 (bars)", min_value=0, value=config.TRADE_COOLDOWN_BARS, step=1)
discount = st.sidebar.slider("券商折數", 0.1, 0.7, 0.3, 0.01)
bad_holding = st.sidebar.checkbox("啟用不良持倉", value=False)
data_source = st.sidebar.selectbox("SMAA 數據源", config.SOURCES, index=0)
n_trials = st.sidebar.number_input("Monte-Carlo 模擬次數", min_value=100, value=1000, step=100)

# 載入數據
df_price, df_factor = load_data(ticker, start_date, end_date, data_source)
if df_price.empty:
    st.error("數據載入失敗，請檢查輸入或網絡連接。")
    st.stop()

# 動態生成時段
def generate_periods(start, end, frequency="year"):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    periods = []
    current = start_dt
    while current < end_dt:
        if frequency == "year":
            next_period = current + timedelta(days=365)
        elif frequency == "half_year":
            next_period = current + timedelta(days=182)
        end_period = min(next_period, end_dt)
        periods.append((current.strftime("%Y-%m-%d"), end_period.strftime("%Y-%m-%d")))
        current = end_period
    return periods

frequency = st.sidebar.selectbox("時段劃分頻率", ["year", "half_year"], index=0)
dynamic_periods = generate_periods(start_date, end_date, frequency)

# 掃描並載入文件
results_files = [f for f in LOG_DIR.glob("optuna_results_*.csv")]
events_files = [f for f in LOG_DIR.glob("optuna_events_*.csv")]
params_files = [f for f in LOG_DIR.glob("optuna_best_params_*.json")]
corr_files = [f for f in LOG_DIR.glob("equity_corr_top_20_with_presets_*.csv")]
plot_files = [f for f in LOG_DIR.glob("*.png")]

# 解析文件名元數據
def parse_filename(filename):
    base_name = filename.stem
    parts = base_name.split("_")
    
    if base_name.startswith("equity_corr_top_20_with_presets"):
        # 僅提取時間戳，無需策略或數據源
        timestamp = parts[-1]
        return {"strategy": "equity_corr", "source": "equity_corr", "timestamp": timestamp}
    elif base_name.endswith(".png"):
        # 處理圖表檔案，假設格式為 <data_source>_mdd_vs_total_return_<timestamp>.png
        if len(parts) >= 4:
            timestamp = parts[-1]
            source_parts = parts[:-3]  # 提取數據源部分
            source = "_".join(source_parts) if source_parts else "unknown"
            return {"strategy": "plot", "source": source, "timestamp": timestamp}
        else:
            # 無法解析的圖表檔案，使用默認值
            return {"strategy": "plot", "source": "unknown", "timestamp": "unknown"}
    else:
        # 處理 optuna_results、optuna_events、optuna_best_params
        if len(parts) >= 4:
            strategy = parts[1]  # 例如 "single"
            source_parts = parts[2:-1]  # 從數據源開始到 timestamp 前
            source = "_".join(source_parts) if source_parts else "Self"
            timestamp = parts[-1]
            return {"strategy": strategy, "source": source, "timestamp": timestamp}
        else:
            # 無法解析的檔案，使用默認值
            return {"strategy": "unknown", "source": "unknown", "timestamp": "unknown"}

results_metadata = {f: parse_filename(f) for f in results_files}
events_metadata = {f: parse_filename(f) for f in events_files}
params_metadata = {f: parse_filename(f) for f in params_files}
corr_metadata = {f: parse_filename(f) for f in corr_files}
plot_metadata = {f: parse_filename(f) for f in plot_files}

# 動態更新側邊欄選項（基於 results_files 提取數據源和時間戳）
available_sources = sorted(list(set(m["source"] for m in results_metadata.values() if m["strategy"] == "single" and m["source"] != "unknown")))
available_timestamps = sorted(list(set(m["timestamp"] for m in results_metadata.values() if m["source"] in available_sources and m["strategy"] == "single" and m["timestamp"] != "unknown")))
selected_source = st.sidebar.selectbox("選擇數據來源", available_sources, index=0 if available_sources else None, disabled=not available_sources)
selected_timestamp = st.sidebar.selectbox("選擇時間戳", available_timestamps, index=0 if available_timestamps else None, disabled=not available_timestamps)

# 載入選定文件（增加錯誤處理）
if not corr_files or not results_files:
    st.warning("目錄中沒有可用的 equity_corr_top_20_with_presets_*.csv 或 optuna_results_*.csv 文件。")
    selected_corr_file = None
    selected_result_file = None
    selected_events_file = None
    selected_params_file = None
    selected_plot_files = []
else:
    try:
        # 匹配與 results 文件對應的 corr 文件
        selected_result_file = next(
            f for f in results_files
            if results_metadata[f]["source"] == selected_source
            and results_metadata[f]["timestamp"] == selected_timestamp
            and results_metadata[f]["strategy"] == "single"
        )
        # 僅根據時間戳匹配 corr 文件
        selected_corr_file = next(
            (f for f in corr_files if corr_metadata[f]["timestamp"] == selected_timestamp),
            None
        )
        selected_events_file = next(
            (f for f in events_files
             if events_metadata[f]["source"] == selected_source
             and events_metadata[f]["timestamp"] == selected_timestamp
             and events_metadata[f]["strategy"] == "single"),
            None
        )
        selected_params_file = next(
            (f for f in params_files
             if params_metadata[f]["source"] == selected_source
             and params_metadata[f]["timestamp"] == selected_timestamp),
            None
        )
        selected_plot_files = [
            f for f in plot_files
            if plot_metadata[f]["source"] == selected_source
            and plot_metadata[f]["timestamp"] == selected_timestamp
        ]
    except StopIteration:
        st.warning(f"未找到與數據源 '{selected_source}' 和時間戳 '{selected_timestamp}' 相符的相關性文件或試驗結果。請檢查文件或選擇其他選項。")
        selected_corr_file = None
        selected_result_file = None
        selected_events_file = None
        selected_params_file = None
        selected_plot_files = []

# 載入選定數據（處理空文件情況）
trial_results = pd.read_csv(selected_result_file) if selected_result_file else pd.DataFrame()
best_params = {}
if selected_params_file:
    with open(selected_params_file, "r", encoding="utf-8") as f:
        best_params = json.load(f)
corr_df = pd.read_csv(selected_corr_file) if selected_corr_file else pd.DataFrame()

# 調試輸出
st.write("Results Metadata:", {f.name: results_metadata[f] for f in results_files})
st.write("Corr Metadata:", {f.name: corr_metadata[f] for f in corr_files})
st.write("Plot Metadata:", {f.name: plot_metadata[f] for f in plot_files})
st.write("Available Sources:", available_sources)
st.write("Available Timestamps:", available_timestamps)

# 頁面導航
page = st.sidebar.selectbox("分析模組", ["相關性熱力圖", "時段排名與報酬", "Monte-Carlo 測試", "CPCV 優化", "壓力時段分析", "其他分析", "圖表顯示"])

# 1. 相關性熱力圖
if page == "相關性熱力圖":
    st.header("權益相關性熱力圖")
    if not corr_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_df.set_index(corr_df.columns[0]), annot=False, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"{selected_source} 權益相關性熱力圖 ({selected_timestamp})")
        st.pyplot(plt)
        plt.close()
    else:
        st.warning("無可用相關性數據。")

# 2. 時段排名與報酬
elif page == "時段排名與報酬":
    st.header("時段內排名與實際報酬率分析")
    if not trial_results.empty:
        top_trials = trial_results.sort_values("score", ascending=False).head(20)
        preset_trials = {k: v for k, v in param_presets.items() if k.startswith("Single") or k.startswith("preset_") or k.startswith("SSMA_turn")}

        rank_data = {"策略": [], "平均排名": []}
        return_data = {"策略": [], "平均報酬": []}
        for period in dynamic_periods:
            rank_data[period[0] + " to " + period[1]] = []
            return_data[period[0] + " to " + period[1]] = []

        for trial in top_trials.itertuples():
            trial_id = trial.trial_number
            ranks = []
            returns = []
            for period in dynamic_periods:
                result = compute_backtest_for_periods(ticker, [period], "single", trial._asdict(), data_source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                if result and result[0]["metrics"]["num_trades"] > 0:
                    rank = top_trials.index[trial] + 1
                    ranks.append(rank)
                    returns.append(result[0]["metrics"]["total_return"])
                else:
                    ranks.append(len(top_trials) + 1)
                    returns.append(0.0)
            rank_data["策略"].append(f"trial_{trial_id}")
            for i, period in enumerate(dynamic_periods):
                rank_data[period[0] + " to " + period[1]].append(ranks[i])
                return_data[period[0] + " to " + period[1]].append(returns[i])
            rank_data["平均排名"].append(np.mean(ranks))
            return_data["平均報酬"].append(np.mean(returns))

        for preset_name, params in preset_trials.items():
            ranks = []
            returns = []
            for period in dynamic_periods:
                result = compute_backtest_for_periods(ticker, [period], params["strategy_type"], params, data_source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                if result and result[0]["metrics"]["num_trades"] > 0:
                    rank = len(top_trials) + 1
                    ranks.append(rank)
                    returns.append(result[0]["metrics"]["total_return"])
                else:
                    ranks.append(len(top_trials) + 1)
                    returns.append(0.0)
            rank_data["策略"].append(f"preset_{preset_name}")
            for i, period in enumerate(dynamic_periods):
                rank_data[period[0] + " to " + period[1]].append(ranks[i])
                return_data[period[0] + " to " + period[1]].append(returns[i])
            rank_data["平均排名"].append(np.mean(ranks))
            return_data["平均報酬"].append(np.mean(returns))

        st.subheader("排名數據")
        st.dataframe(pd.DataFrame(rank_data))
        st.subheader("報酬數據")
        st.dataframe(pd.DataFrame(return_data))
    else:
        st.warning("無可用試驗結果數據。")

# 3. Monte-Carlo 測試
elif page == "Monte-Carlo 測試":
    st.header("Monte-Carlo 穩健性測試")
    if not trial_results.empty:
        top_trial = trial_results.sort_values("score", ascending=False).iloc[0]
        params = {k.split("param_")[1]: v for k, v in top_trial._asdict().items() if k.startswith("param_")}
        original_result = _backtest_once("single", params, [], data_source, df_price, df_factor)
        mc_returns = []
        for _ in range(n_trials):
            shuffled_trades = original_result[5].copy()
            np.random.shuffle(shuffled_trades)
            mc_result = _backtest_once("single", params, [], data_source, df_price, df_factor, trades=shuffled_trades)
            mc_returns.append(mc_result[0])
        st.write(f"原始總報酬: {original_result[0]*100:.2f}%, Monte-Carlo 平均報酬: {np.mean(mc_returns)*100:.2f}%, 標準差: {np.std(mc_returns)*100:.2f}%")
    else:
        st.warning("無可用試驗結果數據。")

# 4. CPCV 優化
elif page == "CPCV 優化":
    st.header("CPCV 優化與排名")
    if not trial_results.empty:
        top_trials = trial_results.sort_values("score", ascending=False).head(5)
        for trial in top_trials.itertuples():
            params = {k.split("param_")[1]: v for k, v in trial._asdict().items() if k.startswith("param_")}
            result = compute_backtest_for_periods(ticker, dynamic_periods, "single", params, data_source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
            oos_returns = [r["metrics"]["total_return"] for r in result if r["metrics"]["num_trades"] > 0]
            st.write(f"試驗 {trial.trial_number}, CPCV 平均 OOS 報酬: {np.mean(oos_returns):.2%}, 標準差: {np.std(oos_returns):.2%}")
    else:
        st.warning("無可用試驗結果數據。")

# 5. 壓力時段分析
elif page == "壓力時段分析":
    st.header("壓力時段績效分析")
    selected_stress_periods = st.multiselect("選擇壓力時段", [f"{p[0]} to {p[1]}" for p in config.STRESS_PERIODS], default=[f"{p[0]} to {p[1]}" for p in config.STRESS_PERIODS[:2]])
    stress_results = {}
    if not trial_results.empty:
        for period_str in selected_stress_periods:
            start, end = period_str.split(" to ")
            result = compute_backtest_for_periods(ticker, [(start, end)], "single", trial_results.sort_values("score", ascending=False).iloc[0]._asdict(), data_source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
            if result and result[0]["metrics"]["num_trades"] > 0:
                stress_results[period_str] = result[0]["metrics"]["total_return"]
        st.dataframe(pd.DataFrame({"壓力時段": list(stress_results.keys()), "總報酬 (%)": [r * 100 for r in stress_results.values()]}))
    else:
        st.warning("無可用試驗結果數據。")

# 6. 其他分析
elif page == "其他分析":
    st.header("其他指標與敏感性分析")
    if not trial_results.empty:
        indicators = ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor"]
        corr_matrix = pd.DataFrame(index=indicators, columns=indicators)
        for i in indicators:
            for j in indicators:
                corr_matrix.loc[i, j] = pearsonr(trial_results[i], trial_results[j])[0]
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        st.pyplot(plt)
        plt.close()

        stress_mdds = trial_results["stress_mdd"].dropna()
        st.write(f"壓力 MDD 平均值: {np.mean(stress_mdds):.3f}, 標準差: {np.std(stress_mdds):.3f}")

        params_to_test = ["linlen", "smaalen", "buy_mult"]
        sensitivity = {}
        for param in params_to_test:
            values = trial_results[f"param_{param}"].unique()
            sens_scores = []
            for v in values:
                subset = trial_results[trial_results[f"param_{param}"] == v]
                sens_scores.append(np.mean(subset["score"]))
            sensitivity[param] = {v: s for v, s in zip(values, sens_scores)}
        st.write("參數敏感性分析:", sensitivity)
    else:
        st.warning("無可用試驗結果數據。")

# 7. 圖表顯示
elif page == "圖表顯示":
    st.header("圖表與視覺化顯示")
    if selected_plot_files:
        for plot_file in selected_plot_files:
            st.image(str(plot_file), caption=plot_file.name, use_container_width=True)
    else:
        st.warning("無可用圖表文件。")

if __name__ == "__main__":
    st.sidebar.button("重新執行分析", on_click=lambda: st.rerun())