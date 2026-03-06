import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import ast
import re
import logging
import plotly.express as px
from scipy.stats import pearsonr

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

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

st.sidebar.title("設定與導航")

# 載入 config 模組
import config
from SSSv096 import load_data, param_presets, compute_backtest_for_periods, calculate_metrics
from Optuna_11 import compute_equity_correlations_with_presets, _backtest_once

# 工具函數：安全轉換 dict(list) 為 DataFrame
def safe_df(d):
    """把 dict(list) 轉成 DataFrame 前，自動補齊長度。"""
    max_len = max(len(v) for v in d.values())
    for k, v in d.items():
        if len(v) < max_len:
            d[k].extend([np.nan] * (max_len - len(v)))  # 用 NaN 補齊
    return pd.DataFrame(d)

# 內部設定參數
ticker = config.TICKER
start_date = config.START_DATE
end_date = "2025-06-06"
trade_cooldown_bars = config.TRADE_COOLDOWN_BARS
discount = 0.3
bad_holding = False
n_trials = 1000

# 有效參數鍵
valid_keys = config.STRATEGY_PARAMS['single']['ind_keys'] + config.STRATEGY_PARAMS['single']['bt_keys']

# 側邊欄輸入
frequency = st.sidebar.selectbox("時段劃分頻率", ["year", "half_year"], index=0)
page = st.sidebar.selectbox("分析模組", ["時段排名與報酬", "相關性熱力圖", "Monte-Carlo 測試", "CPCV 優化", "壓力時段分析", "其他分析", "圖表顯示"])

# 載入數據
df_price, df_factor = load_data(ticker, start_date, end_date, "Self")
if df_price.empty:
    st.error("數據載入失敗，請檢查輸入或網絡連接。")
    st.stop()
logger.info("成功載入價格數據，形狀：%s", df_price.shape)

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

dynamic_periods = generate_periods(start_date, end_date, frequency)

# 掃描並載入文件
results_files = [f for f in LOG_DIR.glob("optuna_results_*.csv")]
events_files = [f for f in LOG_DIR.glob("optuna_events_*.csv")]
params_files = [f for f in LOG_DIR.glob("optuna_best_params_*.json")]
corr_files = [f for f in LOG_DIR.glob("equity_corr_top_20_with_presets_*.csv")]
plot_files = [f for f in LOG_DIR.glob("*.png")]
logger.info("掃描到 %d 個 results 文件，%d 個 corr 文件，%d 個 plot 文件", len(results_files), len(corr_files), len(plot_files))

# 提取時間戳
def extract_timestamp(fname: str) -> datetime:
    m = re.search(r"(\d{8}_\d{6})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            logger.warning("無法解析時間戳：%s", m.group(1))
            return datetime.min
    logger.warning("無法從檔名抽取時間戳：%s", fname)
    return datetime.min

# 解析文件名元數據
def parse_filename(filename):
    base_name = filename.stem
    parts = base_name.split("_")
    
    if base_name.startswith("equity_corr_top_20_with_presets"):
        timestamp = extract_timestamp(base_name).strftime("%Y%m%d_%H%M%S") if extract_timestamp(base_name) != datetime.min else "unknown"
        return {"strategy": "equity_corr", "source": "equity_corr", "timestamp": timestamp}
    
    elif base_name.endswith(".png"):
        if any(x in base_name for x in ["mdd_vs_total_return", "sharpe_vs_total_return"]):
            if len(parts) >= 5:
                timestamp = parts[-2] + "_" + parts[-1]
                source = "_".join(parts[:-3]) if parts[:-3] else "unknown"
                return {"strategy": "plot", "source": source, "timestamp": timestamp}
        elif base_name.startswith("equity_corr_top_20_with_presets"):
            timestamp = extract_timestamp(base_name).strftime("%Y%m%d_%H%M%S") if extract_timestamp(base_name) != datetime.min else "unknown"
            return {"strategy": "plot", "source": "equity_corr", "timestamp": timestamp}
        logger.warning("無法解析圖表文件名：%s", base_name)
        return {"strategy": "plot", "source": "unknown", "timestamp": "unknown"}
    
    elif base_name.startswith("optuna_"):
        if len(parts) >= 5 and parts[2] == "single":
            strategy = parts[2]
            source_parts = parts[3:-2]
            source = "_".join(source_parts) if source_parts else "Self"
            timestamp = parts[-2] + "_" + parts[-1]
            return {"strategy": strategy, "source": source, "timestamp": timestamp}
        logger.warning("無法解析 optuna 文件名：%s", base_name)
        return {"strategy": "unknown", "source": "unknown", "timestamp": "unknown"}
    
    logger.warning("未知文件名格式：%s", base_name)
    return {"strategy": "unknown", "source": "unknown", "timestamp": "unknown"}

results_metadata = {f: parse_filename(f) for f in results_files}
events_metadata = {f: parse_filename(f) for f in events_files}
params_metadata = {f: parse_filename(f) for f in params_files}
corr_metadata = {f: parse_filename(f) for f in corr_files}
plot_metadata = {f: parse_filename(f) for f in plot_files}

# 排序 corr_files 並對應到數據源
corr_files_sorted = sorted(corr_files, key=lambda f: extract_timestamp(f.name))
data_sources = ["Self", "Factor_TWII_2412_TW", "Factor_TWII_2414_TW"]
corr_file_mapping = {f: src for f, src in zip(corr_files_sorted, data_sources)}
logger.info("Corr 文件映射：%s", {f.name: src for f, src in corr_file_mapping.items()})

# 為每個數據源顯示分析結果
for source in data_sources:
    # 查找對應的文件
    result_file = next((f for f in results_files if results_metadata[f]["source"] == source and results_metadata[f]["strategy"] == "single"), None)
    event_file = next((f for f in events_files if events_metadata[f]["source"] == source and events_metadata[f]["strategy"] == "single"), None)
    param_file = next((f for f in params_files if params_metadata[f]["source"] == source), None)
    corr_file = next((f for f, s in corr_file_mapping.items() if s == source), None)
    plot_files_for_source = [
        f for f in plot_files
        if plot_metadata[f]["source"] == source or
           (plot_metadata[f]["source"] == "equity_corr" and f in corr_file_mapping and corr_file_mapping[f] == source)
    ]

    # 載入數據
    try:
        if result_file:
            trial_result = pd.read_csv(result_file, dtype={
                "total_return": str,
                "sharpe_ratio": float,
                "max_drawdown": float,
                "profit_factor": float,
                "stress_mdd": float,
                "num_trades": int,
                "avg_hold_days": str,
                "score": float,
                "parameters": str
            })
            # 清洗資料
            trial_result["total_return"] = trial_result["total_return"].str.replace("%", "").astype(float)
            trial_result["avg_hold_days"] = trial_result["avg_hold_days"].str.replace("天", "").astype(float)
            numeric_cols = ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor", "stress_mdd", "score"]
            for col in numeric_cols:
                trial_result[col] = pd.to_numeric(trial_result[col], errors="coerce")
        else:
            trial_result = pd.DataFrame()
    except Exception as e:
        logger.error(f"載入試驗結果文件 {result_file} 失敗：{e}")
        trial_result = pd.DataFrame()

    best_params = {}
    if param_file:
        try:
            with open(param_file, "r", encoding="utf-8") as f:
                best_params = json.load(f)
        except Exception as e:
            logger.error(f"載入參數文件 {param_file} 失敗：{e}")

    try:
        corr_df = pd.read_csv(corr_file) if corr_file else pd.DataFrame()
        if not corr_df.empty:
            corr_df = corr_df.set_index(corr_df.columns[0])
            corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        logger.error(f"載入相關性文件 {corr_file} 失敗：{e}")
        corr_df = pd.DataFrame()

    logger.info("數據源 %s：結果文件=%s，相關性文件=%s，圖表文件數=%d",
                source, result_file.name if result_file else "None",
                corr_file.name if corr_file else "None", len(plot_files_for_source))

    # 1. 相關性熱力圖
    if page == "相關性熱力圖":
        if not corr_df.empty:
            # 添加刻度調整滑塊
            zmin = st.slider("最小相關性刻度", -1.0, 0.0, -1.0, 0.1, key=f"zmin_{source}")
            zmax = st.slider("最大相關性刻度", 0.0, 1.0, 1.0, 0.1, key=f"zmax_{source}")
            show_heatmap = st.checkbox("顯示熱力圖", value=True, key=f"show_heatmap_{source}")

            if show_heatmap:
                fig = px.imshow(
                    corr_df,
                    color_continuous_scale="RdBu_r",
                    zmin=zmin,
                    zmax=zmax,
                    title=f"權益相關性熱力圖 ({source})",
                    width=800,
                    height=800
                )
                fig.update_traces(hovertemplate="%{x}<br>%{y}<br>相關性: %{z:.2f}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("熱力圖已隱藏，點擊顯示以查看詳細數據。")
                if st.button("顯示個別標的詳細數據", key=f"detail_{source}"):
                    selected_target = st.selectbox("選擇標的", corr_df.index.tolist(), key=f"target_{source}")
                    st.write(f"標的 {selected_target} 的相關性數據:", corr_df[selected_target])
        else:
            st.warning(f"數據源 {source} 無可用相關性數據。")
        # 新增重新計算相關性按鈕
        if st.button("重新計算相關性", key=f"recalc_corr_{source}"):
            with st.spinner("重新計算相關性中..."):
                try:
                    # 使用當前 trial_result 重新計算
                    trial_results = trial_result.to_dict('records')
                    corr_df = compute_equity_correlations_with_presets(
                        trial_results=trial_results,
                        param_presets=param_presets,
                        top_n=20,
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        output_dir=LOG_DIR
                    )
                    st.success(f"數據源 {source} 的相關性已重新計算並更新。")
                    # 強制刷新頁面以顯示新數據
                    st.experimental_rerun()
                except Exception as e:
                    logger.error(f"重新計算相關性失敗 (數據源: {source})：{e}")
                    st.error(f"重新計算相關性失敗 (數據源: {source})：{e}")

    # 2. 時段排名與報酬
    elif page == "時段排名與報酬":
        st.subheader(f"時段內排名與實際報酬率分析 (數據源: {source})")
        if not trial_result.empty:
            # 排序並獲取所有試驗
            all_trials = trial_result.sort_values("score", ascending=False)
            params_list = []
            for _, trial in all_trials.iterrows():
                try:
                    params = ast.literal_eval(trial["parameters"]) if isinstance(trial["parameters"], str) else trial["parameters"]
                    if isinstance(params, dict):
                        params_list.append([params.get(k, 0) for k in valid_keys if k in params])
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"試驗 {trial['trial_number']} 的參數解析失敗，跳過：{e}")
                    continue

            # 記錄 WF_PERIODS 和 STRESS_PERIODS 內容以供除錯
            logger.info("WF_PERIODS content: %s", config.WF_PERIODS)
            logger.info("STRESS_PERIODS content: %s", config.STRESS_PERIODS)

            # 參數多樣性篩選
            if len(params_list) > 1:
                from scipy.spatial.distance import cdist
                try:
                    distance_matrix = cdist(np.array(params_list), np.array(params_list), metric='euclidean')
                    diverse_indices = []
                    for i in range(len(params_list)):
                        # 計算當前向量與其他向量的最小距離（排除自身）
                        min_dist = np.min(distance_matrix[i, [j for j in range(len(params_list)) if j != i]]) if len(params_list) > 1 else 0
                        threshold = np.percentile(distance_matrix.flatten(), 25)
                        if min_dist >= threshold:
                            diverse_indices.append(i)
                    top_trials = all_trials.iloc[diverse_indices].head(20)
                except Exception as e:
                    logger.error(f"參數多樣性篩選失敗：{e}, 使用原始前 20 試驗")
                    top_trials = all_trials.head(20)
            else:
                top_trials = all_trials.head(20)

            # 存入 session_state
            st.session_state[f"top_trials_{source}"] = top_trials

            preset_trials = {k: v for k, v in param_presets.items() if k.startswith("Single") or k.startswith("SSMA_turn")}

            rank_data = {"策略": [], "平均排名": [], "WF最小報酬": [], "壓力平均報酬": []}
            return_data = {"策略": [], "平均報酬": [], "WF最小報酬": [], "壓力平均報酬": []}
            trial_data = {
                "策略": [], "試驗編號": [], "分數": [], "參數": [],
                "總報酬率 (%)": [], "夏普比率": [], "最大回撤": [], "盈虧因子": [],
                "交易次數": [], "平均持倉天數": []
            }
            for period in dynamic_periods:
                rank_data[period[0] + " to " + period[1]] = []
                return_data[period[0] + " to " + period[1]] = []

            # 驗證並清理 WF_PERIODS
            wf_periods_clean = []
            for period in config.WF_PERIODS:
                if isinstance(period, dict) and "test" in period and isinstance(period["test"], (list, tuple)) and len(period["test"]) == 2:
                    wf_periods_clean.append(period["test"])
                else:
                    logger.error(f"Invalid WF_PERIOD entry (skipped): {period}")
            if not wf_periods_clean:
                logger.warning("No valid WF_PERIODS found, using empty list")
                wf_periods_clean = []

            # 驗證並清理 STRESS_PERIODS
            stress_periods_clean = []
            for period in config.STRESS_PERIODS:
                if isinstance(period, (list, tuple)) and len(period) == 2:
                    stress_periods_clean.append(period)
                else:
                    logger.error(f"Invalid STRESS_PERIOD entry (skipped): {period}")
            if not stress_periods_clean:
                logger.warning("No valid STRESS_PERIODS found, using empty list")
                stress_periods_clean = []
            # 加入計算訊息
            st.write(f"開始處理 {len(top_trials)} 個試驗的回測與排名計算...")
            with st.spinner("計算中..."):
                # 處理前 20 試驗
                for idx, trial in top_trials.iterrows():
                    trial_id = trial["trial_number"]
                    logger.info(f"處理試驗 {trial_id}...")
                    ranks = []
                    returns = []
                    wf_returns = []
                    stress_returns = []
                    try:
                        trial_params = ast.literal_eval(trial["parameters"]) if isinstance(trial["parameters"], str) else trial["parameters"]
                        if not isinstance(trial_params, dict):
                            logger.warning(f"試驗 {trial_id} 的參數格式無效，跳過：{trial['parameters']}")
                            continue
                        
                        # 動態調整 valid_keys 根據 strategy_type
                        strategy_type = trial.get("strategy", "single")
                        valid_keys_dynamic = (
                            config.STRATEGY_PARAMS.get(strategy_type, {}).get("ind_keys", []) +
                            config.STRATEGY_PARAMS.get(strategy_type, {}).get("bt_keys", [])
                        ) or valid_keys
                        trial_params = {k: v for k, v in trial_params.items() if k in valid_keys_dynamic}

                        # 確保 ssma_turn 參數完整
                        if strategy_type == "ssma_turn":
                            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
                            default_ssma_params = next((v for k, v in param_presets.items() if k.startswith("SSMA_turn")), {})
                            for key in calc_keys:
                                if key not in trial_params:
                                    trial_params[key] = default_ssma_params.get(key, 0)
                            # 檢查並記錄參數
                            missing_keys = [k for k in calc_keys if k not in trial_params]
                            if missing_keys:
                                logger.warning(f"試驗 {trial_id} 缺少參數：{missing_keys}")
                            logger.info(f"試驗 {trial_id} ssma_turn 參數：{trial_params}")

                    except (ValueError, SyntaxError) as e:
                        logger.error(f"試驗 {trial_id} 的參數解析失敗：{e}")
                        continue

                    for period in dynamic_periods:
                        start, end = period
                        period_data = df_price.loc[start:end]
                        if len(period_data) < 2 or len(period_data) < max(trial_params.get("smaalen", 60), trial_params.get("linlen", 60)) * 1.5:
                            logger.warning(f"時段 {start} 至 {end} 數據不足或無效（長度={len(period_data)}），跳過試驗 {trial_id}")
                            ranks.append(len(top_trials) + 1)
                            returns.append(0.0)
                            wf_returns.append(0.0)
                            stress_returns.append(0.0)
                            continue

                        result = compute_backtest_for_periods(ticker, [period], strategy_type, trial_params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                        if result and result[0]["metrics"].get("num_trades", 0) > 0:
                            rank = top_trials.index.get_loc(idx) + 1
                            ranks.append(rank)
                            returns.append(result[0]["metrics"].get("total_return", 0.0))
                        else:
                            ranks.append(len(top_trials) + 1)
                            returns.append(0.0)

                        # 計算 WF 最小報酬
                        valid_wf_periods = []
                        warmup_days = max(trial_params.get("smaalen", 60), trial_params.get("linlen", 60))
                        for wf_start, wf_end in wf_periods_clean:
                            start_candidates = df_price.index[df_price.index >= pd.Timestamp(wf_start)]
                            end_candidates = df_price.index[df_price.index <= pd.Timestamp(wf_end)]
                            if not start_candidates.empty and not end_candidates.empty and len(start_candidates) >= 2 and len(end_candidates) >= 2:
                                adjusted_start = start_candidates[0]
                                adjusted_end = end_candidates[-1]
                                if (adjusted_end - adjusted_start).days >= warmup_days * 1.5:
                                    valid_wf_periods.append((adjusted_start, adjusted_end))
                            else:
                                logger.warning(f"試驗 {trial_id} WF 期間 {wf_start} 至 {wf_end} 數據範圍無效")
                        if valid_wf_periods:
                            logger.info(f"試驗 {trial_id} WF 期間：{valid_wf_periods}")
                            wf_result = compute_backtest_for_periods(ticker, valid_wf_periods, strategy_type, trial_params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                            wf_returns.append(min([r["metrics"].get("total_return", 0.0) for r in wf_result if r["metrics"].get("num_trades", 0) > 0] or [0.0]))
                        else:
                            wf_returns.append(0.0)
                            logger.warning(f"試驗 {trial_id} 無有效 WF 期間")

                        # 計算壓力平均報酬
                        valid_stress_periods = []
                        for stress_start, stress_end in stress_periods_clean:
                            start_candidates = df_price.index[df_price.index >= pd.Timestamp(stress_start)]
                            end_candidates = df_price.index[df_price.index <= pd.Timestamp(stress_end)]
                            if not start_candidates.empty and not end_candidates.empty and len(start_candidates) >= 2 and len(end_candidates) >= 2:
                                adjusted_start = start_candidates[0]
                                adjusted_end = end_candidates[-1]
                                if (adjusted_end - adjusted_start).days >= warmup_days * 2:
                                    valid_stress_periods.append((adjusted_start, adjusted_end))
                            else:
                                logger.warning(f"試驗 {trial_id} 壓力期間 {stress_start} 至 {stress_end} 數據範圍無效")
                        if valid_stress_periods:
                            logger.info(f"試驗 {trial_id} 壓力期間：{valid_stress_periods}")
                            stress_result = compute_backtest_for_periods(ticker, valid_stress_periods, strategy_type, trial_params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                            stress_returns.append(np.mean([r["metrics"].get("total_return", 0.0) for r in stress_result if r["metrics"].get("num_trades", 0) > 0] or [0.0]))
                        else:
                            stress_returns.append(0.0)
                            logger.warning(f"試驗 {trial_id} 無有效壓力期間")

                    rank_data["策略"].append(f"trial_{trial_id}")
                    for i, period in enumerate(dynamic_periods):
                        rank_data[period[0] + " to " + period[1]].append(ranks[i])
                        return_data[period[0] + " to " + period[1]].append(returns[i])
                    rank_data["平均排名"].append(np.mean(ranks) if ranks else 0.0)
                    rank_data["WF最小報酬"].append(np.mean(wf_returns) if wf_returns else 0.0)
                    rank_data["壓力平均報酬"].append(np.mean(stress_returns) if stress_returns else 0.0)
                    return_data["平均報酬"].append(np.mean(returns) if returns else 0.0)
                    return_data["WF最小報酬"].append(np.mean(wf_returns) if wf_returns else 0.0)
                    return_data["壓力平均報酬"].append(np.mean(stress_returns) if stress_returns else 0.0)

                    trial_data["策略"].append(f"trial_{trial_id}")
                    trial_data["試驗編號"].append(trial_id)
                    trial_data["分數"].append(trial["score"])
                    trial_data["參數"].append(trial["parameters"])
                    trial_data["總報酬率 (%)"].append(trial["total_return"])
                    trial_data["夏普比率"].append(trial["sharpe_ratio"])
                    trial_data["最大回撤"].append(trial["max_drawdown"])
                    trial_data["盈虧因子"].append(trial["profit_factor"])
                    trial_data["交易次數"].append(trial["num_trades"])
                    trial_data["平均持倉天數"].append(trial["avg_hold_days"])

            # 處理預設參數
            for preset_name, params in preset_trials.items():
                ranks = []
                returns = []
                wf_returns = []
                stress_returns = []
                strategy_type = params.get("strategy_type", "single")
                valid_keys_dynamic = (
                    config.STRATEGY_PARAMS.get(strategy_type, {}).get("ind_keys", []) +
                    config.STRATEGY_PARAMS.get(strategy_type, {}).get("bt_keys", [])
                ) or valid_keys
                valid_params = {k: v for k, v in params.items() if k in valid_keys_dynamic}
                
                    # 確保 ssma_turn 參數完整
                if strategy_type == "ssma_turn":
                    calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
                    for key in calc_keys:
                        if key not in valid_params:
                            valid_params[key] = params.get(key, 0)
                        logger.info(f"預設 {preset_name} ssma_turn 參數：{valid_params}")

                for period in dynamic_periods:
                    start, end = period
                    period_data = df_price.loc[start:end]
                    if len(period_data) < 2 or len(period_data) < max(valid_params.get("smaalen", 60), valid_params.get("linlen", 60)) * 1.5:
                        logger.warning(f"時段 {start} 至 {end} 數據不足或無效（長度={len(period_data)}），跳過預設參數 {preset_name}")
                        ranks.append(len(top_trials) + 1)
                        returns.append(0.0)
                        wf_returns.append(0.0)
                        stress_returns.append(0.0)
                        continue

                        result = compute_backtest_for_periods(ticker, [period], strategy_type, valid_params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                    if result and result[0]["metrics"].get("num_trades", 0) > 0:
                        rank = len(top_trials) + 1
                        ranks.append(rank)
                        returns.append(result[0]["metrics"].get("total_return", 0.0))
                    else:
                        ranks.append(len(top_trials) + 1)
                        returns.append(0.0)

                    # 計算 WF 最小報酬
                    valid_wf_periods = []
                    warmup_days = max(valid_params.get("smaalen", 60), valid_params.get("linlen", 60))
                    for wf_start, wf_end in wf_periods_clean:
                        start_candidates = df_price.index[df_price.index >= pd.Timestamp(wf_start)]
                        end_candidates = df_price.index[df_price.index <= pd.Timestamp(wf_end)]
                        if not start_candidates.empty and not end_candidates.empty and len(start_candidates) >= 2 and len(end_candidates) >= 2:
                            adjusted_start = start_candidates[0]
                            adjusted_end = end_candidates[-1]
                            if (adjusted_end - adjusted_start).days >= warmup_days * 1.5:
                                valid_wf_periods.append((adjusted_start, adjusted_end))
                        else:
                            logger.warning(f"預設 {preset_name} WF 期間 {wf_start} 至 {wf_end} 數據範圍無效")
                    if valid_wf_periods:
                        logger.info(f"預設 {preset_name} WF 期間：{valid_wf_periods}")
                        wf_result = compute_backtest_for_periods(ticker, valid_wf_periods, strategy_type, valid_params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                        wf_returns.append(min([r["metrics"].get("total_return", 0.0) for r in wf_result if r["metrics"].get("num_trades", 0) > 0] or [0.0]))
                    else:
                        wf_returns.append(0.0)
                        logger.warning(f"預設 {preset_name} 無有效 WF 期間")

                    # 計算壓力平均報酬
                    valid_stress_periods = []
                    for stress_start, stress_end in stress_periods_clean:
                        start_candidates = df_price.index[df_price.index >= pd.Timestamp(stress_start)]
                        end_candidates = df_price.index[df_price.index <= pd.Timestamp(stress_end)]
                        if not start_candidates.empty and not end_candidates.empty and len(start_candidates) >= 2 and len(end_candidates) >= 2:
                            adjusted_start = start_candidates[0]
                            adjusted_end = end_candidates[-1]
                            if (adjusted_end - adjusted_start).days >= warmup_days * 2:
                                valid_stress_periods.append((adjusted_start, adjusted_end))
                        else:
                            logger.warning(f"預設 {preset_name} 壓力期間 {stress_start} 至 {stress_end} 數據範圍無效")
                    if valid_stress_periods:
                        logger.info(f"預設 {preset_name} 壓力期間：{valid_stress_periods}")
                        stress_result = compute_backtest_for_periods(ticker, valid_stress_periods, strategy_type, valid_params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                        stress_returns.append(np.mean([r["metrics"].get("total_return", 0.0) for r in stress_result if r["metrics"].get("num_trades", 0) > 0] or [0.0]))
                    else:
                        stress_returns.append(0.0)
                        logger.warning(f"預設 {preset_name} 無有效壓力期間")

                rank_data["策略"].append(f"preset_{preset_name}")
                for i, period in enumerate(dynamic_periods):
                    rank_data[period[0] + " to " + period[1]].append(ranks[i])
                    return_data[period[0] + " to " + period[1]].append(returns[i])
                rank_data["平均排名"].append(np.mean(ranks) if ranks else 0.0)
                rank_data["WF最小報酬"].append(np.mean(wf_returns) if wf_returns else 0.0)
                rank_data["壓力平均報酬"].append(np.mean(stress_returns) if stress_returns else 0.0)
                return_data["平均報酬"].append(np.mean(returns) if returns else 0.0)
                return_data["WF最小報酬"].append(np.mean(wf_returns) if wf_returns else 0.0)
                return_data["壓力平均報酬"].append(np.mean(stress_returns) if stress_returns else 0.0)

            st.success(f"完成 {len(top_trials)} 個試驗的回測與排名計算。")
            st.subheader("前 20 試驗詳細資料")
            trial_df = safe_df(trial_data)
            st.dataframe(trial_df.style.format({
                "分數": "{:.3f}",
                "總報酬率 (%)": "{:.2f}",
                "夏普比率": "{:.3f}",
                "最大回撤": "{:.3f}",
                "盈虧因子": "{:.2f}",
                "交易次數": "{:d}",
                "平均持倉天數": "{:.1f}"
            }))

            st.subheader("排名數據")
            st.dataframe(safe_df(rank_data))
            st.subheader("報酬數據")
            st.dataframe(safe_df(return_data))
        else:
            st.warning(f"數據源 {source} 無可用試驗結果數據。")

    # 3. Monte-Carlo 測試
    elif page == "Monte-Carlo 測試":
        st.subheader(f"Monte-Carlo 穩健性測試 (數據源: {source})")
        if not trial_result.empty:
            top_trials = st.session_state.get(f"top_trials_{source}", trial_result.sort_values("score", ascending=False).head(1))
            top_trial = top_trials.iloc[0]
            try:
                params = ast.literal_eval(top_trial["parameters"]) if isinstance(top_trial["parameters"], str) else top_trial["parameters"]
                if not isinstance(params, dict):
                    logger.warning(f"頂尖試驗的參數格式無效，跳過 Monte-Carlo 測試")
                    st.warning(f"數據源 {source} 的參數格式無效，無法進行 Monte-Carlo 測試。")
                    continue
                # 過濾有效參數
                strategy_type = top_trial.get("strategy", "single")
                valid_keys_dynamic = (
                    config.STRATEGY_PARAMS.get(strategy_type, {}).get("ind_keys", []) +
                    config.STRATEGY_PARAMS.get(strategy_type, {}).get("bt_keys", [])
                ) or valid_keys
                params = {k: v for k, v in params.items() if k in valid_keys_dynamic}
                if strategy_type == "ssma_turn":
                    default_ssma_params = next((v for k, v in param_presets.items() if k.startswith("SSMA_turn")), {})
                    for key in ["prom_factor", "min_dist", "buy_shift", "exit_shift", "vol_window", "quantile_win", "signal_cooldown_days"]:
                        if key not in params:
                            params[key] = default_ssma_params.get(key, 0)
            except (ValueError, SyntaxError) as e:
                logger.error(f"頂尖試驗的參數解析失敗：{e}")
                st.warning(f"數據源 {source} 的參數解析失敗，無法進行 Monte-Carlo 測試。")
                continue
            try:
                _, source_factor = load_data(ticker, start_date, end_date, source)
                original_result = _backtest_once(strategy_type, params, [], source, df_price, source_factor)
                mc_returns = []
                for _ in range(n_trials):
                    shuffled_trades = original_result[5].copy()
                    np.random.shuffle(shuffled_trades)
                    mc_result = _backtest_once(strategy_type, params, [], source, df_price, source_factor, trades=shuffled_trades)
                    mc_returns.append(mc_result[0])
                st.write(f"原始總報酬: {original_result[0]*100:.2f}%, Monte-Carlo 平均報酬: {np.mean(mc_returns)*100:.2f}%, 標準差: {np.std(mc_returns)*100:.2f}%")
            except Exception as e:
                logger.error(f"Monte-Carlo 測試失敗 (數據源: {source})：{e}")
                st.warning(f"數據源 {source} 的 Monte-Carlo 測試失敗。")
        else:
            st.warning(f"數據源 {source} 無可用試驗結果數據。")

    # 4. CPCV 優化
    elif page == "CPCV 優化":
        st.subheader(f"CPCV 優化與排名 (數據源: {source})")
        if not trial_result.empty:
            top_trials = st.session_state.get(f"top_trials_{source}", trial_result.sort_values("score", ascending=False).head(5))
            for _, trial in top_trials.iterrows():
                try:
                    params = ast.literal_eval(trial["parameters"]) if isinstance(trial["parameters"], str) else trial["parameters"]
                    if not isinstance(params, dict):
                        logger.warning(f"試驗 {trial['trial_number']} 的參數格式無效，跳過")
                        continue
                    # 過濾有效參數
                    strategy_type = trial.get("strategy", "single")
                    valid_keys_dynamic = (
                        config.STRATEGY_PARAMS.get(strategy_type, {}).get("ind_keys", []) +
                        config.STRATEGY_PARAMS.get(strategy_type, {}).get("bt_keys", [])
                    ) or valid_keys
                    params = {k: v for k, v in params.items() if k in valid_keys_dynamic}
                    if strategy_type == "ssma_turn":
                        default_ssma_params = next((v for k, v in param_presets.items() if k.startswith("SSMA_turn")), {})
                        for key in ["prom_factor", "min_dist", "buy_shift", "exit_shift", "vol_window", "quantile_win", "signal_cooldown_days"]:
                            if key not in params:
                                params[key] = default_ssma_params.get(key, 0)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"試驗 {trial['trial_number']} 的參數解析失敗：{e}")
                    continue
                try:
                    result = compute_backtest_for_periods(ticker, dynamic_periods, strategy_type, params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                    oos_returns = [
                        r["metrics"].get("total_return", np.nan)
                        for r in result if r["metrics"].get("num_trades", 0) > 0
                    ]
                    oos_returns = [x for x in oos_returns if not np.isnan(x)]
                    if oos_returns:
                        st.write(f"試驗 {trial['trial_number']}, CPCV 平均 OOS 報酬: {np.mean(oos_returns):.2%}, 標準差: {np.std(oos_returns):.2%}")
                    else:
                        st.warning(f"CPCV：試驗 {trial['trial_number']} 所有折衝均無有效交易")
                except Exception as e:
                    logger.error(f"CPCV 優化失敗 (試驗: {trial['trial_number']}, 數據源: {source})：{e}")
                    st.warning(f"試驗 {trial['trial_number']} 的 CPCV 優化失敗。")
        else:
            st.warning(f"數據源 {source} 無可用試驗結果數據。")

    # 5. 壓力時段分析
    elif page == "壓力時段分析":
        st.subheader(f"壓力時段績效分析 (數據源: {source})")
        selected_stress_periods = st.multiselect(
            f"選擇壓力時段 (數據源: {source})",
            [f"{p[0]} to {p[1]}" for p in config.STRESS_PERIODS],
            default=[f"{p[0]} to {p[1]}" for p in config.STRESS_PERIODS[:2]],
            key=f"stress_periods_{source}"
        )
        if not trial_result.empty:
            top_trials = st.session_state.get(f"top_trials_{source}", trial_result.sort_values("score", ascending=False).head(1))
            top_trial = top_trials.iloc[0]
            try:
                params = ast.literal_eval(top_trial["parameters"]) if isinstance(top_trial["parameters"], str) else top_trial["parameters"]
                if not isinstance(params, dict):
                    logger.warning(f"頂尖試驗的參數格式無效，跳過壓力時段分析")
                    st.warning(f"數據源 {source} 的參數格式無效，無法進行壓力時段分析。")
                    continue
                # 過濾有效參數
                strategy_type = top_trial.get("strategy", "single")
                valid_keys_dynamic = (
                    config.STRATEGY_PARAMS.get(strategy_type, {}).get("ind_keys", []) +
                    config.STRATEGY_PARAMS.get(strategy_type, {}).get("bt_keys", [])
                ) or valid_keys
                params = {k: v for k, v in params.items() if k in valid_keys_dynamic}
                if strategy_type == "ssma_turn":
                    default_ssma_params = next((v for k, v in param_presets.items() if k.startswith("SSMA_turn")), {})
                    for key in ["prom_factor", "min_dist", "buy_shift", "exit_shift", "vol_window", "quantile_win", "signal_cooldown_days"]:
                        if key not in params:
                            params[key] = default_ssma_params.get(key, 0)
            except (ValueError, SyntaxError) as e:
                logger.error(f"頂尖試驗的參數解析失敗：{e}")
                st.warning(f"數據源 {source} 的參數解析失敗，無法進行壓力時段分析。")
                continue
            stress_results = {}
            for period_str in selected_stress_periods:
                start, end = period_str.split(" to ")
                try:
                    period_data = df_price.loc[start:end]
                    if len(period_data) < 2 or len(period_data) < max(params.get("smaalen", 0), params.get("linlen", 0)):
                        logger.warning(f"壓力時段 {start} 至 {end} 數據不足（長度={len(period_data)}），跳過")
                        continue
                    result = compute_backtest_for_periods(ticker, [(start, end)], strategy_type, params, source, trade_cooldown_bars, discount, bad_holding, df_price, df_factor)
                    val = result[0]["metrics"].get("total_return", np.nan)
                    if not np.isnan(val) and result[0]["metrics"].get("num_trades", 0) > 0:
                        stress_results[period_str] = val
                    else:
                        logger.warning(f"壓力時段 {period_str} 無有效交易或結果為 NaN，跳過")
                except Exception as e:
                    logger.error(f"壓力時段分析失敗 (時段: {period_str}, 數據源: {source})：{e}")
                    st.warning(f"時段 {period_str} 的壓力分析失敗。")
            if stress_results:
                st.dataframe(pd.DataFrame({"壓力時段": list(stress_results.keys()), "總報酬 (%)": [r * 100 for r in stress_results.values()]}))
            else:
                st.info(f"數據源 {source} 選定時段過短或無交易，已自動略過")
        else:
            st.warning(f"數據源 {source} 無可用試驗結果數據。")

    # 6. 其他分析
    elif page == "其他分析":
        st.subheader(f"其他指標與敏感性分析 (數據源: {source})")
        if not trial_result.empty:
            top_trials = st.session_state.get(f"top_trials_{source}", trial_result.sort_values("score", ascending=False).head(20))
            indicators = ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor"]
            try:
                corr_matrix = pd.DataFrame(index=indicators, columns=indicators)
                for i in indicators:
                    for j in indicators:
                        if top_trials[i].notna().all() and top_trials[j].notna().all():
                            corr_matrix.loc[i, j] = pearsonr(top_trials[i], top_trials[j])[0]
                        else:
                            corr_matrix.loc[i, j] = np.nan
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=f"指標相關性熱力圖 ({source})",
                    width=600,
                    height=600
                )
                fig.update_traces(hovertemplate="%{x}<br>%{y}<br>相關性: %{z:.2f}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"指標相關性分析失敗 (數據源: {source})：{e}")
                st.warning(f"數據源 {source} 的指標相關性分析失敗。")

            stress_mdds = pd.to_numeric(top_trials["stress_mdd"], errors="coerce").dropna()
            if not stress_mdds.empty:
                st.write(f"壓力 MDD 平均值: {np.mean(stress_mdds):.3f}, 標準差: {np.std(stress_mdds):.3f}")
            else:
                st.warning(f"數據源 {source} 無有效的壓力 MDD 數據。")

            params_to_test = ["linlen", "smaalen", "buy_mult"]
            sensitivity = {}
            for param in params_to_test:
                param_col = f"param_{param}"
                if param_col in top_trials.columns:
                    values = top_trials[param_col].dropna().unique()
                    sens_scores = []
                    for v in values:
                        subset = top_trials[top_trials[param_col] == v]
                        sens_scores.append(np.mean(pd.to_numeric(subset["score"], errors="coerce")))
                    sensitivity[param] = {v: s for v, s in zip(values, sens_scores) if not np.isnan(s)}
            st.write("參數敏感性分析:", sensitivity)
        else:
            st.warning(f"數據源 {source} 無可用試驗結果數據。")

    # 7. 圖表顯示
    elif page == "圖表顯示":
        st.subheader(f"圖表與視覺化顯示 (數據源: {source})")
        if plot_files_for_source:
            for plot_file in plot_files_for_source:
                st.image(str(plot_file), caption=plot_file.name, use_container_width=True)
        else:
            st.warning(f"數據源 {source} 無可用圖表文件。")

# 調試輸出
st.write("Results Metadata:", {f.name: results_metadata[f] for f in results_files})
st.write("Corr Metadata:", {f.name: corr_metadata[f] for f in corr_files})
st.write("Plot Metadata:", {f.name: plot_metadata[f] for f in plot_files})
st.write("Corr File Mapping:", {f.name: src for f, src in corr_file_mapping.items()})

# 重跑按鈕
if st.sidebar.button("重新執行分析"):
    st.experimental_rerun()