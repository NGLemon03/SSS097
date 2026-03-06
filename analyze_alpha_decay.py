# analyze_alpha_decay.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path
import sys
import os  # <--- [修復1] 補上漏掉的 os

# 引用現有模組
sys.path.append(os.getcwd())
try:
    from sss_core.logic import load_data
    from analysis.strategy_manager import manager  # 🔥 引入 manager
except ImportError:
    print("❌ 請在專案根目錄執行")
    sys.exit(1)

# 預設設定
TICKER = "00631L.TW"
# 移除預設日期，改為自動偵測
# DEFAULT_SPLIT_DATE = "2025-06-30" 

def main():
    parser = argparse.ArgumentParser()
    # 🔥 將 split_date 設為選填，預設為 None
    parser.add_argument('--split_date', type=str, default=None, help='如果不填，自動從倉庫讀取')
    args = parser.parse_args()

    # 🔥 自動讀取 Metadata
    split_date = args.split_date
    if not split_date:
        meta = manager.load_metadata()
        split_date = meta.get("split_date")
        score_mode = meta.get("score_mode", "Unknown")
        mode = meta.get("mode", "Unknown")
        print(f"ℹ️  從倉庫讀取設定: 訓練截止日={split_date}, 模式={mode}, 評分={score_mode}")

    if not split_date:
        print("❌ 無法自動偵測 Split Date，請手動輸入 --split_date")
        return

    print(f"🕵️‍♀️ 開始分析 Alpha 衰退 (基準日: {split_date})...")

    # 1. 讀取大盤 (Benchmark)
    df_raw, _ = load_data(TICKER)
    mask_oos = df_raw.index > split_date
    bench = df_raw.loc[mask_oos, 'close'].copy()
    
    if bench.empty:
        print("❌ OOS 區間無數據")
        return

    # 2. 讀取 Ensemble 結果
    # 我們這裡讀取比例法 (Proportional) 的結果，通常較具代表性
    target_files = list(Path("sss_backtest_outputs").glob("*Ensemble_Proportional*.csv"))

    if not target_files:
        print("❌ 找不到 Ensemble_Proportional 結果，請先執行 run_oos_analysis.py 生成數據")
        return

    # 取最新的檔案
    strat_file = sorted(target_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"📄 讀取策略檔: {strat_file.name}")

    df_strat = pd.read_csv(strat_file)
    df_strat['date'] = pd.to_datetime(df_strat.iloc[:, 0] if 'date' not in df_strat.columns else df_strat['date'])
    df_strat = df_strat.set_index('date')

    # 截取與對齊
    strat = df_strat.loc[df_strat.index > split_date, 'equity'].copy()
    
    # 確保兩者長度一致 (取交集)
    common_idx = bench.index.intersection(strat.index)
    bench = bench.loc[common_idx]
    strat = strat.loc[common_idx]

    if strat.empty:
        print("❌ 策略數據與大盤數據無交集 (日期對不上)")
        return

    # 3. 計算累計超額報酬 (Cumulative Alpha)
    # 歸一化
    bench_norm = bench / bench.iloc[0]
    strat_norm = strat / strat.iloc[0]
    
    # Alpha = 策略累積報酬 - 大盤累積報酬
    cumulative_alpha = (strat_norm - 1) - (bench_norm - 1)

    # 4. 計算滾動 Alpha (Rolling Win Rate)
    rolling_window = 60
    strat_ret_roll = strat.pct_change(rolling_window)
    bench_ret_roll = bench.pct_change(rolling_window)
    rolling_alpha = strat_ret_roll - bench_ret_roll

    # 5. 繪圖
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=(
                            "1. 策略 vs 大盤 (歸一化)", 
                            "2. 累計超額報酬 (Alpha Curve)", 
                            f"3. 滾動 {rolling_window} 天超額績效 (短期動能)"
                        ))

    # 子圖 1: 走勢比較
    fig.add_trace(go.Scatter(x=strat_norm.index, y=strat_norm, name="策略 (Strategy)", line=dict(color='#00CC96', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name="大盤 (00631L)", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)

    # 子圖 2: Alpha Curve
    fig.add_hline(y=0, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Scatter(x=cumulative_alpha.index, y=cumulative_alpha, name="累計 Alpha", 
                             line=dict(color='#FFA15A', width=2), fill='tozeroy'), row=2, col=1)

    # 子圖 3: 滾動檢驗
    colors = np.where(rolling_alpha > 0, 'green', 'red')
    fig.add_trace(go.Bar(x=rolling_alpha.index, y=rolling_alpha, name="滾動 Alpha", marker_color=colors), row=3, col=1)

    # 標註建議重練點 (Alpha Curve 顯著反轉點)
    peak_alpha = cumulative_alpha.cummax()
    drawdown_from_peak = cumulative_alpha - peak_alpha

    # 簡單規則：如果 Alpha 從高點回落超過 10%，標記為「失效警示」
    decay_points = drawdown_from_peak[drawdown_from_peak < -0.10]
    if not decay_points.empty:
        first_decay = decay_points.index[0]
        days_lasted = (first_decay - pd.Timestamp(split_date)).days

        # <--- [修復2] 關鍵修正：轉成數值時間戳 --->
        first_decay_ts = pd.Timestamp(first_decay).timestamp() * 1000

        fig.add_vline(x=first_decay_ts, line_color="red", line_dash="dash",
                      annotation_text=f"建議重練點 (第 {days_lasted} 天)", row=2, col=1)

    fig.update_layout(title=f"策略生命週期分析 (起點: {split_date})", template="plotly_dark", height=900, hovermode="x unified")
    fig.show()

    # 6. 文字報告
    print("\n" + "="*60)
    print("📊 策略生命週期診斷報告")
    print("-" * 60)
    
    # 計算半衰期 (Half-Life) 概念
    alpha_peaks = cumulative_alpha[(cumulative_alpha == cumulative_alpha.cummax())]
    if len(alpha_peaks) > 1:
        last_peak = alpha_peaks.index[-1]
        days_since_peak = (pd.Timestamp.now() - last_peak).days
        valid_duration = (last_peak - pd.Timestamp(split_date)).days

        print(f"✅ 有效增長期: {valid_duration} 天 (Alpha 持續創新高直到 {last_peak.date()})")
        print(f"⚠️ 停滯期: 最近 {days_since_peak} 天 Alpha 未創新高")

        if days_since_peak > 180:
            print("🔴 結論: 策略可能已失效，建議立即重新訓練！")
        elif days_since_peak > 90:
            print("🟡 結論: 策略動能減弱，請密切觀察或準備重練。")
        else:
            print("🟢 結論: 策略仍處於強勢期，暫無需重練。")
    else:
        print("⚪ 數據不足或 Alpha 從未轉正。")

    print("="*60)

if __name__ == "__main__":
    main()