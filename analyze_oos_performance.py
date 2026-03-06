# analyze_oos_performance.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import argparse
from pathlib import Path
import sys
import os

# 引用現有模組
sys.path.append(os.getcwd())
try:
    from analysis.strategy_manager import manager
    from sss_core.logic import load_data, calculate_metrics
    # 嘗試匯入新的序列化工具
    from sss_core.data_utils import unpack_df_robust as df_from_pack
except ImportError:
    print("❌ 請在專案根目錄執行此腳本")
    sys.exit(1)

# ================= 設定 =================
# 預設 OOS 切分點 (請與 run_full_pipeline_3.py 保持一致)
DEFAULT_SPLIT_DATE = "2023-12-31" 
TICKER = "00631L.TW"
# =======================================

def load_active_strategies():
    """從倉庫讀取現役策略"""
    warehouse = Path("analysis/strategy_warehouse.json")
    if not warehouse.exists():
        print("❌ 找不到策略倉庫，請先執行 init_warehouse.py")
        return []
    
    strategies = manager.load_strategies()
    print(f"📂 載入現役策略: {len(strategies)} 個")
    return strategies

def get_strategy_equity(strat_name):
    """讀取策略的權益曲線"""
    # 嘗試在 outputs 找
    files = list(Path("sss_backtest_outputs").glob(f"*{strat_name}*.csv"))
    # 如果找不到，去 archive 找
    if not files:
        files = list(Path("archive").rglob(f"*{strat_name}*.csv"))
    
    if not files:
        return None
    
    # 找最新的
    target_file = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    try:
        df = pd.read_csv(target_file)
        # 如果是交易檔，需要重建權益曲線 (簡易版)
        # 這裡簡化處理：假設我們有 daily_state 或者我們用 benchmark 模擬
        # 為了精確，我們直接讀取 run_ensemble 產出的 ensemble_equity (如果是組合)
        # 或是重新回測 (太慢)。
        
        # 這裡我們假設使用者主要想看 Ensemble 的表現
        # 如果是單策略，我們用簡易模擬
        return df # 這裡回傳原始 DF，稍後處理
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_date', type=str, default=DEFAULT_SPLIT_DATE, help='訓練截止日 (OOS 開始日)')
    parser.add_argument('--ticker', type=str, default=TICKER)
    args = parser.parse_args()

    print(f"🔍 分析目標: {args.ticker}")
    print(f"📅 OOS 開始日: {args.split_date} (以此日為基準歸零重算)")

    # 1. 載入大盤 (B&H)
    df_raw, _ = load_data(args.ticker)
    if df_raw.empty:
        print("❌ 無法載入大盤數據")
        return
    
    # 截取 OOS 區段
    mask_oos = df_raw.index > args.split_date
    df_oos = df_raw.loc[mask_oos].copy()
    
    if df_oos.empty:
        print("❌ OOS 區段無數據，請檢查日期設定")
        return

    # B&H 歸一化 (Rebase to 1.0)
    bh_curve = df_oos['close'] / df_oos['close'].iloc[0]
    
    # 2. 載入策略 (這裡示範載入 Ensemble 的結果)
    # 我們直接去讀 sss_backtest_outputs 裡面的 ensemble 檔案
    # 因為通常我們最關心的是最終組合
    ens_files = list(Path("sss_backtest_outputs").glob("ensemble_daily_state_*.csv"))
    
    if not ens_files:
        print("⚠️ 找不到 Ensemble 結果，請先執行 Dash 或 run_enhanced_ensemble")
        # 嘗試讀取單策略... (略，為保持簡潔，先專注於 Ensemble)
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=("OOS 實質報酬比較 (歸零重算)", "OOS 期間回撤比較"))

    # 畫大盤
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve, name="Buy & Hold", 
                             line=dict(color='gray', width=2, dash='dot')), row=1, col=1)
    
    # 計算大盤回撤
    bh_dd = (bh_curve / bh_curve.cummax() - 1)
    fig.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="B&H DD",
                             line=dict(color='gray', width=1), showlegend=False), row=2, col=1)

    colors = ['#00CC96', '#EF553B', '#AB63FA', '#FFA15A']
    
    summary_stats = []

    for i, f in enumerate(ens_files):
        try:
            df = pd.read_csv(f)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 截取 OOS
            strat_oos = df.loc[df.index > args.split_date].copy()
            if strat_oos.empty: continue
            
            # 策略歸一化
            strat_curve = strat_oos['equity'] / strat_oos['equity'].iloc[0]
            
            # 策略回撤
            strat_dd = (strat_curve / strat_curve.cummax() - 1)
            
            name = f.stem.replace("ensemble_daily_state_", "")
            color = colors[i % len(colors)]
            
            # 繪圖
            fig.add_trace(go.Scatter(x=strat_curve.index, y=strat_curve, name=f"策略: {name}",
                                     line=dict(color=color, width=2)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, name=f"{name} DD",
                                     line=dict(color=color, width=1), fill='tozeroy'), row=2, col=1)

            # 統計數據
            total_ret = strat_curve.iloc[-1] - 1
            max_dd = strat_dd.min()
            bh_ret = bh_curve.iloc[-1] - 1
            bh_mdd = bh_dd.min()
            
            summary_stats.append({
                "Strategy": name,
                "OOS Return": f"{total_ret:.2%}",
                "OOS MDD": f"{max_dd:.2%}",
                "Alpha (vs B&H)": f"{total_ret - bh_ret:.2%}",
                "Risk Reduct": f"{max_dd - bh_mdd:.2%}" # 負越多越好
            })

        except Exception as e:
            print(f"讀取 {f.name} 失敗: {e}")

    # 標示重大事件 (例如 2024/08/05)
    crash_date = "2024-08-05"
    if pd.Timestamp(crash_date) > pd.Timestamp(args.split_date):
        fig.add_vline(x=crash_date, line_dash="dash", line_color="red", annotation_text="0805大跌")

    fig.update_layout(title=f"OOS 實戰表現檢驗 (起點: {args.split_date})",
                      hovermode="x unified", template="plotly_dark", height=800)
    
    # 顯示圖表
    fig.show()
    
    # 打印報告
    print("\n" + "="*60)
    print(f"📊 OOS 期間表現報告 ({args.split_date} ~ Today)")
    print(f"📉 B&H 基準: 報酬 {bh_curve.iloc[-1]-1:.2%} | MDD {bh_dd.min():.2%}")
    print("-" * 60)
    print(f"{'策略名稱':<25} | {'報酬':<10} | {'MDD':<10} | {'超額報酬':<10}")
    print("-" * 60)
    for s in summary_stats:
        print(f"{s['Strategy']:<25} | {s['OOS Return']:<10} | {s['OOS MDD']:<10} | {s['Alpha (vs B&H)']:<10}")
    print("="*60)

if __name__ == "__main__":
    main()