# run_oos_analysis.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import argparse
from pathlib import Path

# 引用現有模組
sys.path.append(os.getcwd())
try:
    from analysis.strategy_manager import manager
    from SSS_EnsembleTab import run_ensemble, RunConfig, EnsembleParams, CostParams
    from sss_core.logic import load_data
except ImportError:
    print("❌ 請在專案根目錄執行")
    sys.exit(1)

# ================= 設定區 =================
TICKER = "00631L.TW"
DEFAULT_SPLIT_DATE = "2025-06-30"  # OOS 切分點

# 定義要測試的 Ensemble 組合
ENSEMBLE_CONFIGS = [
    {
        "name": "Ensemble_Proportional (穩健版)",
        "method": "proportional",
        "floor": 0.5,       # 底倉 50%
        "delta_cap": 0.1,   # 每日變動限制 10%
        "color": "#00CC96"  # 綠色
    },
    {
        "name": "Ensemble_Majority (激進版)",
        "method": "majority",
        "floor": 0.2,       # 底倉 20%
        "delta_cap": 0.3,
        "majority_k_pct": 0.55,
        "color": "#EF553B"  # 紅色
    }
]
# =========================================

def generate_data(ticker):
    """步驟 1: 生成數據 (Generation)"""
    print("🔄 [Step 1] 正在計算 Ensemble 策略數據...")
    
    # 讀取倉庫
    active_strats = manager.load_strategies()
    if not active_strats:
        print("❌ 倉庫是空的！請先執行 python init_warehouse.py --top_k 5")
        return {}

    strat_names = [s['name'].replace('.csv', '') for s in active_strats]
    
    # 建立檔案映射
    file_map = {}
    search_dirs = [Path("sss_backtest_outputs"), Path("archive")]
    
    for s_name in strat_names:
        for d in search_dirs:
            candidates = list(d.rglob(f"*{s_name}*.csv"))
            if candidates:
                file_map[s_name] = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                break
    
    if not file_map:
        print("❌ 找不到策略檔案，無法計算")
        return {}

    cost = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30)
    generated_results = {}

    for cfg_data in ENSEMBLE_CONFIGS:
        print(f"   ⚙️ 計算中: {cfg_data['name']} ...")
        
        ens_params = EnsembleParams(
            floor=cfg_data['floor'],
            ema_span=3,
            delta_cap=cfg_data['delta_cap'],
            min_cooldown_days=1,
            min_trade_dw=0.01
        )

        cfg = RunConfig(
            ticker=ticker,
            method=cfg_data['method'],
            strategies=list(file_map.keys()),
            file_map=file_map,
            params=ens_params,
            cost=cost,
            majority_k_pct=cfg_data.get('majority_k_pct')
        )

        # 執行回測
        _, _, _, _, _, _, daily_state, _ = run_ensemble(cfg)
        
        if daily_state is not None and not daily_state.empty:
            # 存入字典 (記憶體傳遞)
            generated_results[cfg_data['name']] = {
                "df": daily_state,
                "config": cfg_data
            }
            
            # 順便存檔備份
            out_file = Path("sss_backtest_outputs") / f"ensemble_daily_state_{cfg_data['name'].split()[0]}.csv"
            daily_state.to_csv(out_file)

    return generated_results

def analyze_performance(results_dict, ticker, split_date):
    """步驟 2: 分析與畫圖 (Analysis)"""
    print(f"\n📊 [Step 2] 正在分析 OOS 表現 (基準日: {split_date})...")

    # 載入大盤
    df_raw, _ = load_data(ticker)
    mask_oos = df_raw.index > split_date
    df_oos = df_raw.loc[mask_oos].copy()
    
    if df_oos.empty:
        print("❌ OOS 區段無大盤數據")
        return

    # B&H 歸一化
    bh_curve = df_oos['close'] / df_oos['close'].iloc[0]
    bh_dd = (bh_curve / bh_curve.cummax() - 1)

    # 繪圖設定
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=(f"OOS 實質報酬 (Base=1.0 @ {split_date})", "回撤幅度 (Drawdown)"))

    # 畫大盤
    fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve, name="Buy & Hold", 
                             line=dict(color='gray', width=2, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="B&H DD",
                             line=dict(color='gray', width=1), showlegend=False), row=2, col=1)

    summary_stats = []

    # 畫策略
    for name, data in results_dict.items():
        df = data['df']
        config = data['config']
        
        # 截取 OOS
        df['date'] = pd.to_datetime(df.index) # 確保有日期
        strat_oos = df.loc[df.index > split_date].copy()
        
        if strat_oos.empty: continue

        # 歸一化
        strat_curve = strat_oos['equity'] / strat_oos['equity'].iloc[0]
        strat_dd = (strat_curve / strat_curve.cummax() - 1)

        # 畫圖
        fig.add_trace(go.Scatter(x=strat_curve.index, y=strat_curve, name=name,
                                 line=dict(color=config['color'], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, name=f"{name} DD",
                                 line=dict(color=config['color'], width=1), fill='tozeroy'), row=2, col=1)

        # 統計
        tot_ret = strat_curve.iloc[-1] - 1
        max_dd = strat_dd.min()
        bh_ret = bh_curve.iloc[-1] - 1
        
        summary_stats.append({
            "Strategy": name,
            "Return": f"{tot_ret:.2%}",
            "MDD": f"{max_dd:.2%}",
            "Alpha": f"{tot_ret - bh_ret:.2%}"
        })

    # 標示 0805 大跌 (🔥 這裡做了關鍵修正：轉成數值時間戳)
    crash_date_str = "2024-08-05"
    if pd.Timestamp(crash_date_str) > pd.Timestamp(split_date):
        # 轉成毫秒時間戳 (epoch)，確保 Plotly 能讀懂
        crash_ts = pd.Timestamp(crash_date_str).timestamp() * 1000 
        fig.add_vline(x=crash_ts, line_dash="dash", line_color="red", annotation_text="0805 Crash")

    fig.update_layout(title=f"OOS 驗證報告: {ticker}", template="plotly_dark", height=800, hovermode="x unified")
    fig.show()

    # 印出表格
    print("\n" + "="*80)
    print(f"📋 OOS 績效摘要 ({split_date} ~ Today)")
    print(f"📉 B&H 基準: 報酬 {bh_curve.iloc[-1]-1:.2%} | MDD {bh_dd.min():.2%}")
    print("-" * 80)
    print(f"{'策略名稱':<35} | {'報酬':<10} | {'MDD':<10} | {'超額報酬':<10}")
    print("-" * 80)
    for s in summary_stats:
        print(f"{s['Strategy']:<35} | {s['Return']:<10} | {s['MDD']:<10} | {s['Alpha']:<10}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_date', type=str, default=DEFAULT_SPLIT_DATE)
    args = parser.parse_args()

    # 1. 生成數據
    results = generate_data(TICKER)
    
    # 2. 分析畫圖
    if results:
        analyze_performance(results, TICKER, args.split_date)

if __name__ == "__main__":
    main()