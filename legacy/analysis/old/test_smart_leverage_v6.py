# test_smart_leverage_v6.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import os
import sys
from pathlib import Path

# 引用現有模組
sys.path.append(os.getcwd())
try:
    from sss_core.logic import load_data
except ImportError:
    print("❌ 請在專案根目錄執行")
    sys.exit(1)

# ================= 設定區 =================
TARGET_ETF = "00631L.TW"  # 攻擊型
SAFE_ETF = "0050.TW"      # 防守型
START_DATE = "2015-01-01" 

# 交易成本設定 (實盤參數)
FEE_RATE = 0.001425 * 0.3  # 手續費 (3折)
TAX_RATE = 0.001           # ETF 交易稅 (0.1%)
# =========================================

def clean_data(df):
    """資料清洗 (V6)"""
    if df is None or df.empty: return df
    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if 'close' in df.columns:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
    if 'w' in df.columns:
        df['w'] = pd.to_numeric(df['w'], errors='coerce')
        
    df = df[df.index.notnull()]
    return df

def calculate_equity_with_cost(df, w_col, r_target_col, r_safe_col):
    """嚴格計算含成本的權益曲線 (金額版)"""
    cash = 1_000_000.0
    equity_curve = [cash]
    w_prev = 0.0
    
    w_arr = df[w_col].fillna(0).values
    r_target_arr = df[r_target_col].fillna(0).values
    r_safe_arr = df[r_safe_col].fillna(0).values
    
    for i in range(len(df)):
        current_equity = equity_curve[-1]
        delta_w = w_arr[i] - w_prev
        cost = 0.0
        
        if abs(delta_w) > 0.0001:
            turnover_amt = abs(delta_w) * current_equity
            cost += turnover_amt * FEE_RATE
            cost += turnover_amt * TAX_RATE 

        post_cost_equity = current_equity - cost
        daily_profit = (post_cost_equity * w_arr[i] * r_target_arr[i]) + \
                       (post_cost_equity * (1 - w_arr[i]) * r_safe_arr[i])
        
        final_equity = post_cost_equity + daily_profit
        equity_curve.append(final_equity)
        w_prev = w_arr[i]

    return pd.Series(equity_curve[1:], index=df.index)

def get_stats(equity):
    """🔥 V6 修復：強制歸一化計算，避免金額單位誤差"""
    if equity.empty: return 0, 0
    
    # 關鍵：全部除以第一天的值，讓起點變成 1.0
    norm_equity = equity / equity.iloc[0]
    
    total_ret = norm_equity.iloc[-1] - 1
    dd = norm_equity / norm_equity.cummax() - 1
    mdd = dd.min()
    return total_ret, mdd

def main():
    print(f"🚀 開始驗證「動態槓桿 (含成本 V6 - 歸一化修復版)」策略...")
    print(f"💰 設定: 手續費 {FEE_RATE*10000:.1f} bp, 交易稅 {TAX_RATE*1000:.1f} ‰")

    # 1. 讀取策略權重
    target_files = list(Path("sss_backtest_outputs").glob("*Ensemble_Proportional*.csv"))
    if not target_files:
        print("❌ 找不到 Ensemble_Proportional")
        return
    strat_file = sorted(target_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"📄 讀取策略: {strat_file.name}")
    
    df_strat = pd.read_csv(strat_file, index_col=0)
    df_strat = clean_data(df_strat)
    print(f"   👉 策略有效日期: {df_strat.index[0].date()} ~ {df_strat.index[-1].date()}")

    # 2. 準備市場數據
    print("📥 載入市場數據...")
    df_target, _ = load_data(TARGET_ETF, start_date=START_DATE)
    df_target = clean_data(df_target)
    
    safe_file = Path(f"data/{SAFE_ETF.replace(':','_')}_data_raw.csv")
    if not safe_file.exists():
        df_safe = yf.download(SAFE_ETF, start=START_DATE, auto_adjust=True, progress=False)
        df_safe.to_csv(safe_file)
        df_safe = pd.read_csv(safe_file, index_col=0)
    else:
        df_safe = pd.read_csv(safe_file, index_col=0)
    df_safe = clean_data(df_safe)
    
    # 3. 對齊與合併
    common_idx = df_strat.index.intersection(df_target.index).intersection(df_safe.index)
    if len(common_idx) == 0:
        print("❌ 日期無交集")
        return
    print(f"✅ 共同交易日: {len(common_idx)} 天")
    
    df_sim = pd.DataFrame(index=common_idx)
    df_sim['w'] = df_strat.loc[common_idx, 'w']
    df_sim['r_target'] = df_target.loc[common_idx, 'close'].pct_change().fillna(0)
    df_sim['r_safe'] = df_safe.loc[common_idx, 'close'].pct_change().fillna(0)
    
    # 4. 模擬
    print("🔄 計算中...")
    
    # A. B&H 00631L
    equity_bh = (1 + df_sim['r_target']).cumprod()
    
    # B. 原策略 (Cash)
    df_sim_cash = df_sim.copy()
    df_sim_cash['r_safe'] = 0.0 
    equity_orig = calculate_equity_with_cost(df_sim_cash, 'w', 'r_target', 'r_safe')
    
    # C. 新策略 (Smart 0050)
    equity_smart = calculate_equity_with_cost(df_sim, 'w', 'r_target', 'r_safe')

    # 5. 統計與顯示
    ret_bh, mdd_bh = get_stats(equity_bh)
    ret_orig, mdd_orig = get_stats(equity_orig)
    ret_smart, mdd_smart = get_stats(equity_smart)

    print("\n" + "="*60)
    print(f"📊 最終驗證報告 ({common_idx[0].date()} ~ {common_idx[-1].date()})")
    print("-" * 60)
    print(f"{'策略模式':<20} | {'總報酬 %':<10} | {'MDD %':<10}")
    print("-" * 60)
    print(f"{'1. B&H (00631L)':<20} | {ret_bh*100:8.2f}% | {mdd_bh*100:8.2f}%")
    print(f"{'2. 原策略 (Cash)':<20} | {ret_orig*100:8.2f}% | {mdd_orig*100:8.2f}%")
    print(f"{'3. 新策略 (0050)':<20} | {ret_smart*100:8.2f}% | {mdd_smart*100:8.2f}%")
    print("="*60)
    
    # 畫圖 (全部歸一化後再畫，視覺才正確)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_bh.index, y=equity_bh/equity_bh.iloc[0], name="B&H (00631L)", line=dict(color='gray', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=equity_orig.index, y=equity_orig/equity_orig.iloc[0], name="原策略 (Cash)", line=dict(color='#EF553B', width=2)))
    fig.add_trace(go.Scatter(x=equity_smart.index, y=equity_smart/equity_smart.iloc[0], name="新策略 (Smart 0050)", line=dict(color='#00CC96', width=3)))
    
    fig.update_layout(title="動態槓桿驗證 (歸一化)", template="plotly_dark", hovermode="x unified", height=600)
    fig.show()

if __name__ == "__main__":
    main()