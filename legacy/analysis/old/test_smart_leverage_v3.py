# test_smart_leverage_v3.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def ensure_naive(df):
    """強制將索引轉為無時區 (Naive) 的 datetime"""
    if df is None or df.empty: return df
    df.index = pd.to_datetime(df.index, errors='coerce')
    # 如果有時區，移除時區
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def calculate_equity_with_cost(df, w_col, r_target_col, r_safe_col):
    """嚴格計算含成本的權益曲線"""
    cash = 1_000_000.0
    equity_curve = [cash]
    w_prev = 0.0
    
    w_arr = df[w_col].values
    r_target_arr = df[r_target_col].values
    r_safe_arr = df[r_safe_col].values
    
    for i in range(len(df)):
        current_equity = equity_curve[-1]
        
        # 權重變化量 (Turnover)
        delta_w = w_arr[i] - w_prev
        cost = 0.0
        
        if abs(delta_w) > 0.0001:
            turnover_amt = abs(delta_w) * current_equity
            cost += turnover_amt * FEE_RATE
            cost += turnover_amt * TAX_RATE 

        post_cost_equity = current_equity - cost
        
        # 計算今日投資報酬
        daily_profit = (post_cost_equity * w_arr[i] * r_target_arr[i]) + \
                       (post_cost_equity * (1 - w_arr[i]) * r_safe_arr[i])
        
        final_equity = post_cost_equity + daily_profit
        equity_curve.append(final_equity)
        w_prev = w_arr[i]

    return pd.Series(equity_curve[1:], index=df.index)

def main():
    print(f"🚀 開始驗證「動態槓桿 (含成本)」策略...")
    print(f"💰 設定: 手續費 {FEE_RATE*10000:.1f} bp, 交易稅 {TAX_RATE*1000:.1f} ‰")

    # 1. 讀取策略權重
    target_files = list(Path("sss_backtest_outputs").glob("*Ensemble_Proportional*.csv"))
    if not target_files:
        print("❌ 找不到 Ensemble_Proportional")
        return
    strat_file = sorted(target_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"📄 讀取策略: {strat_file.name}")
    
    df_strat = pd.read_csv(strat_file, parse_dates=['date']).set_index('date')
    df_strat = ensure_naive(df_strat).sort_index()
    print(f"   👉 策略日期範圍: {df_strat.index[0].date()} ~ {df_strat.index[-1].date()} (共 {len(df_strat)} 天)")

    # 2. 準備市場數據
    print("📥 載入市場數據...")
    
    # Target (00631L)
    df_target, _ = load_data(TARGET_ETF, start_date=START_DATE)
    df_target = ensure_naive(df_target)
    
    if df_target.empty:
        print(f"❌ 無法載入 {TARGET_ETF}")
        return
    print(f"   👉 {TARGET_ETF} 範圍: {df_target.index[0].date()} ~ {df_target.index[-1].date()}")

    # Safe (0050)
    safe_file = Path(f"data/{SAFE_ETF.replace(':','_')}_data_raw.csv")
    if not safe_file.exists():
        print(f"   ⬇️ 下載 {SAFE_ETF}...")
        df_safe = yf.download(SAFE_ETF, start=START_DATE, auto_adjust=True, progress=False)
        df_safe.to_csv(safe_file)
        # 重新讀取以統一格式
        df_safe = pd.read_csv(safe_file, index_col=0)
    else:
        df_safe = pd.read_csv(safe_file, index_col=0)
    
    # 統一欄位與格式
    df_safe.columns = [c.lower() for c in df_safe.columns]
    df_safe = ensure_naive(df_safe)
    print(f"   👉 {SAFE_ETF}    範圍: {df_safe.index[0].date()} ~ {df_safe.index[-1].date()}")

    # 3. 對齊與合併
    common_idx = df_strat.index.intersection(df_target.index).intersection(df_safe.index)
    
    if len(common_idx) == 0:
        print("\n❌ 錯誤：日期完全沒有交集！請檢查上述日期範圍。")
        return

    print(f"\n✅ 共同交易日: {len(common_idx)} 天 ({common_idx[0].date()} ~ {common_idx[-1].date()})")
    
    df_sim = pd.DataFrame(index=common_idx)
    df_sim['w'] = df_strat.loc[common_idx, 'w']
    df_sim['r_target'] = df_target.loc[common_idx, 'close'].pct_change().fillna(0)
    df_sim['r_safe'] = df_safe.loc[common_idx, 'close'].pct_change().fillna(0)
    
    # 4. 模擬
    print("🔄 計算中...")
    
    # A. B&H 00631L
    equity_bh = (1 + df_sim['r_target']).cumprod()
    
    # B. 原策略 (Cash) - 含成本 (將 Safe 報酬設為 0 來模擬)
    df_sim_cash = df_sim.copy()
    df_sim_cash['r_safe'] = 0.0 
    equity_orig = calculate_equity_with_cost(df_sim_cash, 'w', 'r_target', 'r_safe')
    
    # C. 新策略 (Smart 0050) - 含成本
    equity_smart = calculate_equity_with_cost(df_sim, 'w', 'r_target', 'r_safe')

    # 5. 統計
    def get_stats(equity):
        if equity.empty: return 0, 0
        total_ret = equity.iloc[-1] - 1
        dd = equity / equity.cummax() - 1
        mdd = dd.min()
        return total_ret, mdd

    ret_bh, mdd_bh = get_stats(equity_bh)
    ret_orig, mdd_orig = get_stats(equity_orig)
    ret_smart, mdd_smart = get_stats(equity_smart)

    print("-" * 60)
    print(f"{'策略模式':<20} | {'總報酬 %':<10} | {'MDD %':<10}")
    print("-" * 60)
    print(f"{'1. B&H (00631L)':<20} | {ret_bh*100:8.2f}% | {mdd_bh*100:8.2f}%")
    print(f"{'2. 原策略 (Cash)':<20} | {ret_orig*100:8.2f}% | {mdd_orig*100:8.2f}%")
    print(f"{'3. 新策略 (0050)':<20} | {ret_smart*100:8.2f}% | {mdd_smart*100:8.2f}%")
    print("="*60)
    
    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_bh.index, y=equity_bh, name="B&H (00631L)", line=dict(color='gray', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=equity_orig.index, y=equity_orig, name="原策略 (Cash)", line=dict(color='#EF553B', width=2)))
    fig.add_trace(go.Scatter(x=equity_smart.index, y=equity_smart, name="新策略 (Smart 0050)", line=dict(color='#00CC96', width=3)))
    
    fig.update_layout(title="動態槓桿 (含成本) 驗證結果", template="plotly_dark", hovermode="x unified", height=600)
    fig.show()

if __name__ == "__main__":
    main()