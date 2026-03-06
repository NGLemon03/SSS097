
# test_smart_leverage_v2.py
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

def calculate_equity_with_cost(df, w_col, r_target_col, r_safe_col):
    """
    嚴格計算含成本的權益曲線
    df: 包含 w, r_target, r_safe 的 DataFrame
    """
    # 初始化
    cash = 1_000_000.0
    equity_curve = [cash]
    
    # 上一期的權重 (初始假設為 0)
    w_prev = 0.0
    
    # 為了加速，轉成 numpy array
    w_arr = df[w_col].values
    r_target_arr = df[r_target_col].values
    r_safe_arr = df[r_safe_col].values
    
    # 逐日計算
    for i in range(len(df)):
        # 1. 計算今日目標持倉金額
        current_equity = equity_curve[-1]
        
        target_val_target = current_equity * w_arr[i]     # 00631L 目標金額
        target_val_safe = current_equity * (1 - w_arr[i]) # 0050 目標金額
        
        # 2. 計算昨日持倉在今日的「調整前」權重
        # (這裡簡化處理：假設每天開盤都重新平衡，實際上 turnover = delta w)
        
        # 計算權重變化量 (Turnover)
        # 這是我們要買賣的比例
        delta_w = w_arr[i] - w_prev
        
        # 3. 計算交易成本
        cost = 0.0
        
        if abs(delta_w) > 0.0001: # 有顯著變動才算
            turnover_amt = abs(delta_w) * current_equity
            
            # 手續費 (買賣都要)
            cost += turnover_amt * FEE_RATE
            
            # 交易稅 (只有賣出要)
            # 如果 delta_w > 0: 買 Target, 賣 Safe -> Safe 要付稅
            # 如果 delta_w < 0: 賣 Target, 買 Safe -> Target 要付稅
            cost += turnover_amt * TAX_RATE 

        # 4. 扣除成本後的權益
        post_cost_equity = current_equity - cost
        
        # 5. 計算今日投資報酬
        # 00631L 賺的 + 0050 賺的
        daily_profit = (post_cost_equity * w_arr[i] * r_target_arr[i]) + \
                       (post_cost_equity * (1 - w_arr[i]) * r_safe_arr[i])
        
        final_equity = post_cost_equity + daily_profit
        equity_curve.append(final_equity)
        
        # 更新上一期權重
        w_prev = w_arr[i]

    # 移除初始資金，對齊長度
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
    df_strat = pd.read_csv(strat_file, parse_dates=['date']).set_index('date').sort_index()
    
    # 2. 準備市場數據
    df_target, _ = load_data(TARGET_ETF, start_date=START_DATE)
    
    safe_file = Path(f"data/{SAFE_ETF.replace(':','_')}_data_raw.csv")
    if not safe_file.exists():
        df_safe = yf.download(SAFE_ETF, start=START_DATE, auto_adjust=True)
        df_safe.to_csv(safe_file)
        df_safe = pd.read_csv(safe_file, index_col=0, parse_dates=True)
        df_safe.columns = [c.lower() for c in df_safe.columns]
    else:
        df_safe, _ = load_data(SAFE_ETF, start_date=START_DATE)

    # 3. 對齊與合併
    common_idx = df_strat.index.intersection(df_target.index).intersection(df_safe.index)
    
    df_sim = pd.DataFrame(index=common_idx)
    df_sim['w'] = df_strat.loc[common_idx, 'w']
    df_sim['r_target'] = df_target.loc[common_idx, 'close'].pct_change().fillna(0)
    df_sim['r_safe'] = df_safe.loc[common_idx, 'close'].pct_change().fillna(0)
    df_sim['r_cash'] = 0.0 # 現金報酬率
    
    # 4. 模擬三種情境
    print("🔄 計算中...")
    
    # A. B&H 00631L (只算持有，不算交易成本，因為是 Buy & Hold)
    equity_bh = (1 + df_sim['r_target']).cumprod()
    
    # B. 原策略 (Cash) - 含成本
    # 這裡 w 是 00631L 權重, (1-w) 是現金
    # 我們要把 r_safe 換成 r_cash (0)
    # 但要記得：現金不需要頻繁買賣成本嗎？
    # 其實「轉現金」=「賣股票」，要付稅；「買股票」要付費。
    # 為了公平，我們用同樣的成本函數，但把 Safe 的報酬率設為 0
    # 注意：這裡假設現金切換也要成本（因為是從股票變現），這樣比較嚴格
    df_sim_cash = df_sim.copy()
    df_sim_cash['r_safe'] = 0.0 
    equity_orig = calculate_equity_with_cost(df_sim_cash, 'w', 'r_target', 'r_safe')
    
    # C. 新策略 (Smart 0050) - 含成本
    equity_smart = calculate_equity_with_cost(df_sim, 'w', 'r_target', 'r_safe')

    # 5. 統計與顯示
    def get_stats(equity):
        if equity.empty: return 0, 0
        total_ret = equity.iloc[-1] - 1
        dd = equity / equity.cummax() - 1
        mdd = dd.min()
        return total_ret, mdd

    ret_bh, mdd_bh = get_stats(equity_bh)
    ret_orig, mdd_orig = get_stats(equity_orig)
    ret_smart, mdd_smart = get_stats(equity_smart)

    print("\n" + "="*60)
    print(f"📊 含成本模擬報告 ({common_idx[0].date()} ~ {common_idx[-1].date()})")
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