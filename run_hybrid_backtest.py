# run_hybrid_backtest.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# 引用現有模組
sys.path.append(os.getcwd())
try:
    from sss_core.logic import load_data
except ImportError:
    print("❌ 請在專案根目錄執行")
    sys.exit(1)

# ================= 設定區 =================
TICKER = "00631L.TW"
START_DATE = "2015-01-01"

# 1. 趨勢策略來源 (Ensemble)
# 程式會自動抓最新的 Ensemble 檔案，你不用改

# 2. 逆勢策略設定 (Buy the Dip)
DIP_TRIGGER = -0.30      # 高點回落 20% 開始接刀
DIP_ADD_WEIGHT = 0.5     # 加碼 50% (例如原本持有 100張，再買 50張)
DIP_EXIT_PROFIT = 0.15   # 抄底部位賺 15% 就跑 (短打)
DIP_MAX_HOLD = 60        # 最多抱 60 天 (時間到就停損/停利)

# 3. 資金設定
INITIAL_CASH = 1_000_000
LOAN_RATE = 0.03 / 365  # 質押年利率 3% (日息)
MAX_TOTAL_WEIGHT = 2.0   # 風控上限：總持倉不能超過 200% (自路 + 融資)
# =========================================

def main():
    print(f"🚀 開始執行混合策略回測 (Hybrid: Trend + Dip)...")
    
    # 1. 載入市場數據
    df_raw, _ = load_data(TICKER, start_date=START_DATE)
    close = df_raw['close']
    
    # 2. 載入 Ensemble 權重 (趨勢策略)
    #target_files = list(Path("sss_backtest_outputs").glob("*Ensemble_Proportional*.csv"))
    target_files = list(Path("sss_backtest_outputs").glob("*Ensemble_Majority*.csv")) # 想測 Majority 改這行
    
    if not target_files:
        print("❌ 找不到 Ensemble 結果，請先執行 run_oos_analysis.py 或 generate_ensemble_outputs.py")
        return
    
    strat_file = sorted(target_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"📄 讀取趨勢策略: {strat_file.name}")
    
    df_strat = pd.read_csv(strat_file)
    df_strat['date'] = pd.to_datetime(df_strat.iloc[:, 0] if 'date' not in df_strat.columns else df_strat['date'])
    df_strat = df_strat.set_index('date').sort_index()
    
    # 對齊數據
    common_idx = close.index.intersection(df_strat.index)
    close = close.loc[common_idx]
    w_trend = df_strat.loc[common_idx, 'w'] # 趨勢策略建議的權重

    # 3. 計算歷史高點與回撤 (供 Dip 使用)
    roll_max = close.rolling(250, min_periods=1).max()
    drawdown = (close - roll_max) / roll_max

    # 4. 逐日回測
    cash = INITIAL_CASH
    shares = 0.0
    equity_curve = []
    leverage_curve = []
    
    # Dip 策略狀態
    dip_holding = False
    dip_entry_price = 0.0
    dip_entry_date = None
    dip_shares = 0.0 # 獨立計算抄底的股數

    print("⚙️  開始模擬混合交易...")
    
    for date in common_idx:
        price = close.loc[date]
        dd = drawdown.loc[date]
        
        # --- A. 計算目標權重 ---
        
        # 1. 趨勢部位 (Trend)
        target_w_trend = w_trend.loc[date]
        
        # 2. 逆勢部位 (Dip)
        w_dip = 0.0
        
        if dip_holding:
            # 檢查出場條件
            profit = (price / dip_entry_price) - 1
            days_held = (date - dip_entry_date).days
            
            if profit >= DIP_EXIT_PROFIT or days_held >= DIP_MAX_HOLD:
                # 出場
                dip_holding = False
                dip_shares = 0.0
                w_dip = 0.0 # 歸零
            else:
                # 續抱
                w_dip = DIP_ADD_WEIGHT
        else:
            # 檢查進場條件
            if dd <= DIP_TRIGGER:
                # 進場
                dip_holding = True
                dip_entry_price = price
                dip_entry_date = date
                w_dip = DIP_ADD_WEIGHT
        
        # 3. 合併權重 & 風控
        total_target_w = target_w_trend + w_dip
        total_target_w = min(total_target_w, MAX_TOTAL_WEIGHT) # 強制風控上限
        
        # --- B. 執行交易 ---
        
        # 計算當前總資產 (Equity)
        curr_equity = cash + shares * price
        
        # 計算目標市值
        target_val = curr_equity * total_target_w
        curr_val = shares * price
        
        # 需要調整的金額
        diff_val = target_val - curr_val
        
        # 簡單模擬交易 (忽略滑價與手續費以簡化邏輯，重點看結構)
        if diff_val != 0:
            shares += diff_val / price
            cash -= diff_val
            
        # --- C. 計算利息 (若有借款) ---
        if cash < 0:
            interest = abs(cash) * LOAN_RATE
            cash -= interest # 扣利息
            
        # 紀錄
        final_equity = cash + shares * price
        equity_curve.append(final_equity)
        leverage_curve.append(total_target_w) # 紀錄當下槓桿率

    # 5. 轉換為 Series
    s_equity = pd.Series(equity_curve, index=common_idx)
    s_leverage = pd.Series(leverage_curve, index=common_idx)
    
    # 基準 (B&H)
    s_bh = (close / close.iloc[0]) * INITIAL_CASH
    
    # 純 Ensemble (不加 Dip)
    # 這裡簡單重算，或者直接拿 df_strat['equity'] (如果有)
    # 為求精準比較，我們用 df_strat 裡的 equity 並歸一化到同樣起始資金
    s_ens = df_strat.loc[common_idx, 'equity']
    s_ens = (s_ens / s_ens.iloc[0]) * INITIAL_CASH

    # 6. 統計數據
    def get_stats(s):
        ret = (s.iloc[-1] / s.iloc[0]) - 1
        dd = s / s.cummax() - 1
        mdd = dd.min()
        return ret, mdd

    ret_hyb, mdd_hyb = get_stats(s_equity)
    ret_ens, mdd_ens = get_stats(s_ens)
    ret_bh, mdd_bh = get_stats(s_bh)

    print("\n" + "="*80)
    print(f"🏆 混合策略PK大賽 ({START_DATE} ~ {common_idx[-1].date()})")
    print(f"   設定: 趨勢策略 + 跌{DIP_TRIGGER:.0%}加碼{DIP_ADD_WEIGHT:.1f}倍")
    print("-" * 80)
    print(f"{'策略':<20} | {'總報酬 %':<12} | {'MDD %':<10} | {'備註'}")
    print("-" * 80)
    print(f"{'1. Buy & Hold':<20} | {ret_bh*100:10.1f}% | {mdd_bh*100:8.1f}% | {'基準 (死抱)'}")
    print(f"{'2. 純趨勢 (Ensemble)':<20} | {ret_ens*100:10.1f}% | {mdd_ens*100:8.1f}% | {'會閃崩盤，但V轉追高'}")
    print(f"{'3. 混合體 (Hybrid)':<20} | {ret_hyb*100:10.1f}% | {mdd_hyb*100:8.1f}% | {'趨勢 + 抄底 (動用質押)'}")
    print("="*80)

    # 7. 繪圖
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("權益曲線比較", "回撤比較", "槓桿使用率 (Hybrid)"))

    # Row 1: Equity
    fig.add_trace(go.Scatter(x=s_equity.index, y=s_equity, name="Hybrid (混合)", line=dict(color='#00CC96', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_ens.index, y=s_ens, name="Ensemble (純趨勢)", line=dict(color='#FFA15A', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_bh.index, y=s_bh, name="B&H (00631L)", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)

    # Row 2: Drawdown
    dd_hyb = s_equity / s_equity.cummax() - 1
    dd_ens = s_ens / s_ens.cummax() - 1
    fig.add_trace(go.Scatter(x=dd_hyb.index, y=dd_hyb, name="Hybrid DD", line=dict(color='#00CC96', width=1), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dd_ens.index, y=dd_ens, name="Ensemble DD", line=dict(color='#FFA15A', width=1)), row=2, col=1)

    # Row 3: Leverage
    fig.add_trace(go.Scatter(x=s_leverage.index, y=s_leverage, name="總槓桿率", line=dict(color='#AB63FA', width=1.5)), row=3, col=1)
    # 畫一條 1.0 的線 (原本的滿倉線)
    fig.add_hline(y=1.0, line_dash="dot", line_color="white", row=3, col=1)

    fig.update_layout(title="混合策略驗證：當趨勢策略遇上抄底外掛", template="plotly_dark", height=900, hovermode="x unified")
    fig.show()

if __name__ == "__main__":
    main()