# run_hybrid_backtest_v2.py
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

# 2. 資金與質押設定
INITIAL_CASH = 1_000_000
INITIAL_LOAN = 0         # 初始借款 (如果你一開始就想開槓桿，可設為 500000)
LOAN_RATE = 0.03 / 365  # 年利率 3%
MAINTENANCE_MARGIN = 1.30 # 維持率死線 130%
WARNING_MARGIN = 1.40     # 警戒線 140%
MAX_LEVERAGE = 2.5        # 最大總槓桿限制 (避免借太多)
# =========================================

def main():
    print(f"🚀 執行混合策略回測 (含維持率監控)...")
    
    # 1. 載入數據 & Ensemble
    df_raw, _ = load_data(TICKER, start_date=START_DATE)
    close = df_raw['close']
    # 2. 載入 Ensemble 權重 (趨勢策略)
    #target_files = list(Path("sss_backtest_outputs").glob("*Ensemble_Proportional*.csv"))
    target_files = list(Path("sss_backtest_outputs").glob("*Ensemble_Majority*.csv")) # 使用 Majority 多數決
    if not target_files:
        print("❌ 找不到 Ensemble 結果")
        return
    
    strat_file = sorted(target_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"📄 讀取策略: {strat_file.name}")
    
    df_strat = pd.read_csv(strat_file)
    df_strat['date'] = pd.to_datetime(df_strat.iloc[:, 0] if 'date' not in df_strat.columns else df_strat['date'])
    df_strat = df_strat.set_index('date').sort_index()
    
    common_idx = close.index.intersection(df_strat.index)
    close = close.loc[common_idx]
    w_trend = df_strat.loc[common_idx, 'w']

    # 2. 準備回測變數
    roll_max = close.rolling(250, min_periods=1).max()
    drawdown = (close - roll_max) / roll_max

    cash = INITIAL_CASH
    loan = INITIAL_LOAN # 負債
    shares = 0.0
    
    # 記錄器
    records = []
    
    # Dip 狀態
    dip_holding = False
    dip_entry_price = 0.0
    dip_entry_date = None

    print("⚙️  模擬中...")
    
    for date in common_idx:
        price = close.loc[date]
        dd = drawdown.loc[date]
        
        # --- A. 策略決策 ---
        target_w_trend = w_trend.loc[date]
        w_dip = 0.0
        
        if dip_holding:
            profit = (price / dip_entry_price) - 1
            days_held = (date - dip_entry_date).days
            if profit >= DIP_EXIT_PROFIT or days_held >= DIP_MAX_HOLD:
                dip_holding = False # 出場
            else:
                w_dip = DIP_ADD_WEIGHT # 續抱
        else:
            if dd <= DIP_TRIGGER:
                dip_holding = True # 進場
                dip_entry_price = price
                dip_entry_date = date
                w_dip = DIP_ADD_WEIGHT
        
        # --- B. 計算目標部位 ---
        # 淨值 = 現金 + 股票市值 - 負債
        # 注意：這裡的 cash 可能包含借來的錢，所以 Net Equity = (Cash + StockValue) - Loan
        # 為了簡化，我們用 "淨資產" 來計算目標持倉
        market_val = shares * price
        net_equity = cash + market_val - loan
        
        # 總目標權重 (槓桿率)
        total_target_lev = target_w_trend + w_dip
        total_target_lev = min(total_target_lev, MAX_LEVERAGE)
        
        target_market_val = net_equity * total_target_lev
        
        # 需要調整的市值
        diff_val = target_market_val - market_val
        
        # --- C. 執行交易與借貸 ---
        if diff_val != 0:
            shares_delta = diff_val / price
            shares += shares_delta
            
            # 資金變動
            cost = diff_val
            
            if cost > 0: # 買入 (需要錢)
                if cash >= cost:
                    cash -= cost
                else:
                    # 現金不夠，借錢 (增加 Loan)
                    borrow_need = cost - cash
                    cash = 0
                    loan += borrow_need
            else: # 賣出 (還錢)
                cash_in = abs(cost)
                if loan > 0:
                    repay = min(loan, cash_in)
                    loan -= repay
                    cash += (cash_in - repay)
                else:
                    cash += cash_in

        # --- D. 計算利息 ---
        if loan > 0:
            interest = loan * LOAN_RATE
            # 利息直接從淨值扣除 (假設從現金扣，若沒現金則增加負債)
            if cash >= interest:
                cash -= interest
            else:
                loan += (interest - cash)
                cash = 0
        
        # --- E. 計算維持率 ---
        # 維持率 = 股票市值 / 負債
        m_margin = (shares * price / loan * 100) if loan > 0 else 999.0 # 無負債時設為無限大
        
        # 記錄
        records.append({
            "date": date,
            "equity": cash + shares * price - loan,
            "loan": loan,
            "margin": m_margin,
            "leverage": total_target_lev,
            "price": price
        })

    df_res = pd.DataFrame(records).set_index("date")
    
    # 3. 統計與繪圖
    final_equity = df_res['equity'].iloc[-1]
    ret = (final_equity / INITIAL_CASH) - 1
    mdd = (df_res['equity'] / df_res['equity'].cummax() - 1).min()

    # 檢查是否有借款
    has_loan = (df_res['loan'] > 0).any()
    min_margin = df_res[df_res['loan'] > 0]['margin'].min() if has_loan else None

    print("\n" + "="*60)
    print(f"🏆 混合策略 + 質押風控報告")
    print("-" * 60)
    print(f"總報酬: {ret:.2%}")
    print(f"MDD:    {mdd:.2%}")

    if has_loan and min_margin is not None:
        print(f"最低維持率: {min_margin:.0f}%  (死線: {MAINTENANCE_MARGIN*100:.0f}%)")
        if min_margin < MAINTENANCE_MARGIN * 100:
            print("❌ 警告：策略在歷史回測中曾跌破維持率，會被斷頭！")
        else:
            print("✅ 通過壓力測試：維持率始終在安全範圍內。")
    else:
        print("📝 策略從未借款，維持率不適用")

    # 顯示槓桿使用情況
    avg_leverage = df_res['leverage'].mean()
    max_leverage = df_res['leverage'].max()
    print(f"平均槓桿: {avg_leverage:.2f}x")
    print(f"最大槓桿: {max_leverage:.2f}x")
    print("="*60)

    # 4. 畫圖
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=("權益曲線", "槓桿使用率", "借款金額 (Loan)", "質押維持率 (重要!)"))

    # Row 1: Equity
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['equity'], name="Net Equity", line=dict(color='#00CC96')), row=1, col=1)
    
    # Row 2: Leverage
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['leverage'], name="Leverage", line=dict(color='#AB63FA'), fill='tozeroy'), row=2, col=1)
    
    # Row 3: Loan
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['loan'], name="Loan Amount", line=dict(color='#EF553B')), row=3, col=1)
    
    # Row 4: Margin (關鍵)
    if has_loan:
        # 只畫有借款的時候，避免 999 影響圖表
        margin_plot = df_res['margin'].copy()
        margin_plot[margin_plot > 300] = None # 超過 300% 不顯示，聚焦危險區

        fig.add_trace(go.Scatter(x=df_res.index, y=margin_plot, name="維持率 %", line=dict(color='orange', width=2)), row=4, col=1)

        # 畫死線 - 使用 add_shape 更可靠
        fig.add_shape(type="line", x0=df_res.index[0], x1=df_res.index[-1],
                     y0=MAINTENANCE_MARGIN*100, y1=MAINTENANCE_MARGIN*100,
                     line=dict(color="red", width=2, dash="solid"),
                     xref="x4", yref="y4")
        fig.add_annotation(x=df_res.index[-1], y=MAINTENANCE_MARGIN*100, text="斷頭線 (130%)",
                          showarrow=False, xref="x4", yref="y4", xanchor="left", font=dict(color="red"))

        fig.add_shape(type="line", x0=df_res.index[0], x1=df_res.index[-1],
                     y0=WARNING_MARGIN*100, y1=WARNING_MARGIN*100,
                     line=dict(color="yellow", width=1, dash="dot"),
                     xref="x4", yref="y4")
        fig.add_annotation(x=df_res.index[-1], y=WARNING_MARGIN*100, text="追繳線 (140%)",
                          showarrow=False, xref="x4", yref="y4", xanchor="left", font=dict(color="yellow"))
    else:
        # 沒有借款，顯示說明文字
        fig.add_annotation(text="策略從未借款<br>維持率不適用",
                          xref="x4", yref="y4",
                          x=df_res.index[len(df_res)//2], y=150,
                          showarrow=False, font=dict(size=16, color="gray"))

    fig.update_layout(title="混合策略壓力測試：你會被斷頭嗎？", template="plotly_dark", height=1000, hovermode="x unified")
    fig.show()

if __name__ == "__main__":
    main()