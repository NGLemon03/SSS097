# run_hybrid_backtest_v3.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 引用現有模組
sys.path.append(os.getcwd())
try:
    from sss_core.logic import load_data
except ImportError:
    print("⚠️ 警告: 找不到 sss_core，請確保你在專案根目錄下執行")
    pass

# ================= 設定區 =================
TICKER = "00631L.TW"
START_DATE = "2015-01-01"

# 1. 趨勢策略來源 (模擬)
# 這裡主要針對 User 的 Buy the Dip + 質押邏輯進行整合

# 2. 逆勢策略設定 (Buy the Dip)
DIP_TRIGGER = -0.30      # 高點回落 30% 開始接刀
DIP_ADD_WEIGHT = 0.7     # 加碼 0.5 倍槓桿
DIP_EXIT_PROFIT = 0.30   # 獲利 30% 跑
DIP_MAX_HOLD = 60       # 持有 60 天

# 3. 資金與質押設定
INITIAL_CASH = 1_000_000
LOAN_RATE = 0.03 / 365   # 質押年利率 3%
MAX_TOTAL_WEIGHT = 2.5   # 最大總曝險
MAINTENANCE_MARGIN = 1.40 # 追繳線
LIQUIDATION_MARGIN = 1.30 # 斷頭線

def run_backtest():
    # 1. 載入資料
    print(f"📥 載入數據: {TICKER}...")
    try:
        # 嘗試從 sss_core 讀取
        raw_data = load_data(TICKER, start_date="2014-01-01")
        
        # --- FIX: 處理 Tuple 回傳值 ---
        if isinstance(raw_data, tuple):
            # 如果回傳是 (df, msg) 或其他 tuple 結構，自動找出裡面的 DataFrame
            print(f"   (偵測到回傳值為 tuple，進行拆包...)")
            df = None
            for item in raw_data:
                if isinstance(item, pd.DataFrame):
                    df = item
                    break
            if df is None:
                raise ValueError("無法從 load_data 的回傳值中找到 DataFrame")
        else:
            df = raw_data
            
    except (NameError, ImportError, ValueError) as e:
        # 如果失敗，使用 yfinance 下載
        print(f"   (使用 yfinance 下載資料，原因: {e})")
        import yfinance as yf
        df = yf.download(TICKER, start="2014-01-01", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # --- FIX: 統一將欄位轉小寫 (放在 try-except 之外確保執行) ---
    df.columns = [str(c).lower() for c in df.columns]

    # 檢查是否有 'close' 欄位
    if 'close' not in df.columns:
        print(f"❌ 錯誤: 找不到 'close' 欄位。現有欄位: {df.columns.tolist()}")
        return

    close = df['close']
    
    # 2. 產生信號
    base_signal = pd.Series(1.0, index=close.index) 

    # 計算歷史高點與回撤
    roll_max = close.rolling(250, min_periods=1).max()
    drawdown = (close - roll_max) / roll_max

    # 3. 初始化回測變數
    cash = INITIAL_CASH
    shares = 0
    loan = 0
    
    records = []
    
    # 抄底狀態
    dip_holding = False
    dip_entry_price = 0
    dip_entry_date = None
    dip_shares = 0

    print("🚀 開始回測 Hybrid 策略 (修正版 v3.1)...")

    for date, price in close.items():
        if date < pd.Timestamp(START_DATE):
            continue
            
        # A. 計算總資產
        stock_val = shares * price
        equity = cash + stock_val - loan
        
        # 維持率計算
        if loan > 0:
            margin = (stock_val / loan) 
        else:
            margin = np.inf 

        # B. 策略邏輯
        target_base_val = equity * base_signal.loc[date]
        current_dd = drawdown.loc[date]
        
        # 進場檢查
        if not dip_holding:
            if current_dd <= DIP_TRIGGER:
                dip_holding = True
                dip_entry_price = price
                dip_entry_date = date
                dip_shares = (equity * DIP_ADD_WEIGHT) / price
        
        # 出場檢查
        if dip_holding:
            days_held = (date - dip_entry_date).days
            profit = (price / dip_entry_price) - 1
            
            if profit >= DIP_EXIT_PROFIT or days_held >= DIP_MAX_HOLD:
                dip_holding = False
                dip_shares = 0
        
        # C. 執行交易
        target_dip_val = dip_shares * price if dip_holding else 0
        total_target_val = target_base_val + target_dip_val
        
        if total_target_val > equity * MAX_TOTAL_WEIGHT:
            total_target_val = equity * MAX_TOTAL_WEIGHT
            
        current_val = shares * price
        diff_val = total_target_val - current_val
        
        shares_delta = diff_val / price
        shares += shares_delta
        cash -= diff_val
        
        # D. 資金管理
        if cash < 0:
            loan += abs(cash)
            cash = 0
        elif cash > 0 and loan > 0:
            repay = min(cash, loan)
            loan -= repay
            cash -= repay
            
        # E. 計算利息
        if loan > 0:
            interest = loan * LOAN_RATE
            loan += interest
            
        records.append({
            'Date': date,
            'Price': price,
            'Equity': equity - loan if cash == 0 else equity,
            'Cash': cash,
            'Loan': loan,
            'Shares': shares,
            'Margin': margin if loan > 0 else None,
            'Drawdown': current_dd,
            'Dip_Active': 1 if dip_holding else 0
        })

    # 4. 整理結果
    df_res = pd.DataFrame(records).set_index('Date')
    
    first_price = df_res['Price'].iloc[0]
    df_res['BH_Equity'] = (df_res['Price'] / first_price) * INITIAL_CASH
    
    final_equity = df_res['Equity'].iloc[-1]
    final_bh = df_res['BH_Equity'].iloc[-1]
    ret_strategy = (final_equity / INITIAL_CASH) - 1
    ret_bh = (final_bh / INITIAL_CASH) - 1
    mdd_strategy = (df_res['Equity'] / df_res['Equity'].cummax() - 1).min()

    print("="*50)
    print(f"📊 回測結果 (v3.1)")
    print(f"策略最終權益: {final_equity:,.0f} (報酬率 {ret_strategy:.2%})")
    print(f"B&H 最終權益: {final_bh:,.0f} (報酬率 {ret_bh:.2%})")
    print(f"策略 MDD: {mdd_strategy:.2%}")
    print("="*50)

    # 5. 繪圖
    plot_results(df_res)

def plot_results(df):
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("權益曲線比較 (Log Scale)", "回撤 (Drawdown)", "槓桿倍數 (Leverage)", "維持率 (Margin)")
    )

    # Row 1: Equity
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name="Hybrid 策略", 
                             line=dict(color='#00CC96', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BH_Equity'], name="00631L B&H", 
                             line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    # Row 2: Drawdown
    dd_strat = df['Equity'] / df['Equity'].cummax() - 1
    dd_bh = df['BH_Equity'] / df['BH_Equity'].cummax() - 1
    fig.add_trace(go.Scatter(x=df.index, y=dd_strat, name="策略 DD", 
                             line=dict(color='#EF553B', width=1), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=dd_bh, name="B&H DD", 
                             line=dict(color='gray', width=1, dash='dot')), row=2, col=1)

    # Row 3: Leverage
    leverage = (df['Shares'] * df['Price']) / df['Equity']
    fig.add_trace(go.Scatter(x=df.index, y=leverage, name="實質槓桿", 
                             line=dict(color='#636EFA', width=1.5)), row=3, col=1)
    
    dip_dates = df[df['Dip_Active'] == 1].index
    if not dip_dates.empty:
        fig.add_trace(go.Scatter(x=dip_dates, y=leverage.loc[dip_dates], mode='markers', 
                                 name="抄底期間", marker=dict(color='red', size=2)), row=3, col=1)

    # Row 4: Margin
    margin_data = df['Margin'].dropna() * 100
    fig.add_trace(go.Scatter(x=margin_data.index, y=margin_data, name="維持率 %", 
                             mode='lines+markers', marker=dict(size=2),
                             line=dict(color='#FFA15A', width=1.5)), row=4, col=1)

    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=130, y1=130,
                  line=dict(color="red", width=2, dash="solid"), row=4, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=140, y1=140,
                  line=dict(color="yellow", width=1, dash="dot"), row=4, col=1)
    
    min_margin = margin_data.min() if not margin_data.empty else 200
    if min_margin < 200:
        fig.update_yaxes(range=[120, 300], row=4, col=1)

    fig.update_layout(height=1000, title_text=f"Hybrid 策略回測 v3.1 - {TICKER}", template="plotly_dark")
    fig.show()

if __name__ == "__main__":
    run_backtest()