import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# ================= 設定區 =================
TICKER = "00631L.TW"
START_DATE = "2015-01-01"

# 策略參數
DIP_TRIGGER = -0.30      # 跌 30% 啟動抄底
DIP_ADD_WEIGHT = 1     # 抄底加碼倍數
DIP_EXIT_PROFIT = 0.30   # 賺 30% 跑
DIP_MAX_HOLD = 60       # 抱 120 天

# 資金與風控
INITIAL_CASH = 1_000_000
LOAN_RATE = 0.03 / 365   # 質押利息 3%
LIQUIDATION_MARGIN = 1.30 # 斷頭線 130%

def load_data_robust():
    """讀取股價資料"""
    print(f"📥 準備載入股價數據: {TICKER}...")
    csv_path = '00631L.TW_data_raw.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            start_idx = -1
            for i in range(10): 
                try:
                    pd.to_datetime(df.iloc[i, 0])
                    start_idx = i
                    break
                except: continue
            
            if start_idx != -1:
                data = df.iloc[start_idx:].reset_index(drop=True)
                data = data.iloc[:, :2]
                data.columns = ['Date', 'Close']
                data['Date'] = pd.to_datetime(data['Date'])
                data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
                data = data.dropna().set_index('Date').sort_index()
                return data['Close']
        except Exception:
            pass
    
    print("   (轉用 yfinance 下載...)")
    import yfinance as yf
    df = yf.download(TICKER, start="2014-01-01", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df['close']

def load_ensemble_strategy(start_date):
    """載入 Ensemble 策略結果"""
    print("🔍 尋找 Ensemble 策略檔案...")
    try:
        output_dir = Path("sss_backtest_outputs")
        target_files = list(output_dir.glob("*Ensemble_Majority*.csv"))
        
        if not target_files:
            print("   ⚠️ 找不到 Ensemble 檔案")
            return None, None
            
        latest_file = max(target_files, key=os.path.getctime)
        print(f"   📖 讀取基底策略: {latest_file.name}")
        
        df_ens = pd.read_csv(latest_file)
        
        # 日期處理
        if 'date' in df_ens.columns:
            df_ens['date'] = pd.to_datetime(df_ens['date'])
            df_ens = df_ens.set_index('date')
        else:
            try:
                df_ens.index = pd.to_datetime(df_ens.iloc[:, 0])
                df_ens.index.name = 'Date'
            except:
                return None, None
        
        df_ens = df_ens.sort_index()
        
        # 尋找權重欄位
        target_col = 'w' if 'w' in df_ens.columns else ('Position' if 'Position' in df_ens.columns else None)
        
        if target_col is None:
            return None, None

        df_ens = df_ens[df_ens.index >= pd.Timestamp(start_date)]

        # 尋找 Equity 欄位 (不區分大小寫)
        equity_col = None
        for col in ['equity', 'Equity', 'EQUITY']:
            if col in df_ens.columns:
                equity_col = df_ens[col]
                break

        return df_ens[target_col], equity_col
        
    except Exception:
        return None, None

def run_backtest():
    close = load_data_robust()
    if close is None: return
    close = close[close.index >= pd.Timestamp(START_DATE)]

    # 載入信號
    ens_signal, ens_equity = load_ensemble_strategy(START_DATE)

    # 建立 B&H 對照組
    bh_equity = (close / close.iloc[0]) * INITIAL_CASH

    # --- FIX: 確保 Ensemble 曲線存在 ---
    if ens_signal is not None:
        # 對齊信號
        base_signal = ens_signal.reindex(close.index).ffill().fillna(0)

        if ens_equity is not None:
             print("   ✅ 找到 Equity 欄位，直接使用策略權益曲線")
             ens_equity = ens_equity.reindex(close.index).ffill()
        else:
             # 如果檔案沒有 Equity 欄位，我們自己算！
             print("   ⚠️ 檔案無 equity 欄位，根據權重 w 重新計算資金曲線...")
             # 簡單回測：Daily_Ret = Close_Pct_Change * Weight(Yesterday)
             daily_ret = close.pct_change().fillna(0)
             strat_ret = daily_ret * base_signal.shift(1).fillna(base_signal.iloc[0])
             ens_equity = (1 + strat_ret).cumprod() * INITIAL_CASH
    else:
        # 沒檔案，用 B&H
        print("   ⚠️ 未找到 Ensemble 策略檔案，使用 Buy & Hold 作為基底")
        base_signal = pd.Series(1.0, index=close.index)
        ens_equity = bh_equity.copy()

    # === 新增：運行兩種策略並比較 ===
    print(f"\n🚀 開始回測 - 雙策略比較")
    print(f"   策略 A: 質押加碼型 (可借錢)")
    print(f"   策略 B: 閒置資金型 (優先使用閒置現金)")

    records_leveraged = run_strategy_leveraged(close, base_signal)
    records_cash_only = run_strategy_cash_only(close, base_signal)

    # 整理結果
    df_lev = pd.DataFrame(records_leveraged).set_index('Date')
    df_cash = pd.DataFrame(records_cash_only).set_index('Date')

    df_lev['Ens_Equity'] = ens_equity
    df_lev['BH_Equity'] = bh_equity
    df_cash['Ens_Equity'] = ens_equity
    df_cash['BH_Equity'] = bh_equity

    # 計算 Ensemble 和 B&H 的統計數據
    ens_ret = (ens_equity.iloc[-1] / INITIAL_CASH - 1) * 100
    ens_mdd = (ens_equity / ens_equity.cummax() - 1).min() * 100
    bh_ret = (bh_equity.iloc[-1] / INITIAL_CASH - 1) * 100
    bh_mdd = (bh_equity / bh_equity.cummax() - 1).min() * 100

    # 輸出比較
    print("\n" + "="*60)
    print("📊 策略比較結果")
    print("-"*60)
    print(f"{'策略名稱':12s} | {'報酬率':>8s} | {'MDD':>8s}")
    print("-"*60)

    # Ensemble 基準
    print(f"{'Ensemble':12s} | {ens_ret:7.1f}% | {ens_mdd:7.1f}%  ⬅️ 基準")
    print(f"{'B&H':12s} | {bh_ret:7.1f}% | {bh_mdd:7.1f}%")
    print("-"*60)

    # 混合策略
    for name, df in [("質押加碼型", df_lev), ("閒置資金型", df_cash)]:
        final_eq = df['Equity'].iloc[-1]
        ret = (final_eq / INITIAL_CASH - 1) * 100
        mdd = (df['Equity'] / df['Equity'].cummax() - 1).min() * 100

        # 計算 vs Ensemble 的差異
        ret_diff = ret - ens_ret
        mdd_diff = mdd - ens_mdd
        ret_sign = "+" if ret_diff > 0 else ""
        mdd_sign = "+" if mdd_diff > 0 else ""

        print(f"{name:12s} | {ret:7.1f}% | {mdd:7.1f}%  ({ret_sign}{ret_diff:.1f}% / {mdd_sign}{mdd_diff:.1f}%)")

    print("="*60)
    print("註: 括號內為 (報酬差異 / MDD差異) vs Ensemble")

    # 繪圖比較
    plot_comparison(df_lev, df_cash)

def run_strategy_leveraged(close, base_signal):
    """策略 A: 質押加碼型（原始邏輯，可以借錢）"""
    cash = INITIAL_CASH
    shares = 0
    loan = 0
    records = []
    is_liquidated = False

    dip_active = False
    dip_entry_price = 0
    dip_entry_date = None

    for date, price in close.items():
        if is_liquidated:
            records.append({
                'Date': date, 'Price': price, 'Equity': 0, 'Cash': 0, 'Loan': 0,
                'Shares': 0, 'Margin': None, 'Dip_Active': 0
            })
            continue

        # A. 利息
        if loan > 0:
            interest = loan * LOAN_RATE
            loan += interest

        # B. 淨值
        stock_val = shares * price
        equity = cash + stock_val - loan
        margin = (stock_val / loan) if loan > 0 else np.inf

        # Drawdown
        current_max = close.loc[:date].max()
        dd = (price - current_max) / current_max

        # C. 斷頭
        if (loan > 0 and margin < LIQUIDATION_MARGIN) or (equity <= 0):
            is_liquidated = True
            records.append({
                'Date': date, 'Price': price, 'Equity': 0, 'Cash': 0, 'Loan': 0,
                'Shares': 0, 'Margin': margin, 'Dip_Active': 0
            })
            continue

        # D. 交易邏輯
        current_base_w = base_signal.loc[date] if date in base_signal.index else 1.0

        if not dip_active:
            if dd <= DIP_TRIGGER:
                dip_active = True
                dip_entry_price = price
                dip_entry_date = date

        if dip_active:
            profit = (price / dip_entry_price) - 1
            days_held = (date - dip_entry_date).days
            if profit >= DIP_EXIT_PROFIT or days_held >= DIP_MAX_HOLD:
                dip_active = False

        # 計算目標槓桿（趨勢 + 抄底，可能超過 1.0）
        target_leverage = current_base_w + (DIP_ADD_WEIGHT if dip_active else 0)
        target_val = equity * target_leverage

        # 執行交易
        diff_val = target_val - (shares * price)
        if diff_val != 0:
            shares += diff_val / price
            cash -= diff_val

        # 資金調度（可以借錢）
        if cash < 0:
            loan += abs(cash)
            cash = 0
        elif cash > 0 and loan > 0:
            repay = min(cash, loan)
            loan -= repay
            cash -= repay

        records.append({
            'Date': date, 'Price': price, 'Equity': equity, 'Cash': cash, 'Loan': loan,
            'Shares': shares, 'Margin': margin if loan > 0 else None,
            'Dip_Active': 1 if dip_active else 0
        })

    return records

def run_strategy_cash_only(close, base_signal):
    """策略 B: 閒置資金型（優先使用閒置現金，不借錢）"""
    cash = INITIAL_CASH
    shares = 0
    loan = 0  # 這個策略不會借錢
    records = []

    dip_active = False
    dip_entry_price = 0
    dip_entry_date = None

    for date, price in close.items():
        # A. 淨值
        stock_val = shares * price
        equity = cash + stock_val

        # Drawdown
        current_max = close.loc[:date].max()
        dd = (price - current_max) / current_max

        # B. 交易邏輯
        current_base_w = base_signal.loc[date] if date in base_signal.index else 1.0

        # 抄底判斷
        if not dip_active:
            if dd <= DIP_TRIGGER:
                dip_active = True
                dip_entry_price = price
                dip_entry_date = date

        if dip_active:
            profit = (price / dip_entry_price) - 1
            days_held = (date - dip_entry_date).days
            if profit >= DIP_EXIT_PROFIT or days_held >= DIP_MAX_HOLD:
                dip_active = False

        # === 關鍵邏輯：優先級 ===
        # 1. 先計算抄底需求
        dip_weight = DIP_ADD_WEIGHT if dip_active else 0

        # 2. 抄底優先，剩下的給趨勢策略
        if dip_active:
            # 抄底佔用資金
            dip_target_val = equity * dip_weight
            # 趨勢策略只能用剩下的
            trend_target_val = max(0, equity * current_base_w)
            # 但總和不能超過本金（不借錢）
            total_target = min(equity, dip_target_val + trend_target_val)
        else:
            # 沒抄底，就照趨勢策略走
            total_target = equity * current_base_w

        # 執行交易
        diff_val = total_target - stock_val
        if diff_val != 0:
            shares += diff_val / price
            cash -= diff_val

        records.append({
            'Date': date, 'Price': price, 'Equity': equity, 'Cash': cash, 'Loan': 0,
            'Shares': shares, 'Margin': None,
            'Dip_Active': 1 if dip_active else 0
        })

    return records

def plot_comparison(df_lev, df_cash):
    """繪製雙策略比較圖"""
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.2],
        subplot_titles=("權益比較", "回撤比較", "槓桿使用", "借款金額", "維持率")
    )

    # Row 1: 權益曲線
    fig.add_trace(go.Scatter(x=df_lev.index, y=df_lev['Equity'],
                             name="質押型", line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_cash.index, y=df_cash['Equity'],
                             name="閒置資金型", line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_lev.index, y=df_lev['Ens_Equity'],
                             name="Ensemble", line=dict(color='orange', width=1.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_lev.index, y=df_lev['BH_Equity'],
                             name="B&H", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    fig.update_yaxes(type="log", title_text="金額 (log)", row=1, col=1)

    # Row 2: 回撤
    dd_lev = df_lev['Equity'] / df_lev['Equity'].cummax() - 1
    dd_cash = df_cash['Equity'] / df_cash['Equity'].cummax() - 1
    dd_ens = df_lev['Ens_Equity'] / df_lev['Ens_Equity'].cummax() - 1
    dd_bh = df_lev['BH_Equity'] / df_lev['BH_Equity'].cummax() - 1

    fig.add_trace(go.Scatter(x=df_lev.index, y=dd_lev, name="質押型 DD",
                             line=dict(color='red', width=1.5), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_cash.index, y=dd_cash, name="閒置型 DD",
                             line=dict(color='green', width=1.5), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_lev.index, y=dd_ens, name="Ensemble DD",
                             line=dict(color='orange', width=1, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_lev.index, y=dd_bh, name="B&H DD",
                             line=dict(color='gray', width=0.8, dash='dot')), row=2, col=1)

    # Row 3: 槓桿
    lev_lev = (df_lev['Shares'] * df_lev['Price']) / df_lev['Equity'].replace(0, 1)
    lev_cash = (df_cash['Shares'] * df_cash['Price']) / df_cash['Equity'].replace(0, 1)
    fig.add_trace(go.Scatter(x=df_lev.index, y=lev_lev, name="質押型槓桿",
                             line=dict(color='red', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_cash.index, y=lev_cash, name="閒置型槓桿",
                             line=dict(color='green', width=1.5)), row=3, col=1)

    # Row 4: 借款
    fig.add_trace(go.Scatter(x=df_lev.index, y=df_lev['Loan'], name="質押型借款",
                             line=dict(color='red', width=1.5)), row=4, col=1)

    # Row 5: 維持率
    margin_data = df_lev['Margin'].dropna() * 100
    if not margin_data.empty:
        fig.add_trace(go.Scatter(x=margin_data.index, y=margin_data, name="維持率",
                                 line=dict(color='orange', width=1.5)), row=5, col=1)
        fig.add_shape(type="line", x0=df_lev.index[0], x1=df_lev.index[-1],
                     y0=130, y1=130, line=dict(color="red", width=2), row=5, col=1)

    fig.update_layout(height=1200, title_text="策略比較：質押加碼 vs 閒置資金",
                     template="plotly_dark", showlegend=True)
    fig.show()

if __name__ == "__main__":
    run_backtest()