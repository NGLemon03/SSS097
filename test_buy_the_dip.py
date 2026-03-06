import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# 設定
TICKER = "00631L.TW"
DROP_THRESHOLD = 0.3  # 跌 20% 買入
EXIT_PROFIT = 0.30     # 賺 20% 賣出
MAX_HOLD_DAYS = 60     # 最多抱 60 天 (搶反彈)

def main():
    print(f"🚀 測試「危機入市」策略: {TICKER}")
    print(f"   規則: 高點回落 {DROP_THRESHOLD:.0%} 買入，獲利 {EXIT_PROFIT:.0%} 或持有 {MAX_HOLD_DAYS} 天賣出")

    # 下載數據
    df = yf.download(TICKER, start="2015-01-01", auto_adjust=True)
    
    # --- 修正開始: 處理 yfinance 新版 MultiIndex 問題 ---
    if isinstance(df.columns, pd.MultiIndex):
        # 如果是多層索引，只取第一層 (Price Type)
        df.columns = df.columns.get_level_values(0)
    # --- 修正結束 ---

    df.columns = [c.lower() for c in df.columns]
    
    # 確保抓取到的 close 是 Series 格式 (有些版本可能會回傳 DataFrame)
    if isinstance(df['close'], pd.DataFrame):
         close = df['close'].iloc[:, 0]
    else:
         close = df['close']

    # 計算歷史高點 (Rolling Max 1年)
    roll_max = close.rolling(250, min_periods=1).max()
    drawdown = (close - roll_max) / roll_max

    trades = []
    in_pos = False
    entry_price = 0
    entry_date = None
    
    # 回測迴圈
    for date, price in close.items():
        # 確保 price 是單一數值 (float)，避免因為數據格式問題報錯
        price = float(price)
        
        try:
            dd = drawdown.loc[date]
            # 如果 dd 是 Series (有時候發生在重複 index)，取第一個值
            if isinstance(dd, pd.Series):
                dd = dd.iloc[0]
        except KeyError:
            continue
        
        if in_pos:
            # 持有中，檢查出場
            days_held = (date - entry_date).days
            profit = (price / entry_price) - 1
            
            # 出場條件
            if profit >= EXIT_PROFIT or days_held >= MAX_HOLD_DAYS:
                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": date,
                    "Entry Price": entry_price,
                    "Exit Price": price,
                    "Return": profit,
                    "Reason": "Take Profit" if profit >= EXIT_PROFIT else "Time Out"
                })
                in_pos = False
        else:
            # 空手中，檢查進場
            if dd <= -DROP_THRESHOLD:
                # 買入！
                in_pos = True
                entry_price = price
                entry_date = date

    # 統計結果
    if not trades:
        print("沒有觸發任何交易")
        return

    res = pd.DataFrame(trades)
    print("\n" + "="*50)
    print(f"📊 交易統計 (共 {len(res)} 次)")
    print("-" * 50)
    print(f"平均報酬: {res['Return'].mean():.2%}")
    print(f"勝率: {(res['Return'] > 0).mean():.2%}")
    print(f"最大虧損: {res['Return'].min():.2%}")
    # 修正複利計算公式，處理可能出現的 NaN
    print(f"總報酬 (複利): {((1 + res['Return']).prod() - 1):.2%}")
    print("="*50)
    
    # 顯示詳細交易
    print("\n🔍 近 10 筆交易:")
    print(res.tail(10)[['Entry Date', 'Exit Date', 'Return', 'Reason']].to_string())

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close", line=dict(color='gray')))
    
    # 標示買點
    buy_dates = res['Entry Date']
    buy_prices = res['Entry Price']
    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy the Dip',
                             marker=dict(color='red', size=10, symbol='triangle-up')))
    
    fig.update_layout(title=f"危機入市策略 ({TICKER})", template="plotly_dark")
    fig.show()

if __name__ == "__main__":
    main()