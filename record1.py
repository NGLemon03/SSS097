import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# ==========================================
# 1. 環境設定：解決中文顯示問題
# ==========================================
# 請根據您的作業系統調整字體，例如：
# Windows: 'Microsoft JhengHei' (微軟正黑體)
# Mac: 'Arial Unicode MS' 或 'Heiti TC'
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

def analyze_trading_performance(file_path):
    # ==========================================
    # 2. 資料讀取與清洗
    # ==========================================
    df = pd.read_csv(file_path)
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 過濾目標證券
    target_stock = "元大台灣50正2"
    df_stock = df[df['證券'] == target_stock].copy().sort_values('日期')
    
    if df_stock.empty:
        print(f"找不到 {target_stock} 的交易紀錄。")
        return

    # 清洗數字格式 (處理逗號)
    for col in ['股數', '報價', '金額']:
        if df_stock[col].dtype == object:
            df_stock[col] = df_stock[col].str.replace(',', '').astype(float)
    
    latest_price = df_stock.iloc[-1]['報價']
    total_invested_capital = df_stock[df_stock['類型'] == '買入']['金額'].sum()

    # ==========================================
    # 3. FIFO (先進先出) 演算法：追蹤每一筆資金
    # ==========================================
    inventory = []  # 庫存池
    sell_decisions = [] # 賣出決策紀錄

    for _, row in df_stock.iterrows():
        if row['類型'] == '買入':
            inventory.append({
                'date': row['日期'], 
                'price': row['報價'], 
                'qty': row['股數'], 
                'amt': row['金額']
            })
        elif row['類型'] == '賣出':
            q_to_sell = row['股數']
            sell_p = row['報價']
            
            while q_to_sell > 0 and inventory:
                node = inventory[0]
                take_qty = min(q_to_sell, node['qty'])
                
                # 計算該筆資金權重
                weight = (take_qty * node['price']) / total_invested_capital
                
                sell_decisions.append({
                    '賣出日期': row['日期'],
                    '持有天數': (row['日期'] - node['date']).days,
                    '賣出價格': sell_p,
                    '賣出後續漲幅%': (latest_price / sell_p - 1) * 100,
                    '實際實現ROI%': (sell_p / node['price'] - 1) * 100,
                    '資金權重%': weight * 100
                })
                
                node['qty'] -= take_qty
                q_to_sell -= take_qty
                if node['qty'] == 0:
                    inventory.pop(0)

    decision_df = pd.DataFrame(sell_decisions)

    # ==========================================
    # 4. 繪製視覺化報告
    # ==========================================
    fig = plt.figure(figsize=(16, 18))
    
    # --- 圖一：股價走勢與交易點位 ---
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df_stock['日期'], df_stock['報價'], color='#1f77b4', alpha=0.6, label='股價走勢')
    buys = df_stock[df_stock['類型']=='買入']
    sells = df_stock[df_stock['類型']=='賣出']
    ax1.scatter(buys['日期'], buys['報價'], color='green', marker='^', label='買入', s=80, alpha=0.8)
    ax1.scatter(sells['日期'], sells['報價'], color='red', marker='v', label='賣出', s=80, alpha=0.8)
    ax1.set_title(f'{target_stock} 交易執行圖', fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # --- 圖二：賣飛回補分析 (Regret Analysis) ---
    ax2 = plt.subplot(3, 1, 2)
    sc = ax2.scatter(decision_df['賣出日期'], decision_df['賣出後續漲幅%'], 
                     s=decision_df['資金權重%'] * 50, # 球體反映資金大小
                     c=decision_df['實際實現ROI%'], cmap='RdYlGn', alpha=0.6, edgecolors='grey')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('決策後悔分析：賣出後「又漲了多少%」？ (球體越大=投入資金越多)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('賣出後的後續漲幅 (%)')
    plt.colorbar(sc, ax=ax2, label='該筆實際報酬率 (%)')
    ax2.grid(True, alpha=0.2)

    # --- 圖三：持倉時間與報酬率 (Efficiency Analysis) ---
    ax3 = plt.subplot(3, 1, 3)
    ax3.scatter(decision_df['持有天數'], decision_df['實際實現ROI%'], 
                s=decision_df['資金權重%'] * 50, alpha=0.5)
    
    # 加上趨勢線
    if len(decision_df) > 1:
        z = np.polyfit(decision_df['持有天數'], decision_df['實際實現ROI%'], 1)
        p = np.poly1d(z)
        ax3.plot(decision_df['持有天數'], p(decision_df['持有天數']), "r--", alpha=0.6, label='報酬趨勢')

    ax3.set_title('持倉效率分析：持有天數與實現報酬率的關係', fontsize=16, fontweight='bold')
    ax3.set_xlabel('持有天數 (Days)')
    ax3.set_ylabel('實際實現報酬率 (%)')
    ax3.grid(True, alpha=0.2)

    plt.tight_layout(pad=4.0)
    plt.show()
    
    # 存檔
    fig.savefig('trading_analysis_report.png', dpi=150)
    print("分析完成！圖表已儲存為 trading_analysis_report.png")

# 執行分析
if __name__ == "__main__":
    analyze_trading_performance('re.csv')