#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試腳本：驗證結算平倉的智能識別功能
==========================================

這個腳本會：
1. 讀取實際的回測 CSV 檔案
2. 檢查最後一筆交易是否為賣出
3. 驗證智能識別邏輯是否正確
"""

import pandas as pd
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta

def test_forced_sell_detection():
    """測試結算平倉識別邏輯"""

    # 讀取一個實際的 CSV 檔案
    csv_path = Path("sss_backtest_outputs/trades_from_results_RMA_Factor_TWII_2414_trial196.csv")

    if not csv_path.exists():
        print(f"❌ 找不到測試檔案: {csv_path}")
        return

    print(f"[FILE] {csv_path}")
    print("=" * 60)

    # Read trades
    trades = pd.read_csv(csv_path)
    trades['trade_date'] = pd.to_datetime(trades['trade_date'])

    print(f"\n[INFO] Total trades: {len(trades)}")
    print(f"[INFO] Type values: {trades['type'].unique()}")
    print(f"\n=== 最後 3 筆交易 ===")
    print(trades[['trade_date', 'type', 'price']].tail(3))

    # 找出所有賣出記錄
    all_sells = trades[trades['type'].str.contains('sell', case=False, na=False)]

    if all_sells.empty:
        print("\n⚠️ 沒有找到任何賣出記錄")
        return

    # 找出最後一筆賣出
    last_sell = all_sells.sort_values('trade_date').iloc[-1]

    print(f"\n=== 最後一筆賣出 ===")
    print(f"日期: {last_sell['trade_date']}")
    print(f"類型: {last_sell['type']}")
    print(f"價格: {last_sell['price']}")

    # 模擬回測結束日期（假設是 CSV 中的最後交易日）
    backtest_end_date = trades['trade_date'].max()

    print(f"\n=== 智能識別邏輯 ===")
    print(f"最後賣出日期: {last_sell['trade_date'].date()}")
    print(f"回測結束日期: {backtest_end_date.date()}")

    # 判斷是否為強制平倉
    is_forced = last_sell['trade_date'].date() == backtest_end_date.date()

    if is_forced:
        print(f"\n✅ 識別為 **結算平倉**")
        print(f"   理由: 最後一筆賣出日期 = 回測結束日期")
    else:
        print(f"\n🔴 識別為 **正常賣出**")
        print(f"   理由: 最後一筆賣出日期 ≠ 回測結束日期")

    # 檢查是否有明確標記
    has_explicit_forced = 'forced' in last_sell['type'].lower()

    print(f"\n=== CSV 明確標記 ===")
    if has_explicit_forced:
        print(f"✅ Type 欄位包含 'forced': {last_sell['type']}")
    else:
        print(f"❌ Type 欄位沒有 'forced' 標記")
        print(f"   → 需要使用智能識別！")

    print("\n" + "=" * 60)
    print("📌 測試結論:")
    if has_explicit_forced:
        print("   這是新版 CSV，已有 'sell_forced' 標記")
    elif is_forced:
        print("   這是舊版 CSV，需要智能識別")
        print("   ✅ 智能識別成功：最後一筆賣出應顯示為「結算平倉」")
    else:
        print("   最後一筆是正常策略賣出")

    return is_forced


def test_with_real_data():
    """使用真實股票數據測試完整流程"""

    print("\n" + "=" * 60)
    print("🧪 測試完整圖表渲染流程")
    print("=" * 60)

    try:
        from sss_core.plotting_unified import create_unified_dashboard

        # 創建模擬數據
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        df_raw = pd.DataFrame({
            'close': 100 + pd.Series(range(len(dates))) * 0.5,
            'volume': 10000
        }, index=dates)

        # 模擬交易記錄（最後一天強制賣出）
        trade_df = pd.DataFrame({
            'trade_date': [
                pd.Timestamp('2024-03-15'),
                pd.Timestamp('2024-06-20'),
                pd.Timestamp('2024-12-31')  # 最後一天賣出
            ],
            'type': ['buy', 'sell', 'sell'],  # 注意：第三個是 'sell'，不是 'sell_forced'
            'price': [105, 140, 250]
        })

        daily_state = pd.DataFrame({
            'equity': 100000 + pd.Series(range(len(dates))) * 100,
            'cash': 50000,
            'w': 0.5
        }, index=dates)

        print("\n📊 創建統一圖表...")
        fig = create_unified_dashboard(df_raw, daily_state, trade_df, "TEST.TW", theme='dark')

        # 檢查圖表中有多少個 trace
        print(f"✅ 圖表創建成功，包含 {len(fig.data)} 個 traces")

        # 檢查是否有「結算平倉」的 trace
        has_forced_sell_trace = any('結算平倉' in str(trace.name) for trace in fig.data)

        if has_forced_sell_trace:
            print("✅ 找到「結算平倉」標記")
        else:
            print("❌ 未找到「結算平倉」標記")
            print("   請檢查智能識別邏輯")

        # 列出所有 traces
        print("\n📋 圖表包含的所有 traces:")
        for i, trace in enumerate(fig.data):
            print(f"   {i+1}. {trace.name}")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("[TEST] Forced Sell Detection")
    print("=" * 60)

    # Test 1: Check actual CSV file
    test_forced_sell_detection()

    # Test 2: Test full rendering
    test_with_real_data()

    print("\n[OK] All tests completed")
