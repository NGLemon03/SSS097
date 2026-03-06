# predict_tomorrow.py
# -*- coding: utf-8 -*-
"""
SSS096 全自動每日營運系統 (含相關性過濾版)
功能：自動掃描策略、過濾高相關性策略、智慧載入數據、產生訊號、紀錄歷史
"""

import pandas as pd
import numpy as np
import os
import sys
import ast
import re
import argparse
from pathlib import Path
from datetime import datetime
import sys
import io
# 強制將輸出編碼設為 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 引用 sss_core (計算核心 - 無 Streamlit 依賴)
sys.path.append(os.getcwd())
try:
    import sss_core as SSS
except ImportError:
    print("❌ 錯誤：找不到 sss_core")
    print("提示：請確保 sss_core 目錄存在且包含 logic.py")
    sys.exit(1)

# 引用管理器
try:
    from analysis.strategy_manager import manager
except ImportError:
    print("❌ 錯誤：找不到 analysis/strategy_manager.py")
    sys.exit(1)

RESULTS_DIR = Path("results")
TRADES_DIR = Path("sss_backtest_outputs")
TICKER = "00631L.TW"
DEFAULT_WAREHOUSE = "strategy_warehouse.json"
TOP_K_AUTO_INIT = 5
MAX_CORRELATION = 0.90  # 🔥 新增：相關性門檻 (高於此值視為重複策略)

_DATA_CACHE = {}

# ... (保留原本 get_data_smart 與 calculate_signal 函式，無須變動) ...
def get_data_smart(ticker, strategy_name):
    """智慧判斷並載入數據 (含快取)"""
    source_type = "Self"
    if "Factor" in strategy_name:
        if "2412" in strategy_name: source_type = "Factor (^TWII / 2412.TW)"
        elif "2414" in strategy_name: source_type = "Factor (^TWII / 2414.TW)"
        else: source_type = "Factor (^TWII / 2414.TW)"

    cache_key = f"{ticker}_{source_type}"
    if cache_key in _DATA_CACHE: return _DATA_CACHE[cache_key]

    # print(f"   ↳ 載入數據: {source_type} ...") # 減少噪音
    try:
        df_p, df_f = SSS.load_data(ticker, smaa_source=source_type)
        _DATA_CACHE[cache_key] = (df_p, df_f)
        return df_p, df_f
    except Exception as e:
        print(f"   ❌ 數據載入失敗: {e}")
        return None, None

def calculate_signal(df_p, df_f, stype, params):
    try:
        if stype == 'ssma_turn':
            need = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 
                    'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
            p = {k: params[k] for k in need if k in params}
            res_df, buys, sells = SSS.compute_ssma_turn_combined(df_p, df_f, **p)
            pos = pd.Series(0, index=res_df.index)
            if buys: pos.loc[[d for d in buys if d in pos.index]] = 1
            if sells: pos.loc[[d for d in sells if d in pos.index]] = 0
            pos = pos.ffill().fillna(0)
            return int(pos.iloc[-1])
        elif stype in ['RMA', 'single']:
            if stype == 'RMA':
                p = {k: params[k] for k in ['linlen', 'factor', 'smaalen', 'rma_len', 'dev_len'] if k in params}
                res_df = SSS.compute_RMA(df_p, df_f, **p)
            else:
                p = {k: params[k] for k in ['linlen', 'factor', 'smaalen', 'devwin'] if k in params}
                res_df = SSS.compute_single(df_p, df_f, **p)
            last = res_df.iloc[-1]
            buy_thr = last['base'] + last['sd'] * params.get('buy_mult', 0.5)
            if last['smaa'] < buy_thr: return 1
            sell_thr = last['base'] + last['sd'] * params.get('sell_mult', 0.5)
            if last['smaa'] > sell_thr: return 0
            return -1 
    except Exception: return 0
    return 0

# 🔥 修改：新增相關性檢查邏輯
def get_strategy_position_series(trade_file_path):
    """從交易檔快速還原持倉序列 (用於計算相關性)"""
    try:
        df = pd.read_csv(trade_file_path)
        # 簡單重建：只要有 buy 就是 1, sell 就是 0
        # 這裡需要一個統一的日期索引，我們用 trade_date
        if 'trade_date' not in df.columns: return None
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').set_index('trade_date')
        
        # 建立一個足夠長的日期範圍 (例如最近3年)
        idx = pd.date_range(end=datetime.now(), periods=750, freq='B') 
        pos = pd.Series(0, index=idx)
        
        current_pos = 0
        # 這裡用簡化算法：把交易日對應到 idx
        # 更精確的做法是 merge，但為了速度我們先只看大方向
        # 這裡改用比較 robust 的方法：
        # 讀取交易紀錄，重現每日持倉
        pos_list = []
        last_p = 0
        for d in idx:
            # 找這天有沒有交易
            # 這裡簡化：假設 CSV 已經是交易紀錄
            # 為了計算相關性，我們直接讀取檔案裡的訊號 (如果有的話) 
            # 或是依賴 SSS 的輸出來建立。
            # 最快的方法：直接假設檔案裡有 action
            pass
        
        # ⚡ 替代方案：直接利用檔案名稱去 call SSS 算一次歷史持倉
        # 這樣最準，但比較慢。我們改用 trade file 裡的訊號分佈。
        # 由於 trade file 只有交易點，我們用 cumsum 概念
        
        # 建立每日序列
        daily_pos = pd.Series(0, index=idx)
        for _, row in df.iterrows():
            d = row['trade_date']
            if d in daily_pos.index:
                if row['type'] == 'buy': daily_pos.loc[d:] = 1
                elif row['type'] == 'sell': daily_pos.loc[d:] = 0
        
        return daily_pos
    except:
        return None

def auto_initialize_strategies():
    """自動掃描並建立策略倉庫 (含去重邏輯)"""
    print("\n🔍 系統初次執行，正在掃描並過濾策略...")
    
    # 1. 建立參數索引
    param_db = {}
    for f in RESULTS_DIR.glob("**/*.csv"):
        try:
            df = pd.read_csv(f)
            if 'trial_number' in df.columns and 'parameters' in df.columns:
                for _, row in df.iterrows():
                    param_db[row['trial_number']] = ast.literal_eval(row['parameters'])
        except: pass

    # 2. 找所有交易檔
    files = list(TRADES_DIR.glob("trades_from_results_*.csv"))
    if not files:
        print("❌ 找不到 sss_backtest_outputs/ 下的檔案")
        return []

    # 依修改時間排序 (假設越新越好，或者你也可以依檔名內的 profit 排序)
    files.sort(key=os.path.getmtime, reverse=True)
    
    selected_strategies = []
    selected_positions = [] # 用來存持倉序列計算相關性

    print(f"   候選策略共 {len(files)} 個，目標選出差異化最大的 Top {TOP_K_AUTO_INIT}...")

    for f in files:
        if len(selected_strategies) >= TOP_K_AUTO_INIT: break
        
        fname = f.stem
        match = re.search(r"trial(\d+)", fname)
        if not match: continue
        tid = int(match.group(1))
        params = param_db.get(tid)
        if not params: continue

        # 🔥 計算持倉序列
        pos_series = get_strategy_position_series(f)
        if pos_series is None or pos_series.sum() == 0: continue # 無效策略

        # 🔥 檢查相關性
        is_duplicate = False
        for existing_pos in selected_positions:
            # 計算 Pearson 相關係數
            corr = pos_series.corr(existing_pos)
            if corr > MAX_CORRELATION:
                # print(f"   ⚠️ 策略 {fname[:15]}... 與現有策略高度相關 ({corr:.2f}) -> 跳過")
                is_duplicate = True
                break
        
        if is_duplicate: continue

        # 通過檢查，加入名單
        stype = "Unknown"
        if "RMA" in fname: stype = "RMA"
        elif "ssma_turn" in fname: stype = "ssma_turn"
        elif "single" in fname: stype = "single"

        selected_strategies.append({
            "name": fname,
            "type": stype,
            "params": params
        })
        selected_positions.append(pos_series)
        print(f"   ✅ 選入: {fname[:30]}... (類型: {stype})")

    if selected_strategies:
        manager.save_strategies(selected_strategies)
        return selected_strategies
    else:
        return []

def main():
    parser = argparse.ArgumentParser(description="SSS096 每日訊號預測")
    parser.add_argument("--ticker", default=TICKER, help="目標標的 (例如 00631L.TW)")
    parser.add_argument("--warehouse", default=DEFAULT_WAREHOUSE, help="策略倉庫檔名 (analysis/ 內)")
    args = parser.parse_args()

    ticker = args.ticker
    warehouse_file = args.warehouse or DEFAULT_WAREHOUSE

    print("="*60)
    print(f"🚀 SSS096 每日營運系統 ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"🎯 Ticker: {ticker} | 🗂️ Warehouse: {warehouse_file}")
    print("="*60)

    strategies = manager.load_strategies(warehouse_file)
    if not strategies:
        strategies = auto_initialize_strategies()
        if not strategies: return

    print(f"📂 使用 {len(strategies)} 個差異化策略進行預測")
    print("-" * 60)
    print(f"{'策略簡稱':<30} | {'類型':<10} | {'訊號'}")
    print("-" * 60)

    votes = []
    
    for strat in strategies:
        name = strat['name']
        short_name = name.replace("trades_from_results_", "")[:25] + "..."
        stype = strat['type']
        params = strat['params']

        df_p, df_f = get_data_smart(ticker, name)
        if df_p is None: continue

        sig = calculate_signal(df_p, df_f, stype, params)
        final_sig = 1 if sig == -1 else sig
        
        sig_str = "🟢 LONG" if final_sig == 1 else "⚪ CASH"
        if sig == -1: sig_str += " (Hold)"

        print(f"{short_name:<30} | {stype:<10} | {sig_str}")
        
        votes.append(final_sig)
        latest_price = df_p['close'].iloc[-1]
        latest_date = df_p.index[-1].strftime('%Y-%m-%d')
        
        manager.log_prediction(latest_date, name, final_sig, latest_price)

    if votes:
        long_votes = sum(votes)
        total = len(votes)
        print("-" * 60)
        print(f"📊 最終決策: {long_votes} 票多單 / {total - long_votes} 票空手")
        print(f"📝 結果已存入 analysis/signal_history.csv")
    else:
        print("⚠️ 無有效運算結果")
    print("="*60)

if __name__ == "__main__":
    main()
