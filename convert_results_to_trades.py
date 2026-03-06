# convert_results_to_trades.py
# -*- coding: utf-8 -*-
"""
將 Optuna 優化後的最佳參數 (results/*.csv) 轉換為實際交易明細 (sss_backtest_outputs/*.csv)
供 Ensemble 使用。
"""

import pandas as pd
import numpy as np
import ast
import os
import sys
import argparse
import re
from pathlib import Path

# 引用核心
sys.path.append(os.getcwd())
try:
    import sss_core as SSS
    # 確保引用 logic 層級
    from sss_core.logic import backtest_unified, load_data
except ImportError:
    print("❌ 錯誤：找不到 sss_core，請確認目錄結構")
    sys.exit(1)

# 設定路徑
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("sss_backtest_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_smart_data_source(strategy_type, filename):
    """
    從檔名或策略類型推斷數據源
    避免 ssma_turn 用到 Self 數據導致全錯
    """
    filename = str(filename)
    if "Factor" in filename:
        if "2412" in filename:
            return "Factor (^TWII / 2412.TW)"
        elif "2414" in filename:
            return "Factor (^TWII / 2414.TW)"
        else:
            # 預設 Factor
            return "Factor (^TWII / 2414.TW)"
    
    # 若檔名沒寫，但策略是 ssma_turn，通常需要 Factor
    if strategy_type == "ssma_turn":
        # 這裡做個保險，預設給它 2414
        return "Factor (^TWII / 2414.TW)"
        
    return "Self"

def run_backtest_and_save(stype, params, output_path, ticker):
    """
    執行單次回測並存檔
    """
    try:
        # 1. 智慧判斷數據源
        data_source = get_smart_data_source(stype, output_path)
        print(f"   ↳ 載入數據: {data_source} ...")
        
        df_p, df_f = load_data(ticker, smaa_source=data_source)
        
        if df_p.empty:
            print(f"   ❌ 數據載入失敗 (Empty DataFrame)")
            return False

        # 2. 計算指標
        if stype == 'RMA':
            # 篩選參數
            calc_keys = ['linlen', 'factor', 'smaalen', 'rma_len', 'dev_len']
            calc_p = {k: params[k] for k in calc_keys if k in params}
            df_ind = SSS.compute_RMA(df_p, df_f, **calc_p)
            res = SSS.backtest_unified(df_ind, 'RMA', params)
            
        elif stype == 'single':
            calc_keys = ['linlen', 'factor', 'smaalen', 'devwin']
            calc_p = {k: params[k] for k in calc_keys if k in params}
            df_ind = SSS.compute_single(df_p, df_f, **calc_p)
            res = SSS.backtest_unified(df_ind, 'single', params)
            
        elif stype == 'ssma_turn':
            calc_keys = [
                'linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days',
                'volume_target_pass_rate', 'volume_target_lookback'
            ]
            calc_p = {k: params[k] for k in calc_keys if k in params}
            df_ind, buys, sells = SSS.compute_ssma_turn_combined(df_p, df_f, **calc_p)
            res = SSS.backtest_unified(df_ind, 'ssma_turn', params, buy_dates=buys, sell_dates=sells)
            
        else:
            print(f"   ❌ 未知策略類型: {stype}")
            return False

        # 3. 儲存結果
        trades = res.get('trade_df')
        if trades is not None and not trades.empty:
            # 確保欄位標準化
            if 'date' in trades.columns: trades.rename(columns={'date': 'trade_date'}, inplace=True)
            if 'action' in trades.columns: trades.rename(columns={'action': 'type'}, inplace=True)
            
            trades.to_csv(output_path, index=False)
            
            # 顯示簡單績效
            metrics = res.get('metrics', {})
            ret = metrics.get('total_return', 0)
            cnt = metrics.get('num_trades', 0)
            print(f"   ✅ 成功產生: {output_path.name} (報酬: {ret*100:.1f}%, 筆數: {cnt})")
            return True
        else:
            print(f"   ⚠️ 無交易產生，跳過存檔")
            return False

    except Exception as e:
        print(f"   ❌ 回測執行錯誤: {e}")
        # import traceback
        # traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=10, help='每個策略類型取前 K 名')
    parser.add_argument('--ticker', type=str, default='00631L.TW')
    args = parser.parse_args()

    print(f"🚀 開始轉換 Optuna 結果為交易檔 (Top {args.top_k})...")
    
    # 1. 讀取所有結果檔
    result_files = list(RESULTS_DIR.glob("optuna_results_*.csv"))
    if not result_files:
        print("❌ 找不到 results/*.csv，請先執行 optuna_16.py")
        return

    # 2. 收集所有 trials
    all_trials = []
    for f in result_files:
        try:
            df = pd.read_csv(f)
            # 解析檔名取得策略類型
            fname = f.name
            stype = "Unknown"
            if "RMA" in fname: stype = "RMA"
            elif "ssma_turn" in fname: stype = "ssma_turn"
            elif "single" in fname: stype = "single"

            # 額外資訊 (用於檔名)
            extra = ""
            if "Factor" in fname:
                if "2412" in fname: extra = "_Factor_TWII_2412"
                elif "2414" in fname: extra = "_Factor_TWII_2414"

            # 🔥 新增：解析檔名以取得 TAG
            # 原始格式: optuna_results_single_Self_OOS_Safe_End20250130_Run1231_1905_20251231_191133.csv
            # 提取 TAG 部分（去掉 "optuna_results_" 前綴和最後的時間戳）
            filename_stem = f.stem  # 去掉 .csv
            clean_name = filename_stem.replace("optuna_results_", "")

            # 進一步清理：移除最後的時間戳（格式：_20251231_191133）
            # 使用正則表達式移除結尾的 _YYYYMMDD_HHMMSS 模式
            clean_name = re.sub(r'_\d{8}_\d{6}$', '', clean_name)

            for _, row in df.iterrows():
                # 確保有 value (score)
                if pd.isna(row.get('value')): continue

                # 解析參數字串
                params = row['parameters']
                if isinstance(params, str):
                    params = ast.literal_eval(params)

                all_trials.append({
                    'score': row['value'],
                    'trial_number': row['trial_number'],
                    'params': params,
                    'type': stype,
                    'extra': extra,
                    'source_file': fname,
                    'tag': clean_name  # 🔥 加入 TAG 資訊
                })
        except Exception as e:
            print(f"⚠️ 讀取 {f.name} 失敗: {e}")

    # 3. 排序並取 Top K
    # 這裡我們按「策略類型」分組取 Top K，以保持多樣性
    all_trials.sort(key=lambda x: x['score'], reverse=True)
    
    processed_count = 0
    type_counts = {}
    
    print("-" * 60)
    for trial in all_trials:
        stype = trial['type']

        # 每個類型只取 Top K
        if type_counts.get(stype, 0) >= args.top_k:
            continue

        tid = trial['trial_number']
        params = trial['params']
        tag = trial.get('tag', '')  # 🔥 取得 TAG

        # 🔥 建構輸出檔名（包含完整 TAG）
        # 新格式: trades_{完整TAG}_trial{N}.csv
        # 範例: trades_single_Self_OOS_Safe_End20250130_Run1231_1905_trial86.csv
        out_name = f"trades_{tag}_trial{tid}.csv"
        out_path = OUTPUT_DIR / out_name

        print(f"處理 #{processed_count+1}: {stype} Trial {tid} (Score: {trial['score']:.2f})")

        success = run_backtest_and_save(stype, params, out_path, args.ticker)

        if success:
            type_counts[stype] = type_counts.get(stype, 0) + 1
            processed_count += 1
            
    print("-" * 60)
    print(f"🎉 轉換完成！共產生 {processed_count} 個交易檔於 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
