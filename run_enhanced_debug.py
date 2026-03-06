# -*- coding: utf-8 -*-
"""
Enhanced Analysis 快速診斷腳本
路徑：#run_enhanced_debug.py
創建時間：2025-08-18 12:05
作者：AI Assistant

用於快速診斷 enhanced analysis 的三大資料來源節點
"""

from pathlib import Path
import logging
import pandas as pd
import sys
import os

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from debug_enhanced_data import diag_df, try_load_file, diag_results_obj

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_backtest_files():
    """尋找可能的回測結果檔案"""
    possible_paths = []
    
    # 常見的回測結果路徑
    search_paths = [
        "results/",
        "sss_backtest_outputs/",
        "cache/",
        "data/",
        "."
    ]
    
    # 常見的檔案名稱模式
    file_patterns = [
        "*backtest*.json",
        "*backtest*.pkl",
        "*results*.json",
        "*results*.pkl",
        "*trades*.json",
        "*trades*.pkl",
        "*enhanced-trades-cache*.pkl",
        "*trades_from_results*.csv",
        "*ensemble_trade_ledger*.csv",
        "*ensemble_daily_state*.csv",
        "*ensemble_trades*.csv",
        "*ensemble_equity*.csv"
    ]
    
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            for pattern in file_patterns:
                files = list(path.glob(pattern))
                possible_paths.extend(files)
    
    return list(set(possible_paths))  # 去重

def main():
    """主診斷函式"""
    print("=== Enhanced Analysis 快速診斷腳本 ===\n")
    
    # 創建輸出目錄
    out_dir = Path("debug_out")
    out_dir.mkdir(exist_ok=True)
    
    print(f"輸出目錄：{out_dir.absolute()}")
    
    # 尋找可能的回測結果檔案
    print("\n正在搜尋回測結果檔案...")
    backtest_files = find_backtest_files()
    
    if not backtest_files:
        print("❌ 未找到任何回測結果檔案")
        print("\n請檢查以下路徑：")
        print("- results/")
        print("- sss_backtest_outputs/")
        print("- cache/")
        print("- data/")
        print("- 專案根目錄")
        return
    
    print(f"✅ 找到 {len(backtest_files)} 個可能的檔案：")
    for i, file_path in enumerate(backtest_files, 1):
        print(f"  {i}. {file_path}")
    
    # 診斷每個找到的檔案
    for i, file_path in enumerate(backtest_files, 1):
        print(f"\n{'='*50}")
        print(f"診斷檔案 {i}/{len(backtest_files)}: {file_path}")
        print(f"{'='*50}")
        
        try:
            # 嘗試載入檔案
            obj = try_load_file(file_path)
            if obj is None:
                print(f"❌ 無法載入檔案：{file_path}")
                continue
            
            # 根據物件類型進行診斷
            if isinstance(obj, pd.DataFrame):
                print(f"✅ 成功載入 DataFrame，形狀：{obj.shape}")
                
                # 輸出到檔案
                output_file = out_dir / f"diag_{file_path.stem}.txt"
                with open(output_file, "w", encoding="utf8") as f:
                    # 重定向 stdout 到檔案
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = f
                    
                    try:
                        diag_df(f"檔案：{file_path.name}", obj)
                    finally:
                        sys.stdout = old_stdout
                
                print(f"✅ 診斷結果已輸出到：{output_file}")
                
                # 在控制台顯示摘要
                print(f"\n--- {file_path.name} 摘要 ---")
                print(f"形狀：{obj.shape}")
                print(f"欄位：{list(obj.columns)}")
                if len(obj) > 0:
                    print(f"前3行：\n{obj.head(3).to_string()}")
                
            else:
                print(f"✅ 成功載入物件，類型：{type(obj)}")
                
                # 輸出到檔案
                output_file = out_dir / f"diag_{file_path.stem}_obj.txt"
                with open(output_file, "w", encoding="utf8") as f:
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = f
                    
                    try:
                        diag_results_obj(obj)
                    finally:
                        sys.stdout = old_stdout
                
                print(f"✅ 物件診斷結果已輸出到：{output_file}")
                
                # 在控制台顯示摘要
                print(f"\n--- {file_path.name} 物件摘要 ---")
                if isinstance(obj, dict):
                    print(f"鍵值數量：{len(obj)}")
                    print(f"主要鍵值：{list(obj.keys())[:10]}")  # 只顯示前10個
                else:
                    print(f"物件類型：{type(obj)}")
                    if hasattr(obj, "__dict__"):
                        print(f"屬性：{list(obj.__dict__.keys())[:10]}")
        
        except Exception as e:
            print(f"❌ 診斷檔案 {file_path} 時出錯：{e}")
            logger.exception(f"診斷失敗：{file_path}")
    
    print(f"\n{'='*50}")
    print("診斷完成！")
    print(f"結果檔案位於：{out_dir.absolute()}")
    print(f"{'='*50}")
    
    # 提供後續步驟建議
    print("\n📋 後續步驟建議：")
    print("1. 檢查 debug_out/ 目錄中的診斷結果")
    print("2. 根據診斷結果識別問題（如重複欄名、日期格式不一致等）")
    print("3. 在 enhanced_analysis_ui.py 中執行實際分析時，診斷輸出會顯示在控制台")
    print("4. 根據診斷結果調整資料處理邏輯")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 診斷被使用者中斷")
    except Exception as e:
        print(f"\n\n❌ 診斷腳本執行失敗：{e}")
        logger.exception("診斷腳本執行失敗")
        import traceback
        traceback.print_exc()
