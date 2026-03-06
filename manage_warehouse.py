# manage_warehouse.py
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import os

# 設定路徑
BASE_DIR = Path(os.getcwd())
ANALYSIS_DIR = BASE_DIR / "analysis"
ACTIVE_FILE = ANALYSIS_DIR / "strategy_warehouse.json"

def list_warehouses():
    """列出所有備份"""
    files = sorted(list(ANALYSIS_DIR.glob("warehouse_*.json")), key=os.path.getmtime, reverse=True)
    
    if not files:
        print("❌ 沒有找到任何備份檔案。")
        return []
    
    print(f"\n📦 發現 {len(files)} 個備份存檔 (按時間排序):")
    print("-" * 80)
    print(f"{'ID':<3} | {'建立時間':<20} | {'檔名 (Tag)':<50}")
    print("-" * 80)
    
    for i, f in enumerate(files):
        # 讀取 metadata 顯示更詳細資訊
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                meta = data.get("metadata", {})
                split_date = meta.get("split_date", "N/A")
                score_mode = meta.get("score_mode", "N/A")
                
                # 取得檔案修改時間
                mod_time = datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                
                # 顯示
                display_name = f.name.replace("warehouse_", "").replace(".json", "")
                print(f"{i:<3} | {mod_time:<20} | {display_name}")
                print(f"    └─ 訓練截止: {split_date} | 模式: {score_mode}")
        except:
            print(f"{i:<3} | Error reading file | {f.name}")
            
    print("-" * 80)
    return files

def restore_warehouse(files, index):
    """還原指定備份"""
    if index < 0 or index >= len(files):
        print("❌ 無效的 ID")
        return

    target_file = files[index]
    print(f"\n🔄 正在還原: {target_file.name} ...")
    
    try:
        shutil.copy(target_file, ACTIVE_FILE)
        print(f"✅ 成功！現在 'strategy_warehouse.json' 已經變成選定的版本。")
        print("👉 你現在可以執行 'python run_oos_analysis.py' 來查看該版本的表現。")
    except Exception as e:
        print(f"❌ 還原失敗: {e}")

if __name__ == "__main__":
    import os
    
    files = list_warehouses()
    
    if files:
        try:
            selection = input("\n請輸入要還原的 ID (直接按 Enter 離開): ")
            if selection.strip():
                restore_warehouse(files, int(selection))
            else:
                print("👋 離開，未做任何變更。")
        except ValueError:
            print("❌ 請輸入數字 ID")