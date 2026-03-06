# analysis/strategy_manager.py
import json
import pandas as pd
import os
import glob
import shutil
from datetime import datetime
from pathlib import Path

# 設定路徑
BASE_DIR = Path(os.getcwd())
ANALYSIS_DIR = BASE_DIR / "analysis"
DEFAULT_WAREHOUSE = ANALYSIS_DIR / "strategy_warehouse.json"
HISTORY_FILE = ANALYSIS_DIR / "signal_history.csv"

class StrategyManager:
    def __init__(self):
        self._ensure_files()

    def _ensure_files(self):
        """確保預設檔案存在"""
        if not DEFAULT_WAREHOUSE.exists():
            # 🔥 初始化時加入 metadata 欄位
            self._write_json(DEFAULT_WAREHOUSE, {
                "last_updated": "",
                "metadata": {},  # 新增：存放 OOS 日期、模式等
                "strategies": []
            })

        if not HISTORY_FILE.exists():
            pd.DataFrame(columns=["date", "strategy_name", "signal", "price", "timestamp"]).to_csv(HISTORY_FILE, index=False)

    def _write_json(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def save_strategies(self, strategy_list, tag=None, metadata=None):
        """
        存入策略參數與元數據

        Args:
            strategy_list: 策略列表
            tag: 版本標籤 (例如 'OOS_2025')
            metadata: dict, 例如 {"split_date": "2024-12-31", "mode": "OOS", "score_mode": "alpha"}
        """
        # 讀取舊資料以保留原本的 metadata (如果沒傳新的)
        current_data = {}
        if DEFAULT_WAREHOUSE.exists():
            try:
                with open(DEFAULT_WAREHOUSE, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
            except Exception:
                pass

        final_metadata = metadata if metadata else current_data.get("metadata", {})

        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tag": tag if tag else "default",
            "metadata": final_metadata,  # 🔥 儲存關鍵資訊
            "strategies": strategy_list
        }

        # 1. 更新現役倉庫
        self._write_json(DEFAULT_WAREHOUSE, data)
        print(f"💾 已更新現役倉庫 (含 Metadata)")

        # 2. 如果有 tag，另外存一份備份
        if tag:
            # 清理 tag 字串，避免非法檔名
            safe_tag = "".join([c for c in tag if c.isalnum() or c in ('-', '_')])

            # 🔥 修改：智慧加時間戳，確保唯一性
            # 如果 tag 本身已包含 "Run" (通常是 Pipeline 自動生成的，已有時間戳)
            # 就不額外加時間，避免檔名過長
            if "Run" in safe_tag:
                filename = f"warehouse_{safe_tag}.json"  # Tag 已有時間，直接用
            else:
                # 手動 Tag (例如 --tag MyTest)，加上時間戳防止覆蓋
                timestamp_suffix = datetime.now().strftime("%H%M%S")
                filename = f"warehouse_{safe_tag}_{timestamp_suffix}.json"

            backup_path = ANALYSIS_DIR / filename
            self._write_json(backup_path, data)
            print(f"📦 已建立備份: {filename}")

    def load_strategies(self, filename="strategy_warehouse.json"):
        """讀取策略，預設讀取現役倉庫"""
        target_path = ANALYSIS_DIR / filename
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("strategies", [])
        except Exception as e:
            print(f"❌ 讀取 {filename} 失敗: {e}")
            return []

    def load_metadata(self, filename="strategy_warehouse.json"):
        """🔥 新增：讀取元數據"""
        target_path = ANALYSIS_DIR / filename
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("metadata", {})
        except Exception:
            return {}

    def list_warehouses(self):
        """列出所有可用的倉庫檔案"""
        files = list(ANALYSIS_DIR.glob("warehouse_*.json"))
        # 加入預設倉庫
        files.append(DEFAULT_WAREHOUSE)
        
        # 整理輸出
        file_list = []
        for f in files:
            if f.exists():
                file_list.append(f.name)
        
        # 排序：預設倉庫排第一，其他的按時間
        file_list = sorted(file_list, key=lambda x: 0 if x == "strategy_warehouse.json" else 1)
        return file_list

    def set_active_warehouse(self, filename):
        """將某個備份檔「扶正」為現役倉庫"""
        if filename == "strategy_warehouse.json":
            return True
            
        source = ANALYSIS_DIR / filename
        if not source.exists():
            return False
            
        shutil.copy(source, DEFAULT_WAREHOUSE)
        print(f"✅ 已將 {filename} 設定為現役倉庫！")
        return True

    def log_prediction(self, date, strategy_name, signal, price):
        # ... (保持原樣)
        new_row = {
            "date": date,
            "strategy_name": strategy_name,
            "signal": "LONG" if signal == 1 else "CASH",
            "price": price,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            df = pd.read_csv(HISTORY_FILE)
            mask = (df['date'] == str(date)) & (df['strategy_name'] == strategy_name)
            if not df[mask].empty:
                df.loc[mask, ['signal', 'price', 'timestamp']] = [new_row['signal'], new_row['price'], new_row['timestamp']]
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(HISTORY_FILE, index=False)
        except Exception as e:
            print(f"❌ 寫入紀錄失敗: {e}")

manager = StrategyManager()
