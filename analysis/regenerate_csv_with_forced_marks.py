#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV 資料重新生成工具
====================

用途：重新執行回測，生成帶有明確 sell_forced 標記的 CSV 檔案

為什麼需要？
-----------
舊版 CSV 只有 'sell' 標記，無法區分「策略賣出」和「強制平倉」。
新版 sss_core/logic.py 已支援 sell_forced，但需要重新執行回測才能生成。

使用方式：
---------
1. 確認備份（系統會自動備份到 archive/）
2. 執行此腳本
3. 等待回測完成（可能需要數小時）
4. 完成後，所有 CSV 都會有明確的 sell_forced 標記

注意事項：
---------
- 此腳本會覆蓋現有的 CSV 檔案
- 舊的 CSV 會自動備份到 archive/ 目錄
- 建議在非工作時間執行（例如晚上睡覺前）
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import shutil

# 設定編碼（Windows）
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def backup_existing_csv():
    """備份現有的 CSV 檔案"""
    output_dir = Path("sss_backtest_outputs")
    archive_dir = Path("archive")

    if not output_dir.exists():
        print("[WARN] sss_backtest_outputs/ 不存在，跳過備份")
        return None

    # 找出所有 CSV 檔案
    csv_files = list(output_dir.glob("*.csv"))

    if not csv_files:
        print("[WARN] 沒有找到任何 CSV 檔案，跳過備份")
        return None

    # 創建備份目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = archive_dir / f"Backup_BeforeRegenerate_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 複製所有 CSV
    print(f"\n[BACKUP] 備份現有 CSV 到: {backup_dir}")
    for csv_file in csv_files:
        shutil.copy2(csv_file, backup_dir / csv_file.name)

    print(f"[OK] 已備份 {len(csv_files)} 個 CSV 檔案")
    return backup_dir


def confirm_regeneration():
    """確認是否要重新生成"""
    print("\n" + "=" * 70)
    print("⚠️  WARNING: CSV 重新生成")
    print("=" * 70)
    print()
    print("此操作將會：")
    print("  1. 備份現有的 CSV 檔案到 archive/")
    print("  2. 重新執行所有策略的回測")
    print("  3. 生成帶有明確 sell_forced 標記的新 CSV")
    print()
    print("預估時間：")
    print("  - 小型專案（<10 個策略）：10-30 分鐘")
    print("  - 中型專案（10-50 個策略）：30-120 分鐘")
    print("  - 大型專案（>50 個策略）：2-4 小時")
    print()
    print("建議：在晚上睡覺前執行，明天早上就有全新的數據")
    print()

    response = input("是否繼續？(yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("\n[CANCEL] 操作已取消")
        return False

    return True


def run_full_pipeline():
    """執行完整的回測流程"""
    print("\n" + "=" * 70)
    print("[START] 開始重新生成 CSV")
    print("=" * 70)
    print()

    # 檢查 run_full_pipeline_3.py 是否存在
    pipeline_script = Path("run_full_pipeline_3.py")

    if not pipeline_script.exists():
        print(f"[ERROR] 找不到 {pipeline_script}")
        print("[INFO] 請確認你在正確的專案目錄中")
        return False

    # 執行 pipeline
    print(f"[INFO] 執行: python {pipeline_script}")
    print("[INFO] 這可能需要一些時間，請耐心等待...")
    print()

    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, str(pipeline_script)],
            check=True,
            text=True,
            encoding='utf-8'
        )

        print()
        print("=" * 70)
        print("[SUCCESS] CSV 重新生成完成！")
        print("=" * 70)
        print()
        print("後續步驟：")
        print("  1. 檢查 sss_backtest_outputs/ 目錄")
        print("  2. 確認新的 CSV 檔案包含 'sell_forced' 標記")
        print("  3. 在 Dash UI 中重新執行回測，查看灰色方塊是否正確顯示")
        print()

        return True

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"[ERROR] 執行失敗: {e}")
        print("=" * 70)
        print()
        print("可能的原因：")
        print("  1. run_full_pipeline.py 內部有錯誤")
        print("  2. 缺少必要的依賴套件")
        print("  3. 資料檔案路徑不正確")
        print()
        print("建議：")
        print("  - 檢查 run_full_pipeline.py 的日誌輸出")
        print("  - 手動執行 'python run_full_pipeline.py' 查看詳細錯誤")
        print()

        return False


def main():
    """主函式"""
    print("=" * 70)
    print("CSV 資料重新生成工具")
    print("=" * 70)
    print()
    print("目的：生成帶有明確 sell_forced 標記的 CSV 檔案")
    print("這樣就不需要依賴「智能識別」的猜測邏輯")
    print()

    # 1. 確認操作
    if not confirm_regeneration():
        return

    # 2. 備份現有 CSV
    backup_dir = backup_existing_csv()

    if backup_dir:
        print(f"\n[INFO] 如果出現問題，可以從這裡恢復: {backup_dir}")

    # 3. 執行 pipeline
    success = run_full_pipeline()

    if success:
        print("[DONE] 全部完成！")
        print()
        print("驗證步驟：")
        print("  1. 打開任意一個 CSV（例如 sss_backtest_outputs/trades_from_results_*.csv）")
        print("  2. 查看最後一行的 'type' 欄位")
        print("  3. 如果看到 'sell_forced'，代表成功！")
        print()
    else:
        print("[FAILED] 重新生成失敗")
        print()
        if backup_dir:
            print(f"[INFO] 你可以從備份恢復: {backup_dir}")
        print()


if __name__ == "__main__":
    main()
