# run_full_pipeline.py
import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# ================= 設定區 =================
# 模式選擇:
# "OOS" = 訓練到 2023，驗證 2024-2025 (實戰推薦)
# "IS"  = 使用全部資料訓練 (適合觀察理論極限)
MODE = "OOS"

TRIALS_PER_STRATEGY = 1000  # 建議 300-500
TRAIN_END_DATE = "2023-12-31"  # OOS 模式下的訓練截止日

# Warehouse 初始化設定
# 流程完成後，如果滿意結果，可執行: python init_warehouse.py --top_k <數量> --tag <標籤>
# 這會從優化結果中挑選表現最好的 N 個策略，部署到每日交易系統
# --tag 參數會建立版本備份，方便在 UI 中切換不同訓練版本
WAREHOUSE_TOP_K = 5      # 建議 3-10，太多會稀釋個別策略權重
WAREHOUSE_TAG = "IS_v1"  # 版本標籤，建議: IS_<date> (樣本內) 或 OOS_<date> (樣本外) 或 v<number>

# 輸出檔名標籤 (用於識別這次訓練的檔案)
# 建議格式: 模式_截止日_版本 (例如: OOS_End231231_v1)
# 這會讓生成的 CSV 檔名更清楚,例如: optuna_results_single_Self_OOS_End231231_v1_timestamp.csv
OUTPUT_TAG = f"{MODE}_End{TRAIN_END_DATE.replace('-','')[2:]}_v1" if MODE == "OOS" else f"{MODE}_v1"

STRATEGIES = [
    ("single", "Self"),
    ("single", "Factor (^TWII / 2414.TW)"),
    ("single", "Factor (^TWII / 2412.TW)"),
    ("RMA", "Self"),
    ("RMA", "Factor (^TWII / 2414.TW)"),
    ("RMA", "Factor (^TWII / 2412.TW)"),
    ("ssma_turn", "Self"),
    ("ssma_turn", "Factor (^TWII / 2414.TW)"),
    ("ssma_turn", "Factor (^TWII / 2412.TW)"),
]
# =========================================

def archive_old_results():
    """將舊結果封存，而不是刪除"""
    results_dir = Path("results")
    trades_dir = Path("sss_backtest_outputs")
    
    # 檢查是否有東西需要封存
    has_files = any(results_dir.glob("*.csv")) or any(trades_dir.glob("trades_from_results_*.csv"))
    
    if has_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_root = Path("archive") / f"{timestamp}_Backup"
        archive_root.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📦 [Step 0] 發現舊資料，正在封存至: {archive_root}")

        # 封存 results
        if results_dir.exists():
            dest = archive_root / "results"
            dest.mkdir(exist_ok=True)
            for f in results_dir.glob("*.csv"):
                shutil.move(str(f), str(dest / f.name))
        
        # 封存 trades (只封存由 Optuna 產生的，保留 ensemble 產生的結果)
        if trades_dir.exists():
            dest = archive_root / "sss_backtest_outputs"
            dest.mkdir(exist_ok=True)
            for f in trades_dir.glob("trades_from_results_*.csv"):
                shutil.move(str(f), str(dest / f.name))
                
        print("   ✅ 封存完成，工作目錄已淨空。")
    else:
        print("\n✨ [Step 0] 工作目錄乾淨，無需封存。")

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🔄 [Step] {desc}")
    print(f"   Cmd: {cmd}")
    print(f"{'='*60}\n")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ 執行失敗: {desc}")

def main():
    start_time = datetime.now()
    print(f"🚀 SSS096 自動化流程啟動 - {start_time}")
    print(f"⚙️ 執行模式: {MODE}")

    # 1. 封存舊檔
    archive_old_results()

    # 2. 決定訓練參數
    date_arg = ""
    if MODE == "OOS":
        date_arg = f"--train_end_date {TRAIN_END_DATE}"
        print(f"📅 訓練資料截止於: {TRAIN_END_DATE} (嚴格執行樣本外驗證)")
    else:
        print(f"📅 使用全歷史資料訓練 (注意過度擬合風險)")

    # 3. 執行 Optuna (挖掘)
    for strat, source in STRATEGIES:
        cmd = (
            f"python analysis/optuna_16.py "
            f"--strategy {strat} "
            f"--n_trials {TRIALS_PER_STRATEGY} "
            f"--data_source \"{source}\" "
            f"--n_jobs 4 "
            f"{date_arg} "
            f"--tag {OUTPUT_TAG}"
        )
        run_cmd(cmd, f"Optuna 優化 ({MODE}): {strat}")

    # 4. 轉換交易檔 (這裡永遠跑全範圍，讓我們看驗證期的表現)
    # 不管訓練用多久，轉檔時我們都要看它在整個歷史的表現
    run_cmd("python convert_results_to_trades.py --top_k 20", "產生全歷史區間交易檔")

    # 5. 執行實驗室 (評估)
    run_cmd("python analysis/optimize_ensemble.py", "Ensemble 組合優化實驗")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n🎉 流程結束！耗時: {duration}")
    print("=" * 70)
    print(f"💡 你的舊檔案已保存在 archive/ 資料夾中。")
    print(f"💡 你的檔案現在都有標籤了: _{OUTPUT_TAG}")
    print(f"💡 如果滿意這次 {MODE} 模式的結果 (看 sss_backtest_outputs 的表現)：")
    print(f"   請執行 `python init_warehouse.py --top_k {WAREHOUSE_TOP_K} --tag {WAREHOUSE_TAG}` 來更新每日交易策略。")
    print(f"   (當前設定: 挑選表現最好的 {WAREHOUSE_TOP_K} 個策略，標籤為 '{WAREHOUSE_TAG}')")
    print(f"   💡 --tag 參數會建立版本備份，可在 UI 中切換不同訓練版本")
    print("=" * 70)

if __name__ == "__main__":
    main()