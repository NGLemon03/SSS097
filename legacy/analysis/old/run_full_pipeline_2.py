# run_full_pipeline.py
import subprocess
import sys
import os
from datetime import datetime

# ================= 設定區 =================
TRIALS_PER_STRATEGY = 1000  # 建議 300-500 次
TRAIN_END_DATE = "2025-12-20"  # 🔥 訓練截止日 (留 2024-2025 當考卷)
STRATEGIES = [
    # (策略名稱, 數據源)
    ("single", "Self"),
    ("single", "Factor (^TWII / 2414.TW)"),
    ("single", "Factor (^TWII / 2412.TW)"),
    ("RMA", "Self"),
    ("RMA", "Factor (^TWII / 2414.TW)"),
    ("RMA", "Factor (^TWII / 2412.TW)"),
    ("ssma_turn", "Self"),
    ("ssma_turn", "Factor (^TWII / 2414.TW)"), 
    ("ssma_turn", "Factor (^TWII / 2412.TW)"),
    # 你可以加更多，例如 ssma_turn + 2412
]
# =========================================

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🔄 [Step] {desc}")
    print(f"   Cmd: {cmd}")
    print(f"{'='*60}\n")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ 執行失敗: {desc}")
        # 不一定要 exit，有些步驟失敗可以繼續
        # sys.exit(1)

def main():
    start_time = datetime.now()
    print(f"🚀 SSS096 OOS 驗證流程啟動 - {start_time}")
    print(f"📅 訓練資料截止於: {TRAIN_END_DATE} (之後的資料為樣本外驗證)")

    # 1. 執行 Optuna (多策略並行或依序)
    for strat, source in STRATEGIES:
        cmd = f"python analysis/optuna_16.py  --n_jobs 4 --strategy {strat} --n_trials {TRIALS_PER_STRATEGY} --data_source \"{source}\" --train_end_date {TRAIN_END_DATE}"
        run_cmd(cmd, f"Optuna OOS優化: {strat} ({source})")

    # 2. 轉換結果為交易檔 (這裡會跑完整歷史數據，包含 2024-2025)
    # 這樣我們才能看到它在「未知領域」的表現
    run_cmd("python convert_results_to_trades.py --top_k 20", "轉換參數為交易檔 (全歷史區間)")

    # 3. 執行 Ensemble 實驗室 (挑選最佳 K)
    # 這裡我們只執行觀察，若要自動化抓取 K 值，可以修改 optimizer 輸出到檔案讀取
    # 這裡先讓它跑出來給你看
    run_cmd("python analysis/optimize_ensemble.py", "Ensemble 組合優化實驗")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n🎉 流程結束！耗時: {duration}")
    print("👉 請檢查 sss_backtest_outputs 的結果。")
    print("👉 重點觀察：2024 年之後的績效曲線是否依然向上？如果是，這才是真策略。")


if __name__ == "__main__":
    main()