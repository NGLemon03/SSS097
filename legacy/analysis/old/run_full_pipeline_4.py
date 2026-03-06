# run_full_pipeline_3.py
import subprocess
import sys
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ================= 設定區 =================
MODE = "OOS" # "OOS" (嚴謹驗證) 或 "IS" (全歷史)
TRIALS_PER_STRATEGY = 300  # 試驗次數 (建議 300-500)
TRAIN_END_DATE = "2023-12-31"  # OOS 模式下的訓練截止日 (IS 模式會忽略此項)

# 轉檔時的 Top K (候選池大小)
CONVERT_TOP_K = 20 

# 最終入庫的 Top K (正式上場數量)
WAREHOUSE_TOP_K = 5

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

def generate_auto_tag():
    """生成包含詳細資訊的自動標籤"""
    run_time = datetime.now().strftime("%m%d_%H%M") # 執行時間 (月日_時分)
    
    if MODE == "OOS":
        # 去掉橫線的日期，例如 20231231
        end_date_str = TRAIN_END_DATE.replace("-", "") 
        tag = f"OOS_TrainEnd{end_date_str}_Run{run_time}"
    else:
        tag = f"IS_Full_Run{run_time}"
    
    return tag

def archive_old_results():
    """將舊結果封存"""
    results_dir = Path("results")
    trades_dir = Path("sss_backtest_outputs")
    
    has_files = any(results_dir.glob("*.csv")) or any(trades_dir.glob("trades_from_results_*.csv"))
    
    if has_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_root = Path("archive") / f"{timestamp}_Backup"
        archive_root.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📦 [Step 0] 發現舊資料，正在封存至: {archive_root}")

        if results_dir.exists():
            dest = archive_root / "results"
            dest.mkdir(exist_ok=True)
            for f in results_dir.glob("*.csv"):
                shutil.move(str(f), str(dest / f.name))
        
        if trades_dir.exists():
            dest = archive_root / "sss_backtest_outputs"
            dest.mkdir(exist_ok=True)
            for f in trades_dir.glob("trades_from_results_*.csv"):
                shutil.move(str(f), str(dest / f.name))
        print("   ✅ 封存完成，工作目錄已淨空。")

def generate_draft_report(tag, top_k):
    """
    生成選秀報告：分析 results/ 下的 CSV，統計 Top K 的表現
    這能讓你知道這次跑出來的策略池品質如何
    """
    print(f"\n📊 [Report] 正在生成本次訓練 ({tag}) 的選秀報告...")
    results_dir = Path("results")
    files = list(results_dir.glob(f"*{tag}*.csv"))
    
    if not files:
        print("   ❌ 找不到結果檔，無法生成報告。")
        return

    report_data = []
    
    print("-" * 80)
    print(f"{'策略類型':<30} | {'平均總報酬%':<12} | {'平均 Sharpe':<12} | {'平均 MDD%':<12}")
    print("-" * 80)

    total_avg_ret = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # 確保有 value (分數) 欄位
            if 'value' not in df.columns: continue
            
            # 排序並取 Top K
            df = df.sort_values(by='value', ascending=False).head(top_k)
            
            # 計算平均數據
            avg_ret = df['total_return'].mean() * 100
            avg_sharpe = df['sharpe_ratio'].mean()
            avg_mdd = df['max_drawdown'].mean() * 100
            
            print(f"{f.stem:<30} | {avg_ret:12.2f}% | {avg_sharpe:12.2f} | {avg_mdd:12.2f}%")
            
            total_avg_ret.append(avg_ret)
            
        except Exception as e:
            print(f"   ⚠️ 無法讀取 {f.name}: {e}")

    print("-" * 80)
    if total_avg_ret:
        print(f"🏆 整體池平均報酬: {np.mean(total_avg_ret):.2f}% (這是候選人的平均素質)")
    print("-" * 80)

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🔄 [Step] {desc}")
    # print(f"   Cmd: {cmd}") 
    print(f"{'='*60}\n")
    subprocess.run(cmd, shell=True, check=True)

def main():
    start_time = datetime.now()
    
    # 1. 自動生成標籤
    TAG = generate_auto_tag()
    
    print(f"🚀 SSS096 自動化流程啟動")
    print(f"🏷️  本次任務標籤 (TAG): {TAG}")
    print(f"⚙️  模式: {MODE} | 訓練截止: {TRAIN_END_DATE if MODE=='OOS' else '全歷史'}")
    print(f"🔢 選秀池 Top K: {CONVERT_TOP_K} | 入庫 Top K: {WAREHOUSE_TOP_K}")

    # 2. 封存舊檔
    archive_old_results()

    # 3. 決定訓練參數
    date_arg = f"--train_end_date {TRAIN_END_DATE}" if MODE == "OOS" else ""

    # 4. 執行 Optuna (挖掘)
    for strat, source in STRATEGIES:
        cmd = (
            f"python analysis/optuna_16.py "
            f"--strategy {strat} "
            f"--n_trials {TRIALS_PER_STRATEGY} "
            f"--data_source \"{source}\" "
            f"--n_jobs 4 "
            f"{date_arg} "
            f"--tag {TAG}" # 傳入自動生成的標籤
        )
        run_cmd(cmd, f"Optuna 優化: {strat}")

    # 5. 轉換交易檔
    run_cmd(f"python convert_results_to_trades.py --top_k {CONVERT_TOP_K}", "產生候選交易檔")

    # 6. 🔥 生成選秀報告 (新增功能)
    generate_draft_report(TAG, CONVERT_TOP_K)

    # 7. 執行實驗室 (評估)
    run_cmd("python analysis/optimize_ensemble.py", "Ensemble 組合優化實驗")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n🎉 流程結束！耗時: {duration}")
    print("=" * 80)
    print(f"💡 你的檔案已自動標記為: _{TAG}")
    print(f"💡 滿意的話，請執行以下指令鎖定策略：")
    print(f"   python init_warehouse.py --top_k {WAREHOUSE_TOP_K} --tag {TAG}")
    print("=" * 80)

if __name__ == "__main__":
    main()