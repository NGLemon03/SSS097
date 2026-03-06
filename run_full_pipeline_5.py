# run_full_pipeline_3.py
import subprocess
import sys
import os
import shutil
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ================= 設定區 =================
MODE = "OOS"  # "OOS" or "IS"
TRIALS_PER_STRATEGY = 2000
TRAIN_END_DATE = "2025-06-30"

# 🔥 新增：評分模式選擇
# "balanced" = 原始平衡 (Sharpe 優先)
# "smart_bh" = 穩健生存 (低 MDD 優先)
# "alpha"    = 超額報酬 (總報酬優先)
SCORE_MODE = "smart_bh" 

CONVERT_TOP_K = 20 
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
    run_time = datetime.now().strftime("%m%d_%H%M")
    
    # 將模式名稱簡化放入 Tag
    mode_map = {"balanced": "Bal", "smart_bh": "Safe", "alpha": "Aggr"}
    mode_str = mode_map.get(SCORE_MODE, "Unk")
    
    if MODE == "OOS":
        end_date_str = TRAIN_END_DATE.replace("-", "") 
        # 例如: OOS_Safe_End20231231_Run1229_1030
        tag = f"OOS_{mode_str}_End{end_date_str}_Run{run_time}"
    else:
        tag = f"IS_{mode_str}_Full_Run{run_time}"
    
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
    """生成選秀報告"""
    print(f"\n📊 [Report] 正在生成本次訓練 ({tag}) 的選秀報告...")
    results_dir = Path("results")
    files = list(results_dir.glob(f"*{tag}*.csv"))
    
    if not files:
        print("   ❌ 找不到結果檔，無法生成報告。")
        return

    print("-" * 90)
    print(f"{'策略類型':<35} | {'報酬%':<8} | {'Sharpe':<8} | {'MDD%':<8} | {'Win%':<8}")
    print("-" * 90)

    total_avg_ret = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'value' not in df.columns: continue
            
            df = df.sort_values(by='value', ascending=False).head(top_k)
            
            avg_ret = df['total_return'].mean() * 100
            avg_sharpe = df['sharpe_ratio'].mean()
            avg_mdd = df['max_drawdown'].mean() * 100
            
            # 嘗試讀取勝率 (有些舊檔可能沒有)
            if 'win_rate' in df.columns:
                avg_win = df['win_rate'].mean() * 100
                win_str = f"{avg_win:6.1f}%"
            else:
                win_str = "   N/A "

            print(f"{f.stem:<35} | {avg_ret:8.2f}% | {avg_sharpe:8.2f} | {avg_mdd:8.2f}% | {win_str}")
            total_avg_ret.append(avg_ret)
            
        except Exception as e:
            pass

    print("-" * 90)
    if total_avg_ret:
        print(f"🏆 整體池平均報酬: {np.mean(total_avg_ret):.2f}%")
    print("-" * 90)

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🔄 [Step] {desc}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, shell=True, check=True)

def main():
    start_time = datetime.now()

    # 🔥 防呆檢查：如果拼錯字，直接報錯停止
    valid_modes = ["OOS", "IS"]
    if MODE not in valid_modes:
        print(f"❌ 設定錯誤！MODE 必須是 {valid_modes} 其中之一。")
        print(f"   你目前輸入的是: '{MODE}' (是不是拼錯了?)")
        sys.exit(1)

    # 1. 自動生成標籤
    TAG = generate_auto_tag()
    
    print(f"🚀 SSS096 自動化流程啟動")
    print(f"🏷️  本次任務標籤 (TAG): {TAG}")
    print(f"🎯 評分模式: {SCORE_MODE}")
    print(f"⚙️  模式: {MODE} | 訓練截止: {TRAIN_END_DATE if MODE=='OOS' else '全歷史'}")

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
            f"--tag {TAG} "
            f"--score_mode {SCORE_MODE}" # 🔥 傳入評分模式
        )
        run_cmd(cmd, f"Optuna 優化: {strat}")

    # 5. 轉換交易檔
    run_cmd(f"python convert_results_to_trades.py --top_k {CONVERT_TOP_K}", "產生候選交易檔")

    # 6. 生成選秀報告
    generate_draft_report(TAG, CONVERT_TOP_K)

    # 7. 執行實驗室 (評估) - 🔥 修改這行，傳入參數
    cmd = (
        f"python analysis/optimize_ensemble.py "
        f"--split_date {TRAIN_END_DATE} "
        f"--mode {MODE} "
        f"--score_mode {SCORE_MODE}"
    )
    run_cmd(cmd, "Ensemble 組合優化實驗 (含 Metadata 傳遞)")

    # 🔥 8. 自動初始化倉庫 (入庫)
    # 從推薦名單自動載入策略並存入倉庫，帶上完整的 TAG 和 Metadata
    cmd = f"python init_warehouse.py --tag {TAG}"
    run_cmd(cmd, "策略入庫 (含 TAG 和 Metadata)")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n🎉 全流程完成！耗時: {duration}")
    print("=" * 80)
    print(f"✅ 策略已自動入庫，標籤: {TAG}")
    print(f"📁 現役倉庫: analysis/strategy_warehouse.json")
    print(f"📦 備份快照: analysis/warehouse_{TAG}.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
