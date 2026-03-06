# init_warehouse.py
"""
策略倉庫初始化工具
===================

功能：
1. 從實驗室推薦名單或備用掃描中載入策略
2. 將策略存入倉庫（strategy_warehouse.json）
3. 支援版本標籤功能，自動建立備份快照

使用範例：
---------
# 基本使用（無標籤）：
python init_warehouse.py

# 使用標籤建立版本快照（推薦）：
python init_warehouse.py --tag OOS_2023Q4
python init_warehouse.py --tag IS_training_v2

# 備用掃描模式（無推薦名單時）：
python init_warehouse.py --top_k 10 --force_raw

標籤功能說明：
------------
- 使用 --tag 參數會自動建立兩個檔案：
  1. strategy_warehouse.json (現役版本)
  2. strategy_warehouse_<tag>_<timestamp>.json (備份快照)
- 備份快照可在 UI 中切換和激活
- 建議為每次實驗/訓練週期使用不同標籤（IS/OOS/版本號等）
"""
import argparse
import sys
from pathlib import Path
import json
import ast
import os
import pandas as pd

sys.path.append(str(Path.cwd()))
from analysis.strategy_manager import manager

RESULTS_DIR = Path("results")
TRADES_DIR = Path("sss_backtest_outputs")
RECOMMEND_FILE = Path("analysis/recommended_strategies.json")

def load_from_recommendation():
    """
    優先從實驗室推薦名單載入策略

    讀取 analysis/recommended_strategies.json，
    這個檔案通常由研究實驗室或優化流程產生。

    Returns:
        tuple: (策略列表, metadata字典)，若檔案不存在或格式錯誤則返回 (None, {})
    """
    if not RECOMMEND_FILE.exists():
        return None, {}
    try:
        with open(RECOMMEND_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        strats = data.get("strategies", [])
        meta = data.get("metadata", {})  # 🔥 讀取 Metadata
        if strats:
            print(f"📂 發現實驗室推薦名單 (共 {len(strats)} 個策略)")
            return strats, meta
    except Exception:
        pass
    return None, {}

def scan_top_k_fallback(top_k):
    """
    備用掃描：從 results/ 和 sss_backtest_outputs/ 掃描策略

    當沒有推薦名單時，掃描歷史回測結果，
    按修改時間排序，取最新的 top_k 個策略。

    Args:
        top_k (int): 要載入的策略數量

    Returns:
        list: 策略列表
    """
    print(f"⚠️  無推薦名單，執行備用掃描 (Top {top_k})...")
    param_db = {}
    for f in RESULTS_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                param_db[row['trial_number']] = ast.literal_eval(row['parameters'])
        except Exception:
            pass
    files = sorted(list(TRADES_DIR.glob("trades_from_results_*.csv")), key=os.path.getmtime, reverse=True)
    strategies = []
    for f in files:
        if len(strategies) >= top_k: break
        fname = f.stem
        import re
        match = re.search(r"trial(\d+)", fname)
        if not match: continue
        tid = int(match.group(1))
        params = param_db.get(tid)
        if not params: continue
        stype = "Unknown"
        if "RMA" in fname: stype = "RMA"
        elif "ssma_turn" in fname: stype = "ssma_turn"
        elif "single" in fname: stype = "single"
        strategies.append({"name": fname, "type": stype, "params": params})
    return strategies

def main():
    """
    主程式：執行倉庫初始化流程

    流程：
    1. 解析命令列參數
    2. 載入策略（推薦名單 > 備用掃描）
    3. 存入倉庫並建立版本快照（若有標籤）
    """
    parser = argparse.ArgumentParser(
        description="策略倉庫初始化工具 - 載入策略並建立版本快照",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  %(prog)s                              # 基本初始化（無標籤）
  %(prog)s --tag OOS_2023Q4             # 帶標籤（推薦！會建立備份快照）
  %(prog)s --tag IS_training_v2         # 訓練集版本標籤
  %(prog)s --top_k 10 --force_raw       # 強制備用掃描，取前10個策略

標籤命名建議：
  IS_<date>    : 樣本內訓練集 (In-Sample)
  OOS_<date>   : 樣本外測試集 (Out-of-Sample)
  v<number>    : 版本號 (v1, v2, v3...)
  <experiment> : 實驗名稱 (experiment_A, strategy_update_dec...)
        """
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='備用掃描模式下要載入的策略數量（預設: 5）'
    )
    parser.add_argument(
        '--force_raw',
        action='store_true',
        help='強制使用備用掃描，忽略推薦名單'
    )
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        metavar='TAG',
        help='為這次入庫加上版本標籤（例如: OOS_2023、IS_training_v2）。'
             '使用標籤會自動建立備份快照，可在 UI 中切換版本。'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 策略倉庫初始化工具")
    print("=" * 60)

    # 1. 載入策略與 Metadata
    strategies_to_save = []
    metadata = {}

    if not args.force_raw:
        strategies_to_save, metadata = load_from_recommendation()

    if not strategies_to_save:
        strategies_to_save = scan_top_k_fallback(args.top_k)
        # 如果是 fallback，我們沒有 metadata，只能留空

    # 2. 存入倉庫
    if strategies_to_save:
        print(f"\n📝 準備存入 {len(strategies_to_save)} 個策略...")

        # 顯示策略清單
        for i, s in enumerate(strategies_to_save[:5], 1):  # 只顯示前5個
            print(f"   {i}. {s.get('name', 'Unknown')[:50]}")
        if len(strategies_to_save) > 5:
            print(f"   ... 及其他 {len(strategies_to_save) - 5} 個策略")

        # 🔥 存檔時傳入 metadata
        manager.save_strategies(strategies_to_save, tag=args.tag, metadata=metadata)

        print(f"\n✅ 初始化完成！Metadata 已寫入。")
        print(f"   - 現役倉庫: strategy_warehouse.json")

        if metadata:
            print(f"   ℹ️  訓練截止日: {metadata.get('split_date', 'N/A')}")
            print(f"   ℹ️  執行模式: {metadata.get('mode', 'N/A')}")
            print(f"   ℹ️  評分模式: {metadata.get('score_mode', 'N/A')}")

        if args.tag:
            print(f"   - 備份快照: strategy_warehouse_{args.tag}.json")
            print(f"   - 🏷️  版本標籤: {args.tag}")
            print(f"\n💡 提示: 可在 Streamlit 或 Dash UI 中切換和激活不同版本")
        else:
            print(f"\n💡 提示: 下次可使用 --tag 參數建立版本快照")
            print(f"   範例: python init_warehouse.py --tag OOS_2023Q4")

    else:
        print("\n❌ 找不到任何策略。")
        print("   請確認以下目錄是否有資料：")
        print(f"   - {RECOMMEND_FILE}")
        print(f"   - {RESULTS_DIR}")
        print(f"   - {TRADES_DIR}")

    print("=" * 60)

if __name__ == "__main__":
    main()
