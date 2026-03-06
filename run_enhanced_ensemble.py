
# run_enhanced_ensemble.py
"""
Ensemble 增強版執行腳本 (支援 Walk-Forward) - Fixed Version
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import random

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 引用核心邏輯
try:
    from SSS_EnsembleTab import run_ensemble, RunConfig, EnsembleParams, CostParams
except ImportError:
    print("❌ 錯誤：找不到 SSS_EnsembleTab.py")
    sys.exit(1)

def run_walk_forward_analysis(args):
    """執行滾動式回測"""
    logger.info("🚀 啟動 Walk-Forward 滾動回測模式...")
    
    trades_dir = Path(args.trades_dir)
    all_files = list(trades_dir.glob("trades_from_results_*.csv"))
    
    if not all_files:
        logger.error(f"❌ 在 {trades_dir} 找不到任何策略檔案！")
        return

    logger.info(f"找到 {len(all_files)} 個候選策略檔案")
    
    # 建立 File Map
    file_map = {}
    for f in all_files:
        name = f.stem.replace("trades_from_results_", "").replace("trades_", "")
        file_map[name] = f

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    for year in years:
        logger.info(f"\n--- 正在處理年份: {year} ---")
        
        # 模擬選將：隨機選取 Top K (在正式版中應基於歷史績效選取)
        current_selection_files = random.sample(all_files, min(args.top_k, len(all_files)))
        selected_strategies = [f.stem.replace("trades_from_results_", "").replace("trades_", "") for f in current_selection_files]
        
        # 1. 設定參數
        ens_params = EnsembleParams(
            floor=0.2, ema_span=3, delta_cap=0.1,
            majority_k=max(1, int(len(selected_strategies)/2) + 1),
            min_cooldown_days=5, min_trade_dw=0.02
        )
        cost_params = CostParams(buy_fee_bp=4.275, sell_fee_bp=4.275, sell_tax_bp=30.0)
        
        # 2. 設定 Config (修正：移除 trades_dir，加入 file_map)
        config = RunConfig(
            ticker="00631L.TW",
            method=args.method,
            strategies=selected_strategies,
            params=ens_params,
            cost=cost_params,
            file_map=file_map
        )
        
        try:
            # 3. 執行 Ensemble
            res = run_ensemble(config)
            # 解包回傳值
            open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = res
            
            # 4. 計算該年績效
            start_dt = f"{year}-01-01"
            end_dt = f"{year}-12-31"
            
            if equity is not None and not equity.empty:
                equity.index = pd.to_datetime(equity.index)
                mask = (equity.index >= start_dt) & (equity.index <= end_dt)
                equity_year = equity[mask]
                
                if not equity_year.empty:
                    ret = (equity_year.iloc[-1] / equity_year.iloc[0]) - 1
                    logger.info(f"  ✅ 年份 {year} 回測成功: 報酬率 {ret*100:.2f}% (天數: {len(equity_year)})")
                else:
                    logger.warning(f"  ⚠️ 年份 {year} 執行成功但該年份無數據")
            else:
                logger.warning(f"  ⚠️ 年份 {year} 無交易或資料不足")
                
        except Exception as e:
            logger.error(f"  ❌ 年份 {year} 執行失敗: {e}")

    logger.info("\n🎉 Walk-Forward 全部完成！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='majority')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--trades_dir', type=str, default='sss_backtest_outputs')
    parser.add_argument('--walk_forward', action='store_true')
    # 忽略舊參數
    parser.add_argument('--scan_params', action='store_true')
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--discount', type=float)
    parser.add_argument('--no_cost', action='store_true')

    args = parser.parse_args()
    
    if args.walk_forward:
        run_walk_forward_analysis(args)
    else:
        logger.info("請加上 --walk_forward 參數執行")

if __name__ == "__main__":
    main()
