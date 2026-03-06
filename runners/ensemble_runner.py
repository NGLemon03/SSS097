# runners/ensemble_runner.py
from __future__ import annotations
import logging
from typing import Optional
import pandas as pd

from sss_core.schemas import BacktestResult
from sss_core.normalize import normalize_trades_for_ui, normalize_daily_state

# 設置日誌
logger = logging.getLogger(__name__)

def run_ensemble_backtest(cfg, price_df: pd.DataFrame = None) -> BacktestResult:
    """
    執行 Ensemble 策略並回傳標準化的 BacktestResult

    Args:
        cfg: 配置參數
        price_df: 可選的價格數據（如果提供，Ensemble 將使用這個而不是讀 CSV）
    """
    try:
        # 導入 Ensemble 相關模組
        from SSS_EnsembleTab import run_ensemble, EnsembleParams, CostParams, RunConfig

        # 運行 ensemble 策略（傳遞 price_df 確保使用最新數據）
        open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(cfg, price_df=price_df)
        
        # 標準化資料
        trades_ui = normalize_trades_for_ui(trades)
        daily_state_std = normalize_daily_state(daily_state)
        
        # 構建 BacktestResult
        result = BacktestResult(
            equity_curve=equity,
            daily_state=daily_state_std,
            trades=trades_ui,
            ledger=trade_ledger,
            stats=stats,
            price_series=open_px,
            weight_curve=w,
            cash_curve=daily_state_std['cash'] if daily_state_std is not None and 'cash' in daily_state_std.columns else None
        )
        
        logger.info(f"[Ensemble] 執行成功: {method_name}, 權益曲線長度={len(equity)}, 交易數={len(trade_ledger) if trade_ledger is not None and not trade_ledger.empty else 0}")
        
        return result
        
    except Exception as e:
        logger.error(f"Ensemble 策略執行失敗: {e}")
        # 創建空的結果
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        
        return BacktestResult(
            equity_curve=empty_series,
            daily_state=empty_df,
            trades=empty_df,
            ledger=empty_df,
            stats={
                'total_return': 0.0, 
                'annual_return': 0.0, 
                'max_drawdown': 0.0, 
                'sharpe_ratio': 0.0, 
                'calmar_ratio': 0.0, 
                'num_trades': 0
            }
        )
