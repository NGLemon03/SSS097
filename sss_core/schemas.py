# sss_core/schemas.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """統一的回測輸出結構，供單一策略與 Ensemble 都遵守"""
    # 曲線類
    equity_curve: pd.Series           # 每日權益（必填）
    trades: pd.DataFrame              # 標準欄位：trade_date, type, price, delta_units, w_before, w_after, ...
    stats: Dict[str, Any]            # total_return, annual_return, max_drawdown, ...
    
    # 可選欄位
    daily_state: Optional[pd.DataFrame] = None  # 含 equity/cash/w 等（可選，但建議提供）
    ledger: Optional[pd.DataFrame] = None       # 若有更細的流水帳（可選）
    price_series: Optional[pd.Series] = None    # 價格序列
    weight_curve: Optional[pd.Series] = None    # 權重序列
    cash_curve: Optional[pd.Series] = None      # 現金序列
    
    def __post_init__(self):
        """驗證必要欄位"""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            raise ValueError("equity_curve 是必填欄位且不能為空")
        if self.trades is None or len(self.trades) == 0:
            raise ValueError("trades 是必填欄位且不能為空")
        if self.stats is None:
            raise ValueError("stats 是必填欄位")
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式，用於序列化"""
        result = {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'stats': self.stats
        }
        
        # 添加可選欄位
        if self.daily_state is not None:
            result['daily_state'] = self.daily_state
        if self.ledger is not None:
            result['ledger'] = self.ledger
        if self.price_series is not None:
            result['price_series'] = self.price_series
        if self.weight_curve is not None:
            result['weight_curve'] = self.weight_curve
        if self.cash_curve is not None:
            result['cash_curve'] = self.cash_curve
            
        return result


def pack_df(df: pd.DataFrame) -> str:
    """將 DataFrame 序列化為 JSON 字串，使用 orient="split" + date_format="iso"
    
    Args:
        df: 要序列化的 DataFrame
        
    Returns:
        JSON 字串，空 DataFrame 回傳空字串
    """
    if df is None or len(df) == 0:
        return ""
    return df.to_json(orient="split", date_format="iso")


def pack_series(s: pd.Series) -> str:
    """將 Series 序列化為 JSON 字串，使用 orient="split" + date_format="iso"
    
    Args:
        s: 要序列化的 Series
        
    Returns:
        JSON 字串，空 Series 回傳空字串
    """
    if s is None or len(s) == 0:
        return ""
    return s.to_json(orient="split", date_format="iso")
