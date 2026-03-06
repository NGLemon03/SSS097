# metrics.py
import numpy as np
import pandas as pd

def calculate_sharpe(returns: pd.Series, rf_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """計算夏普比率"""
    if returns.empty or len(returns) < 2:
        return 0.0
    excess_returns = returns - rf_rate / periods_per_year
    mean = excess_returns.mean()
    std = excess_returns.std()
    return mean / std * np.sqrt(periods_per_year) if std > 0 else 0.0

def calculate_max_drawdown(equity: pd.Series) -> float:
    """計算最大回撤"""
    if equity.empty or len(equity) < 2:
        return 0.0
    drawdown = (equity / equity.cummax()) - 1
    return drawdown.min() if not drawdown.empty else 0.0

def calculate_profit_factor(trades: list) -> float:
    """計算盈虧因子"""
    if not trades:
        return 0.0
    gains = [t[1] for t in trades if t[1] > 0]
    losses = [abs(t[1]) for t in trades if t[1] < 0]
    return sum(gains) / sum(losses) if losses and sum(losses) > 0 else np.inf