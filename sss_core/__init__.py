# sss_core/__init__.py

# ============ 核心计算逻辑 (logic.py) ============
from .logic import (
    # 数据加载与验证
    load_data,
    load_data_wrapper,
    validate_params,
    fetch_yf_data,
    is_price_data_up_to_date,
    clear_all_caches,
    force_update_price_data,

    # 技术指标计算
    calc_smaa,
    linreg_last_vectorized,
    linreg_last_original,

    # 策略计算
    compute_single,
    compute_dual,
    compute_RMA,
    compute_ssma_turn_combined,

    # 回测与指标
    backtest_unified,
    calculate_metrics,
    calculate_trade_mmds,
    calculate_holding_periods,

    # 数据类
    TradeSignal,
)

# ============ 数据结构与转换 (schemas.py) ============
from .schemas import BacktestResult, pack_df, pack_series

# ============ 数据标准化 (normalize.py) ============
from .normalize import normalize_trades_for_ui, normalize_trades_for_plots, normalize_daily_state

# ============ 绘图工具 (plotting.py) ============
from .plotting import (
    plot_weight_series,
    plot_equity_cash,
    plot_trades_on_price,
    plot_performance_metrics,
    create_combined_dashboard
)

__all__ = [
    # 核心计算逻辑
    'load_data',
    'load_data_wrapper',
    'validate_params',
    'fetch_yf_data',
    'is_price_data_up_to_date',
    'clear_all_caches',
    'force_update_price_data',
    'calc_smaa',
    'linreg_last_vectorized',
    'linreg_last_original',
    'compute_single',
    'compute_dual',
    'compute_RMA',
    'compute_ssma_turn_combined',
    'backtest_unified',
    'calculate_metrics',
    'calculate_trade_mmds',
    'calculate_holding_periods',
    'TradeSignal',

    # 数据结构
    'BacktestResult',
    'pack_df',
    'pack_series',

    # 数据标准化
    'normalize_trades_for_ui',
    'normalize_trades_for_plots',
    'normalize_daily_state',

    # 绘图工具
    'plot_weight_series',
    'plot_equity_cash',
    'plot_trades_on_price',
    'plot_performance_metrics',
    'create_combined_dashboard'
]
