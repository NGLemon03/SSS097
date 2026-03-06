# -*- coding: utf-8 -*-
"""
使用 sss_core.logic 模块的示例
完全独立于 Streamlit，可在命令行或自动化脚本中使用
"""

from sss_core.logic import (
    load_data,
    compute_single,
    backtest_unified,
    calculate_metrics
)

# 示例 1: 加载数据
print("示例 1: 加载价格数据")
df_price, df_factor = load_data(
    ticker="2330.TW",
    start_date="2023-01-01",
    end_date="2024-12-31",
    smaa_source="Self"
)
print(f"加载了 {len(df_price)} 行价格数据")

# 示例 2: 计算单周期策略
print("\n示例 2: 计算单周期策略指标")
df_indicators = compute_single(
    df=df_price,
    smaa_source_df=df_factor,
    linlen=90,
    factor=40,
    smaalen=30,
    devwin=30,
    smaa_source="Self"
)
print(f"生成了 {len(df_indicators)} 行指标数据")
print(f"指标列: {list(df_indicators.columns)}")

# 示例 3: 执行回测
print("\n示例 3: 执行回测")
results = backtest_unified(
    df_ind=df_indicators,
    strategy_type="single",
    params={
        'buy_mult': 0.5,
        'sell_mult': 1.5,
        'stop_loss': 0.2,
        'min_dist': 5,
        'prom_factor': 0.5
    },
    discount=0.30,
    trade_cooldown_bars=3,
    bad_holding=False
)

# 示例 4: 查看回测结果
print("\n示例 4: 回测结果")
metrics = results['metrics']
print(f"交易次数: {metrics.get('num_trades', 0)}")
print(f"总回报率: {metrics.get('total_return', 0):.2%}")
print(f"年化回报率: {metrics.get('annual_return', 0):.2%}")
print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
print(f"胜率: {metrics.get('win_rate', 0):.2%}")

# 示例 5: 查看交易详情
print("\n示例 5: 交易详情")
trades_df = results['trade_df']
print(trades_df.head())

print("\n所有示例都使用了 sss_core.logic 模块，无需 Streamlit 依赖！")
