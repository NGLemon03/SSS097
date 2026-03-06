# 投资组合流水帐（Portfolio Ledger）功能说明

## 概述

在 `SSS_EnsembleTab.py` 中新增了「投资组合流水帐（ledger）」功能，用于追踪每日资产状态和交易明细。

## 新增功能

### 1. `build_portfolio_ledger()` 函数

```python
def build_portfolio_ledger(open_px, w, cost: CostParams, initial_capital=1_000_000.0, lot_size=None):
    """
    依照每日 open 價與目標權重 w_t（含 floor、delta_cap 等限制後的最終 w_t），
    產出兩個 DataFrame：
      1) daily_state: 每日現金/持倉/總資產/權重
      2) trades: 只有權重變動日的交易明細（買賣金額、費用、稅、交易後資產）
    """
```

**输入参数：**
- `open_px`: 开盘价序列
- `w`: 目标权重序列（已包含 floor、ema_span、delta_cap、cooldown 等限制）
- `cost`: 成本参数（CostParams）
- `initial_capital`: 初始资金（默认 1,000,000）
- `lot_size`: 整股单位（如 100 或 1000，None 表示允许零股）

**新增配置参数：**
- `initial_capital`: 可在 `RunConfig` 中设置初始资金
- `lot_size`: 可在 `RunConfig` 中设置整股单位，支持 CLI 和 Streamlit UI 配置

**输出：**
- `daily_state`: 每日资产状态表
- `trade_ledger`: 交易明细表

### 2. 每日资产状态表 (daily_state)

包含以下字段：
- `date`: 日期
- `open`: 开盘价
- `w_prev`: 前一日权重
- `w`: 当前权重
- `dw`: 权重变化
- `units`: 持仓单位数
- `cash_before`: 交易前现金（可选）
- `position_value_before`: 交易前持仓价值
- `equity_before`: 交易前总资产
- `cash`: 现金余额
- `position_value`: 持仓价值
- `equity`: 总资产
- `cash_pct`: 现金比例
- `invested_pct`: 投资比例
- `actual_w`: 实际权重（持仓价值/总资产）

### 3. 交易明细表 (trade_ledger)

包含以下字段：
- `date`: 交易日期
- `open`: 开盘价
- `side`: 交易方向（BUY/SELL/HOLD）
- `w_prev`: 前一日权重
- `w`: 当前权重
- `dw`: 权重变化
- `delta_units`: 单位数变化
- `exec_notional`: 执行金额
- `fee_buy`: 买进费用
- `fee_sell`: 卖出费用
- `tax`: 证交税
- `fees_total`: 总费用（买进费用+卖出费用+证交税）
- `trade_pct`: 交易比例（执行金额/交易前总资产）
- `cash_after`: 交易后现金
- `position_value_after`: 交易后持仓价值
- `equity_after`: 交易后总资产
- `actual_w_after`: 交易后实际权重

## 使用方法

### 1. 在 `run_ensemble()` 中自动生成

```python
# 建立投資組合流水帳（ledger）
daily_state, trade_ledger = build_portfolio_ledger(
    open_px=px["open"],
    w=w,                            # 最終權重序列
    cost=cfg.cost,                  # CostParams
    initial_capital=cfg.initial_capital,  # 從配置讀取初始資金
    lot_size=cfg.lot_size or None   # 從配置讀取整股單位
)

# 自动保存到 results/ 目录
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
outdir = Path("results")
outdir.mkdir(exist_ok=True, parents=True)
name_tag = f"{method_name}_{ts}"
daily_state.to_csv(outdir / f"ensemble_daily_state_{name_tag}.csv", encoding="utf-8-sig")
trade_ledger.to_csv(outdir / f"ensemble_trade_ledger_{name_tag}.csv", encoding="utf-8-sig")
```

### 2. 手动调用

```python
from SSS_EnsembleTab import build_portfolio_ledger, CostParams

# 创建成本参数
cost = CostParams(
    buy_fee_bp=4.27,    # 买进费率 4.27 bp
    sell_fee_bp=4.27,   # 卖出费率 4.27 bp
    sell_tax_bp=30.0    # 卖出证交税 30 bp
)

# 生成流水帐
daily_state, trade_ledger = build_portfolio_ledger(
    open_px=your_open_prices,
    w=your_weights,
    cost=cost,
    initial_capital=1_000_000.0,
    lot_size=None
)
```

### 3. 配置参数

#### CLI 参数
```bash
python SSS_EnsembleTab.py --ticker 00631L.TW --method majority \
    --initial_capital 2000000 --lot_size 100
```

#### Streamlit UI 参数
- **初始资金**：可设置 100,000 到 10,000,000 之间的任意值
- **整股单位**：可选择"允许零股"、"100股"、"1000股"

## 输出文件

### 1. 每日资产状态表
- 文件名：`ensemble_daily_state_{method_name}_{timestamp}.csv`
- 位置：`results/` 目录
- 用途：追踪每日资产配置状态

### 2. 交易明细表
- 文件名：`ensemble_trade_ledger_{method_name}_{timestamp}.csv`
- 位置：`results/` 目录
- 用途：记录所有交易明细，包含成本计算

## Streamlit UI 增强

在 Streamlit 界面中新增了以下显示：

### 1. 投资组合流水帐摘要
- 当前权重和权重变化
- 现金比例和投入比例
- 现金、持仓价值、总资产
- 持仓单位数

### 2. 最新交易信息
- 交易方向、日期、价格
- 权重变化、交易金额
- 各项费用明细

### 3. 数据表格
- 每日资产表
- 交易明细表

### 4. 导出功能
- 一键下载交易流水帐 CSV 文件

## 重要特性

### 1. 成本计算
- 买进费用：`buy_fee_bp / 10000.0 * 交易金额`
- 卖出费用：`sell_fee_bp / 10000.0 * 交易金额`
- 证交税：`sell_tax_bp / 10000.0 * 交易金额`
- 总费用：`fees_total = fee_buy + fee_sell + tax`

### 2. 新增字段
- **交易前状态**：`cash_before`、`position_value_before`、`equity_before`
- **实际权重**：`actual_w = position_value / equity`（交易后实际持仓比例）
- **交易比例**：`trade_pct = exec_notional / equity_before`（单笔交易相对规模）
- **总费用**：`fees_total`（所有费用的汇总）

### 3. 权重对齐
- 使用经过所有限制（floor、ema_span、delta_cap、cooldown）后的最终权重
- 确保与现有回测逻辑完全一致

### 4. 时间对齐
- 严格遵循 T+1 规则
- 以次日开盘价执行交易
- 与现有 `equity_open_to_open` 逻辑保持一致

### 5. 整股支持
- 通过 `lot_size` 参数支持整股交易
- 默认允许零股交易（适合 ETF 等）
- 支持 CLI 和 Streamlit UI 配置

## 注意事项

1. **成本不重复计算**：流水帐中的成本计算与现有回测逻辑保持一致
2. **权重来源**：使用最终权重序列，包含所有限制和调整
3. **文件命名**：使用时间戳确保文件名唯一性
4. **编码格式**：使用 UTF-8-SIG 编码，支持中文显示

## 示例输出

### 每日资产状态表示例
```csv
date,open,w_prev,w,dw,units,cash_before,position_value_before,equity_before,cash,position_value,equity,cash_pct,invested_pct,actual_w
2024-01-01,100.0,0.0,0.2,0.2,2000.0,800000.0,0.0,1000000.0,799914.6,200000.0,999914.6,0.799983,0.200017,0.200017
2024-01-02,101.0,0.2,0.4,0.2,3967.98,601148.76,202000.0,1001914.6,601063.89,400765.84,1001829.73,0.599966,0.400034,0.400034
```

### 交易明细表示例
```csv
date,open,side,w_prev,w,dw,delta_units,exec_notional,fee_buy,fee_sell,tax,fees_total,trade_pct,cash_after,position_value_after,equity_after,actual_w_after
2024-01-01,100.0,BUY,0.0,0.2,0.2,2000.0,200000.0,85.4,0.0,0.0,85.4,0.2,799914.6,200000.0,999914.6,0.200017
2024-01-05,98.0,SELL,0.8,0.5,-0.3,-2936.04,287732.36,0.0,122.86,863.20,986.06,0.294,489038.53,490024.59,979063.12,0.500504
```

## 总结

新增的投资组合流水帐功能提供了完整的资产追踪和交易记录，使得：

1. **资产状态透明化**：每日现金、持仓、总资产一目了然
2. **交易明细完整化**：包含所有交易的成本和影响
3. **数据导出标准化**：统一的 CSV 格式，便于后续分析
4. **UI 展示增强化**：直观的摘要卡片和详细表格
5. **成本计算精确化**：精确的手续费和税费计算

这些功能为投资组合管理和风险控制提供了强有力的支持。
