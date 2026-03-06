# SSS096 量化交易系统 - 综合使用指南

> **工业级量化策略优化与管理系统**
> 版本: v3.0 | 更新日期: 2026-01-05

---

## 📚 目录

1. [系统架构](#系统架构)
2. [核心脚本说明](#核心脚本说明)
3. [完整工作流程](#完整工作流程)
4. [实战测试结果](#实战测试结果)
5. [最佳实践](#最佳实践)
6. [故障排除](#故障排除)

---

## 🏗️ 系统架构

### 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│  1. run_full_pipeline_5.py                                      │
│     策略训练与优化 (全自动)                                       │
│                                                                 │
│  设置 → Optuna 优化 → 转换交易文件 → Ensemble 评估 → 入库      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. manage_warehouse.py                                         │
│     策略仓库管理                                                 │
│                                                                 │
│  查看仓库列表 → 切换现役版本 → 元数据查询                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. run_oos_analysis.py                                         │
│     样本外验证 (OOS 回测)                                        │
│                                                                 │
│  加载仓库策略 → OOS 回测 → 生成绩效报告                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. run_batch_smart_leverage.py                                 │
│     Smart Leverage 批量测试                                     │
│                                                                 │
│  遍历所有仓库 → 计算 Smart Leverage → 生成排行榜                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. analyze_alpha_decay.py                                      │
│     Alpha 衰退分析                                              │
│                                                                 │
│  自动读取元数据 → 计算 Alpha 曲线 → 判断策略生命周期            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 核心脚本说明

### 1️⃣ `run_full_pipeline_5.py` - 全自动优化流程

**功能：** 一键完成从参数优化到策略入库的完整流程

**配置参数：**
```python
MODE = "OOS"                    # OOS 或 IS
TRAIN_END_DATE = "2025-01-30"   # 训练截止日
SCORE_MODE = "smart_bh"         # 评分模式: smart_bh/alpha/balanced
TRIALS_PER_STRATEGY = 1000      # 每个策略的优化次数
```

**执行流程：**
```bash
python run_full_pipeline_5.py
```

**自动完成：**
1. ✅ 封存旧数据（避免覆盖）
2. ✅ Optuna 优化（9 个策略类型）
3. ✅ 转换交易文件（Top K 试验）
4. ✅ Ensemble 组合优化
5. ✅ **自动入库**（带完整 TAG 和 Metadata）

**输出示例：**
```
🏷️  本次任务标签 (TAG): OOS_Safe_End20250130_Run0102_1430
🎯 评分模式: smart_bh
⚙️  模式: OOS | 训练截止: 2025-01-30

✅ 策略已自动入库，标签: OOS_Safe_End20250130_Run0102_1430
📁 现役仓库: analysis/strategy_warehouse.json
📦 备份快照: analysis/warehouse_OOS_Safe_End20250130_Run0102_1430.json
```

---

### 2️⃣ `manage_warehouse.py` - 策略仓库管理工具

**功能：** 查看、切换、对比不同版本的策略仓库

**基本用法：**
```bash
# 1. 查看所有仓库
python manage_warehouse.py

# 2. 切换现役仓库
python manage_warehouse.py --activate 3
```

**输出示例：**
```
📦 发现 13 个备份存档 (按时间排序):
-----------------------------------------------------------
ID  | 建立时间          | 档名 (Tag)
-----------------------------------------------------------
0   | 2026-01-05 22:12 | OOS_Safe_End20250630_Run0105_1914
    └─ 训练截止: 2025-06-30 | 模式: smart_bh
1   | 2026-01-05 19:07 | OOS_Aggr_End20250630_Run0105_1607
    └─ 训练截止: 2025-06-30 | 模式: alpha
2   | 2026-01-05 14:20 | OOS_Aggr_End20250630_Run0105_1131
    └─ 训练截止: 2025-06-30 | 模式: alpha
...

请输入要切换的仓库 ID (或按 Enter 跳过): 0
✅ 已将 warehouse_OOS_Safe_End20250630_Run0105_1914.json 设置为现役仓库！
```

**应用场景：**
- 📊 对比不同训练日期的策略表现
- 🔄 快速切换到历史最佳版本
- 🔍 查询策略的训练参数

---

### 3️⃣ `run_oos_analysis.py` - 样本外验证

**功能：** 使用现役仓库的策略进行完整的 OOS 回测

**执行：**
```bash
python run_oos_analysis.py
```

**自动流程：**
1. ✅ 从 `strategy_warehouse.json` 读取策略
2. ✅ 自动读取训练截止日（从 metadata）
3. ✅ OOS 区间回测（训练截止日之后）
4. ✅ 生成 Ensemble 组合结果
5. ✅ 输出绩效报告

**输出文件：**
```
sss_backtest_outputs/
  ├─ trades_Ensemble_Majority_oos.csv
  └─ trades_Ensemble_Proportional_oos.csv
```

---

### 4️⃣ `run_batch_smart_leverage.py` - Smart Leverage 批量测试

**功能：** 遍历所有仓库，计算 Smart Leverage 模式下的绩效，生成排行榜

**执行：**
```bash
python run_batch_smart_leverage.py
```

**工作原理：**
```python
# 对每个仓库：
1. 激活该仓库
2. 运行 OOS 回测
3. 计算 Smart Leverage (00631L + 0050)
4. 记录绩效指标
5. 恢复原仓库
```

**输出示例：**
```
🏆 Smart Leverage (00631L + 0050) 批次评比
📉 B&H 基准: 报酬 1881.0% | MDD -55.1%
----------------------------------------------------------------------------------------------------
Tag Info (模式_评分_截止日)                | 总报酬 %   | MDD %    | 档名
----------------------------------------------------------------------------------------------------
OOS_smart_bh_End2025-03-30                 | 2403.7%    | -51.9%   | warehouse_OOS_Safe_End20250330...
OOS_smart_bh_End2024-12-31                 | 2348.9%    | -52.5%   | warehouse_OOS_Safe_End20241231...
OOS_smart_bh_End2023-12-31                 | 2346.4%    | -53.3%   | warehouse_OOS_Safe_End20231231...
OOS_alpha_End2025-06-30                    | 2195.8%    | -50.2%   | warehouse_OOS_Aggr_End20250630...
...
```

**关键指标：**
- **总报酬**：越高越好（相对于 B&H 基准）
- **MDD**：最大回撤，越小（绝对值）越好
- **超额报酬**：策略报酬 - B&H 报酬

---

### 5️⃣ `analyze_alpha_decay.py` - Alpha 衰退分析

**功能：** 诊断策略的生命周期，判断何时需要重新训练

**执行：**
```bash
# 自动模式（从仓库读取训练日期）
python analyze_alpha_decay.py

# 手动指定训练日期
python analyze_alpha_decay.py --split_date 2025-01-30
```

**分析内容：**
1. **Alpha 曲线**：策略相对大盘的累计超额报酬
2. **滚动 Alpha**：短期（60 天）超额绩效动能
3. **衰退警示点**：Alpha 从峰值回落 >10% 的时间点
4. **生命周期诊断**：

```
📊 策略生命周期诊断报告
----------------------------------------------------------
✅ 有效增长期: 287 天 (Alpha 持续创新高直到 2025-10-15)
⚠️ 停滞期: 最近 82 天 Alpha 未创新高

🟡 结论: 策略动能减弱，请密切观察或准备重练。
```

**判断标准：**
- 🟢 停滞期 < 90 天：策略仍强势
- 🟡 停滞期 90-180 天：准备重练
- 🔴 停滞期 > 180 天：立即重练

---

## 🚀 完整工作流程

### Scenario 1: 新策略训练与部署

```bash
# Step 1: 设定参数并运行全流程
# 编辑 run_full_pipeline_5.py
MODE = "OOS"
TRAIN_END_DATE = "2025-06-30"
SCORE_MODE = "smart_bh"

python run_full_pipeline_5.py
# ✅ 自动完成训练、评估、入库

# Step 2: 验证策略表现
python run_oos_analysis.py
# ✅ OOS 回测，查看真实表现

# Step 3: 测试 Smart Leverage
python run_batch_smart_leverage.py
# ✅ 对比现金躺平 vs 买0050

# Step 4: 分析策略生命周期
python analyze_alpha_decay.py
# ✅ 判断策略是否仍有效
```

---

### Scenario 2: 版本管理与回滚

```bash
# Step 1: 查看所有历史版本
python manage_warehouse.py

# Step 2: 切换到历史最佳版本
python manage_warehouse.py --activate 3

# Step 3: 验证该版本表现
python run_oos_analysis.py

# Step 4: 如果满意，保持现状；不满意，再切换
python manage_warehouse.py --activate 0
```

---

### Scenario 3: 批量对比不同配置

```bash
# 训练多个版本（不同评分模式）
# 版本 A: Safe 模式
SCORE_MODE = "smart_bh"
python run_full_pipeline_5.py

# 版本 B: Aggr 模式
SCORE_MODE = "alpha"
python run_full_pipeline_5.py

# 版本 C: Balanced 模式
SCORE_MODE = "balanced"
python run_full_pipeline_5.py

# 一键批量测试所有版本
python run_batch_smart_leverage.py
# ✅ 自动生成排行榜，找出最优配置
```

---

## 📊 实战测试结果

### 测试环境
- **标的资产**: 00631L.TW (元大台灣50正2)
- **防守资产**: 0050.TW (元大台灣50)
- **B&H 基准**: 买入持有 00631L，报酬 1881.0%，MDD -55.1%
- **测试期间**: 约 2010-2026
- **仓库数量**: 13 个不同配置

### Top 5 最佳配置（Smart Leverage 模式）

| 排名 | TAG | 训练截止日 | 评分模式 | 总报酬 | MDD | 超额报酬 |
|------|-----|-----------|---------|--------|-----|----------|
| 🥇 | OOS_Safe_End20250330 | 2025-03-30 | smart_bh | **2403.7%** | -51.9% | +522.7% |
| 🥈 | OOS_Safe_End20241231 | 2024-12-31 | smart_bh | **2348.9%** | -52.5% | +467.9% |
| 🥉 | OOS_Safe_End20231231 | 2023-12-31 | smart_bh | **2346.4%** | -53.3% | +465.4% |
| 4 | OOS_Aggr_End20250630 | 2025-06-30 | alpha | **2195.8%** | -50.2% | +314.8% |
| 5 | OOS_Safe_End20250430 | 2025-04-30 | smart_bh | **2113.7%** | -52.1% | +232.7% |

### 关键发现

#### 1️⃣ Safe 模式 (smart_bh) 表现优异
- ✅ **Top 3 全部是 Safe 模式**
- ✅ 平均报酬 2366.3%，MDD -52.6%
- ✅ 相对 B&H 平均超额报酬 +485%

#### 2️⃣ Aggr 模式 (alpha) 风险略高
- ⚠️ 报酬范围：1935.8% - 2195.8%
- ⚠️ 最佳 MDD -50.2%，最差 -54.4%
- ⚠️ 波动较大，适合激进投资者

#### 3️⃣ 训练截止日影响显著
- 📅 2025-03-30 训练 → 最佳报酬 2403.7%
- 📅 2023-12-31 训练 → 报酬 2346.4%（仍优秀）
- 💡 **结论**：需要定期更新训练数据

#### 4️⃣ Smart Leverage 效果明显
- 🔥 所有配置均超越 B&H 基准
- 🔥 平均超额报酬 +300% 以上
- 🔥 MDD 普遍改善 2-5%

---

## 💡 最佳实践

### 1️⃣ 训练频率建议

```
┌─────────────────────────────────────────┐
│  每季度重新训练 (推荐)                    │
│                                         │
│  Q1: 3/31 截止 → 训练 → 4/1 上线         │
│  Q2: 6/30 截止 → 训练 → 7/1 上线         │
│  Q3: 9/30 截止 → 训练 → 10/1 上线        │
│  Q4: 12/31 截止 → 训练 → 1/1 上线        │
└─────────────────────────────────────────┘
```

**理由：**
- ✅ 及时捕捉市场变化
- ✅ Alpha 衰退期控制在 90 天内
- ✅ 避免过度拟合（太频繁训练）

---

### 2️⃣ 评分模式选择

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 保守型投资者 | `smart_bh` | 低 MDD 优先，报酬稳定 |
| 激进型投资者 | `alpha` | 追求最大报酬，容忍高回撤 |
| 平衡型投资者 | `balanced` | Sharpe 优先，风险调整后收益 |
| 不确定时 | `smart_bh` | **根据测试结果，Safe 模式最优** |

---

### 3️⃣ 仓库管理策略

```bash
# 命名规范（自动生成）
warehouse_{MODE}_{ScoreMode}_{EndDate}_{RunTime}.json

# 保留策略
├─ 每季度保留 2 个版本（Safe + Aggr）
├─ 年度最佳版本永久保留
└─ 其他版本可定期清理（>6 个月且表现不佳）

# 快速切换
# 策略失效时立即切换到上一季度版本
python manage_warehouse.py --activate <best_id>
```

---

### 4️⃣ 监控与维护

**每月检查清单：**
- [ ] 运行 `analyze_alpha_decay.py` 检查 Alpha 状态
- [ ] 运行 `run_oos_analysis.py` 查看 OOS 绩效
- [ ] 查看日志文件：`analysis/log/app/app_*.log`
- [ ] 备份仓库文件到云端/外部存储

**警示信号：**
- 🔴 Alpha 停滞期 > 180 天 → 立即重练
- 🟡 连续 3 个月 MDD 恶化 > 5% → 准备切换版本
- 🟢 Alpha 持续创新高 → 保持现状

---

## 🔧 故障排除

### 问题 1: `ImportError: No module named 'sss_core'`

**解决方案：**
```bash
# 确保在项目根目录执行
cd c:\Stock_reserach\002g
python run_full_pipeline_5.py
```

---

### 问题 2: 仓库切换后策略不更新

**解决方案：**
```bash
# 重启 Dash UI
# Ctrl+C 停止，然后重新运行
python app_dash.py
```

---

### 问题 3: Smart Leverage 计算失败

**检查点：**
```bash
# 1. 确认 0050 数据存在
ls data/0050.TW_data_raw.csv

# 2. 手动下载（如果缺失）
python -c "import yfinance as yf; yf.download('0050.TW', start='2010-01-01').to_csv('data/0050.TW_data_raw.csv')"

# 3. 查看日志
cat analysis/log/app/app_*.log | grep "Smart Leverage"
```

---

### 问题 4: Metadata 读取失败

**症状：**
```
❌ 无法自动侦测 Split Date，请手动输入 --split_date
```

**解决方案：**
```bash
# 方案 1: 手动指定日期
python analyze_alpha_decay.py --split_date 2025-01-30

# 方案 2: 重新入库（会自动写入 metadata）
python init_warehouse.py --tag Fix_Metadata
```

---

## 📈 性能优化建议

### 1️⃣ Optuna 优化加速

```python
# run_full_pipeline_5.py
TRIALS_PER_STRATEGY = 500  # 降低试验次数（快速测试）
# 或
TRIALS_PER_STRATEGY = 2000 # 提高试验次数（精细优化）
```

### 2️⃣ 并行化处理

```python
# analysis/optuna_16.py
parser.add_argument('--n_jobs', type=int, default=8)  # 增加并行核心数
```

### 3️⃣ 缓存管理

```bash
# 定期清理旧的 Optuna 结果
rm -rf results/*_old.csv
rm -rf archive/*_Backup  # 保留最近 3 个月即可
```

---

## 🎓 进阶技巧

### 技巧 1: 批量测试不同训练截止日

```bash
# 创建批处理脚本 batch_train.sh
for date in "2024-12-31" "2025-03-31" "2025-06-30"
do
  # 修改配置
  sed -i "s/TRAIN_END_DATE = .*/TRAIN_END_DATE = \"$date\"/" run_full_pipeline_5.py

  # 运行训练
  python run_full_pipeline_5.py
done

# 批量测试
python run_batch_smart_leverage.py
```

---

### 技巧 2: 自动化 Alpha 监控

```bash
# 创建定时任务（每周一早上 9:00）
# crontab -e
0 9 * * 1 cd /path/to/002g && python analyze_alpha_decay.py > alpha_report.txt && mail -s "Alpha Report" your@email.com < alpha_report.txt
```

---

### 技巧 3: 策略对冲组合

```python
# 同时激活两个仓库的策略
# 例如：Safe 模式 70% + Aggr 模式 30%
# 在 app_dash.py 中手动混合策略参数
```

---

## 📞 支持与反馈

### 日志位置
```
analysis/log/app/app_YYYYMMDD_HHMMSS.log
```

### 关键指标查询
```bash
# 查看最近的回测记录
tail -100 analysis/log/app/app_*.log | grep "总报酬"

# 查看策略入库记录
grep "已更新现役仓库" analysis/log/app/app_*.log
```

---

## 🎯 总结

### 核心优势

1. ✅ **全自动化流程** - 从训练到部署一键完成
2. ✅ **元数据追溯** - 每个策略都有完整的身世信息
3. ✅ **版本管理** - 轻松切换和对比不同配置
4. ✅ **Smart Leverage** - 显著提升收益，降低风险
5. ✅ **Alpha 监控** - 自动诊断策略生命周期

### 推荐工作流

```
每季度:
  1. python run_full_pipeline_5.py        (训练新策略)
  2. python run_batch_smart_leverage.py   (批量测试)
  3. python manage_warehouse.py           (切换最佳版本)

每月:
  4. python analyze_alpha_decay.py        (检查 Alpha)

每周:
  5. python run_oos_analysis.py           (验证绩效)
```

---

**系统版本**: SSS096 v3.0
**最后更新**: 2026-01-05
**维护者**: SSS096 Team

🚀 **祝交易顺利！**
