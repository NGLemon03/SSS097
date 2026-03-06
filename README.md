# SSS096 股票策略回測與分析系統

**版本：** v0.96
**最後更新：** 2025-12-16
**作者：** SSS096 開發團隊

---

## 📋 目錄

- [專案概述](#專案概述)
- [核心功能](#核心功能)
- [系統架構](#系統架構)
- [快速開始](#快速開始)
- [詳細功能說明](#詳細功能說明)
- [開發指南](#開發指南)
- [故障排除](#故障排除)
- [附錄](#附錄)

---

## 專案概述

SSS096 是一個專業的股票策略回測與分析系統，提供完整的策略優化、回測、組合管理和風險控制功能。

### 主要特性

- ✅ **多策略支援**：Single、RMA、SSMA Turn 等多種策略類型
- ✅ **Ensemble 組合**：支援 Majority 和 Proportional 兩種組合方法
- ✅ **參數優化**：整合 Optuna 進行超參數優化
- ✅ **增強分析**：風險閥門、交易貢獻拆解、加碼梯度優化
- ✅ **Web UI**：基於 Dash 的互動式操作界面
- ✅ **投資組合流水帳**：完整的交易明細和資產狀態追蹤
- ✅ **日誌系統**：統一的日誌管理和記錄機制

### 技術棧

- **語言**：Python 3.8+
- **數據處理**：pandas, numpy
- **優化引擎**：Optuna
- **視覺化**：matplotlib, seaborn, plotly
- **Web 框架**：Dash, Dash Bootstrap Components
- **數據來源**：Yahoo Finance (yfinance)

---

## 核心功能

### 1. 策略回測引擎 (SSSv096.py)

主要的策略回測引擎，支援多種技術指標策略。

**核心策略類型：**
- **Single**：單一指標策略
- **RMA**：相對移動平均策略
- **SSMA Turn**：雙重平滑移動平均轉折策略

**使用範例：**
```bash
# 執行單一策略回測
python SSSv096.py --strategy RMA_Factor --param_preset op.json

# 自訂參數回測
python SSSv096.py --strategy single --ticker 2330.TW --initial_capital 1000000
```

### 2. Ensemble 策略組合 (SSS_EnsembleTab.py)

提供策略組合功能，支援多策略聚合和風險控制。

**組合方法：**
- **Majority（多數決）**：根據投票機制決定持倉
- **Proportional（比例配置）**：按多頭策略比例配置權重

**核心參數：**
- `floor`：最低權重下限（預設：0.2）
- `ema_span`：權重平滑 EMA span（預設：3）
- `delta_cap`：單日權重變動上限（預設：0.10）
- `min_cooldown_days`：交易冷卻日數（預設：5）
- `min_trade_dw`：最小資金動用比例（預設：0.02）

**使用範例：**
```bash
# Majority 方法
python SSS_EnsembleTab.py --ticker 00631L.TW --method majority --floor 0.2

# Proportional 方法，啟用參數掃描
python run_enhanced_ensemble.py --method proportional --scan_params
```

### 3. Web UI 界面 (app_dash.py)

基於 Dash 的互動式 Web 界面，提供視覺化操作和分析。

**主要功能：**
- 策略回測配置和執行
- 所有策略買賣點比較
- 增強分析功能（風險閥門、交易貢獻、加碼優化）
- 即時圖表和數據表格
- 結果匯出功能

**啟動方式：**
```bash
python app_dash.py
# 訪問 http://localhost:8050
```

### 4. 參數優化 (analysis/OSv3.py, analysis/optuna_16.py)

使用 Optuna 進行超參數優化。

**優化流程：**
```bash
# 執行 Optuna 優化
python analysis/OSv3.py --strategy RMA_Factor --trials 100

# 分析優化結果
python analysis/check_simple_results_v18.py
```

### 5. 增強交易分析 (analysis/enhanced_trade_analysis.py)

提供三大核心分析功能：

#### 5.1 風險閥門回測
識別高風險期間，評估暫停加碼的潛在效果。

**觸發條件：**
- TWII 20日斜率 < 0（短期趨勢向下）
- TWII 60日斜率 < 0（長期趨勢向下）
- ATR比率 > 1.5（波動率異常）

#### 5.2 交易貢獻拆解
分析加碼/減碼階段對總績效的貢獻。

**階段識別：**
- **accumulation（加碼階段）**：連續權重增加
- **distribution（減碼階段）**：連續權重減少

#### 5.3 加碼梯度優化
優化加碼頻率和時機，避免過度交易。

**優化策略：**
- 設定最小間距（兩次加碼之間的最小天數）
- 設定冷卻期（連續加碼後的冷卻階段）

### 6. 投資組合流水帳 (PORTFOLIO_LEDGER_README.md)

追蹤每日資產狀態和交易明細。

**輸出內容：**
- **daily_state**：每日現金/持倉/總資產/權重
- **trade_ledger**：交易明細（買賣金額、費用、稅）

**配置參數：**
```bash
# CLI 參數
python SSS_EnsembleTab.py --ticker 00631L.TW --method majority \
    --initial_capital 2000000 --lot_size 100
```

---

## 系統架構

### 目錄結構

```
SSS096/
├── SSSv096.py                          # 主策略回測引擎
├── app_dash.py                         # Web UI 主應用
├── SSS_EnsembleTab.py                 # Ensemble 策略核心
├── ensemble_wrapper.py                 # Ensemble 包裝器
├── convert_results_to_trades.py        # 結果轉換工具
├── run_enhanced_ensemble.py            # Ensemble 執行腳本
├── config.yaml                         # 全局配置
│
├── analysis/                           # 分析模組
│   ├── OSv3.py                        # 策略分析主程式
│   ├── enhanced_trade_analysis.py     # 增強交易分析
│   ├── logging_config.py              # 日誌配置
│   ├── data_loader.py                 # 數據載入器
│   ├── metrics.py                     # 指標計算
│   └── past/                          # 舊版本檔案
│
├── runners/                            # 執行器模組
│   └── ensemble_runner.py             # Ensemble 執行器
│
├── sss_core/                          # 核心工具
│   ├── schemas.py                     # 數據 Schema
│   ├── normalize.py                   # 數據標準化
│   └── plotting.py                    # 繪圖工具
│
├── data/                              # 數據目錄
│   ├── 2330.TW_data_raw.csv          # 原始數據
│   └── ...
│
├── results/                           # Optuna 優化結果
│   ├── optuna_results_*.csv
│   └── ensemble_*.csv
│
├── sss_backtest_outputs/              # 回測輸出
│   ├── trades_*.csv                   # 交易文件
│   ├── ensemble_equity_*.csv          # 權益曲線
│   └── ensemble_trade_ledger_*.csv    # 交易流水帳
│
├── assets/                            # UI 資源
│   └── custom.css
│
├── presets/                           # 參數預設
│   └── op.json
│
└── test/                              # 測試檔案
    └── test_payload_smoke.py
```

### 核心模組說明

| 模組 | 功能 | 主要檔案 |
|------|------|----------|
| 策略引擎 | 策略回測和計算 | SSSv096.py |
| Ensemble | 多策略組合 | SSS_EnsembleTab.py, ensemble_wrapper.py |
| 優化引擎 | 參數優化 | analysis/OSv3.py, analysis/optuna_16.py |
| 增強分析 | 風險分析和優化 | analysis/enhanced_trade_analysis.py |
| Web UI | 互動式界面 | app_dash.py |
| 數據處理 | 數據載入和清理 | analysis/data_loader.py |
| 日誌系統 | 統一日誌管理 | analysis/logging_config.py |

---

## 快速開始

### 環境設置

#### 1. 安裝依賴
```bash
# 核心依賴
pip install pandas numpy matplotlib seaborn openpyxl yfinance pyyaml

# 優化和分析
pip install optuna scikit-learn scipy statsmodels

# Web UI
pip install dash dash-bootstrap-components plotly kaleido

# 其他工具
pip install joblib
```

#### 2. 自動化設置（Codex 環境）
```bash
chmod +x setup.sh
./setup.sh
```

### 基本使用流程

#### Step 1: 執行單一策略回測
```bash
python SSSv096.py --strategy RMA_Factor --ticker 00631L.TW
```

#### Step 2: 執行參數優化
```bash
python analysis/OSv3.py --strategy single --trials 100
```

#### Step 3: 轉換優化結果為交易文件
```bash
python convert_results_to_trades.py \
  --results_dir results \
  --output_dir sss_backtest_outputs \
  --top_k 5
```

#### Step 4: 執行 Ensemble 策略
```bash
python run_enhanced_ensemble.py \
  --method majority \
  --scan_params
```

#### Step 5: 啟動 Web UI
```bash
python app_dash.py
# 訪問 http://localhost:8050
```

---

## 詳細功能說明

### Ensemble 工作流程

完整的從 Optuna 優化到 Ensemble 執行的流程。

#### 1. 結果轉換

將 Optuna 結果轉換為標準交易文件：

```bash
python convert_results_to_trades.py \
  --results_dir "C:\Stock_reserach\SSS096\results" \
  --output_dir "C:\Stock_reserach\SSS096\sss_backtest_outputs" \
  --top_k 5 \
  --ticker 00631L.TW
```

**參數說明：**
- `--results_dir`：Optuna 結果目錄
- `--output_dir`：輸出交易文件目錄
- `--top_k`：每個策略取前 K 個最佳結果
- `--ticker`：標的代碼
- `--dry_run`：只預覽不實際轉換

#### 2. Ensemble 執行

執行 Ensemble 策略並進行參數掃描：

```bash
# Majority 方法
python run_enhanced_ensemble.py \
  --method majority \
  --scan_params \
  --trades_dir "C:\Stock_reserach\SSS096\sss_backtest_outputs"

# Proportional 方法
python run_enhanced_ensemble.py \
  --method proportional \
  --scan_params \
  --trades_dir "C:\Stock_reserach\SSS096\sss_backtest_outputs"
```

**參數掃描網格（預設）：**
```python
min_cooldown_days: [1, 3, 5]
delta_cap: [0.10, 0.20, 0.30]
min_trade_dw: [0.01, 0.02, 0.03]
majority_k: 根據策略數量動態決定
```

#### 3. 輸出文件

- `ensemble_equity_*.csv`：權益曲線
- `ensemble_weights_*.csv`：每日權重
- `ensemble_debug_*.csv`：調試明細
- `ensemble_daily_state_*.csv`：每日資產狀態
- `ensemble_trade_ledger_*.csv`：交易流水帳

### 交易成本設置

#### 預設費率（與 SSS 一致）
- **基礎手續費**：0.1425%
- **賣出稅率**：0.3%
- **預設折扣**：0.3
- **實際買進費率**：0.04275%
- **實際賣出費率**：0.34275%

#### 成本參數
```python
cost_params = {
    'buy_fee_bp': 4.27,    # 買進費率 (bp)
    'sell_fee_bp': 4.27,   # 賣出費率 (bp)
    'sell_tax_bp': 30.0    # 稅率 (bp)
}
```

**注意**：Ensemble 聚合層預設不再套疊成本，避免重複計算。

### 日誌系統

統一的日誌管理系統，基於 `logging_config.py`。

#### 日誌器分類

| 日誌器類型 | 用途 | 日誌檔案 |
|------------|------|----------|
| SSS 系列 | 策略計算、回測 | SSS_{timestamp}.log |
| Optuna 系列 | 參數優化 | Optuna_{timestamp}.log |
| OS 系列 | 策略分析 | OS_{timestamp}.log |
| Data | 數據處理 | Data_{timestamp}.log |
| Backtest | 回測執行 | Backtest_{timestamp}.log |
| System | 系統訊息 | System_{timestamp}.log |
| Errors | 錯誤記錄 | Errors_{timestamp}.log |

#### 使用方法

```python
from analysis.logging_config import setup_logging, get_logger

# 初始化日誌系統
setup_logging()

# 獲取日誌器
logger = get_logger('SSSv095b2')

# 記錄日誌
logger.info("這是一條信息")
logger.debug("這是一條調試信息")
logger.warning("這是一條警告")
logger.error("這是一條錯誤")
```

### 數據格式規範

#### 回測輸出標準格式

所有回測輸出必須遵循以下 Schema：

```json
{
  "trade_date": "2025-01-01",
  "trade_df": [...],
  "trade_ledger": [...],
  "equity": [...],
  "trades": [...]
}
```

#### 欄位名稱映射表

| 英文欄位名 | 中文欄位名 | 說明 | 是否必要 |
|------------|------------|------|----------|
| `trade_date` | `交易日期` | 交易執行日期 | ✅ |
| `weight_change` | `權重變化` | 持倉權重變化 | ✅ |
| `type` | `交易類型` | 買入/賣出/持有 | ✅ |
| `price` | `價格` | 交易價格 | ✅ |

---

## 開發指南

### AI 代理工作指南

#### 工作重點區域

**核心檔案：**
- `SSSv096.py` - 主要策略回測引擎
- `app_dash.py` - Web UI 主應用
- `ensemble_wrapper.py` - Ensemble 策略包裝器
- `analysis/` - 分析模組目錄

**避免修改的檔案：**
- `tools/quick_check.ps1` - 自動化檢查腳本
- 已標記為 `past/` 的舊版本檔案
- 編譯後的 `.pyc` 檔案

#### 程式碼規範

**註解與輸出：**
- 一律使用繁體中文進行註解和輸出
- 修改記錄需加入日期時間戳記
- 路徑說明格式：`#子資料夾/檔案名`

**日誌記錄：**
- 使用 `analysis/logging_config.py` 中的日誌器
- 重要操作需記錄到日誌檔案
- 錯誤處理需包含詳細的錯誤信息

**資料格式：**
- 日期欄位統一使用 ISO 格式：`YYYY-MM-DD`
- 數值欄位使用 float 類型
- 避免使用中文欄位名稱（除非必要）

#### 測試與驗證

**快速檢查：**
```bash
powershell -ExecutionPolicy Bypass -File tools\quick_check.ps1
```

**回測測試：**
```bash
# 單一策略
python SSSv096.py --strategy RMA_Factor --param_preset op.json

# Ensemble 策略
python run_enhanced_ensemble.py --method majority --top_k 5
```

### 防禦性檢查機制

為避免上游數據格式變更導致程式崩潰，已實施以下機制：

1. **欄位存在性檢查**：在關鍵函式中檢查必要欄位
2. **統一欄位 Schema**：明確規範回測輸出格式
3. **最小單元測試**：確保欄位合併邏輯正常

**相關更新：**
- `analysis/enhanced_trade_analysis.py`：加入防禦性檢查
- `analysis/UI_INTEGRATION_GUIDE.md`：新增 Schema 規範
- `test/test_payload_smoke.py`：新增單元測試

---

## 故障排除

### 常見問題

#### Q1: 模組導入失敗
**症狀**：`ModuleNotFoundError: No module named 'xxx'`
**解決**：
```bash
pip install [缺少的模組]
# 或執行完整安裝
pip install -r requirements.txt
```

#### Q2: 數據格式錯誤
**症狀**：`KeyError: '交易日期'` 或欄位缺失
**解決**：
- 檢查數據文件格式是否符合 Schema 規範
- 確認必要欄位存在
- 使用防禦性檢查機制

#### Q3: Optuna 優化失敗
**症狀**：優化過程中出現錯誤
**解決**：
- 檢查 `analysis/optuna_version_config.py` 版本設定
- 確認 Optuna 版本與程式碼相容
- 查看 `Optuna_{timestamp}.log` 日誌

#### Q4: Web UI 無法啟動
**症狀**：`app_dash.py` 啟動失敗
**解決**：
- 檢查端口 8050 是否被占用
- 確認 Dash 相關套件已安裝
- 查看錯誤訊息並檢查依賴

#### Q5: 策略文件未找到
**症狀**：`No strategy files found in directory`
**解決**：
- 檢查 `sss_backtest_outputs/` 目錄是否存在
- 確認已執行結果轉換步驟
- 檢查文件命名格式

### 除錯工具

#### 啟用詳細日誌
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 檢查數據結構
```python
print(df.info())
print(df.head())
print(df.columns.tolist())
```

#### 驗證欄位對齊
```python
# 檢查重複欄位
if df.columns.duplicated().any():
    print("重複欄位：", df.columns[df.columns.duplicated()])
```

---

## 附錄

### A. 完整參數列表

#### SSSv096.py 參數
```bash
--strategy          # 策略類型：single, RMA_Factor, ssma_turn
--ticker            # 標的代碼，如：2330.TW, 00631L.TW
--param_preset      # 參數預設檔案路徑
--initial_capital   # 初始資金（預設：1000000）
--lot_size          # 整股單位（預設：None，允許零股）
```

#### SSS_EnsembleTab.py 參數
```bash
--ticker            # 標的代碼
--method            # 組合方法：majority, proportional
--floor             # 最低權重下限（預設：0.2）
--ema_span          # 權重平滑 EMA span（預設：3）
--delta_cap         # 單日權重變動上限（預設：0.10）
--min_cooldown_days # 交易冷卻日數（預設：5）
--min_trade_dw      # 最小資金動用比例（預設：0.02）
--initial_capital   # 初始資金（預設：1000000）
--lot_size          # 整股單位（預設：None）
```

#### run_enhanced_ensemble.py 參數
```bash
--method            # 組合方法：majority, proportional
--scan_params       # 啟用參數掃描
--trades_dir        # 交易文件目錄
--discount          # 手續費折扣（預設：0.3）
--no_cost           # 無成本對比測試
```

### B. 策略選擇邏輯

#### 策略多樣性
- 按策略類型分組（single, RMA, ssma_turn）
- 每個類型取前 `top_k` 個最佳結果
- 確保數據源多樣性

#### 策略命名規範
- **輸入**：`optuna_results_single_Factor_TWII_2414_TW_20250624_054804.csv`
- **輸出**：`trades_from_results_single_Factor_TWII___2414.TW_trial1679.csv`

#### 策略數量控制
- 預設每個策略類型取 5 個最佳結果
- 建議總數控制在 20-30 個策略以內

### C. 性能優化建議

1. **策略數量控制**：每個策略類型取 3-5 個最佳結果
2. **參數掃描優化**：使用較小的參數網格進行初步掃描
3. **並行處理**：轉換和 Ensemble 階段支援並行處理
4. **快取機制**：大數據集使用快取避免重複計算

### D. 文檔更新歷史

| 版本 | 日期 | 更新內容 |
|------|------|----------|
| v1.0 | 2025-08-18 | 初始整合版本 |
| v1.1 | 2025-12-16 | 整合所有說明文件，統一格式 |

---

## 📞 技術支援

### 問題回報
- 提供完整的錯誤訊息和堆疊追蹤
- 包含重現步驟和環境信息
- 檢查相關的日誌檔案

### 文檔維護
- 修改功能時同步更新相關文檔
- 使用清晰的範例和說明
- 保持文檔的時效性

---

**專案維護者**：SSS096 開發團隊
**授權協議**：內部使用
**最後更新**：2025-12-16
