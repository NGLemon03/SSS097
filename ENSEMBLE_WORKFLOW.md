# Ensemble 策略工作流程完整説明

## 概述

本工作流程將 Optuna 優化結果轉換為標準交易文件，並執行強化的 Ensemble 策略（Majority/Proportional），支持參數掃描與最佳化。

## 目錄結構

```
SSS096/
├── results/                          # Optuna 產生的優化結果
│   ├── optuna_results_single_*.csv   # Single 策略結果
│   ├── optuna_results_RMA_*.csv      # RMA 策略結果
│   ├── optuna_results_ssma_turn_*.csv # SSMA Turn 策略結果
│   └── ...
├── convert_results_to_trades.py      # 結果轉換工具
├── sss_backtest_outputs/             # 轉換後的標準交易文件
│   ├── trades_from_results_*.csv     # 標準格式交易文件
│   └── ensemble_*.csv                # Ensemble 輸出文件
├── run_enhanced_ensemble.py          # Ensemble 執行腳本
├── ensemble_wrapper.py               # 底層轉換與 Ensemble 邏輯
└── SSSv096.py                       # 統一回測接口
```

## 核心文件説明

### 1. convert_results_to_trades.py
**功能**: 將 `optuna_results_*.csv` 轉換為 `trades_from_results_*.csv`
**輸入**: `results/` 目錄下的 Optuna 結果文件
**輸出**: `sss_backtest_outputs/` 目錄下的標準交易文件

### 2. run_enhanced_ensemble.py
**功能**: 執行 Ensemble 策略，支持參數掃描優化
**輸入**: `sss_backtest_outputs/` 目錄下的交易文件
**輸出**: Ensemble 結果、權益曲線、權重變化等

### 3. ensemble_wrapper.py
**功能**: 封裝轉換與 Ensemble 執行的底層邏輯
**核心函數**: 
- `convert_optuna_results_to_trades()`: 結果轉換
- `select_top_strategies_from_results()`: 策略選擇

## 詳細操作步驟

### 步驟 1: 結果轉換

將 Optuna 結果轉換為標準交易文件：

```bash
# Windows PowerShell
python convert_results_to_trades.py ^
  --results_dir "C:\Stock_reserach\SSS096\results" ^
  --output_dir "C:\Stock_reserach\SSS096\sss_backtest_outputs" ^
  --top_k 5 ^
  --ticker 00631L.TW
```

**參數説明**:
- `--results_dir`: Optuna 結果目錄（默認: `results`）
- `--output_dir`: 輸出交易文件目錄（默認: `sss_backtest_outputs`）
- `--top_k`: 每個策略取前 K 個最佳結果（默認: 5）
- `--ticker`: 標的代碼（默認: `00631L.TW`）
- `--dry_run`: 只預覽不實際轉換

**轉換邏輯**:
1. 讀取 `results/optuna_results_*.csv`
2. 按 score 排序，每個策略取前 `top_k` 名
3. 解析優化參數，調用 SSSv096 進行實際回測
4. 產出標準化的 `trades_from_results_<strategy>_<datasource>_trial<X>.csv`

### 步驟 2: Ensemble 執行

執行 Ensemble 策略，支持參數掃描：

```bash
# Majority 方法，啓用參數掃描
python run_enhanced_ensemble.py ^
  --method majority ^
  --scan_params ^
  --trades_dir "C:\Stock_reserach\SSS096\sss_backtest_outputs"

# Proportional 方法，啓用參數掃描
python run_enhanced_ensemble.py ^
  --method proportional ^
  --scan_params ^
  --trades_dir "C:\Stock_reserach\SSS096\sss_backtest_outputs"
```

**支持的方法**:
- `majority`: 多數決策略
- `proportional`: 按多頭比例配置策略

**基礎參數**（可掃描優化）:
- `floor`: 最低權重下限（默認: 0.2）
- `ema_span`: 權重平滑用 EMA span（默認: 3）
- `delta_cap`: 單日權重變動上限（默認: 0.10）
- `min_cooldown_days`: 交易冷卻日數（默認: 5）
- `min_trade_dw`: 單筆交易最小資金動用比例（默認: 0.02）
- `majority_k`: 多數決門檻（majority 方法專用，默認自動計算）

**參數掃描網格**（默認）:
```python
min_cooldown_days: [1, 3, 5]
delta_cap: [0.10, 0.20, 0.30]
min_trade_dw: [0.01, 0.02, 0.03]
majority_k: 根據策略數量動態決定
```

## 輸出文件説明

### 1. 轉換階段輸出
- `trades_from_results_<strategy>_<datasource>_trial<id>.csv`: 標準交易文件
- 控制枱輸出: 策略列表和轉換統計

### 2. Ensemble 階段輸出
- `results/ensemble_results_<method>_<timestamp>.txt`: 最佳參數與統計摘要
- `sss_backtest_outputs/ensemble_equity_*.csv`: 權益曲線
- `sss_backtest_outputs/ensemble_weights_*.csv`: 每日權重
- `sss_backtest_outputs/ensemble_debug_*.csv`: 調試明細

## 交易成本設置

### 默認費率（與 SSS 一致）
- **基礎手續費**: 0.1425%
- **賣出税率**: 0.3%
- **默認折扣**: 0.3
- **實際買進費率**: 0.04275%
- **實際賣出費率**: 0.34275%

### 成本參數傳遞
```python
cost_params = {
    'buy_fee_bp': 4.27,    # 買進費率 (bp)
    'sell_fee_bp': 4.27,   # 賣出費率 (bp)
    'sell_tax_bp': 30.0    # 税率 (bp)
}
```

**注意**: Ensemble 聚合層默認不再套疊成本，避免重複計算。

## 策略選擇邏輯

### 1. 策略多樣性
- 按策略類型分組（single, RMA, ssma_turn 等）
- 每個類型取前 `top_k` 個最佳結果
- 確保數據源多樣性（Factor_TWII_2412.TW, Factor_TWII_2414.TW, Self 等）

### 2. 策略命名規範
- **輸入**: `optuna_results_single_Factor_TWII_2414_TW_20250624_054804.csv`
- **輸出**: `trades_from_results_single_Factor_TWII___2414.TW_trial1679.csv`

### 3. 策略數量控制
- 默認每個策略類型取 5 個最佳結果
- 可通過 `--top_k` 參數調整
- 建議總數控制在 20-30 個策略以內

## 常見問題解答

### Q1: 30 檔策略怎麼來的？
**A**: 來自 Optuna 的策略族（如 single/dual/RMA/ssma_turn）依 score 取前 N，並確保多樣性（分羣或數據源組合）。

### Q2: trades_*.csv 與 trades_from_results_*.csv 有差嗎？
**A**: Ensemble 會兩者都接受，但建議用 `trades_from_results_*` 命名規格，保留 trial 與策略類型，方便追溯。

### Q3: 可以只預覽不轉換嗎？
**A**: 使用 `--dry_run` 參數，會顯示會被處理的策略清單，方便確認 `top_k` 與命名。

### Q4: 如何調整參數掃描範圍？
**A**: 修改 `run_enhanced_ensemble.py` 中的 `scan_ensemble_parameters()` 函數內的參數網格定義。

### Q5: 交易成本如何設置？
**A**: 使用 `--discount` 參數調整折扣率，或使用 `--no_cost` 進行無成本對比測試。

## 擴展建議

### 1. 配置文件管理
將常用參數抽取到 `config.yaml`：
```yaml
default_ticker: "00631L.TW"
default_top_k: 5
cost_settings:
  base_fee_rate: 0.001425
  tax_rate: 0.003
  default_discount: 0.30
```

### 2. 結果組織
在 `results/` 下分子文件夾（例如 `results/op16/`）以避免混雜不同版本試驗。

### 3. 輸出文件命名
統一掛上方法、N、日期的前綴：
```
ensemble_equity_Majority_16_of_30_20250812.csv
ensemble_weights_Proportional_N30_20250812.csv
```

## 故障排除

### 1. 導入錯誤
**症狀**: `ModuleNotFoundError: No module named 'ensemble_wrapper'`
**解決**: 確保 `ensemble_wrapper.py` 在當前目錄

### 2. 策略文件未找到
**症狀**: `No strategy files found in directory`
**解決**: 檢查 `sss_backtest_outputs/` 目錄是否包含 `trades_from_results_*.csv` 文件

### 3. 參數掃描失敗
**症狀**: 參數掃描過程中出現錯誤
**解決**: 檢查策略文件格式，確保包含必要的列（date, position, signal 等）

### 4. 內存不足
**症狀**: 處理大量策略時出現內存錯誤
**解決**: 減少 `--top_k` 參數值，或分批處理不同策略類型

## 性能優化建議

### 1. 策略數量控制
- 建議每個策略類型取 3-5 個最佳結果
- 總策略數控制在 20-30 個以內
- 避免過度擬合和計算複雜度

### 2. 參數掃描優化
- 使用較小的參數網格進行初步掃描
- 在最佳區域進行精細掃描
- 考慮使用 Optuna 進行 Ensemble 參數優化

### 3. 並行處理
- 轉換階段可以並行處理不同策略類型
- Ensemble 執行階段支持多進程參數掃描

## 總結

本工作流程提供了一個完整的從 Optuna 優化結果到 Ensemble 策略執行的解決方案：

1. **自動化轉換**: 將 Optuna 結果自動轉換為標準交易文件
2. **智能選擇**: 基於評分和多樣性選擇最佳策略組合
3. **參數優化**: 支持 Ensemble 參數的網格搜索優化
4. **成本管理**: 與 SSS 保持一致的交易成本設置
5. **結果分析**: 提供完整的績效分析和調試信息

通過這個工作流程，您可以：
- 快速驗證 Optuna 優化結果
- 構建多樣化的策略組合
- 優化 Ensemble 參數設置
- 獲得更穩定和可靠的策略表現
