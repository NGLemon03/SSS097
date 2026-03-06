# 日誌系統使用指南

## 概述

本專案使用統一的日誌配置系統，基於 `logging_config.py` 進行管理。日誌系統提供了清晰的模組劃分和不同級別的日誌記錄。

## 日誌器分類

### 1. SSS 系列日誌器
- **用途**: 策略計算、回測、指標計算等核心功能
- **日誌器名稱**: `SSSv095b2`, `SSSv095b1`, `SSSv095a2`, `SSSv095a1`, `SSSv094a4`
- **日誌檔案**: `SSS_{timestamp}.log`, `SSS_errors_{timestamp}.log`
- **適用場景**: 
  - 策略指標計算
  - 回測執行
  - 交易信號生成
  - 數據處理

### 2. Optuna 系列日誌器
- **用途**: 超參數優化相關功能
- **日誌器名稱**: `optuna_13`, `optuna_12`, `optuna_10`, `optuna_9`, `optuna_F`
- **日誌檔案**: `Optuna_{timestamp}.log`, `Optuna_errors_{timestamp}.log`, `Optuna_trials_{timestamp}.log`
- **適用場景**:
  - 試驗執行
  - 參數優化
  - 結果評估
  - 試驗記錄

### 3. OS 系列日誌器
- **用途**: 策略分析和走查功能
- **日誌器名稱**: `OSv3`, `OSv2`, `OSv1`
- **日誌檔案**: `OS_{timestamp}.log`, `OS_errors_{timestamp}.log`
- **適用場景**:
  - 策略分析
  - 走查測試
  - 結果展示

### 4. 數據處理日誌器
- **用途**: 數據載入和處理
- **日誌器名稱**: `data_loader`, `data`
- **日誌檔案**: `Data_{timestamp}.log`
- **適用場景**:
  - 數據載入
  - 數據清理
  - 快取管理

### 5. 回測日誌器
- **用途**: 回測相關功能
- **日誌器名稱**: `backtest`, `metrics`
- **日誌檔案**: `Backtest_{timestamp}.log`
- **適用場景**:
  - 回測執行
  - 指標計算
  - 性能評估

### 6. 系統日誌器
- **用途**: 系統級別訊息
- **日誌器名稱**: `""` (根日誌器)
- **日誌檔案**: `System_{timestamp}.log`
- **適用場景**:
  - 系統初始化
  - 配置載入
  - 一般訊息

### 7. 錯誤日誌器
- **用途**: 錯誤和異常處理
- **日誌器名稱**: `errors`
- **日誌檔案**: `Errors_{timestamp}.log`
- **適用場景**:
  - 異常捕獲
  - 錯誤記錄
  - 調試信息

## 使用方法

### 基本使用

```python
from analysis.logging_config import setup_logging, get_logger

# 初始化日誌系統
setup_logging()

# 獲取日誌器
logger = get_logger('SSSv095b2')  # 或使用預定義常量
logger = get_logger('optuna_13')

# 記錄日誌
logger.info("這是一條信息")
logger.debug("這是一條調試信息")
logger.warning("這是一條警告")
logger.error("這是一條錯誤")
```

### 使用預定義常量

```python
from analysis.logging_config import LOGGER_NAMES, get_logger

# 使用預定義的日誌器名稱
logger = get_logger(LOGGER_NAMES["SSS"])      # 等同於 get_logger("SSSv095b2")
logger = get_logger(LOGGER_NAMES["OPTUNA"])   # 等同於 get_logger("optuna_13")
logger = get_logger(LOGGER_NAMES["OS"])       # 等同於 get_logger("OSv3")
```

### 模組級別設置

```python
from analysis.logging_config import setup_module_logging

# 為特定模組設置日誌器
logger = setup_module_logging("my_module", level="DEBUG")
```

## 日誌級別

- **DEBUG**: 詳細的調試信息
- **INFO**: 一般信息
- **WARNING**: 警告信息
- **ERROR**: 錯誤信息
- **CRITICAL**: 嚴重錯誤

## 日誌格式

### 標準格式
```
2024-01-01 12:00:00,123 INFO [SSSv095b2] 這是一條信息
```

### 詳細格式
```
2024-01-01 12:00:00,123 INFO [SSSv095b2:123] 這是一條詳細信息
```

## 最佳實踐

### 1. 選擇正確的日誌器
- 根據功能選擇對應的日誌器
- 避免混用不同模組的日誌器

### 2. 使用適當的日誌級別
- DEBUG: 用於調試和開發
- INFO: 用於一般信息
- WARNING: 用於警告
- ERROR: 用於錯誤

### 3. 記錄有用的信息
```python
# 好的做法
logger.info(f"開始處理策略 {strategy_name}, 參數: {params}")

# 避免的做法
logger.info("開始處理")
```

### 4. 錯誤處理
```python
try:
    # 執行操作
    result = some_function()
except Exception as e:
    logger.error(f"操作失敗: {e}", exc_info=True)
```

### 5. 性能考慮
- 避免在循環中使用高級別的日誌
- 使用條件日誌記錄
```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"複雜的調試信息: {expensive_operation()}")
```

## 日誌檔案管理

- 日誌檔案按時間戳記命名，避免覆蓋
- 定期清理舊的日誌檔案
- 錯誤日誌單獨存儲，便於問題排查

## 常見問題

### Q: 如何更改日誌級別？
A: 在 `logging_config.py` 中修改對應日誌器的 `level` 設置。

### Q: 如何添加新的日誌器？
A: 在 `LOGGING_DICT` 的 `loggers` 部分添加新的配置。

### Q: 如何禁用某些日誌？
A: 將對應日誌器的 `level` 設置為 `CRITICAL` 或更高的級別。

### Q: 如何查看特定模組的日誌？
A: 查看對應的日誌檔案，或使用 `grep` 命令過濾。 