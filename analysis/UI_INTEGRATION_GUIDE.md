# 增強分析UI整合指南

**路徑：** `#analysis/UI_INTEGRATION_GUIDE.md`  
**創建時間：** 2025-08-18 04:45  
**最後更新：** 2025-08-18 11:40  
**作者：** AI Assistant

## 📋 概述

本指南說明如何將增強分析功能整合到現有的 `app_dash.py` 中，讓用戶可以通過Web界面使用三大核心功能：
1. 風險閥門回測
2. 交易貢獻拆解  
3. 加碼梯度優化

## 🔗 統一欄位 Schema 規範

### 回測輸出標準格式

為了確保 UI 端不會出現 KeyError 或欄位不齊的問題，所有回測輸出（`backtest-store.results[*]`）必須遵循以下 Schema：

#### 必要欄位（Required Fields）

```json
{
  "trade_date": "2025-01-01",           // 交易日期（ISO 格式字串）
  "trade_df": [...],                    // 交易明細 DataFrame
  "trade_ledger": [...],                // 交易帳冊 DataFrame
  "equity": [...],                      // 權益曲線 DataFrame
  "trades": [...]                       // 交易記錄 DataFrame
}
```

#### 日期欄位統一規範

所有日期欄位必須：
- 使用 ISO 格式字串：`"YYYY-MM-DD"` 或 `"YYYY-MM-DD HH:MM:SS"`
- 或使用 pandas Timestamp 物件
- 避免使用其他格式（如 "DD/MM/YYYY" 或中文格式）

#### 欄位名稱映射表

| 英文欄位名 | 中文欄位名 | 說明 | 是否必要 |
|------------|------------|------|----------|
| `trade_date` | `交易日期` | 交易執行日期 | ✅ |
| `date` | `交易日期` | 交易日期（備用） | ⚠️ |
| `weight_change` | `權重變化` | 持倉權重變化 | ✅ |
| `delta_units` | `權重變化` | 單位變化（備用） | ⚠️ |
| `type` | `交易類型` | 買入/賣出/持有 | ✅ |
| `action` | `交易類型` | 交易動作（備用） | ⚠️ |
| `side` | `交易類型` | 交易方向（備用） | ⚠️ |
| `price` | `價格` | 交易價格 | ✅ |
| `exec_price` | `執行價格` | 實際執行價格 | ⚠️ |
| `px` | `價格` | 價格（備用） | ⚠️ |

#### 防禦性檢查機制

1. **日期欄位檢查**：如果沒有 `日期` 欄位，系統會嘗試從 index 建立
2. **重複欄位合併**：當多個英文欄位映射到同一中文欄位時，自動合併並記錄
3. **欄位缺失處理**：必要欄位缺失時提供合理的預設值

#### 整合層欄位處理

參考 `df_from_pack` 與 `format_trade_like_df_for_display` 的欄位映射邏輯：

```python
# 欄位名稱標準化
column_mapping = {
    'trade_date': '交易日期',
    'date': '交易日期',
    'weight_change': '權重變化',
    'delta_units': '權重變化',
    # ... 其他映射
}

# 重複欄位合併
if d.columns.duplicated().any():
    # 自動合併重複欄位
    # 記錄合併過程到 logger
```

## 🚀 快速整合步驟

### 步驟1：導入增強分析模組

在 `app_dash.py` 的開頭添加以下導入語句：

```python
# 在現有導入語句後添加
try:
    from analysis.enhanced_analysis_ui import create_enhanced_analysis_tab, setup_enhanced_analysis_callbacks
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False
    print("警告：無法導入增強分析模組")
```

### 步驟2：在標籤頁中添加增強分析

找到 `app_dash.py` 中的標籤頁定義部分（通常在 `app.layout` 中），在現有標籤頁後添加：

```python
dcc.Tabs(
    id='main-tabs',
    value='backtest',
    children=[
        dcc.Tab(label="策略回測", value="backtest"),
        dcc.Tab(label="所有策略買賣點比較", value="compare"),
        # 添加新的增強分析標籤頁
        create_enhanced_analysis_tab() if ENHANCED_ANALYSIS_AVAILABLE else None
    ],
    className='main-tabs-bar'
)
```

### 步驟3：設置回調函數

在 `app_dash.py` 的最後（在 `if __name__ == '__main__':` 之前）添加：

```python
# 設置增強分析回調函數
if ENHANCED_ANALYSIS_AVAILABLE:
    enhanced_ui = setup_enhanced_analysis_callbacks(app)
```

### 步驟4：更新標籤頁內容回調

找到現有的標籤頁內容回調函數，在 `elif value == "compare":` 後添加：

```python
elif value == "enhanced_analysis":
    if ENHANCED_ANALYSIS_AVAILABLE:
        return html.Div([
            html.H3("🔍 增強交易分析"),
            html.P("請使用上方的增強分析標籤頁進行分析")
        ])
    else:
        return html.Div([
            html.H3("❌ 增強分析不可用"),
            html.P("請檢查模組安裝狀態")
        ])
```

## 🔧 詳細整合說明

### 1. 文件結構

整合後的文件結構：

```
SSS096/
├── app_dash.py                    # 主UI應用
├── analysis/
│   ├── enhanced_trade_analysis.py    # 核心分析模組
│   ├── enhanced_analysis_ui.py       # UI整合模組
│   └── UI_INTEGRATION_GUIDE.md      # 本指南
└── ...
```

### 2. 依賴關係

確保以下套件已安裝：

```bash
pip install dash dash-bootstrap-components pandas numpy matplotlib seaborn openpyxl
```

### 3. 模組導入順序

建議的導入順序：

```python
# 1. 標準庫
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd

# 2. 現有模組
from SSSv096 import param_presets, load_data, compute_single
from analysis import config as cfg

# 3. 新增的增強分析模組
try:
    from analysis.enhanced_analysis_ui import create_enhanced_analysis_tab, setup_enhanced_analysis_callbacks
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False
    print("警告：無法導入增強分析模組")
```

## 📊 UI組件說明

### 1. 增強分析標籤頁

新增的標籤頁包含：

- **數據載入區域**：支持CSV和Excel文件上傳
- **分析執行按鈕**：一鍵執行所有分析
- **結果顯示區域**：分層顯示各項分析結果
- **報告生成功能**：支持多種格式輸出

### 2. 數據格式要求

**交易數據**（必需）：
```csv
交易日期,權重變化,盈虧%
2020-01-01,0.1,5.2
2020-01-15,-0.05,-2.1
```

**基準數據**（可選）：
```csv
日期,收盤價,最高價,最低價
2020-01-01,10000,10100,9900
2020-01-02,10050,10150,10000
```

### 3. 自動欄位映射

系統會自動識別常見的欄位名稱：

- `Date`/`date` → `交易日期`
- `Weight_Change`/`weight_change` → `權重變化`
- `Pnl_Pct`/`pnl_pct` → `盈虧%`
- `Close`/`close` → `收盤價`

## 🔄 回調函數說明

### 1. 主要回調函數

- **`run_enhanced_analysis`**：執行增強分析
- **`generate_enhanced_report`**：生成分析報告
- **`reset_enhanced_analysis`**：重置分析結果

### 2. 回調觸發條件

- 點擊"執行增強分析"按鈕
- 點擊"生成報告"按鈕
- 點擊"重置"按鈕

### 3. 錯誤處理

系統包含完整的錯誤處理機制：

- 數據格式驗證
- 分析執行異常捕獲
- 用戶友好的錯誤訊息顯示

## 🎯 使用流程

### 1. 基本使用流程

1. **上傳數據**：拖拽或點擊上傳交易數據文件
2. **執行分析**：點擊"執行增強分析"按鈕
3. **查看結果**：在結果區域查看各項分析結果
4. **生成報告**：點擊"生成報告"按鈕下載Excel報告

### 2. 進階使用

- **自定義基準數據**：上傳基準數據進行風險閥門分析
- **參數調整**：在控制台調整分析參數
- **結果導出**：支持多種格式的報告輸出

## ⚠️ 注意事項

### 1. 兼容性

- 確保Python版本 >= 3.7
- 檢查所有依賴套件版本兼容性
- 測試現有功能不受影響

### 2. 性能考慮

- 大數據集分析可能需要較長時間
- 建議添加進度條或載入指示器
- 考慮添加分析結果緩存機制

### 3. 錯誤處理

- 所有用戶輸入都經過驗證
- 分析失敗時提供詳細錯誤信息
- 支持分析過程的中斷和重置

## 🔧 故障排除

### 1. 常見問題

**Q: 模組導入失敗**
A: 檢查文件路徑和Python環境

**Q: 分析執行失敗**
A: 檢查數據格式和必要欄位

**Q: UI顯示異常**
A: 檢查Dash版本兼容性

### 2. 調試建議

- 啟用詳細日誌記錄
- 檢查瀏覽器控制台錯誤
- 驗證數據文件格式

## 📈 擴展功能

### 1. 未來改進方向

- 添加更多圖表類型
- 支持實時數據分析
- 集成機器學習模型

### 2. 自定義開發

- 修改分析參數
- 添加新的分析指標
- 自定義報告格式

## 📞 技術支援

如有問題，請檢查：

1. **模組安裝**：確保所有依賴套件正確安裝
2. **文件路徑**：檢查模組文件路徑是否正確
3. **Python環境**：確認Python版本和環境變量
4. **錯誤日誌**：查看控制台和瀏覽器錯誤信息

---

**版本：** v1.0  
**最後更新：** 2025-08-18 04:45  
**適用於：** SSS096 專案 Dash UI 整合
