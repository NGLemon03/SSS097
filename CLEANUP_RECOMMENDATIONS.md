# SSS096 專案清理建議

**生成日期：** 2025-12-16
**目的：** 識別可以刪除或歸檔的檔案，優化專案結構

---

## 📋 可以刪除的檔案清單

### 1. 已整合的舊 MD 說明檔

以下 MD 檔案的內容已經整合到新的 `README.md` 中，可以刪除或移到 `docs/archive/` 目錄：

#### 主目錄的舊 MD 檔案
```
✅ 可刪除：
├── AGENTS.md                          → 已整合到 README.md (開發指南章節)
├── DEFENSIVE_CHECKS_UPDATE_SUMMARY.md → 已整合到 README.md (防禦性檢查章節)
├── ENSEMBLE_WORKFLOW.md               → 已整合到 README.md (Ensemble 工作流程章節)
└── PORTFOLIO_LEDGER_README.md         → 已整合到 README.md (投資組合流水帳章節)
```

#### analysis/ 目錄的舊 MD 檔案
```
✅ 可刪除：
├── analysis/ENHANCED_ANALYSIS_README.md → 已整合到 README.md (增強分析章節)
├── analysis/UI_INTEGRATION_GUIDE.md    → 已整合到 README.md (Web UI 章節)
└── analysis/logging_guide.md           → 已整合到 README.md (日誌系統章節)
```

**建議操作：**
```bash
# 選項 1: 直接刪除（建議先備份）
rm AGENTS.md DEFENSIVE_CHECKS_UPDATE_SUMMARY.md ENSEMBLE_WORKFLOW.md PORTFOLIO_LEDGER_README.md
rm analysis/ENHANCED_ANALYSIS_README.md analysis/UI_INTEGRATION_GUIDE.md analysis/logging_guide.md

# 選項 2: 移到 archive 目錄（推薦）
mkdir -p docs/archive
mv AGENTS.md DEFENSIVE_CHECKS_UPDATE_SUMMARY.md ENSEMBLE_WORKFLOW.md PORTFOLIO_LEDGER_README.md docs/archive/
mv analysis/ENHANCED_ANALYSIS_README.md analysis/UI_INTEGRATION_GUIDE.md analysis/logging_guide.md docs/archive/
```

### 2. .history 目錄（VSCode Local History）

`.history` 目錄包含大量歷史版本文件（超過 100 個 MD 檔案），已經被 `.gitignore` 排除。

```
⚠️ 建議清理：
└── .history/                          → VSCode 本地歷史，可定期清理舊檔案
```

**建議操作：**
```bash
# 檢視 .history 目錄大小
du -sh .history

# 刪除超過 30 天的歷史記錄
find .history -type f -mtime +30 -delete

# 或完全刪除 .history（如果不需要歷史記錄）
rm -rf .history
```

### 3. pytest cache 相關

Pytest 的快取目錄，可以安全刪除（會自動重建）。

```
✅ 可刪除：
├── .pytest_cache/
└── test/.pytest_cache/
```

**建議操作：**
```bash
# 刪除 pytest cache
find . -type d -name ".pytest_cache" -exec rm -rf {} +
# 或使用 pytest 命令清理
pytest --cache-clear
```

### 4. Python 編譯檔案（__pycache__）

Python 的編譯快取，已被 `.gitignore` 排除，可以定期清理。

```
✅ 可刪除：
├── analysis/__pycache__/
├── analysis/past/__pycache__/
├── runners/__pycache__/
├── sss_core/__pycache__/
└── test/__pycache__/
```

**建議操作：**
```bash
# 刪除所有 __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

# 或使用 Python 命令
python -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
```

### 5. 可能不再使用的 Python 檔案

#### 主目錄
```
⚠️ 需確認是否使用：
├── debug_enhanced_data.py             → 調試用，可能只在開發時使用
├── extract_params.py                  → 參數提取工具，需確認是否還在使用
├── leverage.py                        → 槓桿計算？需確認用途
├── list_folder_structure.py           → 列出目錄結構，工具類，可保留
├── run_enhanced_debug.py              → 調試腳本，開發用
├── run_workflow.py                    → 工作流程執行，需確認
├── run_workflow_example.py            → 範例檔案，可能已過時
├── utils_payload.py                   → 工具函式，需檢查是否被引用
└── version_history.py                 → 版本歷史，可能已過時
```

**建議操作：**
```bash
# 檢查檔案是否被其他檔案引用
grep -r "import debug_enhanced_data" .
grep -r "from debug_enhanced_data" .

# 如果沒有被引用，可以移到 tools/ 目錄或刪除
mkdir -p tools/deprecated
mv debug_enhanced_data.py extract_params.py run_enhanced_debug.py tools/deprecated/
```

#### sss_backtest_outputs/ 目錄的工具腳本
```
⚠️ 需確認：
├── sss_backtest_outputs/fix_trades_columns.py          → 一次性修復腳本？
└── sss_backtest_outputs/fix_trades_columns_enhanced.py → 一次性修復腳本？
```

**建議操作：**
如果已經執行過修復，這些腳本可以刪除或移到工具目錄。

### 6. 臨時或測試用的 CSV/TXT 檔案

```
⚠️ 需確認：
├── list.txt                           → 用途不明
├── re.txt                             → 用途不明
├── workflow_summary_*.txt             → 工作流程摘要，是否需要保留？
```

**建議操作：**
```bash
# 檢視檔案內容確認用途
cat list.txt
cat re.txt

# 如果是臨時檔案，可以刪除
rm list.txt re.txt
```

### 7. analysis/past/ 目錄

包含大量舊版本的分析腳本（約 23 個檔案）。

```
⚠️ 建議歸檔：
└── analysis/past/                     → 舊版本分析腳本
    ├── adaptive_clustering_analysis_v7.py
    ├── balanced_clustering_analysis_v10.py
    ├── clustering_comparison_analysis_v9.py
    └── ... (共 23 個檔案)
```

**建議操作：**
```bash
# 選項 1: 保留作為歷史參考（當前做法，無需操作）
# 選項 2: 壓縮歸檔
tar -czf analysis/past_archive_$(date +%Y%m%d).tar.gz analysis/past/
rm -rf analysis/past/

# 選項 3: 如果確定不再需要，可以刪除
# rm -rf analysis/past/
```

---

## 🔍 需要確認使用狀況的檔案

### 1. 資料檔案
```
⚠️ 需確認：
└── data/
    ├── twii_regime_checker.py         → 檢查 TWII 狀態的腳本，是否仍在使用？
    └── *.csv                          → 各股票資料，是否都需要？
```

### 2. 配置檔案
```
✅ 保留：
├── config.yaml                        → 全局配置，需保留
├── .vscode/tasks.json                 → VSCode 任務配置，需保留
├── .claude/settings.local.json        → Claude 配置，需保留
├── ruff.toml                          → Ruff linter 配置，需保留
└── presets/op.json                    → 參數預設，需保留
```

### 3. 設置腳本
```
✅ 保留：
└── setup.sh                           → 環境設置腳本，需保留
```

### 4. 測試檔案
```
✅ 保留：
└── test/
    ├── test_payload_smoke.py          → 單元測試，需保留
    └── __init___.py                   → 注意：檔名有錯誤（應為 __init__.py）
```

**建議操作：**
```bash
# 修正檔名
mv test/__init___.py test/__init__.py
```

---

## 📊 整理後的建議目錄結構

```
SSS096/
├── README.md                          ← 新整合的主說明檔
│
├── 核心檔案（保留）
├── SSSv096.py
├── app_dash.py
├── SSS_EnsembleTab.py
├── ensemble_wrapper.py
├── convert_results_to_trades.py
├── run_enhanced_ensemble.py
├── config.yaml
│
├── 分析模組（保留）
├── analysis/
│   ├── OSv3.py
│   ├── enhanced_trade_analysis.py
│   ├── logging_config.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── past/                          ← 可考慮歸檔或刪除
│
├── 執行器（保留）
├── runners/
│   └── ensemble_runner.py
│
├── 核心工具（保留）
├── sss_core/
│   ├── schemas.py
│   ├── normalize.py
│   └── plotting.py
│
├── 資料目錄（保留）
├── data/
│   └── *.csv
│
├── 輸出目錄（保留）
├── results/
├── sss_backtest_outputs/
│
├── UI 資源（保留）
├── assets/
│   └── custom.css
│
├── 測試（保留，修正檔名）
├── test/
│   ├── __init__.py                    ← 修正後
│   └── test_payload_smoke.py
│
├── 工具目錄（新建，整理調試/臨時腳本）
└── tools/
    ├── quick_check.ps1
    └── deprecated/                    ← 不再使用的腳本
        ├── debug_enhanced_data.py
        ├── extract_params.py
        └── ...
│
└── 文檔歸檔（新建，存放舊 MD 檔案）
    └── docs/
        └── archive/
            ├── AGENTS.md
            ├── DEFENSIVE_CHECKS_UPDATE_SUMMARY.md
            ├── ENSEMBLE_WORKFLOW.md
            └── ...
```

---

## 🚀 建議的清理步驟

### 步驟 1: 建立歸檔目錄
```bash
mkdir -p docs/archive
mkdir -p tools/deprecated
```

### 步驟 2: 歸檔舊 MD 檔案
```bash
mv AGENTS.md docs/archive/
mv DEFENSIVE_CHECKS_UPDATE_SUMMARY.md docs/archive/
mv ENSEMBLE_WORKFLOW.md docs/archive/
mv PORTFOLIO_LEDGER_README.md docs/archive/
mv analysis/ENHANCED_ANALYSIS_README.md docs/archive/
mv analysis/UI_INTEGRATION_GUIDE.md docs/archive/
mv analysis/logging_guide.md docs/archive/
```

### 步驟 3: 清理快取檔案
```bash
# 清理 Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# 清理 pytest cache
find . -type d -name ".pytest_cache" -exec rm -rf {} +
```

### 步驟 4: 清理歷史檔案（可選）
```bash
# 只刪除超過 30 天的歷史
find .history -type f -mtime +30 -delete
```

### 步驟 5: 整理可能不再使用的檔案
```bash
# 移動可能過時的調試腳本
mv debug_enhanced_data.py tools/deprecated/
mv run_enhanced_debug.py tools/deprecated/
mv run_workflow_example.py tools/deprecated/

# 移動一次性修復腳本
mv sss_backtest_outputs/fix_trades_columns.py tools/deprecated/
mv sss_backtest_outputs/fix_trades_columns_enhanced.py tools/deprecated/
```

### 步驟 6: 修正檔名錯誤
```bash
mv test/__init___.py test/__init__.py
```

### 步驟 7: 刪除臨時檔案（確認後）
```bash
# 確認內容後刪除
rm list.txt re.txt
```

---

## ⚠️ 注意事項

1. **備份優先**：在刪除任何檔案之前，建議先建立專案的完整備份
   ```bash
   # 建立備份
   tar -czf SSS096_backup_$(date +%Y%m%d).tar.gz ../SSS096/
   ```

2. **逐步清理**：建議按步驟逐一清理，每次清理後執行測試確認功能正常
   ```bash
   # 執行快速檢查
   powershell -ExecutionPolicy Bypass -File tools\quick_check.ps1
   ```

3. **Git 追蹤**：如果使用 Git，清理前先提交當前狀態
   ```bash
   git add .
   git commit -m "整理前備份"
   ```

4. **團隊確認**：如果是團隊專案，清理前與團隊成員確認哪些檔案可以刪除

---

## 📈 預期效果

完成清理後，預期可達到：

1. **專案結構更清晰**：移除重複和過時的說明文件
2. **儲存空間節省**：刪除快取和歷史檔案可節省磁碟空間
3. **維護更容易**：統一的 README.md 降低維護成本
4. **減少混淆**：避免多個版本的說明檔造成的混淆

---

**生成日期：** 2025-12-16
**維護建議：** 定期（如每季）執行一次專案清理
