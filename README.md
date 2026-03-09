# SSS097 README（當前版本）

最後更新：2026-03-06

## 1. 專案定位
SSS097 是台股槓桿 ETF 的策略研究與執行專案，主流程涵蓋：
- 參數搜尋（Optuna）
- 交易資料轉換
- Ensemble 與 OOS 驗證
- 槓桿配置批次評估
- Dash 視覺化檢視

## 2. 主線入口（目前有效）
以下為主目錄/分析目錄中目前有效的主線入口：
- `analysis/optuna_16.py`
- `convert_results_to_trades.py`
- `analysis/optimize_ensemble.py`
- `init_warehouse.py`
- `run_oos_analysis.py`
- `run_enhanced_ensemble.py`
- `run_batch_smart_leverage.py`
- `app_dash.py`

## 3. 主線執行順序
建議依序執行：
1. `analysis/optuna_16.py`
2. `convert_results_to_trades.py`
3. `analysis/optimize_ensemble.py`
4. `init_warehouse.py`
5. `run_oos_analysis.py` 或 `run_enhanced_ensemble.py`
6. `run_batch_smart_leverage.py`（選用）
7. `app_dash.py`（介面檢視）

## 4. 快速開始

### 4.1 建立環境
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 範例指令
```bash
python analysis/optuna_16.py --strategy single --n_trials 50 --data_source Self
python convert_results_to_trades.py --top_k 5
python analysis/optimize_ensemble.py --split_date 2025-06-30 --mode OOS --score_mode smart_bh
python init_warehouse.py --top_k 5
python run_oos_analysis.py --split_date 2025-06-30
python run_enhanced_ensemble.py --method majority --top_k 5
python run_batch_smart_leverage.py
python app_dash.py
```

## 5. 相容與封存說明

### 5.1 相容包裝腳本（保留）
- `run_workflow.py`
- `run_workflow_example.py`
- `auto_pipeline.py`（歷史相容用途）

### 5.2 已封存入口（不在主目錄）
以下歷史測試/流程已集中封存到：
- `archive/test_series_one_folder_20260306_182919/`

包含：
- `run_full_pipeline.txt`
- `run_full_pipeline_5.py`
- `run_hybrid_backtest*.py`
- `run_round3*`、`run_round4*`、`run_round5*`、`run_round6*`

## 6. 目錄概要
- `analysis/`：研究與優化模組
- `sss_core/`：核心策略/計算邏輯
- `results/`：輸出結果
- `data/`：輸入資料
- `docs/`：正式文檔
- `archive/`：封存快照與歷史資產
- `legacy/`：舊版歷史代碼

## 7. 文檔入口（單一主手冊 + 輔助索引）
- 主作業手冊（整併三分法）：`docs/PROJECT_OPERATIONS_GUIDE.md`
- 主線流程定義：`docs/MAINLINE.md`
- 研究代碼索引：`docs/research_catalog.md`
- 主目錄整理報告：`docs/ROOT_DIR_TIDY_REPORT_20260306.md`

## 8. 維護規範
- 主目錄只保留活躍入口與核心模組。
- 研究型、測試型、一次性腳本優先歸檔，不與主線混放。
- 新增研究腳本時，同步更新 `docs/research_catalog.md`。
- 主線流程或入口有變動時，同步更新本 README 與 `docs/MAINLINE.md`。
