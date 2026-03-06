# SSS096 全專案一次到位重整方案

版本: v1.0  
日期: 2026-03-03  
策略: 高相容 + 功能切分 + 強驗收

## 1. 目標與邊界

### 1.1 目標
- 保留現有外部使用方式不變:
  - `python app_dash.py`
  - 既有 CLI 腳本參數
  - 既有輸出目錄與檔名規則
- 將 `app_dash.py` 巨型單檔拆分為可測試模組。
- 統一資料更新/快取邏輯，只保留 `sss_core/logic.py` 一套來源。
- 導入分層日誌策略，降低噪音訊息。
- 建立「可量化驗收」與「可回滾」機制。

### 1.2 非目標
- 不改變策略演算法行為（除非是 bug 修復且有測試證明）。
- 不改變 CSV/JSON schema。
- 不重寫前端視覺樣式（保持現有 UI 結構與 ID）。

## 2. 現況痛點（重整動機）

- `app_dash.py` 規模過大（5k+ 行），UI、資料更新、回測、子程序執行耦合在同檔。
- 更新邏輯分散，容易出現「資料日期停在舊日期」與「0050 未同步」問題。
- `joblib` 快取若未妥善失效，可能導致 UI 顯示非最新交易日。
- Debug 訊息過多，影響讀 log 效率與排障速度。
- 目前測試多為點狀，缺少完整 E2E 驗收閘門。

## 3. 目標架構（相容優先）

## 3.1 目錄切分
```text
app/
  __init__.py
  bootstrap.py              # create_app(), wiring
  settings.py               # 環境變數/常數集中
  callbacks/
    __init__.py
    market_callbacks.py     # 價格更新、日期刷新
    strategy_callbacks.py   # 策略/ensemble 相關 callback
    process_callbacks.py    # 子程序觸發
  services/
    __init__.py
    market_service.py       # 封裝 sss_core.logic 的更新入口
    backtest_service.py     # 回測資料載入/重算編排
    signal_service.py       # signal_history 讀取與防呆
    process_service.py      # subprocess 安全呼叫
  repositories/
    __init__.py
    file_repo.py            # 檔案系統存取抽象
```

### 3.2 既有相容層
- `app_dash.py` 保留為薄入口與相容 shim:
  - 匯入 `app.bootstrap.create_app()`
  - 暴露既有測試依賴函式（例如 `run_prediction_script`）
  - 轉呼叫新 service，不改函式名稱與參數簽名

## 3.3 單一真相來源（SSOT）
- 價格更新、交易日判斷、快取失效，只走 `sss_core/logic.py`。
- UI 不再自行拼接另一套「是否更新」判斷。

## 4. 一次到位執行包（Work Packages）

## WP1: 相容骨架建立
- 新增 `app/` 分層目錄。
- `app_dash.py` 改為入口 + shim。
- 保留所有 Dash component IDs 與 callback outputs。

驗收:
- `python app_dash.py` 可啟動。
- 現有測試（尤其 `tests/test_app_dash_subprocess_safety.py`）不破壞。

## WP2: 市場資料更新統一
- `app.services.market_service` 僅調用 `sss_core.logic.update_prices_if_needed`。
- 加入 `0050.TW` 到統一更新清單（與其他 ticker 同步）。
- 規則: 「最後交易日不是今天就更新」。

驗收:
- 模擬舊日期 CSV 時，啟動後觸發更新。
- 更新成功後 UI 最後日期前進。

## WP3: load_data 快取每日失效
- `sss_core.logic._build_load_data_cache_key()` 保留「當日 key + 檔案末日期」。
- 確保跨日自動失效、同日重用。
- 更新資料後可選擇 `clear_joblib_cache()`。

驗收:
- 同日重複呼叫命中快取。
- 跨日/檔案末日期變更後自動重載。

## WP4: 回測重算編排
- 交易日更新後，自動觸發必要重算（最小範圍，不全量暴力重跑）。
- 把「是否需要重算」判斷集中在 `backtest_service`。

驗收:
- 價格更新時，回測圖表日期同步更新。
- 無更新時，不重算（避免性能回退）。

## WP5: signal_history 隔離
- signal_history 僅作「訊號展示層」輸入，不阻斷主資料流。
- 加入 fallback: signal_history 缺漏時，仍以價格/ensemble 主線正常顯示。

驗收:
- 刻意移除或損壞 signal_history，不影響主圖載入。

## WP6: 日誌分級與降噪
- 新增 `LOG_LEVEL`（預設 `INFO`）。
- 將大量 debug 訊息降級到 `DEBUG`。
- 保留關鍵 INFO:
  - 更新判斷
  - 實際更新結果
  - 回測重算觸發
  - 錯誤摘要

驗收:
- `LOG_LEVEL=INFO` 時，日誌可讀且不刷屏。
- `LOG_LEVEL=DEBUG` 時，可追完整排障細節。

## WP7: 測試補強（強驗收）
- 單元測試:
  - 更新判斷
  - 快取 key 失效規則
  - signal_history fallback
  - subprocess 安全參數
- 整合測試:
  - 啟動 -> 檢查價格更新 -> 圖表日期一致
  - archive 掃描仍可讀新備份格式
- 回歸測試:
  - 關鍵輸出 CSV schema 不變
  - 舊工作流命令可跑

## WP8: 上線與回滾
- 分支: `refactor/sss096-wp-all-in-one`
- 上線前打包備份（已使用 `archive/YYYYMMDD_HHMMSS_Backup`）。
- 若驗收任一關卡失敗，回到上一 tag/commit（非破壞式回滾）。

## 5. 驗收閘門（Definition of Done）

必須全部通過才算完成:
- G1: 相容性
  - `app_dash.py` 啟動與主要頁面可用
  - 既有 CLI 命令與輸出路徑不變
- G2: 正確性
  - 價格資料非今日時可自動更新
  - 更新後 UI/回測日期同步，不再卡在舊日期
- G3: 穩定性
  - pytest 全綠
  - 主要流程連跑 3 次結果一致（允許市場資料本身變化）
- G4: 可觀測性
  - `LOG_LEVEL=INFO` 可讀
  - `LOG_LEVEL=DEBUG` 可排障
- G5: 性能
  - 無更新日的啟動時間不劣化超過 10%

## 6. 風險與對策

- 風險: callback 拆分後循環依賴
  - 對策: callback register 採單向注入（bootstrap -> callbacks）
- 風險: 測試 monkeypatch 路徑改變
  - 對策: `app_dash.py` 保留 shim 與可 patch 入口
- 風險: 快取策略改動導致效能抖動
  - 對策: 只做「每日失效 + 檔案末日期」最小改動

## 7. 建議落地順序（3 天版）

1. Day 1: WP1 + WP2 + WP3  
2. Day 2: WP4 + WP5 + WP6  
3. Day 3: WP7 + WP8 + 驗收報告

---

此方案的核心是:  
先鎖定外部相容，再把資料更新/快取/重算集中化，最後用測試與閘門收斂品質。
