
#version_history.py
def get_version_history_html():
    """
    返回包含各版本歷史沿革紀錄，並標註路徑A棄用與路徑B啟用
    """
    html = """
  <style>
  .ver-block {margin-bottom: 2rem; line-height: 1.65;}
  .ver-title {font-size: 1.2rem; font-weight: 700; color: #2563eb;}
  .ver-date  {font-size: 0.85rem; color: #6b7280;}
  ul {margin-top: 0.3rem;}
  </style>

  <h2>🔖 各版本歷史沿革紀錄 (V010 ~ V092)</h2>

  <div class="ver-block">
    <div class="ver-title">V010 - 初版架構 <span class="ver-date">(2025-05-07)</span></div>
    <ul>
      <li>建立 <code>Single Strategy</code> 基本回測框架，支援 Best/Median/Worst 參數組。</li>
      <li>匯入 <code>yfinance</code> 做歷史資料下載，採用簡易成交成本模型。</li>
      <li>輸出 ROI、Sharpe、MDD 至側邊欄；圖表採用 Matplotlib。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V020 - Dual-Scale 與 RMA-Method <span class="ver-date">(2025-05-07)</span></div>
    <ul>
      <li>新增 <code>Dual-Scale</code>（日週混合）與 <code>Wilder RMA</code> 策略。</li>
      <li>引入 <code>st.tabs</code> 呈現多策略結果，並可勾選顯示/隱藏各線。</li>
      <li>重構資料快取邏輯，避免重複下載。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V030 - 自訂 SMAA 資料源 <span class="ver-date">(2025-05-07)</span></div>
    <ul>
      <li>允許使用者選擇 Self/其他因子股 作為 SMAA 計算基礎。</li>
      <li>加入 try/except 下載容錯；錯誤訊息顯示於側欄。</li>
      <li>統一時間區間檢查，避免資料缺口導致策略失效。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V040 - UI/UX 優化與通知機制 <span class="ver-date">(2025-05-07)</span></div>
    <ul>
      <li>側邊欄參數中文化、加入 Tooltip 說明。</li>
      <li>整合 <code>st.toast()</code> / <code>st.exception()</code> 作為即時錯誤提示。</li>
      <li>將舊版 Matplotlib 圖表改為 Plotly，支援縮放與匯出 PNG。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V050 - Peak-Trough 峰谷檢測 <span class="ver-date">(2025-05-08)</span></div>
    <ul>
      <li>導入 <code>scipy.signal.find_peaks</code> 搭配動態標準差門檻，生成買賣訊號。</li>
      <li>參數 <code>min_prominence</code>、<code>min_dist</code> 可視覺化疊圖調整。</li>
      <li>新增 交易日冷卻期（Cooldown）與 滑價 (Slippage) 控制。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V060 - 多策略批次回測 & Plotly 優化 <span class="ver-date">(2025-05-09)</span></div>
    <ul>
      <li>重構 <code>run_backtests()</code>，以 <code>ThreadPoolExecutor</code> 並行運算。</li>
      <li>圖表改用 Plotly Subplots，標示買/賣箭頭與資金曲線。</li>
      <li>匯出 CSV 與 PNG 雙格式，一鍵下載。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V070 - FSD & SSMA_turn 策略整合 <span class="ver-date">(2025-05-11)</span></div>
    <ul>
      <li>導入 Front‑Safe Dynamic（FSD）策略，支援 Walk‑Forward 測試。</li>
      <li>整合 SSMA_turn 版本，可同時比較三種策略績效。</li>
      <li>使用 st.tabs 展示「績效總覽/交易紀錄/參數研究」三分頁。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V080 - 性能調校與動態閾值優化 <span class="ver-date">(2025-05-13)</span></div>
    <ul>
      <li>SSMA_turn 引入 自動參數粗掃→局部細掃 流程，加速尋優。</li>
      <li>冷卻期 (Bars) 與 交易成本 改為輸入框，支援小數四捨五入。</li>
      <li>後端 Backtrader 改為 runstrat() 單趟執行，減少記憶體佔用 35%。</li>
      <li>修正 Plotly 負值區塊浮動 bug，新增「同步 Y 軸」開關。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V090 - exit_shift 支援與信號對接 <span class="ver-date">(2025-05-15)</span></div>
    <ul>
      <li>重構 <code>backtest()</code>，新增 <code>exit_shift</code>、<code>external_buys/sells</code> 參數。</li>
      <li>統一 Single/Dual/RMA/PeakTrough 策略的平移邏輯，基於「隔日開盤+N根棒」延遲執行。</li>
      <li>優化 <code>optimize_parameters()</code> 以回傳 <code>signals_df</code>，方便外部 Grid/WF 分析。</li>
    </ul>
  </div>

  <!-- 路徑A已棄用 -->
  <div class="ver-block">
    <div class="ver-title">路徑A 棄用：V091a ~ V095a <span class="ver-date">(2025-05-16 ~ 2025-05-20)</span></div>
    <ul>
      <li>原路徑A版本（091a、091a2、092a...095a）因分拆模組失敗且維護困難，已集成並整合到後續路徑B中，正式棄用。</li>
      <li>整合說明：包含峰谷檢測前視偏差修復、冷卻期限制、接口重構與UI優化等。</li>
    </ul>
  </div>

  <!-- 啟用路徑B -->
  <div class="ver-block">
    <div class="ver-title">V091b (b1 ~ b7) - PeakTrough 前視偏差修復及模組化測試 <span class="ver-date">(2025-05-16 ~ 2025-05-22)</span></div>
    <ul>
      <li>整合 V091b1 至 V091b7 所有改動，包括：滾動窗口局部峰谷檢測、函式重構、緩存鍵改進與錯誤提示優化。</li>
      <li>統一參數驗證與日誌邏輯，完善 <code>validate_params</code>、<code>compute_cache_key</code> 等工具函式。</li>
      <li>修復峰谷算法前視偏差，並更新 <code>compute_ssma_turn_combined</code> 以支持冷卻期與成交量過濾。</li>
    </ul>
  </div>

  <div class="ver-block">
    <div class="ver-title">V092 - 指標計算優化與 RMA 方法改進 <span class="ver-date">(2025-05-21)</span></div>
    <ul>
      <li>版本號提升為 092，新增 <code>backtest_unified</code> 函式以集中管理回測流程。</li>
      <li>優化 <code>fetch_yf_data</code> 添加當日更新檢查與多層容錯下載邏輯。</li>
      <li>重構 <code>load_data</code>，支持多種 SMAA 因子來源與緩存機制，加強資料完整性驗證。</li>
      <li>統一指標計算接口，保留 <code>compute_single</code>、<code>compute_dual</code>、<code>compute_RMA</code> 及 <code>compute_ssma_turn_combined</code> 四大策略函式。</li>
    </ul>
  </div>
  <div class="ver-block"> <div class="ver-title">V092b2 – 快取整合與 Preset 機制優化 <span class="ver-date">(2025-05-15)</span></div> <ul> <li>統一 Preset 讀取流程，修正 op.json 中缺少 strategy_type 時的跳過邏輯。</li> <li>保留 Joblib 快取，移除 Strea​mlit UI 無關之 st.cache_data。</li> <li>參數驗證、日誌檔案路徑等小幅重構，強化可維護性。</li> </ul> </div> <div class="ver-block"> <div class="ver-title">V092b3 – 版本號修正與程式結構調整 <span class="ver-date">(2025-05-18)</span></div> <ul> <li>更正內部 VERSION 標記為 “092b3”，同步更新檔名與匯出 __all__。</li> <li>精簡 import 順序，移除未使用的 uuid4。</li> <li>統一全域變數宣告位置，便於日後擴充。</li> </ul> </div> <div class="ver-block"> <div class="ver-title">V092b4 – 界面分離與錯誤提示優化 <span class="ver-date">(2025-05-20)</span></div> <ul> <li>新增 Streamlit 側欄錯誤提示，細化 fetch_yf_data 的容錯顯示。</li> <li>將 UI 端與批次回測完全分離，批次腳本僅留 Joblib 快取。</li> <li>修正 DATA_DIR、CACHE_DIR 的外部參數注入邏輯。</li> </ul> </div> <div class="ver-block"> <div class="ver-title">V093 – 指標計算重構與回測主流程優化 <span class="ver-date">(2025-05-22)</span></div> <ul> <li>重寫 compute_smaa、compute_RMA，改用 Pandas rolling.mean/std，保留對齊與 NaN 規則。</li> <li>整合 compute_single/dual/RMA 至統一回測函式 backtest_unified。</li> <li>增強參數驗證，統一日誌級別與錯誤處理顯示。</li> </ul> </div> <div class="ver-block"> <div class="ver-title">V093a1 – 快取編寫問題（已棄用） <span class="ver-date">(2025-05-24)</span></div> <ul> <li><strong>資料來源選取</strong><br> 093a1 直接用 price_df，完全忽略 smaa_source_df<br> → GUI 切換外部因子無效，仍以自收盤價計算，訊號大跑偏<br> ↔ 093／094：先判 smaa_source_df 是否為空，再決定價格來源。</li> <li><strong>指標公式重寫</strong><br> ① compute_smaa 改用 get_rolling_mean，失去 min_periods 容錯；<br> ② compute_RMA 用 get_rolling_std(smaa.values)，未扣均線偏差<br> → 前 window-1 無值卻混用 0/NaN；通道過窄導致頻繁出場或爆倉<br> ↔ 093／094：沿用 Pandas rolling(mean/std) 完整索引與 NaN 規則。</li> <li><strong>快取機制衝突</strong><br> 同時掛 @st.cache_data、joblib.Memory、自訂 save_to_cache，三套共存；<br> 多 worker 並行下易讀到舊 pickle、讀寫互相干擾、Hot-reload 未清快取<br> ↔ 094：只保留 cfg.MEMORY.cache，移除 st.cache_data，單一路徑與鎖定。</li> <li><strong>Numba 與陣列長度對齊</strong><br> rolling_linreg + get_rolling_mean/std 回傳裸陣列長度 n，NaN 區段未補齊，強套回 Series.index<br> → 價量 DataFrame 長度對但前幾列為舊值／隨機值，後段計算皆被污染<br> ↔ 093／094：完全使用 Pandas rolling，天生保留對齊，min_periods 內一律 NaN。</li> </ul> <p><strong>教訓與建議：</strong></p> <ul> <li>API 行為不可隨意改；若不支援參數，應移除簽名或 raise 明確錯誤。</li> <li>重寫前先列比對：<br> <code>diff = pd.concat([smaa_093, smaa_093a1], axis=1).diff(axis=1).abs().sum()</code><br> 若第一天就不一致，必是公式錯誤。</li> <li>快取機制僅用一種：UI 端才用 st.cache_data，批量回測用 Joblib。<br> 設定 <code>Memory(CACHE_DIR, verbose=1)</code>，觀察 hit/miss。</li> <li>Numba 或 C-ext 加速前，先驗證輸入＝輸出：<br> <code>assert np.allclose(ref_result, numba_result, equal_nan=True)</code></li> <li>若改用裸 NumPy，務必填回同長度 Series，處理 window-1 的 NaN，並確認 <code>df.align()</code> 後長度一致。</li> </ul> </div> <div class="ver-block"> <div class="ver-title">V094 – 修復快取與指標問題 <span class="ver-date">(2025-05-26)</span></div> <ul> <li>保留唯一 cfg.MEMORY.cache，移除所有 st.cache_data、joblib 與自訂重複快取。</li> <li>恢復正確的 smaa_source_df 判別邏輯，GUI 切換即生效。</li> <li>指標計算全面回歸 Pandas rolling.mean/std，並新增單元測試比對樣本序列。</li> <li>完善日誌與錯誤提示，確保並行運算中不再產生 collision。</li> </ul> </div>
  """
    return html