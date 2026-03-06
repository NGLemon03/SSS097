# run_batch_smart_leverage.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import os
import sys
import json
from pathlib import Path

from analysis.logging_config import get_logger, init_logging

logger = get_logger("SSS.Ensemble")



# 引用現有模組
sys.path.append(os.getcwd())
try:
    from sss_core.logic import load_data
    from SSS_EnsembleTab import run_ensemble, RunConfig, EnsembleParams, CostParams
    from analysis.strategy_manager import manager
except ImportError:
    logger.error("❌ 請在專案根目錄執行")
    sys.exit(1)

# ================= 設定區 =================
TARGET_ETF = "00631L.TW"
SAFE_ETF = "0050.TW"
START_DATE = "2015-01-01" 
TOP_N_FILES = 10  # 要測試最近的幾個備份檔？

# Smart Leverage 參數 (穩健版)
SMART_PARAMS = {
    "floor": 0.5,       # 底倉 50%
    "delta_cap": 0.1,   # 每日變動限制 10%
    "method": "proportional"
}

# 交易成本
FEE_RATE = 0.001425 * 0.3
TAX_RATE = 0.001
# =========================================

def clean_data(df):
    """資料清洗"""
    if df is None or df.empty:
        return df
    idx = df.index
    if not pd.api.types.is_datetime64_any_dtype(idx):
        idx_str = pd.Index(idx).astype(str)
        # 針對 yfinance CSV 的 Ticker/Date 行，先濾掉非日期行，避免 pandas 推斷格式警告
        mask_valid = idx_str.str.match(r"^\d{4}-\d{2}-\d{2}")
        if mask_valid.any():
            df = df[mask_valid]
            idx_str = idx_str[mask_valid].str.slice(0, 10)
            df.index = pd.to_datetime(idx_str, format="%Y-%m-%d", errors="coerce")
        else:
            df.index = pd.to_datetime(idx_str, errors="coerce")
    else:
        df.index = pd.to_datetime(idx, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.columns = [str(c).lower().strip() for c in df.columns]
    if 'close' in df.columns:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
    df = df[df.index.notnull()]
    return df

def _fetch_yf(ticker, start_date):
    df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df

def load_or_update_safe_data(safe_file: Path, ticker: str, start_date: str, min_end_date=None):
    if safe_file.exists():
        df_safe = pd.read_csv(safe_file, index_col=0)
        df_safe = clean_data(df_safe)
    else:
        df_safe = pd.DataFrame()

    last_date = df_safe.index.max() if not df_safe.empty else None
    need_update = False
    if last_date is None:
        need_update = True
        fetch_start = start_date
    else:
        if min_end_date is not None and last_date < min_end_date:
            need_update = True
        if need_update:
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if need_update:
        logger.info("   更新 %s：%s -> 下載最新", ticker, last_date.date() if last_date is not None else 'None')
        df_new = _fetch_yf(ticker, fetch_start)
        if df_new is not None and not df_new.empty:
            df_new = clean_data(df_new)
            if df_safe.empty:
                df_safe = df_new
            else:
                df_safe = pd.concat([df_safe, df_new], axis=0)
            df_safe = df_safe[~df_safe.index.duplicated(keep='last')].sort_index()
            safe_file.parent.mkdir(parents=True, exist_ok=True)
            df_safe.to_csv(safe_file)
        else:
            logger.warning("   WARNING: 無法取得新資料，沿用舊的 0050 檔案")

    return df_safe

def calculate_smart_equity(df_w, df_target, df_safe):
    """計算 Smart 0050 的權益曲線 (含成本)"""
    # 對齊
    common_idx = df_w.index.intersection(df_target.index).intersection(df_safe.index)
    if len(common_idx) == 0: return pd.Series(dtype=float)
    
    w_arr = df_w.loc[common_idx, 'w'].fillna(0).values
    r_target = df_target.loc[common_idx, 'close'].pct_change().fillna(0).values
    r_safe = df_safe.loc[common_idx, 'close'].pct_change().fillna(0).values
    
    cash = 1.0 # 歸一化起點
    equity_curve = [cash]
    w_prev = 0.0
    
    for i in range(len(common_idx)):
        curr_eq = equity_curve[-1]
        delta_w = w_arr[i] - w_prev
        
        # 成本計算
        cost = 0.0
        if abs(delta_w) > 0.0001:
            turnover = abs(delta_w) * curr_eq
            cost = turnover * (FEE_RATE + TAX_RATE)
            
        post_cost_eq = curr_eq - cost
        
        # 損益計算
        profit = (post_cost_eq * w_arr[i] * r_target[i]) + \
                 (post_cost_eq * (1 - w_arr[i]) * r_safe[i])
        
        equity_curve.append(post_cost_eq + profit)
        w_prev = w_arr[i]
        
    return pd.Series(equity_curve[1:], index=common_idx)

def get_stats(equity):
    if equity.empty: return 0, 0
    # 強制歸一化
    norm = equity / equity.iloc[0]
    ret = norm.iloc[-1] - 1
    mdd = (norm / norm.cummax() - 1).min()
    return ret, mdd

def main():
    logger.info("🚀 批次驗證 Smart Leverage (%s 替代現金)...", SAFE_ETF)
    logger.info("📋 掃描最近 %s 個倉庫備份...", TOP_N_FILES)

    # 1. 準備市場數據 (只載入一次)
    logger.info("📥 載入市場數據 (00631L & 0050)...")
    df_target, _ = load_data(TARGET_ETF, start_date=START_DATE)
    df_target = clean_data(df_target)
    
    safe_file = Path(f"data/{SAFE_ETF.replace(':','_')}_data_raw.csv")
    target_end = df_target.index.max() if df_target is not None and not df_target.empty else None
    df_safe = load_or_update_safe_data(safe_file, SAFE_ETF, START_DATE, min_end_date=target_end)

    # 2. 取得倉庫列表
    warehouses = manager.list_warehouses()
    # 排除現役，只看備份 (通常現役跟最新的備份是一樣的，這裡為了列出 Tag 名稱方便)
    backups = [f for f in warehouses if f != "strategy_warehouse.json"]
    # 按時間倒序 (最新的在前面)
    backups.sort(key=lambda x: os.path.getmtime(Path("analysis")/x), reverse=True)
    target_list = backups[:TOP_N_FILES]

    summary = []
    
    # 3. 迴圈測試
    cost_params = CostParams(buy_fee_bp=4.27, sell_fee_bp=4.27, sell_tax_bp=30)
    
    for wh_name in target_list:
        logger.info("⚙️  測試倉庫: %s", wh_name)
        
        # 讀取該倉庫的策略名單
        try:
            with open(Path("analysis")/wh_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            strategies = [s['name'].replace('.csv','') for s in data.get('strategies', [])]
            meta = data.get('metadata', {})
            tag_info = f"{meta.get('mode','?')}_{meta.get('score_mode','?')}_End{meta.get('split_date','?')}"
        except Exception:
            logger.warning("   讀取 JSON 失敗，跳過")
            continue

        # 尋找 CSV 檔案
        file_map = {}
        for s_name in strategies:
            # 搜尋 outputs 和 archive
            found_csv = None
            for d in [Path("sss_backtest_outputs"), Path("archive")]:
                matches = list(d.rglob(f"*{s_name}*.csv"))
                if matches:
                    found_csv = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    break
            if found_csv:
                file_map[s_name] = found_csv
        
        if not file_map:
            logger.warning("   ❌ 找不到任何策略 CSV，跳過")
            continue

        # 執行 Ensemble (記憶體運算)
        ens_params = EnsembleParams(
            floor=SMART_PARAMS['floor'],
            ema_span=3,
            delta_cap=SMART_PARAMS['delta_cap'],
            min_cooldown_days=1
        )
        
        cfg = RunConfig(
            ticker=TARGET_ETF,
            method=SMART_PARAMS['method'],
            strategies=list(file_map.keys()),
            file_map=file_map,
            params=ens_params,
            cost=cost_params
        )
        
        try:
            # 跑回測取得權重
            _, w_series, _, _, _, _, _, _ = run_ensemble(cfg)
            
            # 轉成 DataFrame
            df_w = pd.DataFrame({'w': w_series})
            
            # 計算 Smart 0050 績效
            equity = calculate_smart_equity(df_w, df_target, df_safe)
            ret, mdd = get_stats(equity)
            
            summary.append({
                "Filename": wh_name,
                "Tag Info": tag_info,
                "Return": ret,
                "MDD": mdd
            })
            logger.info("   ✅ 結果: 報酬 %.1f%% | MDD %.1f%%", ret * 100, mdd * 100)
            
        except Exception as e:
            logger.exception("   ❌ 計算錯誤: %s", e)

    # 4. 總結報告
    if not summary:
        logger.warning("沒有成功測試任何倉庫。")
        return

    df_res = pd.DataFrame(summary)
    df_res = df_res.sort_values("Return", ascending=False) # 按報酬排名
    
    # 加入 B&H 基準
    ret_bh, mdd_bh = get_stats((1 + df_target['close'].pct_change().fillna(0)).cumprod())
    
    logger.info("=" * 100)
    logger.info("🏆 Smart Leverage (00631L + 0050) 批次評比")
    logger.info("📉 B&H 基準: 報酬 %.1f%% | MDD %.1f%%", ret_bh * 100, mdd_bh * 100)
    logger.info("-" * 100)
    logger.info("%s | %s | %s | %s", "Tag Info (模式_評分_截止日)".ljust(40), "總報酬 %".ljust(10), "MDD %".ljust(10), "檔案")
    logger.info("-" * 100)
    
    for _, row in df_res.iterrows():
        logger.info("%s | %8.1f%% | %8.1f%% | %s", f"{row['Tag Info']:<40}", row['Return']*100, row['MDD']*100, row['Filename'])
    logger.info("=" * 100)

if __name__ == "__main__":
    init_logging(enable_file=True)
    main()
