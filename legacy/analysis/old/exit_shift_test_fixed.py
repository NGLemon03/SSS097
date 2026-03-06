# analysis/exit_shift_test.py
"""
# exit_shift_test.py – 綠圈策略平倉誤差測試
==========================================================
• 載入資料：讀取 data/wf_results_v13_20250515.csv → 篩選 ROI_mean ≥ 1.5 且 ROI_total ≥ 10
• 對所有策略（ssma_turn / single / dual / RMA）進行平倉平移（exit shift）測試
  - ssma_turn：將賣出訊號 Series 做平移再進行 backtest
  - 其他策略：指標本身無明確賣出訊號 → 雖然產生多個 shift 結果，實際 ROI 無變化
• 所有缺值欄位皆使用預設值填補，以避免 NaN 轉 int 時產生錯誤
• 測試結果輸出至：data/green_exit_shift_results.csv
• 加入特定「崩盤時期」測試樣本（歷史極端區間）

• 輸出結果增加欄位：num_trades（交易次數）、max_drawdown（最大回撤）

針對 SSSv096 的更新
────────────────────
• 調整以支援新的回測回傳結構（metrics 為字典形式）
• STRAT_MAP 更新為使用 compute_ssma_turn_combined
• 確保 exit_shift 正確傳入回測函數
• 新增錯誤記錄檔：shift_errors.log，用於追蹤各策略錯誤原因
• 所有常數統一集中於 core.config 管理
"""

import pandas as pd
import pathlib
import json
import numpy as np
import logging
import ast
from config import TICKER, START_DATE, COST, COOLDOWN, FACTOR, DATA_DIR, SHIFTS, CRASH_PERIODS
try:
    import SSSv091b5 as SSS
except ImportError:
    import SSSv091b4 as SSS  # fallback to previous version

WF_FILE = DATA_DIR / "wf_results_v13_20250515.csv"
OUT_FILE = DATA_DIR / "green_exit_shift_results.csv"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(str(DATA_DIR / "shift_errors.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('shift')

# ─── 小工具 ────────────────────────────────────────────────────────────

def _is_nan(val):
    return pd.isna(val) or str(val).lower() in {"nan", "", "none"}

def _as_int(val, default=0):
    return default if _is_nan(val) else int(float(val))

def _safe_tuple(val, default=(0, 0)):
    if _is_nan(val):
        return default
    if isinstance(val, str):
        try:
            return tuple(ast.literal_eval(val)) if val.strip().startswith("[") else default
        except Exception:
            return default
    if isinstance(val, (tuple, list)):
        return tuple(val)
    return default

def _to_series(sig, idx):
    """確保回傳 Boolean Series 與 idx 對齊"""
    if hasattr(sig, "shift"):
        return sig.astype(bool)
    if isinstance(sig, (list, np.ndarray)):
        if len(sig) == len(idx):
            return pd.Series(sig, index=idx, dtype=bool)
        s = pd.Series(False, index=idx)
        for p in sig:
            if isinstance(p, int) and 0 <= p < len(s):
                s.iloc[p] = True
        return s
    return pd.Series(False, index=idx)

# ─── Strategy map ───────────────────────────────────────────────────────
STRAT_MAP = {
    "ssma_turn": SSS.compute_ssma_turn_combined,
    "single": SSS.compute_single,
    "dual": SSS.compute_dual,
    "RMA": SSS.compute_RMA,
}

# ─── 讀資料 & 篩綠圈 ───────────────────────────────────────────────────
log.info("Loading WF results…")
df = pd.read_csv(WF_FILE)
df = df[pd.to_numeric(df["ROI_wf"], errors="coerce").notnull()].copy()
df["ROI_wf"] = df["ROI_wf"].astype(float)
PARAM_COLS = [
    'strategy', 'data_source', 'linlen', 'smaalen', 'devwin', 'buy_mult', 'sell_mult',
    'rma_len', 'dev_len', 'prom_factor', 'min_dist', 'cooldown', 'prom_q', 'offsets',
    'vol_window', 'cooldown_days'
]
for c in PARAM_COLS:
    df[c] = df[c].astype(str)
full3 = df.groupby(PARAM_COLS)["period"].nunique().reset_index(name="cnt")
full3 = full3.loc[full3["cnt"] == 3, PARAM_COLS]
df3 = df.merge(full3, on=PARAM_COLS, how="inner")
agg = df3.groupby(PARAM_COLS).agg(ROI_total=('ROI', 'first'), ROI_mean=('ROI_wf', 'mean')).reset_index()
GREEN = agg.query("ROI_mean >= 1.5 and ROI_total >= 10").reset_index(drop=True)
log.info(f"Green params: {len(GREEN)}")

# ─── 主迴圈 ────────────────────────────────────────────────────────────
results = []
for _, row in GREEN.iterrows():
    strat = row['strategy']
    fn = STRAT_MAP.get(strat)
    if fn is None:
        log.warning(f"Strategy {strat} not implemented – skip")
        continue

    df_raw, df_fac = SSS.load_data_wrapper(TICKER, start_date=START_DATE, smaa_source=row['data_source'])
    if df_raw.empty:
        log.warning(f"No price data for {row['data_source']} – skip")
        continue

    # kwarg construction
    kwargs = {}
    if strat == 'ssma_turn':
        kwargs = dict(
            linlen=_as_int(row['linlen'], 20),
            factor=FACTOR,
            smaalen=_as_int(row['smaalen'], 120),
            prom_q=_as_int(row['prom_q'], 30),
            min_dist=_as_int(row['min_dist'], 5),
            buy_shift=_safe_tuple(row['offsets'], (3, 7))[0],
            exit_shift=_safe_tuple(row['offsets'], (3, 7))[1],
            vol_window=_as_int(row['vol_window'], 20),
            cooldown_days=_as_int(row['cooldown_days'], 10),
        )
    elif strat == 'single':
        kwargs = dict(
            linlen=_as_int(row['linlen'], 40),
            factor=FACTOR,
            smaalen=_as_int(row['smaalen'], 30),
            devwin=_as_int(row['devwin'], 40),
        )
    elif strat == 'dual':
        kwargs = dict(
            linlen=_as_int(row['linlen'], 40),
            factor=FACTOR,
            smaalen=_as_int(row['smaalen'], 30),
            short_win=_as_int(row['devwin'], 20),
            long_win=_as_int(row['prom_factor'], 80),
        )
    elif strat == 'RMA':
        kwargs = dict(
            linlen=_as_int(row['linlen'], 40),
            factor=FACTOR,
            smaalen=_as_int(row['smaalen'], 30),
            rma_len=_as_int(row['rma_len'], 60),
            dev_len=_as_int(row['dev_len'], 20),
        )

    # compute signals / indicator
    try:
        out = fn(df_raw, df_fac, **kwargs)
    except Exception as e:
        log.error(f"compute_{strat} failed: {e}")
        continue

    if strat == 'ssma_turn' and isinstance(out, tuple) and len(out) == 3:
        ind, buys_raw, sells_raw = out
        buys = _to_series(buys_raw, ind.index)
        sells = _to_series(sells_raw, ind.index)
        cd = kwargs['cooldown_days']

        for sh in SHIFTS:
            try:
                shifted = sells.shift(sh).fillna(False)
                result = SSS.backtest_ssma_turn(ind, buys, shifted, cost=COST, cooldown=cd)
                roi = result['metrics']['total_return'] if 'total_return' in result['metrics'] else float('nan')
                num_trades = len(result['trade_df'])
                max_drawdown = result['metrics'].get('max_drawdown', float('nan'))
                rec = row.to_dict()
                rec.update(exit_shift=sh, ROI_shift=roi, num_trades=num_trades, max_drawdown=max_drawdown)
                results.append(rec)
            except Exception as e:
                log.warning(f"ssma_turn backtest shift={sh} err: {e}")
                rec = row.to_dict()
                rec.update(exit_shift=sh, ROI_shift=float('nan'), num_trades=0, max_drawdown=float('nan'))
                results.append(rec)
    else:
        # other strategies – compute once, reuse ROI for all shifts
        if not isinstance(out, pd.DataFrame) or out.empty:
            log.warning(f"Indicator empty for {strat}")
            continue
        try:
            result = SSS.backtest(out,
                                  _as_int(row['buy_mult'], 1),
                                  _as_int(row['sell_mult'], 1),
                                  _as_int(row['cooldown'], COOLDOWN),
                                  COST, False, strat, exit_shift=0)
            base_roi = result['metrics']['total_return'] if 'total_return' in result['metrics'] else float('nan')
            num_trades = len(result['trade_df'])
            max_drawdown = result['metrics'].get('max_drawdown', float('nan'))
        except Exception as e:
            log.warning(f"backtest {strat} err: {e}")
            base_roi = num_trades = max_drawdown = float('nan')
        for sh in SHIFTS:
            rec = row.to_dict()
            rec.update(exit_shift=sh, ROI_shift=base_roi, num_trades=num_trades, max_drawdown=max_drawdown)
            results.append(rec)

    # Crash-only backtest
    try:
        stress = SSS.compute_backtest_for_periods(TICKER, strat, kwargs, CRASH_PERIODS, row['data_source'], COST, COOLDOWN)
        for _, m in stress.iterrows():
            rec2 = row.to_dict()
            rec2.update(
                exit_shift='stress',
                period=m['period'],
                ROI_shift=m['total_return'],
                num_trades=m.get('num_trades', 0),
                max_drawdown=m.get('max_drawdown', float('nan'))
            )
            results.append(rec2)
    except Exception as e:
        log.warning(f"Crash-only backtest for {strat} failed: {e}")

# ─── 輸出 ──────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv(OUT_FILE, index=False)
log.info(f"Saved {len(results)} rows → {OUT_FILE}")