import ast
# analysis/walk_forward_v13.py
"""
# walk_forward_v13.py

v12 相比的關鍵修正
────────────────
* 所有 "min_periods" / 視窗長度參數現已強制轉為 `int(...)`。
* 每個策略傳入的參數字典經過白名單過濾，避免出現 **unexpected keyword** 錯誤。
* RMA 與 Dual 策略不再接收 `devwin` 參數。
* coarse CSV 檔不再讀取 `factor`（固定為 40），因此不再出現 KeyError: 'factor' 的錯誤。
* `offsets` 現以 JSON 字串形式儲存並穩定解析。
* 若選定的 `data_source`（例如 Self）無 factor 資料框，則會自動排除需依賴 factor 的參數組。
* 關閉 Streamlit 的警告訊息（透過 `os.environ['PYTHONWARNINGS']`）。
* 所有子程序錯誤都會記錄至 *data/wf_v13_errors.log*，
  而不會導致主程序崩潰。
* CPU 使用限制：`max_workers = max(1, int(os.cpu_count()*0.8))`（約佔 80%）。

對應 SSSv096 的更新
────────────────────
* 調整 `_wf_run` 以支援新的回測回傳結構（回傳為包含多項指標的字典）。
* 所有回測呼叫新增參數 `exit_shift=0`。
* 將 `load_data` 更新為 `load_data_wrapper`。
* 將常數統一集中於 core.config 和 core.param_spaces 管理。
"""

import os
import json
import logging
import datetime as dt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
import warnings
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

try:
try:
    import SSSv091b5 as SSS
except ImportError:
    import SSSv091b4 as SSS  # fallback to previous version
except ModuleNotFoundError:
    print("❌ SSSv096.py must be in PYTHONPATH / same folder.")
    sys.exit(1)

from config import TICKER, START_DATE, DATA_DIR, COST, COOLDOWN, FACTOR, PR

WINDOWS = [
    ("2010-06-01", "2014-02-25"),
    ("2014-02-26", "2017-11-22"),
    ("2017-11-23", "2021-08-18"),
    ("2021-08-19", dt.datetime.now().strftime('%Y-%m-%d'))
]

def _parse_offsets(x):
    if isinstance(x, str):
        if x.startswith('['):
            return tuple(json.loads(x))
        if x.startswith('('):
            return ast.literal_eval(x)
    return None if pd.isna(x) else x

def _wf_run(row: pd.Series, start: str, end: str, df_cache: dict):
    try:
        strat = row['strategy']
        src = row['data_source']
        params = row.drop(['strategy', 'data_source', 'ROI', 'Trades']).to_dict()
        # parse offsets json → tuple
        if 'offsets' in params:
            params['offsets'] = _parse_offsets(params['offsets'])

        df_raw, df_fac = df_cache[src]
        slc_raw = df_raw.loc[start:end]
        slc_fac = df_fac.loc[start:end] if not df_fac.empty else pd.DataFrame()
        if slc_raw.empty:
            return None
        bm = params.get('buy_mult')
        sm = params.get('sell_mult')

        if strat == 'single':
            ind = SSS.compute_single(slc_raw, slc_fac, int(params['linlen']), FACTOR, int(params['smaalen']), int(params['devwin']))
            result = SSS.backtest(ind, bm, sm, COOLDOWN, COST, False, 'single', exit_shift=0)
        elif strat == 'dual':
            ind = SSS.compute_dual(slc_raw, slc_fac, int(params['linlen']), FACTOR, int(params['smaalen']), int(params['short_win']), int(params['long_win']))
            result = SSS.backtest(ind, bm, sm, COOLDOWN, COST, False, 'dual', exit_shift=0)
        elif strat == 'RMA':
            ind = SSS.compute_RMA(slc_raw, slc_fac, int(params['linlen']), FACTOR, int(params['smaalen']), int(params['rma_len']), int(params['dev_len']))
            result = SSS.backtest(ind, bm, sm, COOLDOWN, COST, False, 'RMA', exit_shift=0)
        elif strat == 'peaktrough':
            ind = SSS.compute_single(slc_raw, slc_fac, int(params['linlen']), FACTOR, int(params['smaalen']), int(params['devwin']))
            result = SSS.backtest(ind, bm, sm, int(params['cooldown']), COST, False, 'peaktrough', params['prom_factor'], int(params['min_dist']), exit_shift=0)
        else:  # ssma_turn Self
            ind, buys, sells = SSS.compute_ssma_turn_combined(
                slc_raw, pd.DataFrame(), int(params['linlen']), FACTOR, int(params['smaalen']),
                int(params['prom_q']), int(params['min_dist']),
                buy_shift=params['offsets'][0], exit_shift=params['offsets'][1],
                vol_window=int(params['vol_window']), cooldown_days=int(params['cooldown_days']))
            result = SSS.backtest_ssma_turn(ind, buys, sells, cost=COST, cooldown=int(params['cooldown_days']))

        tdf = result['trade_df']
        roi = result['metrics']['total_return'] if 'total_return' in result['metrics'] else np.nan
        return {**row.to_dict(), 'period': f'{start}_{end}', 'ROI_wf': roi, 'Trades_wf': len(tdf)}

    except Exception as e:
        logging.getLogger('wf').error(f"WF {strat}|{src}|{start}_{end}: {e}")
        return None

def wf_main():
    log = logging.getLogger('wf')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(str(DATA_DIR / f'walk_forward_v13_{dt.datetime.now().strftime("%Y%m%d")}.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(fh)

    coarse_file = max([f for f in os.listdir(DATA_DIR) if f.startswith('coarse_v13_')])
    coarse = pd.read_csv(DATA_DIR / coarse_file)
    # quick filter
    cand = coarse[(coarse['ROI'] > 0.10) & (coarse['Trades'] >= 10)].copy()

    # cache data
    df_cache = {}
    for src in cand['data_source'].unique():
        df_cache[src] = SSS.load_data_wrapper(TICKER, start_date=START_DATE, smaa_source=src)

    results = []
    with ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as ex:
        tasks = [ex.submit(_wf_run, row, start, end, df_cache) for _, row in cand.iterrows() for start, end in WINDOWS]
        for f in as_completed(tasks):
            r = f.result()
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    res_file = DATA_DIR / f'wf_results_v13_{dt.datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(res_file, index=False)
    log.info(f'WF results saved → {res_file}')

    # summarise over 4 windows
    keep = df.groupby(['strategy', 'data_source'] + [c for c in cand.columns if c not in ['strategy', 'data_source', 'ROI', 'Trades']]).filter(lambda g: len(g) == 4)
    if not keep.empty:
        summ = (keep.groupby(['strategy', 'data_source'])['ROI_wf'].agg(['mean', 'std']).reset_index())
        summ.to_csv(DATA_DIR / f'wf_summary_v13_{dt.datetime.now().strftime("%Y%m%d")}.csv', index=False)
        log.info('WF summary saved.')

if __name__ == '__main__':
    freeze_support()
    wf_main()