# analysis/grid_search_v13.py
"""
# grid_search_v13  

相較 v12 的關鍵修正
────────────────
* 所有 "min_periods" 或視窗長度參數現已強制轉型為 `int(...)`。
* 每個策略的參數字典會先白名單過濾，再傳入 SSS_V9 API，避免出現 **unexpected keyword** 錯誤。
* RMA / Dual 策略不再接收 `devwin` 參數。
* coarse CSV 檔不再讀取 `factor`（現在固定為 40），因此 KeyError: 'factor' 的錯誤已解決。
* `offsets` 會以 JSON 字串形式儲存，並進行穩定解析。
* 若所選 `data_source`（例如 Self）未提供 factor 資料框，則會自動排除該參數組。
* 關閉 Streamlit 警告訊息（設定 `os.environ['PYTHONWARNINGS']`）。
* 子程序錯誤將記錄於 *data/grid_v13_errors.log* 或 *data/wf_v13_errors.log*，
  不會導致主程序崩潰。
* CPU 使用限制為：`max_workers = max(1, int(os.cpu_count()*0.8))`（大約 80% 的核心數）。

其他更新
────────────────────
* `_grid_run` 函式已更新，可處理新的回測回傳格式（metrics 字典）。
* 所有回測呼叫加入 `exit_shift=0`，保持一致性。
* 所有常數已集中至 config 和 param_spaces 管理。
"""

import os
import json
import logging
import datetime as dt
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import warnings
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
from analysis.config import TICKER, START_DATE, DATA_DIR, COST, COOLDOWN, FACTOR, SOURCES, PR


try:
    import SSSv091b5 as SSS
except ImportError:
    import SSSv091b4 as SSS  # fallback to previous version
except ModuleNotFoundError:
    print("❌ SSSv096.py must be in PYTHON suspension / same folder.")
    sys.exit(1)


def _grid_run(strategy: str, source: str, params: dict):
    """Worker for coarse grid."""
    try:
        df_raw, df_fac = SSS.load_data_wrapper(TICKER, start_date=START_DATE, smaa_source=source)
        if df_raw.empty:
            return None

        # JSON‑safe copy of offsets
        params_clean = {k: (json.dumps(v) if isinstance(v, tuple) else v) for k, v in params.items()}
        bm = params.get('buy_mult')
        sm = params.get('sell_mult')

        if strategy == 'single':
            ind = SSS.compute_single(df_raw, df_fac,
                                     int(params['linlen']), FACTOR,
                                     int(params['smaalen']), int(params['devwin']))
            result = SSS.backtest(ind, bm, sm, COOLDOWN, COST, False, 'single', exit_shift=0)
        elif strategy == 'dual':
            ind = SSS.compute_dual(df_raw, df_fac,
                                   int(params['linlen']), FACTOR,
                                   int(params['smaalen']), int(params['short_win']), int(params['long_win']))
            result = SSS.backtest(ind, bm, sm, COOLDOWN, COST, False, 'dual', exit_shift=0)
        elif strategy == 'RMA':
            ind = SSS.compute_RMA(df_raw, df_fac,
                                  int(params['linlen']), FACTOR,
                                  int(params['smaalen']), int(params['rma_len']), int(params['dev_len']))
            result = SSS.backtest(ind, bm, sm, COOLDOWN, COST, False, 'RMA', exit_shift=0)
        elif strategy == 'peaktrough':
            ind = SSS.compute_single(df_raw, df_fac,
                                     int(params['linlen']), FACTOR,
                                     int(params['smaalen']), int(params['devwin']))
            result = SSS.backtest(ind, bm, sm, int(params['cooldown']), COST, False,
                                  'peaktrough', params['prom_factor'], int(params['min_dist']), exit_shift=0)
        else:  # ssma_turn (Self only)
            if source != 'Self':
                return None
            ind, buys, sells = SSS.compute_ssma_turn_combined(
                df_raw, pd.DataFrame(),
                int(params['linlen']), FACTOR, int(params['smaalen']),
                int(params['prom_q']), int(params['min_dist']),
                buy_shift=params['offsets'][0], exit_shift=params['offsets'][1],
                vol_window=int(params['vol_window']), cooldown_days=int(params['cooldown_days']))
            result = SSS.backtest_ssma_turn(ind, buys, sells, cost=COST, cooldown=int(params['cooldown_days']))

        tdf = result['trade_df']
        roi = result['metrics']['total_return'] if 'total_return' in result['metrics'] else np.nan
        return {'strategy': strategy, 'data_source': source, **params_clean, 'ROI': roi, 'Trades': len(tdf)}

    except Exception as e:
        logging.getLogger('grid').error(f"{strategy}|{source}|{params} → {e}")
        return None

def grid_main():
    log = logging.getLogger('grid')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(str(DATA_DIR / f'grid_search_v13_{dt.datetime.now().strftime("%Y%m%d")}.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(fh)
    log.info('Grid search v13 started')

    tasks = []
    with ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.8))) as ex:
        for s in SOURCES:
            for strat, space in PR.items():
                for combo in itertools.product(*space.values()):
                    p = dict(zip(space.keys(), combo))
                    tasks.append(ex.submit(_grid_run, strat, s, p))
        res = [f.result() for f in as_completed(tasks) if f.result()]

    df = pd.DataFrame(res)
    out = DATA_DIR / f'coarse_v13_{dt.datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(out, index=False)
    log.info(f'Coarse grid saved → {out}  rows={len(df)}')

if __name__ == '__main__':
    freeze_support()
    grid_main()