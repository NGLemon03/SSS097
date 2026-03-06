# grid_search_v14.py
import sys, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import logging

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SSSv095a1 as SSS
from analysis import config as cfg
from analysis import data_loader  # 確保已匯入
ticker = cfg.TICKER
PR = cfg.PR
DEFAULT_FACTOR = getattr(cfg, 'FACTOR', 40)
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
TRADE_COOLDOWN_BARS = cfg.TRADE_COOLDOWN_BARS
RESULT_DIR = cfg.RESULT_DIR

STRAT_SIG = {
    strat: (getattr(SSS, info['compute_func']), info['ind_keys'])
    for strat, info in cfg.STRATEGY_PARAMS.items()
}


def split_params(all_params: dict, ind_keys: list):
    ind, bt = {}, {}
    for k, v in all_params.items():
        if k in ind_keys:
            ind[k] = v
        else:
            bt[k] = v
    if 'factor' in ind_keys and 'factor' not in ind:
        ind['factor'] = DEFAULT_FACTOR

    return ind, bt

def backtest(df_ind, strat, bt_params, **extra):
    return SSS.backtest_unified(
        df_ind=df_ind,
        strategy_type=strat,
        params=bt_params,
        discount=COST_PER_SHARE / 0.001425,
        trade_cooldown_bars=TRADE_COOLDOWN_BARS,
        **extra
    )

def _worker(params_all, strat, compute_f, ind_keys, ticker, df_price, df_factor):
    smaa_source = params_all.get('data_source', 'Self')
    source_df = df_factor if smaa_source != 'Self' else df_price
    
    ind_p, bt_p = split_params(params_all, ind_keys)
    ind_key = tuple((k, ind_p[k]) for k in ind_keys if k in ind_p)
    try:
        if strat == 'ssma_turn':
            df_ind, buys, sells = compute_f(df_price, source_df,smaa_source=smaa_source,**ind_p)
            if isinstance(df_ind, pd.Series):
                df_ind = pd.DataFrame({'smaa': df_ind, 'close': df_price['close']}, index=df_ind.index)
            metrics = backtest(df_ind, strat, bt_p, buy_dates=buys, sell_dates=sells)
        else:
            df_ind = compute_f(df_price, source_df, smaa_source=smaa_source, **ind_p)
            if isinstance(df_ind, pd.Series):
                df_ind = pd.DataFrame({'smaa': df_ind, 'close': df_price['close']}, index=df_ind.index)
            metrics = backtest(df_ind, strat, bt_p)
        return {**ind_p, **bt_p, 'data_source': smaa_source, **metrics['metrics']}
    except Exception as e:
        logging.error(f"子進程失敗 ({ticker}, {smaa_source}, {ind_p}): {e}")
        return None

def run_strategy(strat: str, grid: list) -> pd.DataFrame:
    if strat not in STRAT_SIG:
        print(f"⚠ 未支援策略 {strat}")
        return pd.DataFrame()
    compute_f, ind_keys = STRAT_SIG[strat]
    
    # 預載並儲存 Feather 檔案
    df_price, df_factor = data_loader.load_data(ticker)
    # 清理數據，確保一致性
    df_price = df_price.dropna(subset=['close'])
    df_factor = df_factor.dropna(subset=['close']) if not df_factor.empty else df_price
    # 對齊索引
    common_index = df_price.index.intersection(df_factor.index)
    df_price = df_price.loc[common_index]
    df_factor = df_factor.loc[common_index]
    
    price_file = cfg.CACHE_DIR / f'{ticker}_price.feather'
    factor_file = cfg.CACHE_DIR / f'{ticker}_factor.feather'
    data_loader.save_price_feather(ticker, df_price)
    data_loader.save_factor_feather(f"{ticker}_factor", df_factor)

    # 收集 SMAA 參數組合並預計算
    combos = {
        (p['linlen'], p.get('factor', DEFAULT_FACTOR), p['smaalen'])
        for p in grid
        if 'linlen' in p and 'smaalen' in p
    }
    if combos:
        smaa_source = grid[0].get('data_source', 'Self')
        source_key = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")
        df_price, df_factor = data_loader.load_data(ticker, smaa_source=smaa_source)
        source_df = df_factor if smaa_source != "Self" else df_price
        df_cleaned = source_df.dropna(subset=['close'])
        data_hash = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())
        
        SSS.precompute_smaa(
            ticker=ticker,
            param_combinations=list(combos),
            cache_dir=str(cfg.SMAA_CACHE_DIR),
            start_date="2010-01-01",
            smaa_source=smaa_source
        )

        # 驗證快取檔案
        missing_files = []
        for linlen, factor, smaalen in combos:
            smaa_path = SSS.build_smaa_path(ticker, source_key, linlen, factor, smaalen, data_hash, str(cfg.SMAA_CACHE_DIR))
            if not smaa_path.exists():
                missing_files.append(smaa_path)
        if missing_files:
            logging.error(f"Missing SMAA cache files, regenerating: {missing_files}")
            SSS.precompute_smaa(
                ticker=ticker,
                param_combinations=list(combos),
                cache_dir=str(cfg.SMAA_CACHE_DIR),
                start_date="2010-01-01",
                smaa_source=smaa_source
            )

    # 並行處理回測
    records = Parallel(n_jobs=cfg.N_JOBS, backend="loky", mmap_mode="r")(
        delayed(_worker)(p, strat, compute_f, ind_keys, ticker, df_price, df_factor) for p in grid
    )
    records = [r for r in records if r is not None]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.insert(0, 'strategy', strat)
    df.insert(1, 'ticker', ticker)
    df.insert(2, 'data_source', df.pop('data_source'))
    metric_cols = list(records[0].keys())
    ordered_cols = ['strategy', 'ticker', 'data_source'] \
                 + [c for c in metric_cols if c in df.columns and c != 'data_source'] \
                 + [c for c in df.columns if c not in metric_cols and c not in ('strategy', 'ticker', 'data_source')]
    df = df[ordered_cols]
    out_file = RESULT_DIR / f'grid_{strat}_{ticker.replace("^","")}.csv'
    df.to_csv(out_file, index=False)
    print(f"✅ {strat}: {len(df)} rows → {out_file}")
    return df

if __name__ == '__main__':
    logging.disable(logging.DEBUG)
    all_frames = []
    for strat in PR.keys():
        df_res = run_strategy(strat, PR[strat])
        if not df_res.empty:
            all_frames.append(df_res)
    logging.disable(logging.NOTSET)
    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
        all_file = RESULT_DIR / f'grid_ALL_{ticker.replace("^","")}.csv'
        df_all.to_csv(all_file, index=False)
        print(f'📦 合併所有結果 CSV → {all_file}')