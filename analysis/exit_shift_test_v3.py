# analysis/exit_shift_test_v3.py
import sys, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import logging
import itertools

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import SSSv095a1 as SSS
from analysis import config as cfg
from analysis import data_loader  # 新增：統一資料載入

ticker = cfg.TICKER
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
COOLDOWN_DAYS = cfg.TRADE_COOLDOWN_BARS
RESULT_DIR = cfg.RESULT_DIR

STRAT_MAP = {
    'single': SSS.compute_single,
    'dual': SSS.compute_dual,
    'RMA': SSS.compute_RMA,
    'ssma_turn': SSS.compute_ssma_turn_combined
}


BASE_PARAMS = {
    'single': {'linlen': 60, 'factor': 40, 'smaalen': 20, 'devwin': 40, 'buy_mult': 0.5, 'sell_mult': 1.5},
    'dual': {'linlen': 60, 'factor': 40, 'smaalen': 20, 'short_win': 40, 'long_win': 100, 'buy_mult': 0.5, 'sell_mult': 1.5},
    'RMA': {'linlen': 60, 'factor': 40, 'smaalen': 20, 'rma_len': 60, 'dev_len': 20, 'buy_mult': 0.5, 'sell_mult': 1.5},
    'ssma_turn': {'linlen': 20, 'factor': 60, 'smaalen': 120, 'prom_factor': 45, 'min_dist': 9, 'buy_shift': 3, 'exit_shift': 7, 'vol_window': 20,
                  'quantile_win': 100, 'signal_cooldown_days': 3, 'buy_mult': 0.5, 'sell_mult': 0.5},

}

GRID = {
    'exit_shift': [0, 1, 2, 3],
    'stop_loss': [0.05, 0.10, 0.15]
}

def build_product(grid: dict):
    keys, vals = list(grid.keys()), list(grid.values())
    return list(itertools.product(*vals))  # 預生成列表

def build_ind_params(strat: str, base_params: dict):
    ind_keys = cfg.STRATEGY_PARAMS.get(strat, {}).get('ind_keys', [])
    return {k: base_params[k] for k in ind_keys if k in base_params}

# exit_shift_test_v3.py
def _worker(strat, base_params, grid_params, compute_func, ind_params, df_price, df_factor):
    es, sl = grid_params['exit_shift'], grid_params['stop_loss']
    smaa_source = base_params.get('data_source', 'Self')
    # 若未傳入 df_price 和 df_factor，則根據 smaa_source 載入
    if df_price is None or df_factor is None:
        df_price, df_factor = data_loader.load_data(ticker, smaa_source=smaa_source)
    
    bt_params = {
        'exit_shift': es,
        'stop_loss': sl,
        'buy_mult': base_params.get('buy_mult', 0.5),
        'sell_mult': base_params.get('sell_mult', 0.5),
        'prom_factor': base_params.get('prom_factor', 30),
        'min_dist': base_params.get('min_dist', 5)
    }
    if strat == 'ssma_turn':
        df_ind, buys, sells = compute_func(df_price, df_factor, smaa_source=smaa_source, **ind_params)
        result = SSS.backtest_unified(
            df_ind=df_ind, strategy_type=strat, params=bt_params,
            discount=COST_PER_SHARE/0.001425, trade_cooldown_bars=COOLDOWN_DAYS,
            buy_dates=buys, sell_dates=sells
        )
    else:
        df_ind = compute_func(df_price, df_factor, smaa_source=smaa_source, **ind_params)
        result = SSS.backtest_unified(
            df_ind=df_ind, strategy_type=strat, params=bt_params,
            discount=COST_PER_SHARE/0.001425, trade_cooldown_bars=COOLDOWN_DAYS
        )
    return {
        'strategy': strat,
        'ticker': ticker,
        'data_source': smaa_source,
        'exit_shift': es,
        'stop_loss': sl,
        'trade_cooldown_bars': COOLDOWN_DAYS,  # 加入 trade_cooldown_bars
        **ind_params,
        **bt_params,
        **result['metrics']
    }

def run_exit_shift_test(top_n=50):
    csv_file = RESULT_DIR / f'grid_ALL_{ticker.replace("^","")}.csv'
    if not csv_file.exists():
        print(f'⚠ {csv_file}未找到，使用預設參數集')
        base_params_list = [BASE_PARAMS]
    else:
        df = pd.read_csv(csv_file)
        base_params_list = []
        for _, row in df.sort_values('total_return', ascending=False).head(top_n).iterrows():
            params = {k: row[k] for k in row.index if pd.notna(row[k]) and k not in ['strategy', 'ticker', 'total_return', 'payoff_ratio', 'sharpe_ratio', 'calmar_ratio', 'win_rate', 'num_trades']}
            if 'prom_q' in params:
                params['prom_factor'] = params.pop('prom_q')
            base_params_list.append(params)
        print(f'📈 讀取{len(base_params_list)}個基礎參數集, 來自 {csv_file}')

    all_dfs = []
    for strat, compute_func in STRAT_MAP.items():
        # 去重技術指標參數
        ind_params_set = {tuple((k, p[k]) for k in build_ind_params(strat, p).keys() if k in p) for p in base_params_list}
        ind_cache = {}
        for ind_p in ind_params_set:
            ind_p_dict = dict(ind_p)
            smaa_source = ind_p_dict.get('data_source', 'Self')
            df_price, df_factor = data_loader.load_data(ticker, smaa_source=smaa_source)
            if strat == 'ssma_turn':
                df_ind, buys, sells = compute_func(df_price, df_factor, **ind_p_dict)
                ind_cache[ind_p] = (df_ind, buys, sells)
            else:
                df_ind = compute_func(df_price, df_factor, **ind_p_dict)
                ind_cache[ind_p] = df_ind
        
        # 預生成 GRID 組合
        grid_combinations = [{k: v for k, v in zip(GRID.keys(), combo)} for combo in build_product(GRID)]
        param_combinations = [(p, g) for p in base_params_list for g in grid_combinations]
        records = Parallel(n_jobs=cfg.N_JOBS, backend="loky", mmap_mode="r")(
            delayed(_worker)(
                strat, base_params, grid_params, compute_func,
                build_ind_params(strat, base_params), None, None
            ) for base_params, grid_params in param_combinations
        )
        
        df = pd.DataFrame(records)
        df.insert(0, 'data_source', df.pop('data_source'))
        out_file = RESULT_DIR / f'exit_shift_grid_{strat}_{ticker.replace("^","")}.csv'
        df.to_csv(out_file, index=False)
        print(f'✅ Exit shift 測試 {strat} → {out_file}')
        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    preferred_cols = ['strategy', 'ticker', 'data_source', 'total_return', 'payoff_ratio', 'sharpe_ratio', 'exit_shift', 'stop_loss']
    cols = preferred_cols + [c for c in all_df.columns if c not in preferred_cols and c != 'data_source']
    all_df = all_df[cols]
    all_file = RESULT_DIR / f'exit_shift_grid_ALL_{ticker.replace("^","")}.csv'
    all_df.to_csv(all_file, index=False)
    print(f'📦 全數策略 exit shift 測試 → {all_file}')
    return all_df

if __name__ == '__main__':
    logging.disable(logging.DEBUG)  # 禁用 DEBUG 日誌
    run_exit_shift_test(top_n=50)
    logging.disable(logging.NOTSET)  # 恢復日誌