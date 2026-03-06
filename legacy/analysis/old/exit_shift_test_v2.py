# analysis/exit_shift_test_v2.py
import sys, itertools, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import SSSv092 as SSS
from analysis import config as cfg

SYMBOL = cfg.TICKER
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
COOLDOWN_DAYS = cfg.COOLDOWN
RESULT_DIR = ROOT / 'results'
RESULT_DIR.mkdir(exist_ok=True)

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
    'ssma_turn': {'linlen': 20, 'factor': 60, 'smaalen': 120, 'prom_factor': 45, 'min_dist': 9,
                  'offsets': (3, 7), 'vol_window': 20, 'cooldown_days': 3, 'buy_mult': 0.5, 'sell_mult': 0.5}
}

GRID = {
    'exit_shift': [0, 1, 2, 3],
    'stop_loss': [0.05, 0.10, 0.15]
}

def build_product(grid: dict):
    keys, vals = list(grid.keys()), list(grid.values())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def build_ind_params(strat: str, base_params: dict):
    ind_keys = {
        'single': ['linlen', 'factor', 'smaalen', 'devwin'],
        'dual': ['linlen', 'factor', 'smaalen', 'short_win', 'long_win'],
        'RMA': ['linlen', 'factor', 'smaalen', 'rma_len', 'dev_len'],
        'ssma_turn': ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'offsets', 'vol_window', 'cooldown_days']
    }.get(strat, [])
    return {k: base_params[k] for k in ind_keys if k in base_params}

def run_exit_shift_test():
    raw = SSS.load_data_wrapper(SYMBOL)
    df_price, df_factor = raw if isinstance(raw, tuple) else (raw, pd.DataFrame())
    
    all_dfs = []
    for strat, compute_func in STRAT_MAP.items():
        base_params = BASE_PARAMS.get(strat, {})
        ind_params = build_ind_params(strat, base_params)
        records = []

        for params in build_product(GRID):
            es, sl = params['exit_shift'], params['stop_loss']
            bt_params = {
                'exit_shift': es,
                'stop_loss':  sl,
                'buy_mult':   base_params.get('buy_mult', 0.5),
                'sell_mult':  base_params.get('sell_mult', 0.5),
                'prom_factor': base_params.get('prom_factor', 30),
                'min_dist':   base_params.get('min_dist', 5)
            }

            if strat == 'ssma_turn':
                df_ind, buys, sells = compute_func(df_price.copy(), df_factor.copy(), **ind_params)
                result = SSS.backtest_unified(
                    df_ind=df_ind, strategy_type=strat, params=bt_params,
                    discount=COST_PER_SHARE/0.001425, cooldown=COOLDOWN_DAYS,
                    buy_dates=buys, sell_dates=sells
                )
            else:
                df_ind = compute_func(df_price.copy(), df_factor.copy(), **ind_params)
                result = SSS.backtest_unified(
                    df_ind=df_ind, strategy_type=strat, params=bt_params,
                    discount=COST_PER_SHARE/0.001425, cooldown=COOLDOWN_DAYS
                )
            
            records.append({
                'strategy': strat,
                'symbol': SYMBOL,
                'exit_shift': es,
                'stop_loss': sl,
                **ind_params,
                **bt_params,
                **result['metrics']
            })

        df = pd.DataFrame(records)
        out_file = RESULT_DIR / f'exit_shift_grid_{strat}_{SYMBOL.replace("^","")}.csv'
        df.to_csv(out_file, index=False)
        print(f'✅ Exit shift test for {strat} → {out_file}')
        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    preferred_cols = ['strategy', 'symbol', 'total_return', 'payoff_ratio', 'sharpe_ratio', 'exit_shift', 'stop_loss']
    cols = preferred_cols + [c for c in all_df.columns if c not in preferred_cols]
    all_df = all_df[cols]
    all_file = RESULT_DIR / f'exit_shift_grid_ALL_{SYMBOL.replace("^","")}.csv'
    all_df.to_csv(all_file, index=False)
    print(f'📦 All strategies exit shift test → {all_file}')
    return all_df

if __name__ == '__main__':
    run_exit_shift_test()