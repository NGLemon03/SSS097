# analysis/stress_testv1.py
import sys, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import logging

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import SSSv095a1 as SSS
from analysis import config as cfg
from analysis import data_loader

ticker = cfg.TICKER
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
COOLDOWN_DAYS = cfg.TRADE_COOLDOWN_BARS
RESULT_DIR = cfg.RESULT_DIR
STRESS_PERIODS = cfg.STRESS_PERIODS

def _single_period_worker(row, period, ticker, perf_metrics):
    strat = row['strategy']
    smaa_source = row.get('data_source', 'Self')
    
    # 定義每種策略的必要參數
    strat_param_keys = {
        'single': ['linlen', 'factor', 'smaalen', 'devwin', 'buy_mult', 'sell_mult'],
        'dual': ['linlen', 'factor', 'smaalen', 'short_win', 'long_win', 'buy_mult', 'sell_mult'],
        'RMA': ['linlen', 'factor', 'smaalen', 'rma_len', 'dev_len', 'buy_mult', 'sell_mult'],
        'ssma_turn': ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win']
    }
        
    # 提取該策略的參數
    params = {k: row.get(k) for k in strat_param_keys if pd.notna(row.get(k))}
    if 'factor' not in params:
        params['factor'] = cfg.FACTOR
    
    # 檢查並轉換數值參數為整數
    numeric_params = ['linlen', 'smaalen', 'rma_len', 'dev_len', 'devwin', 'short_win', 'long_win', 'vol_window', 'signal_cooldown_days', 'min_dist', 'trade_cooldown_bars']
    for param in numeric_params:
        if param in params and pd.notna(params[param]):
            try:
                params[param] = int(params[param])
            except (ValueError, TypeError):
                logging.error(f"Invalid {param} value {params[param]} for strategy {strat}, period {period}. Skipping row.")
                return None
        elif param in strat_param_keys:
            logging.error(f"Missing required parameter {param} for strategy {strat}, period {period}. Skipping row.")
            return None
    
    # 檢查並轉換浮點數參數
    float_params = ['prom_factor', 'buy_mult', 'sell_mult']
    for param in float_params:
        if param in params and pd.notna(params[param]):
            try:
                params[param] = float(params[param])
            except (ValueError, TypeError):
                logging.error(f"Invalid {param} value {params[param]} for strategy {strat}, period {period}. Skipping row.")
                return None
        elif param in strat_param_keys:
            logging.error(f"Missing required parameter {param} for strategy {strat}, period {period}. Skipping row.")
            return None
    
    try:
        perf = SSS.compute_backtest_for_periods(
            ticker=ticker, periods=[period], strategy_type=strat, params=params,
            smaa_source=smaa_source, trade_cooldown_bars=COOLDOWN_DAYS, discount=COST_PER_SHARE/0.001425
        )
        period_key = f"{perf[0]['period']['start_date']}_{perf[0]['period']['end_date']}"
        perf_metrics_copy = perf_metrics.copy()
        perf_metrics_copy.update({
            'strategy': strat, 'ticker': ticker, 'period': period_key, 'data_source': smaa_source,
            'trade_cooldown_bars': COOLDOWN_DAYS,  # 加入 trade_cooldown_bars
            **params, **perf[0]['metrics']
        })
        return perf_metrics_copy
    except Exception as e:
        logging.error(f"Error in _single_period_worker for strategy {strat}, period {period}: {e}")
        return None

def run_stress_test(top_n=10000, input_csv='grid_ALL'):
    csv_file = RESULT_DIR / f'{input_csv}_{ticker.replace("^","")}.csv'
    if not csv_file.exists():
        print(f'⚠ {csv_file} not found, skipping stress test.')
        return pd.DataFrame()
    
    df = pd.read_csv(csv_file)
    
    # 檢查 CSV 中每種策略的必要參數
    strat_param_keys = cfg.STRATEGY_PARAMS.get(strat, {}).get('ind_keys', []) + cfg.STRATEGY_PARAMS.get(strat, {}).get('bt_keys', [])
    for strat in df['strategy'].unique():
        strat_df = df[df['strategy'] == strat]
        required_params = strat_param_keys.get(strat, [])
        for param in required_params:
            if param not in strat_df.columns:
                logging.error(f"Column {param} missing for strategy {strat} in {csv_file}. Skipping {strat} rows.")
                df = df[df['strategy'] != strat]
            elif strat_df[param].isna().any():
                logging.warning(f"NaN values found in {param} for strategy {strat}. Removing affected rows.")
                df = df[~((df['strategy'] == strat) & df[param].isna())]
    
    # 選擇 top_n 候選參數集
    df_cand = df.sort_values('total_return', ascending=False).head(top_n)
    print(f'📈 Selected {len(df_cand)} candidate parameter sets for stress testing')
    
    if df_cand.empty:
        print(f'⚠ No valid rows after filtering. Skipping stress test.')
        return pd.DataFrame()
    
    # 準備數據
    for source in cfg.SOURCES:
        try:
            data_loader.load_data(ticker, smaa_source=source)
        except Exception as e:
            logging.error(f"Failed to load data for source {source}: {e}")
    
    perf_metrics = {
        'total_return': 0.0,
        'payoff_ratio': float('nan'),
        'sharpe_ratio': float('nan'),
        'sortino_ratio': float('nan'),
        'calmar_ratio': float('nan'),
        'win_rate': 0.0,
        'num_trades': 0
    }
    
    tasks = [(row, period) for _, row in df_cand.iterrows() for period in STRESS_PERIODS]
    all_results = Parallel(n_jobs=cfg.N_JOBS, backend="loky", mmap_mode="r")(
        delayed(_single_period_worker)(row, period, ticker, perf_metrics)
        for row, period in tasks
    )
    
    all_results = [r for r in all_results if r is not None]
    df_stress = pd.DataFrame(all_results)
    if not df_stress.empty:
        metric_cols = [c for c in df_stress.columns if c in perf_metrics.keys()]
        param_cols = [c for c in df_stress.columns if c not in ('strategy', 'ticker', 'data_source', 'period') and c not in metric_cols]
        ordered_cols = ['strategy', 'ticker', 'data_source', 'period'] + metric_cols + param_cols
        df_stress = df_stress[ordered_cols]
        out_file = RESULT_DIR / f'stress_test_results_{ticker.replace("^","")}.csv'
        df_stress.to_csv(out_file, index=False)
        print(f'✅ Stress test results → {out_file}')
    else:
        print(f'⚠ No valid results after stress test.')
    return df_stress

if __name__ == '__main__':
    logging.disable(logging.DEBUG)
    run_stress_test(top_n=30000, input_csv='grid_ALL')
    logging.disable(logging.NOTSET)