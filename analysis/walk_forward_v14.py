# walk_forward_v14.py
import sys, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import logging
import numpy as np
import datetime as dt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import SSSv095a1 as SSS
from analysis import config as cfg
from analysis import data_loader

RESULT_DIR = cfg.RESULT_DIR
DEFAULT_FACTOR = getattr(cfg, 'FACTOR', 40)
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
COOLDOWN_DAYS = cfg.TRADE_COOLDOWN_BARS
ticker = cfg.TICKER
PR = cfg.PR
STRESS_PERIODS = cfg.STRESS_PERIODS
STRAT_FUNC_MAP = {
    'single': SSS.compute_single,
    'dual': SSS.compute_dual,
    'RMA': SSS.compute_RMA,
    'ssma_turn': SSS.compute_ssma_turn_combined
}


def split_params(params_all: dict, ind_keys: list):
    ind_params, bt_params = {}, {}
    for k, v in params_all.items():
        if k in ind_keys:
            ind_params[k] = v
        else:
            bt_params[k] = v
    if 'factor' in ind_keys and 'factor' not in ind_params:
        ind_params['factor'] = DEFAULT_FACTOR
    return ind_params, bt_params

def _worker_strategy(params_all, strat, compute_func, ind_keys, ticker):
    smaa_source = params_all.get('data_source', 'Self')
    try:
        df_price, df_factor = data_loader.load_data(ticker, smaa_source=smaa_source)
    except Exception as e:
        logging.error(f"子進程載入數據失敗 ({ticker}, {smaa_source}): {e}")
        return []

    ind_params, bt_params = split_params(params_all, ind_keys)
    results = []
    
    # 按走查期間執行回測
    for period in cfg.WF_PERIODS:
        start, end = period['test']
        # 過濾測試期間的數據
        try:
            df_price_period = df_price.loc[start:end]
            df_factor_period = df_factor.loc[start:end] if not df_factor.empty else pd.DataFrame()
        except Exception as e:
            logging.warning(f"無效的日期區間 {start}_{end},策略 {strat},資料來源 {smaa_source}: {e}")
            continue
        
        if df_price_period.empty:
            logging.warning(f"{start}_{end} 區間價格資料為空,策略 {strat},跳過該區間.")
            continue
        if smaa_source != 'Self' and df_factor_period.empty:
            logging.warning(f"{start}_{end} 區間因子資料為空,來源 {smaa_source},策略 {strat},跳過該區間.")
            continue

        try:
            if strat == 'ssma_turn':
                df_ind, buys, sells = compute_func(df_price_period, df_factor_period, smaa_source=smaa_source, **ind_params)
                metrics = SSS.backtest_unified(
                    df_ind=df_ind, strategy_type=strat, params=bt_params,
                    discount=COST_PER_SHARE/0.001425, trade_cooldown_bars=COOLDOWN_DAYS,
                    buy_dates=buys, sell_dates=sells)
            else:
                df_ind = compute_func(df_price_period, df_factor_period, smaa_source=smaa_source, **ind_params)
                if isinstance(df_ind, pd.Series):
                    logging.warning(f"df_ind 是 Series 格式，策略 {strat}，將自動轉換為 DataFrame")
                    smaa = df_ind
                    base = smaa.ewm(span=ind_params.get('devwin', 20), adjust=False).mean()
                    sd = (smaa - base).abs().ewm(alpha=1/ind_params.get('devwin', 20), adjust=False).mean()
                    df_ind = pd.DataFrame({
                        'open': df_price_period['open'],
                        'high': df_price_period['high'],
                        'low': df_price_period['low'],
                        'close': df_price_period['close'],
                        'smaa': smaa,
                        'base': base,
                        'sd': sd
                    }, index=df_ind.index).dropna()

                metrics = SSS.backtest_unified(
                    df_ind=df_ind, strategy_type=strat, params=bt_params,
                    discount=COST_PER_SHARE/0.001425, trade_cooldown_bars=COOLDOWN_DAYS)

            results.append({
                'period': f"{start}_{end}",
                'ROI_wf': metrics['metrics']['total_return'],
                **ind_params, **bt_params, 'data_source': smaa_source, **metrics['metrics']
            })
        except Exception as e:
            logging.error(f"回測錯誤，期間 {start}_{end}，策略 {strat}，來源 {smaa_source}: {e}")
            continue

    if not results:
        logging.warning(f"策略 {strat}、參數 {ind_params}、來源 {smaa_source} 無有效結果")
    
    # 計算累積回報
    if results:
        total_roi = np.prod([1 + r['ROI_wf'] for r in results]) - 1
        for r in results:
            r['ROI'] = total_roi

    return results

def run_strategy(strat: str):
    if strat not in STRAT_FUNC_MAP:
        print(f"⚠️ 不支援的策略類型：{strat}")
        return pd.DataFrame()
    compute_func = STRAT_FUNC_MAP[strat]
    grid = PR.get(strat, [])
    
    # 預載和儲存數據
    df_price, df_factor = data_loader.load_data(ticker)
    logging.info(f"Price data range: {df_price.index.min()} to {df_price.index.max()}")
    logging.info(f"Factor data range: {df_factor.index.min()} to {df_factor.index.max() if not df_factor.empty else 'N/A'}")
    data_loader.save_price_feather(ticker, df_price)
    data_loader.save_factor_feather(f"{ticker}_factor", df_factor)
    
    # 預計算 SMAA
    combos = {
        (p['linlen'], p.get('factor', DEFAULT_FACTOR), p['smaalen'])
        for p in grid
        if 'linlen' in p and 'smaalen' in p
    }
    if combos:
        smaa_source = grid[0].get('data_source', 'Self')
        source_key = smaa_source.replace(" ", "_").replace("/", "_").replace("^", "")
        df_cleaned = (df_factor if smaa_source != "Self" else df_price).dropna(subset=['close'])
        data_hash = str(pd.util.hash_pandas_object(df_cleaned['close']).sum())
        SSS.precompute_smaa(
            ticker=ticker,
            param_combinations=list(combos),
            cache_dir=str(cfg.SMAA_CACHE_DIR),
            start_date="2010-01-01",
            smaa_source=smaa_source
        )
        # 驗證快取
        missing_files = []
        for linlen, factor, smaalen in combos:
            smaa_path = SSS.build_smaa_path(ticker, source_key, linlen, factor, smaalen, data_hash, str(cfg.SMAA_CACHE_DIR))
            if not smaa_path.exists():
                missing_files.append(smaa_path)
        if missing_files:
            logging.error(f"Walk-forward 缺少 SMAA 快取檔案: {missing_files}")
            SSS.precompute_smaa(
                ticker=ticker,
                param_combinations=list(combos),
                cache_dir=str(cfg.SMAA_CACHE_DIR),
                start_date="2010-01-01",
                smaa_source=smaa_source
            )
    
    ind_keys = cfg.STRATEGY_PARAMS.get(strat, {}).get('ind_keys', [])
    # 並行回測
    records = Parallel(n_jobs=cfg.N_JOBS, backend="loky")(
        delayed(_worker_strategy)(p, strat, compute_func, ind_keys, ticker) for p in grid
    )
    # 展平多期間結果
    records = [item for sublist in records if sublist is not None for item in sublist if item is not None]
    if not records:
        print(f"⚠ 策略 {strat} 無有效結果")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df.insert(0, 'strategy', strat)
    df.insert(1, 'ticker', ticker)
    df.insert(2, 'data_source', df.pop('data_source'))
    # 確保欄位順序
    preferred_cols = ['strategy', 'ticker', 'data_source', 'period', 'ROI_wf', 'ROI', 'total_return', 'payoff_ratio', 'sharpe_ratio']
    cols = preferred_cols + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    out_file = RESULT_DIR / f'wf_grid_{strat}_{ticker.replace("^","")}.csv'
    df.to_csv(out_file, index=False)
    print(f'✅ 策略 {strat} walk-forward grid 完成 → {out_file}')
    return df

def _worker_stress(row: pd.Series, ticker: str, perf_metrics: dict, param_keys: list):
    strat = row['strategy']
    smaa_source = row.get('data_source', 'Self')
    params = {k: row.get(k) for k in param_keys if pd.notna(row.get(k))}
    if 'factor' not in params:
        params['factor'] = DEFAULT_FACTOR
    perfs = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(SSS.compute_backtest_for_periods)(
            ticker=ticker, periods=[period], strategy_type=strat, params=params,
            smaa_source=smaa_source, trade_cooldown_bars=COOLDOWN_DAYS, discount=COST_PER_SHARE/0.001425
        ) for period in STRESS_PERIODS
    )
    results = []
    for perf in perfs:
        period = f"{perf[0]['period']['start_date']}_{perf[0]['period']['end_date']}"
        perf_metrics_copy = perf_metrics.copy()
        perf_metrics_copy.update({
            'strategy': strat, 'ticker': ticker, 'period': period, 'data_source': smaa_source, **params, **perf[0]['metrics']
        })
        results.append(perf_metrics_copy)
    return results

def run_stress_test(df_cand: pd.DataFrame):
    all_results = []
    param_keys = [
        'linlen', 'factor', 'smaalen', 'devwin', 'buy_mult', 'sell_mult',
        'rma_len', 'dev_len', 'prom_factor', 'min_dist', 'signal_cooldown_days',
        'buy_shift', 'exit_shift', 'vol_window', 'stop_loss',
        'short_win', 'long_win'
    ]
    perf_metrics = {
        'total_return': 0.0,
        'payoff_ratio': float('nan'),
        'sharpe_ratio': float('nan'),
        'calmar_ratio': float('nan'),
        'win_rate': 0.0,
        'num_trades': 0
    }
    # 並行處理候選參數與壓力時段
    all_results = Parallel(n_jobs=cfg.N_JOBS, backend="loky")(
        delayed(_worker_stress)(row, ticker, perf_metrics, param_keys) for _, row in df_cand.iterrows()
    )
    all_results = [item for sublist in all_results for item in sublist]
    df_stress = pd.DataFrame(all_results)
    if not df_stress.empty:
        metric_cols = [c for c in df_stress.columns if c in perf_metrics.keys()]
        param_cols = [c for c in df_stress.columns if c in param_keys]
        ordered_cols = ['strategy', 'ticker', 'data_source', 'period'] + metric_cols + param_cols
        df_stress = df_stress[ordered_cols]
        out_file = RESULT_DIR / f'stress_test_results_{ticker.replace("^","")}.csv'
        df_stress.to_csv(out_file, index=False)
        print(f'✅ 壓力測試結果已輸出 → {out_file}')
    return df_stress

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        filename=cfg.LOG_DIR / f'walk_forward_v14_{dt.datetime.now().strftime("%Y%m%d")}.log',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    dfs = []
    for strat in PR.keys():
        dfs.append(run_strategy(strat))
    # 1) 合併所有 period
    all_df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)

    # 2) 先把原有的 period-level total_return 改名保留
    if 'total_return' in all_df.columns:
        all_df = all_df.rename(columns={'total_return':'period_return'})

    # 3) 計算跨期累積報酬 total_return_wf = ∏(1+ROI_wf)−1
    #    group_cols 要包含所有參數欄位（strategy, ticker, data_source, 以及其它 ind_keys、bt_keys）
    group_cols = [c for c in all_df.columns
                if c not in ['period','ROI','period_return','payoff_ratio','sharpe_ratio','ROI_wf']]
    cum = (all_df
        .groupby(group_cols)['ROI_wf']
        # 真正的複利累積：Π(1+ret) − 1
        .agg(lambda x: np.prod(1 + x) - 1)
        .reset_index(name='total_return_wf'))

    # 4) 把它合併回 all_df，並以 total_return_wf 覆蓋原 total_return 欄
    all_df = all_df.merge(cum, on=group_cols, how='left')
    all_df['total_return'] = all_df['total_return_wf']

    # 5) 最終欄位順序
    preferred_cols = ['strategy','ticker','data_source','period','ROI_wf','ROI','total_return','payoff_ratio','sharpe_ratio']
    cols = preferred_cols + [c for c in all_df.columns if c not in preferred_cols]
    all_df = all_df[cols]

    # 6) 輸出
    all_file = RESULT_DIR / f'wf_grid_ALL_{ticker.replace("^","")}.csv'
    all_df.to_csv(all_file, index=False)
    print(f'📦 所有策略 walk-forward 匯總完成 → {all_file}')
    df_cand = all_df.sort_values('total_return', ascending=False).head(20)
    print(f'📈 已挑選 {len(df_cand)} 組最佳參數進行壓力測試')
    df_stress = run_stress_test(df_cand)
    logging.disable(logging.NOTSET)