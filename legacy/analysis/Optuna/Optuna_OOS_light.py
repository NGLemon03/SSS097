import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import optuna
import ast
import shutil
from datetime import datetime
from analysis.data_loader import load_data
from SSSv096 import backtest_unified, compute_single, compute_dual, compute_RMA, compute_ssma_turn_combined
from analysis.metrics import calculate_max_drawdown
import logging

# 設定訓練/OOS區間
TRAIN_START = "2014-10-23"
TRAIN_END = "2021-12-31"
OOS_START = "2022-01-01"
OOS_END = "2025-06-15"
TICKER = "00631L.TW"

# 使用獨立 cache 目錄，避免與 v13 衝突
CACHE_DIR = Path("../cache_OOS_light")
# 移除自動清理快取目錄，避免與其他程式衝突
# if CACHE_DIR.exists():
#     shutil.rmtree(CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 參數空間（可依需求擴充）
PARAM_SPACE = {
    "single": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), devwin=(5, 180, 1),
        factor=(40, 40, 1), buy_mult=(0.1, 2.5, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.2)),
    "dual": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), short_win=(10, 100, 5), long_win=(40, 240, 10),
        factor=(40, 40, 1), buy_mult=(0.2, 2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.1)),
    "RMA": dict(
        linlen=(5, 240, 1), smaalen=(7, 240, 5), rma_len=(20, 100, 5), dev_len=(10, 100, 5),
        factor=(40, 40, 1), buy_mult=(0.2, 2, 0.05), sell_mult=(0.5, 4.0, 0.05), stop_loss=(0.00, 0.55, 0.1)),
    "ssma_turn": dict(
        linlen=(10, 240, 5), smaalen=(10, 240, 5), factor=(40.0, 40.0, 1), prom_factor=(5, 70, 1),
        min_dist=(5, 20, 1), buy_shift=(0, 7, 1), exit_shift=(0, 7, 1), vol_window=(5, 90, 5), quantile_win=(5, 180, 10),
        signal_cooldown_days=(1, 7, 1), buy_mult=(0.5, 2, 0.05), sell_mult=(0.2, 3, 0.1), stop_loss=(0.00, 0.55, 0.1)),
}

SCORE_WEIGHTS = dict(
    total_return=2.5,
    sharpe_ratio=0.2,
    max_drawdown=0.1
)

STRAT_FUNC_MAP = {
    'single': compute_single,
    'dual': compute_dual,
    'RMA': compute_RMA,
    'ssma_turn': compute_ssma_turn_combined
}

# 參數採樣（參數名稱唯一化）
def sample_params(trial, strat):
    space = PARAM_SPACE[strat]
    params = {}
    for k, v in space.items():
        param_name = f"{strat}_{k}"
        if isinstance(v[0], int):
            low, high, step = int(v[0]), int(v[1]), int(v[2])
            params[k] = trial.suggest_int(param_name, low, high, step=step)
        else:
            low, high, step = v
            params[k] = round(trial.suggest_float(param_name, low, high, step=step), 3)
    return params

def run_backtest(strat, params, df_price, df_factor):
    if strat == "ssma_turn":
        calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
        ssma_params = {k: v for k, v in params.items() if k in calc_keys}
        buy_mult = params.get('buy_mult', 0.5)
        sell_mult = params.get('sell_mult', 0.5)
        stop_loss = params.get('stop_loss', 0.0)
        df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(df_price, df_factor, **ssma_params)
        if df_ind.empty:
            return np.nan, np.nan, np.nan
        result = backtest_unified(df_ind, strat, params, buy_dates, sell_dates, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
    else:
        if strat == 'single':
            df_ind = compute_single(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['devwin'])
        elif strat == 'dual':
            df_ind = compute_dual(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['short_win'], params['long_win'])
        elif strat == 'RMA':
            df_ind = compute_RMA(df_price, df_factor, params['linlen'], params['factor'], params['smaalen'], params['rma_len'], params['dev_len'])
        else:
            return np.nan, np.nan, np.nan
        if df_ind.empty:
            return np.nan, np.nan, np.nan
        result = backtest_unified(df_ind, strat, params, discount=0.30, trade_cooldown_bars=3, bad_holding=False)
    eq = result.get('equity_curve')
    if eq is None or eq.empty:
        return np.nan, np.nan, np.nan
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    daily_returns = eq.pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    mdd = calculate_max_drawdown(eq)
    return total_return, sharpe, mdd

def objective(trial):
    strat = trial.suggest_categorical('strategy', list(PARAM_SPACE.keys()))
    params = sample_params(trial, strat)
    # 載入訓練集
    df_price_train, df_factor_train = load_data(TICKER, start_date=TRAIN_START, end_date=TRAIN_END, smaa_source='Self')
    train_ret, train_sharpe, train_mdd = run_backtest(strat, params, df_price_train, df_factor_train)
    # 載入 OOS
    df_price_oos, df_factor_oos = load_data(TICKER, start_date=OOS_START, end_date=OOS_END, smaa_source='Self')
    oos_ret, oos_sharpe, oos_mdd = run_backtest(strat, params, df_price_oos, df_factor_oos)
    # 分數 = 0.5 * (訓練加權分數) + 0.5 * (OOS加權分數)
    def score_func(ret, sharpe, mdd):
        return (SCORE_WEIGHTS["total_return"] * ret +
                SCORE_WEIGHTS["sharpe_ratio"] * sharpe +
                SCORE_WEIGHTS["max_drawdown"] * (1 - abs(mdd)))
    if np.isnan(train_ret) or np.isnan(oos_ret):
        score = -np.inf
    else:
        train_score = score_func(train_ret, train_sharpe, train_mdd)
        oos_score = score_func(oos_ret, oos_sharpe, oos_mdd)
        score = 0.5 * train_score + 0.5 * oos_score
    trial.set_user_attr('train_return', train_ret)
    trial.set_user_attr('train_sharpe', train_sharpe)
    trial.set_user_attr('train_mdd', train_mdd)
    trial.set_user_attr('oos_return', oos_ret)
    trial.set_user_attr('oos_sharpe', oos_sharpe)
    trial.set_user_attr('oos_mdd', oos_mdd)
    # 也記錄原始參數（不帶前綴）方便輸出
    for k, v in params.items():
        trial.set_user_attr(k, v)
    return score

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=2000, show_progress_bar=True)
    # 匯出結果
    results = []
    for t in study.trials:
        row = dict()
        row['strategy'] = t.params.get('strategy', None)
        # 只輸出原始參數名（不帶前綴）
        strat = row['strategy']
        if strat is not None:
            for k in PARAM_SPACE[strat].keys():
                row[k] = t.user_attrs.get(k, np.nan)
        row['score'] = t.value
        row['train_return'] = t.user_attrs.get('train_return', np.nan)
        row['train_sharpe'] = t.user_attrs.get('train_sharpe', np.nan)
        row['train_mdd'] = t.user_attrs.get('train_mdd', np.nan)
        row['oos_return'] = t.user_attrs.get('oos_return', np.nan)
        row['oos_sharpe'] = t.user_attrs.get('oos_sharpe', np.nan)
        row['oos_mdd'] = t.user_attrs.get('oos_mdd', np.nan)
        results.append(row)
    df = pd.DataFrame(results)
    out_csv = Path("../results/optuna_oos_light_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"已輸出 {len(df)} 筆結果到 {out_csv}")

if __name__ == "__main__":
    main() 