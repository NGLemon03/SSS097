# analysis/optuna_16.py
# -*- coding: utf-8 -*-
"""
Optuna 16 - 高速版 (支援 sss_core, 多核心, 關閉硬碟快取)
"""

import optuna
import pandas as pd
import numpy as np
import logging
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import joblib 

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 1. 引用核心
try:
    import sss_core as SSS
except ImportError:
    print("❌ 錯誤：找不到 sss_core")
    sys.exit(1)

# 2. 引用設定檔 (修正錯誤點)
try:
    from analysis import config as cfg
except ImportError:
    print("❌ 錯誤：找不到 analysis.config")
    sys.exit(1)

try:
    from SSS_EnsembleTab import run_ensemble, RunConfig, EnsembleParams, CostParams
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Optuna16")

RESULT_DIR = ROOT / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 參數空間
param_presets = {
    'single': { 'linlen': (20, 240, 5), 'factor': (10, 50, 5), 'smaalen': (20, 200, 5), 'devwin': (10, 100, 5), 'buy_mult': (0.5, 2.0, 0.1), 'sell_mult': (0.5, 2.0, 0.1), 'stop_loss': (0.05, 0.3, 0.05) },
    'RMA': { 'linlen': (20, 240, 5), 'smaalen': (20, 200, 5), 'rma_len': (10, 100, 5), 'dev_len': (10, 100, 5), 'factor': (10, 50, 5), 'buy_mult': (0.5, 2.0, 0.1), 'sell_mult': (0.5, 2.0, 0.1), 'stop_loss': (0.05, 0.3, 0.05) },
    'ssma_turn': { 'linlen': (20, 240, 5), 'factor': (10, 100, 10), 'smaalen': (20, 200, 5), 'prom_factor': (3, 20, 1), 'min_dist': (3, 20, 1), 'buy_shift': (0, 5, 1), 'exit_shift': (0, 5, 1), 'vol_window': (10, 50, 5), 'quantile_win': (50, 200, 10), 'signal_cooldown_days': (3, 15, 1) },
    'ensemble': { 'method': ['majority', 'proportional'], 'floor': (0.0, 0.5, 0.1), 'ema_span': (1, 10, 1), 'delta_cap': (0.05, 0.5, 0.05), 'majority_k_pct': (0.3, 0.8, 0.05), 'min_cooldown_days': (1, 5, 1), 'min_trade_dw': (0.01, 0.05, 0.01) }
}

_data_cache = {}

def get_data(ticker, source="Self", end_date=None):
    key = f"{ticker}_{source}_{end_date}"
    if key in _data_cache: return _data_cache[key]
    
    # 傳入 end_date 給 SSS.load_data
    df_p, df_f = SSS.load_data(ticker, smaa_source=source, end_date=end_date)
    _data_cache[key] = (df_p, df_f)
    return df_p, df_f

def run_ensemble_backtest(params, ticker):
    # Ensemble 需要讀檔，這部分如果沒有交易檔會回傳錯誤
    # 在此腳本中通常假設 convert_results_to_trades 已經跑過，或者跳過
    return -999, 0, 0, 0, 0

def run_single_backtest(stype, params, df_p, df_f):
    try:
        if stype == 'RMA':
            calc_keys = ['linlen', 'factor', 'smaalen', 'rma_len', 'dev_len']
            calc_params = {k: params[k] for k in calc_keys if k in params}
            df = SSS.compute_RMA(df_p, df_f, **calc_params)
            res = SSS.backtest_unified(df, 'RMA', params)
        elif stype == 'single':
            calc_keys = ['linlen', 'factor', 'smaalen', 'devwin']
            calc_params = {k: params[k] for k in calc_keys if k in params}
            df = SSS.compute_single(df_p, df_f, **calc_params)
            res = SSS.backtest_unified(df, 'single', params)
        elif stype == 'ssma_turn':
            calc_keys = ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist',
                         'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']
            calc_params = {k: params[k] for k in calc_keys if k in params}
            df, buys, sells = SSS.compute_ssma_turn_combined(df_p, df_f, **calc_params)
            res = SSS.backtest_unified(df, 'ssma_turn', params, buy_dates=buys, sell_dates=sells)
        else:
            return -999, 0, 0, 0, 0

        m = res['metrics']
        return m.get('total_return', -999), m.get('num_trades', 0), m.get('sharpe_ratio', 0), m.get('max_drawdown', 0), m.get('profit_factor', 0)
    except Exception:
        return -999, 0, 0, 0, 0

def objective(trial):
    args = trial.study.user_attrs['args']
    stype = args.strategy
    score_mode = args.score_mode  # 取得評分模式

    params = {}
    preset = param_presets.get(stype, {})
    for k, v in preset.items():
        if isinstance(v, list): params[k] = trial.suggest_categorical(k, v)
        elif isinstance(v, tuple):
            if isinstance(v[0], int): params[k] = trial.suggest_int(k, v[0], v[1], step=v[2])
            else: params[k] = trial.suggest_float(k, v[0], v[1], step=v[2])

    trial.set_user_attr('parameters', params)

    if stype == 'ensemble':
        ret, n, sharpe, mdd, pf = run_ensemble_backtest(params, args.ticker)
    else:
        df_p = trial.study.user_attrs['df_price']
        df_f = trial.study.user_attrs['df_factor']
        ret, n, sharpe, mdd, pf = run_single_backtest(stype, params, df_p, df_f)

    trial.set_user_attr('total_return', ret)
    trial.set_user_attr('num_trades', n)
    trial.set_user_attr('sharpe_ratio', sharpe)
    trial.set_user_attr('max_drawdown', mdd)

    # === 根據模式計算分數 ===
    score = -9999

    # 共同過濾條件：交易次數太少直接出局
    if n < 5:
        return -9999

    if score_mode == 'smart_bh':
        # 【模式 A：穩健生存 (Smart B&H)】
        # 目標：高卡瑪比率，極度厭惡 MDD > 30%
        score_return = ret * 100
        score_risk = (abs(mdd) * 100) ** 1.5
        score = score_return - score_risk
        if abs(mdd) > 0.30:
            score -= 1000  # 重罰
        if n < 10:
            score -= 200
        trial.set_user_attr('score_mode', 'Smart_BH')

    elif score_mode == 'alpha':
        # 【模式 B：超額報酬 (Alpha Hunter)】
        # 目標：總報酬優先，MDD 容忍度較高
        score = ret * 100
        if pf > 0:
            score += pf * 5
        if abs(mdd) > 0.45:  # 容忍到 45%
            score -= (abs(mdd) - 0.45) * 2000
        else:
            score -= abs(mdd) * 10  # 輕微懲罰
        if n < 10:
            score -= 200
        trial.set_user_attr('score_mode', 'Alpha_Hunter')

    else:  # default / balanced
        # 【模式 C：原始平衡 (Balanced)】
        # Sharpe 導向
        score = (sharpe * 10) - (abs(mdd) * 20)
        if ret < 0:
            score -= 100
        trial.set_user_attr('score_mode', 'Balanced')

    return score

def save_results(study, args):
    results = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
        results.append({
            'trial_number': t.number,
            'value': t.value,
            'parameters': t.user_attrs.get('parameters'),
            'total_return': t.user_attrs.get('total_return'),
            'num_trades': t.user_attrs.get('num_trades'),
            'sharpe_ratio': t.user_attrs.get('sharpe_ratio'),
            'max_drawdown': t.user_attrs.get('max_drawdown'),
            'strategy': args.strategy,
            'data_source': args.data_source
        })
        
    if not results: return
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_source = args.data_source.replace('/', '_').replace(' ', '').replace('^', '')

    # 組合檔名：如果有 tag 就加底線，沒有就維持原樣
    tag_suffix = f"_{args.tag}" if args.tag else ""
    fname = f"optuna_results_{args.strategy}_{safe_source}{tag_suffix}_{timestamp}.csv"

    df.to_csv(RESULT_DIR / fname, index=False)
    logger.info(f"✅ 結果已存：{fname}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--data_source', default='Self')
    parser.add_argument('--ticker', default='00631L.TW')
    parser.add_argument('--n_jobs', type=int, default=-1, help='並行數')
    parser.add_argument('--train_end_date', type=str, default=None, help='訓練資料截止日 (YYYY-MM-DD)')
    parser.add_argument('--tag', type=str, default='', help='附加在檔名上的標籤 (例如: OOS_v1)')
    parser.add_argument('--score_mode', type=str, default='balanced',
                        choices=['balanced', 'smart_bh', 'alpha'],
                        help='評分模式: balanced(平衡), smart_bh(穩健), alpha(積極)')
    args = parser.parse_args()
    
    # 🔥🔥🔥 關鍵修正：正確設定 config，關閉硬碟快取 🔥🔥🔥
    try:
        # 直接修改 import 進來的 cfg 物件
        cfg.MEMORY = joblib.Memory(location=None)
        logger.info("已暫時關閉 Joblib 硬碟快取以加速 Optuna")
    except Exception as e:
        logger.warning(f"無法關閉快取，將繼續執行: {e}")
    
    study = optuna.create_study(direction='maximize')
    study.set_user_attr('args', args)
    
    if args.strategy != 'ensemble':
        # 🔥 這裡傳入 train_end_date
        df_p, df_f = get_data(args.ticker, args.data_source, end_date=args.train_end_date)
        
        # 檢查資料長度，避免切太短
        if not df_p.empty:
            logger.info(f"📅 訓練資料區間: {df_p.index[0].date()} ~ {df_p.index[-1].date()}")
        else:
            logger.error("❌ 訓練資料為空！請檢查日期設定。")
            return

        study.set_user_attr('df_price', df_p)
        study.set_user_attr('df_factor', df_f)
        
    try:
        mode_name = {'balanced': '平衡', 'smart_bh': '穩健', 'alpha': '積極'}[args.score_mode]
        logger.info(f"🚀 開始優化: {args.strategy} | 截止日 {args.train_end_date} | 評分模式: {mode_name} ({args.score_mode}) | n_jobs={args.n_jobs}")
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        
    save_results(study, args)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()