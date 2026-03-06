
import argparse
from pathlib import Path
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import sys
from logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger("optuna_9")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -*- coding: utf-8 -*-
"""
批次檢視多組 SSMA_turn 參數
--------------------------------------------------
* SSMA_turn_4 用 Factor (^TWII / 2414.TW)；
  其餘使用 Self（K 線自身）當作 smaa_source
"""
import argparse
from pathlib import Path
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import sys
from logging_config import setup_logging
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

setup_logging()
logger = logging.getLogger("optuna_F")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import config as cfg
from analysis import data_loader
import SSSv096 as SSS
from Optuna_9 import (
    COST_PER_SHARE, COOLDOWN_BARS, TICKER, WF_PERIODS, STRESS_PERIODS,
    compute_period_return, analyze_trade_timing, compute_simplified_sra,
    compute_stress_mdd_equity, compute_excess_return_in_stress_equity,
    buy_and_hold_return, get_fold_period, compute_pbo_score, minmax,
    _wf_min_return, _stress_avg_return
)

# -------- 參數表 -------------
PARAM_SETS = {
    "SSMA_turn_1": {"linlen": 10, "factor": 40, "smaalen": 80, "prom_factor": 70, "min_dist": 9,
                    "buy_shift": 5, "exit_shift": 0, "vol_window": 20, "stop_loss": 0.15,
                    "quantile_win": 100, "signal_cooldown_days": 3, "buy_mult": 0.0, "sell_mult": 0.0},
    "SSMA_turn_2": {"linlen": 5, "factor": 80, "smaalen": 82, "prom_factor": 61, "min_dist": 11,
                    "buy_shift": 0, "exit_shift": 1, "vol_window": 15, "stop_loss": 0.06,
                    "quantile_win": 155, "signal_cooldown_days": 6, "buy_mult": 0.75, "sell_mult": 2.4},
    "SSMA_turn_3": {"linlen": 5, "factor": 80, "smaalen": 72, "prom_factor": 52, "min_dist": 13,
                    "buy_shift": 0, "exit_shift": 0, "vol_window": 80, "stop_loss": 0.03,
                    "quantile_win": 145, "signal_cooldown_days": 5, "buy_mult": 0.85, "sell_mult": 1.6},
    "SSMA_turn_4": {"linlen": 25, "factor": 80, "smaalen": 85, "prom_factor": 9, "min_dist": 8,
                    "buy_shift": 0, "exit_shift": 6, "vol_window": 90, "stop_loss": 0.13,
                    "quantile_win": 65, "signal_cooldown_days": 7, "buy_mult": 0.15, "sell_mult": 0.1}
}

STRATEGY = "ssma_turn"
OUT = []  # 收集結果
CPCV_NUM_SPLITS = 7
CPCV_EMBARGO_DAYS = 15
SCORE_WEIGHTS = dict(total_return=2.5, profit_factor=0.2, wf_min_return=0.2, sharpe_ratio=0.2, max_drawdown=0.1)

for name, p in PARAM_SETS.items():
    # 選擇資料來源
    source = "Factor (^TWII / 2414.TW)" if name == "SSMA_turn_4" else "Self"
    df_price, df_factor = data_loader.load_data(
        TICKER,
        start_date=cfg.START_DATE,
        end_date=datetime.now().strftime("%Y-%m-%d"),  # 更新至當前日期
        smaa_source=source
    )

    # 指標與買賣點
    ind_keys = cfg.STRATEGY_PARAMS[STRATEGY]["ind_keys"]
    ind_kwargs = {k: p[k] for k in ind_keys}
    ind_kwargs["smaa_source"] = source
    df_ind, buys, sells = SSS.compute_ssma_turn_combined(df_price, df_factor, **ind_kwargs)

    # 回測
    bt = SSS.backtest_unified(
        df_ind=df_ind,
        strategy_type=STRATEGY,
        params=p,
        buy_dates=buys,
        sell_dates=sells,
        discount=COST_PER_SHARE / 100,
        trade_cooldown_bars=COOLDOWN_BARS
    )

    metrics = bt.get("metrics", {})
    trades = bt.get("trades", [])
    equity_curve = bt.get("equity_curve", pd.Series(index=df_price.index, data=100000.0))

    # 計算進階指標
    valid_stress_periods = []
    warmup_days = max(p.get('linlen', 60), p.get('smaalen', 60), p.get('vol_window', 60), p.get('quantile_win', 60))
    valid_dates = df_price.index
    for start, end in STRESS_PERIODS:
        start_date = pd.Timestamp(start) - timedelta(days=warmup_days)
        start_candidates = valid_dates[valid_dates >= start_date]
        end_candidates = valid_dates[valid_dates <= pd.Timestamp(end)]
        if start_candidates.empty or end_candidates.empty:
            logger.warning(f"跳過無效壓力測試期間：{start} → {end}")
            continue
        adjusted_start = start_candidates[0]
        adjusted_end = end_candidates[-1]
        if (adjusted_end - adjusted_start).days < warmup_days * 2:
            logger.warning(f"壓力測試期間過短: {adjusted_start} → {adjusted_end}")
            continue
        valid_stress_periods.append((adjusted_start, adjusted_end))

    min_wf_ret = _wf_min_return(STRATEGY, p, source)
    avg_stress_ret = _stress_avg_return(STRATEGY, p, source, valid_stress_periods, df_price, trades)
    stress_mdd = compute_stress_mdd_equity(equity_curve, valid_stress_periods)
    excess_return_stress = compute_excess_return_in_stress_equity(equity_curve, df_price, valid_stress_periods)
    avg_buy_dist, avg_sell_dist = analyze_trade_timing(df_price, trades)

    # 模擬 CPCV 計算 oos_returns 和 pbo_score
    oos_returns = []
    excess_returns = []
    sra_scores = []
    n_splits = min(CPCV_NUM_SPLITS, len(df_price) // (CPCV_EMBARGO_DAYS + 60))
    min_test_len = (p.get('smaalen', 60) + p.get('quantile_win', 60)) * 1.5
    if len(df_price) / n_splits < min_test_len:
        n_splits = max(3, int(len(df_price) // min_test_len))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, test_idx in tscv.split(df_price.index):
        train_start = df_price.index[train_idx[0]]
        train_end = df_price.index[train_idx[-1]]
        test_start = df_price.index[test_idx[0]] + pd.Timedelta(days=CPCV_EMBARGO_DAYS)
        test_end = df_price.index[test_idx[-1]]
        test_start_candidates = valid_dates[valid_dates >= test_start]
        test_end_candidates = valid_dates[valid_dates <= test_end]
        if test_start_candidates.empty or test_end_candidates.empty:
            continue
        adjusted_start = test_start_candidates[0]
        adjusted_end = test_end_candidates[-1]
        folds.append(([train_start, train_end], [adjusted_start, adjusted_end]))

    for train_block, test_block in folds:
        test_results = SSS.compute_backtest_for_periods(
            ticker=TICKER, periods=[(test_block[0], test_block[1])], strategy_type=STRATEGY,
            params=p, trade_cooldown_bars=COOLDOWN_BARS, discount=COST_PER_SHARE / 100,
            df_price=df_price, df_factor=df_factor, smaa_source=source
        )
        if not test_results or 'metrics' not in test_results[0]:
            continue
        strategy_return = test_results[0]['metrics'].get('total_return', 0.0)
        test_start, test_end = test_block
        bh_return = buy_and_hold_return(df_price, test_start, test_end)
        excess_return = strategy_return - bh_return
        excess_returns.append(excess_return)
        oos_returns.append(strategy_return)
        strategy_sharpe, bh_sharpe, p_value = compute_simplified_sra(df_price, trades, [test_block])
        sra_scores.append((strategy_sharpe, p_value))

    excess_return_mean = np.mean(excess_returns) if excess_returns else 0.0
    pbo_score = compute_pbo_score(oos_returns)
    avg_p_value = np.mean([score[1] for score in sra_scores]) if sra_scores else 1.0

    # Min-Max Scaling
    total_ret = metrics.get("total_return", 0.0)
    profit_factor = metrics.get("profit_factor", 0.1)
    sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
    max_drawdown = metrics.get("max_drawdown", 0.0)
    tr_s = minmax(total_ret, 5, 25)
    pf_s = minmax(profit_factor, 0.5, 8)
    wf_s = minmax(min_wf_ret, 0, 2)
    sh_s = minmax(sharpe_ratio, 0.5, 0.8)
    mdd_s = 1 - minmax(abs(max_drawdown), 0, 0.3)
    excess_return_scaled = minmax(excess_return_mean, -1, 1)
    stress_mdd_scaled = minmax(abs(stress_mdd), 0, 0.5)
    pbo_score_scaled = pbo_score

    # Robust Score
    robust_score = 0.4 * excess_return_scaled + 0.3 * (1 - stress_mdd_scaled) + 0.3 * (1 - pbo_score_scaled)

    # 加權分數
    score = (SCORE_WEIGHTS["total_return"] * tr_s + SCORE_WEIGHTS["profit_factor"] * pf_s +
             SCORE_WEIGHTS["sharpe_ratio"] * sh_s + SCORE_WEIGHTS["max_drawdown"] * mdd_s +
             SCORE_WEIGHTS["wf_min_return"] * wf_s + robust_score)

    # 儲存結果
    OUT.append({
        "name": name,
        "data_source": source,
        **metrics,
        "min_wf_return": min_wf_ret,
        "avg_stress_return": avg_stress_ret,
        "stress_mdd": stress_mdd,
        "excess_return_stress": excess_return_stress,
        "pbo_score": pbo_score,
        "sra_p_value": avg_p_value,
        "robust_score": robust_score,
        "score": score,
        "avg_buy_dist": f"{avg_buy_dist:.1f}天",
        "avg_sell_dist": f"{avg_sell_dist:.1f}天"
    })

# ====== 匯出 / 顯示 ======
df = pd.DataFrame(OUT)
display_cols = [
    "name", "data_source", "total_return", "sharpe_ratio", "max_drawdown",
    "profit_factor", "num_trades", "min_wf_return", "avg_stress_return",
    "stress_mdd", "excess_return_stress", "pbo_score", "sra_p_value",
    "robust_score", "score", "avg_buy_dist", "avg_sell_dist"
]
print(df[display_cols].to_string(index=False))

csv_path = Path(cfg.RESULT_DIR) / f"quick_check_{TICKER.replace('^', '')}_{STRATEGY}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
logger.info(f"✔ 已存 CSV：{csv_path}")