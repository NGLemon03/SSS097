
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import sys

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analysis import config as cfg
import SSSv096 as SSS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_ssma_turn_2_v3.log')
    ]
)
logger = logging.getLogger("Test_SSMA_turn_2_v3")

COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
TICKER = cfg.TICKER
COOLDOWN_BARS = cfg.TRADE_COOLDOWN_BARS
WF_PERIODS = [(p["test"][0], p["test"][1]) for p in cfg.WF_PERIODS]
STRESS_PERIODS = cfg.STRESS_PERIODS
DATA_SOURCES = cfg.SOURCES
MIN_NUM_TRADES = 15
MAX_NUM_TRADES = 60

PARAMS_SSMA_TURN_2 = {
    "linlen": 23,
    "factor": 80.0,
    "smaalen": 82,
    "prom_factor": 52,
    "min_dist": 6,
    "buy_shift": 5,
    "exit_shift": 10,
    "vol_window": 70,
    "stop_loss": 0.0,
    "quantile_win": 50,
    "signal_cooldown_days": 7,
    "buy_mult": 0.5,
    "sell_mult": 3.3,
    "strategy_type": "ssma_turn"
}

TRIAL_RESULTS = [
    {"total_return": 15.1335, "num_trades": 19, "data_source": "Factor (^TWII / 2414.TW)", "linlen": 23, "smaalen": 82, "buy_mult": 0.5}
]

def analyze_trade_timing(df_price, trades, window=20):
    logger.info(f"價格數據索引範圍：{df_price.index.min()} 至 {df_price.index.max()}")
    buy_distances = []
    sell_distances = []
    for t in trades:
        entry, _, exit = t[0], t[1], t[2] if len(t) > 2 else (None, None, None)
        logger.info(f"交易日期：entry={entry}, exit={exit}")
        if not entry or not exit or entry not in df_price.index or exit not in df_price.index:
            logger.warning(f"無效交易日期：entry={entry}, exit={exit}")
            continue
        window_data = df_price.loc[:entry, 'close'].tail(window)
        low_idx = window_data.idxmin()
        buy_dist = (pd.Timestamp(entry) - pd.Timestamp(low_idx)).days if low_idx else 0
        buy_distances.append(buy_dist)
        window_data = df_price.loc[:exit, 'close'].tail(window)
        high_idx = window_data.idxmax()
        sell_dist = (pd.Timestamp(exit) - pd.Timestamp(high_idx)).days if high_idx else 0
        sell_distances.append(sell_dist)
    avg_buy_dist = np.mean(buy_distances) if buy_distances else 20.0
    avg_sell_dist = np.mean(sell_distances) if sell_distances else 20.0
    return avg_buy_dist, avg_sell_dist

def compute_knn_stability(df_results: list, params: list, k: int = 5, metric: str = 'total_return') -> float:
    if len(df_results) < k + 1:
        logger.warning(f"試驗數量 {len(df_results)} 不足以計算 KNN 穩定性 (需要至少 {k+1})")
        return 0.0
    param_cols = [p for p in params if p in df_results[0]]
    if not param_cols:
        logger.warning(f"無有效參數用於 KNN 穩定性計算,參數: {params}")
        return 0.0
    X = np.array([[r[p] for p in param_cols] for r in df_results])
    y = np.array([r[metric] for r in df_results])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    stability_scores = []
    for i, idx in enumerate(indices):
        roi = y[i]
        roi_neighbors = np.mean(y[idx[1:]])
        diff = min(abs(roi - roi_neighbors), 2 * roi_neighbors if roi_neighbors > 0 else 2.0)
        stability_scores.append(diff)
    return float(np.mean(stability_scores))

def _wf_min_return(strat: str, params: dict, data_source: str) -> float:
    try:
        df_price, df_factor = SSS.load_data(TICKER, start_date=cfg.START_DATE, end_date="2025-06-06", smaa_source=data_source)
        if df_price.empty:
            logger.warning(f"Walk-forward 測試數據為空,策略: {strat}, 數據源: {data_source}")
            return np.nan
        logger.info(f"WF 數據源確認：smaa_source={data_source}")
        logger.info(f"因子數據頭部：{df_factor.head()}")
        valid_dates = df_price.index
        valid_periods = []
        min_lookback = params.get('smaalen', 60) + params.get('quantile_win', 60)
        for start, end in WF_PERIODS:
            start_candidates = valid_dates[valid_dates >= pd.Timestamp(start)]
            end_candidates = valid_dates[valid_dates <= pd.Timestamp(end)]
            if start_candidates.empty or end_candidates.empty:
                logger.warning(f"無效 walk_forward 期間: {start} → {end}")
                continue
            adjusted_start = start_candidates[0]
            adjusted_end = end_candidates[-1]
            if (adjusted_end - adjusted_start).days < min_lookback * 1.5:
                logger.warning(f"Walk-forward 期間過短: {adjusted_start} → {adjusted_end}, 要求至少 {min_lookback * 1.5} 天")
                continue
            valid_periods.append((adjusted_start, adjusted_end))
        if not valid_periods:
            logger.warning(f"無有效的 walk_forward 期間,策略: {strat}, 數據源: {data_source}")
            return np.nan
        results = SSS.compute_backtest_for_periods(
            ticker=TICKER,
            periods=valid_periods,
            strategy_type=strat,
            params=params,
            trade_cooldown_bars=COOLDOWN_BARS,
            discount=COST_PER_SHARE / 100,
            df_price=df_price,
            df_factor=df_factor,
            smaa_source=data_source
        )
        valid_returns = []
        for i, r in enumerate(results):
            try:
                total_return = r["metrics"]["total_return"]
                num_trades = r["metrics"]["num_trades"]
                logger.info(f"Walk-forward 時段 {valid_periods[i][0]} 至 {valid_periods[i][1]}: 報酬={total_return:.2f}, 交易數={num_trades}")
                if np.isnan(total_return):
                    logger.warning(f"Walk-forward 時段 {valid_periods[i][0]} 至 {valid_periods[i][1]} 無有效交易, 交易數: {num_trades}")
                    continue
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logger.warning(f"Walk-forward 期間缺少 total_return,時段: {valid_periods[i]}, 錯誤: {e}")
                continue
        if not valid_returns:
            logger.warning(f"Walk-forward 無有效報酬,策略: {strat}, 數據源: {data_source}")
            return 0.0
        return min(valid_returns)
    except Exception as e:
        logger.error(f"Walk-forward 測試失敗,策略: {strat}, 數據源: {data_source}, 錯誤: {e}")
        return np.nan

def _stress_avg_return(strat: str, params: dict, data_source: str, valid_stress_periods: list) -> float:
    if not valid_stress_periods:
        logger.warning(f"無有效的壓力測試期間,策略: {strat}, 數據源: {data_source}")
        return np.nan
    try:
        df_price, df_factor = SSS.load_data(TICKER, start_date=cfg.START_DATE, end_date="2025-06-06", smaa_source=data_source)
        if df_price.empty:
            logger.warning(f"壓力測試數據為空,策略: {strat}, 數據源: {data_source}")
            return np.nan
        logger.info(f"壓力測試數據源確認：smaa_source={data_source}")
        logger.info(f"因子數據頭部：{df_factor.head()}")
        valid_periods = [(start, end) for start, end in valid_stress_periods]
        results = SSS.compute_backtest_for_periods(
            ticker=TICKER,
            periods=valid_periods,
            strategy_type=strat,
            params=params,
            trade_cooldown_bars=COOLDOWN_BARS,
            discount=COST_PER_SHARE / 100,
            df_price=df_price,
            df_factor=df_factor,
            smaa_source=data_source
        )
        valid_returns = []
        for i, r in enumerate(results):
            try:
                total_return = r["metrics"]["total_return"]
                num_trades = r["metrics"]["num_trades"]
                logger.info(f"壓力測試時段 {valid_periods[i][0]} 至 {valid_periods[i][1]}: 報酬={total_return:.2f}, 交易數={num_trades}")
                if np.isnan(total_return):
                    logger.warning(f"壓力測試時段 {valid_periods[i][0]} 至 {valid_periods[i][1]} 無有效交易")
                    continue
                valid_returns.append(total_return)
            except (KeyError, TypeError) as e:
                logger.warning(f"壓力測試期間缺少 total_return,時段: {valid_periods[i]}, 錯誤: {e}")
                continue
        return 0.0 if not valid_returns else float(np.mean(valid_returns))
    except Exception as e:
        logger.error(f"壓力測試失敗,策略: {strat}, 數據源: {data_source}, 錯誤: {e}")
        return np.nan

def _backtest_once(strat: str, params: dict, trial_results: list, data_source: str, 
                   df_price: pd.DataFrame, df_factor: pd.DataFrame) -> tuple:
    try:
        if df_price.empty:
            logger.error(f"價格數據為空,策略: {strat}, 數據源: {data_source}")
            return -np.inf, 0, 0.0, 0.0, 0.0, []
        df_price.name = TICKER.replace(':', '_')
        if not df_factor.empty and (not hasattr(df_factor, 'name') or df_factor.name is None):
            df_factor.name = f"{TICKER}_factor"
        compute_f = SSS.compute_ssma_turn_combined
        ind_keys = cfg.STRATEGY_PARAMS[strat]["ind_keys"]
        ind_p = {k: params[k] for k in ind_keys}
        ind_p["smaa_source"] = data_source
        df_ind, buys, sells = compute_f(df_price, df_factor, **ind_p)
        if df_ind.empty:
            logger.warning(f"計算指標失敗,策略: {strat}, 數據源: {data_source}")
            return -np.inf, 0, 0.0, 0.0, 0.0, []
        bt = SSS.backtest_unified(
            df_ind=df_ind,
            strategy_type=strat,
            params=params,
            buy_dates=buys,
            sell_dates=sells,
            discount=COST_PER_SHARE / 100,
            trade_cooldown_bars=COOLDOWN_BARS,
        )
        metrics = bt["metrics"]
        trades_df = bt.get("trades_df")
        if trades_df is None:
            logger.warning(f"回測未返回 trades_df,策略: {strat}, 數據源: {data_source}")
            return -np.inf, 0, 0.0, 0.0, 0.0, []
        profit_factor = metrics.get("profit_factor", 0.1)
        if "profit_factor" not in metrics or pd.isna(profit_factor):
            trades = bt["trades"]
            if trades:
                gains = [t[1] for t in trades if t[1] > 0]
                losses = [abs(t[1]) for t in trades if t[1] < 0]
                profit_factor = sum(gains) / sum(losses) if losses and sum(losses) > 0 else np.inf
                profit_factor = min(profit_factor, 10.0)
        trial_results.append({
            "total_return": metrics.get("total_return", 0.0),
            "num_trades": metrics.get("num_trades", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "profit_factor": profit_factor,
            "data_source": data_source,
            **params
        })
        logger.info(f"回測完成,策略: {strat}, 數據源: {data_source}, 總報酬={metrics.get('total_return', 0.0)*100:.2f}%, 交易次數={metrics.get('num_trades', 0)}")
        return (
            metrics.get("total_return", 0.0),
            metrics.get("num_trades", 0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("max_drawdown", 0.0),
            profit_factor,
            bt["trades"]
        )
    except Exception as e:
        logger.error(f"回測失敗,策略: {strat}, 數據源: {data_source}, 錯誤: {e}")
        return -np.inf, 0, 0.0, 0.0, 0.0, []

def run_test():
    strat = "ssma_turn"
    data_source = "Factor (^TWII / 2414.TW)"
    params = PARAMS_SSMA_TURN_2.copy()
    params.pop("strategy_type")

    logger.info(f"開始測試，策略: {strat}, 數據源: {data_source}, 參數: {params}")

    df_price, df_factor = SSS.load_data(TICKER, start_date=cfg.START_DATE, end_date="2025-06-06", smaa_source=data_source)
    if df_price.empty:
        logger.error(f"價格數據為空，策略: {strat}, 數據源: {data_source}")
        return
    logger.info(f"完整回測數據源確認：smaa_source={data_source}")
    df_factor.to_csv("s1_factor_data.csv")

    total_ret, n_trades, sharpe_ratio, max_drawdown, profit_factor, trades = _backtest_once(
        strat, params, TRIAL_RESULTS, data_source, df_price, df_factor
    )

    if total_ret == -np.inf:
        logger.error("回測失敗，退出測試")
        return

    if not (MIN_NUM_TRADES <= n_trades <= MAX_NUM_TRADES):
        logger.warning(f"交易次數 {n_trades} 不在 {MIN_NUM_TRADES}-{MAX_NUM_TRADES} 範圍內")
        return

    min_wf_ret = _wf_min_return(strat, params, data_source)
    logger.info(f"WF 最差報酬: {min_wf_ret:.2f}")

    valid_dates = df_price.index
    valid_stress_periods = []
    for start, end in STRESS_PERIODS:
        start_candidates = valid_dates[valid_dates >= pd.Timestamp(start)]
        end_candidates = valid_dates[valid_dates <= pd.Timestamp(end)]
        if start_candidates.empty or end_candidates.empty:
            logger.warning(f"跳過無效壓力測試期間：{start} → {end}")
            continue
        adjusted_start = start_candidates[0]
        adjusted_end = end_candidates[-1]
        valid_stress_periods.append((adjusted_start, adjusted_end))

    avg_stress_ret = _stress_avg_return(strat, params, data_source, valid_stress_periods)
    logger.info(f"壓力平均報酬: {avg_stress_ret:.3f}")

    stab = compute_knn_stability(TRIAL_RESULTS, params=['linlen', 'smaalen', 'buy_mult'], k=5)
    logger.info(f"穩定性得分: {stab:.2f}")

    avg_buy_dist, avg_sell_dist = analyze_trade_timing(df_price, trades)
    logger.info(f"買入距離低點: {avg_buy_dist:.1f} 天, 賣出距離高點: {avg_sell_dist:.1f} 天")

    result = {
        "strategy": strat,
        "data_source": data_source,
        "total_return": total_ret,
        "num_trades": n_trades,
        "min_wf_return": min_wf_ret,
        "avg_stress_return": avg_stress_ret,
        "stability_score": stab,
        "avg_buy_dist": avg_buy_dist,
        "avg_sell_dist": avg_sell_dist
    }
    pd.DataFrame([result]).to_csv("test_ssma_turn_2_v3_results.csv", index=False, encoding='utf-8-sig', na_rep='0.0')
    logger.info("測試結果已保存至 test_ssma_turn_2_v3_results.csv")

if __name__ == "__main__":
    run_test()
