# optuna_optimization.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
from datetime import datetime

# 將專案根目錄加入模組搜尋路徑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import config as cfg
from analysis import data_loader
import SSSv095a1 as SSS
from walk_forward_v14 import run_strategy, run_stress_test
from ROEAv4 import compute_knn_stability

# 配置參數
ticker = cfg.TICKER
RESULT_DIR = cfg.RESULT_DIR
LOG_DIR = cfg.LOG_DIR
COST_PER_SHARE = cfg.BUY_FEE + cfg.SELL_FEE
TRADE_COOLDOWN_BARS = cfg.TRADE_COOLDOWN_BARS
STRESS_PERIODS = cfg.STRESS_PERIODS
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / f'optuna_optimization_{TIMESTAMP}.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 參數範圍（根據 analysis_params_v1 和 grid_search_v14 定義）
PARAM_RANGES = {
    'single': {
        'linlen': (20, 100, 10),  # (min, max, step)
        'factor': (10, 60, 10),
        'smaalen': (10, 80, 10),
        'devwin': (20, 100, 10),
        'buy_mult': (0.2, 1.0, 0.1),
        'sell_mult': (0.5, 2.0, 0.1),
        'stop_loss': (0.05, 0.15, 0.01)
    },
    'dual': {
        'linlen': (20, 100, 10),
        'factor': (10, 60, 10),
        'smaalen': (10, 80, 10),
        'short_win': (20, 60, 10),
        'long_win': (60, 120, 10),
        'buy_mult': (0.2, 1.0, 0.1),
        'sell_mult': (0.5, 2.0, 0.1),
        'stop_loss': (0.05, 0.15, 0.01)
    },
    'RMA': {
        'linlen': (20, 100, 10),
        'factor': (10, 60, 10),
        'smaalen': (10, 80, 10),
        'rma_len': (20, 80, 10),
        'dev_len': (10, 50, 10),
        'buy_mult': (0.2, 1.0, 0.1),
        'sell_mult': (0.5, 2.0, 0.1),
        'stop_loss': (0.05, 0.15, 0.01)
    },
    'ssma_turn': {
        'linlen': (10, 50, 5),
        'factor': (20, 80, 10),
        'smaalen': (20, 120, 10),
        'prom_factor': (30, 80, 5),
        'min_dist': (5, 15, 1),
        'buy_shift': (0, 5, 1),
        'exit_shift': (0, 7, 1),
        'vol_window': (10, 30, 5),
        'quantile_win': (50, 150, 10),
        'signal_cooldown_days': (1, 7, 1),
        'buy_mult': (0.2, 0.8, 0.1),
        'sell_mult': (0.2, 0.8, 0.1),
        'stop_loss': (0.05, 0.15, 0.01)
    }
}

# 目標函數
def objective(trial: optuna.Trial):
    # 選擇策略
    strategy = trial.suggest_categorical('strategy', ['single', 'dual', 'RMA', 'ssma_turn'])
    smaa_source = trial.suggest_categorical('data_source', cfg.SOURCES)
    
    # 動態生成參數
    params = {}
    for param, (min_val, max_val, step) in PARAM_RANGES[strategy].items():
        if param in ['linlen', 'smaalen', 'devwin', 'short_win', 'long_win', 'rma_len', 'dev_len',
                     'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'quantile_win', 'signal_cooldown_days']:
            params[param] = trial.suggest_int(param, min_val, max_val, step=int(step))
        else:
            params[param] = trial.suggest_float(param, min_val, max_val, step=step)
    
    # 預載數據
    try:
        df_price, df_factor = data_loader.load_data(ticker, smaa_source=smaa_source)
        if df_price.empty:
            raise ValueError(f"Failed to load price data for {ticker}")
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return -float('inf')
    
    # 執行 walk-forward 回測
    try:
        df_wf = run_strategy(strategy)
        if df_wf.empty:
            logging.warning(f"Walk-forward returned empty for {strategy}, params: {params}")
            return -float('inf')
        
        # 計算平均報酬率和交易次數
        total_return = df_wf['total_return'].mean()
        num_trades = df_wf['num_trades'].mean()
        
        # 交易次數約束
        if not (15 <= num_trades <= 60):
            penalty = abs(num_trades - 37.5) * 0.1  # 37.5 是 15~60 的中間值
            total_return -= penalty
            logging.info(f"Trade count penalty applied: {num_trades}, penalty={penalty}")
        
    except Exception as e:
        logging.error(f"Walk-forward failed: {e}")
        return -float('inf')
    
    # 執行 stress test
    try:
        df_stress = run_stress_test(df_wf.head(1))  # 使用最佳參數進行壓力測試
        if df_stress.empty:
            logging.warning(f"Stress test returned empty for {strategy}, params: {params}")
            return -float('inf')
        
        stress_return = df_stress['total_return'].mean()
        stress_std = df_stress['total_return'].std()
        
        # 穩固性得分：normal 和 stress 報酬率差距
        return_diff = abs(total_return - stress_return)
        stability_penalty = return_diff * 0.5 if return_diff > 0.03 else 0
        total_return -= stability_penalty
        
    except Exception as e:
        logging.error(f"Stress test failed: {e}")
        return -float('inf')
    
    # 檢查過擬合（使用 ROEA 的 knn_stability）
    try:
        knn_stability = compute_knn_stability(df_wf, params=['linlen', 'smaalen', 'buy_mult'], k=5)
        stability_score = np.mean(knn_stability)
        if stability_score > 0.05:  # 假設穩定性閾值
            overfit_penalty = stability_score * 0.2
            total_return -= overfit_penalty
            logging.info(f"Overfit penalty applied: stability={stability_score}, penalty={overfit_penalty}")
    except Exception as e:
        logging.error(f"KNN stability calculation failed: {e}")
        return -float('inf')
    
    # 保存試驗結果
    trial.set_user_attr('total_return', total_return)
    trial.set_user_attr('num_trades', num_trades)
    trial.set_user_attr('stress_return', stress_return)
    trial.set_user_attr('stability_score', stability_score)
    trial.set_user_attr('params', params)
    
    return total_return

# 主程式
def run_optuna_optimization(n_trials=500):
    # 建立 Optuna 研究
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # 運行優化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 保存最佳參數
    best_trial = study.best_trial
    best_params = best_trial.user_attrs['params']
    best_params['strategy'] = best_trial.params['strategy']
    best_params['data_source'] = best_trial.params['data_source']
    
    results = {
        'best_total_return': best_trial.user_attrs['total_return'],
        'best_num_trades': best_trial.user_attrs['num_trades'],
        'best_stress_return': best_trial.user_attrs['stress_return'],
        'best_stability_score': best_trial.user_attrs['stability_score'],
        'best_params': best_params
    }
    
    # 保存結果
    result_file = RESULT_DIR / f'optuna_best_params_{ticker.replace("^","")}_{TIMESTAMP}.json'
    pd.Series(results).to_json(result_file, indent=2)
    logging.info(f"Best parameters saved to {result_file}")
    
    # 繪製參數重要性
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(PLOT_DIR / f'optuna_param_importance_{ticker.replace("^","")}_{TIMESTAMP}.html')
        logging.info(f"Parameter importance plot saved")
    except Exception as e:
        logging.error(f"Failed to plot parameter importance: {e}")
    
    return study, results

if __name__ == '__main__':
    PLOT_DIR = cfg.PLOT_DIR
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    study, results = run_optuna_optimization(n_trials=500)
    print(f"最佳報酬率: {results['best_total_return']:.2%}")
    print(f"最佳交易次數: {results['best_num_trades']:.0f}")
    print(f"壓力測試報酬率: {results['best_stress_return']:.2%}")
    print(f"穩定性得分: {results['best_stability_score']:.4f}")
    print(f"最佳參數: {results['best_params']}")