# -*- coding: utf-8 -*-
"""
Ensemble 策略包裝器 - 可插拔策略接口

提供統一的策略接口，讓 Ensemble 策略可以被 OSv3 和 Optuna 共用。
這是一個薄包裝，不復制計算核心，直接調用現有的 run_ensemble 函數。
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import ast
import glob
import os

from SSS_EnsembleTab import run_ensemble, RunConfig, EnsembleParams, CostParams

# 設置日誌
logger = logging.getLogger(__name__)

class EnsembleStrategyWrapper:
    """Ensemble 策略包裝器，提供統一的策略接口"""
    
    def __init__(self, trades_dir: str = "sss_backtest_outputs", data_path: str = "data"):
        """
        初始化 Ensemble 策略包裝器
        
        Args:
            trades_dir: 包含 trades_*.csv 文件的目錄
            data_path: 包含價格數據的目錄
        """
        self.trades_dir = Path(trades_dir)
        self.data_path = Path(data_path)
        
        # 驗證目錄存在，如果不存在則嘗試使用絕對路徑或創建目錄
        if not self.trades_dir.exists():
            # 嘗試使用絕對路徑
            abs_trades_dir = Path.cwd() / trades_dir
            if abs_trades_dir.exists():
                self.trades_dir = abs_trades_dir
                logger.info(f"使用絕對路徑: {abs_trades_dir}")
            else:
                # 嘗試創建目錄
                try:
                    self.trades_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"創建目錄: {self.trades_dir}")
                except Exception as e:
                    logger.warning(f"無法創建目錄 {trades_dir}，使用當前目錄: {e}")
                    self.trades_dir = Path.cwd()
        
        if not self.data_path.exists():
            # 嘗試使用絕對路徑
            abs_data_path = Path.cwd() / data_path
            if abs_data_path.exists():
                self.data_path = abs_data_path
                logger.info(f"使用絕對路徑: {abs_data_path}")
            else:
                # 嘗試創建目錄
                try:
                    self.data_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"創建目錄: {self.data_path}")
                except Exception as e:
                    logger.warning(f"無法創建目錄 {data_path}，使用當前目錄: {e}")
                    self.data_path = Path.cwd()
    
    def get_available_strategies(self) -> List[str]:
        """獲取可用的子策略列表"""
        trades_files = list(self.trades_dir.glob("trades_*.csv"))
        strategies = []
        
        for f in sorted(trades_files):
            name = f.stem.replace("trades_", "")
            
            # 處理新的文件名格式：trades_from_results_<strategy>_<datasource>_trial<id>
            if name.startswith("from_results_"):
                # 移除 "from_results_" 前綴
                name = name.replace("from_results_", "")
                # 將底線轉換為空白，保持一致性
                strategy_name = name.replace("_", " ")
            else:
                # 處理舊的下劃線分隔的情況
                if "_" in name:
                    parts = name.split("_")
                    if len(parts) >= 2:
                        if parts[0] in ["Single", "RMA", "STM"] and parts[1].isdigit():
                            strategy_name = f"{parts[0]} {parts[1]}"
                        else:
                            strategy_name = " ".join(parts)
                    else:
                        strategy_name = name
                else:
                    strategy_name = name
            
            # 剝掉 from_results_ 前綴（如果還有的話）
            strategy_name = strategy_name.replace('from_results_', '')
            
            strategies.append(strategy_name)
        
        return strategies
    
    def ensemble_strategy(self, 
                         method: str, 
                         params: dict, 
                         ticker: str = "00631L.TW",
                         strategies: Optional[List[str]] = None,
                         cost_params: Optional[Dict] = None) -> Tuple[pd.Series, pd.DataFrame, Dict[str, float], str, pd.DataFrame, pd.DataFrame]:
        """
        Ensemble 策略主函數
        
        Args:
            method: 集成方法 ("majority" 或 "proportional")
            params: 集成參數字典，包含：
                - floor: 底倉 (0.0-1.0)
                - ema_span: EMA 平滑天數 (1-50)
                - delta_cap: 每日權重變化上限 (0.0-1.0)
                - majority_k: 多數決門檻 (1-N)
                - min_cooldown_days: 最小冷卻天數 (1-10)
                - min_trade_dw: 最小權重變化閾值 (0.0-0.1)
            ticker: 股票代碼
            strategies: 子策略列表，None 表示自動推斷
            cost_params: 成本參數，包含：
                - buy_fee_bp: 買進費率 (bp)
                - sell_fee_bp: 賣出費率 (bp)
                - sell_tax_bp: 賣出證交税 (bp)
        
        Returns:
            equity: 權益曲線 (pd.Series)
            trades: 交易記錄 (pd.DataFrame)
            metrics: 績效指標 (dict)
            method_name: 方法名稱 (str)
            daily_state: 每日資產狀態 (pd.DataFrame)
            trade_ledger: 交易流水帳 (pd.DataFrame)
        """
        try:
            # 構建 EnsembleParams
            ensemble_params = EnsembleParams(
                floor=params.get('floor', 0.2),
                ema_span=params.get('ema_span', 3),
                delta_cap=params.get('delta_cap', 0.3),
                majority_k=params.get('majority_k', 6),
                min_cooldown_days=params.get('min_cooldown_days', 3),
                min_trade_dw=params.get('min_trade_dw', 0.02)
            )
            
            # 構建 CostParams
            cost = None
            if cost_params:
                cost = CostParams(
                    buy_fee_bp=cost_params.get('buy_fee_bp', 0.0),
                    sell_fee_bp=cost_params.get('sell_fee_bp', 0.0),
                    sell_tax_bp=cost_params.get('sell_tax_bp', 0.0)
                )
            
            # 構建 RunConfig
            config = RunConfig(
                ticker=ticker,
                method=method,
                strategies=strategies,
                params=ensemble_params,
                cost=cost
            )
            
            # 調用核心函數
            open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger = run_ensemble(config)
            
            # 確保返回的 trades 不為空
            if trades.empty:
                logger.warning("Ensemble 策略返回空的交易記錄")
                # 創建一個空的交易記錄，避免 UI 誤判
                trades = pd.DataFrame(columns=['date', 'action', 'weight', 'price'])
            
            logger.info(f"Ensemble 策略執行成功: {method_name}")
            logger.info(f"權益曲線長度: {len(equity)}, 交易記錄數: {len(trade_ledger) if trade_ledger is not None and not trade_ledger.empty else 0}")
            
            return equity, trades, stats, method_name, daily_state, trade_ledger
            
        except Exception as e:
            logger.error(f"Ensemble 策略執行失敗: {str(e)}")
            raise
    
    def get_strategy_info(self) -> Dict[str, any]:
        """獲取策略信息"""
        strategies = self.get_available_strategies()
        return {
            "name": "Ensemble Strategy",
            "type": "ensemble",
            "available_methods": ["majority", "proportional"],
            "sub_strategies_count": len(strategies),
            "sub_strategies": strategies[:10],  # 只顯示前10個
            "parameter_ranges": {
                "floor": (0.0, 1.0),
                "ema_span": (1, 50),
                "delta_cap": (0.0, 1.0),
                "majority_k": (1, 20),
                "min_cooldown_days": (1, 10),
                "min_trade_dw": (0.0, 0.1)
            }
        }


# 兼容性函數，保持與原有接口一致
def ensemble_strategy(method: str, 
                     params: dict, 
                     trades_dir: str = "sss_backtest_outputs", 
                     data_path: str = "data",
                     ticker: str = "00631L.TW",
                     strategies: Optional[List[str]] = None,
                     cost_params: Optional[Dict] = None) -> Tuple[pd.Series, pd.DataFrame, Dict[str, float], str, pd.DataFrame, pd.DataFrame]:
    """
    兼容性函數，提供與原有接口一致的調用方式
    
    Returns:
        equity: 權益曲線 (pd.Series)
        trades: 交易記錄 (pd.DataFrame) 
        metrics: 績效指標 (dict)
        method_name: 方法名稱 (str)
        daily_state: 每日資產狀態 (pd.DataFrame)
        trade_ledger: 交易流水帳 (pd.DataFrame)
    """
    wrapper = EnsembleStrategyWrapper(trades_dir, data_path)
    return wrapper.ensemble_strategy(method, params, ticker, strategies, cost_params)


# 結果轉換器函數
def convert_optuna_results_to_trades(results_dir: str = "results", 
                                   output_dir: str = "sss_backtest_outputs",
                                   top_k_per_strategy: int = 5,
                                   ticker: str = "00631L.TW") -> List[str]:
    """
    將 results 目錄下的 optuna_results_*.csv 轉換為 trades_*.csv
    
    Args:
        results_dir: 包含 optuna_results_*.csv 的目錄
        output_dir: 輸出 trades_*.csv 的目錄
        top_k_per_strategy: 每個策略取前K個最佳結果
        ticker: 股票代碼
    
    Returns:
        生成的策略名稱列表
    """
    import sys
    from pathlib import Path
    
    # 添加 SSSv096 到路徑
    sys.path.append(str(Path.cwd()))
    
    try:
        from SSSv096 import backtest_unified, load_data
    except ImportError as e:
        logger.error(f"無法導入 SSSv096.backtest_unified，請確保 SSSv096.py 在當前目錄: {e}")
        return []
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    
    # 確保輸出目錄存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 optuna_results_*.csv 文件
    optuna_files = glob.glob(str(results_path / "optuna_results_*.csv"))
    
    if not optuna_files:
        logger.warning(f"在 {results_dir} 目錄下未找到 optuna_results_*.csv 文件")
        return []
    
    generated_strategies = []
    
    for file_path in optuna_files:
        try:
            logger.info(f"處理文件: {file_path}")
            
            # 讀取CSV文件
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"文件 {file_path} 為空，跳過")
                continue
            
            # 按score排序，取前K個
            if 'score' in df.columns:
                df_sorted = df.sort_values('score', ascending=False).head(top_k_per_strategy)
            else:
                logger.warning(f"文件 {file_path} 沒有score列，跳過")
                continue
            
            # 處理每一行
            for idx, row in df_sorted.iterrows():
                try:
                    # 解析參數
                    if 'parameters' in row and pd.notna(row['parameters']):
                        try:
                            params = ast.literal_eval(row['parameters'])
                        except:
                            logger.warning(f"無法解析參數: {row['parameters']}")
                            continue
                    else:
                        continue
                    
                    # 獲取策略信息
                    strategy_name = row.get('strategy', 'unknown')
                    data_source = row.get('data_source', 'unknown')
                    trial_id = row.get('trial_number', idx)
                    
                    # 生成策略標識符
                    strategy_id = f"{strategy_name}_{data_source.replace(' ', '_').replace('^', '').replace('/', '_').replace('(', '').replace(')', '')}_trial{trial_id}"
                    
                    # 調用回測函數
                    try:
                        logger.info(f"開始回測 {strategy_id}，參數: {params}")
                        
                        # 加載數據
                        try:
                            df_price, df_factor = load_data(ticker, "2000-01-01", None, data_source)
                            if df_price.empty or df_factor.empty:
                                logger.warning(f"無法加載數據，跳過 {strategy_id}")
                                continue
                        except Exception as e:
                            logger.warning(f"加載數據失敗，跳過 {strategy_id}: {e}")
                            continue
                        
                        # 根據策略類型計算指標數據
                        if strategy_name == 'ssma_turn':
                            # 導入ssma_turn計算函數
                            try:
                                from SSSv096 import compute_ssma_turn_combined
                                # 提取ssma_turn所需參數
                                ssma_params = {
                                    'linlen': params.get('linlen', 25),
                                    'factor': params.get('factor', 80.0),
                                    'smaalen': params.get('smaalen', 85),
                                    'prom_factor': params.get('prom_factor', 9),
                                    'min_dist': params.get('min_dist', 8),
                                    'buy_shift': params.get('buy_shift', 0),
                                    'exit_shift': params.get('exit_shift', 0),
                                    'vol_window': params.get('vol_window', 20),
                                    'quantile_win': params.get('quantile_win', 100),
                                    'signal_cooldown_days': params.get('signal_cooldown_days', 10),
                                    'volume_target_pass_rate': params.get('volume_target_pass_rate', 0.65),
                                    'volume_target_lookback': params.get('volume_target_lookback', 252)
                                }
                                
                                df_ind, buy_dates, sell_dates = compute_ssma_turn_combined(
                                    df_price, df_factor, **ssma_params
                                )
                                
                                if df_ind.empty:
                                    logger.warning(f"ssma_turn 計算失敗，跳過 {strategy_id}")
                                    continue
                                
                                # 調用回測
                                result = backtest_unified(
                                    df_ind, 'ssma_turn', params, 
                                    buy_dates=buy_dates, sell_dates=sell_dates
                                )
                                
                            except ImportError as e:
                                logger.error(f"無法導入 compute_ssma_turn_combined: {e}")
                                continue
                                
                        elif strategy_name == 'single':
                            try:
                                from SSSv096 import compute_single
                                # 提取single策略所需參數
                                single_params = {
                                    'linlen': params.get('linlen', 25),
                                    'factor': params.get('factor', 80.0),
                                    'smaalen': params.get('smaalen', 85),
                                    'devwin': params.get('devwin', 9)
                                }
                                
                                df_ind = compute_single(
                                    df_price, df_factor, **single_params
                                )
                                
                                if df_ind.empty:
                                    logger.warning(f"single 計算失敗，跳過 {strategy_id}")
                                    continue
                                
                                # 調用回測
                                result = backtest_unified(df_ind, 'single', params)
                                
                            except ImportError as e:
                                logger.error(f"無法導入 compute_single: {e}")
                                continue
                                
                        elif strategy_name == 'dual':
                            try:
                                from SSSv096 import compute_dual
                                # 提取dual策略所需參數
                                dual_params = {
                                    'linlen': params.get('linlen', 25),
                                    'factor': params.get('factor', 80.0),
                                    'smaalen': params.get('smaalen', 85),
                                    'short_win': params.get('short_win', 9),
                                    'long_win': params.get('long_win', 18)
                                }
                                
                                df_ind = compute_dual(
                                    df_price, df_factor, **dual_params
                                )
                                
                                if df_ind.empty:
                                    logger.warning(f"dual 計算失敗，跳過 {strategy_id}")
                                    continue
                                
                                # 調用回測
                                result = backtest_unified(df_ind, 'dual', params)
                                
                            except ImportError as e:
                                logger.error(f"無法導入 compute_dual: {e}")
                                continue
                                
                        elif strategy_name == 'RMA':
                            try:
                                from SSSv096 import compute_RMA
                                # 提取RMA策略所需參數
                                rma_params = {
                                    'linlen': params.get('linlen', 25),
                                    'factor': params.get('factor', 80.0),
                                    'smaalen': params.get('smaalen', 85),
                                    'rma_len': params.get('rma_len', 9),
                                    'dev_len': params.get('dev_len', 18)
                                }
                                
                                df_ind = compute_RMA(
                                    df_price, df_factor, **rma_params
                                )
                                
                                if df_ind.empty:
                                    logger.warning(f"RMA 計算失敗，跳過 {strategy_id}")
                                    continue
                                
                                # 調用回測
                                result = backtest_unified(df_ind, 'RMA', params)
                                
                            except ImportError as e:
                                logger.error(f"無法導入 compute_RMA: {e}")
                                continue
                        else:
                            logger.warning(f"未知策略類型: {strategy_name}，跳過")
                            continue
                        
                        # 檢查回測結果
                        if not result or 'trade_df' not in result:
                            logger.warning(f"回測結果無效，跳過 {strategy_id}")
                            continue
                        
                        # 轉換交易記錄為標準格式
                        trades_df = result['trade_df']
                        if trades_df.empty:
                            logger.warning(f"沒有交易記錄，跳過 {strategy_id}")
                            continue
                        
                        # 轉換為標準trades格式（統一輸出 date/action/price/weight 格式）
                        standard_trades = []
                        for _, trade in trades_df.iterrows():
                            # 支持多種輸入格式，統一輸出為標準格式
                            trade_date = trade.get('trade_date') or trade.get('date')
                            trade_type = trade.get('type') or trade.get('action')
                            trade_price = trade.get('price')
                            
                            if trade_date is not None and trade_type is not None and trade_price is not None:
                                standard_trade = {
                                    'date': trade_date,  # 統一使用 date 列名
                                    'action': trade_type,  # 統一使用 action 列名
                                    'price': trade_price,
                                    'weight': 1.0 if str(trade_type).lower() == 'buy' else -1.0
                                }
                                if 'shares' in trade:
                                    standard_trade['shares'] = trade['shares']
                                standard_trades.append(standard_trade)
                        
                        if not standard_trades:
                            logger.warning(f"轉換後的交易記錄為空，跳過 {strategy_id}")
                            continue
                        
                        # 保存交易記錄
                        output_file = output_path / f"trades_from_results_{strategy_id}.csv"
                        trades_output_df = pd.DataFrame(standard_trades)
                        trades_output_df.to_csv(output_file, index=False)
                        
                        generated_strategies.append(strategy_id)
                        logger.info(f"成功生成策略: {strategy_id}，交易次數: {len(standard_trades)}")
                        
                    except Exception as e:
                        logger.error(f"回測失敗 {strategy_id}: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"處理行 {idx} 失敗: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"處理文件 {file_path} 失敗: {e}")
            continue
    
    logger.info(f"成功生成 {len(generated_strategies)} 個策略")
    return generated_strategies


def select_top_strategies_from_results(results_dir: str = "results",
                                     top_k_per_strategy: int = 5,
                                     diversity_method: str = "score_based") -> List[str]:
    """
    從results目錄中選擇最佳策略
    
    Args:
        results_dir: 包含 optuna_results_*.csv 的目錄
        top_k_per_strategy: 每個策略取前K個最佳結果
        diversity_method: 多樣性選擇方法 ("score_based", "clustering", "bucketing")
    
    Returns:
        選擇的策略名稱列表
    """
    results_path = Path(results_dir)
    
    # 查找所有 optuna_results_*.csv 文件
    optuna_files = glob.glob(str(results_path / "optuna_results_*.csv"))
    
    if not optuna_files:
        logger.warning(f"在 {results_dir} 目錄下未找到 optuna_results_*.csv 文件")
        return []
    
    selected_strategies = []
    
    for file_path in optuna_files:
        try:
            # 讀取CSV文件
            df = pd.read_csv(file_path)
            
            if df.empty:
                continue
            
            # 按score排序，取前K個
            if 'score' in df.columns:
                df_sorted = df.sort_values('score', ascending=False).head(top_k_per_strategy)
                
                for idx, row in df_sorted.iterrows():
                    strategy_name = row.get('strategy', 'unknown')
                    data_source = row.get('data_source', 'unknown')
                    trial_id = row.get('trial_number', idx)
                    
                    strategy_id = f"{strategy_name}_{data_source.replace(' ', '_').replace('^', '').replace('/', '_').replace('(', '').replace(')', '')}_trial{trial_id}"
                    selected_strategies.append(strategy_id)
                    
        except Exception as e:
            logger.error(f"處理文件 {file_path} 失敗: {e}")
            continue
    
    return selected_strategies


# 測試函數
if __name__ == "__main__":
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 測試包裝器
    wrapper = EnsembleStrategyWrapper()
    
    # 獲取可用策略信息
    info = wrapper.get_strategy_info()
    print("策略信息:", info)
    
    # 測試結果轉換器
    print("\n測試結果轉換器...")
    try:
        generated_strategies = convert_optuna_results_to_trades(
            results_dir="results",
            top_k_per_strategy=3
        )
        print(f"生成的策略: {generated_strategies}")
    except Exception as e:
        print(f"結果轉換器測試失敗: {e}")
    
    # 測試 Majority 策略
    try:
        params = {
            'floor': 0.2,
            'ema_span': 3,
            'delta_cap': 0.3,
            'majority_k': 6,
            'min_cooldown_days': 3,
            'min_trade_dw': 0.02
        }
        
        equity, trades, stats, method_name, daily_state, trade_ledger = wrapper.ensemble_strategy(
            method="majority",
            params=params,
            ticker="00631L.TW"
        )
        
        print(f"\n策略執行成功: {method_name}")
        print(f"權益曲線長度: {len(equity)}")
        print(f"交易記錄數: {len(trades)}")
        print(f"績效指標: {stats}")
        
    except Exception as e:
        print(f"測試失敗: {e}")
