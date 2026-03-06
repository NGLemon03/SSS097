
from __future__ import annotations
import argparse
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-
"""
SSS_EnsembleTab.py

兩種可直接在 SSS 專案中使用的「多策略組合」方法：
1) Majority K-of-N（多數決，預設使用比例門檻 majority_k_pct=0.55）
2) Proportional（依多頭比例分配）

嚴格遵守 T+1：N 日收盤產生訊號 -> N+1 開盤用於交易；
權益以 Open-to-Open 報酬遞推。

使用方式（在專案根目錄執行）：
    python SSS_EnsembleTab.py --ticker 00631L.TW --method majority \
        --floor 0.2 --ema 3 --delta 0.3
    python SSS_EnsembleTab.py --ticker 00631L.TW --method proportional \
        --floor 0.2 --ema 3 --delta 0.3

輸出：
- sss_backtest_outputs/ensemble_*.csv：
  - ensemble_weights_<name>.csv  : 每日權重 w[t]
  - ensemble_equity_<name>.csv   : 權益曲線（Open→Open）
  - ensemble_trades_<name>.csv   : 依權重變化生成的交易事件（t 開盤生效）
- sss_backtest_outputs/ensemble_summary.csv（附加模式）：各組合方法摘要績效

注意：
- 本檔讀取 SSS 在 sss_backtest_outputs/ 下既有的 trades_*.csv 來重建各子策略的「次日開盤生效」部位序列。
- 優先使用 trades_from_results_*.csv（120檔策略），找不到才使用舊的 trades_*.csv（11檔策略）。
- 成本與滑點可在參數中設定（預設使用 param_presets 中的配置）；
  台股實盤：buy_fee_bp=4.27、sell_fee_bp=4.27、sell_tax_bp=30（單位為 bp=萬分之一）。
"""



""" 2025/12/20 00:28
這個版本做了兩個關鍵改動：
1.  **內建標準化函式**：移除對 `sss_core` 的強制依賴。這樣你直接跑 `run_enhanced_ensemble.py` 就不會因為找不到模組而報錯。
2.  **強化 `RunConfig`**：確保它完全支援 Walk-Forward 需要的 `file_map` 參數。
SSS_EnsembleTab.py - Ensemble 策略核心邏輯 (獨立運行版)
移除外部依賴，確保 Walk-Forward 與預測腳本能順利介接。
"""

# 設置 logger
logger = logging.getLogger("SSS.Ensemble")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 路徑設定
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "sss_backtest_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 內建標準化工具 (移除對 sss_core 的依賴以確保獨立運行)
# ---------------------------------------------------------------------
def normalize_trades_for_ui(trades_df: pd.DataFrame) -> pd.DataFrame:
    """標準化交易明細格式供 UI 顯示"""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    # 統一欄位名稱
    col_map = {
        'date': 'trade_date',
        'action': 'type',
        'side': 'type',
        'open': 'price',
        'price_open': 'price'
    }
    df = df.rename(columns=col_map)
    # 確保必要欄位存在
    if 'trade_date' not in df.columns:
        df['trade_date'] = pd.NaT
    if 'type' not in df.columns:
        df['type'] = 'unknown'
    if 'price' not in df.columns:
        df['price'] = np.nan
    return df

# ---------------------------------------------------------------------
# 資料讀取
# ---------------------------------------------------------------------
def _read_market_csv_auto(path: Path) -> pd.DataFrame:
    """讀取市場數據 (Open/High/Low/Close)"""
    try:
        if not path.exists():
            logger.error(f"找不到價格檔案: {path}")
            return pd.DataFrame()
            
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        
        # 處理日期索引
        date_col = cols.get("date", df.columns[0])
        df["date"] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        
        # 處理價格欄位
        for k in ["open", "high", "low", "close", "volume"]:
            if k in cols:
                df[k] = pd.to_numeric(df[cols[k]], errors="coerce")
            else:
                # 兼容舊格式
                if k == "open" and "open" not in cols:
                    df["open"] = df.iloc[:, 0] if df.shape[1] > 0 else np.nan
        
        return df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"])
    except Exception as e:
        logger.error(f"讀取價格檔案失敗 {path}: {e}")
        return pd.DataFrame()

def build_position_from_trades(trade_csv: Path, index: pd.DatetimeIndex) -> pd.Series:
    """從交易記錄重建持倉序列 (0/1)"""
    try:
        df = pd.read_csv(trade_csv)
        # 標準化欄位
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get('trade_date', cols.get('date'))
        type_col = cols.get('type', cols.get('action'))
        
        if not date_col or not type_col:
            return pd.Series(0.0, index=index)

        df['dt'] = pd.to_datetime(df[date_col])
        df['act'] = df[type_col].astype(str).str.lower()
        
        pos = pd.Series(0.0, index=index)
        for _, row in df.iterrows():
            dt, act = row['dt'], row['act']
            if dt in pos.index:
                if act in ('buy', 'long', 'entry'):
                    pos.loc[dt:] = 1.0
                elif act in ('sell', 'short', 'exit', 'sell_forced'):
                    pos.loc[dt:] = 0.0
        return pos
    except Exception as e:
        logger.warning(f"處理交易檔 {trade_csv.name} 失敗: {e}")
        return pd.Series(0.0, index=index)

def load_positions_matrix(index: pd.DatetimeIndex, strategies: List[str], file_map: Dict[str, Path] = None) -> pd.DataFrame:
    """載入多策略持倉矩陣"""
    positions = {}

    # ⚠️ 防呆保護：檢查策略清單是否為空
    if not strategies:
        logger.warning("⚠️ load_positions_matrix: 未提供策略清單 (strategies is None/Empty)")
        return pd.DataFrame(index=index)

    # 自動補全 file_map (如果沒傳入)
    if file_map is None:
        file_map = {}
        # 搜尋 outputs 目錄
        for f in OUT_DIR.glob("trades_from_results_*.csv"):
            # 建立多種 key 格式以增加命中率
            name_full = f.stem  # 完整檔名
            name_short = f.stem.replace("trades_from_results_", "").replace("trades_", "")

            file_map[name_full] = f
            file_map[name_short] = f

    for strat in strategies:
        # 嘗試直接獲取
        csv_path = file_map.get(strat)

        # 如果找不到，嘗試模糊匹配 (這是關鍵修復！)
        if not csv_path:
            # 1. 嘗試加上前綴找
            csv_path = file_map.get(f"trades_from_results_{strat}")

            # 2. 嘗試在 file_map 的所有 key 中找包含 strat 的
            if not csv_path:
                for k, v in file_map.items():
                    if strat in k:
                        csv_path = v
                        break

        if csv_path and csv_path.exists():
            pos = build_position_from_trades(csv_path, index)
            # 只有當持倉有變化（不是全 0）才加入，避免無效策略
            if pos.std() > 0 or pos.sum() > 0:
                positions[strat] = pos
            else:
                logger.warning(f"⚠️ 策略 {strat} 持倉全為 0，已忽略")
        else:
            logger.warning(f"⚠️ 找不到策略檔案: {strat} (嘗試路徑: {csv_path})")

    if not positions:
        logger.error("❌ 無法載入任何有效策略持倉")
        return pd.DataFrame(index=index)

    return pd.DataFrame(positions, index=index)

# ---------------------------------------------------------------------
# Ensemble 核心參數與邏輯
# ---------------------------------------------------------------------

@dataclass
class EnsembleParams:
    floor: float = 0.1 # 底倉比例,原始設定=0.1
    ema_span: int = 3
    delta_cap: float = 0.3  # 舊版單一限制（保留向後兼容）
    majority_k: int = 6
    min_cooldown_days: int = 1 # 權重變化冷卻天數 原始設定=1
    min_trade_dw: float = 0.01 # 權重變化最小門檻 原始設定=0.01

    # 🔥 不對稱平滑參數 (可選，預設為 None = 不啟用)
    delta_cap_buy: float = None   # 加倉限制 (上樓慢)
    delta_cap_sell: float = None  # 減倉限制 (下樓快)
    enable_asymmetric: bool = False  # 是否啟用不對稱平滑

    def __post_init__(self):
        """初始化後處理：確保不對稱參數的向後兼容性"""
        # 如果啟用不對稱但沒設定參數，使用 delta_cap 作為預設值
        if self.enable_asymmetric:
            if self.delta_cap_buy is None:
                self.delta_cap_buy = self.delta_cap
            if self.delta_cap_sell is None:
                self.delta_cap_sell = self.delta_cap
        else:
            # 未啟用不對稱時，強制使用對稱邏輯
            self.delta_cap_buy = self.delta_cap
            self.delta_cap_sell = self.delta_cap

@dataclass
class CostParams:
    buy_fee_bp: float = 4.27
    sell_fee_bp: float = 4.27
    sell_tax_bp: float = 30.0
    
    @property
    def buy_rate(self): return self.buy_fee_bp / 10000.0
    @property
    def sell_rate(self): return (self.sell_fee_bp + self.sell_tax_bp) / 10000.0

@dataclass
class RunConfig:
    ticker: str
    method: str  # 'majority' or 'proportional'
    strategies: List[str] | None = None
    params: EnsembleParams = field(default_factory=EnsembleParams)
    cost: CostParams = field(default_factory=CostParams)
    file_map: Dict[str, Path] = None  # 關鍵：傳入檔案路徑對照表
    majority_k_pct: float = None      # 關鍵：比例門檻

def weights_majority(pos_df: pd.DataFrame, p: EnsembleParams) -> pd.Series:
    """多數決權重計算"""
    S = pos_df.sum(axis=1)
    w_raw = (S >= p.majority_k).astype(float)
    # 底倉緩衝
    w = w_raw * (1 - p.floor) + p.floor
    return _smooth_and_cap(w, p)

def weights_proportional(pos_df: pd.DataFrame, p: EnsembleParams) -> pd.Series:
    """比例權重計算"""
    N = pos_df.shape[1]
    S = pos_df.sum(axis=1)
    w_raw = S / max(N, 1)
    w = w_raw * (1 - p.floor) + p.floor
    return _smooth_and_cap(w, p)

def _smooth_and_cap(w: pd.Series, p: EnsembleParams) -> pd.Series:
    """
    平滑與限制邏輯 (支持不對稱平滑)

    Args:
        w: 原始權重序列
        p: Ensemble 參數 (包含不對稱設定)

    Returns:
        處理後的權重序列
    """
    # 1. EMA 平滑
    if p.ema_span > 1:
        w_smooth = w.ewm(span=p.ema_span, adjust=False).mean()
    else:
        w_smooth = w.copy()

    # 2. Delta Cap (每日變化上限) - 🔥 支持不對稱
    w_out = w_smooth.copy()

    for i in range(1, len(w_smooth)):
        delta = w_smooth.iloc[i] - w_out.iloc[i-1]

        if p.enable_asymmetric:
            # 🔥 不對稱模式：加倉嚴格、減倉寬鬆
            if delta > 0:
                # 加倉：使用 delta_cap_buy (更嚴格)
                if delta > p.delta_cap_buy:
                    w_out.iloc[i] = w_out.iloc[i-1] + p.delta_cap_buy
            else:
                # 減倉：使用 delta_cap_sell (更寬鬆)
                if abs(delta) > p.delta_cap_sell:
                    w_out.iloc[i] = w_out.iloc[i-1] - p.delta_cap_sell
        else:
            # 傳統對稱模式
            if abs(delta) > p.delta_cap:
                w_out.iloc[i] = w_out.iloc[i-1] + (p.delta_cap if delta > 0 else -p.delta_cap)

    # 3. Cooldown & Min Threshold
    w_final = w_out.copy()
    last_val = w_final.iloc[0]
    last_chg_idx = 0

    for i in range(1, len(w_final)):
        curr = w_final.iloc[i]
        if abs(curr - last_val) < p.min_trade_dw:
            w_final.iloc[i] = last_val
        elif (i - last_chg_idx) < p.min_cooldown_days:
            w_final.iloc[i] = last_val
        else:
            last_val = curr
            last_chg_idx = i

    return w_final.clip(0, 1)

def run_ensemble(cfg: RunConfig, price_df: pd.DataFrame = None):
    """
    Ensemble 主程式

    Args:
        cfg: 配置參數
        price_df: 可選的價格數據 DataFrame（如果提供，將優先使用；否則從 CSV 讀取）

    回傳: (open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger)
    """
    # 1. 讀取價格（優先使用傳入的 price_df，確保使用最新數據）
    if price_df is not None and not price_df.empty:
        px_df = price_df
        logger.info(f"[Ensemble] 使用傳入的最新價格數據（結束日期: {px_df.index[-1].date()}）")
    else:
        # 回退到讀取 CSV
        px_path = DATA_DIR / f"{cfg.ticker.replace(':','_')}_data_raw.csv"
        px_df = _read_market_csv_auto(px_path)
        if px_df.empty:
            logger.error(f"無法讀取 {cfg.ticker} 價格數據")
            return None, None, pd.DataFrame(), {}, "Error", None, None, None
        logger.info(f"[Ensemble] 從 CSV 讀取價格數據（結束日期: {px_df.index[-1].date()}）")

    open_px = px_df['open']
    
    # 2. 載入策略持倉
    pos_df = load_positions_matrix(px_df.index, cfg.strategies, cfg.file_map)
    N = pos_df.shape[1]
    
    if N == 0:
        return open_px, pd.Series(0, index=open_px.index), pd.DataFrame(), {}, "NoStrategies", None, None, None

    # 3. 計算多多數決門檻 (若有指定比例)
    if cfg.method == 'majority' and cfg.majority_k_pct:
        cfg.params.majority_k = max(1, int(math.ceil(N * cfg.majority_k_pct)))
        logger.info(f"多數決動態門檻: {cfg.params.majority_k} / {N} ({(cfg.majority_k_pct*100):.0f}%)")

    # 4. 計算權重
    if cfg.method == 'majority':
        w = weights_majority(pos_df, cfg.params)
        method_name = f"Majority_{cfg.params.majority_k}_of_{N}"
    else:
        w = weights_proportional(pos_df, cfg.params)
        method_name = f"Prop_{N}"

    # 5. 計算權益與交易 (Open-to-Open)
    equity, trades, daily_state, trade_ledger = calculate_performance(open_px, w, cfg.cost)

    # 6. 計算績效統計（完整版，包含勝率、盈虧比等）
    if len(equity) < 2:
        # 數據不足，返回空統計
        stats = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'annualized_volatility': 0.0,
            'time_in_market': 0.0,
            'turnover_py': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_win': np.nan,
            'avg_loss': np.nan,
            'payoff_ratio': np.nan,
            'profit_factor': np.nan
        }
    else:
        # === 基礎指標（基於權益曲線）===
        # 日報酬率
        daily_ret = equity.pct_change().dropna()

        # 總報酬與年化
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        days = (equity.index[-1] - equity.index[0]).days
        years = max(days / 365.25, 0.01)  # 避免除以零
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 最大回撤
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()

        # 回撤持續天數
        is_in_dd = drawdown < 0
        if is_in_dd.any():
            dd_duration = is_in_dd.astype(int).groupby(is_in_dd.ne(is_in_dd.shift()).cumsum()).cumsum().max()
        else:
            dd_duration = 0

        # 風險調整指標
        sharpe_ratio = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
        calmar_ratio = (annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        downside_ret = daily_ret[daily_ret < 0]
        sortino_ratio = (daily_ret.mean() / downside_ret.std() * np.sqrt(252)) if len(downside_ret) > 0 and downside_ret.std() > 0 else 0.0

        annualized_volatility = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 0 else 0.0

        # 市場參與度與週轉率
        time_in_market = (w > 0.05).mean()  # 持倉 > 5% 算在市場內
        turnover_py = w.diff().abs().sum() / len(w) * 252 if len(w) > 0 else 0.0

        # === 進階交易指標（基於 LIFO 計算）===
        num_trades = len(trades)
        win_rate = 0.0
        avg_win = np.nan
        avg_loss = np.nan
        payoff_ratio = np.nan
        profit_factor = np.nan

        if not trades.empty:
            # 使用 LIFO 計算每筆賣出的報酬率
            sell_returns = _calculate_lifo_returns(trades)
            # 過濾出有效的賣出交易
            sell_returns_valid = sell_returns[~np.isnan(sell_returns)]

            if len(sell_returns_valid) > 0:
                wins = sell_returns_valid[sell_returns_valid > 0]
                losses = sell_returns_valid[sell_returns_valid <= 0]

                win_rate = len(wins) / len(sell_returns_valid)
                avg_win = wins.mean() if len(wins) > 0 else 0.0
                avg_loss = losses.mean() if len(losses) > 0 else 0.0

                if avg_loss != 0:
                    payoff_ratio = abs(avg_win / avg_loss)

                # 獲利因子 (總獲利 / 總虧損)
                gross_profit = wins.sum() if len(wins) > 0 else 0.0
                gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
                if gross_loss != 0:
                    profit_factor = gross_profit / gross_loss

        # 最大連續盈利/虧損
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        if not trades.empty:
            sell_returns = _calculate_lifo_returns(trades)
            sell_returns_valid = sell_returns[~np.isnan(sell_returns)]
            if len(sell_returns_valid) > 0:
                win_flag = pd.Series(sell_returns_valid) > 0
                grp = (win_flag != win_flag.shift()).cumsum()
                consec = win_flag.groupby(grp).cumcount() + 1
                max_consecutive_wins = int(consec[win_flag].max() if True in win_flag.values else 0)
                max_consecutive_losses = int(consec[~win_flag].max() if False in win_flag.values else 0)

        stats = {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_duration': int(dd_duration),
            'sharpe_ratio': float(sharpe_ratio),
            'calmar_ratio': float(calmar_ratio),
            'sortino_ratio': float(sortino_ratio),
            'annualized_volatility': float(annualized_volatility),
            'time_in_market': float(time_in_market),
            'turnover_py': float(turnover_py),
            'num_trades': int(num_trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win) if not np.isnan(avg_win) else np.nan,
            'avg_loss': float(avg_loss) if not np.isnan(avg_loss) else np.nan,
            'payoff_ratio': float(payoff_ratio) if not np.isnan(payoff_ratio) else np.nan,
            'profit_factor': float(profit_factor) if not np.isnan(profit_factor) else np.nan,
            'max_consecutive_wins': int(max_consecutive_wins),
            'max_consecutive_losses': int(max_consecutive_losses)
        }

    return open_px, w, trades, stats, method_name, equity, daily_state, trade_ledger

def _calculate_lifo_returns(trades_df: pd.DataFrame) -> np.ndarray:
    """
    使用 LIFO (Last In First Out) 演算法計算每筆賣出交易的報酬率

    Args:
        trades_df: 交易記錄，必須包含 trade_date, type, price, weight_change 欄位

    Returns:
        numpy array of returns for sell trades (NaN for buy trades)
    """
    if trades_df is None or trades_df.empty:
        return np.array([])

    # 建立副本並確保欄位名稱一致
    df = trades_df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    # 檢查必要欄位
    if 'type' not in df.columns or 'price' not in df.columns:
        return np.array([])

    # 尋找數量欄位
    qty_col = None
    if 'shares' in df.columns:
        qty_col = 'shares'
    elif 'weight_change' in df.columns:
        qty_col = 'weight_change'
    elif 'delta_units' in df.columns:
        qty_col = 'delta_units'

    if not qty_col:
        return np.array([])

    # === LIFO (先進後出) 計算核心 ===
    inventory = []  # 庫存堆疊: [{'price': float, 'qty': float}, ...]
    returns = []

    # 確保按時間正序排列
    df = df.sort_values('trade_date') if 'trade_date' in df.columns else df

    for idx, row in df.iterrows():
        try:
            trade_type = str(row['type']).lower()
            price = float(row['price'])
            qty = abs(float(row[qty_col]))  # 取絕對值

            if qty == 0:
                returns.append(np.nan)
                continue

            if 'buy' in trade_type or 'add' in trade_type or 'long' in trade_type:
                # 買入：推入堆疊
                inventory.append({'price': price, 'qty': qty})
                returns.append(np.nan)

            elif 'sell' in trade_type or 'exit' in trade_type:
                # 賣出：從堆疊尾端開始扣 (LIFO)
                remaining_sell_qty = qty
                total_cost = 0.0
                matched_qty = 0.0

                # 倒著遍歷 inventory
                while remaining_sell_qty > 0 and inventory:
                    last_batch = inventory[-1]

                    if last_batch['qty'] <= remaining_sell_qty:
                        # 這批不夠賣，全部吃掉
                        cost = last_batch['qty'] * last_batch['price']
                        total_cost += cost
                        matched_qty += last_batch['qty']
                        remaining_sell_qty -= last_batch['qty']
                        inventory.pop()
                    else:
                        # 這批夠賣，吃掉一部分
                        cost = remaining_sell_qty * last_batch['price']
                        total_cost += cost
                        matched_qty += remaining_sell_qty
                        inventory[-1]['qty'] -= remaining_sell_qty
                        remaining_sell_qty = 0

                # 計算損益
                if matched_qty > 0:
                    avg_buy_price = total_cost / matched_qty
                    ret = (price - avg_buy_price) / avg_buy_price
                    returns.append(ret)
                else:
                    # 空庫存賣出
                    returns.append(0.0)
            else:
                returns.append(np.nan)

        except Exception:
            returns.append(np.nan)

    return np.array(returns)

def calculate_performance(open_px: pd.Series, w: pd.Series, cost: CostParams):
    """計算回測績效，並輸出可直接顯示的完整交易欄位。"""
    cash = 1_000_000.0
    shares = 0.0
    equity_curve = []
    trades = []
    ledger = []

    # 對齊 index
    common_idx = open_px.index.intersection(w.index)
    price = open_px.loc[common_idx]
    weight = w.loc[common_idx]

    # 交易日 -> 訊號日（前一個交易日）
    signal_date_map = {}
    for i, dt in enumerate(common_idx):
        signal_date_map[dt] = common_idx[i - 1] if i > 0 else pd.NaT

    sell_fee_rate = float(cost.sell_fee_bp) / 10000.0
    sell_tax_rate = float(cost.sell_tax_bp) / 10000.0

    curr_w = 0.0

    for dt, px in price.items():
        target_w = float(weight.loc[dt])
        px = float(px)

        # 計算當前資產
        curr_equity = cash + shares * px

        # 權重變化（有變化才記交易）
        if np.isfinite(px) and px > 0 and abs(target_w - curr_w) > 0.001:
            w_before = float(curr_w)
            target_val = curr_equity * target_w
            curr_val = shares * px
            diff_val = float(target_val - curr_val)

            trade_type = None
            reason = None
            traded_shares = 0.0
            fee_amt = 0.0
            tax_amt = 0.0
            net_amount = 0.0

            if diff_val > 0:  # 買入
                fee_amt = diff_val * cost.buy_rate
                buy_val = max(diff_val - fee_amt, 0.0)
                traded_shares = buy_val / px if px > 0 else 0.0
                shares += traded_shares
                cash -= diff_val  # 現金減少（含手續費）

                trade_type = "buy"
                reason = "ensemble_rebalance_buy"
                net_amount = -diff_val

            elif diff_val < 0:  # 賣出
                sell_val = abs(diff_val)
                fee_amt = sell_val * sell_fee_rate
                tax_amt = sell_val * sell_tax_rate
                total_cost = fee_amt + tax_amt
                real_get = max(sell_val - total_cost, 0.0)
                traded_shares = sell_val / px if px > 0 else 0.0
                shares -= traded_shares
                if abs(shares) < 1e-10:
                    shares = 0.0
                cash += real_get

                trade_type = "sell"
                reason = "ensemble_rebalance_sell"
                net_amount = real_get

            if trade_type is not None:
                curr_w = target_w
                total_equity_after = cash + shares * px
                position_value = shares * px
                total_for_pct = total_equity_after if total_equity_after != 0 else np.nan

                trades.append(
                    {
                        "trade_date": dt,
                        "signal_date": signal_date_map.get(dt, pd.NaT),
                        "type": trade_type,
                        "price": px,
                        "weight_change": target_w - w_before,
                        "w_before": w_before,
                        "w_after": target_w,
                        "shares": traded_shares,
                        "reason": reason,
                        "fee": fee_amt,
                        "tax": tax_amt,
                        "net_amount": net_amount,
                        "leverage_ratio": 1.0,
                        "strategy_version": "ensemble_v1",
                        "exec_notional": abs(diff_val),
                        "equity_after": total_equity_after,
                        "cash_after": cash,
                        "equity_pct": (position_value / total_for_pct) if pd.notna(total_for_pct) else np.nan,
                        "cash_pct": (cash / total_for_pct) if pd.notna(total_for_pct) else np.nan,
                        "invested_pct": (position_value / total_for_pct) if pd.notna(total_for_pct) else np.nan,
                        "position_value": position_value,
                    }
                )

        # 紀錄每日資產
        total_equity = cash + shares * px
        equity_curve.append(total_equity)

        ledger_row = {
            "date": dt,
            "equity": total_equity,
            "cash": cash,
            "w": curr_w,
            "position_value": shares * px,
        }
        ledger.append(ledger_row)

    equity_series = pd.Series(equity_curve, index=common_idx)
    trades_df = pd.DataFrame(trades)
    ledger_df = pd.DataFrame(ledger).set_index("date")

    # 補上單筆報酬（僅賣出行會有值）
    if not trades_df.empty:
        try:
            lifo_ret = _calculate_lifo_returns(trades_df)
            if len(lifo_ret) == len(trades_df):
                trades_df["return"] = lifo_ret
        except Exception as e:
            logger.warning(f"LIFO 報酬計算失敗，略過 return 欄位: {e}")

    return equity_series, trades_df, ledger_df, trades_df  # trade_ledger 沿用交易明細

# ---------------------------------------------------------------------
# 🔥 風險閥門 (Risk Valve) 與增強分析功能
# ---------------------------------------------------------------------

def compute_risk_valve_signals(
    benchmark_df: pd.DataFrame,
    slope20_thresh: float = 0.0,
    slope60_thresh: float = 0.0,
    atr_win: int = 20,
    atr_ref_win: int = 60,
    atr_ratio_mult: float = 1.5,
    use_slopes: bool = True,
    slope_method: str = "polyfit",  # "polyfit" or "diff"
    atr_cmp: str = "gt",            # "gt" (>) or "lt" (<)
    combine_mode: str = "or"        # "or" / "and"
) -> pd.DataFrame:
    """
    計算風險閥門訊號

    Args:
        benchmark_df: 基準資料 (必須包含 close，可選 high/low)
        slope20_thresh: 20日斜率門檻
        slope60_thresh: 60日斜率門檻
        atr_win: ATR 計算窗口
        atr_ref_win: ATR 參考窗口
        atr_ratio_mult: ATR 比值倍數門檻
        use_slopes: 是否使用斜率條件
        slope_method: 斜率計算方法
        atr_cmp: ATR 比較符號

    Returns:
        DataFrame 包含 risk_trigger 欄位
    """
    df = benchmark_df.copy()

    # 取得收盤價
    if 'close' in df.columns:
        close = df['close']
    elif 'Close' in df.columns:
        close = df['Close']
    elif '收盤價' in df.columns:
        close = df['收盤價']
    else:
        close = df.iloc[:, 0]  # 回退到第一欄

    close = pd.to_numeric(close, errors='coerce')

    # 1. 斜率計算
    if use_slopes:
        if slope_method == "polyfit":
            def _slope(s, w):
                if len(s) < w:
                    return np.nan
                y = s.values
                x = np.arange(w)
                coef = np.polyfit(x, y, 1)
                return coef[0] / y[-1]  # 正規化斜率

            df['slope20'] = close.rolling(20).apply(lambda x: _slope(x, 20), raw=False)
            df['slope60'] = close.rolling(60).apply(lambda x: _slope(x, 60), raw=False)
        else:
            df['slope20'] = close.diff(20) / close.shift(20)
            df['slope60'] = close.diff(60) / close.shift(60)

        slope_cond = (df['slope20'] < slope20_thresh) | (df['slope60'] < slope60_thresh)
    else:
        slope_cond = pd.Series(False, index=df.index)

    # 2. ATR 比值計算
    if {'high', 'low', 'close'}.issubset(df.columns) or {'High', 'Low', 'Close'}.issubset(df.columns):
        # 有高低價，計算 True Range
        high = df.get('high', df.get('High', df.get('最高價')))
        low = df.get('low', df.get('Low', df.get('最低價')))

        high = pd.to_numeric(high, errors='coerce')
        low = pd.to_numeric(low, errors='coerce')

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        # 只有收盤價，降級處理
        tr = close.diff().abs()

    atr_curr = tr.rolling(atr_win).mean()
    atr_ref = tr.rolling(atr_ref_win).mean()

    # 避免除以 0
    atr_ratio = atr_curr / atr_ref.replace(0, np.nan)

    if atr_cmp == "gt":
        atr_cond = atr_ratio > atr_ratio_mult
    else:
        atr_cond = atr_ratio < atr_ratio_mult

    # 3. 綜合觸發
    if str(combine_mode).lower() == "and":
        trigger = slope_cond & atr_cond
    else:
        trigger = slope_cond | atr_cond

    df['atr_ratio'] = atr_ratio
    df['risk_trigger'] = trigger
    df['risk_trigger'] = df['risk_trigger'].fillna(False)

    return df[['risk_trigger', 'atr_ratio']]


def risk_valve_backtest(
    open_px: pd.Series,
    w: pd.Series,
    cost: CostParams,
    benchmark_df: pd.DataFrame,
    mode: str = "cap",          # "cap" / "adaptive_cap" / "ban_add"
    cap_level: float = 0.5,
    slope20_thresh: float = 0.0,
    slope60_thresh: float = 0.0,
    atr_win: int = 20,
    atr_ref_win: int = 60,
    atr_ratio_mult: float = 1.5,
    use_slopes: bool = True,
    slope_method: str = "polyfit",
    atr_cmp: str = "gt",
    combine_mode: str = "or"
) -> Dict:
    """
    風險閥門回測：比較原始策略 vs 套用風險控制後的差異

    Args:
        open_px: 開盤價序列
        w: 權重序列
        cost: 成本參數
        benchmark_df: 基準資料
        mode: 閥門模式
        cap_level: 上限值
        ... (其他風險閥門參數)

    Returns:
        Dict 包含 metrics, signals, daily_state_orig, daily_state_valve, ...
    """
    # 1. 原始回測
    if cost is None:
        cost = CostParams()

    eq_orig, tr_orig, ld_orig, _ = calculate_performance(open_px, w, cost)

    # 2. 計算風險訊號
    risk_df = compute_risk_valve_signals(
        benchmark_df, slope20_thresh, slope60_thresh,
        atr_win, atr_ref_win, atr_ratio_mult,
        use_slopes, slope_method, atr_cmp, combine_mode
    )

    # 對齊 index
    mask = risk_df['risk_trigger'].reindex(w.index).fillna(False).astype(bool)

    # 3. 調整權重
    w_valve = w.copy()

    if mode == "cap":
        # 觸發時，權重上限壓到 cap_level
        w_valve[mask] = np.minimum(w_valve[mask], cap_level)

    elif mode == "adaptive_cap":
        # 依 ATR 超標幅度動態降倉：
        # ratio 接近門檻時幾乎不壓；ratio 越高，越接近 cap_level。
        atr_ratio_series = pd.to_numeric(risk_df.get('atr_ratio'), errors='coerce').reindex(w.index)
        th = max(float(atr_ratio_mult), 1e-9)
        severity = (atr_ratio_series / th) - 1.0
        severity = severity.clip(lower=0.0, upper=1.0).fillna(0.0)
        cap_series = 1.0 - (1.0 - float(cap_level)) * severity
        cap_series = cap_series.clip(lower=float(cap_level), upper=1.0)
        active_mask = mask & cap_series.notna()
        w_valve[active_mask] = np.minimum(w_valve[active_mask], cap_series[active_mask])

    elif mode == "ban_add":
        # 禁止加碼：如果觸發，且原本 w 變大，強制 w 不變 (或只能變小)
        w_vals = w_valve.values
        m_vals = mask.values
        for i in range(1, len(w_vals)):
            if m_vals[i]:
                if w_vals[i] > w_vals[i-1]:
                    w_vals[i] = w_vals[i-1]
        w_valve = pd.Series(w_vals, index=w.index)

    # 4. 閥門回測
    eq_valve, tr_valve, ld_valve, _ = calculate_performance(open_px, w_valve, cost)

    # 5. 計算指標差異
    def _quick_stats(eq, tr):
        if len(eq) < 2:
            return {}
        ret = (eq.iloc[-1]/eq.iloc[0]) - 1
        dd = eq/eq.cummax() - 1
        mdd = dd.min()

        # 右尾計算
        sell_rets = _calculate_lifo_returns(tr)
        pos_rets = sell_rets[sell_rets > 0]
        right_tail_sum = pos_rets.sum() if len(pos_rets) > 0 else 0

        gross_profit = pos_rets.sum()
        gross_loss = abs(sell_rets[sell_rets<=0].sum())
        pf = gross_profit/gross_loss if gross_loss!=0 else 0

        return {"ret": ret, "mdd": mdd, "right_tail_sum": right_tail_sum, "pf": pf}

    s_orig = _quick_stats(eq_orig, tr_orig)
    s_valve = _quick_stats(eq_valve, tr_valve)

    metrics = {
        "pf_orig": s_orig.get("pf", 0),
        "pf_valve": s_valve.get("pf", 0),
        "mdd_orig": s_orig.get("mdd", 0),
        "mdd_valve": s_valve.get("mdd", 0),
        "right_tail_sum_orig": s_orig.get("right_tail_sum", 0),
        "right_tail_sum_valve": s_valve.get("right_tail_sum", 0),
        "right_tail_reduction": s_orig.get("right_tail_sum", 0) - s_valve.get("right_tail_sum", 0)
    }

    return {
        "metrics": metrics,
        "signals": risk_df,
        "daily_state_orig": ld_orig,
        "daily_state_valve": ld_valve,
        "trade_ledger_valve": tr_valve,
        "weights_valve": w_valve
    }


def _mdd_from_daily_equity(equity: pd.Series) -> float:
    """從權益曲線計算最大回撤"""
    if equity is None or len(equity) < 2:
        return 0.0
    dd = equity / equity.cummax() - 1
    return dd.min()


def _sell_returns_pct_from_ledger(trades: pd.DataFrame) -> pd.Series:
    """從交易記錄提取賣出交易的報酬率"""
    arr = _calculate_lifo_returns(trades)
    return pd.Series(arr).dropna()


# ---------------------------------------------------------------------
# 主程式入口
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--method", choices=["majority", "proportional"], default="majority")
    args = parser.parse_args()

    print("請使用 run_enhanced_ensemble.py 執行完整流程。")
