# sss_core/plotting.py
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
import numpy as np

def plot_weight_series(weight_series: pd.Series, title: str = "權重變化", 
                       figsize: Tuple[int, int] = (800, 400)) -> go.Figure:
    """繪製權重變化圖"""
    fig = go.Figure()
    
    # 權重線
    fig.add_trace(go.Scatter(
        x=weight_series.index,
        y=weight_series.values,
        mode='lines',
        name='權重',
        line=dict(color='blue', width=2)
    ))
    
    # 設定圖表
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="權重",
        height=figsize[1],
        width=figsize[0],
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def plot_equity_cash(equity_or_ds, cash_series=None,
                     title: str = "權益與現金變化", figsize=(800, 400),
                     debug_csv_path: str | None = None) -> go.Figure:
    """
    統一處理：不管傳入 DataFrame 還是 Series，都會經過 normalize_daily_state() 糾偏
    1) plot_equity_cash(daily_state_df, None)  # 推薦：自動標準化與糾偏
    2) plot_equity_cash(equity_series, cash_series)  # 舊簽名，現在也會糾偏
    
    Args:
        equity_or_ds: daily_state DataFrame 或 equity Series
        cash_series: cash Series（當 equity_or_ds 是 Series 時使用）
        title: 圖表標題
        figsize: 圖表尺寸 (width, height)
        debug_csv_path: 可選的 debug CSV 輸出路徑，用於比對兩邊資料
    """
    fig = go.Figure()

    # 統一轉成 DataFrame 再丟去 normalize_daily_state()
    if isinstance(equity_or_ds, pd.DataFrame):
        ds = equity_or_ds.copy()
        ds.columns = [str(c).lower() for c in ds.columns]
    else:
        eq = equity_or_ds if equity_or_ds is not None else pd.Series(dtype=float)
        ca = cash_series if cash_series is not None else pd.Series(dtype=float)
        ds = pd.DataFrame({"equity": eq, "cash": ca}).sort_index()

        # (加一道保險) 若只有 equity/cash 而且強負相關，先把 equity 當成 position_value
        try:
            if ds["equity"].notna().any() and ds["cash"].notna().any():
                corr = ds["equity"].corr(ds["cash"])
                if corr is not None and corr < -0.95:
                    ds["position_value"] = ds["equity"]
        except Exception:
            pass

    from .normalize import normalize_daily_state
    ds = normalize_daily_state(ds)

    # << 新增：若提供路徑就把實際用來畫圖的資料噴成 CSV >>
    if debug_csv_path and {"equity", "cash"}.issubset(ds.columns):
        try:
            ds[["equity", "cash"]].to_csv(debug_csv_path, encoding="utf-8-sig")
        except Exception:
            pass

    if not {"equity", "cash"}.issubset(ds.columns):
        fig.update_layout(title=title, height=figsize[1], width=figsize[0])
        return fig

    fig.add_trace(go.Scatter(x=ds.index, y=ds["equity"], name="Equity"))
    fig.add_trace(go.Scatter(x=ds.index, y=ds["cash"],   name="Cash"))
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="金額",
                      height=figsize[1], width=figsize[0], hovermode='x unified')
    return fig

def prepare_equity_cash_inputs(equity_or_ds, cash_series=None) -> pd.DataFrame:
    """
    回傳真正要畫的 equity/cash DataFrame（標準化/糾偏後），讓你能另存 CSV 比對。
    
    Args:
        equity_or_ds: daily_state DataFrame 或 equity Series
        cash_series: cash Series（當 equity_or_ds 是 Series 時使用）
        
    Returns:
        標準化後的 equity/cash DataFrame，可用於 CSV 輸出比對
    """
    if isinstance(equity_or_ds, pd.DataFrame):
        try:
            # 嘗試相對導入
            from .normalize import normalize_daily_state
        except ImportError:
            try:
                # 嘗試絕對導入
                from sss_core.normalize import normalize_daily_state
            except ImportError:
                # 如果都失敗，直接返回原始數據
                print("⚠️ 無法導入 normalize_daily_state，返回原始數據")
                cols = [c for c in ["equity", "cash"] if c in equity_or_ds.columns]
                if cols:
                    return equity_or_ds[cols].copy()
                else:
                    return equity_or_ds.copy()
        
        ds = normalize_daily_state(equity_or_ds.copy())
        cols = [c for c in ["equity", "cash"] if c in ds.columns]
        if cols:
            return ds[cols].copy()
        else:
            return ds.copy()
    else:
        eq = equity_or_ds if equity_or_ds is not None else pd.Series(dtype=float)
        ca = cash_series if cash_series is not None else pd.Series(dtype=float)
        df = pd.DataFrame({"equity": eq}).join(pd.DataFrame({"cash": ca}), how="outer")
        return df.sort_index()

def plot_trades_on_price(price_series: pd.Series, trades_df: pd.DataFrame,
                         title: str = "交易信號", figsize: Tuple[int, int] = (800, 400)) -> go.Figure:
    """在價格圖上標示交易信號"""
    fig = go.Figure()
    
    # 價格線
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode='lines',
        name='價格',
        line=dict(color='black', width=1)
    ))
    
    if not trades_df.empty and 'trade_date' in trades_df.columns and 'type' in trades_df.columns:
        # 買入信號
        buy_trades = trades_df[trades_df['type'].str.lower() == 'buy']
        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['trade_date'],
                y=buy_trades['price'] if 'price' in buy_trades.columns else price_series.loc[buy_trades['trade_date']],
                mode='markers',
                name='買入',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        # 賣出信號
        sell_trades = trades_df[trades_df['type'].str.lower() == 'sell']
        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['trade_date'],
                y=sell_trades['price'] if 'price' in sell_trades.columns else price_series.loc[sell_trades['trade_date']],
                mode='markers',
                name='賣出',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
    
    # 設定圖表
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="價格",
        height=figsize[1],
        width=figsize[0],
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def plot_performance_metrics(stats: dict, title: str = "績效指標", 
                           figsize: Tuple[int, int] = (600, 400)) -> go.Figure:
    """繪製績效指標圖"""
    # 選擇要顯示的指標
    key_metrics = ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio']
    display_names = ['總報酬率', '年化報酬率', '最大回撤', '夏普比率']
    
    values = []
    labels = []
    
    for metric, display_name in zip(key_metrics, display_names):
        if metric in stats:
            value = stats[metric]
            if isinstance(value, (int, float)):
                values.append(value)
                labels.append(display_name)
    
    if not values:
        # 如果沒有有效指標，創建空圖
        fig = go.Figure()
        fig.add_annotation(text="無可用績效指標", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=['green', 'blue', 'red', 'orange'])
        ])
    
    fig.update_layout(
        title=title,
        xaxis_title="指標",
        yaxis_title="數值",
        height=figsize[1],
        width=figsize[0],
        showlegend=False
    )
    
    return fig

def create_combined_dashboard(equity_series: pd.Series, trades_df: pd.DataFrame,
                             price_series: Optional[pd.Series] = None,
                             weight_series: Optional[pd.Series] = None,
                             stats: Optional[dict] = None,
                             title: str = "策略回測儀表板") -> go.Figure:
    """創建綜合儀表板"""
    # 創建子圖
    if price_series is not None and weight_series is not None:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('權益曲線', '價格與交易信號', '權重變化'),
            vertical_spacing=0.1
        )
        
        # 權益曲線
        fig.add_trace(
            go.Scatter(x=equity_series.index, y=equity_series.values, name='權益'),
            row=1, col=1
        )
        
        # 價格與交易信號
        if price_series is not None:
            fig.add_trace(
                go.Scatter(x=price_series.index, y=price_series.values, name='價格'),
                row=2, col=1
            )
        
        # 權重變化
        if weight_series is not None:
            fig.add_trace(
                go.Scatter(x=weight_series.index, y=weight_series.values, name='權重'),
                row=3, col=1
            )
        
        fig.update_layout(height=900, title_text=title)
        
    else:
        # 簡化版本
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('權益曲線', '交易信號'),
            vertical_spacing=0.1
        )
        
        # 權益曲線
        fig.add_trace(
            go.Scatter(x=equity_series.index, y=equity_series.values, name='權益'),
            row=1, col=1
        )
        
        # 交易信號
        if not trades_df.empty and 'trade_date' in trades_df.columns:
            buy_trades = trades_df[trades_df['type'].str.lower() == 'buy']
            sell_trades = trades_df[trades_df['type'].str.lower() == 'sell']
            
            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(x=buy_trades['trade_date'], y=[1]*len(buy_trades), 
                              mode='markers', name='買入', marker=dict(color='green', size=8)),
                    row=2, col=1
                )
            
            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(x=sell_trades['trade_date'], y=[0]*len(sell_trades), 
                              mode='markers', name='賣出', marker=dict(color='red', size=8)),
                    row=2, col=1
                )
        
        fig.update_layout(height=600, title_text=title)
    
    return fig

# === DEBUG helpers: dump what we actually plot ===
import os
from pathlib import Path

def _debug_dir() -> Path:
    """創建偵錯輸出目錄"""
    try:
        # 方法1：從當前檔案位置向上找專案根目錄
        current_file = Path(__file__)
        project_root = current_file.parent.parent  # sss_core -> 專案根目錄
        
        # 檢查是否有 sss_backtest_outputs 目錄
        debug_dir = project_root / "sss_backtest_outputs" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 偵錯目錄：{debug_dir}")
        return debug_dir
    except Exception as e:
        # 方法2：如果失敗，使用當前工作目錄
        try:
            import os
            current_dir = Path.cwd()
            debug_dir = current_dir / "sss_backtest_outputs" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"🔍 偵錯目錄（備用）：{debug_dir}")
            return debug_dir
        except Exception as e2:
            # 方法3：最後備用，使用臨時目錄
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "sss_debug"
            temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"🔍 偵錯目錄（臨時）：{temp_dir}")
            return temp_dir

def dump_equity_cash(tag: str, equity_or_ds, cash_series=None) -> Path:
    """
    取用 prepare_equity_cash_inputs() 的結果（即真正畫圖用到的資料）直接輸出 CSV。
    tag 例如: 'streamlit_backtest' / 'dash_backtest' / 'ensemble_majority' 等。
    回傳 CSV 路徑。
    """
    try:
        df = prepare_equity_cash_inputs(equity_or_ds, cash_series)
        if df is None or len(df) == 0:
            # 也把空的吐出，避免「以為沒呼叫」
            df = (pd.DataFrame({"equity": pd.Series(dtype=float), "cash": pd.Series(dtype=float)}))
        # 統一 index/欄位
        df = df.copy().sort_index()
        df.index.name = "date"
        # 簡單的健檢欄位
        if {"equity","cash"}.issubset(df.columns):
            df["position_value_implied"] = (pd.to_numeric(df["equity"], errors="coerce")
                                            - pd.to_numeric(df["cash"], errors="coerce"))
            df["check_sum_eq≈pos+cash"] = (df["equity"] - (df["position_value_implied"] + df["cash"])).abs()
        # 寫檔
        out = _debug_dir() / f"equity_cash_{tag}.csv"
        df.to_csv(out, float_format="%.8f")
        print(f"🔍 偵錯輸出成功：{out}")  # 加入成功訊息
        return out
    except Exception as e:
        # 錯誤處理：即使失敗也要輸出錯誤信息
        error_out = _debug_dir() / f"equity_cash_{tag}_ERROR.txt"
        with open(error_out, 'w', encoding='utf-8') as f:
            f.write(f"dump_equity_cash 執行失敗: {e}\n")
            f.write(f"輸入參數: equity_or_ds={type(equity_or_ds)}, cash_series={type(cash_series)}\n")
            if hasattr(equity_or_ds, 'shape'):
                f.write(f"equity_or_ds shape: {equity_or_ds.shape}\n")
            if hasattr(equity_or_ds, 'columns'):
                f.write(f"equity_or_ds columns: {list(equity_or_ds.columns)}\n")
        print(f"❌ 偵錯輸出失敗：{error_out}")  # 加入錯誤訊息
        return error_out

def dump_timeseries(tag: str, **series_dict) -> Path:
    """
    任意命名的多條 Series 一起吐出（例如 weight_curve, price, cash_series…）
    用於對齊檢查。
    """
    try:
        out = _debug_dir() / f"series_{tag}.csv"
        # 將所有 series outer-join 在一起
        df = pd.DataFrame()
        for k, s in series_dict.items():
            if s is None:
                continue
            ss = s.copy()
            if not isinstance(ss, pd.Series):
                try:
                    ss = pd.Series(ss)
                except Exception:
                    continue
            if not isinstance(ss.index, pd.DatetimeIndex):
                ss.index = pd.to_datetime(ss.index, errors="coerce")
            df = df.join(ss.rename(k), how="outer") if len(df) else ss.rename(k).to_frame()
        df = df.sort_index()
        df.index.name = "date"
        df.to_csv(out, float_format="%.8f")
        print(f"🔍 時間序列偵錯輸出成功：{out}")  # 加入成功訊息
        return out
    except Exception as e:
        # 錯誤處理：即使失敗也要輸出錯誤信息
        error_out = _debug_dir() / f"series_{tag}_ERROR.txt"
        with open(error_out, 'w', encoding='utf-8') as f:
            f.write(f"dump_timeseries 執行失敗: {e}\n")
            f.write(f"輸入參數: {series_dict}\n")
        print(f"❌ 時間序列偵錯輸出失敗：{error_out}")  # 加入錯誤訊息
        return error_out

# === 測試偵錯功能 ===
def test_debug_functions():
    """測試偵錯功能是否正常工作"""
    try:
        print("🧪 測試偵錯功能...")
        
        # 測試目錄創建
        debug_dir = _debug_dir()
        print(f"✅ 偵錯目錄創建成功：{debug_dir}")
        
        # 測試 dump_equity_cash
        test_df = pd.DataFrame({
            'equity': [100, 101, 102],
            'cash': [20, 21, 22]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        result = dump_equity_cash("test", test_df)
        print(f"✅ dump_equity_cash 測試成功：{result}")
        
        # 測試 dump_timeseries
        test_series = pd.Series([0.5, 0.6, 0.7], index=pd.date_range('2024-01-01', periods=3))
        result2 = dump_timeseries("test", weight=test_series)
        print(f"✅ dump_timeseries 測試成功：{result2}")
        
        print("🎉 所有偵錯功能測試通過！")
        return True
    except Exception as e:
        print(f"❌ 偵錯功能測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False

# 如果直接執行此檔案，則運行測試
if __name__ == "__main__":
    test_debug_functions()
