"""sss_core.plotting_unified

統一儀表板繪圖：
1. 股價與買賣點
2. 權益與現金
3. 持倉權重
4. LIFO 單筆報酬
5. (Ensemble 時) 投票線
6. 回撤 + 相對淨值比
7. 60 日滾動超額報酬
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def calculate_lifo_returns(trades_df: pd.DataFrame) -> pd.DataFrame:
    """依交易序列計算 LIFO 單筆賣出報酬。"""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    qty_col = None
    for c in ["shares", "weight_change", "delta_units"]:
        if c in df.columns:
            qty_col = c
            break

    if not qty_col or "price" not in df.columns or "type" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values("trade_date")
    inventory: list[dict[str, float]] = []
    returns: list[float] = []

    for _, row in df.iterrows():
        try:
            t_type = str(row["type"]).lower()
            price = float(row["price"])
            qty = abs(float(row[qty_col]))

            if qty == 0:
                returns.append(np.nan)
                continue

            if any(x in t_type for x in ["buy", "add", "long"]):
                inventory.append({"price": price, "qty": qty})
                returns.append(np.nan)
            elif any(x in t_type for x in ["sell", "exit"]):
                rem_qty = qty
                total_cost = 0.0
                matched_qty = 0.0

                while rem_qty > 0 and inventory:
                    last = inventory[-1]
                    if last["qty"] <= rem_qty:
                        total_cost += last["qty"] * last["price"]
                        matched_qty += last["qty"]
                        rem_qty -= last["qty"]
                        inventory.pop()
                    else:
                        total_cost += rem_qty * last["price"]
                        matched_qty += rem_qty
                        inventory[-1]["qty"] -= rem_qty
                        rem_qty = 0.0

                if matched_qty > 0:
                    avg_cost = total_cost / matched_qty
                    returns.append((price - avg_cost) / avg_cost)
                else:
                    returns.append(0.0)
            else:
                returns.append(np.nan)
        except Exception:
            returns.append(np.nan)

    df["lifo_return"] = returns
    return df


def create_unified_dashboard(
    df_raw: pd.DataFrame,
    daily_state: pd.DataFrame,
    trade_df: pd.DataFrame,
    ticker: str,
    theme: str = "dark",
    votes_series: Optional[pd.Series] = None,
    votes_threshold: Optional[float] = None,
    indicator_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """建立統一儀表板（X 軸同步）。"""

    labels = {
        "title_price": "股價與買賣點",
        "title_indicator": "每日 SMAA 與閥值",
        "title_eq": "權益與現金",
        "title_weight": "持倉權重(%)",
        "title_lifo": "LIFO 交易報酬",
        "title_votes": "Ensemble 投票",
        "title_dd_nav": "回撤與相對淨值比（DD / NAV）",
        "title_rolling": "30/60日滾動超額報酬",
        "close": "收盤價",
        "buy": "買點",
        "sell": "賣點",
        "forced_sell": "強制賣出",
        "smaa": "SMAA",
        "base": "基準線",
        "buy_threshold": "買入閥值",
        "sell_threshold": "賣出閥值",
        "prom_threshold": "Prominence閥值",
        "equity": "權益",
        "cash": "現金",
        "weight": "持倉權重(%)",
        "lifo_ret": "LIFO 報酬",
        "votes": "多頭票數",
        "vote_threshold": "門檻票數",
        "bh_dd": "BH回撤",
        "strat_dd": "策略回撤",
        "rel_nav": "相對淨值比（策略/BH）",
        "rel_nav_base": "相對淨值基準（1.0）",
        "roll30": "30日滾動超額",
        "roll60": "60日滾動超額",
        "axis_price": "價格",
        "axis_indicator": "SMAA/閥值",
        "axis_asset": "資產",
        "axis_weight": "持倉權重(%)",
        "axis_ret": "報酬率",
        "axis_votes": "票數",
        "axis_dd": "回撤",
        "axis_nav": "相對淨值比",
        "axis_roll": "30/60日滾動超額報酬",
    }

    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # 對齊與清洗時間索引
    df_raw_aligned = df_raw.copy() if df_raw is not None else pd.DataFrame()
    if not df_raw_aligned.empty:
        df_raw_aligned.index = pd.to_datetime(df_raw_aligned.index, errors="coerce")
        df_raw_aligned = df_raw_aligned[df_raw_aligned.index.notna()].sort_index()

    ds = daily_state.copy() if daily_state is not None else pd.DataFrame()
    if not ds.empty:
        ds.index = pd.to_datetime(ds.index, errors="coerce")
        ds = ds[ds.index.notna()].sort_index()
        if not df_raw_aligned.empty:
            df_raw_aligned = df_raw_aligned.loc[df_raw_aligned.index >= ds.index.min()].copy()

    indicator = indicator_df.copy() if indicator_df is not None else pd.DataFrame()
    if not indicator.empty:
        indicator.index = pd.to_datetime(indicator.index, errors="coerce")
        indicator = indicator[indicator.index.notna()].sort_index()
        if not ds.empty:
            indicator = indicator.loc[indicator.index >= ds.index.min()].copy()
        if not df_raw_aligned.empty:
            indicator = indicator.loc[indicator.index >= df_raw_aligned.index.min()].copy()
    has_indicator = (not indicator.empty) and ("smaa" in indicator.columns)

    close_col = _pick_col(df_raw_aligned, ["close", "Close", "收盤價"])
    has_votes = votes_series is not None and len(votes_series) > 0

    row_price = 1
    row_indicator = 2 if has_indicator else None
    row_eq = 3 if has_indicator else 2
    row_weight = row_eq + 1
    row_lifo = row_weight + 1
    if has_votes:
        row_votes = row_lifo + 1
        row_dd_nav = row_votes + 1
        row_rolling = row_dd_nav + 1
    else:
        row_votes = None
        row_dd_nav = row_lifo + 1
        row_rolling = row_dd_nav + 1

    total_rows = row_rolling
    subplot_titles: list[str] = [f"{ticker} {labels['title_price']}"]
    if has_indicator:
        subplot_titles.append(labels["title_indicator"])
    subplot_titles.extend(
        [
            labels["title_eq"],
            labels["title_weight"],
            labels["title_lifo"],
        ]
    )
    if has_votes:
        subplot_titles.append(labels["title_votes"])
    subplot_titles.extend([labels["title_dd_nav"], labels["title_rolling"]])

    if has_indicator and has_votes:
        row_heights = [0.18, 0.14, 0.14, 0.11, 0.13, 0.08, 0.13, 0.09]
    elif has_indicator and not has_votes:
        row_heights = [0.20, 0.15, 0.16, 0.12, 0.15, 0.14, 0.08]
    elif (not has_indicator) and has_votes:
        row_heights = [0.23, 0.16, 0.12, 0.15, 0.09, 0.15, 0.10]
    else:
        row_heights = [0.27, 0.19, 0.14, 0.16, 0.15, 0.09]

    specs = [[{"secondary_y": False}] for _ in range(total_rows)]
    specs[row_dd_nav - 1] = [{"secondary_y": True}]

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=tuple(subplot_titles),
        row_heights=row_heights,
        specs=specs,
    )

    row_legend_map: dict[int, str] = {}

    def _row_legend_id(row: int) -> str:
        if row in row_legend_map:
            return row_legend_map[row]
        legend_id = "legend" if not row_legend_map else f"legend{len(row_legend_map) + 1}"
        row_legend_map[row] = legend_id
        return legend_id

    def _add_trace(trace: go.BaseTraceType, row: int, *, secondary_y: bool = False, use_row_legend: bool = True) -> None:
        if use_row_legend:
            trace.legend = _row_legend_id(row)
        fig.add_trace(trace, row=row, col=1, secondary_y=secondary_y)

    # Row 1: 股價與買賣點
    if not df_raw_aligned.empty and close_col:
        _add_trace(
            go.Scatter(
                x=df_raw_aligned.index,
                y=pd.to_numeric(df_raw_aligned[close_col], errors="coerce"),
                name=labels["close"],
                line=dict(color="#636EFA", width=1.5),
            ),
            row=1,
        )

    if trade_df is not None and not trade_df.empty:
        tdf = trade_df.copy()
        tdf.columns = [str(c).lower() for c in tdf.columns]
        if "trade_date" in tdf.columns:
            tdf["trade_date"] = pd.to_datetime(tdf["trade_date"], errors="coerce")
        type_col = "type" if "type" in tdf.columns else ("action" if "action" in tdf.columns else None)
        if type_col and {"trade_date", "price"}.issubset(tdf.columns):
            buys = tdf[tdf[type_col].astype(str).str.contains("buy|add", case=False, na=False)]
            sells = tdf[tdf[type_col].astype(str).str.contains("sell", case=False, na=False)]
            forced = sells[sells[type_col].astype(str).str.contains("forced", case=False, na=False)]
            normal_sells = sells[~sells[type_col].astype(str).str.contains("forced", case=False, na=False)]

            if not buys.empty:
                _add_trace(
                    go.Scatter(
                        x=buys["trade_date"],
                        y=pd.to_numeric(buys["price"], errors="coerce"),
                        mode="markers",
                        name=labels["buy"],
                        marker=dict(symbol="triangle-up", size=9, color="#00CC96"),
                    ),
                    row=1,
                )
            if not normal_sells.empty:
                _add_trace(
                    go.Scatter(
                        x=normal_sells["trade_date"],
                        y=pd.to_numeric(normal_sells["price"], errors="coerce"),
                        mode="markers",
                        name=labels["sell"],
                        marker=dict(symbol="triangle-down", size=9, color="#EF553B"),
                    ),
                    row=1,
                )
            if not forced.empty:
                _add_trace(
                    go.Scatter(
                        x=forced["trade_date"],
                        y=pd.to_numeric(forced["price"], errors="coerce"),
                        mode="markers",
                        name=labels["forced_sell"],
                        marker=dict(symbol="square", size=7, color="#888888"),
                    ),
                    row=1,
                )

    if has_indicator and row_indicator is not None:
        smaa_series = pd.to_numeric(indicator.get("smaa"), errors="coerce")
        _add_trace(
            go.Scatter(
                x=indicator.index,
                y=smaa_series,
                name=labels["smaa"],
                line=dict(color="#4dabf7", width=1.6),
            ),
            row=row_indicator,
        )
        if "base" in indicator.columns:
            _add_trace(
                go.Scatter(
                    x=indicator.index,
                    y=pd.to_numeric(indicator["base"], errors="coerce"),
                    name=labels["base"],
                    line=dict(color="#f59f00", width=1.2),
                ),
                row=row_indicator,
            )
        if "buy_threshold" in indicator.columns:
            _add_trace(
                go.Scatter(
                    x=indicator.index,
                    y=pd.to_numeric(indicator["buy_threshold"], errors="coerce"),
                    name=labels["buy_threshold"],
                    line=dict(color="#40c057", width=1.2, dash="dot"),
                ),
                row=row_indicator,
            )
        if "sell_threshold" in indicator.columns:
            _add_trace(
                go.Scatter(
                    x=indicator.index,
                    y=pd.to_numeric(indicator["sell_threshold"], errors="coerce"),
                    name=labels["sell_threshold"],
                    line=dict(color="#fa5252", width=1.2, dash="dash"),
                ),
                row=row_indicator,
            )
        if "prom_threshold" in indicator.columns:
            _add_trace(
                go.Scatter(
                    x=indicator.index,
                    y=pd.to_numeric(indicator["prom_threshold"], errors="coerce"),
                    name=labels["prom_threshold"],
                    line=dict(color="#FFD166", width=1.2, dash="dashdot"),
                ),
                row=row_indicator,
            )

    # Row 權益與現金
    if not ds.empty:
        if "equity" in ds.columns:
            _add_trace(
                go.Scatter(
                    x=ds.index,
                    y=pd.to_numeric(ds["equity"], errors="coerce"),
                    name=labels["equity"],
                    line=dict(color="#00CC96", width=2),
                ),
                row=row_eq,
            )
        if "cash" in ds.columns:
            _add_trace(
                go.Scatter(
                    x=ds.index,
                    y=pd.to_numeric(ds["cash"], errors="coerce"),
                    name=labels["cash"],
                    line=dict(color="#FFA15A", width=1.5, dash="dot"),
                ),
                row=row_eq,
            )

    # Row 持倉權重
    if not ds.empty and "w" in ds.columns:
        _add_trace(
            go.Scatter(
                x=ds.index,
                y=pd.to_numeric(ds["w"], errors="coerce") * 100,
                name=labels["weight"],
                fill="tozeroy",
                line=dict(color="#AB63FA", width=1.5),
            ),
            row=row_weight,
        )

    # Row LIFO 報酬
    lifo_y_range = None
    if trade_df is not None and not trade_df.empty:
        lifo_df = calculate_lifo_returns(trade_df)
        valid_lifo = lifo_df[lifo_df["lifo_return"].notna()] if "lifo_return" in lifo_df.columns else pd.DataFrame()
        if not valid_lifo.empty:
            colors = ["#00CC96" if x > 0 else "#EF553B" for x in valid_lifo["lifo_return"]]
            _add_trace(
                go.Bar(
                    x=valid_lifo["trade_date"],
                    y=valid_lifo["lifo_return"],
                    name=labels["lifo_ret"],
                    marker=dict(color=colors, line=dict(width=0)),
                    opacity=0.95,
                    hovertemplate="日期: %{x}<br>報酬: %{y:.2%}<extra></extra>",
                ),
                row=row_lifo,
            )
            fig.add_hline(y=0, line_color="gray", line_width=1, row=row_lifo, col=1)

            returns_array = valid_lifo["lifo_return"].to_numpy(dtype=float)
            returns_array = returns_array[np.isfinite(returns_array)]
            if returns_array.size:
                p95 = np.percentile(np.abs(returns_array), 95)
                if p95 > 0:
                    # 下限放寬到 1%，避免小幅策略被壓扁
                    y_limit = max(0.01, min(p95 * 1.15, 0.5))
                    lifo_y_range = [-y_limit, y_limit]

    # 共同資料：DD / NAV / 30/60日滾動超額
    strat_dd = pd.Series(dtype=float)
    bench_dd = pd.Series(dtype=float)
    relative_nav = pd.Series(dtype=float)
    rolling_excess_30 = pd.Series(dtype=float)
    rolling_excess_60 = pd.Series(dtype=float)

    if not ds.empty and "equity" in ds.columns and not df_raw_aligned.empty and close_col:
        common_idx = pd.DatetimeIndex(ds.index.intersection(df_raw_aligned.index)).sort_values()
        if len(common_idx) >= 2:
            panel = pd.DataFrame(index=common_idx)
            panel["equity"] = pd.to_numeric(ds.loc[common_idx, "equity"], errors="coerce")
            panel["bh"] = pd.to_numeric(df_raw_aligned.loc[common_idx, close_col], errors="coerce")
            panel = panel.replace([np.inf, -np.inf], np.nan).dropna()

            if len(panel) >= 2:
                strat_equity = panel["equity"]
                bench_close = panel["bh"]

                strat_dd = strat_equity / strat_equity.cummax() - 1.0
                bench_dd = bench_close / bench_close.cummax() - 1.0

                strat_ret = strat_equity.pct_change()
                bh_ret = bench_close.pct_change()
                strat_30 = (1.0 + strat_ret).rolling(window=30, min_periods=30).apply(np.prod, raw=True) - 1.0
                bh_30 = (1.0 + bh_ret).rolling(window=30, min_periods=30).apply(np.prod, raw=True) - 1.0
                strat_60 = (1.0 + strat_ret).rolling(window=60, min_periods=60).apply(np.prod, raw=True) - 1.0
                bh_60 = (1.0 + bh_ret).rolling(window=60, min_periods=60).apply(np.prod, raw=True) - 1.0
                rolling_excess_30 = strat_30 - bh_30
                rolling_excess_60 = strat_60 - bh_60
                # 不再額外平滑，避免增加滯後。

                if bench_close.iloc[0] != 0:
                    relative_nav = (strat_equity / strat_equity.iloc[0]) / (bench_close / bench_close.iloc[0])

    # Row 5 (有投票時): 投票
    if has_votes and row_votes is not None:
        try:
            vs = votes_series.copy()
            if not isinstance(vs.index, pd.DatetimeIndex):
                vs.index = pd.to_datetime(vs.index, errors="coerce")
            vs = vs[vs.index.notna()].sort_index()
            if not ds.empty:
                vs = vs.reindex(ds.index)
        except Exception:
            vs = votes_series

        _add_trace(
            go.Scatter(
                x=vs.index,
                y=vs.values,
                name=labels["votes"],
                line=dict(color="#28a745", width=1.8),
            ),
            row=row_votes,
        )
        if votes_threshold is not None:
            _add_trace(
                go.Scatter(
                    x=vs.index,
                    y=[votes_threshold] * len(vs),
                    name=labels["vote_threshold"],
                    line=dict(color="#ff6b6b", dash="dash", width=1.2),
                ),
                row=row_votes,
            )

    # Row DD/NAV
    if not bench_dd.empty:
        _add_trace(
            go.Scatter(
                x=bench_dd.index,
                y=bench_dd,
                name=f"{labels['bh_dd']} (MDD: {bench_dd.min():.1%})",
                line=dict(color="rgba(140, 140, 140, 0.60)", width=1.0, dash="dot"),
                hovertemplate="BH回撤: %{y:.2%}<extra></extra>",
            ),
            row=row_dd_nav,
            secondary_y=False,
        )
        _add_trace(
            go.Scatter(
                x=strat_dd.index,
                y=strat_dd,
                name=f"{labels['strat_dd']} (MDD: {strat_dd.min():.1%})",
                line=dict(color="rgba(239, 85, 59, 0.70)", width=1.5),
                hovertemplate="策略回撤: %{y:.2%}<extra></extra>",
            ),
            row=row_dd_nav,
            secondary_y=False,
        )
        _add_trace(
            go.Scatter(
                x=relative_nav.index,
                y=relative_nav,
                name=labels["rel_nav"],
                line=dict(color="#FFD166", width=2.0),
                hovertemplate="相對淨值比: %{y:.3f}<extra></extra>",
            ),
            row=row_dd_nav,
            secondary_y=True,
        )
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="rgba(128, 128, 128, 0.35)",
            line_width=1,
            row=row_dd_nav,
            col=1,
        )
        _add_trace(
            go.Scatter(
                x=relative_nav.index,
                y=np.ones(len(relative_nav), dtype=float),
                name=labels["rel_nav_base"],
                line=dict(color="rgba(255, 209, 102, 0.6)", width=1.0, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_dd_nav,
            secondary_y=True,
            use_row_legend=False,
        )

    # Row 滾動超額（30/60 日）
    if not rolling_excess_30.empty:
        _add_trace(
            go.Scatter(
                x=rolling_excess_30.index,
                y=rolling_excess_30,
                name=labels["roll30"],
                line=dict(color="rgba(255, 0, 0, 0.92)", width=1.8, dash="dot", shape="spline", smoothing=0.5),
                hovertemplate="30日超額: %{y:.2%}<extra></extra>",
            ),
            row=row_rolling,
        )
    if not rolling_excess_60.empty:
        _add_trace(
            go.Scatter(
                x=rolling_excess_60.index,
                y=rolling_excess_60,
                name=labels["roll60"],
                line=dict(color="rgba(0, 166, 139, 0.98)", width=2.4, dash="dot", shape="spline", smoothing=0.5),
                hovertemplate="60日超額: %{y:.2%}<extra></extra>",
            ),
            row=row_rolling,
        )
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(128, 128, 128, 0.35)",
        line_width=1,
        row=row_rolling,
        col=1,
    )

    theme_cfg = {
        "dark": {"template": "plotly_dark", "paper_bgcolor": "#1a1a1a", "plot_bgcolor": "#0e0e0e", "font_color": "#e0e0e0"},
        "light": {"template": "plotly_white", "paper_bgcolor": "#ffffff", "plot_bgcolor": "#f8f9fa", "font_color": "#2c3e50"},
    }
    cfg = theme_cfg.get(theme, theme_cfg["dark"])
    if has_indicator:
        height = 1900 if has_votes else 1680
    else:
        height = 1700 if has_votes else 1500

    fig.update_layout(
        height=height,
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"], size=11),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=80, b=40),
        bargap=0.1,
    )

    fig.update_xaxes(hoverformat="%Y-%m-%d")
    fig.update_yaxes(title_text=labels["axis_price"], row=1, col=1)
    if has_indicator and row_indicator is not None:
        fig.update_yaxes(title_text=labels["axis_indicator"], row=row_indicator, col=1)
    fig.update_yaxes(title_text=labels["axis_asset"], row=row_eq, col=1)
    fig.update_yaxes(title_text=labels["axis_weight"], range=[0, 110], row=row_weight, col=1)

    if lifo_y_range is not None:
        fig.update_yaxes(
            title_text=labels["axis_ret"],
            tickformat=".1%",
            range=lifo_y_range,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            row=row_lifo,
            col=1,
        )
    else:
        fig.update_yaxes(
            title_text=labels["axis_ret"],
            tickformat=".1%",
            range=[-0.1, 0.1],
            row=row_lifo,
            col=1,
        )

    if has_votes and row_votes is not None:
        fig.update_yaxes(title_text=labels["axis_votes"], row=row_votes, col=1)

    fig.update_yaxes(
        title_text=labels["axis_dd"],
        tickformat=".1%",
        zeroline=True,
        zerolinecolor="gray",
        zerolinewidth=1,
        row=row_dd_nav,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=labels["axis_nav"],
        tickformat=".2f",
        row=row_dd_nav,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text=labels["axis_roll"],
        tickformat=".1%",
        row=row_rolling,
        col=1,
    )

    legend_bg = "rgba(255,255,255,0.18)" if theme == "light" else "rgba(10,10,10,0.18)"
    legend_updates = {}
    for row in sorted(row_legend_map.keys()):
        subplot = fig.get_subplot(row, 1)
        if subplot is None or subplot.yaxis is None or subplot.yaxis.domain is None:
            continue
        y_top = float(subplot.yaxis.domain[1])
        legend_cfg = dict(
            orientation="h",
            yanchor="top",
            y=max(y_top - 0.004, 0.0),
            xanchor="left",
            x=0.0,
            bgcolor=legend_bg,
            borderwidth=0,
            font=dict(color=cfg["font_color"], size=10),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        )
        lid = row_legend_map[row]
        legend_updates["legend" if lid == "legend" else lid] = legend_cfg

    if legend_updates:
        fig.update_layout(**legend_updates)

    return fig


def plot_drawdown_comparison(
    equity_series: pd.Series,
    price_series: pd.Series,
    title: str = "策略 vs B&H 回撤比較",
    theme: str = "dark",
) -> go.Figure:
    """繪製策略與 B&H 回撤比較圖。"""
    if equity_series is None or equity_series.empty or price_series is None or price_series.empty:
        return go.Figure()

    common_idx = equity_series.index.intersection(price_series.index)
    if len(common_idx) < 2:
        logger.warning("Drawdown 比較圖：時間軸對齊失敗")
        return go.Figure()

    strat = equity_series.loc[common_idx]
    bench = price_series.loc[common_idx]

    strat_dd = strat / strat.cummax() - 1.0
    bench_dd = bench / bench.cummax() - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bench_dd.index,
            y=bench_dd,
            name=f"B&H (MDD: {bench_dd.min():.1%})",
            line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(128, 128, 128, 0.2)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=strat_dd.index,
            y=strat_dd,
            name=f"策略 (MDD: {strat_dd.min():.1%})",
            line=dict(color="#EF553B", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(239, 85, 59, 0.1)",
        )
    )
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(128, 128, 128, 0.3)", line_width=1)

    cfg = {
        "dark": {"template": "plotly_dark", "paper_bgcolor": "#1a1a1a", "plot_bgcolor": "#0e0e0e", "font_color": "#e0e0e0"},
        "light": {"template": "plotly_white", "paper_bgcolor": "#ffffff", "plot_bgcolor": "#f8f9fa", "font_color": "#2c3e50"},
    }.get(theme, {"template": "plotly_dark", "paper_bgcolor": "#1a1a1a", "plot_bgcolor": "#0e0e0e", "font_color": "#e0e0e0"})

    fig.update_layout(
        title=title,
        height=400,
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"], size=11),
        hovermode="x unified",
        xaxis=dict(title="日期"),
        yaxis=dict(title="回撤比例", tickformat=".1%"),
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(hoverformat="%Y-%m-%d")
    return fig
