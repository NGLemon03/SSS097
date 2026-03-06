# record.py
# 交易記錄互動式分析（Plotly 版）
# - 預設顯示實際金額/股數（record1 功能）
# - --anonymous 旗標：匿名模式（權重歸一化，不顯示股數/金額）
# - 全程使用 Plotly，可在圖中 hover 查詢資料
# - 輸出：瀏覽器互動圖（fig.show()）

import os
import re
import sys
import math
import argparse
import traceback
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# =========================
# 讀取 / 清洗
# =========================
def read_csv_robust(path: str) -> pd.DataFrame:
    last_err = None
    for enc in [None, "utf-8-sig", "cp950"]:
        try:
            return pd.read_csv(path) if enc is None else pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def parse_float_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def load_transactions(csv_path: str) -> pd.DataFrame:
    df = read_csv_robust(csv_path)
    needed = ["日期", "類型", "證券", "股數", "報價", "金額"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少欄位：{missing}\n現有欄位：{df.columns.tolist()}")

    df["日期"] = pd.to_datetime(df["日期"])
    df["股數_num"] = parse_float_series(df["股數"])
    df["報價_num"] = parse_float_series(df["報價"])
    df["金額_num"] = parse_float_series(df["金額"])
    # 費用（可選）
    if "費用" in df.columns:
        df["費用_num"] = parse_float_series(df["費用"])
    else:
        df["費用_num"] = 0.0
    return df


# =========================
# yfinance 取最新收盤
# =========================
def try_get_latest_close_yf(ticker: str) -> Optional[float]:
    try:
        import yfinance as yf  # type: ignore
        hist = yf.download(ticker, progress=False, auto_adjust=False)
        if hist is None or len(hist) == 0:
            return None
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            return float(close.iloc[-1, 0])
        return float(close.iloc[-1])
    except Exception:
        return None


# =========================
# FIFO 配對
# =========================
def build_fifo_df(df_stock: pd.DataFrame, latest_price: float, anonymous: bool = False) -> pd.DataFrame:
    total_invested = float(df_stock.loc[df_stock["類型"] == "買入", "金額_num"].sum())
    if total_invested <= 0:
        total_invested = 1.0

    inventory: List[Dict] = []
    records: List[Dict] = []

    for _, row in df_stock.iterrows():
        t = row["類型"]
        if t == "買入":
            inventory.append({
                "date": row["日期"],
                "price": float(row["報價_num"]),
                "qty": float(row["股數_num"]),
                "amt": float(row["金額_num"]),
            })
        elif t == "賣出":
            q_left = float(row["股數_num"])
            sell_p = float(row["報價_num"])
            sell_date = row["日期"]
            sell_fee = float(row.get("費用_num", 0) or 0)

            while q_left > 1e-9 and inventory:
                node = inventory[0]
                take = min(q_left, node["qty"])
                cost_per_share = node["price"]
                roi = (sell_p / cost_per_share - 1.0) * 100.0 if cost_per_share > 0 else np.nan
                post_rise = (latest_price / sell_p - 1.0) * 100.0 if sell_p > 0 else np.nan
                weight_raw = (take * cost_per_share) / total_invested * 100.0

                # 少賺金額：正=賣後還在漲（機會成本），負=賣在高點（賣對了）
                missed_gain = take * (latest_price - sell_p)
                # 若持有到今的總損益（買入到最新價）
                hold_to_now = take * (latest_price - cost_per_share)

                records.append({
                    "買入日期": node["date"],
                    "賣出日期": sell_date,
                    "持有天數": int((sell_date - node["date"]).days),
                    "買入價格": round(cost_per_share, 3),
                    "賣出價格": round(sell_p, 3),
                    "實現ROI%": round(roi, 2) if np.isfinite(roi) else np.nan,
                    "賣出後續漲幅%": round(post_rise, 2) if np.isfinite(post_rise) else np.nan,
                    "資金權重%_raw": round(weight_raw, 3),
                    # record1 欄位（完整模式）
                    "股數": round(take, 0),
                    "買入金額": round(take * cost_per_share, 0),
                    "賣出金額": round(take * sell_p, 0),
                    "實現損益": round(take * (sell_p - cost_per_share), 0),
                    "少賺金額": round(missed_gain, 0),        # 賣後機會成本（正=後悔）
                    "若持有到今損益": round(hold_to_now, 0),  # 若完全不動的理想損益
                })
                node["qty"] -= take
                q_left -= take
                if node["qty"] <= 1e-9:
                    inventory.pop(0)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # 匿名化：相對權重（最大=100）
    w = df["資金權重%_raw"].values.astype(float)
    w = np.clip(w, 0, None)
    df["資金權重%"] = (w / np.max(w)) * 100.0 if np.max(w) > 0 else 1.0

    if anonymous:
        df = df.drop(columns=["股數", "買入金額", "賣出金額", "實現損益"])

    return df


# =========================
# 持倉中部位
# =========================
def get_open_positions(df_stock: pd.DataFrame) -> pd.DataFrame:
    inventory: List[Dict] = []
    for _, row in df_stock.iterrows():
        t = row["類型"]
        if t == "買入":
            inventory.append({
                "買入日期": row["日期"],
                "買入價格": float(row["報價_num"]),
                "股數": float(row["股數_num"]),
                "買入金額": float(row["金額_num"]),
            })
        elif t == "賣出":
            q_left = float(row["股數_num"])
            while q_left > 1e-9 and inventory:
                node = inventory[0]
                take = min(q_left, node["股數"])
                node["股數"] -= take
                q_left -= take
                if node["股數"] <= 1e-9:
                    inventory.pop(0)
    return pd.DataFrame(inventory)


# =========================
# 清倉→買回偵測
# =========================
def detect_exit_reentry(df_stock: pd.DataFrame) -> pd.DataFrame:
    t = df_stock.sort_values("日期").copy()
    t["signed"] = np.where(t["類型"] == "買入", t["股數_num"], -t["股數_num"])
    t["pos"] = t["signed"].cumsum().round(6)

    exit_rows = t[(t["類型"] == "賣出") & (t["pos"] == 0)]
    events = []
    for _, r in exit_rows.iterrows():
        exit_date = r["日期"]
        exit_price = float(r["報價_num"])
        nxt = t[(t["日期"] > exit_date) & (t["類型"] == "買入")].head(1)
        if nxt.empty:
            continue
        re_date = nxt.iloc[0]["日期"]
        re_price = float(nxt.iloc[0]["報價_num"])
        gap = int((re_date - exit_date).days)
        pct = (re_price / exit_price - 1.0) * 100.0 if exit_price > 0 else np.nan
        events.append({
            "清倉日": exit_date.strftime("%Y-%m-%d"),
            "清倉價": round(exit_price, 2),
            "買回日": re_date.strftime("%Y-%m-%d"),
            "買回價": round(re_price, 2),
            "空手天數": gap,
            "踏空追價%": round(pct, 2) if np.isfinite(pct) else None,
            "賣飛分數": round((pct if np.isfinite(pct) else 0.0) * gap, 1),
        })
    return pd.DataFrame(events)


# =========================
# 月度績效
# =========================
def build_monthly_pnl(fifo_df: pd.DataFrame) -> pd.DataFrame:
    if fifo_df.empty or "實現損益" not in fifo_df.columns:
        return pd.DataFrame()
    df = fifo_df.copy()
    df["月份"] = pd.to_datetime(df["賣出日期"]).dt.to_period("M")
    monthly = df.groupby("月份").agg(
        月度損益=("實現損益", "sum"),
        交易次數=("實現ROI%", "count"),
        平均ROI=("實現ROI%", "mean"),
        勝率=("實現ROI%", lambda x: (x > 0).mean() * 100),
    ).reset_index()
    monthly["月份"] = monthly["月份"].astype(str)
    return monthly


# =========================
# 繪圖：Plotly 互動式圖表
# =========================
def build_figure(
    df_stock: pd.DataFrame,
    fifo_df: pd.DataFrame,
    open_pos: pd.DataFrame,
    events_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    stock_name: str,
    latest_price: float,
    latest_src: str,
    anonymous: bool,
) -> go.Figure:

    has_sells = not fifo_df.empty
    has_events = not events_df.empty
    has_monthly = not monthly_df.empty

    # 累計損益序列（有賣出才有）
    cumulative_pnl = None
    if has_sells and "實現損益" in fifo_df.columns:
        cp = fifo_df.sort_values("賣出日期")[["賣出日期", "實現損益", "少賺金額"]].copy()
        cp["累計損益"] = cp["實現損益"].cumsum()
        cp["累計少賺"] = cp["少賺金額"].cumsum()
        cumulative_pnl = cp

    # --- 決定子圖數量 ---
    n_rows = 3
    if cumulative_pnl is not None:
        n_rows += 1   # 圖4：累計損益 + 少賺金額對比
    if has_monthly:
        n_rows += 1   # 月度損益
    if has_sells:
        n_rows += 2   # ROI分佈直方圖 + 持倉時間軸

    # row heights: 圖1最高，其餘等分
    row_heights = [0.28] + [0.16] * (n_rows - 1)
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    subplot_titles = [
        f"{stock_name}｜交易執行圖（▲買入 ▼賣出）",
        "賣出後悔分析：Y>0=賣太早（後悔區）｜Y<0=賣在高點（✓）",
        "持倉效率四象限（持有天數 vs ROI%）",
    ]
    if cumulative_pnl is not None:
        subplot_titles.append("累計實現損益 vs 累計少賺金額（機會成本）")
    if has_monthly:
        subplot_titles.append("月度實現損益")
    if has_sells:
        subplot_titles.append("ROI 分佈直方圖（盈虧分布一覽）")
        subplot_titles.append("持倉時間軸（每筆進出場甘特圖）")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.08,
    )

    # -------- 圖1：交易執行圖 --------
    buys = df_stock[df_stock["類型"] == "買入"]
    sells = df_stock[df_stock["類型"] == "賣出"]
    # 股息/其他
    others = df_stock[~df_stock["類型"].isin(["買入", "賣出"])]

    # 連線（報價序列）
    fig.add_trace(
        go.Scatter(
            x=df_stock["日期"], y=df_stock["報價_num"],
            mode="lines",
            line=dict(color="#5588aa", width=1.2),
            name="報價序列",
            hovertemplate="%{x|%Y-%m-%d}<br>報價：%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    def _hover_buy_sell(row_series, ttype):
        if row_series.empty:
            return []
        lines = []
        for _, r in row_series.iterrows():
            qty = f"{r['股數_num']:,.0f}" if not anonymous else "隱藏"
            amt = f"{r['金額_num']:,.0f}" if not anonymous else "隱藏"
            fee = f"{r['費用_num']:,.0f}" if not anonymous else "隱藏"
            lines.append(
                f"<b>{ttype}</b><br>"
                f"日期：{r['日期'].strftime('%Y-%m-%d')}<br>"
                f"報價：{r['報價_num']:.3f}<br>"
                f"股數：{qty}<br>"
                f"金額：{amt}<br>"
                f"費用：{fee}"
            )
        return lines

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["日期"], y=buys["報價_num"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#22bb66", line=dict(width=1, color="white")),
                name="買入",
                text=_hover_buy_sell(buys, "買入"),
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["日期"], y=sells["報價_num"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#ee4444", line=dict(width=1, color="white")),
                name="賣出",
                text=_hover_buy_sell(sells, "賣出"),
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    if not others.empty:
        for _, r in others.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[r["日期"]], y=[r["報價_num"]],
                    mode="markers",
                    marker=dict(symbol="circle", size=8, color="#aaaaaa"),
                    name=str(r["類型"]),
                    hovertemplate=(
                        f"<b>{r['類型']}</b><br>"
                        f"日期：{r['日期'].strftime('%Y-%m-%d')}<br>"
                        f"金額：{r['金額_num']:,.0f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # 持倉中：加一條最新價水平線
    fig.add_hline(
        y=latest_price, row=1, col=1,
        line=dict(color="gold", width=1, dash="dot"),
        annotation_text=f"最新 {latest_price:.2f}（{latest_src}）",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    # -------- 泡泡大小：統一歸一化到 [8, 55] px --------
    def _scale_sizes(arr, lo=8, hi=55):
        a = np.clip(np.asarray(arr, dtype=float), 0, None)
        mx = a.max()
        if mx <= 0:
            return np.full(len(a), lo, dtype=float)
        return lo + (a / mx) * (hi - lo)

    # -------- 圖2：決策後悔 (FIFO Bubble) --------
    if has_sells:
        size_col = "資金權重%" if anonymous else "資金權重%_raw"
        if size_col not in fifo_df.columns:
            size_col = "資金權重%"
        sizes = np.clip(fifo_df[size_col].fillna(0).values, 0, None)
        sizes_display = _scale_sizes(sizes)

        custom_data = np.column_stack([
            fifo_df["買入日期"].dt.strftime("%Y-%m-%d").values,   # 0
            fifo_df["買入價格"].values,                            # 1
            fifo_df["賣出價格"].values,                            # 2
            fifo_df["持有天數"].values,                            # 3
            fifo_df.get("股數", pd.Series([np.nan] * len(fifo_df))).values,        # 4
            fifo_df.get("實現損益", pd.Series([np.nan] * len(fifo_df))).values,    # 5
            fifo_df[size_col].fillna(0).values,                    # 6 資金佔比%
        ])

        w_label = "資金佔比" if not anonymous else "相對權重"
        hover_tmpl = (
            "<b>賣出日：%{x|%Y-%m-%d}</b><br>"
            "買入日：%{customdata[0]}<br>"
            "買入價：%{customdata[1]:.3f}<br>"
            "賣出價：%{customdata[2]:.3f}<br>"
            "持有天數：%{customdata[3]:.0f} 天<br>"
            f"{w_label}：%{{customdata[6]:.2f}}%<br>"
            + (
                "股數：%{customdata[4]:.0f}<br>"
                "實現損益：%{customdata[5]:,.0f}<br>"
                if not anonymous else ""
            ) +
            "賣後漲幅：%{y:.2f}%<br>"
            "實現ROI：%{marker.color:.2f}%"
            "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=fifo_df["賣出日期"],
                y=fifo_df["賣出後續漲幅%"],
                mode="markers",
                marker=dict(
                    size=sizes_display,
                    color=fifo_df["實現ROI%"],
                    colorscale="RdYlGn",
                    colorbar=dict(title="實現ROI%", x=1.02, thickness=12),
                    showscale=True,
                    line=dict(color="grey", width=0.5),
                    opacity=0.75,
                ),
                name="賣出 tranche",
                customdata=custom_data,
                hovertemplate=hover_tmpl,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, row=2, col=1, line=dict(color="white", width=1.5, dash="dash"))
        # 後悔區（Y>0）= 紅色背景；適時出場（Y<0）= 綠色背景
        _y2 = fifo_df["賣出後續漲幅%"].dropna()
        _y_hi = max(_y2.max() if len(_y2) else 1, 1) * 1.4
        _y_lo = min(_y2.min() if len(_y2) else -1, -1) * 1.4
        for _y0, _y1, _fc in [
            (0, _y_hi, "rgba(220,60,60,0.10)"),   # 後悔區
            (_y_lo, 0, "rgba(60,200,80,0.10)"),   # 適時出場
        ]:
            fig.add_shape(type="rect",
                          xref="x2 domain", yref="y2",
                          x0=0, x1=1, y0=_y0, y1=_y1,
                          fillcolor=_fc, line_width=0, layer="below")
        fig.add_annotation(
            text="▲ 後悔區（賣太早，股票繼續漲）", textangle=0,
            xref="x2 domain", yref="y2 domain", x=0.01, y=0.97,
            showarrow=False, font=dict(color="#ff8888", size=11), xanchor="left",
        )
        fig.add_annotation(
            text="▼ 適時出場（賣後股票沒再漲 ✓）", textangle=0,
            xref="x2 domain", yref="y2 domain", x=0.01, y=0.06,
            showarrow=False, font=dict(color="#88ee88", size=11), xanchor="left",
        )
    else:
        fig.add_annotation(
            text="沒有賣出紀錄，無法進行 FIFO 分析",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="grey"),
            row=2, col=1,
        )

    # -------- 圖3：持倉效率 --------
    if has_sells:
        ok = fifo_df[["持有天數", "實現ROI%"]].notna().all(axis=1)
        df_eff = fifo_df[ok].copy()
        sizes3 = np.clip(df_eff[size_col].fillna(0).values, 0, None)
        sizes3_display = _scale_sizes(sizes3)

        fig.add_trace(
            go.Scatter(
                x=df_eff["持有天數"],
                y=df_eff["實現ROI%"],
                mode="markers",
                marker=dict(
                    size=sizes3_display,
                    color=df_eff["實現ROI%"],
                    colorscale="RdYlGn",
                    showscale=False,
                    opacity=0.65,
                    line=dict(color="grey", width=0.5),
                ),
                name="持倉效率",
                customdata=np.column_stack([
                    df_eff["買入日期"].dt.strftime("%Y-%m-%d").values,
                    df_eff["賣出日期"].dt.strftime("%Y-%m-%d").values,
                    df_eff["買入價格"].values,
                    df_eff["賣出價格"].values,
                ]),
                hovertemplate=(
                    "持有：%{x} 天<br>"
                    "ROI：%{y:.2f}%<br>"
                    "買入：%{customdata[0]} @ %{customdata[2]:.3f}<br>"
                    "賣出：%{customdata[1]} @ %{customdata[3]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=3, col=1,
        )

        # 趨勢線
        x_vals = df_eff["持有天數"].values.astype(float)
        y_vals = df_eff["實現ROI%"].values.astype(float)
        fin = np.isfinite(x_vals) & np.isfinite(y_vals)
        if fin.sum() >= 3:
            z = np.polyfit(x_vals[fin], y_vals[fin], 1)
            xs = np.linspace(x_vals[fin].min(), x_vals[fin].max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=np.polyval(z, xs),
                    mode="lines",
                    line=dict(color="tomato", dash="dash", width=1.5),
                    name="ROI 趨勢",
                    hoverinfo="skip",
                ),
                row=3, col=1,
            )

        fig.add_hline(y=0, row=3, col=1, line=dict(color="white", width=1.2, dash="dash"))
        # 四象限：以持有天數中位數為垂直分割線
        _med_days = float(df_eff["持有天數"].median())
        fig.add_shape(type="line",
                      xref="x3", yref="y3 domain",
                      x0=_med_days, x1=_med_days, y0=0, y1=1,
                      line=dict(color="white", width=1, dash="dash"))
        # 四個角標籤（paper-domain 參考系）
        for _tx, _ty, _label, _col in [
            (0.02, 0.97, "快賺 ✓✓", "#88ee88"),    # 左上：短持倉獲利
            (0.98, 0.97, "慢賺 ✓",  "#aaddaa"),    # 右上：長持倉獲利
            (0.02, 0.05, "快虧 ✗✗", "#ee8888"),    # 左下：短持倉虧損
            (0.98, 0.05, "慢虧 ✗",  "#dd9999"),    # 右下：長持倉虧損
        ]:
            fig.add_annotation(
                text=_label,
                xref="x3 domain", yref="y3 domain",
                x=_tx, y=_ty,
                showarrow=False,
                font=dict(color=_col, size=12, family="Microsoft JhengHei, sans-serif"),
                xanchor="left" if _tx < 0.5 else "right",
                yanchor="top" if _ty > 0.5 else "bottom",
            )

    # -------- 圖4（可選）：累計實現損益 vs 累計少賺金額 --------
    current_row = 4
    if cumulative_pnl is not None:
        cp = cumulative_pnl

        # 單筆損益柱狀（綠=賺 紅=虧）
        colors = ["#22bb66" if v >= 0 else "#ee4444" for v in cp["實現損益"]]
        fig.add_trace(
            go.Bar(
                x=cp["賣出日期"], y=cp["實現損益"],
                name="單筆實現損益",
                marker_color=colors,
                opacity=0.45,
                hovertemplate="日期：%{x|%Y-%m-%d}<br>單筆損益：%{y:,.0f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        # 累計實現損益線（黃金）
        fig.add_trace(
            go.Scatter(
                x=cp["賣出日期"], y=cp["累計損益"],
                mode="lines+markers",
                line=dict(color="gold", width=2.5),
                name="累計實現損益",
                hovertemplate="日期：%{x|%Y-%m-%d}<br>累計損益：%{y:,.0f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        # 累計少賺金額線（橘紅虛線）= 若不賣持有至今的多賺部分
        fig.add_trace(
            go.Scatter(
                x=cp["賣出日期"], y=cp["累計少賺"],
                mode="lines",
                line=dict(color="#ff7744", width=1.8, dash="dot"),
                name="累計少賺金額（賣後機會成本）",
                hovertemplate=(
                    "日期：%{x|%Y-%m-%d}<br>"
                    "累計少賺：%{y:,.0f}<br>"
                    "<i>若持有至今多賺的總金額</i>"
                    "<extra></extra>"
                ),
            ),
            row=current_row, col=1,
        )
        fig.add_hline(y=0, row=current_row, col=1, line=dict(color="grey", width=1, dash="dash"))

        # 終點標註：告訴用戶缺口是多少
        last_missed = cp["累計少賺"].iloc[-1]
        gap = last_missed  # 若持有到今還能多賺（少賺累計值）
        gap_color = "#ff7744" if gap > 0 else "#22bb66"
        gap_text = f"尚有機會成本 {gap:+,.0f}" if gap > 0 else f"已超越持有策略 {gap:+,.0f}"
        fig.add_annotation(
            x=cp["賣出日期"].iloc[-1],
            y=last_missed,
            text=gap_text,
            showarrow=True, arrowhead=2, arrowcolor=gap_color,
            font=dict(color=gap_color, size=11),
            ax=60, ay=-30,
            row=current_row, col=1,
        )
        current_row += 1

    # -------- 圖5（可選）：月度損益 --------
    if has_monthly:
        m_colors = ["#22bb66" if v >= 0 else "#ee4444" for v in monthly_df["月度損益"]]
        fig.add_trace(
            go.Bar(
                x=monthly_df["月份"],
                y=monthly_df["月度損益"],
                name="月度損益",
                marker_color=m_colors,
                customdata=np.column_stack([
                    monthly_df["交易次數"].values,
                    monthly_df["平均ROI"].values,
                    monthly_df["勝率"].values,
                ]),
                hovertemplate=(
                    "%{x}<br>"
                    "損益：%{y:,.0f}<br>"
                    "交易次數：%{customdata[0]}<br>"
                    "平均ROI：%{customdata[1]:.1f}%<br>"
                    "勝率：%{customdata[2]:.0f}%"
                    "<extra></extra>"
                ),
            ),
            row=current_row, col=1,
        )
        fig.add_hline(y=0, row=current_row, col=1, line=dict(color="grey", width=1, dash="dash"))
        current_row += 1

    # -------- ROI 分佈直方圖 --------
    if has_sells:
        roi_vals = fifo_df["實現ROI%"].dropna()
        avg_roi_hist = roi_vals.mean()
        win_cnt = (roi_vals > 0).sum()
        loss_cnt = (roi_vals <= 0).sum()
        fig.add_trace(
            go.Histogram(
                x=roi_vals,
                nbinsx=30,
                name="ROI 分佈",
                marker=dict(
                    color=[
                        "#22bb66" if v > 0 else "#ee4444"
                        for v in roi_vals
                    ],
                    line=dict(color="#111", width=0.5),
                ),
                hovertemplate="ROI：%{x:.1f}%<br>筆數：%{y}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.add_shape(type="line",
                      xref="x" + str(current_row), yref="y" + str(current_row) + " domain",
                      x0=0, x1=0, y0=0, y1=1,
                      line=dict(color="white", width=1.5, dash="dash"))
        fig.add_shape(type="line",
                      xref="x" + str(current_row), yref="y" + str(current_row) + " domain",
                      x0=avg_roi_hist, x1=avg_roi_hist, y0=0, y1=1,
                      line=dict(color="gold", width=1.5, dash="dot"))
        fig.add_annotation(
            text=f"平均 {avg_roi_hist:.1f}%  |  獲利 {win_cnt} 筆  虧損 {loss_cnt} 筆",
            xref="x" + str(current_row) + " domain", yref="y" + str(current_row) + " domain",
            x=0.99, y=0.97, showarrow=False,
            font=dict(size=11, color="gold"), xanchor="right",
        )
        current_row += 1

    # -------- 持倉時間軸（甘特圖）--------
    if has_sells:
        gantt_row = current_row
        sold = fifo_df.sort_values("買入日期").copy()
        for i, (_, r) in enumerate(sold.iterrows()):
            roi_val = r["實現ROI%"]
            bar_color = "#22bb66" if (np.isfinite(roi_val) and roi_val > 0) else "#ee4444"
            hover = (
                f"買入：{r['買入日期'].strftime('%Y-%m-%d')} @ {r['買入價格']:.2f}<br>"
                f"賣出：{r['賣出日期'].strftime('%Y-%m-%d')} @ {r['賣出價格']:.2f}<br>"
                f"持有：{r['持有天數']} 天   ROI：{roi_val:.1f}%"
            )
            fig.add_trace(
                go.Scatter(
                    x=[r["買入日期"], r["賣出日期"]],
                    y=[i, i],
                    mode="lines",
                    line=dict(color=bar_color, width=6),
                    name=f"tranche {i+1}",
                    hovertext=hover,
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=gantt_row, col=1,
            )
        # 現持倉中（未賣出部分）= 灰色延伸線
        if not open_pos.empty:
            today = pd.Timestamp.today().normalize()
            for j, (_, op) in enumerate(open_pos.iterrows()):
                fig.add_trace(
                    go.Scatter(
                        x=[op["買入日期"], today],
                        y=[len(sold) + j, len(sold) + j],
                        mode="lines",
                        line=dict(color="#aaaaaa", width=5, dash="dot"),
                        hovertext=(
                            f"持倉中：買入 {op['買入日期'].strftime('%Y-%m-%d')} "
                            f"@ {op['買入價格']:.2f}，尚未賣出"
                        ),
                        hoverinfo="text",
                        showlegend=False,
                    ),
                    row=gantt_row, col=1,
                )
        fig.update_yaxes(visible=False, row=gantt_row, col=1)

    # -------- 全圖佈局 --------
    anon_label = "（匿名版）" if anonymous else "（完整版）"

    # 一眼看懂摘要
    verdict_parts = []
    if has_sells:
        n_win  = (fifo_df["實現ROI%"] > 0).sum()
        n_total = fifo_df["實現ROI%"].notna().sum()
        avg_roi = fifo_df["實現ROI%"].mean()
        win_rate = n_win / n_total * 100 if n_total > 0 else 0
        verdict_parts.append(
            f"{'✅' if win_rate>=60 else '⚠️' if win_rate>=50 else '🔴'} 勝率 {win_rate:.0f}%"
        )
        verdict_parts.append(
            f"{'✅' if avg_roi>=5 else '⚠️' if avg_roi>=0 else '🔴'} 平均ROI {avg_roi:.1f}%"
        )
        if "少賺金額" in fifo_df.columns:
            missed_pos = fifo_df[fifo_df["少賺金額"] > 0]["少賺金額"].sum()
            if missed_pos > 0:
                verdict_parts.append(f"🔴 賣太早損失機會成本 {missed_pos:,.0f}")
            else:
                verdict_parts.append("✅ 無明顯賣太早問題")
    verdict_text = "  |  ".join(verdict_parts) if verdict_parts else ""

    total_pnl_str = ""
    if has_sells and "實現損益" in fifo_df.columns:
        total_pnl = fifo_df["實現損益"].sum()
        total_pnl_str = f"  |  累計實現損益：{total_pnl:,.0f}"

    cjk_fonts = "Microsoft JhengHei, Microsoft YaHei, SimHei, Noto Sans CJK TC, Arial Unicode MS, sans-serif"
    fig.update_layout(
        title=dict(
            text=f"{stock_name} 交易記錄分析 {anon_label}{total_pnl_str}",
            font=dict(size=16, family=cjk_fonts),
        ),
        font=dict(family=cjk_fonts, size=13),
        height=230 * n_rows + 100,
        template="plotly_dark",
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", y=-0.03, x=0, font=dict(family=cjk_fonts)),
        margin=dict(l=60, r=80, t=110, b=60),
        paper_bgcolor="#111",
        plot_bgcolor="#1a1a1a",
    )

    # 一眼看懂摘要文字（固定在最頂端）
    if verdict_text:
        fig.add_annotation(
            text=verdict_text,
            xref="paper", yref="paper", x=0.5, y=1.03,
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=12, color="#dddddd", family=cjk_fonts),
            bgcolor="rgba(40,40,40,0.85)",
            bordercolor="#555", borderwidth=1, borderpad=6,
        )

    # 子圖標題字型（subplot_titles 是 annotations，需另設）
    for ann in fig.layout.annotations:
        ann.font = dict(family=cjk_fonts, size=13)

    # 軸標籤
    fig.update_xaxes(gridcolor="#333", title_font=dict(family=cjk_fonts))
    fig.update_yaxes(gridcolor="#333", title_font=dict(family=cjk_fonts))
    fig.update_yaxes(title_text="報價", row=1, col=1)
    fig.update_yaxes(title_text="賣後漲幅 (%)", row=2, col=1)
    fig.update_xaxes(title_text="持有天數", row=3, col=1)
    fig.update_yaxes(title_text="實現 ROI (%)", row=3, col=1)
    if cumulative_pnl is not None:
        fig.update_yaxes(title_text="損益 (元)", row=4, col=1)
    if has_monthly:
        fig.update_yaxes(title_text="月度損益 (元)", row=current_row, col=1)

    return fig


# =========================
# 摘要統計（文字輸出）
# =========================
def print_summary(
    fifo_df: pd.DataFrame,
    open_pos: pd.DataFrame,
    events_df: pd.DataFrame,
    stock_name: str,
    latest_price: float,
    latest_src: str,
    anonymous: bool,
):
    print(f"\n{'='*60}")
    print(f"  {stock_name}  交易分析摘要")
    print(f"  最新價格：{latest_price:.2f}（來源：{latest_src}）")
    print(f"{'='*60}")

    if fifo_df.empty:
        print("  ⚠️  尚無賣出紀錄，無法進行 FIFO 分析。")
    else:
        n = len(fifo_df)
        wins = (fifo_df["實現ROI%"] > 0).sum()
        avg_roi = fifo_df["實現ROI%"].mean()
        avg_days = fifo_df["持有天數"].mean()
        print(f"  賣出 tranche 數：{n}")
        print(f"  勝率：{wins}/{n} = {wins/n*100:.1f}%")
        print(f"  平均 ROI：{avg_roi:.2f}%")
        print(f"  平均持倉天數：{avg_days:.1f} 天")
        if not anonymous and "實現損益" in fifo_df.columns:
            total_pnl = fifo_df["實現損益"].sum()
            print(f"  累計實現損益：{total_pnl:,.0f} 元")

    # ── 補回分析（完整模式）──
    if not anonymous and not fifo_df.empty and "少賺金額" in fifo_df.columns:
        total_pnl    = fifo_df["實現損益"].sum() if "實現損益" in fifo_df.columns else 0
        total_missed = fifo_df["少賺金額"].sum()     # 已賣出部分：正=後悔，負=賣對了
        missed_pos   = fifo_df[fifo_df["少賺金額"] > 0]["少賺金額"].sum()  # 只算後悔的部分

        # 未實現損益（持倉中）
        unrealized = 0.0
        if not open_pos.empty:
            total_qty = open_pos["股數"].sum()
            avg_cost  = (open_pos["股數"] * open_pos["買入價格"]).sum() / total_qty if total_qty > 0 else 0
            unrealized = (latest_price - avg_cost) * total_qty if avg_cost > 0 else 0.0

        net_total = total_pnl + unrealized   # 實際到手 + 帳面
        print(f"\n  ── 少賺 vs 補回分析 ──")
        print(f"  累計實現損益：          {total_pnl:>12,.0f} 元")
        print(f"  現持倉未實現損益（估）： {unrealized:>12,.0f} 元")
        print(f"  合計（實現+未實現）：   {net_total:>12,.0f} 元")
        print(f"  ────────────────────────────────────")
        if total_missed > 0:
            print(f"  賣後機會成本（少賺）：  {missed_pos:>12,.0f} 元  ← 已賣出後股票繼續漲")
            gap = missed_pos - unrealized
            if gap > 0:
                print(f"  🔴 尚未補回缺口：       {gap:>12,.0f} 元（持倉未實現不足以覆蓋少賺）")
            else:
                print(f"  🟢 現持倉已補回：        多了 {-gap:>10,.0f} 元（持倉未實現 > 少賺機會成本）")
        else:
            print(f"  🟢 整體賣得不錯，賣後股票並未持續上漲（累計少賺 {total_missed:,.0f} 元）")
    elif not open_pos.empty:
        total_qty = open_pos["股數"].sum()
        avg_cost  = (open_pos["股數"] * open_pos["買入價格"]).sum() / total_qty if total_qty > 0 else 0
        unrealized = (latest_price - avg_cost) * total_qty if avg_cost > 0 else 0

    if not open_pos.empty:
        total_qty = open_pos["股數"].sum()
        avg_cost  = (open_pos["股數"] * open_pos["買入價格"]).sum() / total_qty if total_qty > 0 else 0
        unrealized = (latest_price - avg_cost) * total_qty if avg_cost > 0 else 0
        print(f"\n  持倉中：{total_qty:,.0f} 股  均成本：{avg_cost:.3f}")
        if not anonymous:
            pct = (latest_price / avg_cost - 1) * 100 if avg_cost > 0 else 0
            print(f"  未實現損益（估）：{unrealized:,.0f} 元（{pct:+.2f}%）")

    if not events_df.empty:
        print(f"\n  清倉→買回事件：{len(events_df)} 次")
        print(events_df.to_string(index=False))

    print(f"{'='*60}\n")


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser(description="交易記錄互動式分析（Plotly 版）")
    parser.add_argument("--csv", default="re.csv")
    parser.add_argument("--target", default="元大台灣50正2")
    parser.add_argument("--use_yf", action="store_true")
    parser.add_argument("--yf_ticker", default="00631L.TW")
    parser.add_argument("--anonymous", action="store_true", help="匿名模式（不顯示股數/金額）")
    parser.add_argument("--list", action="store_true", help="列出 CSV 中所有證券名稱後退出")
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(work_dir, args.csv)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")

    df = load_transactions(csv_path)

    if args.list:
        stocks = df["證券"].dropna().unique()
        print("CSV 中的所有證券：")
        for s in sorted(stocks):
            cnt = len(df[df["證券"] == s])
            print(f"  {s}（{cnt} 筆）")
        return 0

    # 過濾目標
    df_stock = df[df["證券"] == args.target].copy().sort_values("日期")
    if df_stock.empty:
        # 模糊匹配
        alt = df[df["證券"].astype(str).str.contains(args.target, na=False)]
        if not alt.empty:
            args.target = alt["證券"].value_counts().index[0]
            df_stock = df[df["證券"] == args.target].copy().sort_values("日期")

    if df_stock.empty:
        raise RuntimeError(f"找不到 '{args.target}' 的交易紀錄。\n使用 --list 查看所有可用證券。")

    # 最新價格
    latest_price = float(df_stock.iloc[-1]["報價_num"])
    latest_src = "CSV 最後一筆報價"
    if args.use_yf:
        p = try_get_latest_close_yf(args.yf_ticker)
        if p and np.isfinite(p) and p > 0:
            latest_price = p
            latest_src = f"yfinance ({args.yf_ticker})"

    # 分析
    fifo_df = build_fifo_df(df_stock, latest_price, anonymous=args.anonymous)
    open_pos = get_open_positions(df_stock)
    events_df = detect_exit_reentry(df_stock)
    monthly_df = build_monthly_pnl(fifo_df) if not fifo_df.empty else pd.DataFrame()

    # 文字摘要
    print_summary(fifo_df, open_pos, events_df, args.target, latest_price, latest_src, args.anonymous)

    # 互動圖（瀏覽器開啟）
    fig = build_figure(
        df_stock, fifo_df, open_pos, events_df, monthly_df,
        args.target, latest_price, latest_src, args.anonymous,
    )
    fig.show()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\n=== 發生錯誤 ===")
        traceback.print_exc()
        input("\n按 Enter 關閉…")
        raise
