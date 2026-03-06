# -*- coding: utf-8 -*-
"""
Enhanced Data Diagnostics Module
路徑：#debug_enhanced_data.py
創建時間：2025-08-18 12:00
作者：AI Assistant

用於診斷 enhanced analysis 的資料來源問題
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import logging

def diag_df(name, df):
    """診斷 DataFrame 的詳細資訊"""
    print(f"\n=== DIAG: {name} ===\n")
    
    if df is None:
        print("DataFrame 為 None")
        return
    
    if not isinstance(df, pd.DataFrame):
        print(f"不是 DataFrame，類型為：{type(df)}")
        return
    
    print("shape:", getattr(df, "shape", None))
    print("columns:", list(df.columns))
    
    # 檢查重複欄名
    cols = list(df.columns)
    dups = [c for c in pd.Index(cols).unique() if cols.count(c) > 1]
    print("duplicated column names:", dups)
    
    print("dtypes:\n", df.dtypes)
    print("null counts:\n", df.isna().sum().to_dict())
    
    # 顯示前10行
    if len(df) > 0:
        print("\nHEAD:\n", df.head(10).to_string())
    else:
        print("\nDataFrame 為空")
    
    # 數值欄位診斷
    numeric_candidates = ['盈虧%', 'return', 'profit', 'price', 'shares', 'weight_change', '權重變化', '收盤價']
    for candidate in numeric_candidates:
        if candidate in df.columns:
            try:
                s = pd.to_numeric(df[candidate], errors='coerce')
                print(f"\n-- {candidate} stats --")
                print("non-numeric count:", s.isna().sum(), "/", len(s))
                if s.dropna().size > 0:
                    print("min/max/mean/median:", s.min(), s.max(), s.mean(), s.median())
                    # 檢測百分點 vs 分數
                    absmax = s.abs().max()
                    print("absmax indicates scale:", absmax)
                    if absmax > 100:
                        print(" -> values likely already in percent (e.g. 23 for 23%) or outliers")
                    elif absmax <= 1:
                        print(" -> values likely fractions (e.g. 0.0023 for 0.23%)")
                    else:
                        print(" -> values likely in percent (e.g. 0.23 for 0.23%)")
            except Exception as e:
                print(f"處理 {candidate} 時出錯：{e}")
    
    # 日期欄位檢查
    date_candidates = [c for c in df.columns if '日期' in c or 'date' in c.lower()]
    print(f"\ndate columns: {date_candidates}")
    for dc in date_candidates:
        try:
            dt_series = pd.to_datetime(df[dc], errors='coerce')
            nat_count = dt_series.isna().sum()
            print(f"{dc} dtype: {df[dc].dtype}, NaT count: {nat_count}")
            if nat_count > 0:
                print(f"  -> 警告：{nat_count} 個日期解析失敗")
        except Exception as e:
            print(f"處理日期欄位 {dc} 時出錯：{e}")
    
    # 重複行檢查
    try:
        dup_rows = df.duplicated().sum()
        print(f"duplicated rows (full-row): {dup_rows}")
        if dup_rows > 0:
            print(f"  -> 警告：有 {dup_rows} 行完全重複")
    except Exception as e:
        print(f"dup check error: {e}")

def diag_results_obj(obj):
    """診斷結果物件的結構"""
    print("\n=== RESULTS OBJ KEYS ===")
    if obj is None:
        print("物件為 None")
        return
    
    if isinstance(obj, dict):
        print("keys:", list(obj.keys()))
        for k in obj.keys():
            try:
                v = obj[k]
                if hasattr(v, "shape"):
                    print(f"{k} -> shape: {v.shape}, type: {type(v)}")
                elif isinstance(v, (list, tuple)):
                    print(f"{k} -> type: {type(v)}, length: {len(v)}")
                else:
                    print(f"{k} -> type: {type(v)}")
            except Exception as e:
                print(f"error introspecting {k}: {e}")
    else:
        print(f"results type: {type(obj)}")
        if hasattr(obj, "__dict__"):
            print("attributes:", list(obj.__dict__.keys()))

def try_load_file(p):
    """嘗試載入各種格式的檔案"""
    p = Path(p)
    if not p.exists():
        print(f"檔案不存在: {p}")
        return None
    
    try:
        if p.suffix in ('.json', '.txt'):
            with p.open('r', encoding='utf8') as f:
                return json.load(f)
        elif p.suffix in ('.pkl', '.pickle'):
            import pickle
            with p.open('rb') as f:
                return pickle.load(f)
        elif p.suffix in ('.csv',):
            return pd.read_csv(p)
        else:
            print(f"未知格式: {p}")
            return None
    except Exception as e:
        print(f"載入檔案 {p} 時出錯: {e}")
        return None

def quick_diag_all(trades_df=None, benchmark_df=None, results_obj=None):
    """快速診斷所有資料來源"""
    print("=== 快速診斷所有資料來源 ===\n")
    
    if trades_df is not None:
        diag_df("trades_df", trades_df)
    
    if benchmark_df is not None:
        diag_df("benchmark_df", benchmark_df)
    
    if results_obj is not None:
        diag_results_obj(results_obj)
    
    print("\n=== 診斷完成 ===")

if __name__ == "__main__":
    print("Enhanced Data Diagnostics Module")
    print("使用方法：")
    print("from debug_enhanced_data import diag_df, diag_results_obj, try_load_file")
    print("diag_df('name', dataframe)")
    print("diag_results_obj(results_object)")
    print("data = try_load_file('path/to/file')")
