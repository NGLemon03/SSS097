#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna 版本配置檔案
用於管理 Optuna 13 和 Optuna 15 的版本差異
"""

from pathlib import Path
from typing import Dict, List, Optional

# === 資料夾配置 ===
RESULT_FOLDERS = {
    "Optuna 13 (預設)": Path("../results"),
    "Optuna 15": Path("../results_op15"),
    "Optuna 13 (備用)": Path("../results_op13"),
}

# === 版本特徵配置 ===
VERSION_FEATURES = {
    "Optuna 13": {
        "description": "原始版本，使用分離的參數欄位",
        "param_format": "separated",  # param_* 和 parameters_* 分離
        "key_fields": [
            "trial_number", "score", "strategy", "data_source",
            "param_linlen", "param_smaalen", "param_devwin", "param_factor",
            "total_return", "num_trades", "sharpe_ratio", "max_drawdown",
            "profit_factor", "avg_hold_days", "pbo_score", "avg_excess_return"
        ],
        "optional_fields": [
            "excess_return_stress"  # Optuna 15 新增，Optuna 13 可能沒有
        ]
    },
    "Optuna 15": {
        "description": "新版本，使用統一的 JSON 參數格式",
        "param_format": "unified",  # 統一的 parameters JSON
        "key_fields": [
            "trial_number", "score", "strategy", "data_source",
            "parameters",  # JSON 格式的參數
            "total_return", "num_trades", "sharpe_ratio", "max_drawdown",
            "profit_factor", "avg_hold_days", "pbo_score", "avg_excess_return",
            "excess_return_stress"  # 新增欄位
        ],
        "optional_fields": []
    }
}

# === 檔案命名模式 ===
FILENAME_PATTERNS = {
    "results": "optuna_results_{strategy}_{data_source}_{timestamp}.csv",
    "events": "optuna_events_{strategy}_{data_source}_{timestamp}.csv",
    "best_params": "optuna_best_params_{strategy}_{data_source}_{timestamp}.json",
    "sqlite": "optuna_{strategy}_{data_source}_{timestamp}.sqlite3"
}

# === 數據源配置 ===
DATA_SOURCES = {
    "Self": "Self",
    "Factor (^TWII / 2412.TW)": "Factor (^TWII / 2412.TW)",
    "Factor (^TWII / 2414.TW)": "Factor (^TWII / 2414.TW)"
}

# === 策略配置 ===
STRATEGIES = {
    "single": "單一指標策略",
    "dual": "雙指標策略", 
    "RMA": "RMA 策略",
    "ssma_turn": "SSMA 轉向策略"
}

def detect_optuna_version(csv_file: Path) -> Optional[str]:
    """
    根據 CSV 檔案內容檢測 Optuna 版本
    
    Args:
        csv_file: CSV 檔案路徑
    
    Returns:
        str: 版本名稱或 None
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_file, nrows=1)  # 只讀取標題行
        
        # 檢查是否有 parameters 欄位（Optuna 15 特徵）
        if 'parameters' in df.columns:
            return "Optuna 15"
        
        # 檢查是否有 param_* 欄位（Optuna 13 特徵）
        param_cols = [col for col in df.columns if col.startswith('param_')]
        if param_cols:
            return "Optuna 13"
        
        return None
        
    except Exception as e:
        print(f"檢測版本失敗 {csv_file}: {e}")
        return None

def get_available_folders() -> Dict[str, Dict]:
    """
    獲取可用的結果資料夾
    
    Returns:
        Dict: 可用資料夾資訊
    """
    available_folders = {}
    
    for folder_name, folder_path in RESULT_FOLDERS.items():
        if folder_path.exists():
            csv_files = list(folder_path.glob("*.csv"))
            if csv_files:
                # 檢測版本
                version_info = {}
                for csv_file in csv_files[:3]:  # 檢查前3個檔案
                    version = detect_optuna_version(csv_file)
                    if version:
                        version_info[version] = version_info.get(version, 0) + 1
                
                # 確定主要版本
                main_version = max(version_info.items(), key=lambda x: x[1])[0] if version_info else "未知"
                
                available_folders[folder_name] = {
                    'path': folder_path,
                    'file_count': len(csv_files),
                    'main_version': main_version,
                    'version_distribution': version_info
                }
    
    return available_folders

def get_version_features(version: str) -> Dict:
    """
    獲取指定版本的特徵
    
    Args:
        version: 版本名稱
    
    Returns:
        Dict: 版本特徵
    """
    return VERSION_FEATURES.get(version, {})

def format_folder_display_name(folder_name: str, folder_info: Dict) -> str:
    """
    格式化資料夾顯示名稱
    
    Args:
        folder_name: 原始資料夾名稱
        folder_info: 資料夾資訊
    
    Returns:
        str: 格式化後的顯示名稱
    """
    file_count = folder_info['file_count']
    main_version = folder_info['main_version']
    return f"{folder_name} ({file_count} 個檔案, {main_version})"

# === 使用範例 ===
if __name__ == "__main__":
    print("=== Optuna 版本配置測試 ===")
    
    # 檢查可用資料夾
    available = get_available_folders()
    print(f"可用資料夾: {len(available)} 個")
    
    for folder_name, folder_info in available.items():
        print(f"\n📁 {folder_name}")
        print(f"   路徑: {folder_info['path']}")
        print(f"   檔案數: {folder_info['file_count']}")
        print(f"   主要版本: {folder_info['main_version']}")
        print(f"   版本分布: {folder_info['version_distribution']}")
    
    # 顯示版本特徵
    print("\n=== 版本特徵 ===")
    for version, features in VERSION_FEATURES.items():
        print(f"\n🔧 {version}")
        print(f"   描述: {features['description']}")
        print(f"   參數格式: {features['param_format']}")
        print(f"   關鍵欄位: {len(features['key_fields'])} 個")
        if features['optional_fields']:
            print(f"   可選欄位: {features['optional_fields']}") 