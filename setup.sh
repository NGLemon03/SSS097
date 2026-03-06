#!/bin/bash

# SSS096 專案 Codex 環境設置腳本
# 路徑：#setup.sh
# 創建時間：2025-08-18 12:00
# 用途：解決 Codex 環境中的依賴安裝問題

echo "🚀 開始設置 SSS096 專案 Codex 環境..."

# 設置環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export SSS_CREATE_LOGS="1"

# 配置 pip 代理設置（如果遇到 403 錯誤）
if [ -n "$CODEX_PROXY_CERT" ]; then
    echo "🔧 配置 pip 代理證書..."
    export PIP_CERT="$CODEX_PROXY_CERT"
    export NODE_EXTRA_CA_CERTS="$CODEX_PROXY_CERT"
    
    # 配置 pip 使用代理
    pip config set global.cert "$CODEX_PROXY_CERT"
    pip config set global.trusted-host "proxy:8080"
fi

# 升級 pip 到最新版本
echo "📦 升級 pip..."
python -m pip install --upgrade pip

# 安裝核心依賴套件
echo "📦 安裝核心依賴套件..."
pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    openpyxl \
    dash \
    dash-bootstrap-components \
    yfinance \
    pyyaml \
    joblib

# 安裝分析相關套件
echo "📦 安裝分析相關套件..."
pip install --no-cache-dir \
    scikit-learn \
    scipy \
    statsmodels \
    plotly \
    kaleido

# 安裝開發工具
echo "📦 安裝開發工具..."
pip install --no-cache-dir \
    pytest \
    black \
    flake8 \
    mypy \
    ruff

# 創建必要的目錄結構
echo "📁 創建必要的目錄結構..."
mkdir -p analysis/log
mkdir -p analysis/cache
mkdir -p analysis/grids
mkdir -p analysis/presets
mkdir -p cache
mkdir -p log
mkdir -p results
mkdir -p sss_backtest_outputs

# 設置日誌配置
echo "🔧 設置日誌配置..."
cat > analysis/logging_config_fallback.py << 'EOF'
# -*- coding: utf-8 -*-
"""
日誌配置回退模組 - 當 joblib 不可用時使用
路徑：#analysis/logging_config_fallback.py
創建時間：2025-08-18 12:00
"""

import logging
import os
from pathlib import Path
from datetime import datetime

def init_logging_fallback(enable_file=True):
    """初始化日誌系統（回退版本）"""
    
    # 創建日誌目錄
    log_root = Path("analysis/log")
    log_root.mkdir(parents=True, exist_ok=True)
    
    # 生成時間戳記
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 配置根日誌器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除現有的 handlers
    root_logger.handlers.clear()
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 檔案 handler（如果啟用）
    if enable_file:
        try:
            file_handler = logging.FileHandler(
                log_root / f"system_{timestamp}.log",
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"✅ 日誌檔案已創建：{file_handler.baseFilename}")
        except Exception as e:
            print(f"⚠️ 無法創建日誌檔案：{e}")
    
    print("✅ 日誌系統初始化完成（回退版本）")
    return root_logger

def get_logger_fallback(name):
    """獲取日誌器（回退版本）"""
    return logging.getLogger(name)

# 導出函數
__all__ = ['init_logging_fallback', 'get_logger_fallback']
EOF

# 創建測試腳本
echo "🧪 創建測試腳本..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設置測試腳本
路徑：#test_setup.py
創建時間：2025-08-18 12:00
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """測試關鍵模組導入"""
    print("🧪 測試模組導入...")
    
    # 測試核心套件
    core_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'dash', 'plotly', 'yfinance', 'pyyaml'
    ]
    
    for package in core_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - 導入成功")
        except ImportError as e:
            print(f"❌ {package} - 導入失敗: {e}")
    
    # 測試專案模組
    project_modules = [
        'analysis.config',
        'analysis.logging_config',
        'ensemble_wrapper'
    ]
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} - 導入成功")
        except ImportError as e:
            print(f"⚠️ {module} - 導入失敗: {e}")

def test_logging():
    """測試日誌系統"""
    print("\n🧪 測試日誌系統...")
    
    try:
        # 嘗試使用標準日誌配置
        from analysis.logging_config import init_logging
        init_logging(enable_file=False)
        print("✅ 標準日誌系統初始化成功")
    except Exception as e:
        print(f"⚠️ 標準日誌系統初始化失敗: {e}")
        
        # 使用回退版本
        try:
            from analysis.logging_config_fallback import init_logging_fallback
            init_logging_fallback(enable_file=False)
            print("✅ 回退日誌系統初始化成功")
        except Exception as e2:
            print(f"❌ 回退日誌系統也失敗: {e2}")

def test_data_access():
    """測試數據存取"""
    print("\n🧪 測試數據存取...")
    
    # 檢查必要目錄
    required_dirs = [
        'analysis', 'cache', 'log', 'results', 
        'sss_backtest_outputs', 'data'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name} - 目錄存在")
        else:
            print(f"⚠️ {dir_name} - 目錄不存在")

def main():
    """主測試函數"""
    print("🚀 開始測試 SSS096 專案設置...")
    print("=" * 50)
    
    test_imports()
    test_logging()
    test_data_access()
    
    print("\n" + "=" * 50)
    print("🎉 設置測試完成！")
    
    # 檢查是否有嚴重錯誤
    print("\n📊 測試結果摘要：")
    print("- 如果看到 ❌ 錯誤，請檢查依賴安裝")
    print("- 如果看到 ⚠️ 警告，功能可能受限但基本可用")
    print("- 如果看到 ✅ 成功，設置完成")

if __name__ == "__main__":
    main()
EOF

# 設置權限
chmod +x setup.sh
chmod +x test_setup.py

# 創建環境變數配置文件
echo "🔧 創建環境變數配置文件..."
cat > .env.codex << 'EOF'
# Codex 環境變數配置
# 路徑：#.env.codex
# 創建時間：2025-08-18 12:00

# Python 路徑
PYTHONPATH=.:${PYTHONPATH}

# 日誌設置
SSS_CREATE_LOGS=1

# 代理設置（如果可用）
if [ -n "$CODEX_PROXY_CERT" ]; then
    export PIP_CERT="$CODEX_PROXY_CERT"
    export NODE_EXTRA_CA_CERTS="$CODEX_PROXY_CERT"
fi
EOF

# 創建 requirements.txt
echo "📦 創建 requirements.txt..."
cat > requirements.txt << 'EOF'
# SSS096 專案依賴套件
# 路徑：#requirements.txt
# 創建時間：2025-08-18 12:00

# 核心數據處理
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Web UI
dash>=2.0.0
dash-bootstrap-components>=1.0.0
plotly>=5.0.0

# 數據分析
scikit-learn>=1.1.0
scipy>=1.9.0
statsmodels>=0.13.0

# 文件處理
openpyxl>=3.0.0
pyyaml>=6.0

# 金融數據
yfinance>=0.1.70

# 快取和並行
joblib>=1.2.0

# 開發工具
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
ruff>=0.0.200
EOF

# 創建 pip 配置
echo "🔧 創建 pip 配置..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
# 使用國內鏡像源（如果代理有問題）
index-url = https://pypi.org/simple/
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org

# 代理設置（如果可用）
[global]
cert = ${CODEX_PROXY_CERT}
trusted-host = proxy:8080
EOF

echo ""
echo "🎉 SSS096 專案 Codex 環境設置完成！"
echo ""
echo "📋 下一步操作："
echo "1. 測試設置：python test_setup.py"
echo "2. 檢查依賴：pip list"
echo "3. 運行快速檢查：powershell -ExecutionPolicy Bypass -File tools\\quick_check.ps1"
echo ""
echo "🔧 如果遇到問題："
echo "- 檢查代理設置：echo \$CODEX_PROXY_CERT"
echo "- 檢查 pip 配置：cat ~/.pip/pip.conf"
echo "- 查看錯誤日誌：tail -f analysis/log/*.log"
echo ""
echo "📚 更多信息請參考 AGENTS.md"
