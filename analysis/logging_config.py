# -*- coding: utf-8 -*-
"""
統一日誌配置系統 - 按需初始化版本
路徑：#analysis/logging_config.py
更新時間：2025-01-20 16:00
作者：AI Assistant

統一的日誌配置，支援檔案和控制台輸出
採用懶加載模式，避免模組載入時自動創建空白日誌檔案
"""

import os
import logging
import logging.config
from pathlib import Path
from datetime import datetime

# 基本路徑設定
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"

# 全域變數來追蹤初始化狀態
_logging_initialized = False
_file_logging_enabled = False


def _env_log_level(default: str = "INFO") -> str:
    return os.environ.get("LOG_LEVEL") or os.environ.get("SSS_LOG_LEVEL", default)


def _normalize_log_level(value: str, default: str = "INFO") -> str:
    level = (value or default).upper()
    return level if level in logging._nameToLevel else default

def ensure_log_dirs():
    """確保日誌目錄存在"""
    subdirs = ["app", "core", "ensemble", "errors", "system"]
    for subdir in subdirs:
        (LOG_DIR / subdir).mkdir(parents=True, exist_ok=True)

class DelayedFileHandler(logging.FileHandler):
    """延遲檔案創建的 FileHandler"""
    def __init__(self, filename, mode='a', encoding=None, delay=True):
        # 使用 delay=True 來延遲檔案創建，直到第一次寫入
        super().__init__(filename, mode, encoding, delay=delay)

def build_logging_config(enable_file: bool = True) -> dict:
    """構建統一的日誌配置"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_level = _normalize_log_level(_env_log_level("INFO"))
    
    # 基本 handlers（總是啟用）
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": log_level,
        }
    }
    
    if enable_file:
        ensure_log_dirs()
        
        # 檔案 handlers - 使用延遲創建
        handlers.update({
            "app_file": {
                "class": "analysis.logging_config.DelayedFileHandler",
                "formatter": "detailed",
                "filename": str((LOG_DIR / "app" / f"app_{timestamp}.log").resolve()),
                "encoding": "utf-8",
                "level": log_level,
                "mode": "w",
                "delay": True,
            },
            "core_file": {
                "class": "analysis.logging_config.DelayedFileHandler",
                "formatter": "detailed",
                "filename": str((LOG_DIR / "core" / f"core_{timestamp}.log").resolve()),
                "encoding": "utf-8",
                "level": log_level,
                "mode": "w",
                "delay": True,
            },
            "ensemble_file": {
                "class": "analysis.logging_config.DelayedFileHandler",
                "formatter": "detailed",
                "filename": str((LOG_DIR / "ensemble" / f"ensemble_{timestamp}.log").resolve()),
                "encoding": "utf-8",
                "level": log_level,
                "mode": "w",
                "delay": True,
            },
            "error_file": {
                "class": "analysis.logging_config.DelayedFileHandler",
                "formatter": "detailed",
                "filename": str((LOG_DIR / "errors" / f"errors_{timestamp}.log").resolve()),
                "encoding": "utf-8",
                "level": "ERROR",
                "mode": "w",
                "delay": True,
            },
            "system_file": {
                "class": "analysis.logging_config.DelayedFileHandler",
                "formatter": "detailed",
                "filename": str((LOG_DIR / "system" / f"system_{timestamp}.log").resolve()),
                "encoding": "utf-8",
                "level": log_level,
                "mode": "w",
                "delay": True,
            }
        })
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": handlers,
        "loggers": {
            "SSS.App": {
                "handlers": ["console"] + (["app_file", "error_file"] if enable_file else []),
                "level": log_level,
                "propagate": False,
            },
            "SSS.Core": {
                "handlers": ["console"] + (["core_file", "error_file"] if enable_file else []),
                "level": log_level,
                "propagate": False,
            },
            "SSS.Ensemble": {
                "handlers": ["console"] + (["ensemble_file", "error_file"] if enable_file else []),
                "level": log_level,
                "propagate": False,
            },
            "SSS.System": {
                "handlers": ["console"] + (["system_file", "error_file"] if enable_file else []),
                "level": log_level,
                "propagate": False,
            }
        },
        "root": {
            "handlers": ["console"] + (["system_file"] if enable_file else []),
            "level": log_level,
        }
    }

def init_logging(enable_file: bool = None) -> None:
    """初始化日誌系統"""
    global _logging_initialized, _file_logging_enabled
    
    # 如果沒有指定，檢查環境變數
    if enable_file is None:
        enable_file = os.environ.get("SSS_CREATE_LOGS", "0") == "1"
    
    # 記錄設定狀態
    _file_logging_enabled = enable_file
    
    # 清除現有的 logger 配置
    for logger_name in ["SSS.App", "SSS.Core", "SSS.Ensemble", "SSS.System"]:
        existing_logger = logging.getLogger(logger_name)
        existing_logger.handlers.clear()
        existing_logger.propagate = False
    
    # 應用新配置
    config_dict = build_logging_config(enable_file)
    logging.config.dictConfig(config_dict)
    
    # 標記為已初始化
    _logging_initialized = True
    
    # 驗證配置（僅在啟用檔案日誌時輸出詳細訊息）
    if enable_file:
        system_logger = logging.getLogger("SSS.System")
        system_logger.info("=== 統一日誌系統初始化完成 ===")
        system_logger.info(f"檔案日誌啟用: {enable_file}")
        system_logger.info(f"日誌目錄: {LOG_DIR.resolve()}")
        
        # 強制刷新所有 handlers
        for handler in system_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

def get_logger(name: str) -> logging.Logger:
    """獲取配置好的日誌器（真正的懶加載）"""
    global _logging_initialized
    
    # 如果還沒初始化，僅返回基本的 logger（不創建檔案）
    if not _logging_initialized:
        # 設定基本的控制台日誌格式
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            level = _normalize_log_level(_env_log_level("INFO"))
            logger.setLevel(getattr(logging, level, logging.INFO))
            logger.propagate = False
        return logger
    
    return logging.getLogger(name)

def ensure_logging_ready(logger_name: str = None) -> None:
    """確保日誌系統已準備就緒，僅在需要時初始化"""
    global _logging_initialized, _file_logging_enabled
    
    if not _logging_initialized:
        # 檢查環境變數決定是否啟用檔案日誌
        enable_file = os.environ.get("SSS_CREATE_LOGS", "0") == "1"
        init_logging(enable_file=enable_file)

def setup_logging():
    """設置日誌系統（向後兼容）"""
    init_logging(False)

def setup_module_logging(module_name: str, level: str = "INFO") -> logging.Logger:
    """設置模組日誌（向後兼容）"""
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, level.upper()))
    return logger
