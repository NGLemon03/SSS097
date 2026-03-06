# analysis/config.py
"""
股票交易策略配置檔案

功能:
    - 定義數據路徑、交易成本、參數範圍等
    - 提供壓力測試時段與參數網格

修改記錄：2025-01-12 - 移除 import 時自動建立目錄的副作用，改為按需建立 API
"""

import os
from pathlib import Path
import json
import pickle
import itertools
from joblib import Memory
import datetime as dt

# 根目錄與子目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULT_DIR = PROJECT_ROOT / "results"
PLOT_DIR = PROJECT_ROOT / "plots"
GRIDS_DIR = PROJECT_ROOT / "analysis" / "grids"
PRESETS_DIR = PROJECT_ROOT / "analysis" / "presets"
SMAA_CACHE_DIR = CACHE_DIR / "cache_smaa"

# 移除自動建立目錄的迴圈，改為按需建立
# 預設只讓 data、cache 自動建立，其它一律按需
_DEFAULT_AUTO_CREATE = {"data", "cache"}

def _want(name: str) -> bool:
    """環境變數精細控制：SSS_CREATE_<NAME>=1/0；未設定時只開 data, cache。"""
    v = os.getenv(f"SSS_CREATE_{name.upper()}", "").lower()
    if v in ("1", "true", "yes"): return True
    if v in ("0", "false", "no"): return False
    return name in _DEFAULT_AUTO_CREATE

def ensure_dir(p: Path, *, force: bool | None = None) -> Path:
    """必要時才建立資料夾；force=True/False 可覆寫環境變數/預設。"""
    name_map = {
        DATA_DIR: "data", CACHE_DIR: "cache", RESULT_DIR: "results",
        PLOT_DIR: "plots", LOG_DIR: "log"
    }
    name = name_map.get(p, None)
    do_create = _want(name) if force is None else force
    if do_create:
        p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_parent(file_path: Path) -> None:
    """寫檔前用這個確保父目錄存在（更精準、不會亂建沒用到的夾）。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

# 快取設定 - 只在需要時建立目錄
MEMORY = Memory(CACHE_DIR / "joblib", verbose=0, mmap_mode=None)

# 交易成本與資本設定
TICKER = "00631L.TW"  # 股票代號
START_DATE = "2014-10-23" # 回測起始日期
BUY_FEE = 0.001425 * 0.3 # 買入手續費 (0.1425% * 0.3)
SELL_FEE = (0.001425 + 0.003) * 0.3 # 賣出手續費 (0.1425% + 0.3% * 0.3)
COST = BUY_FEE + SELL_FEE # 交易成本
COST_DEFAULT = 0.005 # 預設交易成本
SLIP_PCT = 0.002 # 滑價百分比 (0.2%)
INITIAL_CAPITAL = 3_000_000 # 初始資本

# 冷卻期與參數
TRADE_COOLDOWN_BARS = 3  # 回測交易冷卻期 (bars)
SIGNAL_COOLDOWN_DAYS = 3  # 峰谷檢測冷卻期 (天)
FACTOR = 40 # 預設因子值

# 資料來源
SOURCES = [
    "Self",
    "Factor (^TWII / 2412.TW)",
    "Factor (^TWII / 2414.TW)"
]

# 並行與平移測試
N_JOBS = os.cpu_count()
SHIFTS = [-10, -5, -2, 0, 2, 5, 10]



STRATEGY_PARAMS = {
    'single': {
        'ind_keys': ['linlen', 'factor', 'smaalen', 'devwin'],
        'bt_keys': ['buy_mult', 'sell_mult', 'stop_loss'],
        'compute_func': 'compute_single'
    },
    'dual': {
        'ind_keys': ['linlen', 'factor', 'smaalen', 'short_win', 'long_win'],
        'bt_keys': ['buy_mult', 'sell_mult', 'stop_loss'],
        'compute_func': 'compute_dual'
    },
    'RMA': {
        'ind_keys': ['linlen', 'factor', 'smaalen', 'rma_len', 'dev_len'],
        'bt_keys': ['buy_mult', 'sell_mult', 'stop_loss'],
        'compute_func': 'compute_RMA'
    },
    'ssma_turn': {
        'ind_keys': ['linlen', 'factor', 'smaalen', 'prom_factor', 'min_dist', 'buy_shift', 'exit_shift', 'vol_window', 'signal_cooldown_days', 'quantile_win'],
        'bt_keys': ['buy_mult', 'sell_mult', 'stop_loss'],
        'compute_func': 'compute_ssma_turn_combined'
    }
}

# 走查期間
WF_PERIODS = [
    #{"test": ("2010-06-01", "2014-02-25")},
    {"test": ("2014-02-26", "2017-11-22")},
    {"test": ("2017-11-23", "2021-08-18")},
    {"test": ("2021-08-19", "2025-06-06")}
]
# 壓力測試時段
STRESS_PERIODS = [
    #("1990-02-12", "1990-10-01"),  # 證交稅實施：12682 → 2551 (-10131 點, -79.88%, 232天)
    #("1995-01-05", "1995-08-15"),  # 中共飛彈試射：7144 → 4474 (-2670 點, -37.37%, 222天)
    #("1997-08-27", "1999-02-05"),  # 亞洲金融風暴：10256 → 5474 (-4782 點, -46.63%, 528天)
    #("2000-02-05", "2001-09-20"),  # 網路泡沫化：10202 → 3446 (-6756 點, -66.22%, 593天)
    #("2003-01-24", "2003-04-28"),  # SARS 疫情：5141 → 4044 (-1097 點, -21.34%, 94天)
    #("2004-03-19", "2004-08-05"),  # 陳水扁 319 槍擊案：6833 → 5355 (-1478 點, -21.63%, 139天)
    #("2008-05-20", "2008-11-21"),  # 金融海嘯：9309 → 3955 (-5354 點, -57.51%, 185天)
    #("2011-02-08", "2011-12-19"),  # 美債危機：9220 → 6609 (-2611 點, -28.32%, 314天)
    ("2015-04-28", "2015-08-24"),  # 中國股災：10014 → 7203 (-2811 點, -28.07%, 119天)
    ("2018-10-02", "2019-01-04"),  # 中美貿易戰：11064 → 9319 (-1745 點, -15.77%, 94天)
    ("2020-01-20", "2020-03-19"),  # COVID-19：12151 → 8523 (-3628 點, -29.86%, 59天)
    ("2022-01-05", "2022-10-26"),  # FED 升息停 QE：18619 → 12635 (-5984 點, -32.14%, 294天)
    ("2024-07-11", "2024-08-06"),  # 日本升息：24416.67 → 19662.74 (-4753.93 點, -24.17%, 26天)
    ("2025-04-02", "2025-04-09"),  # 川普關稅政策：21298.22 → 17391.76 (-3906.46 點, -18.34%, 7天)
]

def build_product(grid: dict):
    """
    生成參數網格的所有組合
    """
    keys, vals = zip(*grid.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def load_grid_params(grid_level: str):
    """
    加載並緩存網格參數
    """
    cache_file = CACHE_DIR / f"{grid_level}_params.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    grid_file = GRIDS_DIR / f"{grid_level}.json"
    with open(grid_file, "r", encoding="utf-8") as f:
        grid_dict = json.load(f)

    params_list = { strat: list(build_product(cfg)) for strat, cfg in grid_dict.items() }

    with open(cache_file, "wb") as f:
        pickle.dump(params_list, f)

    return params_list


# 默認網格級別
GRID_LEVEL = os.environ.get("GRID_LEVEL", "triple_full")
PR = load_grid_params(GRID_LEVEL)

def get_data_filename(ticker: str) -> str:
    """
    根據 Ticker 生成數據文件名
    """
    return f"{ticker.replace(':', '_')}_data_raw.csv"
