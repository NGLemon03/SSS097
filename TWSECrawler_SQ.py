import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# 設定日誌輸出格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


class TWSECrawler:
    MI_INDEX_MIN_DATE = datetime(2004, 2, 11)
    PROGRESS_KEY_FMTQIK = "FMTQIK"
    PROGRESS_KEY_MI_INDEX = "MI_INDEX"

    def __init__(
        self,
        db_path=None,
        min_delay=4.0,
        max_delay=7.0,
        request_timeout=10,
        max_retries=3,
        retry_sleep_base=5,
    ):
        base_dir = Path(__file__).resolve().parent
        default_db = base_dir / "data" / "twse_data.db"
        legacy_db = base_dir / "twse_data.db"

        if db_path is None:
            # Prefer data/twse_data.db, keep legacy fallback for compatibility.
            db_path = default_db if default_db.exists() else (legacy_db if legacy_db.exists() else default_db)
        else:
            db_path = Path(db_path)
            if not db_path.is_absolute():
                db_path = (base_dir / db_path).resolve()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_sleep_base = retry_sleep_base
        self.session = requests.Session()
        # 偽裝一般瀏覽器，降低被阻擋機率
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 資料庫與資料表"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # 建立 FMTQIK (大盤與加權指數) 資料表，欄位已解構
            c.execute(
                """CREATE TABLE IF NOT EXISTS fmtqik (
                            date TEXT PRIMARY KEY,
                            trade_volume TEXT,
                            trade_value TEXT,
                            transaction_count TEXT,
                            taiex TEXT,
                            change TEXT
                        )"""
            )
            # 建立 FMTQIK 爬取月份紀錄表 (用來判斷哪些月份已經抓完，避免重複抓)
            c.execute(
                """CREATE TABLE IF NOT EXISTS fmtqik_months (
                            month TEXT PRIMARY KEY,
                            fetched_at TEXT
                        )"""
            )
            # 建立 MI_INDEX (每日所有資訊) 資料表，直接儲存原始 JSON 確保所有額外資訊都在
            c.execute(
                """CREATE TABLE IF NOT EXISTS mi_index (
                            date TEXT PRIMARY KEY,
                            has_data BOOLEAN,
                            raw_json TEXT
                        )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS crawler_progress (
                            crawler TEXT PRIMARY KEY,
                            last_download_date TEXT,
                            updated_at TEXT
                        )"""
            )
            conn.commit()

    @staticmethod
    def _next_month_start(date_obj):
        return (date_obj.replace(day=28) + timedelta(days=4)).replace(day=1)

    @staticmethod
    def _is_terminal_no_data(stat_text):
        if not stat_text:
            return False
        return any(k in stat_text for k in ("沒有符合條件", "查無資料", "無資料"))

    @staticmethod
    def _parse_ymd(date_text):
        try:
            return datetime.strptime(date_text, "%Y-%m-%d")
        except (TypeError, ValueError):
            return None

    def _get_last_download_date(self, cursor, crawler_key):
        cursor.execute("SELECT last_download_date FROM crawler_progress WHERE crawler=?", (crawler_key,))
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        return self._parse_ymd(row[0])

    def _set_last_download_date(self, cursor, crawler_key, date_obj):
        if date_obj is None:
            return
        date_str = date_obj.strftime("%Y-%m-%d")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """INSERT INTO crawler_progress (crawler, last_download_date, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(crawler) DO UPDATE SET
                 last_download_date=excluded.last_download_date,
                 updated_at=excluded.updated_at""",
            (crawler_key, date_str, now_str),
        )

    def _get_max_table_date(self, cursor, table_name):
        if table_name not in {"fmtqik", "mi_index"}:
            return None
        cursor.execute(f"SELECT MAX(date) FROM {table_name}")
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        return self._parse_ymd(row[0])

    def _request_json(self, url, tag):
        """有界重試的 HTTP JSON 請求，避免無限重試卡住。"""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.request_timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                LOGGER.warning("[%s] 第 %d/%d 次請求失敗: %s", tag, attempt, self.max_retries, e)
            except ValueError as e:
                LOGGER.warning("[%s] 第 %d/%d 次 JSON 解析失敗: %s", tag, attempt, self.max_retries, e)

            if attempt < self.max_retries:
                sleep_s = self.retry_sleep_base * attempt
                LOGGER.info("[%s] %d 秒後重試...", tag, sleep_s)
                time.sleep(sleep_s)

        LOGGER.error("[%s] 連續失敗 %d 次，放棄本次請求。", tag, self.max_retries)
        return None

    def delay(self):
        """隨機延遲，避免被 TWSE 鎖 IP"""
        sleep_time = random.uniform(self.min_delay, self.max_delay)
        time.sleep(sleep_time)

    def fetch_fmtqik(self, start_year=2000):
        """爬取每月的大盤總結資料 (FMTQIK)"""
        today = datetime.now()
        requested_start = datetime(start_year, 1, 1)

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            current_date = requested_start
            last_download_date = self._get_last_download_date(c, self.PROGRESS_KEY_FMTQIK)
            if last_download_date is None:
                last_download_date = self._get_max_table_date(c, "fmtqik")
                if last_download_date is not None:
                    self._set_last_download_date(c, self.PROGRESS_KEY_FMTQIK, last_download_date)
                    conn.commit()
            if last_download_date and last_download_date > requested_start:
                current_date = last_download_date
                LOGGER.info("[FMTQIK] 從上次最後下載日期續抓: %s", last_download_date.strftime("%Y-%m-%d"))

            while current_date <= today:
                month_str = current_date.strftime("%Y%m")
                query_date_str = current_date.strftime("%Y%m%d")
                is_current_month = current_date.year == today.year and current_date.month == today.month

                # 如果不是當月，且資料庫已經標記抓過了，就直接跳過 (續傳機制)
                if not is_current_month:
                    c.execute("SELECT 1 FROM fmtqik_months WHERE month=?", (month_str,))
                    if c.fetchone():
                        current_date = self._next_month_start(current_date)
                        continue

                LOGGER.info("[FMTQIK] 正在抓取 %s 月份資料...", month_str)
                url = (
                    "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
                    f"?date={query_date_str}&response=json"
                )
                data = self._request_json(url, tag=f"FMTQIK {month_str}")
                fetch_ok = False

                if data is not None:
                    stat = str(data.get("stat", "")).strip()
                    if stat == "OK":
                        latest_row_date = None
                        for row in data.get("data", []):
                            if len(row) < 6:
                                LOGGER.warning("[FMTQIK] %s 存在異常資料列: %s", month_str, row)
                                continue

                            try:
                                # 將民國年 (例如 89/01/04) 轉為西元年 (2000-01-04)
                                tw_date = str(row[0]).split("/")
                                if len(tw_date) != 3:
                                    raise ValueError(f"日期格式錯誤: {row[0]}")
                                date_obj = datetime(
                                    int(tw_date[0]) + 1911,
                                    int(tw_date[1]),
                                    int(tw_date[2]),
                                )
                                gregorian_date = date_obj.strftime("%Y-%m-%d")
                                if latest_row_date is None or date_obj > latest_row_date:
                                    latest_row_date = date_obj
                            except Exception as parse_err:
                                LOGGER.warning(
                                    "[FMTQIK] %s 日期解析失敗 (%s): %s", month_str, row[0], parse_err
                                )
                                continue

                            c.execute(
                                """INSERT OR REPLACE INTO fmtqik
                                   (date, trade_volume, trade_value, transaction_count, taiex, change)
                                   VALUES (?, ?, ?, ?, ?, ?)""",
                                (gregorian_date, row[1], row[2], row[3], row[4], row[5]),
                            )

                        # 只有真正成功時，才標記該月已抓取
                        c.execute(
                            "INSERT OR REPLACE INTO fmtqik_months (month, fetched_at) VALUES (?, ?)",
                            (month_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        )
                        progress_date = latest_row_date if latest_row_date is not None else current_date
                        self._set_last_download_date(c, self.PROGRESS_KEY_FMTQIK, progress_date)
                        conn.commit()
                        fetch_ok = True
                    else:
                        LOGGER.warning("[FMTQIK] %s 回傳非 OK 狀態: %s", month_str, stat)

                if not fetch_ok:
                    LOGGER.error("[FMTQIK] %s 本次不標記完成，待下次補抓。", month_str)

                self.delay()
                current_date = self._next_month_start(current_date)

    def fetch_mi_index(self, start_year=2000):
        """爬取每日詳細收盤行情與所有額外資訊 (MI_INDEX)"""
        today = datetime.now()
        requested_start = datetime(start_year, 1, 1)
        current_date = max(requested_start, self.MI_INDEX_MIN_DATE)

        if requested_start < self.MI_INDEX_MIN_DATE:
            LOGGER.info(
                "[MI_INDEX] 起始日期早於 TWSE 可查下限，調整為 %s",
                self.MI_INDEX_MIN_DATE.strftime("%Y-%m-%d"),
            )

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            last_download_date = self._get_last_download_date(c, self.PROGRESS_KEY_MI_INDEX)
            if last_download_date is None:
                last_download_date = self._get_max_table_date(c, "mi_index")
                if last_download_date is not None:
                    self._set_last_download_date(c, self.PROGRESS_KEY_MI_INDEX, last_download_date)
                    conn.commit()
            if last_download_date and last_download_date > current_date:
                current_date = last_download_date
                LOGGER.info("[MI_INDEX] 從上次最後下載日期續抓: %s", last_download_date.strftime("%Y-%m-%d"))

            while current_date <= today:
                # 六日不開盤，直接跳過減少不必要的請求
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue

                date_str_db = current_date.strftime("%Y-%m-%d")
                date_str_query = current_date.strftime("%Y%m%d")

                # 如果資料庫已經有這一天的紀錄 (無論當天是否有資料)，就跳過 (續傳/每日維護機制)
                # 除非該日期是「今天」，我們可能需要覆寫最新資料
                if current_date.date() != today.date():
                    c.execute("SELECT 1 FROM mi_index WHERE date=?", (date_str_db,))
                    if c.fetchone():
                        current_date += timedelta(days=1)
                        continue

                LOGGER.info("[MI_INDEX] 正在抓取 %s 每日資料...", date_str_db)
                url = (
                    "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
                    f"?date={date_str_query}&response=json"
                )
                data = self._request_json(url, tag=f"MI_INDEX {date_str_db}")

                if data is None:
                    LOGGER.error("[MI_INDEX] %s 連續失敗，略過本日待下次補抓。", date_str_db)
                    self.delay()
                    current_date += timedelta(days=1)
                    continue

                stat = str(data.get("stat", "")).strip()
                if stat == "OK":
                    has_data = True
                elif self._is_terminal_no_data(stat):
                    has_data = False
                else:
                    # 非終態錯誤(例如暫時性服務異常)不寫入，避免把失敗固化成「已抓取」
                    LOGGER.warning(
                        "[MI_INDEX] %s 回傳非終態狀態: %s，待下次補抓。",
                        date_str_db,
                        stat,
                    )
                    self.delay()
                    current_date += timedelta(days=1)
                    continue

                raw_json_str = json.dumps(data, ensure_ascii=False)
                c.execute(
                    """INSERT OR REPLACE INTO mi_index (date, has_data, raw_json)
                       VALUES (?, ?, ?)""",
                    (date_str_db, has_data, raw_json_str),
                )
                self._set_last_download_date(c, self.PROGRESS_KEY_MI_INDEX, current_date)
                conn.commit()

                self.delay()
                current_date += timedelta(days=1)


if __name__ == "__main__":
    crawler = TWSECrawler()

    # 1. 先抓取大盤月度摘要 (速度快)
    LOGGER.info("=== 開始執行 FMTQIK (大盤指數) 爬蟲 ===")
    crawler.fetch_fmtqik(start_year=2000)

    # 2. 再抓取每日詳細資訊 (速度較慢，初次執行需時數小時)
    LOGGER.info("=== 開始執行 MI_INDEX (每日所有詳細資料) 爬蟲 ===")
    crawler.fetch_mi_index(start_year=2000)

    LOGGER.info("=== 所有爬蟲任務完成！ ===")
