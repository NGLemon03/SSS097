# sss_core/data_utils.py
"""
資料序列化工具 - 健壯的 DataFrame/Series 打包/解包
=================================================

解決問題：
1. JSON 轉換會丟失型別資訊 (datetime -> str, int -> float)
2. 編碼問題 (Windows cp950 vs UTF-8)
3. 傳輸體積過大

使用 Pickle + Gzip + Base64 方案：
- 完美保留 Pandas 型別
- 壓縮率 30-50%
- 避免編碼問題
"""
import pickle
import base64
import gzip
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def pack_df_robust(df: pd.DataFrame) -> str:
    """
    健壯的 DataFrame 序列化

    Args:
        df: 要序列化的 DataFrame

    Returns:
        Base64 編碼的壓縮字串（空 DataFrame 返回空字串）

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> packed = pack_df_robust(df)
        >>> unpacked = unpack_df_robust(packed)
        >>> assert df.equals(unpacked)
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ""

    try:
        # 使用最高協議的 Pickle 序列化
        pickled = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)

        # Gzip 壓縮（壓縮率 30-50%）
        compressed = gzip.compress(pickled, compresslevel=6)

        # Base64 編碼為 ASCII 字串（安全傳輸）
        encoded = base64.b64encode(compressed).decode('ascii')

        return encoded

    except Exception as e:
        logger.error(f"❌ 打包 DataFrame 失敗: {e}\n形狀: {df.shape if hasattr(df, 'shape') else 'N/A'}")
        return ""


def unpack_df_robust(data) -> Optional[pd.DataFrame]:
    """
    健壯的 DataFrame 反序列化（支援自動格式偵測）

    Args:
        data: pack_df_robust 產生的字串 或 舊的 JSON 字串

    Returns:
        還原的 DataFrame（失敗時返回空 DataFrame）

    Examples:
        >>> packed = "..."  # 來自 pack_df_robust
        >>> df = unpack_df_robust(packed)
        >>> print(df.head())
    """
    # Avoid ambiguous truth checks on pandas objects.
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not isinstance(data, str):
        return pd.DataFrame()
    if data == "":
        return pd.DataFrame()

    # 🔑 自動格式偵測：嘗試新格式 (Pickle)，失敗則退回舊格式 (JSON)
    try:
        # Base64 解碼
        decoded = base64.b64decode(data.encode('ascii'))

        # Gzip 解壓縮
        decompressed = gzip.decompress(decoded)

        # Pickle 反序列化
        df = pickle.loads(decompressed)

        return df

    except (base64.binascii.Error, gzip.BadGzipFile, pickle.UnpicklingError, ValueError) as e:
        # 這些錯誤表示不是新格式，嘗試舊格式 (JSON)
        logger.debug(f"[Fallback] 偵測到舊格式資料，使用 JSON 解析（錯誤: {type(e).__name__}）")

        try:
            import io
            # 先嘗試 split orient（常見格式）
            for orient in ("split", None):
                try:
                    kw = {"orient": orient} if orient else {}
                    return pd.read_json(io.StringIO(data), **kw)
                except Exception:
                    pass

            # 最後嘗試直接從 list/dict 建立
            if data.startswith('[') or data.startswith('{'):
                import json
                parsed = json.loads(data)
                return pd.DataFrame(parsed)

        except Exception as json_error:
            logger.error(f"❌ 解包 DataFrame 失敗（新舊格式均失敗）: Pickle錯誤={e}, JSON錯誤={json_error}")

        return pd.DataFrame()

    except Exception as e:
        logger.error(f"❌ 解包 DataFrame 失敗（未知錯誤）: {e}\n資料長度: {len(data) if data else 0}")
        return pd.DataFrame()


def pack_series_robust(series: pd.Series) -> str:
    """
    健壯的 Series 序列化

    Args:
        series: 要序列化的 Series

    Returns:
        Base64 編碼的壓縮字串
    """
    if series is None or (isinstance(series, pd.Series) and series.empty):
        return ""

    try:
        pickled = pickle.dumps(series, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(pickled, compresslevel=6)
        encoded = base64.b64encode(compressed).decode('ascii')
        return encoded

    except Exception as e:
        logger.error(f"❌ 打包 Series 失敗: {e}\n長度: {len(series) if hasattr(series, '__len__') else 'N/A'}")
        return ""


def unpack_series_robust(data) -> Optional[pd.Series]:
    """
    健壯的 Series 反序列化（支援自動格式偵測）

    Args:
        data: pack_series_robust 產生的字串 或 舊的 JSON 字串

    Returns:
        還原的 Series（失敗時返回空 Series）
    """
    if data is None:
        return pd.Series(dtype=float)
    if isinstance(data, pd.Series):
        return data.copy()
    if not isinstance(data, str):
        return pd.Series(dtype=float)
    if data == "":
        return pd.Series(dtype=float)

    # 🔑 自動格式偵測：嘗試新格式 (Pickle)，失敗則退回舊格式 (JSON)
    try:
        decoded = base64.b64decode(data.encode('ascii'))
        decompressed = gzip.decompress(decoded)
        series = pickle.loads(decompressed)
        return series

    except (base64.binascii.Error, gzip.BadGzipFile, pickle.UnpicklingError, ValueError) as e:
        # 這些錯誤表示不是新格式，嘗試舊格式 (JSON)
        logger.debug(f"[Fallback] 偵測到舊格式資料，使用 JSON 解析（錯誤: {type(e).__name__}）")

        try:
            import io
            # 嘗試 JSON 解析
            return pd.read_json(io.StringIO(data), typ='series')

        except Exception as json_error:
            logger.error(f"❌ 解包 Series 失敗（新舊格式均失敗）: Pickle錯誤={e}, JSON錯誤={json_error}")

        return pd.Series(dtype=float)

    except Exception as e:
        logger.error(f"❌ 解包 Series 失敗（未知錯誤）: {e}\n資料長度: {len(data) if data else 0}")
        return pd.Series(dtype=float)


# 向後相容：提供舊函式名稱的別名
pack_df = pack_df_robust
df_from_pack = unpack_df_robust
pack_series = pack_series_robust
series_from_pack = unpack_series_robust


if __name__ == "__main__":
    # 簡單測試（Windows cp950 相容）
    import numpy as np

    print("[TEST] DataFrame serialization test...")
    df_test = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    df_test.set_index('date', inplace=True)

    # 打包
    packed = pack_df_robust(df_test)
    print(f"[OK] Packed successfully, size: {len(packed)} bytes")

    # 解包
    unpacked = unpack_df_robust(packed)
    print(f"[OK] Unpacked successfully, shape: {unpacked.shape}")

    # 驗證
    assert df_test.equals(unpacked), "[ERROR] Data mismatch!"
    print("[OK] All tests passed!")

    # 壓縮率測試
    import sys
    original_size = sys.getsizeof(df_test)
    packed_size = len(packed)
    ratio = (1 - packed_size / original_size) * 100
    print(f"[INFO] Compression ratio: {ratio:.1f}%")
