核心目標

確保跨平台（特別是 Windows）環境下，繁體中文路徑與檔案內容（CSV/TXT）不產生亂碼。
1. 編碼規範 (Encoding)

    原始碼檔案： 所有 .py 檔案必須存為 UTF-8 (無 BOM)。

    註解： 保持現有註解，中文註解必須維持 UTF-8 編碼。

    現有檔案安全： 除非明確要求，否則禁止變更現有檔案的編碼格式（防止 UTF-8 被誤轉為 ANSI）。

2. 檔案路徑 (File Paths)

    路徑類型： 必須支援非 ASCII 字元（如：繁體中文路徑）。

    路徑處理： 優先使用 pathlib.Path 或確保使用 Python 3 str 路徑。
    Python

    from pathlib import Path
    data_path = Path("數據/策略輸出.csv")

3. 檔案 I/O 操作 (File Read/Write)

    顯式編碼： 所有文字檔案讀寫操作 必須 顯式指定 encoding="utf-8"。

        禁止：open(file, "r")

        正確：open(file, "r", encoding="utf-8")

4. CSV 輸出與 Excel 相容 (CSV Output)

    Excel 相容性： 所有預計由 Excel 開啟的 CSV 檔案，必須 使用 utf-8-sig 編碼（加入 BOM 標記）。
    Python

    # Pandas 範例
    df.to_csv(path, index=False, encoding="utf-8-sig")

    # 標準庫範例
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        # writer logic...

5. 代碼修改規則 (Modification Logic)

    最小改動原則： 除非明確要求重構，否則僅針對特定區塊進行 Search → Replace 修改。

    一致性： 修改時需觀察檔案現有的縮進（Space vs Tab）並嚴格遵循。
