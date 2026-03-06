# -*- coding: utf-8 -*-
"""
臨時腳本：清理 SSSv096.py 中的重複代碼
"""
import re

input_file = "SSSv096.py"
output_file = "SSSv096_cleaned.py"

# 讀取原始文件內容
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 標記需要刪除的區域
# 從 "from functools import wraps" 之後，到 "def compute_backtest_for_periods" 之前
start_marker = None
end_marker = None

for i, line in enumerate(lines):
    if 'from functools import wraps' in line and start_marker is None:
        start_marker = i + 1  # 從下一行開始刪除

    # 找到第一個未被破壞的 compute_backtest_for_periods 定義
    # （完整的參數列表應該有 'smaa_source' 參數）
    if ('def compute_backtest_for_periods(' in line and
        'smaa_source' in ''.join(lines[i:min(i+5, len(lines))])):
        end_marker = i
        break

print(f"檢測到重複代碼區域: Line {start_marker+1} 到 Line {end_marker}")

if start_marker is not None and end_marker is not None:
    # 構建新文件內容
    new_lines = []

    # 保留開頭部分
    new_lines.extend(lines[:start_marker])

    # 插入註釋標記
    new_lines.append("\n")
    new_lines.append("# " + "="*79 + "\n")
    new_lines.append("# ⚠️ 重複代碼已移除 ⚠️\n")
    new_lines.append("# 以下函數已從 sss_core.logic 導入（見文件頂部）:\n")
    new_lines.append("# - TradeSignal, validate_params, load_data, calc_smaa, linreg_last_vectorized\n")
    new_lines.append("# - compute_single, compute_dual, compute_RMA, compute_ssma_turn_combined\n")
    new_lines.append("# - backtest_unified, calculate_metrics, calculate_holding_periods, calculate_atr\n")
    new_lines.append("#\n")
    new_lines.append(f"# 原始代碼約 {end_marker - start_marker} 行已刪除，如需恢復請使用版本控制系統\n")
    new_lines.append("# " + "="*79 + "\n")
    new_lines.append("\n")
    new_lines.append("# --- SSSv096 專用的輔助函數 ---\n")
    new_lines.append("\n")

    # 保留 compute_backtest_for_periods 及之後的內容
    new_lines.extend(lines[end_marker:])

    # 寫入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"清理完成！")
    print(f"刪除了 {end_marker - start_marker} 行重複代碼")
    print(f"新文件已保存到: {output_file}")
    print(f"請檢查新文件無誤後，執行: mv {output_file} {input_file}")
else:
    print("未找到標記位置，清理失敗")
