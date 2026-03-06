#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列出整個文件夾的架構和文件
排除以 . 開頭的隱藏文件夾
"""

import os
from pathlib import Path
import json
from datetime import datetime
import logging

# 設置 logger - 按需初始化
from analysis.logging_config import get_logger
logger = get_logger("SSS.System")

def get_file_size_str(size_bytes):
    """將字節數轉換為人類可讀的文件大小字符串"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    if i == 0:
        return f"{size_bytes:.0f} {size_names[i]}"
    else:
        return f"{size_bytes:.1f} {size_names[i]}"

def scan_directory(root_path, max_depth=10, current_depth=0):
    """掃描目錄結構"""
    if current_depth > max_depth:
        return None
    
    root_path = Path(root_path)
    if not root_path.exists():
        return None
    
    result = {
        "name": root_path.name,
        "path": str(root_path),
        "type": "directory",
        "size": 0,
        "items": [],
        "file_count": 0,
        "dir_count": 0
    }
    
    try:
        # 獲取目錄內容，排除隱藏文件夾
        items = [item for item in root_path.iterdir() 
                if not item.name.startswith('.')]
        
        # 分別處理文件和文件夾
        files = [item for item in items if item.is_file()]
        dirs = [item for item in items if item.is_dir()]
        
        # 統計文件數量
        result["file_count"] = len(files)
        result["dir_count"] = len(dirs)
        
        # 處理文件
        for file_path in sorted(files):
            try:
                file_stat = file_path.stat()
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "type": "file",
                    "size": file_stat.st_size,
                    "size_str": get_file_size_str(file_stat.st_size),
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "extension": file_path.suffix.lower()
                }
                result["items"].append(file_info)
                result["size"] += file_stat.st_size
            except (OSError, PermissionError):
                # 跳過無法訪問的文件
                continue
        
        # 處理文件夾（遞歸）
        for dir_path in sorted(dirs):
            try:
                dir_info = scan_directory(dir_path, max_depth, current_depth + 1)
                if dir_info:
                    result["items"].append(dir_info)
                    result["size"] += dir_info["size"]
            except (OSError, PermissionError):
                # 跳過無法訪問的文件夾
                continue
        
        # 添加總大小字符串
        result["size_str"] = get_file_size_str(result["size"])
        
    except (OSError, PermissionError):
        # 無法訪問目錄
        return None
    
    return result

def print_structure(data, indent=0, show_details=True):
    """打印目錄結構到控制枱"""
    if not data:
        return
    
    prefix = "  " * indent
    
    if data["type"] == "directory":
        logger.info(f"{prefix}📁 {data['name']}/")
        if show_details:
            logger.info(f"{prefix}   📊 文件: {data['file_count']}, 文件夾: {data['dir_count']}, 總大小: {data['size_str']}")
        
        # 遞歸打印子項目
        for item in data["items"]:
            print_structure(item, indent + 1, show_details)
    
    elif data["type"] == "file":
        if show_details:
                    logger.info(f"{prefix}📄 {data['name']} ({data['size_str']}, {data['modified']})")
    else:
        logger.info(f"{prefix}📄 {data['name']}")

def print_structure_to_list(data, output_lines, indent=0, show_details=True):
    """將目錄結構輸出到列表中"""
    if not data:
        return
    
    prefix = "  " * indent
    
    if data["type"] == "directory":
        output_lines.append(f"{prefix}📁 {data['name']}/")
        if show_details:
            output_lines.append(f"{prefix}   📊 文件: {data['file_count']}, 文件夾: {data['dir_count']}, 總大小: {data['size_str']}")
        
        # 遞歸處理子項目
        for item in data["items"]:
            print_structure_to_list(item, output_lines, indent + 1, show_details)
    
    elif data["type"] == "file":
        if show_details:
            output_lines.append(f"{prefix}📄 {data['name']} ({data['size_str']}, {data['modified']})")
        else:
            output_lines.append(f"{prefix}📄 {data['name']}")

def generate_summary(data):
    """生成統計摘要"""
    if not data:
        return {}
    
    summary = {
        "total_files": 0,
        "total_dirs": 0,
        "total_size": 0,
        "file_extensions": {},
        "largest_files": [],
        "largest_dirs": []
    }
    
    def collect_stats(item):
        if item["type"] == "file":
            summary["total_files"] += 1
            summary["total_size"] += item["size"]
            
            # 統計文件擴展名
            ext = item["extension"]
            if ext:
                summary["file_extensions"][ext] = summary["file_extensions"].get(ext, 0) + 1
            
            # 記錄大文件
            summary["largest_files"].append({
                "name": item["name"],
                "path": item["path"],
                "size": item["size"],
                "size_str": item["size_str"]
            })
        
        elif item["type"] == "directory":
            summary["total_dirs"] += 1
            summary["total_size"] += item["size"]
            
            # 記錄大文件夾
            summary["largest_dirs"].append({
                "name": item["name"],
                "path": item["path"],
                "size": item["size"],
                "size_str": item["size_str"]
            })
            
            # 遞歸統計子項目
            for sub_item in item["items"]:
                collect_stats(sub_item)
    
    collect_stats(data)
    
    # 排序大文件和大文件夾
    summary["largest_files"].sort(key=lambda x: x["size"], reverse=True)
    summary["largest_dirs"].sort(key=lambda x: x["size"], reverse=True)
    
    return summary

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='列出文件夾結構')
    parser.add_argument('path', nargs='?', default='.', help='要掃描的路徑（默認為當前目錄）')
    parser.add_argument('--max-depth', type=int, default=10, help='最大掃描深度（默認10）')
    parser.add_argument('--no-details', action='store_true', help='不顯示詳細信息')
    parser.add_argument('--json', action='store_true', help='輸出JSON格式')
    parser.add_argument('--summary', action='store_true', help='只顯示統計摘要')
    parser.add_argument('--output', default='list.txt', help='輸出文件名（默認list.txt）')
    
    args = parser.parse_args()
    
    # 準備輸出內容
    output_lines = []
    
    output_lines.append(f"🔍 掃描目錄: {os.path.abspath(args.path)}")
    output_lines.append(f"📅 掃描時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("=" * 80)
    
    # 掃描目錄
    data = scan_directory(args.path, args.max_depth)
    
    if not data:
        output_lines.append("❌ 無法訪問指定目錄")
        # 保存到文件
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        logger.error("❌ 無法訪問指定目錄")
        return
    
    if args.json:
        # 輸出JSON格式
        json_output = json.dumps(data, indent=2, ensure_ascii=False)
        output_lines.append(json_output)
        logger.info(json_output)
    elif args.summary:
        # 只顯示統計摘要
        summary = generate_summary(data)
        output_lines.append("\n📊 統計摘要:")
        output_lines.append(f"總文件數: {summary['total_files']:,}")
        output_lines.append(f"總文件夾數: {summary['total_dirs']:,}")
        output_lines.append(f"總大小: {get_file_size_str(summary['total_size'])}")
        
        if summary['file_extensions']:
            output_lines.append("\n📁 文件類型分佈:")
            for ext, count in sorted(summary['file_extensions'].items(), key=lambda x: x[1], reverse=True):
                output_lines.append(f"  {ext}: {count:,} 個文件")
        
        if summary['largest_files']:
            output_lines.append("\n📄 最大的10個文件:")
            for i, file_info in enumerate(summary['largest_files'][:10], 1):
                output_lines.append(f"  {i:2d}. {file_info['name']} ({file_info['size_str']})")
        
        if summary['largest_dirs']:
            output_lines.append("\n📁 最大的10個文件夾:")
            for i, dir_info in enumerate(summary['largest_dirs'][:10], 1):
                output_lines.append(f"  {i:2d}. {dir_info['name']}/ ({dir_info['size_str']})")
        
        # 打印到控制枱
        logger.info('\n'.join(output_lines))
    
    else:
        # 顯示完整結構
        structure_lines = []
        print_structure_to_list(data, structure_lines, show_details=not args.no_details)
        output_lines.extend(structure_lines)
        
        # 顯示統計摘要
        summary = generate_summary(data)
        output_lines.append("\n" + "=" * 80)
        output_lines.append("📊 統計摘要:")
        output_lines.append(f"總文件數: {summary['total_files']:,}")
        output_lines.append(f"總文件夾數: {summary['total_dirs']:,}")
        output_lines.append(f"總大小: {get_file_size_str(summary['total_size'])}")
        
        if summary['file_extensions']:
            output_lines.append(f"文件類型數: {len(summary['file_extensions'])}")
        
        # 打印到控制枱
        logger.info('\n'.join(output_lines))
    
    # 保存到文件
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        logger.info(f"💾 結果已保存到: {args.output}")
    except Exception as e:
        logger.error(f"❌ 保存文件失敗: {e}")

if __name__ == "__main__":
    main()
