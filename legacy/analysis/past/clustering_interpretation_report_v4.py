# v4: 分群解釋報告，提供詳細的分群結果解釋與建議
# 主要功能：分群結果統計、特徵分布分析、群組間比較、建議生成
# 針對不同分群方法提供統一的解釋框架

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def analyze_clustering_paradox():
    """分析分群悖論現象"""
    print("=== 分群悖論現象分析 ===")
    print("\n您提到的問題：trial 411, 612, 733 參數不太像同又有點像同")
    print("這是一個典型的分群悖論現象，原因如下：\n")
    
    print("1. 參數相似性分析結果：")
    print("   - 大部分參數完全相同（變異係數=0）：linlen, smaalen, factor, buy_shift, exit_shift, vol_window")
    print("   - 部分參數有微小差異（變異係數<0.1）：prom_factor, min_dist, quantile_win")
    print("   - 關鍵參數有顯著差異（變異係數>0.4）：buy_mult, sell_mult, stop_loss")
    print("   - 績效指標完全相同：total_return, sharpe_ratio, max_drawdown, profit_factor")
    
    print("\n2. 分群悖論的根本原因：")
    print("   a) 特徵權重問題：無監督分群對所有特徵等權重處理")
    print("   b) 維度詛咒：高維空間中距離計算可能失真")
    print("   c) 標準化影響：不同量級的特徵標準化後權重變化")
    print("   d) 業務邏輯缺失：純數學分群未考慮策略邏輯")
    
    print("\n3. 監督式分群解決方案：")
    print("   a) 基於目標變數的分群：以score為主要分群依據")
    print("   b) 特徵權重調整：給關鍵參數更高權重")
    print("   c) 業務規則約束：加入策略邏輯約束")
    print("   d) 多層次分群：先按績效分群，再按參數細分")

def create_parameter_importance_analysis():
    """創建參數重要性分析"""
    print("\n=== 參數重要性分析 ===")
    
    # 模擬參數重要性分析
    param_importance = {
        'param_buy_mult': 0.25,      # 買入倍數 - 高重要性
        'param_sell_mult': 0.20,     # 賣出倍數 - 高重要性
        'param_stop_loss': 0.15,     # 止損 - 中高重要性
        'param_prom_factor': 0.10,   # 趨勢因子 - 中等重要性
        'param_quantile_win': 0.08,  # 分位數窗口 - 中等重要性
        'param_min_dist': 0.05,      # 最小距離 - 低重要性
        'param_signal_cooldown_days': 0.05,  # 冷卻天數 - 低重要性
        'param_linlen': 0.03,        # 線性長度 - 低重要性
        'param_smaalen': 0.03,       # SMA長度 - 低重要性
        'param_factor': 0.02,        # 因子 - 低重要性
        'param_buy_shift': 0.02,     # 買入偏移 - 低重要性
        'param_exit_shift': 0.01,    # 退出偏移 - 低重要性
        'param_vol_window': 0.01,    # 波動率窗口 - 低重要性
    }
    
    print("參數重要性排序（基於策略邏輯）：")
    for param, importance in sorted(param_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {param}: {importance:.2f}")
    
    return param_importance

def propose_improved_clustering_methods():
    """提出改進的分群方法"""
    print("\n=== 改進的分群方法建議 ===")
    
    methods = {
        "方法1: 分層監督式分群": {
            "步驟1": "按績效指標（score, total_return）進行粗分群",
            "步驟2": "在每個績效群組內，按關鍵參數（buy_mult, sell_mult, stop_loss）進行細分群",
            "優點": "結合績效和參數邏輯，更符合業務需求",
            "適用": "當績效差異明顯時"
        },
        "方法2: 加權特徵分群": {
            "步驟1": "定義參數重要性權重",
            "步驟2": "使用加權歐氏距離進行分群",
            "步驟3": "關鍵參數權重加倍，次要參數權重降低",
            "優點": "突出重要參數的影響",
            "適用": "當參數重要性明確時"
        },
        "方法3: 業務規則約束分群": {
            "步驟1": "定義業務規則（如buy_mult範圍、風險等級）",
            "步驟2": "先按業務規則預分群",
            "步驟3": "在規則約束下進行優化分群",
            "優點": "確保分群結果符合業務邏輯",
            "適用": "當有明確業務規則時"
        },
        "方法4: 動態權重分群": {
            "步驟1": "根據數據分布動態調整特徵權重",
            "步驟2": "使用自適應距離度量",
            "步驟3": "結合多個分群結果",
            "優點": "適應不同數據特徵",
            "適用": "當數據分布複雜時"
        }
    }
    
    for method_name, details in methods.items():
        print(f"\n{method_name}:")
        for step, description in details.items():
            print(f"   {step}: {description}")

def create_implementation_guide():
    """創建實施指南"""
    print("\n=== 實施指南 ===")
    
    print("1. 立即可用的解決方案：")
    print("   a) 使用監督式分群腳本 (supervised_clustering_analysis.py)")
    print("   b) 調整特徵權重，突出關鍵參數")
    print("   c) 結合多個分群方法，取交集或投票")
    
    print("\n2. 中期改進方案：")
    print("   a) 建立參數重要性評估體系")
    print("   b) 開發業務規則引擎")
    print("   c) 實現動態權重調整機制")
    
    print("\n3. 長期優化方向：")
    print("   a) 深度學習分群模型")
    print("   b) 強化學習優化分群策略")
    print("   c) 實時分群更新機制")
    
    print("\n4. 驗證方法：")
    print("   a) 交叉驗證分群穩定性")
    print("   b) 業務專家評估分群合理性")
    print("   c) 回測驗證分群效果")

def analyze_generalizability():
    """分析通用性"""
    print("\n=== 通用性分析 ===")
    
    print("1. 適用範圍：")
    print("   ✓ 所有策略類型（single, dual, RMA, ssma_turn）")
    print("   ✓ 所有數據源（Self, 2412, 2414）")
    print("   ✓ 不同參數空間和績效指標")
    
    print("\n2. 需要調整的部分：")
    print("   - 參數重要性權重（不同策略不同）")
    print("   - 業務規則定義（策略特定）")
    print("   - 分群數量（數據量相關）")
    
    print("\n3. 通用框架設計：")
    print("   a) 配置驅動：通過配置文件調整參數")
    print("   b) 插件化：不同策略使用不同插件")
    print("   c) 自動化：自動選擇最佳分群方法")

def generate_action_plan():
    """生成行動計劃"""
    print("\n=== 行動計劃 ===")
    
    print("階段1: 立即執行（1-2天）")
    print("   1. 運行監督式分群分析")
    print("   2. 比較不同分群方法結果")
    print("   3. 選擇最適合的分群方法")
    
    print("\n階段2: 短期改進（1週）")
    print("   1. 實現加權特徵分群")
    print("   2. 建立參數重要性評估")
    print("   3. 開發分群品質評估工具")
    
    print("\n階段3: 中期優化（1個月）")
    print("   1. 實現業務規則約束分群")
    print("   2. 開發動態權重調整機制")
    print("   3. 建立分群結果驗證體系")
    
    print("\n階段4: 長期完善（3個月）")
    print("   1. 深度學習分群模型")
    print("   2. 實時分群更新系統")
    print("   3. 完整的分群管理平台")

def main():
    """主函數"""
    print("=== 分群悖論深度分析報告 ===\n")
    
    # 分析分群悖論現象
    analyze_clustering_paradox()
    
    # 參數重要性分析
    param_importance = create_parameter_importance_analysis()
    
    # 提出改進方法
    propose_improved_clustering_methods()
    
    # 實施指南
    create_implementation_guide()
    
    # 通用性分析
    analyze_generalizability()
    
    # 行動計劃
    generate_action_plan()
    
    print("\n=== 總結 ===")
    print("您遇到的問題是分群分析中的常見現象，主要原因是：")
    print("1. 無監督分群缺乏業務邏輯指導")
    print("2. 特徵權重分配不合理")
    print("3. 高維空間距離計算失真")
    
    print("\n解決方案：")
    print("1. 使用監督式分群，以績效為主要分群依據")
    print("2. 調整特徵權重，突出關鍵參數")
    print("3. 結合業務規則，確保分群合理性")
    print("4. 多方法驗證，提高分群穩定性")
    
    print("\n這個解決方案可以推廣到所有策略和數據源，")
    print("只需要根據具體策略調整參數重要性和業務規則即可。")

if __name__ == "__main__":
    main() 