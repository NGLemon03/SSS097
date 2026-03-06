#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速距離度量方法測試
簡化版本，專注於核心指標比較
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from trial_diversity_analyzer import TrialDiversityAnalyzer
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_distance_test():
    """快速測試不同距離度量方法"""
    logger.info("開始快速距離度量方法測試")
    
    # 載入數據
    csv_file = "results_op15/optuna_results_ssma_turn_Self_20250630_075809.csv"
    df = pd.read_csv(csv_file)
    logger.info(f"載入數據: {df.shape}")
    
    # 測試配置
    test_configs = [
        {
            'name': 'euclidean',
            'description': '標準歐氏距離',
            'config': {
                'distance_method': 'euclidean',
                'normalize_method': 'standard',
                'use_performance_weighted': False
            }
        },
        {
            'name': 'weighted_euclidean',
            'description': '加權歐氏距離',
            'config': {
                'distance_method': 'weighted_euclidean',
                'normalize_method': 'standard',
                'parameter_weights': {
                    'weight_0': 2.0,  # linlen
                    'weight_1': 1.5,  # smaalen
                    'weight_2': 1.0,  # buy_mult
                    'weight_3': 1.0,  # sell_mult
                    'weight_4': 0.5   # stop_loss
                },
                'use_performance_weighted': False
            }
        },
        {
            'name': 'mahalanobis',
            'description': '馬氏距離',
            'config': {
                'distance_method': 'mahalanobis',
                'normalize_method': 'standard',
                'use_mahalanobis': True,
                'use_performance_weighted': False
            }
        },
        {
            'name': 'cosine',
            'description': '餘弦距離',
            'config': {
                'distance_method': 'cosine',
                'normalize_method': 'standard',
                'use_performance_weighted': False
            }
        }
    ]
    
    results = {}
    
    for config in test_configs:
        logger.info(f"\n=== 測試 {config['description']} ===")
        
        try:
            # 創建分析器
            analyzer = TrialDiversityAnalyzer(
                performance_metrics=['num_trades', 'avg_excess_return', 'avg_hold_days'],
                parameter_metrics=['linlen', 'smaalen', 'buy_mult', 'sell_mult', 'stop_loss'],
                **config['config']
            )
            
            # 預處理數據
            df_processed = analyzer.preprocess_trials(df)
            
            # 計算距離矩陣
            distance_matrix = analyzer.calculate_distance_matrix(df_processed, 'performance')
            
            if distance_matrix.size > 0:
                # 計算動態門檻
                threshold = analyzer._calculate_dynamic_threshold(distance_matrix)
                
                # 識別相似試驗
                similar_trials = analyzer.identify_similar_trials(
                    df_processed, 
                    distance_threshold=threshold,
                    use_dynamic_threshold=False
                )
                
                # 檢測過擬合風險
                overfitting_risk = analyzer.detect_overfitting_risk(df_processed)
                
                # 分析參數相關性
                param_analysis = analyzer._analyze_parameter_correlations(df_processed)
                
                results[config['name']] = {
                    'description': config['description'],
                    'similar_pairs_count': similar_trials.get('total_similar_pairs', 0),
                    'avg_distance': similar_trials.get('avg_distance', 0),
                    'distance_threshold': threshold,
                    'overfitting_risk_score': overfitting_risk.get('risk_score', 0),
                    'overfitting_risk_level': overfitting_risk.get('overfitting_risk', 'unknown'),
                    'parameter_correlation_avg': param_analysis.get('avg_correlation', 0),
                    'distance_matrix_shape': distance_matrix.shape,
                    'distance_stats': {
                        'mean': float(np.mean(distance_matrix)),
                        'std': float(np.std(distance_matrix)),
                        'min': float(np.min(distance_matrix)),
                        'max': float(np.max(distance_matrix))
                    }
                }
                
                logger.info(f"✓ {config['description']} 測試完成")
                logger.info(f"  相似試驗對數: {results[config['name']]['similar_pairs_count']}")
                logger.info(f"  平均距離: {results[config['name']]['avg_distance']:.4f}")
                logger.info(f"  過擬合風險分數: {results[config['name']]['overfitting_risk_score']:.3f}")
                logger.info(f"  參數相關性: {results[config['name']]['parameter_correlation_avg']:.3f}")
                
            else:
                logger.error(f"✗ {config['description']} 距離矩陣為空")
                
        except Exception as e:
            logger.error(f"✗ {config['description']} 測試異常: {e}")
            continue
    
    # 顯示比較結果
    if results:
        logger.info("\n=== 距離方法比較結果 ===")
        comparison_df = pd.DataFrame(results).T
        
        # 顯示關鍵指標
        key_metrics = ['similar_pairs_count', 'avg_distance', 'overfitting_risk_score', 'parameter_correlation_avg']
        logger.info(f"\n{comparison_df[key_metrics]}")
        
        # 保存結果
        output_file = Path("quick_distance_test_results.csv")
        comparison_df.to_csv(output_file)
        logger.info(f"\n結果已保存至: {output_file}")
        
        # 分析結果
        logger.info("\n=== 結果分析 ===")
        
        # 找出最佳方法
        best_methods = {}
        
        # 相似試驗對數（適中為好）
        similar_counts = comparison_df['similar_pairs_count']
        best_methods['similar_pairs'] = similar_counts.idxmin()  # 較少相似對表示更好的區分度
        
        # 平均距離（適中為好）
        avg_distances = comparison_df['avg_distance']
        best_methods['avg_distance'] = avg_distances.idxmax()  # 較大平均距離表示更好的區分度
        
        # 過擬合風險分數（越低越好）
        risk_scores = comparison_df['overfitting_risk_score']
        best_methods['overfitting_risk'] = risk_scores.idxmin()
        
        # 參數相關性（越低越好）
        param_corr = comparison_df['parameter_correlation_avg']
        best_methods['parameter_correlation'] = param_corr.idxmin()
        
        logger.info("各指標最佳方法:")
        for metric, method in best_methods.items():
            logger.info(f"  {metric}: {method} ({comparison_df.loc[method, 'description']})")
        
        # 綜合評估
        logger.info("\n=== 綜合評估 ===")
        logger.info("基於測試結果的建議:")
        
        if 'mahalanobis' in best_methods.values():
            logger.info("✓ 馬氏距離在多個指標上表現良好，建議用於參數相關性高的情況")
        
        if 'cosine' in best_methods.values():
            logger.info("✓ 餘弦距離在某些指標上表現良好，適合關注參數模式而非絕對值")
        
        if 'weighted_euclidean' in best_methods.values():
            logger.info("✓ 加權歐氏距離在某些指標上表現良好，適合參數重要性差異大的情況")
        
        # 檢查是否有明顯的改進
        euclidean_risk = results.get('euclidean', {}).get('overfitting_risk_score', 1.0)
        other_methods_risk = [v.get('overfitting_risk_score', 1.0) for k, v in results.items() if k != 'euclidean']
        
        if other_methods_risk and min(other_methods_risk) < euclidean_risk:
            improvement = (euclidean_risk - min(other_methods_risk)) / euclidean_risk * 100
            logger.info(f"✓ 改進方法相比標準歐氏距離降低了 {improvement:.1f}% 的過擬合風險")
        else:
            logger.info("⚠ 改進方法相比標準歐氏距離沒有明顯改善")
    
    else:
        logger.error("沒有成功的測試結果")
    
    return results

if __name__ == "__main__":
    quick_distance_test() 