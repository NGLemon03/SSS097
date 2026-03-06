#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面的距離度量方法測試
實際比較不同方法在策略參數分析中的效果
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trial_diversity_analyzer import TrialDiversityAnalyzer
import logging
import json
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDistanceTest:
    """全面的距離度量方法測試類"""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.results = {}
        self.test_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def load_data(self):
        """載入數據"""
        logger.info(f"載入數據: {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        logger.info(f"數據形狀: {self.df.shape}")
        logger.info(f"數據欄位: {list(self.df.columns)}")
        
        # 檢查必要欄位
        required_cols = ['trial_number', 'score', 'total_return', 'num_trades']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            logger.error(f"缺少必要欄位: {missing_cols}")
            return False
        
        # 檢查參數欄位
        param_cols = [col for col in self.df.columns if col.startswith('param_')]
        logger.info(f"找到參數欄位: {param_cols}")
        
        return True
    
    def test_distance_methods(self):
        """測試不同的距離度量方法"""
        logger.info("開始測試不同的距離度量方法")
        
        # 定義測試配置
        test_configs = [
            {
                'name': 'standard_euclidean',
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
            },
            {
                'name': 'performance_weighted',
                'description': '性能加權歐氏距離',
                'config': {
                    'distance_method': 'euclidean',
                    'normalize_method': 'standard',
                    'use_performance_weighted': True
                }
            }
        ]
        
        # 執行測試
        for config in test_configs:
            logger.info(f"\n=== 測試 {config['description']} ===")
            
            try:
                # 創建分析器
                analyzer = TrialDiversityAnalyzer(
                    performance_metrics=['num_trades', 'avg_excess_return', 'avg_hold_days'],
                    parameter_metrics=['linlen', 'smaalen', 'buy_mult', 'sell_mult', 'stop_loss'],
                    **config['config']
                )
                
                # 執行分析
                output_dir = Path(f"test_output_{config['name']}_{self.test_timestamp}")
                result = analyzer.run_complete_analysis(self.df, output_dir)
                
                if result:
                    self.results[config['name']] = {
                        'description': config['description'],
                        'config': config['config'],
                        'result': result,
                        'output_dir': str(output_dir)
                    }
                    logger.info(f"✓ {config['description']} 測試完成")
                else:
                    logger.error(f"✗ {config['description']} 測試失敗")
                    
            except Exception as e:
                logger.error(f"✗ {config['description']} 測試異常: {e}")
                continue
    
    def analyze_results(self):
        """分析測試結果"""
        logger.info("\n=== 分析測試結果 ===")
        
        if not self.results:
            logger.error("沒有可用的測試結果")
            return
        
        # 提取關鍵指標
        comparison_data = {}
        
        for method_name, data in self.results.items():
            result = data['result']
            
            # 相似試驗分析
            similar_trials = result.get('similar_trials', {})
            overfitting_risk = result.get('overfitting_risk', {})
            parameter_analysis = result.get('parameter_analysis', {})
            
            comparison_data[method_name] = {
                'description': data['description'],
                'similar_pairs_count': similar_trials.get('total_similar_pairs', 0),
                'avg_distance': similar_trials.get('avg_distance', 0),
                'distance_threshold': similar_trials.get('distance_threshold', 0),
                'overfitting_risk_score': overfitting_risk.get('risk_score', 0),
                'overfitting_risk_level': overfitting_risk.get('overfitting_risk', 'unknown'),
                'parameter_correlation_avg': parameter_analysis.get('avg_param_distance', 0),
                'diverse_trials_count': len(result.get('diverse_trials', {}).get('selected_trials', [])),
                'clustering_n_clusters': result.get('clustering', {}).get('n_clusters', 0)
            }
        
        # 創建比較DataFrame
        self.comparison_df = pd.DataFrame(comparison_data).T
        
        # 保存比較結果
        comparison_file = Path(f"distance_methods_comparison_{self.test_timestamp}.csv")
        self.comparison_df.to_csv(comparison_file)
        logger.info(f"比較結果已保存至: {comparison_file}")
        
        # 顯示比較結果
        logger.info("\n=== 距離方法比較結果 ===")
        logger.info(f"\n{self.comparison_df}")
        
        return self.comparison_df
    
    def create_visualizations(self):
        """創建視覺化圖表"""
        logger.info("\n=== 創建視覺化圖表 ===")
        
        if not hasattr(self, 'comparison_df'):
            logger.error("沒有比較數據，無法創建視覺化")
            return
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建圖表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('距離度量方法比較分析', fontsize=16)
        
        # 1. 相似試驗對數量比較
        ax1 = axes[0, 0]
        self.comparison_df['similar_pairs_count'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('相似試驗對數量')
        ax1.set_ylabel('數量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 平均距離比較
        ax2 = axes[0, 1]
        self.comparison_df['avg_distance'].plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('平均距離')
        ax2.set_ylabel('距離值')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 過擬合風險分數比較
        ax3 = axes[0, 2]
        self.comparison_df['overfitting_risk_score'].plot(kind='bar', ax=ax3, color='salmon')
        ax3.set_title('過擬合風險分數')
        ax3.set_ylabel('風險分數')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 多樣性試驗數量比較
        ax4 = axes[1, 0]
        self.comparison_df['diverse_trials_count'].plot(kind='bar', ax=ax4, color='gold')
        ax4.set_title('多樣性試驗數量')
        ax4.set_ylabel('數量')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 分群數量比較
        ax5 = axes[1, 1]
        self.comparison_df['clustering_n_clusters'].plot(kind='bar', ax=ax5, color='lightcoral')
        ax5.set_title('分群數量')
        ax5.set_ylabel('群組數')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 參數相關性比較
        ax6 = axes[1, 2]
        self.comparison_df['parameter_correlation_avg'].plot(kind='bar', ax=ax6, color='plum')
        ax6.set_title('平均參數相關性')
        ax6.set_ylabel('相關性')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存圖表
        plot_file = Path(f"distance_methods_comparison_{self.test_timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"視覺化圖表已保存至: {plot_file}")
        
        plt.show()
    
    def generate_detailed_report(self):
        """生成詳細報告"""
        logger.info("\n=== 生成詳細報告 ===")
        
        report = {
            'test_info': {
                'timestamp': self.test_timestamp,
                'data_file': self.csv_file,
                'data_shape': self.df.shape if self.df is not None else None,
                'test_methods': list(self.results.keys())
            },
            'comparison_results': self.comparison_df.to_dict() if hasattr(self, 'comparison_df') else {},
            'detailed_analysis': {}
        }
        
        # 詳細分析每個方法
        for method_name, data in self.results.items():
            result = data['result']
            
            # 分析相似試驗分布
            similar_trials = result.get('similar_trials', {})
            similar_pairs = similar_trials.get('similar_pairs', [])
            
            if similar_pairs:
                distances = [pair['distance'] for pair in similar_pairs]
                score_diffs = [pair['score_diff'] for pair in similar_pairs if pair['score_diff'] is not None]
                
                report['detailed_analysis'][method_name] = {
                    'similar_trials_stats': {
                        'total_pairs': len(similar_pairs),
                        'avg_distance': np.mean(distances),
                        'std_distance': np.std(distances),
                        'min_distance': np.min(distances),
                        'max_distance': np.max(distances),
                        'avg_score_diff': np.mean(score_diffs) if score_diffs else None,
                        'std_score_diff': np.std(score_diffs) if score_diffs else None
                    },
                    'overfitting_risk': result.get('overfitting_risk', {}),
                    'parameter_analysis': result.get('parameter_analysis', {}),
                    'diverse_trials': result.get('diverse_trials', {}),
                    'clustering': result.get('clustering', {})
                }
        
        # 保存詳細報告
        report_file = Path(f"detailed_analysis_report_{self.test_timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"詳細報告已保存至: {report_file}")
        
        return report
    
    def run_comprehensive_test(self):
        """運行全面測試"""
        logger.info("開始全面距離度量方法測試")
        
        # 1. 載入數據
        if not self.load_data():
            logger.error("數據載入失敗")
            return
        
        # 2. 測試距離方法
        self.test_distance_methods()
        
        # 3. 分析結果
        if self.results:
            self.analyze_results()
            self.create_visualizations()
            self.generate_detailed_report()
            
            logger.info("全面測試完成！")
            logger.info(f"測試了 {len(self.results)} 種距離度量方法")
            logger.info("請查看生成的CSV、PNG和JSON文件了解詳細結果")
        else:
            logger.error("沒有成功的測試結果")

def main():
    """主函數"""
    # 測試文件
    csv_file = "results_op15/optuna_results_ssma_turn_Self_20250630_075809.csv"
    
    # 創建測試實例
    tester = ComprehensiveDistanceTest(csv_file)
    
    # 運行全面測試
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 