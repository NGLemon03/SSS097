# v14: 多階段分群，分層次進行分群分析
# 主要功能：多階段分群、層次化分析、穩定性評估、結果整合
# 提供更複雜但更精確的分群策略

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats
import json
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 設定路徑
INPUT_DIR = 'results/risk_enhanced_clustering'
OUTPUT_DIR = 'results/multi_stage_clustering'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_risk_clustered_data():
    """載入風險分群後的數據"""
    all_files = glob.glob(f'{INPUT_DIR}/*_risk_clustered.csv')
    data_dict = {}
    
    for f in all_files:
        fname = os.path.basename(f)
        print(f"處理文件: {fname}")
        
        # 提取策略和數據源
        if 'RMA_Self' in fname:
            strategy = 'RMA_Self'
            datasource = 'Self'
        elif 'single_Self' in fname:
            strategy = 'single_Self'
            datasource = 'Self'
        elif 'ssma_turn_Self' in fname:
            strategy = 'ssma_turn_Self'
            datasource = 'Self'
        else:
            continue
        
        df = pd.read_csv(f)
        print(f"載入 {strategy} - {datasource}: {len(df)} 行")
        data_dict[(strategy, datasource)] = df
    
    return data_dict

def multi_stage_clustering(df, strategy, datasource):
    """多階段分群：先按return/score分組，再用his/risk細分"""
    print(f"\n=== {strategy} - {datasource} 多階段分群 ===")
    
    # 第一階段：按return/score分組
    print("第一階段：按return/score分組")
    
    # 選擇return/score相關特徵
    return_score_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['return', 'score', 'sharpe', 'calmar', 'sortino']):
            return_score_cols.append(col)
    
    print(f"找到return/score相關特徵: {return_score_cols}")
    
    if len(return_score_cols) == 0:
        print("沒有找到return/score特徵，使用預設特徵")
        return_score_cols = ['total_return', 'annual_return', 'sharpe_ratio']
        return_score_cols = [col for col in return_score_cols if col in df.columns]
    
    # 準備return/score特徵數據
    X_return_score = df[return_score_cols].fillna(0).values
    scaler_return = StandardScaler()
    X_return_score_scaled = scaler_return.fit_transform(X_return_score)
    
    # 使用較少的群組數進行第一階段分群
    n_samples = len(df)
    stage1_k = max(5, min(20, n_samples // 50))  # 每群約50個數據
    
    print(f"第一階段群組數: {stage1_k}")
    
    # 第一階段分群（階層式）
    Z = linkage(X_return_score_scaled, method='ward')
    stage1_labels = fcluster(Z, t=stage1_k, criterion='maxclust')
    
    # 統計第一階段結果
    stage1_sizes = np.bincount(stage1_labels)
    print(f"第一階段群組大小: {stage1_sizes}")
    
    # 第二階段：在每個第一階段群組內進行細分
    print("\n第二階段：在每個群組內進行細分")
    
    # 選擇his/risk特徵
    his_risk_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['volatility', 'risk', 'var', 'cvar', 'downside', 'drawdown']):
            his_risk_cols.append(col)
    
    print(f"找到his/risk相關特徵: {his_risk_cols}")
    
    if len(his_risk_cols) == 0:
        print("沒有找到his/risk特徵，使用預設特徵")
        his_risk_cols = ['annual_volatility', 'downside_risk', 'var_95_annual', 'cvar_95_annual']
        his_risk_cols = [col for col in his_risk_cols if col in df.columns]
    
    # 準備his/risk特徵數據
    X_his_risk = df[his_risk_cols].fillna(0).values
    scaler_his_risk = StandardScaler()
    X_his_risk_scaled = scaler_his_risk.fit_transform(X_his_risk)
    
    # 最終分群標籤
    final_labels = np.zeros(len(df), dtype=int)
    current_cluster_id = 1
    
    stage2_results = {}
    
    # 對每個第一階段群組進行細分
    for stage1_id in np.unique(stage1_labels):
        mask = stage1_labels == stage1_id
        group_size = np.sum(mask)
        
        print(f"\n處理第一階段群組 {stage1_id} (大小: {group_size})")
        
        if group_size <= 5:
            # 如果群組已經很小，直接保留
            final_labels[mask] = current_cluster_id
            stage2_results[stage1_id] = {
                'method': 'no_split',
                'n_clusters': 1,
                'sizes': [group_size],
                'silhouette': 0
            }
            current_cluster_id += 1
            continue
        
        # 提取該群組的his/risk特徵
        X_group = X_his_risk_scaled[mask]
        
        # 決定第二階段群組數
        if group_size <= 10:
            stage2_k = 2
        elif group_size <= 20:
            stage2_k = 3
        elif group_size <= 50:
            stage2_k = 5
        else:
            stage2_k = max(5, group_size // 10)  # 每群約10個數據
        
        print(f"  第二階段群組數: {stage2_k}")
        
        try:
            # 第二階段分群（KMeans）
            kmeans = KMeans(n_clusters=stage2_k, random_state=42, n_init=10)
            stage2_labels = kmeans.fit_predict(X_group)
            
            # 計算該群組的silhouette score
            if len(np.unique(stage2_labels)) > 1:
                silhouette = silhouette_score(X_group, stage2_labels)
            else:
                silhouette = 0
            
            # 統計第二階段結果
            stage2_sizes = np.bincount(stage2_labels)
            
            print(f"  第二階段群組大小: {stage2_sizes}")
            print(f"  Silhouette Score: {silhouette:.3f}")
            
            # 分配最終標籤
            for i, stage2_id in enumerate(stage2_labels):
                final_labels[mask][i] = current_cluster_id + stage2_id
            
            stage2_results[stage1_id] = {
                'method': 'kmeans',
                'n_clusters': stage2_k,
                'sizes': stage2_sizes.tolist(),
                'silhouette': silhouette
            }
            
            current_cluster_id += stage2_k
            
        except Exception as e:
            print(f"  第二階段分群失敗: {e}")
            # 如果失敗，直接保留原群組
            final_labels[mask] = current_cluster_id
            stage2_results[stage1_id] = {
                'method': 'failed',
                'n_clusters': 1,
                'sizes': [group_size],
                'silhouette': 0
            }
            current_cluster_id += 1
    
    # 重新編號標籤（從1開始）
    unique_labels = np.unique(final_labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels, 1)}
    final_labels = np.array([label_mapping[label] for label in final_labels])
    
    # 統計最終結果
    final_sizes = np.bincount(final_labels)
    print(f"\n最終分群結果:")
    print(f"總群組數: {len(final_sizes)}")
    print(f"群組大小分布: {final_sizes}")
    print(f"平均群組大小: {np.mean(final_sizes):.1f}")
    print(f"最大群組大小: {np.max(final_sizes)}")
    print(f"小群組比例 (≤5): {np.sum(final_sizes <= 5) / len(final_sizes):.3f}")
    
    return {
        'final_labels': final_labels,
        'stage1_labels': stage1_labels,
        'stage1_sizes': stage1_sizes,
        'stage2_results': stage2_results,
        'final_sizes': final_sizes,
        'return_score_cols': return_score_cols,
        'his_risk_cols': his_risk_cols
    }

def analyze_multi_stage_quality(clustering_result, df, strategy, datasource):
    """分析多階段分群品質"""
    print(f"\n=== {strategy} - {datasource} 分群品質分析 ===")
    
    final_labels = clustering_result['final_labels']
    stage1_labels = clustering_result['stage1_labels']
    
    # 基本統計
    unique_labels = np.unique(final_labels)
    final_sizes = clustering_result['final_sizes']
    
    # 計算小群組統計
    small_clusters = final_sizes[final_sizes <= 5]
    medium_clusters = final_sizes[(final_sizes > 5) & (final_sizes <= 10)]
    large_clusters = final_sizes[final_sizes > 10]
    
    print(f"小群組 (≤5): {len(small_clusters)} 個")
    print(f"中群組 (6-10): {len(medium_clusters)} 個")
    print(f"大群組 (>10): {len(large_clusters)} 個")
    
    # 特徵重要性分析
    all_features = clustering_result['return_score_cols'] + clustering_result['his_risk_cols']
    feature_importance = {}
    
    for feature in all_features:
        if feature in df.columns:
            feature_values = df[feature].values
            
            # ANOVA檢定
            try:
                groups = [feature_values[final_labels == i] for i in unique_labels]
                f_stat, p_value = stats.f_oneway(*groups)
                
                feature_importance[feature] = {
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                feature_importance[feature] = {
                    'f_stat': 0,
                    'p_value': 1,
                    'significant': False
                }
    
    n_significant_features = sum(1 for f in feature_importance.values() if f['significant'])
    print(f"顯著特徵數: {n_significant_features}")
    
    return {
        'final_sizes': final_sizes,
        'small_clusters': len(small_clusters),
        'medium_clusters': len(medium_clusters),
        'large_clusters': len(large_clusters),
        'mean_size': np.mean(final_sizes),
        'max_size': np.max(final_sizes),
        'small_ratio': len(small_clusters) / len(final_sizes),
        'feature_importance': feature_importance,
        'n_significant_features': n_significant_features
    }

def create_multi_stage_visualizations(df, clustering_result, quality_analysis, strategy, datasource):
    """創建多階段分群可視化"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    final_labels = clustering_result['final_labels']
    stage1_labels = clustering_result['stage1_labels']
    final_sizes = clustering_result['final_sizes']
    stage1_sizes = clustering_result['stage1_sizes']
    
    # 1. 第一階段群組大小分布
    ax1 = axes[0, 0]
    ax1.bar(range(len(stage1_sizes)), stage1_sizes)
    ax1.set_xlabel('第一階段群組編號')
    ax1.set_ylabel('群組大小')
    ax1.set_title('第一階段群組大小分布')
    ax1.grid(True, alpha=0.3)
    
    # 2. 最終群組大小分布
    ax2 = axes[0, 1]
    ax2.bar(range(len(final_sizes)), final_sizes)
    ax2.set_xlabel('最終群組編號')
    ax2.set_ylabel('群組大小')
    ax2.set_title('最終群組大小分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 群組大小分布比較
    ax3 = axes[0, 2]
    small_sizes = final_sizes[final_sizes <= 5]
    medium_sizes = final_sizes[(final_sizes > 5) & (final_sizes <= 10)]
    large_sizes = final_sizes[final_sizes > 10]
    
    ax3.hist([small_sizes, medium_sizes, large_sizes], 
             label=['小群組(≤5)', '中群組(6-10)', '大群組(>10)'], 
             alpha=0.7, bins=10)
    ax3.set_xlabel('群組大小')
    ax3.set_ylabel('群組數量')
    ax3.set_title('群組大小分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 第一階段 vs 最終群組數
    ax4 = axes[0, 3]
    ax4.bar(['第一階段', '最終'], [len(stage1_sizes), len(final_sizes)])
    ax4.set_ylabel('群組數量')
    ax4.set_title('分群階段比較')
    ax4.grid(True, alpha=0.3)
    
    # 5. 最佳分群的PCA散點圖
    ax5 = axes[1, 0]
    
    # 使用主要特徵進行PCA
    main_cols = clustering_result['return_score_cols'] + clustering_result['his_risk_cols']
    available_cols = [col for col in main_cols if col in df.columns][:10]
    
    if len(available_cols) >= 2:
        X_pca = df[available_cols].fillna(0).values
        scaler = StandardScaler()
        X_pca_scaled = scaler.fit_transform(X_pca)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca_2d = pca.fit_transform(X_pca_scaled)
        
        # 只顯示前20個群組（避免圖表太亂）
        unique_labels = np.unique(final_labels)
        for cluster_id in unique_labels[:20]:
            mask = final_labels == cluster_id
            ax5.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                       label=f'Cluster {cluster_id}', alpha=0.7)
        
        ax5.set_xlabel('PCA Component 1')
        ax5.set_ylabel('PCA Component 2')
        ax5.set_title('最終分群PCA散點圖\n(顯示前20個群組)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. 群組大小分布
    ax6 = axes[1, 1]
    ax6.scatter(range(len(final_sizes)), final_sizes, alpha=0.6)
    ax6.set_xlabel('群組編號')
    ax6.set_ylabel('群組大小')
    ax6.set_title('群組大小分布')
    ax6.grid(True, alpha=0.3)
    
    # 7. 第二階段分群統計
    ax7 = axes[1, 2]
    stage2_results = clustering_result['stage2_results']
    methods = [result['method'] for result in stage2_results.values()]
    method_counts = pd.Series(methods).value_counts()
    
    ax7.bar(method_counts.index, method_counts.values)
    ax7.set_xlabel('第二階段方法')
    ax7.set_ylabel('群組數量')
    ax7.set_title('第二階段分群方法統計')
    ax7.grid(True, alpha=0.3)
    
    # 8. 特徵重要性比較
    ax8 = axes[1, 3]
    feature_importance = quality_analysis['feature_importance']
    important_features = [(f, v['f_stat']) for f, v in feature_importance.items() if v['significant']]
    important_features.sort(key=lambda x: x[1], reverse=True)
    
    if important_features:
        features, f_stats = zip(*important_features[:10])
        ax8.barh(range(len(features)), f_stats)
        ax8.set_yticks(range(len(features)))
        ax8.set_yticklabels(features)
        ax8.set_xlabel('F-statistic')
        ax8.set_title('重要特徵')
        ax8.grid(True, alpha=0.3)
    
    # 9. 群組大小統計
    ax9 = axes[2, 0]
    size_stats = {
        '小群組(≤5)': quality_analysis['small_clusters'],
        '中群組(6-10)': quality_analysis['medium_clusters'],
        '大群組(>10)': quality_analysis['large_clusters']
    }
    
    ax9.bar(size_stats.keys(), size_stats.values())
    ax9.set_ylabel('群組數量')
    ax9.set_title('群組大小統計')
    ax9.grid(True, alpha=0.3)
    
    # 10. 第二階段群組數分布
    ax10 = axes[2, 1]
    stage2_clusters = [result['n_clusters'] for result in stage2_results.values()]
    
    ax10.hist(stage2_clusters, bins=10, alpha=0.7)
    ax10.set_xlabel('第二階段群組數')
    ax10.set_ylabel('頻率')
    ax10.set_title('第二階段群組數分布')
    ax10.grid(True, alpha=0.3)
    
    # 11. 第二階段Silhouette Score分布
    ax11 = axes[2, 2]
    stage2_silhouettes = [result['silhouette'] for result in stage2_results.values() if result['method'] == 'kmeans']
    
    if stage2_silhouettes:
        ax11.hist(stage2_silhouettes, bins=10, alpha=0.7)
        ax11.set_xlabel('Silhouette Score')
        ax11.set_ylabel('頻率')
        ax11.set_title('第二階段Silhouette Score分布')
        ax11.grid(True, alpha=0.3)
    
    # 12. 最終推薦
    ax12 = axes[2, 3]
    ax12.text(0.1, 0.9, f'策略: {strategy}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.8, f'總群組數: {len(final_sizes)}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.7, f'平均群組大小: {quality_analysis["mean_size"]:.1f}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.6, f'最大群組大小: {quality_analysis["max_size"]}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.5, f'小群組比例: {quality_analysis["small_ratio"]:.3f}', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.4, f'顯著特徵數: {quality_analysis["n_significant_features"]}', fontsize=12, transform=ax12.transAxes)
    
    # 推薦信息
    if quality_analysis['small_ratio'] >= 0.7:
        ax12.text(0.1, 0.3, '✅ 小群組比例良好', fontsize=12, transform=ax12.transAxes, color='green')
    else:
        ax12.text(0.1, 0.3, '⚠️ 小群組比例偏低', fontsize=12, transform=ax12.transAxes, color='orange')
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    ax12.set_title('最終推薦')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_multi_stage_clustering.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def generate_multi_stage_recommendations(clustering_result, quality_analysis, strategy, datasource):
    """生成多階段分群建議"""
    print(f"\n=== {strategy} - {datasource} 多階段分群建議 ===")
    
    final_sizes = clustering_result['final_sizes']
    stage2_results = clustering_result['stage2_results']
    
    print(f"總群組數: {len(final_sizes)}")
    print(f"平均群組大小: {quality_analysis['mean_size']:.1f}")
    print(f"最大群組大小: {quality_analysis['max_size']}")
    print(f"小群組比例: {quality_analysis['small_ratio']:.3f}")
    print(f"顯著特徵數: {quality_analysis['n_significant_features']}")
    
    # 分析群組大小分布
    print(f"\n群組大小分布:")
    print(f"小群組 (≤5): {quality_analysis['small_clusters']} 個")
    print(f"中群組 (6-10): {quality_analysis['medium_clusters']} 個")
    print(f"大群組 (>10): {quality_analysis['large_clusters']} 個")
    
    # 分析第二階段分群效果
    print(f"\n第二階段分群效果:")
    kmeans_results = [result for result in stage2_results.values() if result['method'] == 'kmeans']
    if kmeans_results:
        avg_silhouette = np.mean([result['silhouette'] for result in kmeans_results])
        print(f"平均Silhouette Score: {avg_silhouette:.3f}")
        print(f"KMeans分群次數: {len(kmeans_results)}")
    
    # 推薦改進建議
    print(f"\n改進建議:")
    if quality_analysis['small_ratio'] < 0.5:
        print("- 小群組比例偏低，建議增加第二階段群組數")
    if quality_analysis['max_size'] > 10:
        print("- 存在大群組，建議調整第二階段分群參數")
    if quality_analysis['n_significant_features'] < 5:
        print("- 顯著特徵較少，建議檢查特徵選擇")
    
    return {
        'total_clusters': len(final_sizes),
        'mean_size': quality_analysis['mean_size'],
        'max_size': quality_analysis['max_size'],
        'small_ratio': quality_analysis['small_ratio'],
        'n_significant_features': quality_analysis['n_significant_features']
    }

def main():
    """主函數"""
    print("開始多階段分群分析...")
    
    # 載入數據
    data_dict = load_risk_clustered_data()
    
    all_recommendations = {}
    
    for (strategy, datasource), df in data_dict.items():
        print(f"\n分析 {strategy} - {datasource}")
        
        # 多階段分群
        clustering_result = multi_stage_clustering(df, strategy, datasource)
        
        # 分析分群品質
        quality_analysis = analyze_multi_stage_quality(clustering_result, df, strategy, datasource)
        
        # 創建可視化
        create_multi_stage_visualizations(df, clustering_result, quality_analysis, strategy, datasource)
        
        # 生成建議
        recommendations = generate_multi_stage_recommendations(clustering_result, quality_analysis, 
                                                             strategy, datasource)
        
        # 保存結果
        result_df = df.copy()
        result_df['multi_stage_cluster'] = clustering_result['final_labels']
        result_df['stage1_cluster'] = clustering_result['stage1_labels']
        
        output_file = f'{OUTPUT_DIR}/{strategy}_{datasource}_multi_stage_clustered.csv'
        result_df.to_csv(output_file, index=False)
        
        all_recommendations[(strategy, datasource)] = recommendations
        
        print(f"結果已保存到 {output_file}")
    
    # 生成總結報告
    print(f"\n=== 多階段分群總結 ===")
    for (strategy, datasource), recs in all_recommendations.items():
        print(f"\n{strategy} - {datasource}:")
        print(f"  總群組數: {recs['total_clusters']}")
        print(f"  平均群組大小: {recs['mean_size']:.1f}")
        print(f"  小群組比例: {recs['small_ratio']:.3f}")
        print(f"  顯著特徵數: {recs['n_significant_features']}")
    
    print(f"\n所有結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 