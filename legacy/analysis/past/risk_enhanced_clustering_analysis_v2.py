# v2: 增強型風險分群分析，加入多種風險指標與權益曲線特徵
# 主要功能：自動展開參數、計算多種風險指標、分群與可視化
# 後續版本有更細緻的分群與多特徵分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, percentileofscore
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import os
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 設定路徑
INPUT_DIR = 'results/op16'
CACHE_DIR = 'cache/optuna16_equity'
OUTPUT_DIR = 'results/risk_enhanced_clustering'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_datasource_from_filename(filename):
    """從檔案名提取數據源"""
    fname = filename.upper().replace(' ', '').replace('(', '').replace(')', '').replace('^', '').replace('/', '').replace('.', '').replace('_', '')
    if 'SELF' in fname:
        return 'Self'
    if '2412' in fname:
        return '2412'
    if '2414' in fname:
        return '2414'
    return None

def extract_strategy_from_filename(filename):
    """從檔案名提取策略名稱"""
    file_pattern = re.compile(r'optuna_results_([a-zA-Z0-9_]+)_([^_]+)')
    m = file_pattern.search(filename)
    if m:
        return m.group(1)
    return None

def calculate_risk_metrics(equity_curve, window=252):
    """計算風險指標"""
    if len(equity_curve) < 10:
        return {}
    
    # 計算日報酬
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # 基本統計量
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # 年化指標
    annual_return = mean_return * 252
    annual_volatility = std_return * np.sqrt(252)
    
    # 夏普比率 (假設無風險利率為0)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # 最大回撤
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # 下行風險 (Downside Risk)
    downside_returns = returns[returns < 0]
    downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    # Sortino Ratio
    sortino_ratio = annual_return / downside_risk if downside_risk > 0 else 0
    
    # VaR (Value at Risk) - 95% 信心水準
    var_95 = np.percentile(returns, 5)
    var_95_annual = var_95 * np.sqrt(252)
    
    # CVaR (Conditional Value at Risk) - 95% 信心水準
    cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
    cvar_95_annual = cvar_95 * np.sqrt(252)
    
    # 尾部風險指標
    min_return = np.min(returns)
    percentile_5 = np.percentile(returns, 5)
    percentile_1 = np.percentile(returns, 1)
    
    # 偏度與峰度
    skewness = np.mean(((returns - mean_return) / std_return) ** 3) if std_return > 0 else 0
    kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) if std_return > 0 else 0
    
    # 波動率穩定性 (使用滾動標準差)
    if len(returns) >= window:
        rolling_vol = pd.Series(returns).rolling(window=window).std().dropna()
        vol_stability = np.std(rolling_vol) if len(rolling_vol) > 0 else 0
    else:
        vol_stability = 0
    
    # 回撤持續時間
    drawdown_duration = np.sum(drawdown < -0.05) / len(drawdown)  # 超過5%回撤的時間比例
    
    # 連續虧損天數
    consecutive_losses = 0
    max_consecutive_losses = 0
    for ret in returns:
        if ret < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'downside_risk': downside_risk,
        'sortino_ratio': sortino_ratio,
        'var_95_annual': var_95_annual,
        'cvar_95_annual': cvar_95_annual,
        'min_return': min_return,
        'percentile_5': percentile_5,
        'percentile_1': percentile_1,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'vol_stability': vol_stability,
        'drawdown_duration': drawdown_duration,
        'max_consecutive_losses': max_consecutive_losses
    }

def load_and_process_data():
    """載入並處理所有數據"""
    all_files = glob.glob(f'{INPUT_DIR}/*.csv')
    data_dict = {}
    
    for f in all_files:
        fname = os.path.basename(f)
        strategy = extract_strategy_from_filename(fname)
        datasource = extract_datasource_from_filename(fname)
        
        if strategy and datasource:
            df = pd.read_csv(f)
            
            # 自動展開 parameters 欄位
            if 'parameters' in df.columns:
                param_df = df['parameters'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
                param_df = pd.json_normalize(param_df.tolist())
                param_df = param_df.add_prefix('param_')
                df = pd.concat([df, param_df], axis=1)
            
            # 載入權益曲線並計算風險指標
            risk_metrics_list = []
            for _, row in df.iterrows():
                trial_number = row['trial_number']
                eq_path = os.path.join(CACHE_DIR, f'trial_{trial_number:05d}_equity.npy')
                
                if os.path.exists(eq_path):
                    equity_curve = np.load(eq_path)
                    risk_metrics = calculate_risk_metrics(equity_curve)
                    risk_metrics_list.append(risk_metrics)
                else:
                    # 如果沒有權益曲線，使用基本指標
                    risk_metrics = {
                        'annual_return': row.get('total_return', 0),
                        'annual_volatility': 0,
                        'sharpe_ratio': row.get('sharpe_ratio', 0),
                        'max_drawdown': row.get('max_drawdown', 0),
                        'downside_risk': 0,
                        'sortino_ratio': 0,
                        'var_95_annual': 0,
                        'cvar_95_annual': 0,
                        'min_return': 0,
                        'percentile_5': 0,
                        'percentile_1': 0,
                        'skewness': 0,
                        'kurtosis': 0,
                        'vol_stability': 0,
                        'drawdown_duration': 0,
                        'max_consecutive_losses': 0
                    }
                    risk_metrics_list.append(risk_metrics)
            
            # 將風險指標添加到DataFrame
            risk_df = pd.DataFrame(risk_metrics_list)
            df = pd.concat([df, risk_df], axis=1)
            
            key = (strategy, datasource)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(df)
    
    # 合併每個策略-數據源組合的所有檔案
    merged_data = {}
    for key, dfs in data_dict.items():
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_data[key] = merged_df
    
    return merged_data

def perform_risk_enhanced_clustering(df, strategy, datasource):
    """執行風險增強的分群分析"""
    # 選擇特徵
    performance_features = [
        'total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor',
        'cpcv_oos_mean', 'cpcv_oos_min'
    ]
    
    risk_features = [
        'annual_volatility', 'downside_risk', 'sortino_ratio', 'var_95_annual',
        'cvar_95_annual', 'skewness', 'kurtosis', 'vol_stability',
        'drawdown_duration', 'max_consecutive_losses'
    ]
    
    param_features = [col for col in df.columns if col.startswith('param_')]
    
    # 組合所有特徵
    all_features = performance_features + risk_features + param_features
    available_features = [f for f in all_features if f in df.columns]
    
    if len(available_features) < 5:
        print(f"警告: {strategy}-{datasource} 可用特徵不足，跳過分群")
        return None
    
    # 準備特徵數據
    X = df[available_features].fillna(0).values
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降維
    n_components = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_components < 2:
        print(f"警告: {strategy}-{datasource} PCA組件不足，跳過分群")
        return None
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 階層式分群
    Z = linkage(X_pca, method='ward')
    
    # 自動選擇最佳分群數
    max_clusters = min(10, len(df) // 5)  # 每群至少5個樣本
    best_k = 2
    best_score = -1
    
    for k in range(2, max_clusters + 1):
        try:
            labels = fcluster(Z, t=k, criterion='maxclust')
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_pca, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except:
            continue
    
    # 使用最佳分群數
    final_labels = fcluster(Z, t=best_k, criterion='maxclust')
    
    # 計算分群品質指標
    sil_score = silhouette_score(X_pca, final_labels) if len(np.unique(final_labels)) > 1 else 0
    
    return {
        'labels': final_labels,
        'X_pca': X_pca,
        'linkage_matrix': Z,
        'best_k': best_k,
        'silhouette_score': sil_score,
        'feature_names': available_features,
        'pca_components': n_components
    }

def create_risk_cluster_visualizations(df, cluster_result, strategy, datasource):
    """創建風險分群可視化"""
    if cluster_result is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA散點圖
    ax1 = axes[0, 0]
    for cluster_id in np.unique(cluster_result['labels']):
        mask = cluster_result['labels'] == cluster_id
        ax1.scatter(cluster_result['X_pca'][mask, 0], cluster_result['X_pca'][mask, 1], 
                   label=f'Cluster {cluster_id}', alpha=0.7)
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_title(f'{strategy} - {datasource}\nRisk-Enhanced Clustering (k={cluster_result["best_k"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 風險指標箱線圖
    ax2 = axes[0, 1]
    risk_metrics = ['annual_volatility', 'downside_risk', 'var_95_annual']
    available_metrics = [m for m in risk_metrics if m in df.columns]
    
    if available_metrics:
        metric = available_metrics[0]
        mask = (~df[metric].isna()) & np.isfinite(df[metric])
        if mask.sum() > 0:
            metric_data = df.loc[mask, metric]
            cluster_data = cluster_result['labels'][mask]
            
            cluster_groups = []
            cluster_labels = []
            for cluster_id in np.unique(cluster_data):
                cluster_mask = cluster_data == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_groups.append(metric_data[cluster_mask])
                    cluster_labels.append(f'Cluster {cluster_id}')
            
            if cluster_groups:
                ax2.boxplot(cluster_groups, labels=cluster_labels)
                ax2.set_ylabel(metric)
                ax2.set_title(f'{metric} by Cluster')
                ax2.grid(True, alpha=0.3)
    
    # 3. 績效指標箱線圖
    ax3 = axes[0, 2]
    perf_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
    available_perf = [m for m in perf_metrics if m in df.columns]
    
    if available_perf:
        metric = available_perf[0]
        mask = (~df[metric].isna()) & np.isfinite(df[metric])
        if mask.sum() > 0:
            metric_data = df.loc[mask, metric]
            cluster_data = cluster_result['labels'][mask]
            
            cluster_groups = []
            cluster_labels = []
            for cluster_id in np.unique(cluster_data):
                cluster_mask = cluster_data == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_groups.append(metric_data[cluster_mask])
                    cluster_labels.append(f'Cluster {cluster_id}')
            
            if cluster_groups:
                ax3.boxplot(cluster_groups, labels=cluster_labels)
                ax3.set_ylabel(metric)
                ax3.set_title(f'{metric} by Cluster')
                ax3.grid(True, alpha=0.3)
    
    # 4. 分群大小分布
    ax4 = axes[1, 0]
    cluster_counts = np.bincount(cluster_result['labels'])
    cluster_labels = [f'Cluster {i}' for i in range(len(cluster_counts))]
    
    ax4.pie(cluster_counts, labels=cluster_labels, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Cluster Size Distribution')
    
    # 5. 風險-收益散點圖
    ax5 = axes[1, 1]
    if 'total_return' in df.columns and 'annual_volatility' in df.columns:
        mask = (~df['total_return'].isna()) & (~df['annual_volatility'].isna()) & np.isfinite(df['total_return']) & np.isfinite(df['annual_volatility'])
        if mask.sum() > 0:
            for cluster_id in np.unique(cluster_result['labels']):
                cluster_mask = (cluster_result['labels'] == cluster_id) & mask
                ax5.scatter(df.loc[cluster_mask, 'annual_volatility'], 
                           df.loc[cluster_mask, 'total_return'],
                           label=f'Cluster {cluster_id}', alpha=0.7)
            ax5.set_xlabel('Annual Volatility')
            ax5.set_ylabel('Total Return')
            ax5.set_title('Risk-Return by Cluster')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
    
    # 6. 分群樹狀圖
    ax6 = axes[1, 2]
    dendrogram(cluster_result['linkage_matrix'], ax=ax6, orientation='top')
    ax6.set_title('Hierarchical Clustering Dendrogram')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{strategy}_{datasource}_risk_cluster_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_risk_comparison_analysis(all_results):
    """創建風險分群比較分析"""
    comparison_data = []
    
    for (strategy, datasource), result in all_results.items():
        if result and 'cluster_result' in result:
            cluster_result = result['cluster_result']
            comparison_data.append({
                'strategy': strategy,
                'datasource': datasource,
                'silhouette_score': cluster_result['silhouette_score'],
                'best_k': cluster_result['best_k'],
                'pca_components': cluster_result['pca_components'],
                'n_features': len(cluster_result['feature_names']),
                'data_size': result['data_size']
            })
    
    if not comparison_data:
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 創建比較圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Silhouette Score 比較
    ax1 = axes[0, 0]
    sns.boxplot(data=comparison_df, x='strategy', y='silhouette_score', ax=ax1)
    ax1.set_title('Risk-Enhanced Clustering Quality by Strategy')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 最佳分群數比較
    ax2 = axes[0, 1]
    sns.countplot(data=comparison_df, x='best_k', ax=ax2)
    ax2.set_title('Distribution of Optimal Cluster Numbers')
    
    # 數據源比較
    ax3 = axes[1, 0]
    sns.boxplot(data=comparison_df, x='datasource', y='silhouette_score', ax=ax3)
    ax3.set_title('Risk-Enhanced Clustering Quality by Datasource')
    
    # 特徵數量比較
    ax4 = axes[1, 1]
    sns.scatterplot(data=comparison_df, x='n_features', y='silhouette_score', 
                   hue='strategy', size='data_size', ax=ax4)
    ax4.set_title('Clustering Quality vs Feature Count')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/risk_clustering_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存比較結果
    comparison_df.to_csv(f'{OUTPUT_DIR}/risk_clustering_comparison.csv', index=False)
    
    return comparison_df

def generate_risk_cluster_report(all_results):
    """生成風險分群分析報告"""
    report = []
    report.append("=== 風險增強分群分析報告 ===\n")
    
    total_combinations = len(all_results)
    successful_clustering = sum(1 for r in all_results.values() if r and 'cluster_result' in r)
    
    report.append(f"1. 整體統計:")
    report.append(f"   - 總策略-數據源組合數: {total_combinations}")
    report.append(f"   - 成功分群組合數: {successful_clustering}")
    report.append(f"   - 分群成功率: {successful_clustering/total_combinations*100:.1f}%\n")
    
    if successful_clustering > 0:
        # 收集所有分群結果
        all_scores = []
        all_clusters = []
        
        for (strategy, datasource), result in all_results.items():
            if result and 'cluster_result' in result:
                cluster_result = result['cluster_result']
                all_scores.append(cluster_result['silhouette_score'])
                all_clusters.append(cluster_result['best_k'])
                
                report.append(f"2. {strategy} - {datasource}:")
                report.append(f"   - 最佳分群數: {cluster_result['best_k']}")
                report.append(f"   - Silhouette Score: {cluster_result['silhouette_score']:.3f}")
                report.append(f"   - PCA組件數: {cluster_result['pca_components']}")
                report.append(f"   - 特徵數量: {len(cluster_result['feature_names'])}")
                report.append(f"   - 數據樣本數: {result['data_size']}")
                report.append("")
        
        report.append(f"3. 整體分群品質:")
        report.append(f"   - 平均 Silhouette Score: {np.mean(all_scores):.3f}")
        report.append(f"   - 最高 Silhouette Score: {np.max(all_scores):.3f}")
        report.append(f"   - 最低 Silhouette Score: {np.min(all_scores):.3f}")
        report.append(f"   - 平均最佳分群數: {np.mean(all_clusters):.1f}")
        report.append(f"   - 分群數分布: {dict(pd.Series(all_clusters).value_counts().sort_index())}")
    
    # 保存報告
    with open(f'{OUTPUT_DIR}/risk_clustering_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("風險增強分群分析報告已生成完成！")
    print('\n'.join(report))

def main():
    """主函數"""
    print("開始風險增強分群分析...")
    
    # 載入數據
    merged_data = load_and_process_data()
    print(f"載入了 {len(merged_data)} 個策略-數據源組合")
    
    all_results = {}
    
    # 對每個策略-數據源組合進行風險增強分群
    for (strategy, datasource), df in merged_data.items():
        print(f"分析 {strategy} - {datasource}...")
        
        # 執行風險增強分群
        cluster_result = perform_risk_enhanced_clustering(df, strategy, datasource)
        
        if cluster_result is not None:
            # 創建可視化
            create_risk_cluster_visualizations(df, cluster_result, strategy, datasource)
            
            # 保存結果
            all_results[(strategy, datasource)] = {
                'cluster_result': cluster_result,
                'data_size': len(df)
            }
            
            # 將分群標籤添加到原始數據
            df['risk_cluster'] = cluster_result['labels']
            
            # 保存帶有風險分群標籤的數據
            output_csv = f'{OUTPUT_DIR}/{strategy}_{datasource}_risk_clustered.csv'
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            
            print(f"  - 最佳分群數: {cluster_result['best_k']}")
            print(f"  - Silhouette Score: {cluster_result['silhouette_score']:.3f}")
            print(f"  - 特徵數量: {len(cluster_result['feature_names'])}")
        else:
            print(f"  - 跳過（數據不足或特徵不足）")
    
    # 創建比較分析
    if all_results:
        comparison_df = create_risk_comparison_analysis(all_results)
        generate_risk_cluster_report(all_results)
    
    print(f"\n分析結果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 