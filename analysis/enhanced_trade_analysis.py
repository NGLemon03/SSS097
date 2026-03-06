# -*- coding: utf-8 -*-
"""
增強交易分析模組 - 2025-08-18 04:38
整合風險閥門回測、交易貢獻拆解、加碼梯度優化

作者：AI Assistant
路徑：#analysis/enhanced_trade_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 導入統一的風險閥門訊號計算函數
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SSS_EnsembleTab import compute_risk_valve_signals

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedTradeAnalyzer:
    """增強交易分析器"""
    
    def __init__(self, trades_df, benchmark_df=None):
        """
        初始化分析器
        
        Args:
            trades_df: 交易記錄DataFrame，需包含交易日期、權重變化、盈虧%等欄位
            benchmark_df: 基準指數DataFrame，需包含日期和收盤價
        """
        self.trades_df = trades_df.copy()
        self.benchmark_df = benchmark_df
        self.analysis_results = {}
        
        # 預處理交易數據
        self._preprocess_trades()
        
    def _preprocess_trades(self):
        """預處理交易數據"""
        # 日期
        if '交易日期' in self.trades_df.columns:
            self.trades_df['交易日期'] = pd.to_datetime(self.trades_df['交易日期'])
        elif 'date' in self.trades_df.columns:
            self.trades_df['交易日期'] = pd.to_datetime(self.trades_df['date'])

        # 排序
        self.trades_df = self.trades_df.sort_values('交易日期').reset_index(drop=True)

        # ✅ 兼容英文字段 → 交易類型
        if '交易類型' not in self.trades_df.columns and 'type' in self.trades_df.columns:
            self.trades_df['交易類型'] = self.trades_df['type'].astype(str).str.lower()

        # ✅ 先「尊重」既有交易類型；沒有才從權重變化推導
        if '交易類型' in self.trades_df.columns:
            self.trades_df['交易類型'] = self.trades_df['交易類型'].astype(str).str.lower()
        else:
            if '權重變化' in self.trades_df.columns:
                self.trades_df['交易類型'] = self.trades_df['權重變化'].apply(
                    lambda x: 'buy' if x > 0 else 'sell' if x < 0 else 'hold'
                )
            elif 'weight_change' in self.trades_df.columns:
                self.trades_df['權重變化'] = self.trades_df['weight_change']
                self.trades_df['交易類型'] = self.trades_df['權重變化'].apply(
                    lambda x: 'buy' if x > 0 else 'sell' if x < 0 else 'hold'
                )
            else:
                self.trades_df['交易類型'] = 'hold'

        # ✅ 權重變化缺或為 0 → 用交易類型方向補
        need_weight = ('權重變化' not in self.trades_df.columns) or \
                      (pd.to_numeric(self.trades_df['權重變化'], errors='coerce').fillna(0).abs().sum() == 0)
        if need_weight and '交易類型' in self.trades_df.columns:
            self.trades_df['權重變化'] = self.trades_df['交易類型'].map({'buy': 1.0, 'sell': -1.0}).fillna(0.0)

        # 累積權重（用數值後再做）
        self.trades_df['累積權重'] = pd.to_numeric(self.trades_df['權重變化'], errors='coerce').fillna(0).cumsum()

        # ✅ 正規化盈虧%：如果像 0.07 這種小數（<=2），自動視為 7%→×100
        if '盈虧%' in self.trades_df.columns:
            r = pd.to_numeric(self.trades_df['盈虧%'], errors='coerce')
            if np.nanmax(np.abs(r)) <= 2:
                r = r * 100.0
            self.trades_df['盈虧%'] = r
        else:
            if '每筆盈虧%' in self.trades_df.columns:
                self.trades_df['盈虧%'] = pd.to_numeric(self.trades_df['每筆盈虧%'], errors='coerce')
            elif {'交易日期', '總資產'}.issubset(self.trades_df.columns):
                self.trades_df = self.trades_df.sort_values('交易日期').reset_index(drop=True)
                self.trades_df['盈虧%'] = self.trades_df['總資產'].pct_change().fillna(0) * 100.0
            else:
                self.trades_df['盈虧%'] = 0.0
        
    def risk_valve_backtest(self, risk_rules=None):
        """
        風險閥門回測
        
        Args:
            risk_rules: 風險規則字典，預設包含TWII斜率和ATR規則
        """
        if risk_rules is None:
            risk_rules = {
                'twii_slope_20d': {'threshold': 0, 'window': 20},
                'twii_slope_60d': {'threshold': 0, 'window': 60},
                'atr_threshold': {'window': 20, 'multiplier': 1.5}
            }
            
        print("=== 風險閥門回測分析 ===")
        print(f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 模擬風險閥門觸發
        self._simulate_risk_valves(risk_rules)
        
        # 計算風險閥門效果
        self._calculate_risk_valve_impact()
        
        return self.analysis_results.get('risk_valve', {})
    
    def _simulate_risk_valves(self, risk_rules):
        """模擬風險閥門觸發"""
        if self.benchmark_df is None:
            print("警告：缺少基準數據，無法進行風險閥門回測")
            return
            
        # 計算TWII技術指標
        benchmark = self.benchmark_df.copy()
        
        # 🔧 索引名與欄位名重複時，取消索引名，避免 pandas 歧義錯誤
        if benchmark.index.name == '日期':
            benchmark.index.name = None
        
        # 防禦性檢查：如果沒有 日期 欄就從 index 建立
        if '日期' not in benchmark.columns:
            print("警告：基準數據缺少 '日期' 欄位，嘗試從 index 建立")
            if benchmark.index.name == '日期':
                benchmark['日期'] = benchmark.index
            elif isinstance(benchmark.index, pd.DatetimeIndex):
                benchmark['日期'] = benchmark.index
            else:
                print("錯誤：無法從 index 建立日期欄位，請檢查基準數據格式")
                return
        
        benchmark['日期'] = pd.to_datetime(benchmark['日期'])
        benchmark = benchmark.sort_values('日期').reset_index(drop=True)
        
        # 使用統一的風險閥門訊號計算函數
        sig = compute_risk_valve_signals(
            benchmark,
            slope20_thresh= risk_rules.get('twii_slope_20d', {}).get('threshold', 0.0),
            slope60_thresh= risk_rules.get('twii_slope_60d', {}).get('threshold', 0.0),
            atr_win=20, atr_ref_win=60,
            atr_ratio_mult= risk_rules.get('atr_threshold', {}).get('multiplier', 1.0),
            use_slopes=True,
            slope_method="polyfit",
            atr_cmp="gt"
        )
        benchmark = benchmark.join(sig[["slope_20d","slope_60d","atr","atr_ratio","risk_trigger"]], how="left")
        benchmark.rename(columns={"risk_trigger":"risk_valve_triggered"}, inplace=True)
        
        self.benchmark_enhanced = benchmark
        
    def _calculate_risk_valve_impact(self):
        """計算風險閥門對績效的影響"""
        if not hasattr(self, 'benchmark_enhanced'):
            return
            
        # 防禦性檢查：確保有必要的欄位
        if '日期' not in self.benchmark_enhanced.columns:
            print("警告：benchmark_enhanced 缺少 '日期' 欄位，無法計算風險閥門影響")
            return
            
        if 'risk_valve_triggered' not in self.benchmark_enhanced.columns:
            print("警告：benchmark_enhanced 缺少 'risk_valve_triggered' 欄位，無法計算風險閥門影響")
            return
            
        # 找出風險閥門觸發的期間
        risk_periods = self.benchmark_enhanced[
            self.benchmark_enhanced['risk_valve_triggered']
        ]['日期'].tolist()
        
        # 分析風險期間的交易表現
        risk_trades = self.trades_df[
            self.trades_df['交易日期'].isin(risk_periods)
        ]
        
        normal_trades = self.trades_df[
            ~self.trades_df['交易日期'].isin(risk_periods)
        ]
        
        # 新增（口徑一致：只對賣出列計數）
        risk_sells = risk_trades[risk_trades['交易類型'].str.lower() == 'sell']
        normal_sells = normal_trades[normal_trades['交易類型'].str.lower() == 'sell']
        
        # 計算對比指標
        risk_metrics = self._calculate_trade_metrics(risk_trades)
        normal_metrics = self._calculate_trade_metrics(normal_trades)
        
        self.analysis_results['risk_valve'] = {
            'risk_periods_count': len(risk_periods),
            'risk_trades_count': int(len(risk_sells)),        # <- 改成賣出筆數
            'normal_trades_count': int(len(normal_sells)),    # <- 改成賣出筆數
            'risk_periods_metrics': risk_metrics,
            'normal_periods_metrics': normal_metrics,
            'improvement_potential': {
                'mdd_reduction': normal_metrics.get('mdd', 0) - risk_metrics.get('mdd', 0),
                'pf_improvement': risk_metrics.get('profit_factor', 0) - normal_metrics.get('profit_factor', 0),
                'win_rate_improvement': risk_metrics.get('win_rate', 0) - normal_metrics.get('win_rate', 0)
            }
        }
        
        # 輸出結果
        print(f"風險閥門觸發期間數：{len(risk_periods)}")
        print(f"風險期間交易數：{len(risk_sells)}")
        print(f"正常期間交易數：{len(normal_sells)}")
        print("\n風險期間表現：")
        self._print_metrics(risk_metrics)
        print("\n正常期間表現：")
        self._print_metrics(normal_metrics)
        
    def trade_contribution_analysis(self):
        """交易貢獻拆解分析"""
        print("\n=== 交易貢獻拆解分析 ===")
        print(f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 識別加碼/減碼階段
        self._identify_trading_phases()
        
        # 分析各階段貢獻
        self._analyze_phase_contributions()
        
        return self.analysis_results.get('phase_analysis', {})
    
    def _identify_trading_phases(self):
        """識別交易階段"""
        # 基於權重變化識別階段
        weight_changes = self.trades_df['權重變化'].values
        phases = []
        current_phase = {'type': None, 'start_idx': 0, 'trades': []}
        
        for i, change in enumerate(weight_changes):
            if change > 0:  # 加碼
                if current_phase['type'] != 'accumulation':
                    if current_phase['type'] is not None:
                        phases.append(current_phase.copy())
                    current_phase = {'type': 'accumulation', 'start_idx': i, 'trades': []}
                current_phase['trades'].append(i)
            elif change < 0:  # 減碼
                if current_phase['type'] != 'distribution':
                    if current_phase['type'] is not None:
                        phases.append(current_phase.copy())
                    current_phase = {'type': 'distribution', 'start_idx': i, 'trades': []}
                current_phase['trades'].append(i)
            else:  # 持有
                if current_phase['type'] is not None:
                    current_phase['trades'].append(i)
                    
        # 添加最後一個階段
        if current_phase['type'] is not None:
            phases.append(current_phase)
            
        self.trading_phases = phases
        
    def _analyze_phase_contributions(self):
        """分析各階段貢獻"""
        phase_results = {}
        
        for phase in self.trading_phases:
            phase_type = phase['type']
            trade_indices = phase['trades']
            
            # 提取該階段的交易
            phase_trades = self.trades_df.iloc[trade_indices]
            
            # 計算階段指標
            phase_metrics = self._calculate_trade_metrics(phase_trades)
            
            # 計算對總績效的貢獻
            total_return = pd.to_numeric(
                self.trades_df.loc[self.trades_df['交易類型'].str.lower() == 'sell', '盈虧%'],
                errors='coerce'
            ).dropna().sum()
            phase_return = pd.to_numeric(
                phase_trades.loc[phase_trades['交易類型'].str.lower() == 'sell', '盈虧%'],
                errors='coerce'
            ).dropna().sum()
            contribution_ratio = phase_return / total_return if total_return != 0 else 0
            
            phase_results[phase_type] = {
                'trade_count': len(phase_trades),
                'total_return': phase_return,
                'contribution_ratio': contribution_ratio,
                'metrics': phase_metrics,
                'start_date': phase_trades['交易日期'].iloc[0],
                'end_date': phase_trades['交易日期'].iloc[-1]
            }
            
        self.analysis_results['phase_analysis'] = phase_results
        
        # 輸出結果
        print(f"識別出 {len(self.trading_phases)} 個交易階段")
        for phase_type, result in phase_results.items():
            print(f"\n{phase_type} 階段：")
            print(f"  交易數：{result['trade_count']}")
            print(f"  總報酬：{result['total_return']:.2f}%")
            print(f"  貢獻比：{result['contribution_ratio']:.2%}")
            print(f"  期間：{result['start_date'].strftime('%Y-%m-%d')} 到 {result['end_date'].strftime('%Y-%m-%d')}")
            
    def position_gradient_optimization(self, min_interval_days=3, cooldown_days=7):
        """
        加碼梯度優化分析
        
        Args:
            min_interval_days: 最小加碼間距（天）
            cooldown_days: 冷卻期（天）
        """
        print(f"\n=== 加碼梯度優化分析 ===")
        print(f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"最小間距：{min_interval_days} 天，冷卻期：{cooldown_days} 天")
        
        # 分析當前加碼模式
        current_pattern = self._analyze_current_accumulation_pattern()
        
        # 模擬優化後的加碼策略
        optimized_pattern = self._simulate_optimized_accumulation(
            min_interval_days, cooldown_days
        )
        
        # 對比分析
        self._compare_accumulation_strategies(current_pattern, optimized_pattern)
        
        return self.analysis_results.get('gradient_optimization', {})
    
    def _analyze_current_accumulation_pattern(self):
        """分析當前加碼模式，保證回傳欄位一致（即使交易少於2筆）"""
        buy_trades = self.trades_df[self.trades_df['權重變化'] > 0].copy()

        # 保證回傳欄位：intervals, avg_interval, min_interval, max_consecutive, consecutive_buys
        if len(buy_trades) < 2:
            return {
                'intervals': [],
                'avg_interval': 0,
                'min_interval': 0,
                'max_consecutive': 0,
                'consecutive_buys': 0
            }

        # 計算加碼間距
        buy_trades = buy_trades.sort_values('交易日期')
        intervals = []
        consecutive_buys = 0
        max_consecutive = 0

        for i in range(1, len(buy_trades)):
            interval = (buy_trades.iloc[i]['交易日期'] - buy_trades.iloc[i-1]['交易日期']).days
            intervals.append(interval)

            if interval <= 1:  # 視作連續加碼（可以改成參數化）
                consecutive_buys += 1
                max_consecutive = max(max_consecutive, consecutive_buys)
            else:
                consecutive_buys = 0

        return {
            'intervals': intervals,
            'avg_interval': float(np.mean(intervals)) if intervals else 0.0,
            'min_interval': int(min(intervals)) if intervals else 0,
            'max_consecutive': int(max_consecutive),
            'consecutive_buys': int(consecutive_buys)
        }
        
    def _simulate_optimized_accumulation(self, min_interval_days, cooldown_days):
        """模擬優化後的加碼策略"""
        buy_trades = self.trades_df[self.trades_df['權重變化'] > 0].copy()
        
        if len(buy_trades) < 2:
            return {'filtered_trades': [], 'reduction_ratio': 0}
            
        buy_trades = buy_trades.sort_values('交易日期')
        filtered_trades = [buy_trades.iloc[0]]  # 保留第一筆
        last_buy_date = buy_trades.iloc[0]['交易日期']
        in_cooldown = False
        
        for i in range(1, len(buy_trades)):
            current_trade = buy_trades.iloc[i]
            days_since_last = (current_trade['交易日期'] - last_buy_date).days
            
            if days_since_last >= min_interval_days and not in_cooldown:
                filtered_trades.append(current_trade)
                last_buy_date = current_trade['交易日期']
                
                # 檢查是否需要進入冷卻期
                if len(filtered_trades) >= 3:  # 連續3筆加碼後進入冷卻
                    in_cooldown = True
            elif in_cooldown and days_since_last >= cooldown_days:
                in_cooldown = False
                
        reduction_ratio = 1 - len(filtered_trades) / len(buy_trades)
        
        return {
            'filtered_trades': filtered_trades,
            'reduction_ratio': reduction_ratio,
            'original_count': len(buy_trades),
            'optimized_count': len(filtered_trades)
        }
        
    def _compare_accumulation_strategies(self, current, optimized):
        """對比加碼策略（防禦式取值）"""
        cur_avg = current.get('avg_interval', 0.0)
        cur_min = current.get('min_interval', 0)
        cur_max_consec = current.get('max_consecutive', 0)

        print(f"\n當前加碼模式：")
        print(f"  平均間距：{cur_avg:.1f} 天")
        print(f"  最小間距：{cur_min} 天")
        print(f"  最大連續加碼：{cur_max_consec} 筆")

        print(f"\n優化後加碼模式：")
        print(f"  加碼次數減少：{optimized.get('reduction_ratio', 0.0):.1%}")
        print(f"  從 {optimized.get('original_count', 0)} 筆減少到 {optimized.get('optimized_count', 0)} 筆")

        # 若有優化後 trades，再計算指標
        if optimized.get('filtered_trades'):
            optimized_trades = pd.DataFrame(optimized['filtered_trades'])
            optimized_metrics = self._calculate_trade_metrics(optimized_trades)
            print(f"\n優化後指標：")
            self._print_metrics(optimized_metrics)

        self.analysis_results['gradient_optimization'] = {
            'current_pattern': current,
            'optimized_pattern': optimized
        }
        
    def _calculate_trade_metrics(self, trades_df):
        """計算交易指標（僅以賣出列計算實現報酬）"""
        if len(trades_df) == 0:
            return {}

        # 只取賣出列（避免買單 0 報酬稀釋指標）
        mask_sell = trades_df.get('交易類型', pd.Series('sell', index=trades_df.index)).astype(str).str.lower().eq('sell')
        sell_returns = pd.to_numeric(trades_df.loc[mask_sell, '盈虧%'], errors='coerce').dropna()

        if sell_returns.empty:
            return {'trade_count': int(len(trades_df)), 'sell_count': 0}

        # 以百分比數值為單位，例如 12.3 代表 +12.3%
        r = sell_returns.values
        # ✅ 二次保險：若還是小數制，就乘 100
        if np.nanmax(np.abs(r)) <= 2:
            r = r * 100.0
        win_rate = float((r > 0).sum()) / len(r)
        avg_win = float(r[r > 0].mean()) if (r > 0).any() else 0.0
        avg_loss = float(r[r < 0].mean()) if (r < 0).any() else 0.0
        profit_factor = (abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'))

        # 風險指標（改用權益曲線計 MDD；把百分比轉成小數）
        rf = r / 100.0
        equity = (1.0 + rf).cumprod()
        # 使用 numpy 的 maximum.accumulate 避免 ndarray.cummax() 錯誤
        run_max = np.maximum.accumulate(equity)
        dd = equity / run_max - 1.0
        mdd = float(dd.min()) if len(dd) else 0.0

        volatility = float(rf.std())  # 以小數衡量
        sharpe_ratio = float(rf.mean() / volatility) if volatility > 0 else 0.0
        total_return = float(r.sum())  # 仍用百分比單位呈現

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'mdd': mdd,
            'trade_count': int(len(trades_df)),
            'sell_count': int(len(sell_returns))
        }
        
    def _print_metrics(self, metrics):
        """輸出指標"""
        if not metrics:
            return
            
        print(f"  勝率：{metrics.get('win_rate', 0):.2%}")
        print(f"  平均獲利：{metrics.get('avg_win', 0):.2f}%")
        print(f"  平均虧損：{metrics.get('avg_loss', 0):.2f}%")
        print(f"  Profit Factor：{metrics.get('profit_factor', 0):.2f}")
        print(f"  總報酬：{metrics.get('total_return', 0):.2f}%")
        print(f"  波動率：{metrics.get('volatility', 0):.2f}%")
        print(f"  夏普比率：{metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤：{metrics.get('mdd', 0):.2%}")
        
    def generate_comprehensive_report(self):
        """生成綜合報告"""
        print("\n=== 生成綜合分析報告 ===")
        
        # 執行所有分析
        self.risk_valve_backtest()
        self.trade_contribution_analysis()
        self.position_gradient_optimization()
        
        # 生成報告摘要
        report_summary = {
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trades': len(self.trades_df),
            'analysis_results': self.analysis_results
        }
        
        print(f"\n分析完成！總共分析了 {len(self.trades_df)} 筆交易")
        print(f"報告生成時間：{report_summary['analysis_timestamp']}")
        
        return report_summary
        
    def plot_enhanced_analysis(self):
        """繪製增強分析圖表"""
        if not self.analysis_results:
            print("請先執行分析")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('增強交易分析圖表', fontsize=16)
        
        # 1. 風險閥門觸發時序
        if 'risk_valve' in self.analysis_results and hasattr(self, 'benchmark_enhanced'):
            # 防禦性檢查：確保有必要的欄位
            if '日期' not in self.benchmark_enhanced.columns or '收盤價' not in self.benchmark_enhanced.columns:
                print("警告：benchmark_enhanced 缺少必要欄位，跳過風險閥門時序圖")
            elif 'risk_valve_triggered' not in self.benchmark_enhanced.columns:
                print("警告：benchmark_enhanced 缺少 'risk_valve_triggered' 欄位，跳過風險閥門時序圖")
            else:
                ax1 = axes[0, 0]
                self.benchmark_enhanced.plot(x='日期', y='收盤價', ax=ax1, color='blue', alpha=0.7)
                
                risk_dates = self.benchmark_enhanced[
                    self.benchmark_enhanced['risk_valve_triggered']
                ]['日期']
                
                if len(risk_dates) > 0:
                    ax1.scatter(risk_dates, 
                               self.benchmark_enhanced.loc[
                                   self.benchmark_enhanced['risk_valve_triggered'], '收盤價'
                               ], 
                               color='red', s=50, alpha=0.8, label='風險閥門觸發')
                    ax1.legend()
                    
                ax1.set_title('風險閥門觸發時序')
                ax1.set_ylabel('收盤價')
        
        # 2. 交易階段貢獻
        if 'phase_analysis' in self.analysis_results:
            ax2 = axes[0, 1]
            phases = list(self.analysis_results['phase_analysis'].keys())
            contributions = [self.analysis_results['phase_analysis'][p]['contribution_ratio'] 
                           for p in phases]
            
            ax2.bar(phases, contributions, color=['green', 'red', 'blue'])
            ax2.set_title('各階段貢獻比例')
            ax2.set_ylabel('貢獻比例')
            ax2.tick_params(axis='x', rotation=45)
            
        # 3. 加碼間距分布
        if 'gradient_optimization' in self.analysis_results:
            ax3 = axes[1, 0]
            current_intervals = self.analysis_results['gradient_optimization']['current_pattern']['intervals']
            
            if current_intervals:
                ax3.hist(current_intervals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(np.mean(current_intervals), color='red', linestyle='--', 
                           label=f'平均: {np.mean(current_intervals):.1f}天')
                ax3.legend()
                
            ax3.set_title('加碼間距分布')
            ax3.set_xlabel('間距（天）')
            ax3.set_ylabel('頻次')
            
        # 4. 優化前後對比
        if 'gradient_optimization' in self.analysis_results:
            ax4 = axes[1, 1]
            opt = self.analysis_results['gradient_optimization']
            
            categories = ['原始加碼次數', '優化後加碼次數']
            values = [opt['optimized_pattern']['original_count'], 
                     opt['optimized_pattern']['optimized_count']]
            
            ax4.bar(categories, values, color=['lightcoral', 'lightgreen'])
            ax4.set_title('加碼策略優化效果')
            ax4.set_ylabel('加碼次數')
            
            # 添加數值標籤
            for i, v in enumerate(values):
                ax4.text(i, v + max(values) * 0.01, str(v), ha='center', va='bottom')
                
        plt.tight_layout()
        plt.show()
        
        return fig
        
def main():
    """主函數 - 示範用法"""
    print("增強交易分析模組")
    print("路徑：#analysis/enhanced_trade_analysis.py")
    print("創建時間：2025-08-18 04:38")
    print("\n使用方法：")
    print("1. 創建分析器實例：analyzer = EnhancedTradeAnalyzer(trades_df, benchmark_df)")
    print("2. 執行風險閥門回測：analyzer.risk_valve_backtest()")
    print("3. 執行交易貢獻拆解：analyzer.trade_contribution_analysis()")
    print("4. 執行加碼梯度優化：analyzer.position_gradient_optimization()")
    print("5. 生成綜合報告：analyzer.generate_comprehensive_report()")
    print("6. 繪製分析圖表：analyzer.plot_enhanced_analysis()")

if __name__ == "__main__":
    main()
