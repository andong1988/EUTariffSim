"""
Ordered Probit模型可视化工具

生成6类可视化图表：
1. 拟合优度：混淆矩阵和准确率
2. 阈值S型曲线：展示Ordered Probit的决策边界
3. 国家权重对比：各国在不同理论维度上的权重
4. 残差诊断：残差分布和QQ图
5. 边际效应：理论得分变化对投票概率的影响
6. 模型比较：不同模型的表现对比
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple
import os
import json


class ProbitVisualizer:
    """Ordered Probit模型可视化器"""
    
    def __init__(self, output_dir: str = "results/probit"):
        """
        初始化可视化器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set English font
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Vote type mapping
        self.vote_labels = {0: 'Oppose', 1: 'Abstain', 2: 'Approve'}
        self.vote_colors = {0: '#ff6b6b', 1: '#ffd93d', 2: '#6bcb77'}
        
        # Theory dimension names
        self.theory_names = [
            'Structural Economic Constraints',
            'Political Economy Mechanisms',
            'External Strategic Interactions'
        ]
        self.theory_names_en = [
            'Structural Economic Constraints',
            'Political Economy Mechanisms',
            'External Strategic Interactions'
        ]
    
    def plot_goodness_of_fit(self, actual: List[int], predicted: List[int],
                            country_names: List[str] = None,
                            save_name: str = "goodness_of_fit.png"):
        """
        1. 拟合优度可视化
        
        参数:
            actual: 实际投票列表 (0, 1, 2)
            predicted: 预测投票列表 (0, 1, 2)
            country_names: 国家名称列表（可选，用于按国家分解）
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ordered Probit Model Goodness of Fit', fontsize=16, fontweight='bold')
        
        # 1.1 Confusion Matrix
        cm = confusion_matrix(actual, predicted)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.vote_labels[i] for i in range(3)],
                   yticklabels=[self.vote_labels[i] for i in range(3)],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Actual Vote')
        axes[0, 0].set_xlabel('Predicted Vote')
        
        # Calculate accuracy
        accuracy = np.mean(np.array(actual) == np.array(predicted))
        axes[0, 0].text(0.5, -0.3, f'Overall Accuracy: {accuracy:.2%}', 
                       ha='center', transform=axes[0, 0].transAxes, fontsize=11)
        
        # 1.2 各类别的准确率
        class_accuracy = []
        for i in range(3):
            mask = np.array(actual) == i
            if np.sum(mask) > 0:
                acc = np.mean(np.array(actual)[mask] == np.array(predicted)[mask])
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        bars = axes[0, 1].bar(range(3), class_accuracy,
                            color=[self.vote_colors[i] for i in range(3)])
        axes[0, 1].set_xticks(range(3))
        axes[0, 1].set_xticklabels([self.vote_labels[i] for i in range(3)])
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Class-wise Prediction Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 1.1)
        
        # Show values on bars
        for bar, acc in zip(bars, class_accuracy):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{acc:.2%}', ha='center', va='bottom')
        
        # 1.3 Predicted Probability Distribution (Histogram)
        # Probability data needed but not in signature, skip for now
        axes[1, 0].text(0.5, 0.5, 'Predicted probability data needed\nfor probability distribution plot',
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=10, style='italic')
        axes[1, 0].set_title('Predicted Probability Distribution (To Implement)', fontsize=12, fontweight='bold')
        
        # 1.4 Overall Statistics
        stats_text = f"""Model Performance Statistics:

Total Samples: {len(actual)}
Correct Predictions: {np.sum(np.array(actual) == np.array(predicted))}
Overall Accuracy: {accuracy:.2%}

Class-wise Accuracy:
• Oppose: {class_accuracy[0]:.2%}
• Abstain: {class_accuracy[1]:.2%}
• Approve: {class_accuracy[2]:.2%}

Class Distribution:
• Oppose: {np.mean(np.array(actual)==0):.2%}
• Abstain: {np.mean(np.array(actual)==1):.2%}
• Approve: {np.mean(np.array(actual)==2):.2%}
"""
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, family='monospace', verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Model Statistics Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"拟合优度图表已保存到 {filepath}")
        plt.close()
    
    def plot_threshold_curves(self, alpha1: float, alpha2: float,
                             save_name: str = "threshold_curves.png"):
        """
        2. 阈值S型曲线可视化
        
        参数:
            alpha1: 第一个阈值
            alpha2: 第二个阈值
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Ordered Probit Model Threshold Decision Boundaries', fontsize=16, fontweight='bold')
        
        # Generate a range of eta values
        eta = np.linspace(-3, 3, 300)
        
        # Calculate probabilities for three categories
        p_oppose = norm.cdf(alpha1 - eta)
        p_abstain = norm.cdf(alpha2 - eta) - norm.cdf(alpha1 - eta)
        p_approve = 1 - norm.cdf(alpha2 - eta)
        
        # 2.1 Probability Curves
        axes[0].plot(eta, p_oppose, label=f'Oppose (Y<=alpha1, alpha1={alpha1:.3f})',
                    color=self.vote_colors[0], linewidth=2.5)
        axes[0].plot(eta, p_abstain, label=f'Abstain (alpha1<Y<=alpha2, alpha2={alpha2:.3f})',
                    color=self.vote_colors[1], linewidth=2.5)
        axes[0].plot(eta, p_approve, label=f'Approve (Y>alpha2)',
                    color=self.vote_colors[2], linewidth=2.5)
        
        # Draw threshold lines
        axes[0].axvline(x=alpha1, color='gray', linestyle='--', alpha=0.7)
        axes[0].axvline(x=alpha2, color='gray', linestyle='--', alpha=0.7)
        axes[0].text(alpha1, 0.95, f'alpha1={alpha1:.3f}', rotation=90,
                    ha='right', va='top', fontsize=9)
        axes[0].text(alpha2, 0.95, f'alpha2={alpha2:.3f}', rotation=90,
                    ha='right', va='top', fontsize=9)
        
        axes[0].set_xlabel('Linear Combination eta = X*beta', fontsize=11)
        axes[0].set_ylabel('Probability', fontsize=11)
        axes[0].set_title('Ordered Probit Probability Functions', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.1)
        
        # 2.2 Decision Regions
        eta_range = np.linspace(-3, 3, 300)
        decision = np.zeros_like(eta_range)
        decision[eta_range <= alpha1] = 0  # Oppose
        decision[(eta_range > alpha1) & (eta_range <= alpha2)] = 1  # Abstain
        decision[eta_range > alpha2] = 2  # Approve
        
        # Draw decision regions
        axes[1].fill_between(eta_range, 0, 1, where=(eta_range <= alpha1),
                           alpha=0.3, color=self.vote_colors[0], label='Oppose Region')
        axes[1].fill_between(eta_range, 0, 1, where=(eta_range > alpha1) & (eta_range <= alpha2),
                           alpha=0.3, color=self.vote_colors[1], label='Abstain Region')
        axes[1].fill_between(eta_range, 0, 1, where=(eta_range > alpha2),
                           alpha=0.3, color=self.vote_colors[2], label='Approve Region')
        
        # Draw threshold lines
        axes[1].axvline(x=alpha1, color='gray', linestyle='--', linewidth=2)
        axes[1].axvline(x=alpha2, color='gray', linestyle='--', linewidth=2)
        
        axes[1].set_xlabel('Linear Combination eta = X*beta', fontsize=11)
        axes[1].set_yticks([])
        axes[1].set_title('Decision Regions', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].set_xlim(-3, 3)
        axes[1].set_ylim(0, 1)
        
        # Add explanation text
        explanation = f"""Threshold Parameters:
alpha1 = {alpha1:.4f}
alpha2 = {alpha2:.4f}

Decision Rules:
• eta <= alpha1 -> Oppose
• alpha1 < eta <= alpha2 -> Abstain  
• eta > alpha2 -> Approve"""
        axes[1].text(0.02, 0.98, explanation, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"阈值曲线图已保存到 {filepath}")
        plt.close()
    
    def plot_country_weights(self, weights: np.ndarray, country_names: List[str],
                            save_name: str = "country_weights.png"):
        """
        3. 国家权重对比可视化
        
        参数:
            weights: 权重矩阵 shape: (n_countries, n_theories)
            country_names: 国家名称列表
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Country Theory Dimension Weight Estimates', fontsize=16, fontweight='bold')
        
        n_countries, n_theories = weights.shape
        x = np.arange(n_countries)
        width = 0.25
        
        # 3.1 堆叠柱状图
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bottom = np.zeros(n_countries)
        
        for i in range(n_theories):
            axes[0, 0].bar(x, weights[:, i], width, bottom=bottom,
                          label=self.theory_names[i], color=colors[i])
            bottom += weights[:, i]
        
        axes[0, 0].set_xlabel('Country', fontsize=11)
        axes[0, 0].set_ylabel('Weight', fontsize=11)
        axes[0, 0].set_title('Country Theory Weight Distribution (Stacked)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(country_names, rotation=45, ha='right')
        axes[0, 0].legend(loc='upper right', fontsize=9)
        axes[0, 0].set_ylim(0, 1.1)
        
        # 3.2 分组柱状图
        x_pos = np.arange(n_countries)
        width = 0.25
        
        axes[0, 1].bar(x_pos - width, weights[:, 0], width, 
                      label=self.theory_names[0], color=colors[0])
        axes[0, 1].bar(x_pos, weights[:, 1], width,
                      label=self.theory_names[1], color=colors[1])
        axes[0, 1].bar(x_pos + width, weights[:, 2], width,
                      label=self.theory_names[2], color=colors[2])
        
        axes[0, 1].set_xlabel('Country', fontsize=11)
        axes[0, 1].set_ylabel('Weight', fontsize=11)
        axes[0, 1].set_title('Country Theory Weight Comparison (Grouped)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(country_names, rotation=45, ha='right')
        axes[0, 1].legend(loc='upper right', fontsize=9)
        
        # 3.3 热力图
        im = axes[1, 0].imshow(weights.T, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_xticks(range(n_countries))
        axes[1, 0].set_xticklabels(country_names, rotation=45, ha='right')
        axes[1, 0].set_yticks(range(n_theories))
        axes[1, 0].set_yticklabels([t[:8]+'...' for t in self.theory_names])
        axes[1, 0].set_title('Weight Heatmap', fontsize=12, fontweight='bold')
        
        # 添加数值标注
        for i in range(n_theories):
            for j in range(n_countries):
                text = axes[1, 0].text(j, i, f'{weights[j, i]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 3.4 雷达图（显示所有国家）
        # 显示所有国家
        n_selected = n_countries
        selected_indices = np.arange(n_countries)
        
        angles = np.linspace(0, 2 * np.pi, n_theories, endpoint=False).tolist()
        angles += angles[:1]
        
        ax_radar = fig.add_subplot(2, 2, 4, projection='polar')
        
        radar_colors = plt.cm.tab10(np.linspace(0, 1, n_selected))
        
        for idx, country_idx in enumerate(selected_indices):
            values = weights[country_idx].tolist()
            values += values[:1]
            
            ax_radar.plot(angles, values, 'o-', linewidth=1.5,
                         label=country_names[country_idx], color=radar_colors[idx])
            ax_radar.fill(angles, values, alpha=0.1, color=radar_colors[idx])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(['w_market', 'w_political', 'w_institutional'], fontsize=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('All Countries Weight Radar Chart', fontsize=12, fontweight='bold', pad=20)
        ax_radar.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=7)
        ax_radar.grid(True)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"国家权重图已保存到 {filepath}")
        plt.close()
    
    def plot_residuals(self, residuals: List[float], save_name: str = "residuals.png"):
        """
        4. 残差诊断可视化
        
        参数:
            residuals: 残差列表
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ordered Probit Model Residual Diagnostics', fontsize=16, fontweight='bold')
        
        residuals = np.array(residuals)
        
        # 4.1 残差直方图
        axes[0, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Residual Value', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        
        # Add statistical information
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[0, 0].text(0.95, 0.95, f'Mean: {mean_res:.3f}\nStd: {std_res:.3f}',
                        transform=axes[0, 0].transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4.2 QQ图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Residual QQ Plot (Normality Test)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 4.3 残差箱线图
        unique_residuals = np.unique(residuals)
        axes[1, 0].boxplot([residuals[residuals == r] for r in unique_residuals],
                          labels=[f'{r:.0f}' for r in unique_residuals])
        axes[1, 0].set_xlabel('Residual Value', fontsize=11)
        axes[1, 0].set_ylabel('Distribution', fontsize=11)
        axes[1, 0].set_title('Residual Boxplot', fontsize=12, fontweight='bold')
        
        # 4.4 Residual Distribution Pie Chart
        residual_counts = {r: np.sum(residuals == r) for r in unique_residuals}
        labels = [f'Residual={r}\n({count} obs)' for r, count in residual_counts.items()]
        colors = ['#ff6b6b' if r < 0 else '#6bcb77' if r > 0 else '#ffd93d' 
                 for r in residual_counts.keys()]
        
        axes[1, 1].pie(residual_counts.values(), labels=labels, colors=colors,
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Residual Distribution Proportion', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"残差诊断图已保存到 {filepath}")
        plt.close()
    
    def plot_marginal_effects(self, mle_model, country_idx: int, country_name: str,
                             theory_idx: int, save_name: str = "marginal_effects.png"):
        """
        5. 边际效应可视化
        
        参数:
            mle_model: OrderedProbitMLE模型实例
            country_idx: 国家索引
            country_name: 国家名称
            theory_idx: 理论维度索引
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{country_name} - Marginal Effects of {self.theory_names[theory_idx]}',
                    fontsize=16, fontweight='bold')
        
        # 生成理论得分范围
        theory_scores = np.linspace(0, 10, 100)
        
        # 固定其他两个维度为均值（或特定值）
        scores_matrix = np.zeros((100, 3))
        scores_matrix[:, theory_idx] = theory_scores
        
        # 计算概率
        p_oppose = []
        p_abstain = []
        p_approve = []
        
        for scores in scores_matrix:
            p = mle_model.predict_proba(scores, country_idx)
            p_oppose.append(p[0])
            p_abstain.append(p[1])
            p_approve.append(p[2])
        
        # 5.1 Probability vs Theory Score Curves
        axes[0].plot(theory_scores, p_oppose, label='Oppose', 
                    color=self.vote_colors[0], linewidth=2.5)
        axes[0].plot(theory_scores, p_abstain, label='Abstain',
                    color=self.vote_colors[1], linewidth=2.5)
        axes[0].plot(theory_scores, p_approve, label='Approve',
                    color=self.vote_colors[2], linewidth=2.5)
        
        axes[0].set_xlabel(f'{self.theory_names[theory_idx]} Score', fontsize=11)
        axes[0].set_ylabel('Vote Probability', fontsize=11)
        axes[0].set_title('Vote Probability vs Theory Score', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.1)
        
        # 5.2 Marginal Effects (Derivatives)
        dp_oppose = np.gradient(p_oppose, theory_scores)
        dp_abstain = np.gradient(p_abstain, theory_scores)
        dp_approve = np.gradient(p_approve, theory_scores)
        
        axes[1].plot(theory_scores, dp_oppose, label='Oppose Marginal Effect',
                    color=self.vote_colors[0], linewidth=2.5)
        axes[1].plot(theory_scores, dp_abstain, label='Abstain Marginal Effect',
                    color=self.vote_colors[1], linewidth=2.5)
        axes[1].plot(theory_scores, dp_approve, label='Approve Marginal Effect',
                    color=self.vote_colors[2], linewidth=2.5)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        axes[1].set_xlabel(f'{self.theory_names[theory_idx]} Score', fontsize=11)
        axes[1].set_ylabel('Marginal Effect (dP/dX)', fontsize=11)
        axes[1].set_title('Marginal Effect Curves', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"边际效应图已保存到 {filepath}")
        plt.close()
    
    def plot_model_comparison(self, results: Dict[str, Dict],
                            save_name: str = "model_comparison.png"):
        """
        6. 模型比较可视化
        
        参数:
            results: 模型结果字典 {model_name: {'accuracy': float, 'log_likelihood': float}}
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Different Model Performance Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]
        log_likelihoods = [results[m]['log_likelihood'] for m in model_names]
        
        # 6.1 Accuracy Comparison
        bars = axes[0].bar(model_names, accuracies, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, 1.1)
        
        # 在柱子上显示数值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2%}', ha='center', va='bottom')
        
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 6.2 Log Likelihood Comparison
        bars = axes[1].bar(model_names, log_likelihoods, color='coral', alpha=0.7)
        axes[1].set_ylabel('Log Likelihood', fontsize=11)
        axes[1].set_title('Model Log Likelihood Comparison', fontsize=12, fontweight='bold')
        
        # 在柱子上显示数值
        for bar, ll in zip(bars, log_likelihoods):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{ll:.1f}', ha='center', va='bottom')
        
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"模型比较图已保存到 {filepath}")
        plt.close()
    
    def create_dashboard(self, alpha1: float, alpha2: float, weights: np.ndarray,
                        country_names: List[str], actual: List[int], predicted: List[int],
                        residuals: List[float]):
        """
        创建完整的仪表板，包含所有6类图表
        
        参数:
            alpha1: 阈值1
            alpha2: 阈值2
            weights: 权重矩阵
            country_names: 国家名称列表
            actual: 实际投票
            predicted: 预测投票
            residuals: 残差列表
        """
        print("正在生成Ordered Probit模型完整分析仪表板...")
        
        # 生成所有图表
        self.plot_goodness_of_fit(actual, predicted, country_names)
        self.plot_threshold_curves(alpha1, alpha2)
        self.plot_country_weights(weights, country_names)
        self.plot_residuals(residuals)
        
        print(f"所有图表已保存到 {self.output_dir} 目录")
