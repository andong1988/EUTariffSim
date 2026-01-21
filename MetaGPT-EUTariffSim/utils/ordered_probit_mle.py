"""
Ordered Probit最大似然估计器

用于估计各国在不同理论维度上的权重以及全局阈值参数。
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Tuple, List
import json
import os


class OrderedProbitMLE:
    """Ordered Probit模型的最大似然估计器"""
    
    def __init__(self, n_countries: int, n_theories: int = 3,
                 manual_thresholds: bool = False,
                 manual_alpha1: float = None,
                 manual_alpha2: float = None):
        """
        初始化MLE估计器
        
        参数:
            n_countries: 国家数量
            n_theories: 理论维度数量（默认为3）
            manual_thresholds: 是否手动控制阈值（默认False，自动优化）
            manual_alpha1: 手动配置的α₁阈值（仅在manual_thresholds=True时使用）
            manual_alpha2: 手动配置的α₂阈值（仅在manual_thresholds=True时使用）
        """
        self.n_countries = n_countries
        self.n_theories = n_theories
        self.alpha1 = None
        self.alpha2 = None
        self.weights = None  # shape: (n_countries, n_theories)
        self.convergence_info = None
        
        # 阈值手动控制参数
        self.manual_thresholds = manual_thresholds
        self.manual_alpha1 = manual_alpha1
        self.manual_alpha2 = manual_alpha2
        
        # 如果启用手动阈值，验证参数
        if self.manual_thresholds:
            self._validate_manual_thresholds()
        
        # 类别权重（样本权重）：按优先级设置权重
        # 优先级：反对 > 弃权 > 赞成
        self.class_weights = {
            0: 10.0,  # 反对票权重 = 10.0（最高优先级）
            1: 5.0,   # 弃权票权重 = 5.0（次优先级）
            2: 1.0    # 赞成票权重 = 1.0（最低优先级）
        }
    
    def _validate_manual_thresholds(self):
        """
        验证手动配置的阈值参数
        
        如果启用了手动阈值控制，验证参数是否合法
        """
        if self.manual_alpha1 is None or self.manual_alpha2 is None:
            raise ValueError("启用手动阈值时，必须提供manual_alpha1和manual_alpha2参数")
        
        if not isinstance(self.manual_alpha1, (int, float)) or not isinstance(self.manual_alpha2, (int, float)):
            raise ValueError("手动阈值必须是数值类型")
        
        if self.manual_alpha2 <= self.manual_alpha1:
            raise ValueError(f"手动阈值必须满足 α₂ > α₁，当前: α₁={self.manual_alpha1}, α₂={self.manual_alpha2}")
        
        if self.manual_alpha1 < -3.0 or self.manual_alpha1 > 3.0:
            raise ValueError(f"α₁必须在[-3, 3]范围内，当前值: {self.manual_alpha1}")
        
        if self.manual_alpha2 < -3.0 or self.manual_alpha2 > 3.0:
            raise ValueError(f"α₂必须在[-3, 3]范围内，当前值: {self.manual_alpha2}")
        
        print(f"手动阈值验证通过: α₁={self.manual_alpha1}, α₂={self.manual_alpha2}")
        
    def log_likelihood(self, params: np.ndarray, data: List[Dict]) -> float:
        """
        计算对数似然函数
        
        参数:
            params: 展平的参数向量 [alpha1, alpha2, w_11, w_12, ..., w_cn]
            data: 观察数据列表，每个元素包含:
                - country_idx: 国家索引 (0, 1, ..., n_countries-1)
                - theory_scores: 三个理论维度的得分 [t1, t2, t3]
                - actual_vote: 实际投票 (0=反对, 1=弃权, 2=赞成)
        
        返回:
            负对数似然（用于最小化）
        """
        # 正则化强度（使用更强的正则化防止权重过度极端化）
        # 考虑到数据量少（16条）且类别不平衡，需要更强的正则化
        lambda_reg = 0.8
        
        # 解析参数
        alpha1 = params[0]
        alpha2 = params[1]
        
        # 权重需要满足约束：alpha1 < alpha2
        if alpha2 <= alpha1:
            return 1e10  # 惩罚项
        
        # 解析国家权重矩阵
        weights = params[2:].reshape(self.n_countries, self.n_theories)
        
        # 对每个国家的权重应用softmax归一化，并强制最小权重
        min_weight = 0.1  # 每个维度至少占10%
        for i in range(self.n_countries):
            w = weights[i]
            exp_w = np.exp(w - np.max(w))  # 数值稳定性
            softmax_weights = exp_w / np.sum(exp_w)
            
            # 强制每个权重至少为min_weight，然后重新归一化
            softmax_weights = np.maximum(softmax_weights, min_weight)
            weights[i] = softmax_weights / np.sum(softmax_weights)
        
        # 添加L2正则化项（鼓励权重接近均匀分布，防止过度极端化）
        uniform_weights = 1.0 / self.n_theories
        regularization = lambda_reg * np.sum((weights - uniform_weights)**2)
        
        # 添加熵正则化，鼓励权重分布更加均匀
        # 熵越大，分布越均匀；熵越小，分布越极端
        entropy_penalty = -np.sum(weights * np.log(weights + 1e-10))
        regularization += 0.1 * entropy_penalty  # 熵正则化系数
        
        # 计算负对数似然
        neg_log_likelihood = 0.0
        eps = 1e-10  # 防止log(0)
        
        for observation in data:
            country_idx = observation['country_idx']
            theory_scores = observation['theory_scores']
            actual_vote = observation['actual_vote']
            
            # 获取该国家的权重
            w = weights[country_idx]
            
            # 计算线性组合
            eta = np.dot(theory_scores, w)
            
            # 计算三个类别的概率
            p0 = norm.cdf(alpha1 - eta)  # 反对
            p1 = norm.cdf(alpha2 - eta) - norm.cdf(alpha1 - eta)  # 弃权
            p2 = 1 - norm.cdf(alpha2 - eta)  # 赞成
            
            # 确保概率在合理范围内
            p0 = np.clip(p0, eps, 1 - eps)
            p1 = np.clip(p1, eps, 1 - eps)
            p2 = np.clip(p2, eps, 1 - eps)
            
            # 归一化（确保和为1）
            total = p0 + p1 + p2
            p0, p1, p2 = p0/total, p1/total, p2/total
            
            # 根据实际投票选择对应的概率
            if actual_vote == 0:
                prob = p0
            elif actual_vote == 1:
                prob = p1
            else:  # actual_vote == 2
                prob = p2
            
            # 应用类别权重（样本权重），让优化器更重视少数类
            sample_weight = self.class_weights[actual_vote]
            neg_log_likelihood -= sample_weight * np.log(prob)
        
        # 添加正则化项到最终结果
        neg_log_likelihood += regularization
        
        return neg_log_likelihood
    
    def fit_iterative(self, data: List[Dict]) -> Dict:
        """
        迭代优化方法：分阶段优化权重和阈值
        
        如果启用手动阈值控制（manual_thresholds=True）：
            - 直接使用用户配置的α₁和α₂
            - 跳过阈值优化，只优化权重
        
        如果未启用手动阈值（默认）：
            - 阶段1: 优先保证反对票100%准确
            - 阶段2: 优化弃权票和赞成票
        
        参数:
            data: 观察数据
        
        返回:
            包含结果的字典
        """
        print("="*80)
        if self.manual_thresholds:
            print("开始迭代优化流程（手动阈值模式）")
        else:
            print("开始迭代优化流程（自动阈值模式）")
        print("="*80)
        
        # 统计数据
        vote_counts = np.array([0, 0, 0])
        for obs in data:
            vote_counts[obs['actual_vote']] += 1
        total = np.sum(vote_counts)
        
        print(f"\n数据分布:")
        print(f"  反对 (0): {vote_counts[0]}")
        print(f"  弃权 (1): {vote_counts[1]}")
        print(f"  赞成 (2): {vote_counts[2]}")
        
        # =====================================================================
        # 手动阈值模式：直接使用用户配置的阈值
        # =====================================================================
        if self.manual_thresholds:
            print("\n" + "="*80)
            print("【手动阈值模式】使用用户配置的阈值")
            print("="*80)
            print(f"\n手动配置的阈值:")
            print(f"  α₁ = {self.manual_alpha1:.6f}")
            print(f"  α₂ = {self.manual_alpha2:.6f}")
            
            # 直接使用手动配置的阈值
            alpha1_fixed = self.manual_alpha1
            alpha2_fixed = self.manual_alpha2
            
            # 初始化权重参数
            weights_init = np.zeros(self.n_countries * self.n_theories)
            for i in range(self.n_countries):
                country_random = np.random.randn(self.n_theories) * 0.5
                for j in range(self.n_theories):
                    idx = i * self.n_theories + j
                    weights_init[idx] = country_random[j] + (i * 0.01)
            
            # 准备初始参数：使用手动配置的阈值
            initial_params = np.concatenate([[alpha1_fixed, alpha2_fixed], weights_init])
            
            # 设置边界：α₁和α₂都固定，权重可以调整
            bounds_manual = [
                (alpha1_fixed, alpha1_fixed),  # α₁固定
                (alpha2_fixed, alpha2_fixed),  # α₂固定
            ]
            bounds_manual += [(-5, 5)] * (self.n_countries * self.n_theories)
            
            # 设置类别权重（使用均衡权重）
            self.class_weights = {
                0: 10.0,   # 反对票权重
                1: 5.0,    # 弃权票权重
                2: 1.0     # 赞成票权重
            }
            
            print(f"\n类别权重设置:")
            print(f"  反对票权重: {self.class_weights[0]}")
            print(f"  弃权票权重: {self.class_weights[1]}")
            print(f"  赞成票权重: {self.class_weights[2]}")
            
            # 优化权重（阈值固定）
            print("\n开始优化权重（α₁和α₂固定）...")
            result_manual = self._optimize_with_params(initial_params, data, bounds=bounds_manual)
            
            # 存储结果
            self.alpha1 = alpha1_fixed
            self.alpha2 = alpha2_fixed
            self.weights = result_manual['weights']
            
            # 评估准确率
            acc_manual = self._evaluate_accuracy(data)
            print(f"\n优化完成！")
            print(f"最终阈值:")
            print(f"  α₁ = {self.alpha1:.6f} (手动配置)")
            print(f"  α₂ = {self.alpha2:.6f} (手动配置)")
            print(f"\n最终准确率:")
            print(f"  反对: {acc_manual[0]:.2%}")
            print(f"  弃权: {acc_manual[1]:.2%}")
            print(f"  赞成: {acc_manual[2]:.2%}")
            print(f"  整体: {np.mean(acc_manual):.2%}")
            
            # 存储收敛信息
            self.convergence_info = {
                'success': True,
                'message': 'Manual threshold optimization completed',
                'nit': 0,
                'fun': result_manual['fun'],
                'method': 'manual_thresholds',
                'manual_thresholds': {
                    'alpha1': alpha1_fixed,
                    'alpha2': alpha2_fixed
                },
                'accuracy': acc_manual.tolist()
            }
            
            print("\n" + "="*80)
            print("手动阈值优化流程完成")
            print("="*80)
            
            return self.convergence_info
        
        # =====================================================================
        # 自动阈值模式：原有的两阶段优化流程
        # =====================================================================
        print("\n使用自动阈值优化流程...")
        
        # 初始化参数（范围扩大到(-3, 3)以获得更好的概率区分度）
        alpha1_init = -1.5
        alpha2_init = 1.5
        
        weights_init = np.zeros(self.n_countries * self.n_theories)
        for i in range(self.n_countries):
            country_random = np.random.randn(self.n_theories) * 0.5
            for j in range(self.n_theories):
                idx = i * self.n_theories + j
                weights_init[idx] = country_random[j] + (i * 0.01)
        
        initial_params = np.concatenate([[alpha1_init, alpha2_init], weights_init])
        
        # =====================================================================
        # 阶段1: 优先保证反对票100%准确
        # =====================================================================
        print("\n" + "="*80)
        print("【阶段1】优先优化反对票准确率")
        print("="*80)
        
        # 使用适度的反对票权重（不要太极端）
        self.class_weights = {
            0: 20.0,   # 反对票权重较高，但不是极端值
            1: 5.0,    # 弃权票权重
            2: 1.0     # 赞成票权重
        }
        
        print("\n阶段1权重设置:")
        print(f"  反对票权重: {self.class_weights[0]}")
        print(f"  弃权票权重: {self.class_weights[1]}")
        print(f"  赞成票权重: {self.class_weights[2]}")
        
        # 阶段1优化
        print("\n开始阶段1优化...")
        result1 = self._optimize_with_params(initial_params, data)
        
        # 存储阶段1结果
        alpha1_phase1 = result1['alpha1']
        alpha2_phase1 = result1['alpha2']
        weights_phase1 = result1['weights']
        
        print(f"\n阶段1优化完成:")
        print(f"  α₁ = {alpha1_phase1:.6f}")
        print(f"  α₂ = {alpha2_phase1:.6f}")
        
        # 评估阶段1结果
        self.alpha1 = alpha1_phase1
        self.alpha2 = alpha2_phase1
        self.weights = weights_phase1
        
        acc1 = self._evaluate_accuracy(data)
        print(f"\n阶段1准确率:")
        print(f"  反对: {acc1[0]:.2%}")
        print(f"  弃权: {acc1[1]:.2%}")
        print(f"  赞成: {acc1[2]:.2%}")
        
        # 如果反对票不是100%，手动调整α₁
        # 或者即使反对票是100%，也要检查α₁是否合理
        if acc1[0] < 1.0 or alpha1_phase1 > 0.5:
            if acc1[0] < 1.0:
                print(f"\n⚠ 反对票准确率未达到100%，进行手动调整...")
            else:
                print(f"\n⚠ α₁值过高（{alpha1_phase1:.6f}），可能导致全部预测为反对，进行调整...")
            
            # 计算所有反对票样本的η值
            oppose_eta_values = []
            for obs in data:
                if obs['actual_vote'] == 0:
                    country_idx = obs['country_idx']
                    theory_scores = obs['theory_scores']
                    w = weights_phase1[country_idx]
                    eta = np.dot(theory_scores, w)
                    oppose_eta_values.append(eta)
            
            # 计算弃权和赞成票样本的η值，确保α₁不会太高
            non_oppose_eta_values = []
            for obs in data:
                if obs['actual_vote'] != 0:
                    country_idx = obs['country_idx']
                    theory_scores = obs['theory_scores']
                    w = weights_phase1[country_idx]
                    eta = np.dot(theory_scores, w)
                    non_oppose_eta_values.append(eta)
            
            oppose_eta_values = np.array(oppose_eta_values)
            non_oppose_eta_values = np.array(non_oppose_eta_values)
            
            max_oppose_eta = np.max(oppose_eta_values)
            min_non_oppose_eta = np.min(non_oppose_eta_values)
            
            print(f"  反对票样本的最大η值: {max_oppose_eta:.6f}")
            print(f"  非反对票样本的最小η值: {min_non_oppose_eta:.6f}")
            
            # 将α₁设置在反对票最大η值和非反对票最小η值之间
            # 理想情况下，α₁应该是一个中间值，比如(max_oppose + min_non_oppose) / 2
            # 但要确保反对票样本都在α₁左侧
            # 安全边界：α₁应该比最大反对票η值大0.05-0.1
            alpha1_adjusted = max_oppose_eta + 0.1
            
            # 同时确保α₁不会太大，不要超过非反对票的最小η值
            # 如果会超过，就取两者的中间值
            if alpha1_adjusted > min_non_oppose_eta - 0.1:
                alpha1_adjusted = (max_oppose_eta + min_non_oppose_eta) / 2
                print(f"  使用中间值避免α₁过高")
            
            # 确保α₁在合理范围内（不要太接近边界）
            alpha1_adjusted = np.clip(alpha1_adjusted, -0.8, 0.8)
            
            print(f"  调整后α₁: {alpha1_phase1:.6f} -> {alpha1_adjusted:.6f}")
            
            alpha1_phase1 = alpha1_adjusted
            
            # 重新评估调整后的准确率
            self.alpha1 = alpha1_phase1
            self.alpha2 = alpha2_phase1
            self.weights = weights_phase1
            acc1_adjusted = self._evaluate_accuracy(data)
            print(f"  调整后准确率: 反对={acc1_adjusted[0]:.2%}, 弃权={acc1_adjusted[1]:.2%}, 赞成={acc1_adjusted[2]:.2%}")
            acc1 = acc1_adjusted
        
        # =====================================================================
        # 阶段2: 优化弃权票和赞成票
        # =====================================================================
        print("\n" + "="*80)
        print("【阶段2】优化弃权票和赞成票准确率（保持反对准确率）")
        print("="*80)
        
        # 使用更均衡的权重
        # 反对票权重不能太高，否则会压倒其他类别
        self.class_weights = {
            0: 20.0,    # 反对票权重适中
            1: 10.0,    # 弃权票权重
            2: 1.0     # 赞成票权重最低
        }
        
        print("\n阶段2权重设置:")
        print(f"  反对票权重: {self.class_weights[0]}")
        print(f"  弃权票权重: {self.class_weights[1]}")
        print(f"  赞成票权重: {self.class_weights[2]}")
        
        # 评估阶段1的准确率（使用阶段1的权重和阈值）
        self.alpha1 = alpha1_phase1
        self.alpha2 = alpha2_phase1
        self.weights = weights_phase1
        acc_phase1 = self._evaluate_accuracy(data)
        print(f"\n阶段1最终准确率:")
        print(f"  反对: {acc_phase1[0]:.2%}")
        print(f"  弃权: {acc_phase1[1]:.2%}")
        print(f"  赞成: {acc_phase1[2]:.2%}")
        
        # =====================================================================
        # 阶段2-步骤1: 固定阶段1的权重，调整α₂使弃权票正确
        # =====================================================================
        print("\n" + "-"*80)
        print("【阶段2-步骤1】固定权重，调整α₂确保弃权票准确率")
        print("-"*80)
        
        # 使用阶段1的权重计算η值（避免权重变化影响α₂调整）
        abstain_eta_values = []
        for obs in data:
            if obs['actual_vote'] == 1:
                country_idx = obs['country_idx']
                theory_scores = obs['theory_scores']
                w = weights_phase1[country_idx]
                eta = np.dot(theory_scores, w)
                abstain_eta_values.append(eta)
        
        # 计算赞成票样本的η值
        approve_eta_values = []
        for obs in data:
            if obs['actual_vote'] == 2:
                country_idx = obs['country_idx']
                theory_scores = obs['theory_scores']
                w = weights_phase1[country_idx]
                eta = np.dot(theory_scores, w)
                approve_eta_values.append(eta)
        
        abstain_eta_values = np.array(abstain_eta_values)
        approve_eta_values = np.array(approve_eta_values)
        
        max_abstain_eta = np.max(abstain_eta_values)
        min_approve_eta = np.min(approve_eta_values)
        
        print(f"使用阶段1的权重统计:")
        print(f"  弃权票样本的最大η值: {max_abstain_eta:.6f}")
        print(f"  赞成票样本的最小η值: {min_approve_eta:.6f}")
        
        # 将α₂设置在弃权票最大η值和赞成票最小η值之间
        # 确保所有弃权票都在α₂左侧，赞成票都在α₂右侧
        # 安全边界：α₂应该比最大弃权票η值大0.05-0.1
        alpha2_adjusted = max_abstain_eta + 0.1
        
        # 同时确保α₂不会太小，不要比赞成票的最小η值还小
        if alpha2_adjusted < min_approve_eta - 0.1:
            alpha2_adjusted = (max_abstain_eta + min_approve_eta) / 2
            print(f"  使用中间值避免α₂过小")
        
        # 确保α₂在合理范围内，且必须大于α₁
        alpha2_adjusted = np.clip(alpha2_adjusted, alpha1_phase1 + 0.2, 1.0)
        
        print(f"调整后α₂: {alpha2_phase1:.6f} -> {alpha2_adjusted:.6f}")
        
        # 评估调整后的准确率（使用阶段1的权重）
        self.alpha1 = alpha1_phase1
        self.alpha2 = alpha2_adjusted
        self.weights = weights_phase1
        acc_step1 = self._evaluate_accuracy(data)
        print(f"步骤1调整后准确率: 反对={acc_step1[0]:.2%}, 弃权={acc_step1[1]:.2%}, 赞成={acc_step1[2]:.2%}")
        
        # =====================================================================
        # 阶段2-步骤2: 固定α₁和调整后的α₂，只优化权重
        # =====================================================================
        print("\n" + "-"*80)
        print("【阶段2-步骤2】固定阈值，优化权重")
        print("-"*80)
        
        # 准备初始参数：使用阶段1的结果，但α₂使用调整后的值
        phase2_init = np.concatenate([[alpha1_phase1, alpha2_adjusted], weights_phase1.flatten()])
        
        # 设置边界：α₁和α₂都固定，权重可以调整
        bounds_phase2 = [
            (alpha1_phase1, alpha1_phase1),  # α₁固定
            (alpha2_adjusted, alpha2_adjusted),  # α₂固定
        ]
        bounds_phase2 += [(-5, 5)] * (self.n_countries * self.n_theories)
        
        print("\n开始阶段2-步骤2优化（α₁和α₂固定，只优化权重）...")
        result2 = self._optimize_with_params(phase2_init, data, bounds=bounds_phase2)
        
        # 存储阶段2最终结果
        weights_phase2 = result2['weights']
        
        # 评估阶段2最终结果
        self.alpha1 = alpha1_phase1
        self.alpha2 = alpha2_adjusted
        self.weights = weights_phase2
        
        acc2 = self._evaluate_accuracy(data)
        print(f"\n阶段2最终准确率:")
        print(f"  反对: {acc2[0]:.2%}")
        print(f"  弃权: {acc2[1]:.2%}")
        print(f"  赞成: {acc2[2]:.2%}")
        acc_final = acc2
        
        # 存储最终结果
        self.alpha1 = alpha1_phase1
        self.alpha2 = alpha2_adjusted
        self.weights = weights_phase2
        
        print(f"\n最终阈值:")
        print(f"  α₁ = {self.alpha1:.6f} (阶段1)")
        print(f"  α₂ = {self.alpha2:.6f} (阶段2调整)")
        
        print(f"\n最终准确率:")
        print(f"  反对: {acc_final[0]:.2%}")
        print(f"  弃权: {acc_final[1]:.2%}")
        print(f"  赞成: {acc_final[2]:.2%}")
        print(f"  整体: {np.mean(acc_final):.2%}")
        
        # 存储收敛信息
        self.convergence_info = {
            'success': True,
            'message': 'Iterative optimization completed',
            'nit': 0,  # 迭代次数不适用
            'fun': 0.0,
            'method': 'iterative',
            'phase_results': {
                'phase1': {'alpha1': alpha1_phase1, 'alpha2': alpha2_phase1, 'accuracy': acc_phase1.tolist()},
                'phase2_step1': {'alpha1': alpha1_phase1, 'alpha2': alpha2_adjusted, 'accuracy': acc_step1.tolist()},
                'phase2_step2': {'alpha1': self.alpha1, 'alpha2': self.alpha2, 'accuracy': acc_final.tolist()}
            }
        }
        
        print("\n" + "="*80)
        print("迭代优化流程完成")
        print("="*80)
        
        return self.convergence_info
    
    def _optimize_with_params(self, initial_params: np.ndarray, data: List[Dict], 
                             bounds: List = None) -> Dict:
        """
        辅助方法：使用给定初始参数进行优化
        
        参数:
            initial_params: 初始参数
            data: 观察数据
            bounds: 参数边界（可选）
        
        返回:
            包含优化结果的字典
        """
        # 定义目标函数
        def objective(params):
            return self.log_likelihood(params, data)
        
        # 设置默认边界（范围扩大到(-3, 3)以获得更好的概率区分度）
        if bounds is None:
            bounds = [(-3.0, 3.0), (-3.0, 3.0)]
            bounds += [(-5, 5)] * (self.n_countries * self.n_theories)
        
        # 运行优化
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 5000,
                'ftol': 1e-12,
                'gtol': 1e-9,
                'disp': False,
                'maxfun': 15000
            }
        )
        
        # 归一化权重
        raw_weights = result.x[2:].reshape(self.n_countries, self.n_theories)
        normalized_weights = np.zeros_like(raw_weights)
        min_weight = 0.2
        
        for i in range(self.n_countries):
            w = raw_weights[i]
            exp_w = np.exp(w - np.max(w))
            softmax_weights = exp_w / np.sum(exp_w)
            softmax_weights = np.maximum(softmax_weights, min_weight)
            normalized_weights[i] = softmax_weights / np.sum(softmax_weights)
        
        return {
            'alpha1': result.x[0],
            'alpha2': result.x[1],
            'weights': normalized_weights,
            'fun': result.fun,
            'success': result.success
        }
    
    def _evaluate_accuracy(self, data: List[Dict]) -> np.ndarray:
        """
        辅助方法：评估各类别准确率
        
        参数:
            data: 观察数据
        
        返回:
            各类别准确率数组 [反对, 弃权, 赞成]
        """
        correct = [0, 0, 0]
        total = [0, 0, 0]
        
        for obs in data:
            country_idx = obs['country_idx']
            theory_scores = obs['theory_scores']
            actual_vote = obs['actual_vote']
            
            proba = self.predict_proba(theory_scores, country_idx)
            predicted_vote = np.argmax(proba)
            
            total[actual_vote] += 1
            if predicted_vote == actual_vote:
                correct[actual_vote] += 1
        
        accuracy = np.array([
            correct[0] / total[0] if total[0] > 0 else 0.0,
            correct[1] / total[1] if total[1] > 0 else 0.0,
            correct[2] / total[2] if total[2] > 0 else 0.0
        ])
        
        return accuracy
    
    def fit(self, data: List[Dict], initial_params: np.ndarray = None) -> Dict:
        """
        拟合Ordered Probit模型（原始方法，向后兼容）
        
        参数:
            data: 观察数据
            initial_params: 初始参数（可选）
        
        返回:
            包含结果的字典
        """
        # 初始化参数
        if initial_params is None:
            # 数据驱动的阈值初始化
            # 统计实际投票分布
            vote_counts = np.array([0, 0, 0])
            for obs in data:
                vote_counts[obs['actual_vote']] += 1
            
            total = np.sum(vote_counts)
            # 根据η的实际范围[-3, 3]来初始化阈值（范围扩大以获得更好的概率区分度）
            # 不再使用正态分布分位数，而是使用合理的中间值
            alpha1_init = -1.5  # 反对/弃权边界在-1.5
            alpha2_init = 1.5   # 弃权/赞成边界在1.5
            
            print(f"固定阈值初始化（基于η的范围[-3, 3]）:")
            print(f"  投票分布: 反对={vote_counts[0]}/{total}, " +
                  f"弃权={vote_counts[1]}/{total}, " +
                  f"赞成={vote_counts[2]}/{total}")
            print(f"  初始阈值: α₁={alpha1_init:.4f}, α₂={alpha2_init:.4f}")
            
            # 权重初值：为每个国家使用不同的随机初始值，打破对称性
            # 移除固定随机种子，允许真正的随机探索
            weights_init = np.zeros(self.n_countries * self.n_theories)
            
            for i in range(self.n_countries):
                # 为每个国家生成不同的随机初始值
                # 范围在[-3, 3]之间，经过softmax后会变成合理的概率分布
                country_random = np.random.randn(self.n_theories) * 0.5  # 标准差0.5
                
                # 添加基于国家索引的小扰动，确保每个国家不同
                for j in range(self.n_theories):
                    idx = i * self.n_theories + j
                    weights_init[idx] = country_random[j] + (i * 0.01)
            
            print(f"权重初始化: 使用随机初始值，每个国家不同")
            
            initial_params = np.concatenate([[alpha1_init, alpha2_init], weights_init])
        
        # 定义目标函数（返回字典形式以兼容优化器）
        def objective(params):
            return self.log_likelihood(params, data)
        
        # 设置参数边界（范围扩大到(-3, 3)以获得更好的概率区分度）
        # α₁和α₂必须在(-3, 3)之间，与η的范围[-3, 3]匹配
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]  # alpha1, alpha2约束在(-3, 3)
        # 扩大原始权重范围到[-5, 5]，允许更大的搜索空间
        bounds += [(-5, 5)] * (self.n_countries * self.n_theories)
        
        # 使用L-BFGS-B优化（支持边界约束）
        # 添加回调函数来监控优化过程
        iteration_info = {'iter': 0, 'best_ll': float('inf')}
        
        def callback(xk):
            iteration_info['iter'] += 1
            current_ll = objective(xk)
            
            # 记录最佳值
            if current_ll < iteration_info['best_ll']:
                iteration_info['best_ll'] = current_ll
            
            # 每50次迭代打印一次
            if iteration_info['iter'] % 50 == 0:
                print(f"  迭代 {iteration_info['iter']}: 负对数似然 = {current_ll:.6f} (最佳: {iteration_info['best_ll']:.6f})")
                # 打印当前的阈值
                alpha1, alpha2 = xk[0], xk[1]
                print(f"    当前阈值: α₁={alpha1:.6f}, α₂={alpha2:.6f}")
        
        print("开始优化过程...")
        print(f"初始负对数似然: {objective(initial_params):.6f}")
        
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            callback=callback,
            options={
                'maxiter': 5000,  # 增加最大迭代次数
                'ftol': 1e-12,   # 提高收敛精度
                'gtol': 1e-9,    # 提高梯度容差
                'disp': True,
                'maxfun': 15000  # 增加最大函数评估次数
            }
        )
        
        print(f"优化完成!")
        print(f"最终负对数似然: {result.fun:.6f}")
        print(f"总迭代次数: {result.nit}")
        
        # 存储结果
        self.alpha1 = result.x[0]
        self.alpha2 = result.x[1]
        
        # 归一化权重，并应用最小权重约束
        raw_weights = result.x[2:].reshape(self.n_countries, self.n_theories)
        self.weights = np.zeros_like(raw_weights)
        min_weight = 0.2  # 每个维度至少占20%（提高到0.2以防止极端化）
        
        for i in range(self.n_countries):
            w = raw_weights[i]
            exp_w = np.exp(w - np.max(w))
            softmax_weights = exp_w / np.sum(exp_w)
            
            # 应用最小权重约束，然后重新归一化
            softmax_weights = np.maximum(softmax_weights, min_weight)
            self.weights[i] = softmax_weights / np.sum(softmax_weights)
        
        self.convergence_info = {
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'nit': result.nit,
            'fun': result.fun
        }
        
        return self.convergence_info
    
    def predict_proba(self, theory_scores: np.ndarray, country_idx: int) -> np.ndarray:
        """
        预测投票概率
        
        参数:
            theory_scores: 理论得分 [t1, t2, t3]
            country_idx: 国家索引
        
        返回:
            概率数组 [p_oppose, p_abstain, p_approve]
        """
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        w = self.weights[country_idx]
        eta = np.dot(theory_scores, w)
        
        p0 = norm.cdf(self.alpha1 - eta)
        p1 = norm.cdf(self.alpha2 - eta) - norm.cdf(self.alpha1 - eta)
        p2 = 1 - norm.cdf(self.alpha2 - eta)
        
        # 归一化
        total = p0 + p1 + p2
        return np.array([p0/total, p1/total, p2/total])
    
    def get_residuals(self, data: List[Dict]) -> List[float]:
        """
        计算残差（观察值与预测概率的差）
        
        参数:
            data: 观察数据
        
        返回:
            残差列表
        """
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        residuals = []
        for observation in data:
            country_idx = observation['country_idx']
            theory_scores = observation['theory_scores']
            actual_vote = observation['actual_vote']
            
            pred_proba = self.predict_proba(theory_scores, country_idx)
            residuals.append(actual_vote - np.argmax(pred_proba))
        
        return residuals
    
    def save_parameters(self, filepath: str, country_names: List[str]):
        """
        保存估计的参数到JSON文件
        
        参数:
            filepath: 保存路径
            country_names: 国家名称列表
        """
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        
        # 准备convergence_info，确保所有值都是JSON可序列化的
        convergence_info_serializable = {}
        if self.convergence_info:
            for key, value in self.convergence_info.items():
                if isinstance(value, np.ndarray):
                    # 将numpy数组转换为列表
                    convergence_info_serializable[key] = value.tolist()
                elif isinstance(value, dict):
                    # 处理嵌套字典（如phase_results）
                    nested_dict = {}
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, np.ndarray):
                            nested_dict[nested_key] = nested_value.tolist()
                        else:
                            nested_dict[nested_key] = nested_value
                    convergence_info_serializable[key] = nested_dict
                else:
                    convergence_info_serializable[key] = value
        
        params_dict = {
            'thresholds': {
                'alpha1': float(self.alpha1),
                'alpha2': float(self.alpha2)
            },
            'country_weights': {},
            'theory_names': [
                'x_market',
                'x_political',
                'x_institutional'
            ],
            'convergence_info': convergence_info_serializable
        }
        
        # 转换权重为字典格式
        for i, country in enumerate(country_names):
            params_dict['country_weights'][country] = {
                'x_market': float(self.weights[i, 0]),
                'x_political': float(self.weights[i, 1]),
                'x_institutional': float(self.weights[i, 2])
            }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=2, ensure_ascii=False)
        
        print(f"参数已保存到 {filepath}")
    
    @staticmethod
    def load_parameters(filepath: str) -> Dict:
        """
        从JSON文件加载参数
        
        参数:
            filepath: 文件路径
        
        返回:
            参数字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
