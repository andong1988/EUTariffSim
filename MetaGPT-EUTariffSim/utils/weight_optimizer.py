"""权重优化工具"""

import json
import logging
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class WeightOptimizer:
    """权重优化工具类"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def monte_carlo_weight_optimization(self, country_id: str, initial_theory_scores: Dict, 
                                         final_theory_scores: Dict, vote_targets: List[float], 
                                         actual_votes: List[str]) -> Optional[Dict[str, float]]:
        """使用蒙特卡洛算法进行权重优化 - 使用同一套权重匹配两轮不同的理论得分"""
        self.logger.info(f"{country_id} 开始蒙特卡洛权重优化")
        
        # 采样参数
        n_samples = 100000  # 采样次数，增加到10万以提高找到完美解的概率
        best_accuracy = 0.0
        best_weights = None
        perfect_weights_list = []  # 存储所有达到100%准确率的权重
        
        start_time = time.time()
        
        for iteration in range(n_samples):
            # 生成3个随机权重（新的三个维度）
            raw_weights = np.random.rand(3)  # [0, 1]
            
            # 归一化权重
            weights_normalized = raw_weights / np.sum(raw_weights)
            
            # 转换为字典格式
            weights = {
                'x_market': weights_normalized[0],
                'x_political': weights_normalized[1],
                'x_institutional': weights_normalized[2]
            }
            
            # 计算准确率 - 传递两轮理论得分
            accuracy = self.calculate_accuracy_with_weights(
                initial_theory_scores, final_theory_scores, weights, actual_votes
            )
            
            # 更新最优解
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()
                
                # 如果达到100%，记录下来
                if accuracy == 1.0:
                    perfect_weights_list.append({
                        'weights': weights.copy(),
                        'iteration': iteration
                    })
            
            # 每10000次迭代报告进度
            if (iteration + 1) % 10000 == 0:
                self.logger.info(f"{country_id} 蒙特卡洛优化进度: {iteration + 1}/{n_samples}, 当前最佳准确率: {best_accuracy:.3f}, 完美解数: {len(perfect_weights_list)}")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"{country_id} 蒙特卡洛优化完成，耗时: {elapsed_time:.2f}秒")
        self.logger.info(f"  采样次数: {n_samples}")
        self.logger.info(f"  找到的100%准确率解: {len(perfect_weights_list)}个")
        self.logger.info(f"  最优准确率: {best_accuracy:.3f}")
        self.logger.info(f"  最优权重: {best_weights}")
        
        # 如果找到了100%准确率的解，从中选择最优的（最均匀分布的）
        if perfect_weights_list:
            self.logger.info(f"  从{len(perfect_weights_list)}个完美解中选择最优解")
            
            # 计算每个完美解的权重熵（熵越大，分布越均匀）
            def calculate_entropy(w):
                total = sum(w.values())
                normalized = {k: v/total for k, v in w.items()}
                return -sum(v * np.log(v + 1e-10) for v in normalized.values())
            
            # 选择熵最大的权重（最均匀的分布）
            best_perfect = max(perfect_weights_list, key=lambda x: calculate_entropy(x['weights']))
            best_weights = best_perfect['weights']
            best_accuracy = 1.0
            
            entropy = calculate_entropy(best_weights)
            self.logger.info(f"  选中的完美解熵值: {entropy:.4f} (越均匀越好)")
        else:
            # 如果没有找到100%准确率的解，尝试增加采样次数进行第二轮搜索
            self.logger.info(f"  未找到完美解，尝试第二轮局部搜索")
            
            # 在最优解附近进行精细搜索 - 传递两轮理论得分
            if best_weights:
                best_weights = self.local_search_around_best_weights(
                    country_id, initial_theory_scores, final_theory_scores, 
                    actual_votes, best_weights, n_local_search=10000
                )
                
                # 重新计算准确率 - 传递两轮理论得分
                final_accuracy = self.calculate_accuracy_with_weights(
                    initial_theory_scores, final_theory_scores, best_weights, actual_votes
                )
                best_accuracy = final_accuracy
        
        # 如果仍然达不到100%，返回None，让其他优化方法尝试
        if best_accuracy < 1.0:
            self.logger.info(f"  蒙特卡洛优化未达到100%准确率，返回None让其他方法尝试")
            return None
        
        return best_weights
    
    def local_search_around_best_weights(self, country_id: str, initial_theory_scores: Dict, 
                                         final_theory_scores: Dict, actual_votes: List[str], 
                                         best_weights: Dict[str, float], n_local_search: int = 5000) -> Dict[str, float]:
        """在最优权重附近进行局部搜索 - 使用两轮理论得分"""
        self.logger.info(f"{country_id} 开始局部搜索，采样{n_local_search}次")
        
        best_accuracy = 0.0
        best_local_weights = best_weights.copy()
        
        for _ in range(n_local_search):
            # 在最优权重附近添加小扰动
            test_weights = {
                'x_market': max(0.0, min(1.0, best_weights['x_market'] + np.random.uniform(-0.2, 0.2))),
                'x_political': max(0.0, min(1.0, best_weights['x_political'] + np.random.uniform(-0.2, 0.2))),
                'x_institutional': max(0.0, min(1.0, best_weights['x_institutional'] + np.random.uniform(-0.2, 0.2)))
            }
            
            # 归一化
            test_weights = self.normalize_and_constrain_weights(test_weights)
            
            # 计算准确率 - 传递两轮理论得分
            accuracy = self.calculate_accuracy_with_weights(
                initial_theory_scores, final_theory_scores, test_weights, actual_votes
            )
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_local_weights = test_weights.copy()
                
                # 如果达到100%，立即返回
                if accuracy == 1.0:
                    break
        
        self.logger.info(f"  局部搜索完成，最优准确率: {best_accuracy:.3f}")
        return best_local_weights
    
    def bayesian_weight_optimization(self, country_id: str, initial_theory_scores: Dict,
                                     final_theory_scores: Dict, vote_targets: List[float], 
                                     actual_votes: List[str]) -> Optional[Dict[str, float]]:
        """使用贝叶斯优化进行权重调整 - 使用两轮理论得分"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
            
            self.logger.info(f"{country_id} 开始贝叶斯权重优化")
            
            # 定义搜索空间：每个权重在0到1之间（新的三个维度）
            space = [
                Real(0.0, 1.0, name='x_market'),
                Real(0.0, 1.0, name='x_political'),
                Real(0.0, 1.0, name='x_institutional')
            ]
            
            @use_named_args(space)
            def objective_function(**params):
                """目标函数：使用两轮理论得分最大化投票准确率（新的三个维度）"""
                # 提取权重
                weights = np.array([
                    params['x_market'],
                    params['x_political'],
                    params['x_institutional']
                ])
                
                # 归一化权重
                weights_normalized = weights / np.sum(weights)
                
                # 计算预测值 - 使用两轮理论得分（新的三个维度）
                predictions = []
                theory_scores_list = [initial_theory_scores, final_theory_scores]
                for target in vote_targets:
                    # 确定使用哪一轮的理论得分
                    i = vote_targets.index(target)
                    theory_scores = theory_scores_list[i]
                    prediction = (
                        theory_scores["x_market"] * weights_normalized[0] +
                        theory_scores["x_political"] * weights_normalized[1] +
                        theory_scores["x_institutional"] * weights_normalized[2]
                    )
                    predictions.append(prediction)
                
                # 计算均方误差
                mse = np.mean((np.array(predictions) - np.array(vote_targets)) ** 2)
                
                # 添加正则化项：惩罚权重极端分布
                # 计算权重熵
                weights_entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-10))
                max_entropy = np.log(len(weights_normalized))
                regularization = 0.1 * (1 - weights_entropy / max_entropy)  # 鼓励更均匀的分布
                
                # 添加平滑性正则化：惩罚权重方差过大
                smoothness = 0.05 * np.var(weights_normalized)
                
                # 总损失函数：MSE + 正则化
                total_loss = mse + regularization + smoothness
                
                # 贝叶斯优化最小化，所以返回总损失
                return total_loss
            
            # 执行贝叶斯优化
            result = gp_minimize(
                objective_function,
                space,
                n_calls=100,  # 优化迭代次数
                n_initial_points=20,  # 随机初始采样点数
                random_state=42,
                n_jobs=-1,  # 并行计算
                verbose=False
            )
            
            # 提取最优权重
            best_weights_array = result.x
            best_loss = result.fun
            
            # 转换为字典格式（新的三个维度）
            best_weights = {
                'x_market': best_weights_array[0],
                'x_political': best_weights_array[1],
                'x_institutional': best_weights_array[2]
            }
            
            # 计算最优准确率 - 传递两轮理论得分
            accuracy = self.calculate_accuracy_with_weights(
                initial_theory_scores, final_theory_scores, best_weights, actual_votes
            )
            
            self.logger.info(f"{country_id} 贝叶斯优化完成:")
            self.logger.info(f"  最优损失: {best_loss:.6f}")
            self.logger.info(f"  最优权重: {best_weights}")
            self.logger.info(f"  最优准确率: {accuracy:.3f}")
            
            return best_weights
            
        except ImportError:
            self.logger.warning("scikit-optimize不可用，贝叶斯优化不可用")
            return None
        except Exception as e:
            self.logger.error(f"贝叶斯优化失败: {e}")
            return None
    
    def calculate_accuracy_with_weights(self, initial_theory_scores: Dict, final_theory_scores: Dict,
                                        weights: Dict[str, float], actual_votes: List[str]) -> float:
        """计算给定权重的预测准确率 - 使用同一套权重同时匹配两轮不同的理论得分
        
        Args:
            initial_theory_scores: 初始轮的理论得分
            final_theory_scores: 最终轮的理论得分
            weights: 权重字典
            actual_votes: 实际投票结果 [第一次, 第二次]
            
        Returns:
            两轮预测的总准确率
        """
        # 理论名称映射：理论得分键 -> 权重键（新的三个维度）
        theory_to_weight_mapping = {
            "x_market": "x_market",
            "x_political": "x_political",
            "x_institutional": "x_institutional"
        }
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights_normalized = {k: v / total_weight for k, v in weights.items()}
        else:
            weights_normalized = weights
        
        correct_predictions = 0
        total_predictions = len(actual_votes)
        
        # 理论得分列表 - 对应两轮投票
        theory_scores_list = [initial_theory_scores, final_theory_scores]
        
        # 为每次投票生成预测
        for i, actual_vote in enumerate(actual_votes):
            theory_scores = theory_scores_list[i]
            
            # 计算决策得分 - 使用键名映射（新的三个维度）
            decision_score = 0.0
            for theory_key, score in theory_scores.items():
                weight_key = theory_to_weight_mapping.get(theory_key)
                if weight_key and weight_key in weights_normalized:
                    decision_score += score * weights_normalized[weight_key]
            
            # 转换为投票立场（新的范围：-1到1）
            if decision_score >= 0.5:  # >= 0.5为赞同
                predicted_stance = "support"
            elif decision_score >= 0.0:  # 0.0-0.499999为弃权
                predicted_stance = "abstain"
            else:  # < 0.0为反对
                predicted_stance = "against"
            
            # 检查预测是否正确
            if predicted_stance == actual_vote:
                correct_predictions += 1
        
        # 计算准确率
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return accuracy
    
    def normalize_and_constrain_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """归一化并约束权重"""
        # 确保权重非负（新的范围：0到1）
        for theory in weights:
            weights[theory] = max(0.0, weights[theory])
        
        # 归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            for theory in weights:
                weights[theory] /= total_weight
        
        # 约束在合理范围内（新的范围：0到1）
        for theory in weights:
            weights[theory] = max(0.0, min(1.0, weights[theory]))
        
        # 重新归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            for theory in weights:
                weights[theory] /= total_weight
        
        return weights
    
    def solve_weight_optimization(self, initial_theory_scores: Dict, final_theory_scores: Dict,
                                 vote_targets: List[float], actual_votes: List[str]) -> Optional[Dict[str, float]]:
        """使用梯度下降方法求解权重（作为贝叶斯优化的后备方案）"""
        try:
            import numpy as np
            from scipy.optimize import minimize
            
            theories = ["x_market", "x_political", "x_institutional"]
            
            # 理论得分列表 - 对应两轮投票
            theory_scores_list = [initial_theory_scores, final_theory_scores]
            
            # 构建目标函数：最小化预测误差
            def objective_function(weights):
                # 确保权重非负
                weights = np.maximum(weights, 0.01)
                # 归一化权重
                weights = weights / np.sum(weights)
                
                # 计算预测值 - 使用两轮理论得分
                predictions = []
                for i, target in enumerate(vote_targets):
                    theory_scores = theory_scores_list[i]
                    prediction = (
                        theory_scores["x_market"] * weights[0] +
                        theory_scores["x_political"] * weights[1] +
                        theory_scores["x_institutional"] * weights[2]
                    )
                    predictions.append(prediction)
                
                # 计算误差
                error = np.mean((np.array(predictions) - np.array(vote_targets)) ** 2)
                return error
            
            # 初始权重（均匀分布，新的三个维度）
            initial_weights = np.array([0.33, 0.33, 0.33])
            
            # 权重约束：每个权重在0到1之间（新的三个维度）
            bounds = [(0.0, 1.0) for _ in range(3)]
            
            # 优化
            result = minimize(objective_function, initial_weights, method='L-BFGS-B', bounds=bounds)
            
            if result.success and result.fun < 0.1:  # 误差阈值
                optimized_weights = result.x
                # 归一化
                optimized_weights = optimized_weights / np.sum(optimized_weights)
                
                return {
                    theories[0]: optimized_weights[0],
                    theories[1]: optimized_weights[1], 
                    theories[2]: optimized_weights[2]
                }
            
            return None
            
        except ImportError:
            self.logger.warning("scipy不可用，使用启发式优化方法")
            return None
        except Exception as e:
            self.logger.warning(f"数值优化失败: {e}")
            return None
