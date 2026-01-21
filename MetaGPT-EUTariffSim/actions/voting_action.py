#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
投票动作 - 处理最终投票决策
"""

from typing import Dict, Any, List, Optional
from metagpt.actions import Action
import logging
import random
import numpy as np
from scipy import stats
import yaml
import os

logger = logging.getLogger(__name__)


class VotingAction(Action):
    """投票动作：处理最终投票决策"""
    
    def __init__(self, **kwargs):
        """
        初始化投票动作
        
        Args:
            **kwargs: Action参数
        """
        super().__init__(**kwargs)
        
        # 投票选项
        self.vote_options = {
            'yes': '支持',
            'no': '反对',
            'abstain': '弃权'
        }
        
        # 默认配置值
        self.qualified_majority_threshold = 0.65
        self.noise_level = 0.05
        
        # 加载Ordered Probit配置
        self._load_ordered_probit_config()
    
    def _load_ordered_probit_config(self):
        """加载Ordered Probit模型配置"""
        try:
            # 尝试从配置文件加载
            config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if 'ordered_probit' in config and 'thresholds' in config['ordered_probit']:
                        self.alpha1 = config['ordered_probit']['thresholds']['alpha1']
                        self.alpha2 = config['ordered_probit']['thresholds']['alpha2']
                        logger.info(f"已加载Ordered Probit配置: α1={self.alpha1}, α2={self.alpha2}")
                        return
        except Exception as e:
            logger.warning(f"加载Ordered Probit配置失败: {e}，使用默认值")
        
        # 使用默认值
        self.alpha1 = 0.0
        self.alpha2 = 0.5
        logger.info(f"使用Ordered Probit默认配置: α1={self.alpha1}, α2={self.alpha2}")
    
    async def run(self,
                  country_features: Dict[str, Any],
                  current_position: float,
                  decision_history: List[Dict[str, Any]],
                  theory_weights: Dict[str, float],
                  context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行投票决策
        
        Args:
            country_features: 国家特征
            current_position: 当前立场（0-1）
            decision_history: 决策历史
            theory_weights: 理论权重
            context: 投票上下文
            
        Returns:
            投票结果
        """
        logger.info(f"执行投票决策，当前立场: {current_position:.3f}")
        
        # 1. 分析投票情境
        voting_context = self._analyze_voting_context(
            current_position, decision_history, context
        )
        
        # 2. 计算投票决策
        vote_decision, confidence = self._calculate_vote_decision(
            current_position, voting_context, theory_weights
        )
        
        # 3. 生成投票理由
        reasoning = self._generate_reasoning(
            vote_decision, confidence, current_position,
            voting_context, theory_weights, decision_history
        )
        
        # 4. 生成投票消息
        vote_message = self._generate_vote_message(vote_decision, confidence)
        
        result = {
            'vote': vote_decision,
            'vote_text': self.vote_options.get(vote_decision, vote_decision),
            'confidence': confidence,
            'current_position': current_position,
            'voting_context': voting_context,
            'reasoning': reasoning,
            'vote_message': vote_message,
            'theory_weights_at_vote': theory_weights.copy()
        }
        
        logger.debug(f"投票结果: {result}")
        return result
    
    def _analyze_voting_context(self,
                               current_position: float,
                               decision_history: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析投票情境
        
        Args:
            current_position: 当前立场
            decision_history: 决策历史
            context: 投票上下文
            
        Returns:
            投票情境分析
        """
        voting_context = {
            'current_position': current_position,
            'decision_history_count': len(decision_history),
            'position_stability': 0.7,  # 默认稳定性
            'external_pressure': context.get('external_pressure', 0.3),
            'coalition_alignment': context.get('coalition_alignment', 0.5),
            'critical_momentum': False
        }
        
        # 分析立场稳定性
        if len(decision_history) >= 2:
            # 计算最近几次决策的立场变化
            recent_positions = []
            for record in decision_history[-3:]:  # 最近3次决策
                if 'position' in record:
                    recent_positions.append(record['position'])
            
            if len(recent_positions) >= 2:
                # 计算标准差作为稳定性指标
                mean_position = sum(recent_positions) / len(recent_positions)
                variance = sum((p - mean_position) ** 2 for p in recent_positions) / len(recent_positions)
                stability = 1.0 - min(1.0, variance * 10)  # 方差越小，稳定性越高
                voting_context['position_stability'] = stability
        
        # 分析关键时刻（是否接近投票阈值）
        if 'coalition_support' in context:
            coalition_support = context['coalition_support']
            # 如果联盟支持率接近阈值，则是关键时刻
            voting_context['critical_momentum'] = abs(coalition_support - self.qualified_majority_threshold) < 0.05
        
        # 分析历史决策模式
        if decision_history:
            # 计算支持/反对的历史倾向
            support_count = sum(1 for record in decision_history 
                              if record.get('position', 0.5) > 0.6)
            oppose_count = sum(1 for record in decision_history 
                             if record.get('position', 0.5) < 0.4)
            
            total_decisions = len(decision_history)
            if total_decisions > 0:
                voting_context['historical_support_rate'] = support_count / total_decisions
                voting_context['historical_oppose_rate'] = oppose_count / total_decisions
        
        return voting_context
    
    def _calculate_vote_decision(self,
                                current_position: float,
                                voting_context: Dict[str, Any],
                                theory_weights: Dict[str, float]) -> tuple:
        """
        使用Ordered Probit模型计算投票决策
        
        Ordered Probit模型设定：
        Y* = Xβ + ε, ε ~ N(0,1)
        Y = 1 (反对) if Y* ≤ α1
        Y = 2 (弃权) if α1 < Y* ≤ α2
        Y = 3 (赞成) if Y* > α2
        
        概率计算：
        P(Y=1) = Φ(α1 - Xβ)
        P(Y=2) = Φ(α2 - Xβ) - Φ(α1 - Xβ)
        P(Y=3) = 1 - Φ(α2 - Xβ)
        
        其中 Φ 是标准正态分布的累积分布函数
        
        Args:
            current_position: 当前立场（作为Xβ，范围0-1）
            voting_context: 投票情境
            theory_weights: 理论权重
            
        Returns:
            (投票选项, 置信度)
        """
        # 将当前立场映射到Ordered Probit的潜变量空间
        # current_position范围是0-1，我们将其标准化到适合的标准正态分布范围
        # 这里我们直接使用current_position作为Xβ，因为它已经是0-1的标准化分数
        
        X_beta = current_position
        
        # 计算各类别的概率
        # P(反对) = P(Y=1) = Φ(α1 - Xβ)
        prob_oppose = stats.norm.cdf(self.alpha1 - X_beta)
        
        # P(弃权) = P(Y=2) = Φ(α2 - Xβ) - Φ(α1 - Xβ)
        prob_abstain = stats.norm.cdf(self.alpha2 - X_beta) - stats.norm.cdf(self.alpha1 - X_beta)
        
        # P(赞成) = P(Y=3) = 1 - Φ(α2 - Xβ)
        prob_approve = 1.0 - stats.norm.cdf(self.alpha2 - X_beta)
        
        # 考虑情境因素对潜变量的调整
        # 稳定性影响：高稳定性减少随机性，低稳定性增加随机性
        stability = voting_context.get('position_stability', 0.7)
        stability_adjustment = (0.5 - stability) * 0.2  # 稳定性越高，调整越小
        
        # 外部压力影响：高压可能使立场向中间移动（更可能弃权）
        external_pressure = voting_context.get('external_pressure', 0.3)
        pressure_adjustment = 0.0
        if external_pressure > 0.7:
            # 高压下，倾向于向中间移动
            pressure_adjustment = (current_position - 0.5) * (external_pressure - 0.7) * 0.3
        
        # 联盟对齐影响：高度对齐增强当前立场
        coalition_alignment = voting_context.get('coalition_alignment', 0.5)
        coalition_adjustment = 0.0
        if coalition_alignment > 0.7:
            # 高度对齐增强当前立场
            coalition_adjustment = (current_position - 0.5) * (coalition_alignment - 0.7) * 0.2
        elif coalition_alignment < 0.3:
            # 低度对齐削弱当前立场
            coalition_adjustment = (0.5 - current_position) * (0.5 - coalition_alignment) * 0.15
        
        # 关键时刻影响：可能采取策略性投票
        critical_adjustment = 0.0
        if voting_context.get('critical_momentum', False):
            # 关键时刻可能增强或减弱立场
            critical_factor = voting_context.get('strategic_direction', 0)  # -1到1
            critical_adjustment = critical_factor * 0.1
        
        # 理论权重影响（新三个维度）
        economic_weight = theory_weights.get('x_market', 0.33)
        domestic_weight = theory_weights.get('x_political', 0.33)
        external_weight = theory_weights.get('x_institutional', 0.33)
        
        # 高结构性经济约束权重：减少随机性，增强基于经济利益的决策
        if economic_weight > 0.4:
            stability_adjustment *= 0.8  # 减少调整幅度
        
        # 高国内政治—经济中介权重：可能受到外部压力影响
        if domestic_weight > 0.4 and external_pressure > 0.6:
            pressure_adjustment *= 1.3  # 增强压力影响
        
        # 高外部战略与互动权重：可能增强联盟对齐的影响
        if external_weight > 0.4:
            coalition_adjustment *= 1.2  # 增强联盟影响
        
        # 应用所有调整
        adjusted_X_beta = X_beta + stability_adjustment + pressure_adjustment + coalition_adjustment + critical_adjustment
        
        # 确保调整后的值在合理范围内（-2到2，覆盖大部分概率质量）
        adjusted_X_beta = max(-2.0, min(2.0, adjusted_X_beta))
        
        # 使用调整后的潜变量重新计算概率
        prob_oppose_adjusted = stats.norm.cdf(self.alpha1 - adjusted_X_beta)
        prob_abstain_adjusted = stats.norm.cdf(self.alpha2 - adjusted_X_beta) - stats.norm.cdf(self.alpha1 - adjusted_X_beta)
        prob_approve_adjusted = 1.0 - stats.norm.cdf(self.alpha2 - adjusted_X_beta)
        
        # 归一化概率（确保总和为1）
        total_prob = prob_oppose_adjusted + prob_abstain_adjusted + prob_approve_adjusted
        if total_prob > 0:
            prob_oppose_adjusted /= total_prob
            prob_abstain_adjusted /= total_prob
            prob_approve_adjusted /= total_prob
        
        # 添加随机噪声到概率（模拟个体差异和不确定性）
        noise = (random.random() * 2 - 1) * self.noise_level
        prob_oppose_adjusted = max(0.01, min(0.99, prob_oppose_adjusted + noise))
        prob_abstain_adjusted = max(0.01, min(0.99, prob_abstain_adjusted - noise))
        prob_approve_adjusted = max(0.01, min(0.99, prob_approve_adjusted))
        
        # 再次归一化
        total_prob = prob_oppose_adjusted + prob_abstain_adjusted + prob_approve_adjusted
        if total_prob > 0:
            prob_oppose_adjusted /= total_prob
            prob_abstain_adjusted /= total_prob
            prob_approve_adjusted /= total_prob
        
        # 根据概率分布进行投票
        # 使用随机抽样模拟概率决策
        rand_val = random.random()
        if rand_val < prob_oppose_adjusted:
            vote_decision = 'no'
            confidence = prob_oppose_adjusted
        elif rand_val < prob_oppose_adjusted + prob_abstain_adjusted:
            vote_decision = 'abstain'
            confidence = prob_abstain_adjusted
        else:
            vote_decision = 'yes'
            confidence = prob_approve_adjusted
        
        # 计算不确定性（熵）
        probs = [prob_oppose_adjusted, prob_abstain_adjusted, prob_approve_adjusted]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = np.log(3)  # 最大熵（均匀分布）
        uncertainty = entropy / max_entropy  # 0（确定）到1（完全不确定）
        
        # 置信度调整：基于选择的概率和不确定性
        confidence = confidence * (1.0 - uncertainty * 0.5)
        confidence = max(0.1, min(1.0, confidence))
        
        # 记录Ordered Probit计算详情（用于调试和分析）
        voting_context['ordered_probit_details'] = {
            'alpha1': self.alpha1,
            'alpha2': self.alpha2,
            'X_beta': X_beta,
            'adjusted_X_beta': adjusted_X_beta,
            'probabilities': {
                'oppose': prob_oppose_adjusted,
                'abstain': prob_abstain_adjusted,
                'approve': prob_approve_adjusted
            },
            'entropy': entropy,
            'uncertainty': uncertainty,
            'adjustments': {
                'stability': stability_adjustment,
                'pressure': pressure_adjustment,
                'coalition': coalition_adjustment,
                'critical': critical_adjustment
            }
        }
        
        logger.debug(f"Ordered Probit计算: Xβ={X_beta:.3f}, 调整后={adjusted_X_beta:.3f}, "
                    f"P(反对)={prob_oppose_adjusted:.3f}, P(弃权)={prob_abstain_adjusted:.3f}, "
                    f"P(赞成)={prob_approve_adjusted:.3f}, 最终投票={vote_decision}, 置信度={confidence:.3f}")
        
        return vote_decision, confidence
    
    def _generate_reasoning(self,
                           vote_decision: str,
                           confidence: float,
                           current_position: float,
                           voting_context: Dict[str, Any],
                           theory_weights: Dict[str, float],
                           decision_history: List[Dict[str, Any]]) -> str:
        """
        生成投票理由（基于Ordered Probit模型）
        
        Args:
            vote_decision: 投票决定
            confidence: 置信度
            current_position: 当前立场
            voting_context: 投票情境
            theory_weights: 理论权重
            decision_history: 决策历史
            
        Returns:
            投票理由文本
        """
        vote_text = self.vote_options.get(vote_decision, vote_decision)
        
        # 获取Ordered Probit计算详情
        probit_details = voting_context.get('ordered_probit_details', {})
        
        # 分析主要影响因素
        influencing_factors = []
        
        # 1. 立场因素
        position_strength = abs(current_position - 0.5) * 2  # 0到1
        if position_strength > 0.6:
            influencing_factors.append("强烈立场倾向")
        elif position_strength > 0.3:
            influencing_factors.append("明确立场倾向")
        else:
            influencing_factors.append("中立立场")
        
        # 2. 稳定性因素
        stability = voting_context.get('position_stability', 0.7)
        if stability > 0.8:
            influencing_factors.append("高度稳定的决策历史")
        elif stability < 0.5:
            influencing_factors.append("立场波动较大")
        
        # 3. 外部因素
        external_pressure = voting_context.get('external_pressure', 0.3)
        if external_pressure > 0.7:
            influencing_factors.append("高强度外部压力")
        elif external_pressure > 0.5:
            influencing_factors.append("中等外部压力")
        
        # 4. 联盟因素
        coalition_alignment = voting_context.get('coalition_alignment', 0.5)
        if coalition_alignment > 0.7:
            influencing_factors.append("高度联盟对齐")
        elif coalition_alignment < 0.3:
            influencing_factors.append("低联盟对齐")
        
        # 5. 关键时刻
        if voting_context.get('critical_momentum', False):
            influencing_factors.append("关键时刻策略性考虑")
        
        # 6. 理论权重影响（新三个维度）
        dominant_theory = max(theory_weights.items(), key=lambda x: x[1])[0] if theory_weights else None
        if dominant_theory:
            theory_names = {
                'x_market': '市场维度',
                'x_political': '政治维度',
                'x_institutional': '制度维度'
            }
            theory_name = theory_names.get(dominant_theory, dominant_theory)
            influencing_factors.append(f"{theory_name}主导")
        
        # Ordered Probit模型信息
        alpha1 = probit_details.get('alpha1', self.alpha1)
        alpha2 = probit_details.get('alpha2', self.alpha2)
        X_beta = probit_details.get('X_beta', current_position)
        adjusted_X_beta = probit_details.get('adjusted_X_beta', current_position)
        probabilities = probit_details.get('probabilities', {})
        uncertainty = probit_details.get('uncertainty', 0.5)
        
        prob_oppose = probabilities.get('oppose', 0)
        prob_abstain = probabilities.get('abstain', 0)
        prob_approve = probabilities.get('approve', 0)
        
        reasoning = f"""
        最终投票决策分析（Ordered Probit模型）：
        
        1. Ordered Probit模型设定：
           - 模型公式：Y* = Xβ + ε, ε ~ N(0,1)
           - 阈值参数：α1 = {alpha1}, α2 = {alpha2}
           - 潜变量 Xβ（原始）：{X_beta:.3f}
           - 潜变量 Xβ（调整后）：{adjusted_X_beta:.3f}
        
        2. 各类别概率分布：
           - P(反对) = Φ(α1 - Xβ) = {prob_oppose:.3f}
           - P(弃权) = Φ(α2 - Xβ) - Φ(α1 - Xβ) = {prob_abstain:.3f}
           - P(赞成) = 1 - Φ(α2 - Xβ) = {prob_approve:.3f}
           - 决策不确定性（熵归一化）：{uncertainty:.3f}
        
        3. 投票决定：
           - 最终投票：{vote_text}
           - 决策置信度：{confidence:.1%}
           - 投票选项：{vote_decision}
        
        4. 情境因素调整：
           - 立场稳定性调整：{probit_details.get('adjustments', {}).get('stability', 0):.3f}
           - 外部压力调整：{probit_details.get('adjustments', {}).get('pressure', 0):.3f}
           - 联盟对齐调整：{probit_details.get('adjustments', {}).get('coalition', 0):.3f}
           - 关键时刻调整：{probit_details.get('adjustments', {}).get('critical', 0):.3f}
        
        5. 基本立场分析：
           - 当前立场得分：{current_position:.3f}
           - 立场强度：{position_strength:.1%}
           - 历史决策次数：{len(decision_history)}
        
        6. 情境因素分析：
           - 立场稳定性：{stability:.1%}
           - 外部压力水平：{external_pressure:.1%}
           - 联盟对齐程度：{coalition_alignment:.1%}
           - 关键时刻：{'是' if voting_context.get('critical_momentum', False) else '否'}
        
        7. 主要影响因素：
           {chr(10).join(f'           - {factor}' for factor in influencing_factors)}
        
        8. 理论框架影响（新三个维度）：
           - 市场维度权重：{theory_weights.get('x_market', 0.33):.1%}
           - 政治维度权重：{theory_weights.get('x_political', 0.33):.1%}
           - 制度维度权重：{theory_weights.get('x_institutional', 0.33):.1%}
        
        9. 决策逻辑：
           - 使用Ordered Probit有序概率模型
           - 基于潜变量 Xβ 和阈值 α1, α2 计算各类别概率
           - 考虑了历史决策模式和外部约束的调整
           - 融入了理论框架的指导原则
           - 最终决策反映了国家利益和战略考量的平衡
        """
        
        return reasoning.strip()
    
    def _generate_vote_message(self, vote_decision: str, confidence: float) -> str:
        """
        生成投票消息
        
        Args:
            vote_decision: 投票决定
            confidence: 置信度
            
        Returns:
            投票消息文本
        """
        vote_text = self.vote_options.get(vote_decision, vote_decision)
        
        # 根据置信度生成不同语气
        if confidence > 0.8:
            confidence_text = "高度确信"
        elif confidence > 0.6:
            confidence_text = "较为确信"
        elif confidence > 0.4:
            confidence_text = "基本确定"
        else:
            confidence_text = "有所保留"
        
        return f"我方最终投票：{vote_text}（{confidence_text}，置信度：{confidence:.1%}）"


if __name__ == "__main__":
    # 测试投票动作
    import asyncio
    
    # 模拟国家特征
    features = {
        'economic': {
            'trade_dependency_china': 'medium',
            'automotive_industry_share': 'high'
        }
    }
    
    # 模拟理论权重（新三个维度）
    theory_weights = {
        'x_market': 0.35,
        'x_political': 0.35,
        'x_institutional': 0.3
    }
    
    # 模拟决策历史
    decision_history = [
        {'timestamp': '2024-01-01', 'position': 0.65, 'situation': 'initial_decision'},
        {'timestamp': '2024-01-02', 'position': 0.68, 'situation': 'negotiation'},
        {'timestamp': '2024-01-03', 'position': 0.70, 'situation': 'negotiation'}
    ]
    
    # 模拟投票上下文
    context = {
        'external_pressure': 0.4,
        'coalition_alignment': 0.7,
        'coalition_support': 0.68,
        'description': '最终投票阶段'
    }
    
    async def test_action():
        action = VotingAction()
        result = await action.run(
            country_features=features,
            current_position=0.7,
            decision_history=decision_history,
            theory_weights=theory_weights,
            context=context
        )
        
        print("投票动作测试结果:")
        print(f"投票决定: {result['vote_text']} ({result['vote']})")
        print(f"置信度: {result['confidence']:.1%}")
        print(f"投票消息: {result['vote_message']}")
        print(f"\n投票情境分析:")
        for key, value in result['voting_context'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
