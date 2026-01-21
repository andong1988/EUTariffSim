#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谈判动作 - 处理国家间的谈判和协商
"""

from typing import Dict, Any, List, Optional
from metagpt.actions import Action
import logging
import random

logger = logging.getLogger(__name__)


class NegotiationAction(Action):
    """谈判动作：处理国家间的谈判和协商"""
    
    name: str = "NegotiationAction"
    desc: str = "处理国家间谈判和协商的动作"
    
    def __init__(self, **kwargs):
        """
        初始化谈判动作
        
        Args:
            **kwargs: Action参数
        """
        super().__init__(**kwargs)
        
        # 谈判策略
        self.strategies = {
            'hardline': '强硬立场',
            'compromise': '妥协立场',
            'cooperative': '合作立场',
            'defensive': '防御立场'
        }
        
        # 不再设置self.config，避免触发set_config方法
        # 将配置作为实例变量直接存储
        self.noise_level = 0.05
    
    async def run(self,
                  country_features: Dict[str, Any],
                  current_position: float,
                  opponents: List[str],
                  negotiation_history: List[Dict[str, Any]],
                  theory_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        执行谈判
        
        Args:
            country_features: 国家特征
            current_position: 当前立场（0-1）
            opponents: 谈判对手列表
            negotiation_history: 谈判历史
            theory_weights: 理论权重
            
        Returns:
            谈判结果
        """
        logger.info(f"执行谈判，对手: {opponents}, 当前立场: {current_position:.3f}")
        
        # 1. 分析谈判情境
        negotiation_context = self._analyze_negotiation_context(
            opponents, negotiation_history, current_position
        )
        
        # 2. 选择谈判策略
        strategy = self._select_negotiation_strategy(
            country_features, negotiation_context, theory_weights
        )
        
        # 3. 计算立场调整
        position_adjusted, new_position, adjustment_reasoning = self._calculate_position_adjustment(
            current_position, negotiation_context, strategy, theory_weights
        )
        
        # 4. 计算学习效应
        learning_effect = self._calculate_learning_effect(
            negotiation_context, strategy, position_adjusted
        )
        
        # 5. 生成谈判消息
        negotiation_message = self._generate_negotiation_message(
            strategy, position_adjusted, new_position, opponents
        )
        
        # 6. 使用LLM生成详细理由
        reasoning = await self._generate_reasoning_with_llm(
            strategy, negotiation_context, position_adjusted,
            new_position, adjustment_reasoning, learning_effect,
            country_features, opponents, theory_weights
        )
        
        result = {
            'position_adjusted': position_adjusted,
            'new_position': new_position,
            'adjustment_magnitude': abs(new_position - current_position) if position_adjusted else 0,
            'strategy': strategy,
            'strategy_name': self.strategies.get(strategy, strategy),
            'negotiation_message': negotiation_message,
            'learning_effect': learning_effect,
            'reasoning': reasoning,
            'opponents': opponents.copy()
        }
        
        logger.debug(f"谈判结果: {result}")
        return result
    
    def _analyze_negotiation_context(self,
                                    opponents: List[str],
                                    history: List[Dict[str, Any]],
                                    current_position: float) -> Dict[str, Any]:
        """
        分析谈判情境
        
        Args:
            opponents: 谈判对手
            history: 谈判历史
            current_position: 当前立场
            
        Returns:
            谈判情境分析
        """
        context = {
            'opponent_count': len(opponents),
            'history_length': len(history),
            'current_position': current_position,
            'opponent_positions': {},
            'consensus_level': 0.5,
            'pressure_level': 0.3
        }
        
        # 分析历史记录中的对手立场
        if history:
            recent_opponent_positions = []
            for record in history[-5:]:  # 最近5条记录
                if 'position' in record:
                    recent_opponent_positions.append(record['position'])
            
            if recent_opponent_positions:
                avg_opponent_position = sum(recent_opponent_positions) / len(recent_opponent_positions)
                context['avg_opponent_position'] = avg_opponent_position
                
                # 计算共识水平（1 - 立场差异）
                position_diff = abs(current_position - avg_opponent_position)
                context['consensus_level'] = 1 - position_diff
                
                # 计算压力水平（对手数量 * 立场差异）
                context['pressure_level'] = min(1.0, len(opponents) * position_diff * 0.5)
        
        # 分析对手特征（简化版）
        for opponent in opponents:
            # 这里可以添加更复杂的对手分析
            context['opponent_positions'][opponent] = random.uniform(0.3, 0.7)
        
        return context
    
    def _select_negotiation_strategy(self,
                                    features: Dict[str, Any],
                                    context: Dict[str, Any],
                                    theory_weights: Dict[str, float]) -> str:
        """
        选择谈判策略
        
        Args:
            features: 国家特征
            context: 谈判情境
            theory_weights: 理论权重
            
        Returns:
            策略标识符
        """
        # 基于理论权重选择策略
        rational_weight = theory_weights.get('rational_choice', 0.25)
        constructivist_weight = theory_weights.get('constructivism', 0.25)
        
        # 基于情境选择策略
        pressure_level = context.get('pressure_level', 0.3)
        consensus_level = context.get('consensus_level', 0.5)
        
        # 决策逻辑
        if pressure_level > 0.7:
            # 高压情境：防御或妥协
            if rational_weight > constructivist_weight:
                return 'defensive'  # 理性选择：保护自身利益
            else:
                return 'compromise'  # 建构主义：寻求共识
        elif consensus_level > 0.7:
            # 高共识情境：合作
            return 'cooperative'
        elif context.get('opponent_count', 0) == 1:
            # 双边谈判：可能采取强硬立场
            if 'political' in features:
                political_orientation = features['political'].get('political_orientation', 'centrist')
                if political_orientation == 'protectionist':
                    return 'hardline'
        
        # 默认策略：妥协
        return 'compromise'
    
    def _calculate_position_adjustment(self,
                                      current_position: float,
                                      context: Dict[str, Any],
                                      strategy: str,
                                      theory_weights: Dict[str, float]) -> tuple:
        """
        计算立场调整
        
        Args:
            current_position: 当前立场
            context: 谈判情境
            strategy: 谈判策略
            theory_weights: 理论权重
            
        Returns:
            (是否调整, 新立场, 调整理由)
        """
        # 获取对手平均立场
        avg_opponent_position = context.get('avg_opponent_position')
        if avg_opponent_position is None:
            return False, current_position, "无对手立场信息，保持原立场"
        
        # 基于策略计算调整幅度
        adjustment_factors = {
            'hardline': 0.0,      # 强硬：不调整
            'compromise': 0.3,    # 妥协：适度调整
            'cooperative': 0.5,   # 合作：较大调整
            'defensive': 0.1      # 防御：轻微调整
        }
        
        base_adjustment = adjustment_factors.get(strategy, 0.2)
        
        # 考虑理论权重的影响
        two_level_weight = theory_weights.get('two_level_games', 0.25)
        interdependence_weight = theory_weights.get('weaponized_interdependence', 0.25)
        
        # 双层博弈：考虑国内约束，调整幅度较小
        domestic_constraint_factor = 1.0 - two_level_weight * 0.5
        
        # 相互依赖武器化：考虑脆弱性，可能避免大幅调整
        vulnerability_factor = 1.0 - interdependence_weight * 0.3
        
        # 综合调整因子
        adjustment_factor = base_adjustment * domestic_constraint_factor * vulnerability_factor
        
        # 计算目标立场（向对手立场靠拢）
        position_diff = avg_opponent_position - current_position
        adjustment = position_diff * adjustment_factor
        
        # 添加随机扰动
        noise = (random.random() * 2 - 1) * self.noise_level
        adjustment += noise
        
        # 应用调整
        new_position = current_position + adjustment
        new_position = max(0, min(1, new_position))  # 限制在0-1范围内
        
        # 判断是否实际调整了立场
        position_adjusted = abs(adjustment) > 0.01  # 调整幅度大于1%才视为调整
        
        # 生成调整理由
        if position_adjusted:
            direction = "向对手靠拢" if adjustment > 0 else "远离对手立场"
            reasoning = f"基于{self.strategies.get(strategy, strategy)}策略，{direction}，调整幅度: {abs(adjustment):.3f}"
        else:
            reasoning = "立场调整幅度过小，保持原立场"
        
        return position_adjusted, new_position, reasoning
    
    def _calculate_learning_effect(self,
                                  context: Dict[str, Any],
                                  strategy: str,
                                  position_adjusted: bool) -> Dict[str, float]:
        """
        计算学习效应（理论权重调整）
        
        Args:
            context: 谈判情境
            strategy: 谈判策略
            position_adjusted: 是否调整了立场
            
        Returns:
            各理论的学习效应（正值表示增加权重，负值表示减少）
        """
        learning_effect = {
            'rational_choice': 0.0,
            'two_level_games': 0.0,
            'constructivism': 0.0,
            'weaponized_interdependence': 0.0
        }
        
        if not position_adjusted:
            # 未调整立场：理性选择和双层博弈理论可能更相关
            learning_effect['rational_choice'] = 0.05
            learning_effect['two_level_games'] = 0.05
            return learning_effect
        
        # 根据策略调整学习效应
        if strategy == 'cooperative':
            # 合作策略：建构主义和相互依赖理论更相关
            learning_effect['constructivism'] = 0.1
            learning_effect['weaponized_interdependence'] = 0.05
        elif strategy == 'compromise':
            # 妥协策略：理性选择和双层博弈理论
            learning_effect['rational_choice'] = 0.08
            learning_effect['two_level_games'] = 0.08
        elif strategy == 'hardline':
            # 强硬策略：理性选择和相互依赖武器化
            learning_effect['rational_choice'] = 0.1
            learning_effect['weaponized_interdependence'] = 0.1
        elif strategy == 'defensive':
            # 防御策略：双层博弈和相互依赖武器化
            learning_effect['two_level_games'] = 0.1
            learning_effect['weaponized_interdependence'] = 0.08
        
        # 根据共识水平调整
        consensus_level = context.get('consensus_level', 0.5)
        if consensus_level > 0.7:
            # 高共识：建构主义更相关
            learning_effect['constructivism'] += 0.05
        elif consensus_level < 0.3:
            # 低共识：理性选择更相关
            learning_effect['rational_choice'] += 0.05
        
        return learning_effect
    
    def _generate_negotiation_message(self,
                                     strategy: str,
                                     position_adjusted: bool,
                                     new_position: float,
                                     opponents: List[str]) -> str:
        """
        生成谈判消息
        
        Args:
            strategy: 谈判策略
            position_adjusted: 是否调整了立场
            new_position: 新立场
            opponents: 谈判对手
            
        Returns:
            谈判消息文本
        """
        strategy_name = self.strategies.get(strategy, strategy)
        
        if not position_adjusted:
            return f"基于{strategy_name}，我方坚持原有立场。"
        
        # 根据立场变化生成消息
        if new_position > 0.7:
            position_text = "强烈支持关税"
        elif new_position > 0.6:
            position_text = "支持关税"
        elif new_position > 0.4:
            position_text = "中立立场"
        elif new_position > 0.3:
            position_text = "反对关税"
        else:
            position_text = "强烈反对关税"
        
        opponent_text = "、".join(opponents) if opponents else "各方"
        
        return f"经过与{opponent_text}的协商，基于{strategy_name}，我方调整立场为：{position_text}。"
    
    async def _generate_reasoning_with_llm(self,
                                          strategy: str,
                                          context: Dict[str, Any],
                                          position_adjusted: bool,
                                          new_position: float,
                                          adjustment_reasoning: str,
                                          learning_effect: Dict[str, float],
                                          country_features: Dict[str, Any],
                                          opponents: List[str],
                                          theory_weights: Dict[str, float]) -> str:
        """
        使用LLM生成详细理由
        
        Args:
            strategy: 谈判策略
            context: 谈判情境
            position_adjusted: 是否调整了立场
            new_position: 新立场
            adjustment_reasoning: 调整理由
            learning_effect: 学习效应
            country_features: 国家特征
            opponents: 谈判对手
            theory_weights: 理论权重
            
        Returns:
            详细理由文本
        """
        strategy_name = self.strategies.get(strategy, strategy)
        
        # 准备LLM提示
        prompt = f"""
        你是一个国际谈判专家，正在分析一个国家在欧盟对华汽车关税谈判中的表现。
        
        国家特征：
        {country_features}
        
        谈判对手：{', '.join(opponents) if opponents else '无'}
        
        谈判情境：
        - 对手数量：{context.get('opponent_count', 0)}
        - 谈判历史记录：{context.get('history_length', 0)}条
        - 共识水平：{context.get('consensus_level', 0.5):.2f}
        - 压力水平：{context.get('pressure_level', 0.3):.2f}
        
        理论权重：
        - 理性选择：{theory_weights.get('rational_choice', 0.25):.1%}
        - 双层博弈：{theory_weights.get('two_level_games', 0.25):.1%}
        - 建构主义：{theory_weights.get('constructivism', 0.25):.1%}
        - 相互依赖武器化：{theory_weights.get('weaponized_interdependence', 0.25):.1%}
        
        谈判策略：{strategy_name}
        
        立场调整：
        - 是否调整：{'是' if position_adjusted else '否'}
        - 调整理由：{adjustment_reasoning}
        - 新立场得分：{new_position:.3f}
        
        学习效应（理论权重调整建议）：
        - 理性选择理论：{learning_effect.get('rational_choice', 0):+.3f}
        - 双层博弈理论：{learning_effect.get('two_level_games', 0):+.3f}
        - 建构主义理论：{learning_effect.get('constructivism', 0):+.3f}
        - 相互依赖武器化理论：{learning_effect.get('weaponized_interdependence', 0):+.3f}
        
        请基于以上信息，生成一个详细的谈判分析报告，包括：
        1. 谈判策略选择的合理性分析
        2. 立场调整的决策逻辑
        3. 学习效应的理论解释
        4. 谈判结果的评估
        5. 对未来谈判的建议
        
        请用中文回答，保持专业但清晰易懂。
        """
        
        try:
            # 使用LLM生成理由
            reasoning = await self._aask(prompt)
            return reasoning
        except Exception as e:
            logger.error(f"LLM生成谈判理由失败: {e}")
            # 回退到简单理由生成
            return self._generate_fallback_reasoning(
                strategy, context, position_adjusted, new_position, 
                adjustment_reasoning, learning_effect
            )
    
    def _generate_fallback_reasoning(self,
                                    strategy: str,
                                    context: Dict[str, Any],
                                    position_adjusted: bool,
                                    new_position: float,
                                    adjustment_reasoning: str,
                                    learning_effect: Dict[str, float]) -> str:
        """
        回退的详细理由生成（当LLM失败时使用）
        """
        strategy_name = self.strategies.get(strategy, strategy)
        
        reasoning = f"""
        谈判分析报告：
        
        1. 谈判情境：
           - 对手数量：{context.get('opponent_count', 0)}
           - 谈判历史记录：{context.get('history_length', 0)}条
           - 共识水平：{context.get('consensus_level', 0.5):.2f}
           - 压力水平：{context.get('pressure_level', 0.3):.2f}
        
        2. 策略选择：
           - 选用策略：{strategy_name}
           - 策略依据：基于谈判情境和理论权重分析
        
        3. 立场调整：
           - 是否调整：{'是' if position_adjusted else '否'}
           - 调整理由：{adjustment_reasoning}
           - 新立场得分：{new_position:.3f}
        
        4. 学习效应（理论权重调整建议）：
           - 理性选择理论：{learning_effect.get('rational_choice', 0):+.3f}
           - 双层博弈理论：{learning_effect.get('two_level_games', 0):+.3f}
           - 建构主义理论：{learning_effect.get('constructivism', 0):+.3f}
           - 相互依赖武器化理论：{learning_effect.get('weaponized_interdependence', 0):+.3f}
        
        5. 谈判结论：
           - 本次谈判{'成功达成共识' if context.get('consensus_level', 0.5) > 0.6 else '未能达成共识'}
           - 建议后续{'继续合作' if strategy == 'cooperative' else '保持当前策略'}
        """
        
        return reasoning.strip()


if __name__ == "__main__":
    # 测试谈判动作
    import asyncio
    
    # 模拟国家特征
    features = {
        'economic': {
            'trade_dependency_china': 'medium',
            'automotive_industry_share': 'high'
        },
        'political': {
            'political_orientation': 'centrist',
            'eu_integration_level': 'high'
        }
    }
    
    # 模拟理论权重
    theory_weights = {
        'rational_choice': 0.3,
        'two_level_games': 0.3,
        'constructivism': 0.2,
        'weaponized_interdependence': 0.2
    }
    
    # 模拟谈判参数
    current_position = 0.6
    opponents = ["Country_B", "Country_C"]
    negotiation_history = [
        {'from': 'Country_B', 'position': 0.7, 'content': '建议支持关税'},
        {'from': 'Country_C', 'position': 0.4, 'content': '建议反对关税'}
    ]
    
    async def test_action():
        action = NegotiationAction()
        result = await action.run(
            country_features=features,
            current_position=current_position,
            opponents=opponents,
            negotiation_history=negotiation_history,
            theory_weights=theory_weights
        )
        
        print("谈判动作测试结果:")
        print(f"立场调整: {'是' if result['position_adjusted'] else '否'}")
        print(f"新立场: {result['new_position']:.3f}")
        print(f"策略: {result['strategy_name']}")
        print(f"谈判消息: {result['negotiation_message']}")
        print(f"\n学习效应:")
        for theory, effect in result['learning_effect'].items():
            print(f"  {theory}: {effect:+.3f}")
        print(f"\n详细理由:\n{result['reasoning']}")
    
    asyncio.run(test_action())
