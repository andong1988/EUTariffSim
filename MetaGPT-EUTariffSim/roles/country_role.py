"""
国家角色模块

定义欧盟成员国的智能体角色，实现基于理论约束的决策机制。
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from metagpt.roles.role import Role
from metagpt.actions.action import Action
from metagpt.schema import Message

logger = logging.getLogger(__name__)

from data_processing.anonymization_processor import AnonymizedCountry
from theory.theory_encoder import TheoryEncoder, TheoryContext, TheoryType, TheoryWeight
from theory.theory_constraint_prompter import TheoryConstraintPrompter
from theory.theory_consistency_checker import TheoryConsistencyChecker, ConsistencyLevel


class VotingPosition(Enum):
    """投票立场枚举"""
    SUPPORT = "support"  # 支持
    OPPOSE = "oppose"    # 反对
    ABSTAIN = "abstain"  # 弃权


@dataclass
class DomesticActor:
    """国内行为体"""
    name: str
    type: str  # "government", "industry", "labor", "public", "party"
    power_weight: float  # 权力权重 (0-1)
    position: VotingPosition  # 立场
    arguments: List[str]  # 论据


@dataclass
class CountryState:
    """国家状态"""
    current_position: VotingPosition
    position_strength: float  # 立场强度 (0-1)
    domestic_actors: List[DomesticActor]
    negotiation_history: List[Dict[str, Any]]
    communication_network: Dict[str, int]  # 与其他国家的通信次数
    theory_weights: TheoryWeight
    learning_rate: float = 0.1  # 学习率


class CountryRole(Role):
    """国家角色智能体"""
    
    def __init__(
        self,
        name: str,
        anonymized_country: AnonymizedCountry,
        profile: str = "EU Member State",
        goal: str = "Make decisions based on national interests and theoretical frameworks",
        constraints: str = "Must follow theoretical constraints and maintain anonymity"
    ):
        super().__init__(
            name=name, 
            profile=profile, 
            goal=goal, 
            constraints=constraints
        )
        
        self.anonymized_country = anonymized_country
        self.theory_encoder = TheoryEncoder()
        self.theory_prompter = TheoryConstraintPrompter()
        self.theory_checker = TheoryConsistencyChecker()
        
        # 初始化国家状态
        self.state = CountryState(
            current_position=VotingPosition.ABSTAIN,
            position_strength=0.5,
            domestic_actors=self._create_domestic_actors(),
            negotiation_history=[],
            communication_network={},
            theory_weights=self._initialize_theory_weights(),
            learning_rate=0.1
        )
        
        # 设置动作
        self.set_actions([
            InitialAnalysisAction(self),
            NegotiationAction(self),
            VotingDecisionAction(self),
            EventResponseAction(self)
        ])
    
    def _create_domestic_actors(self) -> List[DomesticActor]:
        """创建国内行为体"""
        actors = []
        
        # 政府部门
        actors.append(DomesticActor(
            name="Government",
            type="government",
            power_weight=0.4,
            position=VotingPosition.ABSTAIN,
            arguments=["National sovereignty", "Economic stability"]
        ))
        
        # 产业界
        actors.append(DomesticActor(
            name="Automotive Industry",
            type="industry", 
            power_weight=0.3,
            position=VotingPosition.ABSTAIN,
            arguments=["Market access", "Supply chain", "Competitiveness"]
        ))
        
        # 工会
        actors.append(DomesticActor(
            name="Labor Unions",
            type="labor",
            power_weight=0.2,
            position=VotingPosition.ABSTAIN,
            arguments=["Employment", "Worker rights", "Fair competition"]
        ))
        
        # 公众
        actors.append(DomesticActor(
            name="Public Opinion",
            type="public",
            power_weight=0.1,
            position=VotingPosition.ABSTAIN,
            arguments=["Consumer prices", "Environmental concerns", "National interests"]
        ))
        
        return actors
    
    def _initialize_theory_weights(self) -> TheoryWeight:
        """初始化理论权重"""
        # 基于国家特征初始化理论权重
        # 使用实际的特征结构
        from data_processing.anonymization_processor import FeatureLevel
        
        gdp_level = self.anonymized_country.economic_features.get("gdp_per_capita", FeatureLevel.MEDIUM)
        trade_dep = self.anonymized_country.economic_features.get("trade_dependency_china", FeatureLevel.MEDIUM)
        
        # 根据经济特征设置理性选择权重
        if gdp_level in [FeatureLevel.HIGH, FeatureLevel.VERY_HIGH]:
            economic_weight = 0.4
        else:
            economic_weight = 0.3
        
        # 根据政治特征设置双层博弈权重
        eu_integration = self.anonymized_country.political_features.get("eu_integration_level", FeatureLevel.MEDIUM)
        if eu_integration in [FeatureLevel.HIGH, FeatureLevel.VERY_HIGH]:
            political_weight = 0.3
        else:
            political_weight = 0.2
        
        constructivist_weight = 0.2
        interdependence_weight = 0.1
        
        return TheoryWeight(
            rational_choice=economic_weight,
            two_level_game=political_weight,
            constructivism=constructivist_weight,
            dependence_weaponization=interdependence_weight
        )
    
    async def _analyze_initial_position(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析初始立场"""
        # 创建理论上下文
        theory_context = TheoryContext(
            country=self.anonymized_country,
            issue="欧盟对华汽车关税投票",
            timestamp="2024-01-01"
        )
        
        # 编码理论
        theory_analysis = self.theory_encoder.encode_all_theories(theory_context)
        
        # 生成理论约束提示
        prompt_data = self.theory_prompter.generate_theory_constrained_prompt(
            theory_context, self.state.theory_weights
        )
        
        # LLM决策
        system_prompt = prompt_data["system_prompt"]
        user_prompt = prompt_data["user_prompt"]
        
        # 构建完整提示（包含系统提示）
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = await self.llm.aask(full_prompt)
        
        # 检查一致性
        consistency_result = self.theory_checker.check_consistency(
            response, theory_context, self.state.theory_weights
        )
        
        # 解析决策
        try:
            decision_data = json.loads(response)
            position = VotingPosition(decision_data.get("position", "abstain"))
            confidence = decision_data.get("confidence", 0.5)
            reasoning = decision_data.get("reasoning", "")
        except:
            # 如果解析失败，使用理论回退
            position = self._theoretical_fallback(theory_analysis, self.state.theory_weights)
            confidence = 0.5
            reasoning = "Theoretical fallback due to parsing error"
        
        # 更新状态
        self.state.current_position = position
        self.state.position_strength = confidence
        
        # 序列化理论分析结果以处理FeatureLevel枚举
        serialized_theory_analysis = self.theory_encoder.serialize_for_json(theory_analysis)
        
        return {
            "country_id": self.anonymized_country.anonymous_id,
            "position": position.value,
            "confidence": confidence,
            "reasoning": reasoning,
            "theory_analysis": serialized_theory_analysis,
            "consistency_result": consistency_result.dict() if hasattr(consistency_result, 'dict') else str(consistency_result)
        }
    
    def _theoretical_fallback(self, theory_analysis: Dict[str, Any], weights: TheoryWeight) -> VotingPosition:
        """理论回退决策"""
        # 基于理论分析结果进行加权决策
        support_score = 0
        oppose_score = 0
        
        # 理性选择理论
        rational = theory_analysis.get("rational_choice", {})
        if rational.get("economic_utility", 0) > 0:
            support_score += weights.rational_choice
        else:
            oppose_score += weights.rational_choice
        
        # 双层博弈理论
        two_level = theory_analysis.get("two_level_game", {})
        if two_level.get("domestic_support", 0) > 0.5:
            support_score += weights.two_level_game
        else:
            oppose_score += weights.two_level_game
        
        # 建构主义理论
        constructivist = theory_analysis.get("constructivism", {})
        if constructivist.get("normative_fit", 0) > 0.5:
            support_score += weights.constructivism
        else:
            oppose_score += weights.constructivism
        
        # 相互依赖武器化理论
        interdependence = theory_analysis.get("dependence_weaponization", {})
        if interdependence.get("vulnerability", 0) < 0.5:
            support_score += weights.dependence_weaponization
        else:
            oppose_score += weights.dependence_weaponization
        
        return VotingPosition.SUPPORT if support_score > oppose_score else VotingPosition.OPPOSE
    
    async def _negotiate(self, negotiation_data: Dict[str, Any]) -> Dict[str, Any]:
        """参与谈判"""
        partner_id = negotiation_data.get("partner_id")
        offer = negotiation_data.get("offer", {})
        round_num = negotiation_data.get("round", 1)
        
        # 记录谈判历史
        self.state.negotiation_history.append({
            "round": round_num,
            "partner": partner_id,
            "offer": offer,
            "timestamp": self._get_current_time()
        })
        
        # 更新通信网络
        if partner_id in self.state.communication_network:
            self.state.communication_network[partner_id] += 1
        else:
            self.state.communication_network[partner_id] = 1
        
        # 基于当前立场和谈判内容生成响应
        response = await self._generate_negotiation_response(offer, partner_id, round_num)
        
        return {
            "country_id": self.anonymized_country.anonymous_id,
            "partner_id": partner_id,
            "round": round_num,
            "response": response,
            "current_position": self.state.current_position.value,
            "position_strength": self.state.position_strength
        }
    
    async def _generate_negotiation_response(self, offer: Dict[str, Any], partner_id: str, round_num: int) -> Dict[str, Any]:
        """生成谈判响应"""
        prompt = f"""
        As Country {self.anonymized_country.anonymous_id}, you are in round {round_num} of negotiations with Country {partner_id}.
        
        Your current position: {self.state.current_position.value}
        Your position strength: {self.state.position_strength}
        
        The offer from partner: {offer}
        
        Based on your country characteristics and theoretical framework, generate a negotiation response:
        
        1. Do you accept the offer? (yes/no/partial)
        2. What are your counter-offers if any?
        3. What are your main concerns?
        4. How flexible is your position? (scale 1-10)
        
        Respond in JSON format:
        {{
            "acceptance": "yes/no/partial",
            "counter_offers": [],
            "concerns": [],
            "flexibility": 5,
            "reasoning": ""
        }}
        """
        
        response = await self.llm.aask(prompt)
        
        try:
            return json.loads(response)
        except:
            return {
                "acceptance": "partial",
                "counter_offers": [],
                "concerns": ["Need more information"],
                "flexibility": 5,
                "reasoning": "Parsing error, default response"
            }
    
    async def _make_voting_decision(self, voting_context: Dict[str, Any]) -> Dict[str, Any]:
        """做出投票决策"""
        # 综合考虑所有因素后的最终决策
        final_position = await self._analyze_final_position(voting_context)
        
        # 序列化final_position以处理FeatureLevel枚举
        try:
            serialized_final_position = self.theory_encoder.serialize_for_json(final_position)
        except Exception as e:
            logger.warning(f"序列化最终立场失败: {e}")
            serialized_final_position = final_position
        
        return {
            "country_id": self.anonymized_country.anonymous_id,
            "final_position": serialized_final_position.get("position", "abstain"),
            "confidence": serialized_final_position.get("confidence", 0.5),
            "reasoning": serialized_final_position.get("reasoning", ""),
            "theoretical_justification": serialized_final_position.get("theory_analysis", {}),
            "domestic_consensus": self._calculate_domestic_consensus(),
            "negotiation_impact": self._assess_negotiation_impact()
        }
    
    async def _analyze_final_position(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析最终立场"""
        # 基于历史互动和学习调整立场
        adjusted_weights = self._adjust_theory_weights()
        
        # 创建更新的理论上下文
        theory_context = TheoryContext(
            country=self.anonymized_country,
            issue="欧盟对华汽车关税最终投票",
            timestamp="2024-01-01"
        )
        
        # 编码理论分析
        theory_analysis = self.theory_encoder.encode_all_theories(theory_context)
        
        # 生成最终决策提示
        prompt_data = self.theory_prompter.generate_theory_constrained_prompt(
            theory_context, adjusted_weights
        )
        
        system_prompt = prompt_data["system_prompt"]
        user_prompt = prompt_data["user_prompt"] + "\n\nThis is your final voting decision. Consider all negotiations and interactions."
        
        # 构建完整提示（包含系统提示）
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = await self.llm.aask(full_prompt)
        
        try:
            decision_data = json.loads(response)
            # 序列化理论分析结果
            serialized_theory_analysis = self.theory_encoder.serialize_for_json(theory_analysis)
            return {
                "position": decision_data.get("position", self.state.current_position.value),
                "confidence": decision_data.get("confidence", self.state.position_strength),
                "reasoning": decision_data.get("reasoning", "Final decision based on comprehensive analysis"),
                "theory_analysis": serialized_theory_analysis
            }
        except:
            # 序列化理论分析结果
            serialized_theory_analysis = self.theory_encoder.serialize_for_json(theory_analysis)
            return {
                "position": self.state.current_position.value,
                "confidence": self.state.position_strength,
                "reasoning": "Final decision based on current position",
                "theory_analysis": serialized_theory_analysis
            }
    
    def _adjust_theory_weights(self) -> TheoryWeight:
        """调整理论权重（基于学习）"""
        # 基于谈判历史调整理论权重
        if len(self.state.negotiation_history) > 0:
            # 简单的学习机制：根据谈判成功率调整权重
            successful_negotiations = sum(1 for n in self.state.negotiation_history if n.get("success", False))
            negotiation_success_rate = successful_negotiations / len(self.state.negotiation_history)
            
            # 如果谈判成功，增加双层博弈权重
            if negotiation_success_rate > 0.7:
                self.state.theory_weights.two_level_game += self.state.learning_rate * 0.1
                self.state.theory_weights.rational_choice -= self.state.learning_rate * 0.05
        
        return self.state.theory_weights
    
    def _calculate_domestic_consensus(self) -> float:
        """计算国内共识度"""
        if not self.state.domestic_actors:
            return 0.0
        
        # 计算加权共识度
        total_weight = sum(actor.power_weight for actor in self.state.domestic_actors)
        consensus_score = 0.0
        
        for actor in self.state.domestic_actors:
            if actor.position == self.state.current_position:
                consensus_score += actor.power_weight
        
        return consensus_score / total_weight if total_weight > 0 else 0.0
    
    def _assess_negotiation_impact(self) -> Dict[str, Any]:
        """评估谈判影响"""
        if not self.state.negotiation_history:
            return {"impact_level": 0, "key_partners": [], "position_changes": 0}
        
        # 分析谈判对立场的影响
        position_changes = len(set(n.get("position_after", self.state.current_position) for n in self.state.negotiation_history))
        
        # 识别关键伙伴
        communication_counts = self.state.communication_network
        key_partners = [partner for partner, count in communication_counts.items() if count >= 3]
        
        impact_level = min(1.0, len(self.state.negotiation_history) / 10.0)
        
        return {
            "impact_level": impact_level,
            "key_partners": key_partners,
            "position_changes": position_changes
        }
    
    async def _respond_to_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """响应事件"""
        event_type = event_data.get("type", "unknown")
        event_intensity = event_data.get("intensity", 0.5)
        
        # 创建事件响应上下文
        theory_context = TheoryContext(
            country=self.anonymized_country,
            issue="事件响应",
            timestamp="2024-01-01"
        )
        
        # 生成事件响应提示
        prompt = f"""
        As Country {self.anonymized_country.anonymous_id}, you need to respond to an event:
        
        Event type: {event_type}
        Event intensity: {event_intensity}
        Event details: {event_data}
        
        Based on your country characteristics and theoretical framework, how would you respond?
        
        Consider:
        1. How does this event affect your national interests?
        2. Which theoretical perspective is most relevant?
        3. What is your preferred response?
        4. How might this change your voting position?
        
        Respond in JSON format:
        {{
            "event_impact": "high/medium/low",
            "response_type": "support/oppose/neutral",
            "position_change": "yes/no",
            "new_position": "support/oppose/abstain",
            "reasoning": "",
            "theoretical_basis": ""
        }}
        """
        
        response = await self.llm.aask(prompt)
        
        try:
            response_data = json.loads(response)
            
            # 如果事件导致立场变化，更新状态
            if response_data.get("position_change") == "yes":
                new_position = VotingPosition(response_data.get("new_position", "abstain"))
                self.state.current_position = new_position
            
            return {
                "country_id": self.anonymized_country.anonymous_id,
                "event_type": event_type,
                "event_intensity": event_intensity,
                "response": response_data
            }
        except:
            return {
                "country_id": self.anonymized_country.anonymous_id,
                "event_type": event_type,
                "event_intensity": event_intensity,
                "response": {
                    "event_impact": "medium",
                    "response_type": "neutral",
                    "position_change": "no",
                    "new_position": self.state.current_position.value,
                    "reasoning": "Error parsing response, maintaining current position",
                    "theoretical_basis": "error_fallback"
                }
            }
    
    def _get_current_time(self) -> int:
        """获取当前时间步"""
        return len(self.state.negotiation_history)
    
    async def get_initial_position(self) -> Dict[str, Any]:
        """获取初始立场"""
        return await self._analyze_initial_position({})
    
    async def analyze_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """分析提案"""
        return await self._analyze_initial_position({"proposal": proposal})
    
    async def cast_vote(self) -> Dict[str, Any]:
        """投票"""
        return await self._make_voting_decision({})
    
    async def respond_to_countermeasures(self, countermeasures: Dict[str, Any], abstract_event: Dict[str, Any]) -> Dict[str, Any]:
        """响应反制措施"""
        return await self._respond_to_event(countermeasures)
    
    async def reassess_position(self, countermeasures: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """重新评估立场"""
        context = {"countermeasures": countermeasures} if countermeasures else {}
        return await self._analyze_initial_position(context)
    
    async def _act(self) -> Message:
        """主要行动方法"""
        # 获取最新消息
        msg = self.rc.memory.get()[-1] if self.rc.memory.get() else None
        
        if not msg:
            return Message(content="No message received", role=self.name, send_to=self.name)
        
        # 根据消息类型执行相应动作
        if hasattr(msg, 'action_type'):
            action_type = msg.action_type
            content = msg.content
            
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    content = {"data": content}
            
            if action_type == "initial_analysis":
                result = await self._analyze_initial_position(content)
            elif action_type == "negotiation":
                result = await self._negotiate(content)
            elif action_type == "voting_decision":
                result = await self._make_voting_decision(content)
            elif action_type == "event_response":
                result = await self._respond_to_event(content)
            else:
                result = {"error": f"Unknown action type: {action_type}"}
        else:
            # 默认进行初始分析
            result = await self._analyze_initial_position({})
        
        # 序列化结果以处理FeatureLevel枚举
        try:
            serialized_result = self.theory_encoder.serialize_for_json(result)
        except Exception as e:
            logger.warning(f"序列化结果失败: {e}")
            serialized_result = result
        
        return Message(
            content=json.dumps(serialized_result, ensure_ascii=False),
            role=self.name,
            send_to=self.name
        )


# 定义动作类
class InitialAnalysisAction(Action):
    """初始分析动作"""
    
    def __init__(self, country_role: "CountryRole"):
        super().__init__()
        self.country_role = country_role
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self.country_role._analyze_initial_position(context)


class NegotiationAction(Action):
    """谈判动作"""
    
    def __init__(self, country_role: "CountryRole"):
        super().__init__()
        self.country_role = country_role
    
    async def run(self, negotiation_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.country_role._negotiate(negotiation_data)


class VotingDecisionAction(Action):
    """投票决策动作"""
    
    def __init__(self, country_role: "CountryRole"):
        super().__init__()
        self.country_role = country_role
    
    async def run(self, voting_context: Dict[str, Any]) -> Dict[str, Any]:
        return await self.country_role._make_voting_decision(voting_context)


class EventResponseAction(Action):
    """事件响应动作"""
    
    def __init__(self, country_role: "CountryRole"):
        super().__init__()
        self.country_role = country_role
    
    async def run(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.country_role._respond_to_event(event_data)
