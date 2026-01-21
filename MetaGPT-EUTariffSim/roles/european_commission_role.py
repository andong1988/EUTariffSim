"""
欧盟委员会角色模块

定义欧盟委员会的智能体角色，负责议程设置和提案生成。
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from metagpt.roles.role import Role
from metagpt.actions.action import Action
from metagpt.schema import Message

from data_processing.anonymization_processor import AnonymizedCountry
from theory.theory_encoder import TheoryEncoder, TheoryContext, TheoryType, TheoryWeight


class ProposalType(Enum):
    """提案类型枚举"""
    TARIFF_PROPOSAL = "tariff_proposal"  # 关税提案
    NEGOTIATION_INITIATIVE = "negotiation_initiative"  # 谈判倡议
    COMPROMISE_PROPOSAL = "compromise_proposal"  # 妥协提案
    EMERGENCY_MEASURE = "emergency_measure"  # 紧急措施


@dataclass
class TariffProposal:
    """关税提案"""
    proposal_id: str
    proposal_type: ProposalType
    tariff_rates: Dict[str, float]  # 产品类别 -> 关税率
    justification: List[str]  # 理由
    expected_impact: Dict[str, Any]  # 预期影响
    implementation_timeline: str  # 实施时间表
    conditions: List[str]  # 条件


@dataclass
class CommissionState:
    """委员会状态"""
    current_proposals: List[TariffProposal]
    negotiation_status: str
    member_state_positions: Dict[str, str]  # 国家ID -> 立场
    voting_history: List[Dict[str, Any]]
    policy_priorities: Dict[str, float]  # 政策优先级


class EuropeanCommissionRole(Role):
    """欧盟委员会角色智能体"""
    
    def __init__(
        self,
        name: str = "European Commission",
        profile: str = "EU Executive Body",
        goal: str = "Promote EU integration and collective interests",
        constraints: str = "Must follow EU treaties and procedures"
    ):
        super().__init__(name=name, profile=profile, goal=goal, constraints=constraints)
        
        self.theory_encoder = TheoryEncoder()
        
        # 初始化委员会状态
        self.state = CommissionState(
            current_proposals=[],
            negotiation_status="preparing",
            member_state_positions={},
            voting_history=[],
            policy_priorities={
                "economic_integration": 0.3,
                "political_cohesion": 0.25,
                "external_relations": 0.2,
                "industrial_competitiveness": 0.15,
                "consumer_protection": 0.1
            }
        )
        
        # 设置监听的动作
        self._watch([ProposalGenerationAction, AgendaSettingAction, NegotiationMediationAction])
    
    async def _generate_initial_proposal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成初始提案"""
        # 分析当前形势
        situation_analysis = await self._analyze_situation(context)
        
        # 生成关税提案
        proposal = await self._create_tariff_proposal(situation_analysis)
        
        # 评估提案可行性
        feasibility_assessment = await self._assess_proposal_feasibility(proposal, context)
        
        # 记录提案
        self.state.current_proposals.append(proposal)
        
        return {
            "commission_id": self.name,
            "proposal": proposal.__dict__,
            "situation_analysis": situation_analysis,
            "feasibility_assessment": feasibility_assessment,
            "next_steps": ["Distribute proposal to member states", "Schedule negotiation rounds"]
        }
    
    async def _analyze_situation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析当前形势"""
        member_states = context.get("member_states", [])
        external_factors = context.get("external_factors", {})
        
        # 创建理论上下文进行分析
        theory_context = TheoryContext(
            country_id="EU_Commission",
            country_features=None,  # 委员会不代表单一国家
            current_event=context.get("current_event", {}),
            historical_interactions=[],
            time_step=0
        )
        
        # 基于政策优先级分析形势
        analysis = {
            "economic_context": await self._analyze_economic_context(member_states),
            "political_context": await self._analyze_political_context(member_states),
            "external_pressure": await self._analyze_external_pressure(external_factors),
            "policy_alignment": await self._assess_policy_alignment(context),
            "recommended_approach": await self._determine_approach(context)
        }
        
        return analysis
    
    async def _analyze_economic_context(self, member_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析经济背景"""
        if not member_states:
            return {"assessment": "insufficient_data"}
        
        # 模拟经济分析
        prompt = f"""
        As the European Commission, analyze the economic context of {len(member_states)} EU member states regarding automotive trade with China.
        
        Consider:
        1. Overall EU automotive industry health
        2. Trade balance concerns
        3. Market access issues
        4. Competitive pressures
        5. Supply chain vulnerabilities
        
        Provide an economic assessment in JSON format:
        {{
            "overall_health": "strong/moderate/weak",
            "trade_concerns": ["concern1", "concern2"],
            "market_access_issues": ["issue1", "issue2"],
            "recommended_tariff_range": {{
                "min": 0.0,
                "max": 0.5,
                "optimal": 0.25
            }},
            "economic_justification": ""
        }}
        """
        
        response = await self._aask(prompt)
        
        try:
            return json.loads(response)
        except:
            return {
                "overall_health": "moderate",
                "trade_concerns": ["Trade imbalance", "Market access barriers"],
                "market_access_issues": ["Regulatory differences", "Subsidy competition"],
                "recommended_tariff_range": {"min": 0.1, "max": 0.3, "optimal": 0.2},
                "economic_justification": "Standard economic analysis"
            }
    
    async def _analyze_political_context(self, member_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析政治背景"""
        prompt = f"""
        As the European Commission, analyze the political context among EU member states regarding China automotive tariffs.
        
        Consider:
        1. Political divisions within the EU
        2. National sovereignty concerns
        3. Integration pressures
        4. External political influences
        5. Public opinion trends
        
        Provide a political assessment in JSON format:
        {{
            "cohesion_level": "high/medium/low",
            "main_divisions": ["division1", "division2"],
            "sovereignty_concerns": ["concern1", "concern2"],
            "political_feasibility": "high/medium/low",
            "recommended_strategy": ""
        }}
        """
        
        response = await self._aask(prompt)
        
        try:
            return json.loads(response)
        except:
            return {
                "cohesion_level": "medium",
                "main_divisions": ["North-South divide", "East-West differences"],
                "sovereignty_concerns": ["National autonomy", "Economic policy control"],
                "political_feasibility": "medium",
                "recommended_strategy": "Balanced approach with flexibility"
            }
    
    async def _analyze_external_pressure(self, external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """分析外部压力"""
        china_factors = external_factors.get("china", {})
        global_factors = external_factors.get("global", {})
        
        analysis = {
            "china_response_potential": await self._assess_china_response(china_factors),
            "global_trade_environment": await self._assess_global_environment(global_factors),
            "alliance_considerations": await self._assess_alliance_factors(external_factors),
            "risk_assessment": await self._assess_external_risks(external_factors)
        }
        
        return analysis
    
    async def _assess_china_response(self, china_factors: Dict[str, Any]) -> Dict[str, Any]:
        """评估中国可能的反应"""
        prompt = f"""
        As the European Commission, assess how China might respond to EU automotive tariffs.
        
        China factors: {china_factors}
        
        Consider:
        1. Retaliatory measures
        2. Economic countermeasures
        3. Diplomatic responses
        4. WTO challenges
        5. Long-term strategic implications
        
        Provide assessment in JSON format:
        {{
            "retaliation_likelihood": "high/medium/low",
            "potential_measures": ["measure1", "measure2"],
            "economic_impact": "severe/moderate/mild",
            "diplomatic_consequences": "severe/moderate/mild",
            "recommended_mitigation": ""
        }}
        """
        
        response = await self._aask(prompt)
        
        try:
            return json.loads(response)
        except:
            return {
                "retaliation_likelihood": "medium",
                "potential_measures": ["Counter-tariffs", "Regulatory barriers"],
                "economic_impact": "moderate",
                "diplomatic_consequences": "moderate",
                "recommended_mitigation": "Gradual implementation with dialogue"
            }
    
    async def _assess_global_environment(self, global_factors: Dict[str, Any]) -> Dict[str, Any]:
        """评估全球环境"""
        return {
            "trade_tensions": "increasing",
            "protectionism_trend": "rising",
            "multilateral_support": "declining",
            "regional_cooperation": "strengthening",
            "overall_stability": "moderate"
        }
    
    async def _assess_alliance_factors(self, external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """评估联盟因素"""
        return {
            "us_alignment": "partial",
            "other_partners": ["Japan", "South Korea"],
            "multilateral_support": "limited",
            "strategic_considerations": ["Supply chain security", "Technology competition"]
        }
    
    async def _assess_external_risks(self, external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """评估外部风险"""
        return {
            "escalation_risk": "medium",
            "economic_spillover": "moderate",
            "political_isolation": "low",
            "legal_challenges": "high",
            "overall_risk_level": "moderate"
        }
    
    async def _assess_policy_alignment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估政策一致性"""
        alignment_scores = {}
        
        for policy, priority in self.state.policy_priorities.items():
            # 简化的政策一致性评估
            alignment_scores[policy] = min(1.0, priority + 0.2)  # 假设基本一致
        
        return {
            "alignment_scores": alignment_scores,
            "overall_alignment": sum(alignment_scores.values()) / len(alignment_scores),
            "policy_conflicts": [],
            "recommended_adjustments": []
        }
    
    async def _determine_approach(self, context: Dict[str, Any]) -> str:
        """确定推荐方法"""
        return "balanced_negotiation_with_tariff_threat"
    
    async def _create_tariff_proposal(self, situation_analysis: Dict[str, Any]) -> TariffProposal:
        """创建关税提案"""
        # 基于分析生成提案
        economic_context = situation_analysis.get("economic_context", {})
        political_context = situation_analysis.get("political_context", {})
        
        # 确定关税率
        tariff_range = economic_context.get("recommended_tariff_range", {"min": 0.1, "max": 0.3, "optimal": 0.2})
        optimal_rate = tariff_range.get("optimal", 0.2)
        
        # 创建提案
        proposal = TariffProposal(
            proposal_id=f"EU_TARIFF_{self._get_current_time()}",
            proposal_type=ProposalType.TARIFF_PROPOSAL,
            tariff_rates={
                "electric_vehicles": optimal_rate,
                "hybrid_vehicles": optimal_rate * 0.8,
                "traditional_vehicles": optimal_rate * 0.6,
                "auto_parts": optimal_rate * 0.4
            },
            justification=[
                "Addressing market access barriers",
                "Leveling playing field for EU industry",
                "Protecting strategic automotive sector",
                "Encouraging fair competition"
            ],
            expected_impact={
                "trade_balance_improvement": "moderate",
                "industry_protection": "significant",
                "consumer_price_impact": "mild to moderate",
                "diplomatic_tensions": "moderate"
            },
            implementation_timeline="Phase 1: 6 months, Phase 2: 12 months",
            conditions=[
                "Parallel negotiations with China",
                "Review mechanism after 18 months",
                "Exemptions for developing countries",
                "WTO compliance verification"
            ]
        )
        
        return proposal
    
    async def _assess_proposal_feasibility(self, proposal: TariffProposal, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估提案可行性"""
        member_states = context.get("member_states", [])
        
        # 模拟可行性评估
        support_count = len(member_states) // 2  # 简化假设
        opposition_count = len(member_states) // 4
        abstain_count = len(member_states) - support_count - opposition_count
        
        # 计算合格多数
        total_population = sum(state.get("population", 10) for state in member_states)
        support_population = total_population * 0.65  # 简化假设
        
        qualified_majority = support_population >= total_population * 0.65 and support_count >= len(member_states) * 0.55
        
        return {
            "support_count": support_count,
            "opposition_count": opposition_count,
            "abstain_count": abstain_count,
            "qualified_majority": qualified_majority,
            "feasibility_level": "high" if qualified_majority else "medium",
            "key_concerns": [
                "Industry competitiveness",
                "Consumer price impact",
                "China retaliation risk",
                "WTO compliance"
            ],
            "recommended_modifications": [
                "Gradual implementation",
                "Sector-specific exemptions",
                "Enhanced negotiation efforts"
            ]
        }
    
    async def _set_agenda(self, agenda_data: Dict[str, Any]) -> Dict[str, Any]:
        """设置议程"""
        agenda_items = agenda_data.get("items", [])
        priority_level = agenda_data.get("priority", "medium")
        timeline = agenda_data.get("timeline", "standard")
        
        # 组织议程
        organized_agenda = await self._organize_agenda_items(agenda_items, priority_level)
        
        # 设置时间表
        schedule = await self._create_schedule(organized_agenda, timeline)
        
        return {
            "commission_id": self.name,
            "agenda": organized_agenda,
            "schedule": schedule,
            "priorities": self.state.policy_priorities,
            "next_meeting": schedule[0] if schedule else None
        }
    
    async def _organize_agenda_items(self, items: List[str], priority: str) -> List[Dict[str, Any]]:
        """组织议程项目"""
        organized = []
        
        for i, item in enumerate(items):
            organized.append({
                "item_id": f"agenda_{i+1}",
                "title": item,
                "priority": priority,
                "estimated_duration": "2 hours",
                "required_participants": ["all_member_states"],
                "expected_outcome": "decision"
            })
        
        return organized
    
    async def _create_schedule(self, agenda: List[Dict[str, Any]], timeline: str) -> List[Dict[str, Any]]:
        """创建时间表"""
        schedule = []
        
        for i, item in enumerate(agenda):
            schedule.append({
                "session": i + 1,
                "date": f"Week {i+1}",
                "agenda_item": item["title"],
                "duration": item["estimated_duration"],
                "location": "Brussels"
            })
        
        return schedule
    
    async def _mediate_negotiation(self, negotiation_data: Dict[str, Any]) -> Dict[str, Any]:
        """调解谈判"""
        conflicting_parties = negotiation_data.get("conflicting_parties", [])
        issues = negotiation_data.get("issues", [])
        current_proposals = negotiation_data.get("proposals", [])
        
        # 分析冲突
        conflict_analysis = await self._analyze_conflicts(conflicting_parties, issues)
        
        # 生成调解方案
        mediation_proposal = await self._generate_mediation_proposal(conflict_analysis, current_proposals)
        
        # 评估调解成功概率
        success_probability = await self._assess_mediation_success(mediation_proposal, conflicting_parties)
        
        return {
            "commission_id": self.name,
            "conflict_analysis": conflict_analysis,
            "mediation_proposal": mediation_proposal,
            "success_probability": success_probability,
            "next_steps": ["Present proposal to parties", "Facilitate discussion", "Monitor implementation"]
        }
    
    async def _analyze_conflicts(self, parties: List[str], issues: List[str]) -> Dict[str, Any]:
        """分析冲突"""
        return {
            "conflict_type": "policy_divergence",
            "core_issues": issues,
            "party_positions": {party: "moderate" for party in parties},
            "common_ground": ["EU_integration", "economic_stability"],
            "resolution_difficulty": "medium"
        }
    
    async def _generate_mediation_proposal(self, conflict_analysis: Dict[str, Any], existing_proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成调解提案"""
        return {
            "proposal_type": "compromise",
            "key_elements": [
                "Phased implementation",
                "Sector-specific adjustments",
                "Review mechanisms",
                "Compensation measures"
            ],
            "timeline": "12-18 months",
            "monitoring": "quarterly_reviews",
            "flexibility_clauses": ["economic_impact_adjustment", "reciprocity_conditions"]
        }
    
    async def _assess_mediation_success(self, proposal: Dict[str, Any], parties: List[str]) -> Dict[str, Any]:
        """评估调解成功概率"""
        return {
            "overall_probability": 0.65,
            "party_acceptance": {party: 0.6 + (i * 0.05) for i, party in enumerate(parties)},
            "key_factors": [
                "Economic benefits",
                "Political feasibility",
                "Implementation timeline",
                "Flexibility level"
            ],
            "potential_obstacles": [
                "National sovereignty concerns",
                "Domestic political pressure",
                "External influences"
            ]
        }
    
    def _get_current_time(self) -> int:
        """获取当前时间步"""
        return len(self.state.voting_history)
    
    async def _act(self) -> Message:
        """主要行动方法"""
        # 获取最新消息
        msg = self._rc.msgs[-1] if self._rc.msgs else None
        
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
            
            if action_type == "proposal_generation":
                result = await self._generate_initial_proposal(content)
            elif action_type == "agenda_setting":
                result = await self._set_agenda(content)
            elif action_type == "negotiation_mediation":
                result = await self._mediate_negotiation(content)
            else:
                result = {"error": f"Unknown action type: {action_type}"}
        else:
            # 默认生成提案
            result = await self._generate_initial_proposal({})
        
        return Message(
            content=json.dumps(result, ensure_ascii=False),
            role=self.name,
            send_to=self.name
        )


# 定义动作类
class ProposalGenerationAction(Action):
    """提案生成动作"""
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "proposal_generation", "context": context}


class AgendaSettingAction(Action):
    """议程设置动作"""
    
    async def run(self, agenda_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "agenda_setting", "data": agenda_data}


class NegotiationMediationAction(Action):
    """谈判调解动作"""
    
    async def run(self, negotiation_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "negotiation_mediation", "data": negotiation_data}
