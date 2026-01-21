"""
中国商务部角色模块

定义中国商务部的智能体角色，负责反制策略制定。
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from metagpt.roles.role import Role
from metagpt.actions.action import Action
from metagpt.schema import Message

from theory.theory_encoder import TheoryEncoder, TheoryContext, TheoryType, TheoryWeight


class CountermeasureType(Enum):
    """反制措施类型枚举"""
    TARIFF_RETALIATION = "tariff_retaliation"  # 关税报复
    REGULATORY_BARRIERS = "regulatory_barriers"  # 监管壁垒
    DIPLOMATIC_PRESSURE = "diplomatic_pressure"  # 外交压力
    ECONOMIC_COERCION = "economic_coercion"  # 经济胁迫
    LEGAL_CHALLENGE = "legal_challenge"  # 法律挑战
    NEGOTIATION_OFFENSIVE = "negotiation_offensive"  # 谈判攻势


@dataclass
class Countermeasure:
    """反制措施"""
    measure_id: str
    measure_type: CountermeasureType
    target_sectors: List[str]  # 目标行业
    severity_level: float  # 严重程度 (0-1)
    economic_cost: Dict[str, float]  # 经济成本
    political_cost: Dict[str, float]  # 政治成本
    implementation_timeline: str  # 实施时间表
    expected_effectiveness: float  # 预期有效性 (0-1)
    escalation_risk: float  # 升级风险 (0-1)


@dataclass
class ChinaCommerceState:
    """中国商务部状态"""
    current_countermeasures: List[Countermeasure]
    eu_tariff_assessment: Dict[str, Any]
    strategic_priorities: Dict[str, float]  # 战略优先级
    escalation_history: List[Dict[str, Any]]
    negotiation_position: str
    domestic_considerations: Dict[str, Any]


class ChinaCommerceRole(Role):
    """中国商务部角色智能体"""
    
    def __init__(
        self,
        name: str = "China Ministry of Commerce",
        profile: str = "Chinese Government Trade Authority",
        goal: str = "Protect China's economic interests and respond to EU trade measures",
        constraints: str = "Must consider international law and WTO obligations"
    ):
        super().__init__(name=name, profile=profile, goal=goal, constraints=constraints)
        
        self.theory_encoder = TheoryEncoder()
        
        # 初始化中国商务部状态
        self.state = ChinaCommerceState(
            current_countermeasures=[],
            eu_tariff_assessment={},
            strategic_priorities={
                "economic_growth": 0.3,
                "technological_sovereignty": 0.25,
                "political_stability": 0.2,
                "international_relations": 0.15,
                "domestic_industry_protection": 0.1
            },
            escalation_history=[],
            negotiation_position="firm_but_flexible",
            domestic_considerations={
                "automotive_industry_health": "moderate",
                "employment_concerns": "high",
                "technological_development": "critical",
                "export_dependency": "significant"
            }
        )
        
        # 设置监听的动作
        self._watch([CountermeasureDevelopmentAction, EUTariffAssessmentAction, NegotiationStrategyAction])
    
    async def _develop_countermeasures(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """制定反制措施"""
        # 评估欧盟关税措施
        tariff_assessment = await self._assess_eu_tariffs(eu_tariff_data)
        
        # 分析影响
        impact_analysis = await self._analyze_tariff_impact(eu_tariff_data)
        
        # 生成反制选项
        countermeasure_options = await self._generate_countermeasure_options(tariff_assessment, impact_analysis)
        
        # 评估和选择最佳措施
        selected_countermeasures = await self._select_optimal_countermeasures(countermeasure_options)
        
        # 更新状态
        self.state.eu_tariff_assessment = tariff_assessment
        self.state.current_countermeasures.extend(selected_countermeasures)
        
        return {
            "commerce_ministry_id": self.name,
            "tariff_assessment": tariff_assessment,
            "impact_analysis": impact_analysis,
            "countermeasure_options": [cm.__dict__ for cm in countermeasure_options],
            "selected_countermeasures": [cm.__dict__ for cm in selected_countermeasures],
            "implementation_strategy": await self._create_implementation_strategy(selected_countermeasures),
            "escalation_risk_assessment": await self._assess_escalation_risks(selected_countermeasures)
        }
    
    async def _assess_eu_tariffs(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估欧盟关税措施"""
        tariff_rates = eu_tariff_data.get("tariff_rates", {})
        justification = eu_tariff_data.get("justification", [])
        
        # 创建理论上下文进行分析
        theory_context = TheoryContext(
            country_id="China",
            country_features=None,  # 中国商务部不代表单一国家特征
            current_event=eu_tariff_data,
            historical_interactions=self.state.escalation_history,
            time_step=len(self.state.escalation_history)
        )
        
        # 评估关税的严重性和合法性
        assessment = {
            "tariff_severity": await self._assess_tariff_severity(tariff_rates),
            "legal_compliance": await self._assess_wto_compliance(eu_tariff_data),
            "economic_impact": await self._assess_economic_impact(tariff_rates),
            "political_motivation": await self._assess_political_motivation(justification),
            "strategic_implications": await self._assess_strategic_implications(eu_tariff_data)
        }
        
        return assessment
    
    async def _assess_tariff_severity(self, tariff_rates: Dict[str, float]) -> Dict[str, Any]:
        """评估关税严重性"""
        if not tariff_rates:
            return {"level": "low", "average_rate": 0, "max_rate": 0}
        
        rates = list(tariff_rates.values())
        avg_rate = sum(rates) / len(rates)
        max_rate = max(rates)
        
        if avg_rate < 0.1:
            severity = "low"
        elif avg_rate < 0.25:
            severity = "moderate"
        else:
            severity = "high"
        
        return {
            "level": severity,
            "average_rate": avg_rate,
            "max_rate": max_rate,
            "sector_impact": {sector: rate for sector, rate in tariff_rates.items()},
            "overall_assessment": f"{severity} tariff levels with average rate of {avg_rate:.2%}"
        }
    
    async def _assess_wto_compliance(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估WTO合规性"""
        prompt = f"""
        As China's Ministry of Commerce, assess the WTO compliance of the EU's proposed automotive tariffs.
        
        EU tariff data: {eu_tariff_data}
        
        Consider:
        1. Most-favored-nation (MFN) principle
        2. National treatment
        3. Quantitative restrictions
        4. Technical barriers to trade
        5. Safeguard measures
        
        Provide WTO compliance assessment in JSON format:
        {{
            "compliance_level": "full/partial/questionable",
            "potential_violations": ["violation1", "violation2"],
            "legal_basis_for_challenge": ["basis1", "basis2"],
            "success_probability": 0.7,
            "recommended_legal_action": ""
        }}
        """
        
        response = await self._aask(prompt)
        
        try:
            return json.loads(response)
        except:
            return {
                "compliance_level": "questionable",
                "potential_violations": ["Discriminatory treatment", "Excessive tariff rates"],
                "legal_basis_for_challenge": ["GATT Article I", "GATT Article III"],
                "success_probability": 0.6,
                "recommended_legal_action": "WTO dispute settlement"
            }
    
    async def _assess_economic_impact(self, tariff_rates: Dict[str, float]) -> Dict[str, Any]:
        """评估经济影响"""
        # 简化的经济影响评估
        total_export_value = 100  # 假设值（十亿美元）
        weighted_tariff = sum(rate * 0.25 for rate in tariff_rates.values())  # 简化权重
        
        annual_cost = total_export_value * weighted_tariff
        employment_impact = annual_cost * 50  # 每十亿美元影响5万个工作岗位
        
        return {
            "annual_cost_billion_usd": annual_cost,
            "employment_impact": int(employment_impact),
            "sector_specific_impact": {
                sector: rate * total_export_value * 0.25
                for sector, rate in tariff_rates.items()
            },
            "long_term_effects": [
                "Market share erosion",
                "Supply chain disruption",
                "Technology transfer slowdown"
            ]
        }
    
    async def _assess_political_motivation(self, justification: List[str]) -> Dict[str, Any]:
        """评估政治动机"""
        if not justification:
            return {"primary_motivation": "unknown", "secondary_motivations": []}
        
        # 分析理由中的政治动机
        economic_protection = any("protect" in reason.lower() or "industry" in reason.lower() for reason in justification)
        political_pressure = any("cohesion" in reason.lower() or "unity" in reason.lower() for reason in justification)
        strategic_competition = any("competition" in reason.lower() or "strategic" in reason.lower() for reason in justification)
        
        motivations = []
        if economic_protection:
            motivations.append("economic_protectionism")
        if political_pressure:
            motivations.append("political_cohesion")
        if strategic_competition:
            motivations.append("strategic_competition")
        
        return {
            "primary_motivation": motivations[0] if motivations else "unknown",
            "secondary_motivations": motivations[1:],
            "assessment": "Mixed economic and political motivations detected"
        }
    
    async def _assess_strategic_implications(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估战略影响"""
        return {
            "eu_china_relations": "deterioration_likely",
            "global_trade_system": "fragmentation_risk",
            "alliance_dynamics": "us_alignment_pressure",
            "technological_competition": "intensification_expected",
            "long_term_strategic_position": "challenged_but_resilient"
        }
    
    async def _analyze_tariff_impact(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析关税影响"""
        return {
            "immediate_impact": await self._analyze_immediate_impact(eu_tariff_data),
            "medium_term_impact": await self._analyze_medium_term_impact(eu_tariff_data),
            "long_term_impact": await self._analyze_long_term_impact(eu_tariff_data),
            "sectoral_analysis": await self._analyze_sectoral_impact(eu_tariff_data),
            "regional_variations": await self._analyze_regional_variations(eu_tariff_data)
        }
    
    async def _analyze_immediate_impact(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析直接影响"""
        return {
            "export_decline": "5-10%",
            "price_increase": "2-5%",
            "market_share_loss": "3-7%",
            "company_profit_impact": "10-15%",
            "timeline": "3-6 months"
        }
    
    async def _analyze_medium_term_impact(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析中期影响"""
        return {
            "supply_chain_adjustment": "significant",
            "production_relocation": "partial",
            "technology_development_acceleration": "moderate",
            "market_diversification": "increased",
            "timeline": "1-2 years"
        }
    
    async def _analyze_long_term_impact(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析长期影响"""
        return {
            "industry_transformation": "accelerated",
            "innovation_investment": "increased",
            "global_market_repositioning": "significant",
            "strategic_autonomy": "enhanced",
            "timeline": "3-5 years"
        }
    
    async def _analyze_sectoral_impact(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析行业影响"""
        return {
            "electric_vehicles": "high_impact",
            "hybrid_vehicles": "moderate_impact",
            "traditional_vehicles": "low_impact",
            "auto_parts": "moderate_impact",
            "battery_technology": "high_impact"
        }
    
    async def _analyze_regional_variations(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析地区差异"""
        return {
            "western_europe": "higher_impact",
            "eastern_europe": "moderate_impact",
            "northern_europe": "mixed_impact",
            "southern_europe": "lower_impact"
        }
    
    async def _generate_countermeasure_options(self, tariff_assessment: Dict[str, Any], impact_analysis: Dict[str, Any]) -> List[Countermeasure]:
        """生成反制措施选项"""
        options = []
        
        # 关税报复措施
        tariff_retaliation = Countermeasure(
            measure_id="CM_TARIFF_001",
            measure_type=CountermeasureType.TARIFF_RETALIATION,
            target_sectors=["automotive", "agriculture", "luxury_goods"],
            severity_level=0.7,
            economic_cost={"china": 0.3, "eu": 0.8},
            political_cost={"china": 0.2, "eu": 0.6},
            implementation_timeline="3-6 months",
            expected_effectiveness=0.8,
            escalation_risk=0.7
        )
        options.append(tariff_retaliation)
        
        # 监管壁垒措施
        regulatory_barriers = Countermeasure(
            measure_id="CM_REG_001",
            measure_type=CountermeasureType.REGULATORY_BARRIERS,
            target_sectors=["automotive", "technology", "pharmaceuticals"],
            severity_level=0.5,
            economic_cost={"china": 0.1, "eu": 0.6},
            political_cost={"china": 0.3, "eu": 0.4},
            implementation_timeline="6-12 months",
            expected_effectiveness=0.6,
            escalation_risk=0.4
        )
        options.append(regulatory_barriers)
        
        # 外交压力措施
        diplomatic_pressure = Countermeasure(
            measure_id="CM_DIPLO_001",
            measure_type=CountermeasureType.DIPLOMATIC_PRESSURE,
            target_sectors=["all_sectors"],
            severity_level=0.3,
            economic_cost={"china": 0.05, "eu": 0.2},
            political_cost={"china": 0.1, "eu": 0.3},
            implementation_timeline="immediate",
            expected_effectiveness=0.4,
            escalation_risk=0.2
        )
        options.append(diplomatic_pressure)
        
        # 法律挑战措施
        legal_challenge = Countermeasure(
            measure_id="CM_LEGAL_001",
            measure_type=CountermeasureType.LEGAL_CHALLENGE,
            target_sectors=["automotive"],
            severity_level=0.4,
            economic_cost={"china": 0.1, "eu": 0.3},
            political_cost={"china": 0.1, "eu": 0.2},
            implementation_timeline="12-24 months",
            expected_effectiveness=0.5,
            escalation_risk=0.1
        )
        options.append(legal_challenge)
        
        return options
    
    async def _select_optimal_countermeasures(self, options: List[Countermeasure]) -> List[Countermeasure]:
        """选择最优反制措施"""
        # 基于战略优先级和成本效益分析选择措施
        selected = []
        
        # 计算每个措施的综合得分
        for option in options:
            effectiveness_score = option.expected_effectiveness
            cost_score = 1 - (option.economic_cost["china"] + option.political_cost["china"]) / 2
            risk_score = 1 - option.escalation_risk
            
            # 综合得分
            composite_score = (
                effectiveness_score * 0.4 +
                cost_score * 0.3 +
                risk_score * 0.3
            )
            
            option.composite_score = composite_score
        
        # 按得分排序并选择前3个
        options.sort(key=lambda x: x.composite_score, reverse=True)
        selected = options[:3]
        
        return selected
    
    async def _create_implementation_strategy(self, selected_measures: List[Countermeasure]) -> Dict[str, Any]:
        """创建实施策略"""
        return {
            "phased_approach": {
                "phase_1": "diplomatic_protests_and_wto_preparation",
                "phase_2": "targeted_countermeasures",
                "phase_3": "escalation_if_needed"
            },
            "coordination_mechanisms": [
                "inter_agency_coordination",
                "industry_consultation",
                "international_alliance_building"
            ],
            "communication_strategy": {
                "domestic_narrative": "defending_national_interests",
                "international_narrative": "promoting_free_trade",
                "eu_narrative": "seeking_fair_resolution"
            },
            "contingency_planning": [
                "escalation_de_escalation",
                "economic_stabilization",
                "alternative_markets"
            ]
        }
    
    async def _assess_escalation_risks(self, selected_measures: List[Countermeasure]) -> Dict[str, Any]:
        """评估升级风险"""
        max_risk = max(measure.escalation_risk for measure in selected_measures)
        avg_risk = sum(measure.escalation_risk for measure in selected_measures) / len(selected_measures)
        
        return {
            "overall_risk_level": "high" if max_risk > 0.7 else "medium" if avg_risk > 0.5 else "low",
            "maximum_risk": max_risk,
            "average_risk": avg_risk,
            "risk_factors": [
                "EU counter-retaliation",
                "Alliance formation against China",
                "Global trade war escalation",
                "Domestic political pressure"
            ],
            "mitigation_strategies": [
                "Gradual escalation",
                "Multilateral engagement",
                "Economic impact mitigation",
                "Diplomatic backchannels"
            ]
        }
    
    async def _assess_eu_tariffs(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估欧盟关税（独立方法）"""
        # 这是一个独立的评估方法，用于响应不同的触发条件
        return await self._assess_eu_tariffs(assessment_data)
    
    async def _develop_negotiation_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """制定谈判策略"""
        eu_position = context.get("eu_position", {})
        current_tensions = context.get("current_tensions", "medium")
        
        strategy = {
            "opening_position": await self._determine_opening_position(eu_position),
            "concession_strategy": await self._plan_concessions(eu_position),
            "red_lines": await self._define_red_lines(),
            "timeline": await self._set_negotiation_timeline(current_tensions),
            "communication_approach": await self._plan_communication_approach()
        }
        
        return {
            "commerce_ministry_id": self.name,
            "negotiation_strategy": strategy,
            "success_probability": await self._assess_negotiation_success(strategy, eu_position),
            "contingency_plans": await self._create_negotiation_contingencies()
        }
    
    async def _determine_opening_position(self, eu_position: Dict[str, Any]) -> Dict[str, Any]:
        """确定开盘立场"""
        return {
            "primary_demand": "complete_tariff_removal",
            "secondary_demands": ["compensation", "future_cooperation"],
            "negotiating_posture": "firm_but_constructive",
            "initial_offers": ["mutual_tariff_reduction", "sector_specific_exemptions"]
        }
    
    async def _plan_concessions(self, eu_position: Dict[str, Any]) -> Dict[str, Any]:
        """规划让步策略"""
        return {
            "concession_sequence": [
                "minor_regulatory_adjustments",
                "selective_tariff_reductions",
                "cooperation_mechanisms",
                "long_term_market_access_commitments"
            ],
            "concession_conditions": [
                "reciprocity_required",
                "implementation_verification",
                "dispute_resolution_mechanism"
            ],
            "red_line_concessions": ["sovereignty", "core_industries"]
        }
    
    async def _define_red_lines(self) -> List[str]:
        """定义红线"""
        return [
            "no_acceptance_of_discriminatory_treatment",
            "protection_of_strategic_industries",
            "maintenance_of_technological_development_rights",
            "preservation_of_market_access_principles"
        ]
    
    async def _set_negotiation_timeline(self, current_tensions: str) -> Dict[str, Any]:
        """设定谈判时间表"""
        if current_tensions == "high":
            timeline = "accelerated_2_weeks"
        elif current_tensions == "medium":
            timeline = "standard_6_weeks"
        else:
            timeline = "extended_12_weeks"
        
        return {
            "timeline_type": timeline,
            "milestone_dates": ["initial_session", "mid_term_review", "final_deadline"],
            "extension_possibility": "conditional"
        }
    
    async def _plan_communication_approach(self) -> Dict[str, Any]:
        """规划沟通方式"""
        return {
            "public_communication": "principled_stance_with_flexibility",
            "private_diplomacy": "constructive_engagement",
            "media_strategy": "balanced_narrative",
            "alliance_coordination": "multilateral_support_building"
        }
    
    async def _assess_negotiation_success(self, strategy: Dict[str, Any], eu_position: Dict[str, Any]) -> float:
        """评估谈判成功概率"""
        # 简化的成功概率评估
        base_probability = 0.5
        
        # 根据欧盟立场调整
        eu_flexibility = eu_position.get("flexibility", 0.5)
        adjusted_probability = base_probability + (eu_flexibility - 0.5) * 0.4
        
        return max(0.1, min(0.9, adjusted_probability))
    
    async def _create_negotiation_contingencies(self) -> List[Dict[str, Any]]:
        """创建谈判应急计划"""
        return [
            {
                "scenario": "negotiation_breakdown",
                "response": "immediate_countermeasure_implementation"
            },
            {
                "scenario": "partial_agreement",
                "response": "phased_implementation_with_monitoring"
            },
            {
                "scenario": "eu_intransigence",
                "response": "escalation_with_multilateral_support"
            }
        ]
    
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
            
            if action_type == "countermeasure_development":
                result = await self._develop_countermeasures(content)
            elif action_type == "eu_tariff_assessment":
                result = await self._assess_eu_tariffs(content)
            elif action_type == "negotiation_strategy":
                result = await self._develop_negotiation_strategy(content)
            else:
                result = {"error": f"Unknown action type: {action_type}"}
        else:
            # 默认制定反制措施
            result = await self._develop_countermeasures({})
        
        return Message(
            content=json.dumps(result, ensure_ascii=False),
            role=self.name,
            send_to=self.name
        )


# 定义动作类
class CountermeasureDevelopmentAction(Action):
    """反制措施开发动作"""
    
    async def run(self, eu_tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "countermeasure_development", "data": eu_tariff_data}


class EUTariffAssessmentAction(Action):
    """欧盟关税评估动作"""
    
    async def run(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "eu_tariff_assessment", "data": assessment_data}


class NegotiationStrategyAction(Action):
    """谈判策略动作"""
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "negotiation_strategy", "data": context}
