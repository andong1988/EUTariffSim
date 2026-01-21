"""
多边谈判动作
实现所有国家间的多边谈判机制
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from metagpt.actions.action import Action
from metagpt.schema import Message

class NegotiationRound(Enum):
    """谈判轮次"""
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"

@dataclass
class NegotiationProposal:
    """谈判提案"""
    from_country: str
    to_country: str
    proposal_type: str  # "tariff_reduction", "compensation", "alliance", "neutral"
    content: str
    expected_response: str
    urgency: float  # 0-1, 提案紧急程度

@dataclass
class NegotiationResponse:
    """谈判响应"""
    from_country: str
    to_country: str
    response_type: str  # "accept", "reject", "counter_proposal", "delay"
    content: str
    new_stance: Optional[str] = None

class MultilateralNegotiationAction(Action):
    """多边谈判动作"""
    
    def __init__(self, name: str = "MultilateralNegotiationAction", 
                 negotiation_round: NegotiationRound = NegotiationRound.FIRST):
        super().__init__(name)
        self.negotiation_round = negotiation_round
        self.logger = logging.getLogger(__name__)
        
        # 谈判策略配置
        self.negotiation_strategies = {
            "Country_A": {  # 德国 - 经济导向
                "primary_focus": "economic_impact",
                "alliance_tendency": 0.3,
                "compensation_acceptance": 0.7,
                "aggression_level": 0.2
            },
            "Country_B": {  # 法国 - 政治导向
                "primary_focus": "political_considerations",
                "alliance_tendency": 0.8,
                "compensation_acceptance": 0.4,
                "aggression_level": 0.6
            },
            "Country_C": {  # 意大利 - 平衡导向
                "primary_focus": "balanced_approach",
                "alliance_tendency": 0.5,
                "compensation_acceptance": 0.6,
                "aggression_level": 0.4
            }
        }
    
    async def run(self, context: Dict) -> Dict:
        """执行多边谈判"""
        self.logger.info(f"开始第{self.negotiation_round.value}轮多边谈判")
        
        countries = context.get("countries", [])
        current_stances = context.get("current_stances", {})
        country_features = context.get("country_features", {})
        
        # 生成所有国家间的谈判提案
        proposals = await self._generate_proposals(
            countries, current_stances, country_features
        )
        
        # 处理谈判响应
        responses = await self._process_responses(
            proposals, current_stances, country_features
        )
        
        # 更新立场
        updated_stances = await self._update_stances(
            current_stances, responses, country_features
        )
        
        # 分析联盟形成
        alliances = await self._analyze_alliances(responses, updated_stances)
        
        # 记录谈判结果
        negotiation_result = {
            "round": self.negotiation_round.value,
            "proposals": [self._serialize_proposal(p) for p in proposals],
            "responses": [self._serialize_response(r) for r in responses],
            "updated_stances": updated_stances,
            "alliances": alliances,
            "negotiation_summary": await self._generate_negotiation_summary(
                proposals, responses, updated_stances, alliances
            )
        }
        
        self.logger.info(f"第{self.negotiation_round.value}轮谈判完成")
        return negotiation_result
    
    async def _generate_proposals(self, countries: List[str], 
                                 current_stances: Dict[str, str],
                                 country_features: Dict) -> List[NegotiationProposal]:
        """生成所有国家间的谈判提案"""
        proposals = []
        
        # 为每对国家生成提案（双向）
        for from_country in countries:
            for to_country in countries:
                if from_country != to_country:
                    # 根据当前立场和特征生成提案
                    proposal = await self._create_specific_proposal(
                        from_country, to_country, current_stances, country_features
                    )
                    if proposal:
                        proposals.append(proposal)
        
        # 根据紧急程度排序
        proposals.sort(key=lambda x: x.urgency, reverse=True)
        
        # 限制提案数量，避免过多
        max_proposals = len(countries) * 2  # 每个国家最多2个提案
        return proposals[:max_proposals]
    
    async def _create_specific_proposal(self, from_country: str, to_country: str,
                                     current_stances: Dict[str, str],
                                     country_features: Dict) -> Optional[NegotiationProposal]:
        """创建特定的谈判提案"""
        from_stance = current_stances.get(from_country, "neutral")
        to_stance = current_stances.get(to_country, "neutral")
        from_strategy = self.negotiation_strategies.get(from_country, {})
        
        # 基于立场差异决定提案类型
        if from_stance == to_stance:
            # 立场相同，寻求联盟
            proposal_type = "alliance"
            content = f"我们两国在关税问题上立场一致，建议形成联盟共同推进{from_stance}立场。"
            expected_response = "accept"
            urgency = 0.6 + from_strategy.get("alliance_tendency", 0.5) * 0.4
            
        elif from_stance == "support" and to_stance == "against":
            # 尝试说服反对国
            if from_strategy.get("compensation_acceptance", 0.5) > 0.5:
                proposal_type = "compensation"
                content = f"理解贵国的关切，我们愿意提供经济补偿以换取支持。"
                expected_response = "counter_proposal"
                urgency = 0.8
            else:
                proposal_type = "tariff_reduction"
                content = f"建议降低关税幅度，寻求折中方案。"
                expected_response = "delay"
                urgency = 0.7
                
        elif from_stance == "against" and to_stance == "support":
            # 尝试说服支持国
            proposal_type = "tariff_reduction"
            content = f"关税措施可能对双方经济造成损害，建议重新考虑。"
            expected_response = "counter_proposal"
            urgency = 0.7
            
        else:
            # 中立立场，寻求信息
            proposal_type = "neutral"
            content = f"希望了解贵国在关税问题上的具体考量。"
            expected_response = "delay"
            urgency = 0.4
        
        return NegotiationProposal(
            from_country=from_country,
            to_country=to_country,
            proposal_type=proposal_type,
            content=content,
            expected_response=expected_response,
            urgency=urgency
        )
    
    async def _process_responses(self, proposals: List[NegotiationProposal],
                               current_stances: Dict[str, str],
                               country_features: Dict) -> List[NegotiationResponse]:
        """处理谈判响应"""
        responses = []
        
        for proposal in proposals:
            response = await self._create_response(
                proposal, current_stances, country_features
            )
            responses.append(response)
        
        return responses
    
    async def _create_response(self, proposal: NegotiationProposal,
                             current_stances: Dict[str, str],
                             country_features: Dict) -> NegotiationResponse:
        """创建谈判响应"""
        to_country = proposal.to_country
        to_stance = current_stances.get(to_country, "neutral")
        to_strategy = self.negotiation_strategies.get(to_country, {})
        
        # 基于提案类型和国家策略决定响应
        response_type = await self._determine_response_type(
            proposal, to_stance, to_strategy
        )
        
        # 生成响应内容
        content = await self._generate_response_content(
            proposal, response_type, to_stance, to_strategy
        )
        
        # 确定是否改变立场
        new_stance = await self._determine_new_stance(
            proposal, response_type, to_stance, to_strategy
        )
        
        return NegotiationResponse(
            from_country=to_country,
            to_country=proposal.from_country,
            response_type=response_type,
            content=content,
            new_stance=new_stance
        )
    
    async def _determine_response_type(self, proposal: NegotiationProposal,
                                     current_stance: str, strategy: Dict) -> str:
        """确定响应类型"""
        proposal_type = proposal.proposal_type
        alliance_tendency = strategy.get("alliance_tendency", 0.5)
        compensation_acceptance = strategy.get("compensation_acceptance", 0.5)
        aggression_level = strategy.get("aggression_level", 0.5)
        
        # 基于提案类型和策略的响应逻辑
        if proposal_type == "alliance":
            if alliance_tendency > 0.6:
                return "accept"
            elif alliance_tendency > 0.3:
                return "delay"
            else:
                return "reject"
                
        elif proposal_type == "compensation":
            if compensation_acceptance > 0.7:
                return "accept"
            elif compensation_acceptance > 0.4:
                return "counter_proposal"
            else:
                return "reject"
                
        elif proposal_type == "tariff_reduction":
            if current_stance == "support":
                if aggression_level > 0.6:
                    return "reject"
                else:
                    return "counter_proposal"
            else:  # against or neutral
                return "accept"
                
        else:  # neutral
            return "delay"
    
    async def _generate_response_content(self, proposal: NegotiationProposal,
                                       response_type: str, current_stance: str,
                                       strategy: Dict) -> str:
        """生成响应内容"""
        templates = {
            "accept": [
                "我们同意贵国的提案，这符合我们的利益。",
                "感谢贵国的建议，我们愿意合作。",
                "这个提案很有建设性，我们表示支持。"
            ],
            "reject": [
                "我们无法接受这个提案，这与我们的核心利益冲突。",
                "抱歉，这个方案不符合我们的立场。",
                "我们坚决反对这个提议。"
            ],
            "counter_proposal": [
                "我们理解贵国的立场，但建议考虑以下替代方案...",
                "部分同意，但我们希望在某些方面进行调整。",
                "我们可以探讨其他可能的解决方案。"
            ],
            "delay": [
                "我们需要更多时间来考虑这个提案。",
                "这个问题比较复杂，建议进一步讨论。",
                "我们暂时无法做出决定，需要内部协商。"
            ]
        }
        
        selected_templates = templates.get(response_type, templates["delay"])
        return random.choice(selected_templates)
    
    async def _determine_new_stance(self, proposal: NegotiationProposal,
                                  response_type: str, current_stance: str,
                                  strategy: Dict) -> Optional[str]:
        """确定是否改变立场"""
        # 只有在接受或提出反提案时才可能改变立场
        if response_type not in ["accept", "counter_proposal"]:
            return None
        
        # 基于提案类型和当前立场决定新立场
        if proposal.proposal_type == "alliance":
            # 联盟提案，采用提案国的立场
            return current_stances.get(proposal.from_country, current_stance)
            
        elif proposal.proposal_type == "compensation":
            # 补偿提案，可能转向支持
            if current_stance == "against" and response_type == "accept":
                return "support"
                
        elif proposal.proposal_type == "tariff_reduction":
            # 减税提案，可能转向反对
            if current_stance == "support" and response_type == "accept":
                return "against"
        
        return None
    
    async def _update_stances(self, current_stances: Dict[str, str],
                            responses: List[NegotiationResponse],
                            country_features: Dict) -> Dict[str, str]:
        """更新各国立场"""
        updated_stances = current_stances.copy()
        
        # 统计每个国家的响应
        response_summary = {}
        for response in responses:
            country = response.from_country
            if country not in response_summary:
                response_summary[country] = {
                    "accept_count": 0,
                    "reject_count": 0,
                    "counter_proposal_count": 0,
                    "new_stances": []
                }
            
            response_summary[country][f"{response.response_type}_count"] += 1
            if response.new_stance:
                response_summary[country]["new_stances"].append(response.new_stance)
        
        # 基于响应更新立场
        for country, summary in response_summary.items():
            total_responses = (summary["accept_count"] + 
                             summary["reject_count"] + 
                             summary["counter_proposal_count"])
            
            if total_responses == 0:
                continue
            
            # 如果有明确的立场改变建议
            if summary["new_stances"]:
                # 采用最常见的建议立场
                from collections import Counter
                stance_votes = Counter(summary["new_stances"])
                most_common_stance = stance_votes.most_common(1)[0][0]
                updated_stances[country] = most_common_stance
                
            # 基于响应倾向微调立场
            accept_ratio = summary["accept_count"] / total_responses
            if accept_ratio > 0.7 and updated_stances[country] == "neutral":
                # 高接受率，可能倾向于支持
                updated_stances[country] = "support"
            elif accept_ratio < 0.3 and updated_stances[country] == "neutral":
                # 低接受率，可能倾向于反对
                updated_stances[country] = "against"
        
        return updated_stances
    
    async def _analyze_alliances(self, responses: List[NegotiationResponse],
                               updated_stances: Dict[str, str]) -> Dict[str, List[str]]:
        """分析联盟形成"""
        alliances = {}
        
        # 基于相同立场形成联盟
        stance_groups = {}
        for country, stance in updated_stances.items():
            if stance not in stance_groups:
                stance_groups[stance] = []
            stance_groups[stance].append(country)
        
        # 为每个立场群体创建联盟标识
        for stance, countries in stance_groups.items():
            if len(countries) >= 2:  # 至少2个国家才能形成联盟
                alliance_name = f"{stance}_alliance"
                alliances[alliance_name] = countries
        
        # 基于谈判响应中的接受情况进一步细化联盟
        positive_responses = [r for r in responses if r.response_type == "accept"]
        for response in positive_responses:
            from_country = response.from_country
            to_country = response.to_country
            
            # 检查是否已有联盟包含这两个国家
            existing_alliance = None
            for alliance_name, members in alliances.items():
                if from_country in members and to_country in members:
                    existing_alliance = alliance_name
                    break
            
            # 如果没有现有联盟，创建新的双边联盟
            if not existing_alliance:
                new_alliance = f"bilateral_{from_country}_{to_country}"
                alliances[new_alliance] = [from_country, to_country]
        
        return alliances
    
    async def _generate_negotiation_summary(self, proposals: List[NegotiationProposal],
                                           responses: List[NegotiationResponse],
                                           updated_stances: Dict[str, str],
                                           alliances: Dict[str, List[str]]) -> str:
        """生成谈判总结"""
        summary_parts = []
        
        # 提案统计
        proposal_types = {}
        for proposal in proposals:
            proposal_types[proposal.proposal_type] = proposal_types.get(proposal.proposal_type, 0) + 1
        
        summary_parts.append(f"本轮谈判共产生{len(proposals)}个提案，{len(responses)}个响应。")
        
        # 主要提案类型
        main_proposal_type = max(proposal_types, key=proposal_types.get)
        summary_parts.append(f"主要提案类型为{main_proposal_type}（{proposal_types[main_proposal_type]}个）。")
        
        # 立场变化
        stance_changes = 0
        for country in updated_stances:
            # 这里需要与初始立场比较，简化处理
            stance_changes += 1
        
        summary_parts.append(f"共有{stance_changes}个国家调整了立场。")
        
        # 联盟形成
        if alliances:
            alliance_count = len(alliances)
            summary_parts.append(f"形成了{alliance_count}个联盟。")
        else:
            summary_parts.append("未形成明确联盟。")
        
        return " ".join(summary_parts)
    
    def _serialize_proposal(self, proposal: NegotiationProposal) -> Dict:
        """序列化提案"""
        return {
            "from_country": proposal.from_country,
            "to_country": proposal.to_country,
            "proposal_type": proposal.proposal_type,
            "content": proposal.content,
            "expected_response": proposal.expected_response,
            "urgency": proposal.urgency
        }
    
    def _serialize_response(self, response: NegotiationResponse) -> Dict:
        """序列化响应"""
        return {
            "from_country": response.from_country,
            "to_country": response.to_country,
            "response_type": response.response_type,
            "content": response.content,
            "new_stance": response.new_stance
        }

async def main():
    """测试多边谈判"""
    # 创建测试数据
    countries = ["Country_A", "Country_B", "Country_C"]
    current_stances = {
        "Country_A": "against",
        "Country_B": "support", 
        "Country_C": "neutral"
    }
    country_features = {
        "Country_A": {"economic_power": "high", "political_orientation": "center"},
        "Country_B": {"economic_power": "medium", "political_orientation": "left"},
        "Country_C": {"economic_power": "medium", "political_orientation": "center"}
    }
    
    # 创建谈判动作
    negotiation = MultilateralNegotiationAction(NegotiationRound.FIRST)
    
    # 执行谈判
    context = {
        "countries": countries,
        "current_stances": current_stances,
        "country_features": country_features
    }
    
    result = await negotiation.run(context)
    
    # 输出结果
    print("=== 多边谈判结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
