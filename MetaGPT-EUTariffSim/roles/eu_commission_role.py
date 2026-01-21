"""欧盟委员会角色智能体"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from metagpt.roles.role import Role


class EUCommissionRole(Role):
    """欧委会角色"""
    
    def __init__(self):
        super().__init__(name="EU_Commission", profile="European Commission")
        self.logger = logging.getLogger(f"{__name__}.EU_Commission")
        self.proposal_history = []
        self.communication_history = []
        self.voting_rules = None  # 将由EUTariffSimulation设置
    
    async def publish_proposal(self, tariff_rate: float = 0.35) -> Dict:
        """发布关税提案"""
        proposal = {
            "title": "欧盟拟对产自中国的电动汽车加征35%关税提案",
            "tariff_rate": tariff_rate,
            "target_sectors": ["automotive", "renewable_energy"],
            "rationale": "反补贴调查发现中国电动汽车获得不公平政府补贴，需要保护欧盟产业",
            "eu_industry_benefit": "保护欧盟汽车产业，维护公平竞争",

        }
        
        self.proposal_history.append(proposal)
        self.logger.info(f"欧委会发布{tariff_rate*100}%关税提案")
        
        return proposal
    
    async def analyze_and_communicate_voting_results(self, votes: Dict[str, str], 
                                             country_populations: Dict[str, int],
                                             voting_rules: Dict[str, float],
                                             country_anonymization_map: Dict[str, str] = None) -> Dict:
        """分析投票结果并按照欧盟规则统计，然后发送给各国
        
        Args:
            votes: 各国投票结果 {"country": "support/against/abstain"}
            country_populations: 各国人口数据 {"country": population}
            voting_rules: 欧盟投票规则 {"against_threshold_countries": 0.55, "against_threshold_population": 0.65}
            country_anonymization_map: 国家匿名化映射 {country_id: anonymous_name}
            
        Returns:
            投票分析结果和沟通信息
        """
        self.logger.info("欧委会开始分析投票结果并统计是否通过")
        
        # 使用欧盟投票规则计算结果
        voting_result = self._calculate_eu_voting_result(votes, country_populations, voting_rules)
        
        # 生成给各国的沟通信息（包含匿名化投票详情）
        communications = await self._generate_voting_result_communications(votes, voting_result, country_anonymization_map)
        
        # 保存投票分析历史
        voting_analysis = {
            "timestamp": datetime.now().isoformat(),
            "votes": votes,
            "voting_result": voting_result,
            "communications_sent": communications
        }
        self.communication_history.append(voting_analysis)
        
        # 记录关键信息
        self.logger.info(f"投票统计完成: {voting_result['reason']}")
        self.logger.info(f"向{len(communications)}个国家发送投票结果通知")
        
        return {
            "voting_analysis": voting_analysis,
            "communications": communications,
            "summary": {
                "total_countries": voting_result["total_countries"],
                "proposal_passed": voting_result["passed"],
                "support_count": voting_result["support_count"],
                "against_count": voting_result["against_count"],
                "abstain_count": voting_result["abstain_count"],
                "against_country_ratio": voting_result["against_country_ratio"],
                "against_population_ratio": voting_result["against_population_ratio"]
            }
        }
    
    def _calculate_eu_voting_result(self, votes: Dict[str, str], 
                                   country_populations: Dict[str, int],
                                   voting_rules: Dict[str, float]) -> Dict[str, Any]:
        """根据欧盟投票规则计算投票结果
        
        欧盟投票规则：有55%及以上数量的国家且占65%人口的国家投反对票，才会否决此项决议
        
        Args:
            votes: 各国投票结果
            country_populations: 各国人口数据
            voting_rules: 投票规则
            
        Returns:
            投票结果统计和是否通过
        """
        total_countries = len(votes)
        if total_countries == 0:
            return {
                "total_countries": 0,
                "support_count": 0,
                "against_count": 0,
                "abstain_count": 0,
                "against_country_ratio": 0.0,
                "against_population_ratio": 0.0,
                "passed": True,
                "reason": "无投票，默认通过",
                "voting_details": {}
            }
        
        # 统计投票数量
        support_count = sum(1 for vote in votes.values() if vote == "support")
        against_count = sum(1 for vote in votes.values() if vote == "against")
        abstain_count = sum(1 for vote in votes.values() if vote == "abstain")
        
        # 计算反对票比例
        against_country_ratio = against_count / total_countries
        
        # 计算反对票人口比例
        total_population = 0
        against_population = 0
        
        for country_id, vote in votes.items():
            if country_id in country_populations:
                population = country_populations[country_id]
                total_population += population
                
                if vote == "against":
                    against_population += population
        
        against_population_ratio = against_population / total_population if total_population > 0 else 0.0
        
        # 判断是否通过
        # 只有同时满足以下两个条件才会否决：
        # 1. 55%及以上数量的国家投反对票
        # 2. 占65%人口的国家投反对票
        # 否则通过（包括弃权情况）
        
        veto_by_countries = against_country_ratio >= voting_rules["against_threshold_countries"]
        veto_by_population = against_population_ratio >= voting_rules["against_threshold_population"]
        
        # 必须同时满足两个条件才会否决
        veto_triggered = veto_by_countries and veto_by_population
        passed = not veto_triggered
        
        if veto_triggered:
            reason = f"否决：反对国家比例{against_country_ratio:.1%}≥{voting_rules['against_threshold_countries']:.0%} 且 反对人口比例{against_population_ratio:.1%}≥{voting_rules['against_threshold_population']:.0%}"
        else:
            reason = f"通过：不满足否决条件（反对国家比例{against_country_ratio:.1%}，反对人口比例{against_population_ratio:.1%}）"
        
        result = {
            "total_countries": total_countries,
            "support_count": support_count,
            "against_count": against_count,
            "abstain_count": abstain_count,
            "against_country_ratio": against_country_ratio,
            "against_population_ratio": against_population_ratio,
            "passed": passed,
            "reason": reason,
            "voting_details": {
                "countries_against": [cid for cid, vote in votes.items() if vote == "against"],
                "countries_support": [cid for cid, vote in votes.items() if vote == "support"],
                "countries_abstain": [cid for cid, vote in votes.items() if vote == "abstain"],
                "total_population": total_population,
                "against_population": against_population,
                "veto_by_countries": veto_by_countries,
                "veto_by_population": veto_by_population
            }
        }
        
        return result
    
    async def _generate_voting_result_communications(self, votes: Dict[str, str], 
                                               voting_result: Dict[str, Any],
                                               country_anonymization_map: Dict[str, str] = None) -> List[Dict]:
        """生成给各国的投票结果通知（包含各国匿名化后的投票详情）
        
        Args:
            votes: 各国投票结果
            voting_result: 投票统计结果
            country_anonymization_map: 国家匿名化映射 {country_id: anonymous_name}
            
        Returns:
            通信信息列表
        """
        communications = []
        
        proposal_passed = voting_result["passed"]
        reason = voting_result["reason"]
        
        # 如果没有匿名化映射，创建一个临时的（使用原始国家ID）
        if country_anonymization_map is None:
            country_anonymization_map = {cid: cid for cid in votes.keys()}
        
        # 构建匿名化的其他国家投票详情
        anonymous_other_countries_voting = []
        for other_country_id, other_vote in votes.items():
            anonymous_name = country_anonymization_map.get(other_country_id, other_country_id)
            vote_display = {
                "support": "支持",
                "against": "反对",
                "abstain": "弃权",
                "neutral": "中立"
            }.get(other_vote, "未知")
            
            anonymous_other_countries_voting.append({
                "anonymous_country_id": anonymous_name,
                "vote": vote_display
            })
        
        # 为每个国家生成个性化通知
        for country_id, vote in votes.items():
            communication = {
                "from": "EU_Commission",
                "to": country_id,
                "type": "voting_result_notification",
                "timestamp": datetime.now().isoformat()
            }
            
            # 获取当前国家的匿名化名称
            current_anonymous_name = country_anonymization_map.get(country_id, country_id)
            vote_display = {
                "support": "支持",
                "against": "反对",
                "abstain": "弃权",
                "neutral": "中立"
            }.get(vote, "未知")
            
            if proposal_passed:
                # 提案通过的情况
                if vote == "support":
                    message = f"感谢贵国支持关税提案。根据上一次咨询性的投票，提案暂未被否决。我们赞赏贵国对维护欧盟产业安全的贡献。"
                    tone = "appreciation"
                elif vote == "against":
                    message = f"虽然贵国反对关税提案，根据上一次咨询性的投票，提案暂未被否决。我们理解贵国的立场，但欧盟整体利益需要得到维护。"
                    tone = "understanding"
                else:  # abstain
                    message = f"贵国在关税提案中弃权。根据上一次咨询性的投票，提案暂未被否决。我们希望贵国在后续实施中能够支持欧盟的共同立场。"
                    tone = "neutral"
            else:
                # 提案被否决的情况
                if vote == "against":
                    message = f"感谢贵国反对关税提案。根据上一次咨询性的投票，提案已被否决。这反映了欧盟成员国对公平贸易的共同关切。"
                    tone = "appreciation"
                elif vote == "support":
                    message = f"虽然贵国支持关税提案，但上一次咨询性的投票，提案已被否决。我们理解贵国对产业保护的关切，但需要寻找其他解决方案。"
                    tone = "reassurance"
                else:  # abstain
                    message = f"贵国在关税提案中弃权。根据上一次咨询性的投票，提案已被否决。欧委会将继续寻求其他方式解决贸易争端。"
                    tone = "informational"
            
            communication["message"] = {
                "content": message,
                "tone": tone,
                "your_anonymous_id": current_anonymous_name,
                "your_vote": vote_display,
                "voting_result_summary": {
                    "proposal_passed": proposal_passed,
                    "your_vote": vote_display,
                    "final_reason": reason,
                    "total_support": voting_result["support_count"],
                    "total_against": voting_result["against_count"],
                    "total_abstain": voting_result["abstain_count"],
                    "against_country_ratio": voting_result["against_country_ratio"],
                    "against_population_ratio": voting_result["against_population_ratio"],
                    "voting_rules_applied": {
                        "against_threshold_countries": "55%",
                        "against_threshold_population": "65%"
                    }
                },
                "other_countries_voting_anonymous": anonymous_other_countries_voting
            }
            
            communications.append(communication)
        
        return communications
    
    async def _generate_china_voting_notification(self, votes: Dict[str, str], 
                                          voting_result: Dict[str, Any]) -> Dict:
        """生成给中国的第一次投票结果通知
        
        Args:
            votes: 各国投票结果
            voting_result: 投票统计结果
            
        Returns:
            给中国的通信信息
        """
        proposal_passed = voting_result["passed"]
        reason = voting_result["reason"]
        
        # 构建投票统计摘要（不匿名化，因为中国不是欧盟成员国）
        voting_summary = {
            "total_countries": voting_result["total_countries"],
            "support_count": voting_result["support_count"],
            "against_count": voting_result["against_count"],
            "abstain_count": voting_result["abstain_count"],
            "proposal_passed": proposal_passed,
            "reason": reason,
            "voting_rules": {
                "against_threshold_countries": "55%",
                "against_threshold_population": "65%"
            }
        }
        
        # 生成给中国的沟通信息
        if proposal_passed:
            message = f"欧盟各成员国已完成第一次投票。根据欧盟投票规则，关税提案已{reason}。我们感谢您的关注，并将继续按照程序推进后续流程。"
        else:
            message = f"欧盟各成员国已完成第一次投票。根据欧盟投票规则，关税提案已{reason}。我们理解各方关切，将考虑下一步行动。"
        
        communication = {
            "from": "EU_Commission",
            "to": "China",
            "type": "voting_result_notification",
            "timestamp": datetime.now().isoformat(),
            "message": {
                "content": message,
                "voting_summary": voting_summary,
                "voting_round": "first"
            }
        }
        
        return communication
    
    async def communicate_with_countries(self, countries: List[str], voting_results: Dict) -> List[Dict]:
        """与特定国家沟通"""
        communications = []
        
        for country_id in countries:
            # 根据投票结果生成不同的沟通内容
            stance = voting_results.get(country_id, "neutral")
            
            if stance == "against":
                message = {
                    "type": "persuasion",
                    "content": f"我们理解贵国对自由贸易的坚持。",
                    "incentives": [""],
                    "urgency": 0.8,
                    "quality": 0.7
                }
            elif stance == "support":
                message = {
                    "type": "coordination",
                    "content": f"感谢贵国的支持。",
                    "coordination_points": [""],
                    "urgency": 0.6,
                    "quality": 0.8
                }
            else:
                message = {
                    "type": "information",
                    "content": f"我们理解贵国的决定。",
                    "additional_info": [""],
                    "urgency": 0.5,
                    "quality": 0.6
                }
            
            communication = {
                "from": "EU_Commission",
                "to": country_id,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            communications.append(communication)
            self.communication_history.append(communication)
        
        self.logger.info(f"欧委会向{len(countries)}个国家发送沟通信息")
        return communications
