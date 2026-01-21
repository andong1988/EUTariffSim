"""中国角色智能体"""

import logging
from typing import Dict, List
from datetime import datetime
from metagpt.roles.role import Role


class ChinaRole(Role):
    """中国角色"""
    
    def __init__(self):
        super().__init__(name="China", profile="People's Republic of China")
        self.logger = logging.getLogger(f"{__name__}.China")
        self.retaliation_history = []
        self.communication_history = []
    
    async def assess_retaliation(self, voting_results: Dict, proposal_passed: bool = False) -> Dict:
        """评估反制措施
        
        Args:
            voting_results: 各国投票结果
            proposal_passed: 提案是否通过（不满足否决条件即为通过）
        """
        support_count = sum(1 for vote in voting_results.values() if vote == "support")
        total_count = len(voting_results)
        support_ratio = support_count / total_count
        
        # 修复：基于提案是否通过来判断是否触发反制，而不是基于支持票是否过半
        # 如果提案通过（不满足否决条件），就应该触发反制
        if proposal_passed:
            severity = "high" if support_ratio > 0.7 else "medium"
            
            retaliation = {
                "triggered": True,
                "severity": severity,
                "support_ratio": support_ratio,
                "reason": f"提案通过（支持比例{support_ratio:.1%}），触发反制措施",
                "measures": [
                    "中方对裁决结果不认同、不接受，已就此在世贸组织争端解决机制下提出诉讼。中方将继续采取一切必要措施坚决维护中国企业的合法权益。",
                    "对产自欧盟的白兰地展开反倾销调查、对产自欧盟的猪肉展开反倾销调查、对产自欧盟的乳制品展开反倾销调查,涉及从欧盟进口的几种奶酪、牛奶和奶油产品。",
                    "中国机电商会向欧委会提交了出口电动车的最低价格承诺方案。"
                    """Implement export controls on rare earth products (including but not limited to tungsten, tellurium, bismuth, molybdenum, indium-related items)
                     exported to the EU. European Commission President Ursula von der Leyen mentioned that 98% of the EU's rare earths, 93% of magnesium,
                      and 97% of lithium come from China, and these materials play a crucial role in the modern industrial system. """
                ],
                "target_countries": [cid for cid, vote in voting_results.items() if vote == "support"],
                "economic_impact": {
                    "eu_export_loss": f"{support_ratio * 2:.1f}%",
                    "job_impact": f"{support_ratio * 50000:.0f}",
                    "gdp_impact": f"{support_ratio * 0.3:.2f}%"
                }
            }
        else:
            retaliation = {
                "triggered": False,
                "reason": "提案未通过（被否决），无需反制",
                "support_ratio": support_ratio,
                "measures": [],
                "economic_impact": {}
            }
        
        self.retaliation_history.append(retaliation)
        self.logger.info(f"中国反制评估: {'触发' if retaliation['triggered'] else '未触发'} (提案通过: {proposal_passed}, 支持比例: {support_ratio:.1%})")
        
        return retaliation
    
    async def communicate_with_countries(self, target_countries: List[str], retaliation: Dict) -> List[Dict]:
        """与特定国家沟通"""
        communications = []
        
        for country_id in target_countries:
            # 根据反制触发状态和目标国家立场生成不同的沟通信息
            triggered = retaliation.get("triggered", False)
            target_support = country_id in retaliation.get("target_countries", [])
            
            if triggered and target_support:
                # 反制已触发且该目标支持关税：发送反制警告
                message = {
                    "type": "warning",
                    "content": f"欧盟的关税措施将严重影响中欧经贸关系。",
                    "retaliation_details": retaliation["measures"],
                    "economic_impact_warning": retaliation["economic_impact"],
                    "urgency": 0.9,
                    "quality": 0.8
                }
            elif triggered and not target_support:
                # 反制已触发但该目标不支持关税：发送外交沟通
                message = {
                    "type": "diplomacy",
                    "content": f"尽管欧盟部分国家支持关税，中方希望与贵国继续保持友好合作关系，避免贸易战波及双边经贸。",
                    "cooperation_opportunities": ["深化投资合作", "技术交流", "市场开放"],
                    "urgency": 0.6,
                    "quality": 0.7
                }
            else:
                # 反制未触发：发送外交沟通
                message = {
                    "type": "diplomacy",
                    "content": f"感谢贵国的理性立场，我们希望继续加强中欧经贸合作，共同维护全球贸易稳定。",
                    "cooperation_opportunities": ["深化投资合作", "技术交流", "市场开放"],
                    "urgency": 0.4,
                    "quality": 0.7
                }
            
            communication = {
                "from": "China",
                "to": country_id,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            communications.append(communication)
            self.communication_history.append(communication)
        
        self.logger.info(f"中国向{len(target_countries)}个国家发送沟通信息（反制触发: {retaliation.get('triggered', False)}）")
        return communications
