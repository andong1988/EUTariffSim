"""
中国商务部角色模块
负责制定反制策略，响应欧盟关税决策
"""

from typing import Dict, Any, List, Optional
from metagpt.roles import Role
from metagpt.actions import Action
from metagpt.schema import Message
import json


class ChineseCommerceRole(Role):
    """中国商务部角色，负责制定反制策略"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        初始化中国商务部角色
        
        Args:
            config: 配置字典，包含角色配置
            **kwargs: 其他参数，传递给父类
        """
        # 设置默认值，但允许通过kwargs覆盖
        default_kwargs = {
            'name': kwargs.get('name', 'chinese_commerce'),
            'profile': kwargs.get('profile', '中国商务部'),
            'goal': kwargs.get('goal', '制定合理的反制措施，保护中国汽车产业利益，维护国际贸易规则')
        }
        
        # 合并参数，确保kwargs中的值覆盖默认值
        for key in ['name', 'profile', 'goal']:
            if key in kwargs:
                default_kwargs[key] = kwargs[key]
        
        # 移除已处理的参数，避免重复传递
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['name', 'profile', 'goal']}
        
        super().__init__(
            name=default_kwargs['name'],
            profile=default_kwargs['profile'],
            goal=default_kwargs['goal'],
            **filtered_kwargs
        )
        
        # 存储配置
        self.config = config
        
        # 初始化反制策略库
        self.countermeasure_library = self._init_countermeasure_library()
        
        # 初始化事件响应映射
        self.event_response_mapping = self._init_event_response_mapping()
    
    def _init_countermeasure_library(self) -> Dict[str, Dict[str, Any]]:
        """初始化反制策略库"""
        return {
            "tariff_retaliation": {
                "name": "关税反制",
                "description": "对欧盟汽车及相关产品加征关税",
                "intensity_levels": ["低", "中", "高"],
                "economic_impact": "直接影响欧盟汽车出口",
                "political_cost": "中等"
            },
            "export_restriction": {
                "name": "出口限制",
                "description": "限制关键原材料或零部件出口",
                "intensity_levels": ["低", "中", "高"],
                "economic_impact": "影响欧盟汽车产业链",
                "political_cost": "高"
            },
            "investment_review": {
                "name": "投资审查",
                "description": "加强对欧盟在华投资的审查",
                "intensity_levels": ["低", "中"],
                "economic_impact": "间接影响欧盟企业",
                "political_cost": "低"
            },
            "wto_complaint": {
                "name": "WTO申诉",
                "description": "向世界贸易组织提起申诉",
                "intensity_levels": ["低"],
                "economic_impact": "法律和外交影响",
                "political_cost": "低"
            },
            "diplomatic_protest": {
                "name": "外交抗议",
                "description": "通过外交渠道表达抗议",
                "intensity_levels": ["低"],
                "economic_impact": "政治影响",
                "political_cost": "低"
            }
        }
    
    def _init_event_response_mapping(self) -> Dict[str, List[str]]:
        """初始化事件响应映射"""
        return {
            "tariff_announcement": ["tariff_retaliation", "wto_complaint", "diplomatic_protest"],
            "tariff_vote_passed": ["tariff_retaliation", "export_restriction", "wto_complaint"],
            "tariff_vote_failed": ["diplomatic_protest", "investment_review"],
            "negotiation_breakdown": ["tariff_retaliation", "export_restriction"],
            "diplomatic_tension": ["diplomatic_protest", "investment_review"]
        }
    
    async def _think(self) -> None:
        """思考过程：分析当前局势，制定反制策略"""
        # 获取当前状态
        current_state = self.rc.state if hasattr(self.rc, 'state') else {}
        eu_decision = current_state.get('eu_decision', {})
        tariff_level = eu_decision.get('tariff_level', 0)
        vote_result = eu_decision.get('vote_result', 'pending')
        
        # 分析局势严重性
        severity = self._assess_situation_severity(tariff_level, vote_result)
        
        # 制定反制策略
        countermeasures = self._formulate_countermeasures(severity, eu_decision)
        
        # 设置下一步行动
        self.rc.todo = self.actions[0] if self.actions else None
        
        # 存储分析结果
        self.rc.memory.add(Message(
            content=f"局势分析完成：严重性={severity}，建议反制措施={countermeasures}",
            role=self.profile
        ))
    
    def _assess_situation_severity(self, tariff_level: float, vote_result: str) -> str:
        """评估局势严重性"""
        if vote_result == 'passed' and tariff_level >= 20:
            return "高"
        elif vote_result == 'passed' and tariff_level >= 10:
            return "中"
        elif vote_result == 'passed':
            return "低"
        elif vote_result == 'failed':
            return "极低"
        else:
            return "待定"
    
    def _formulate_countermeasures(self, severity: str, eu_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """制定反制措施"""
        # 根据严重性选择策略类型
        if severity == "高":
            strategy_types = ["tariff_retaliation", "export_restriction", "wto_complaint"]
        elif severity == "中":
            strategy_types = ["tariff_retaliation", "wto_complaint"]
        elif severity == "低":
            strategy_types = ["wto_complaint", "diplomatic_protest"]
        else:
            strategy_types = ["diplomatic_protest"]
        
        # 生成具体措施
        countermeasures = []
        for strategy_type in strategy_types:
            if strategy_type in self.countermeasure_library:
                strategy = self.countermeasure_library[strategy_type].copy()
                
                # 根据严重性设置强度
                if severity == "高":
                    strategy['intensity'] = "高"
                    strategy['implementation_timing'] = "立即"
                elif severity == "中":
                    strategy['intensity'] = "中"
                    strategy['implementation_timing'] = "1周内"
                else:
                    strategy['intensity'] = "低"
                    strategy['implementation_timing'] = "2周内"
                
                # 添加针对性信息
                strategy['target'] = "欧盟汽车产业"
                strategy['rationale'] = f"应对欧盟{eu_decision.get('tariff_level', 0)}%的汽车关税"
                
                countermeasures.append(strategy)
        
        return countermeasures
    
    async def _act(self) -> Message:
        """执行行动：生成反制措施报告"""
        if not self.rc.todo:
            return Message(content="无待执行行动", role=self.profile)
        
        # 获取思考结果
        recent_messages = self.rc.memory.get()
        analysis_result = ""
        for msg in recent_messages[-3:]:  # 获取最近3条消息
            if "局势分析完成" in msg.content:
                analysis_result = msg.content
                break
        
        # 生成反制措施报告
        if analysis_result:
            # 从分析结果中提取信息
            severity = analysis_result.split("严重性=")[1].split("，")[0] if "严重性=" in analysis_result else "未知"
            
            # 生成正式报告
            report = self._generate_countermeasure_report(severity)
            
            return Message(
                content=report,
                role=self.profile,
                cause_by=type(self.rc.todo)
            )
        else:
            # 生成默认报告
            report = self._generate_default_countermeasure_report()
            
            return Message(
                content=report,
                role=self.profile,
                cause_by=type(self.rc.todo)
            )
    
    def _generate_countermeasure_report(self, severity: str) -> str:
        """生成反制措施报告"""
        report = {
            "report_title": "中国商务部关于欧盟汽车关税的反制措施建议",
            "report_date": "2024-12-14",
            "situation_assessment": {
                "severity": severity,
                "description": f"欧盟对华汽车关税投票结果严重性评估为{severity}级"
            },
            "recommended_countermeasures": [],
            "implementation_timeline": "根据严重性级别确定",
            "expected_impact": "保护中国汽车产业利益，维护国际贸易规则",
            "diplomatic_considerations": "在维护国家利益的同时，保持外交沟通渠道畅通"
        }
        
        # 根据严重性添加具体措施
        if severity == "高":
            report["recommended_countermeasures"] = [
                {
                    "measure": "对欧盟汽车加征对等关税",
                    "intensity": "高",
                    "timing": "立即实施",
                    "rationale": "应对欧盟高额关税，维护贸易公平"
                },
                {
                    "measure": "限制关键汽车零部件出口",
                    "intensity": "中",
                    "timing": "1周内实施",
                    "rationale": "影响欧盟汽车产业链，增加其成本压力"
                },
                {
                    "measure": "向WTO提起正式申诉",
                    "intensity": "低",
                    "timing": "2周内提交",
                    "rationale": "通过多边机制解决贸易争端"
                }
            ]
        elif severity == "中":
            report["recommended_countermeasures"] = [
                {
                    "measure": "对欧盟汽车加征适度关税",
                    "intensity": "中",
                    "timing": "1周内实施",
                    "rationale": "适度回应，保持升级空间"
                },
                {
                    "measure": "向WTO表达关切",
                    "intensity": "低",
                    "timing": "立即进行",
                    "rationale": "通过外交和法律渠道施压"
                }
            ]
        else:
            report["recommended_countermeasures"] = [
                {
                    "measure": "通过外交渠道表达抗议",
                    "intensity": "低",
                    "timing": "立即进行",
                    "rationale": "表达立场，保持沟通"
                }
            ]
        
        return json.dumps(report, ensure_ascii=False, indent=2)
    
    def _generate_default_countermeasure_report(self) -> str:
        """生成默认反制措施报告"""
        return json.dumps({
            "report_title": "中国商务部关于欧盟汽车关税的初步反制措施建议",
            "report_date": "2024-12-14",
            "situation_assessment": {
                "severity": "待评估",
                "description": "等待欧盟投票结果正式公布"
            },
            "recommended_countermeasures": [
                {
                    "measure": "准备多种反制预案",
                    "intensity": "待定",
                    "timing": "结果公布后24小时内",
                    "rationale": "根据投票结果灵活应对"
                }
            ],
            "implementation_timeline": "结果公布后确定",
            "expected_impact": "保护国家利益，维护贸易公平",
            "diplomatic_considerations": "保持战略定力，维护多边贸易体系"
        }, ensure_ascii=False, indent=2)
    
    async def react(self) -> Message:
        """重写react方法，整合思考和行动"""
        await self._think()
        return await self._act()


# 配置示例
config = {
    "countermeasure_intensity": "proportional",  # 反制强度：proportional（比例对应）、escalating（逐步升级）、minimal（最小化）
    "diplomatic_channel": "active",  # 外交渠道：active（积极）、reserved（保留）、minimal（最小）
    "economic_consideration": "balanced",  # 经济考量：balanced（平衡）、protective（保护性）、strategic（战略性）
    "implementation_speed": "moderate"  # 实施速度：immediate（立即）、moderate（适度）、gradual（渐进）
}


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test_role():
        role = ChineseCommerceRole(config, name="test_chinese_commerce")
        print(f"角色创建成功: {role.name}")
        print(f"角色目标: {role.goal}")
        print(f"反制策略库: {list(role.countermeasure_library.keys())}")
        
        # 测试局势评估
        severity = role._assess_situation_severity(25, "passed")
        print(f"高关税通过时的严重性评估: {severity}")
        
        # 测试反制措施制定
        eu_decision = {"tariff_level": 25, "vote_result": "passed"}
        countermeasures = role._formulate_countermeasures("高", eu_decision)
        print(f"建议的反制措施数量: {len(countermeasures)}")
        
        # 测试报告生成
        report = role._generate_countermeasure_report("高")
        print(f"报告生成成功，长度: {len(report)} 字符")
    
    asyncio.run(test_role())
