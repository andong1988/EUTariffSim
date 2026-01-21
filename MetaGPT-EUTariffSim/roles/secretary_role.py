"""秘书智能体 - 分析沟通信息并生成后续效应"""

import logging
import os
from typing import Dict, List, Any
from datetime import datetime
from metagpt.roles.role import Role
from metagpt.actions import Action
import json


class SecretaryAnalysisAction(Action):
    """秘书分析动作：分析沟通信息并生成后续效应"""
    
    name: str = "SecretaryAnalysisAction"
    desc: str = "分析沟通信息并生成后续效应"
    
    async def run(self, 
                  country_id: str,
                  country_features: Dict[str, Any],
                  communications: Dict[str, Any],
                  round_name: str = 'initial') -> Dict[str, Any]:
        """
        执行秘书分析
        
        Args:
            country_id: 国家ID
            country_features: 国家特征（匿名化数据）
            communications: 沟通信息字典
            round_name: 轮次名称（'initial' 或 'final'）
            
        Returns:
            后续效应分析结果
        """
        # 构建分析prompt
        prompt = self._build_analysis_prompt(country_id, country_features, communications, round_name)
        
        # 保存prompt到文件
        try:
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"secretary_prompt_{country_id}_{round_name}_{timestamp}.txt"
            
            # 定义文件路径（与理论决策prompt保存在同一目录）
            prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'actions', 'prompts')
            file_path = os.path.join(prompt_dir, filename)
            
            # 确保目录存在
            os.makedirs(prompt_dir, exist_ok=True)
            
            # 将prompt写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            logging.info(f"秘书分析prompt已保存到: {file_path}")
        except Exception as e:
            logging.error(f"保存秘书分析prompt失败: {e}")
        
        # 调用LLM进行分析
        try:
            response = await self._aask(prompt)
            
            # 解析响应
            effect_analysis = self._parse_effect_analysis(response)
            
            return {
                "country_id": country_id,
                "effect_analysis": effect_analysis,
                "raw_response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"秘书分析失败: {e}")
            # 返回默认分析结果
            return {
                "country_id": country_id,
                "effect_analysis": {
                    "market_effect": "未能分析沟通信息对市场维度的影响",
                    "political_effect": "未能分析沟通信息对政治维度的影响",
                    "institutional_effect": "未能分析沟通信息对制度维度的影响",
                    "overall_impact": "由于分析失败，无法评估后续效应"
                },
                "raw_response": "",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _build_analysis_prompt(self, 
                              country_id: str,
                              country_features: Dict[str, Any],
                              communications: Dict[str, Any],
                              round_name: str = 'initial') -> str:
        """
        构建分析提示词
        
        Args:
            country_id: 国家ID
            country_features: 国家特征（匿名化数据）
            communications: 沟通信息字典
            round_name: 轮次名称（'initial' 或 'final'）
        """
        # 提取国家特征
        anonymized_text = country_features.get('anonymized_text', {})
        
        # 提取沟通信息
        country_to_country = communications.get('country_to_country', [])
        eu_commission = communications.get('eu_commission', [])
        china_targeted = communications.get('china_targeted', [])
        china_general = communications.get('china_general', [])
        retaliation = communications.get('retaliation', {})
        
        # 构建沟通详情
        comm_details = self._format_communications(
            country_to_country, eu_commission, china_targeted, china_general, retaliation, country_id
        )
        
        # 🔴 特殊处理：如果当前国家是爱尔兰且是第二轮投票，添加欧委会的单独承诺
        if country_id == 'Ireland' and round_name == 'final':
            eu_commission_promise_section = """### 欧盟委员会对Ireland的单独承诺：

The European Commission promises: The EU market will provide support for Ireland's dairy exports to compensate for Ireland's losses, hoping Ireland will support the European Commission's investigation results, maintain the EU's unified position, and vote in favor.
"""
            comm_details = eu_commission_promise_section + "\n\n" + comm_details
        
        prompt = f"""作为{country_id}的决策顾问秘书，请分析针对{country_id}的所有沟通信息，评估这些沟通对该国决策的后续效应。

【{country_id}的基本信息】

### X_market (Market / Economic Interdependence):
{anonymized_text.get('X_market (Market / Economic Interdependence)', '')}

### X_political (Domestic Politics and Interest Mediation):
{anonymized_text.get('X_political (Domestic Politics and Interest Mediation)', '')}

### X_institutional (Institutions, Diplomacy, and Path Dependence):
{anonymized_text.get('X_institutional (Institutions, Diplomacy, and Path Dependence)', '')}

【收到的沟通信息】

{comm_details}

【分析任务】

请综合分析以上所有沟通信息，评估这些沟通对{country_id}在欧盟对华汽车关税议题上的后续影响。请从以下三个维度进行分析：

## 分析维度

### 1. 市场维度后续效应（X_market）
分析各方的沟通对该国经济利益的潜在影响：
- 其他欧盟国家及中国的实质性承诺对该国经济利益的直接影响
- **欧委会的正式承诺**（如补偿措施、市场准入支持等）是确定性的制度性安排，应作为核心考量，能有效对冲反制风险
- 中国的反制威胁或合作提议对经济贸易的实际影响，当存在欧委会承诺时可降低其权重
- 中国的反制措施，对本国关键行业经济的影响，非关键行业可降低权重

### 2. 政治维度后续效应（X_political）
分析各方的沟通对该国国内政治的潜在影响：
- 欧委会的团结呼吁如何影响该国的欧盟立场
- 考虑欧盟统一立场的影响
- 中国的警告或合作提议如何影响国内政治压力，若经济影响过大，会增大政治压力
- 中国的反制措施是否会引起国内政治势力的反对


### 3. 制度维度后续效应（X_institutional）
分析各方的沟通对该国外交和制度关系的潜在影响：

**首要分析：欧委会及高层领导人互动**
- **领导人会晤级别**：国事访问、正式访问、工作访问、礼节性会晤等不同级别的影响权重差异显著
- **协议达成情况**：是否达成具体合作意向、合作协议、谅解备忘录、联合声明等具有法律或政治约束力的文件、发布了什么声明
- **制度化程度**：领导人达成的协议是否建立长期对话机制、合作框架或制度化安排
- **欧委会的实质性沟通内容**欧委会的实质性沟通，对本国有利，应作为重要的正面因素考量

**其他制度因素分析**：
- 与他国的双边关系如何影响制度性决策
- 与中国的外交关系如何影响长期制度选择，需结合领导人互动的深度和成果


## 分析原则

- 欧委会和中国等核心行为体的沟通是主要分析对象

- 其他国家的沟通作为参考，权重较低

- 只关注其他国家的实质性承诺及中国的制裁措施，一般性沟通可忽略

【输出要求】

请输出结构化的JSON格式分析结果，格式如下：

{{
  "market_effect": "评估沟通对贸易依赖、产业保护、供应链等方面的影响...",
  "political_effect": "评估沟通对国内政治压力、欧盟团结、政策成本等方面的影响...",
  "institutional_effect": "评估沟通对领导人会晤、外交关系、欧盟一体化、对华政策等方面的影响...",
}}

注意事项：
1. 分析要基于{country_id}的具体特征，不要泛泛而谈
2. 评估要客观理性，考虑沟通的可信度和实际影响力
3. 每个维度的分析都要有具体依据
4. 字数不大于200字。

只输出JSON，不要输出其他内容。"""
        
        return prompt
    
    def _format_communications(self,
                              country_to_country: List[Dict],
                              eu_commission: List[Dict],
                              china_targeted: List[Dict],
                              china_general: List[Dict],
                              retaliation: Dict,
                              country_id: str,
                              round_name: str = 'initial') -> str:
        """
        格式化沟通信息
        
        Args:
            country_to_country: 国家间沟通列表
            eu_commission: 欧委会沟通列表
            china_targeted: 中国针对性沟通列表
            china_general: 中国一般性沟通列表
            retaliation: 中国反制措施信息
            country_id: 国家ID
            round_name: 轮次名称（'initial' 或 'final'）
        """
        details = []
        
        # 1. 来自其他国家的沟通
        targeted_comms = [
            comm for comm in country_to_country 
            if comm.get('to') == country_id
        ]
        
        if targeted_comms:
            details.append(f"### 来自其他国家的沟通（共{len(targeted_comms)}条）：")
            for i, comm in enumerate(targeted_comms, 1):
                from_country = comm.get('from', 'unknown')
                content = self._extract_communication_content(comm)
                details.append(f"\n{i}. {from_country}的沟通：\n{content}")
        else:
            details.append("### 来自其他国家的沟通：无")
        
        # 2. 来自欧盟委员会的单独承诺
        individual_promises = [
            comm for comm in eu_commission 
            if comm.get('type') == 'individual_promise' or comm.get('to') == country_id
        ]
        
        if individual_promises:
            details.append(f"\n### 来自欧盟委员会的单独承诺：")
            for i, comm in enumerate(individual_promises, 1):
                from_entity = comm.get('from', comm.get('sender', 'EU_Commission'))
                content = self._extract_communication_content(comm)
                details.append(f"\n{i}. {from_entity}的沟通：\n{content}")
        else:
            details.append("\n### 来自欧盟委员会的单独承诺：无")
        
        # 4. 来自中国的针对性沟通
        if china_targeted:
            details.append(f"\n### 来自中国的针对性沟通（共{len(china_targeted)}条）：")
            for i, comm in enumerate(china_targeted, 1):
                content = self._extract_communication_content(comm)
                details.append(f"\n{i}. {content}")
        else:
            details.append("\n### 来自中国的针对性沟通：无")
        
        # 5. 中国反制措施信息
        if retaliation:
            triggered = retaliation.get('triggered', False)
            if triggered:
                details.append(f"\n### 中国反制措施：已触发")
                measures = retaliation.get('measures', [])
                if measures:
                    details.append("\n具体措施：")
                    for measure in measures:
                        details.append(f"- {measure}")
            else:
                details.append(f"\n### 中国反制措施：未触发")
        else:
            details.append("\n### 中国反制措施：无")
        
        return "\n".join(details)
    
    def _extract_communication_content(self, comm: Dict) -> str:
        """
        从沟通字典中提取内容
        """
        # 尝试从不同位置获取内容
        if 'content' in comm:
            content = comm['content']
            if isinstance(content, dict):
                if 'message' in content:
                    return content['message']
                elif 'communication' in content:
                    return content['communication']
                else:
                    return str(content)
            elif isinstance(content, str):
                return content
        
        if 'message' in comm:
            message = comm['message']
            if isinstance(message, dict):
                if 'content' in message:
                    return message['content']
                else:
                    return str(message)
            elif isinstance(message, str):
                return message
        
        if 'communication' in comm:
            return comm['communication']
        
        # 如果都找不到，返回整个comm的字符串表示
        return str(comm)
    
    def _parse_effect_analysis(self, response: str) -> Dict[str, str]:
        """
        解析LLM响应，提取后续效应分析
        """
        import re
        
        try:
            # 尝试直接解析JSON
            response = response.strip()
            if response.startswith('{'):
                return json.loads(response)
            
            # 尝试从文本中提取JSON
            json_pattern = r'\{[^}]*"market_effect"[^}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
            
            # 如果JSON解析失败，尝试手动提取
            result = {
                "market_effect": self._extract_field(response, "market_effect"),
                "political_effect": self._extract_field(response, "political_effect"),
                "institutional_effect": self._extract_field(response, "institutional_effect"),
                "overall_impact": self._extract_field(response, "overall_impact")
            }
            
            # 确保所有字段都有值
            for key in result:
                if not result[key]:
                    result[key] = "未能提取到该维度的分析"
            
            return result
            
        except Exception as e:
            logging.error(f"解析后续效应分析失败: {e}")
            return {
                "market_effect": "解析失败",
                "political_effect": "解析失败",
                "institutional_effect": "解析失败",
                "overall_impact": "解析失败"
            }
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """
        从文本中提取特定字段
        """
        patterns = [
            rf'"{field_name}"\s*:\s*"([^"]*)"',
            rf"'{field_name}'\s*:\s*'([^']*)'",
            rf'{field_name}\s*:\s*"([^"]*)"',
            rf'{field_name}\s*:\s*"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return ""


class SecretaryRole(Role):
    """秘书智能体：为特定国家分析沟通信息并生成后续效应"""
    
    def __init__(self, country_id: str, anonymized_data: Dict[str, Any], **kwargs):
        """
        初始化秘书智能体
        
        Args:
            country_id: 国家ID
            anonymized_data: 国家匿名化数据（基本信息）
        """
        super().__init__(
            name=f"Secretary_{country_id}",
            profile=f"Decision Secretary for {country_id}",
            **kwargs
        )
        
        self.country_id = country_id
        self.anonymized_data = anonymized_data
        self.logger = logging.getLogger(f"{__name__}.Secretary_{country_id}")
        
        # 初始化分析动作
        self.analysis_action = SecretaryAnalysisAction()
        
        # 存储分析历史
        self.analysis_history = []
    
    async def analyze_communications(self, communications: Dict[str, Any], round_name: str = 'initial') -> Dict[str, Any]:
        """
        分析针对该国家的所有沟通信息
        
        Args:
            communications: 沟通信息字典，包含：
                - country_to_country: 国家间沟通列表
                - eu_commission: 欧委会沟通列表
                - china_targeted: 中国针对性沟通列表
                - china_general: 中国一般性沟通列表
                - retaliation: 中国反制措施信息
            round_name: 轮次名称（'initial' 或 'final'）
            
        Returns:
            后续效应分析结果
        """
        self.logger.info(f"开始分析{self.country_id}的沟通信息（轮次：{round_name}）")
        
        # 筛选针对该国家的沟通
        targeted_communications = {
            "country_to_country": [
                comm for comm in communications.get('country_to_country', [])
                if comm.get('to') == self.country_id
            ],
            "eu_commission": communications.get('eu_commission', []),
            "china_targeted": [
                comm for comm in communications.get('china_targeted', [])
                if comm.get('to') == self.country_id
            ],
            "china_general": communications.get('china_general', []),
            "retaliation": communications.get('retaliation', {})
        }
        
        # 执行分析
        analysis_result = await self.analysis_action.run(
            country_id=self.country_id,
            country_features={"anonymized_text": self.anonymized_data.get("anonymized_text", {})},
            communications=targeted_communications,
            round_name=round_name
        )
        
        # 记录分析历史
        self.analysis_history.append(analysis_result)
        
        self.logger.info(f"完成{self.country_id}的沟通信息分析")
        
        return analysis_result
    
    def get_analysis_summary(self) -> str:
        """
        获取分析历史摘要
        """
        if not self.analysis_history:
            return "尚无分析记录"
        
        summary = f"{self.country_id}秘书分析历史（共{len(self.analysis_history)}次）：\n"
        
        for i, analysis in enumerate(self.analysis_history, 1):
            timestamp = analysis.get('timestamp', 'unknown')
            effect = analysis.get('effect_analysis', {})
            overall = effect.get('overall_impact', 'unknown')
            summary += f"{i}. {timestamp}: {overall}\n"
        
        return summary
