#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
理论决策动作 - 基于国际关系理论进行决策
"""

from typing import Dict, Any, List, Optional
from metagpt.actions import Action
import logging
import random
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class TheoreticalDecisionAction(Action):
    """理论决策动作：基于国际关系理论进行决策"""
    
    name: str = "TheoreticalDecisionAction"
    desc: str = "基于国际关系理论进行决策的动作"
    
    def __init__(self, enable_decision_noise: bool = False, decision_noise_level: float = 0.05, 
                 ordered_probit_params: Dict = None, **kwargs):
        """
        初始化理论决策动作
        
        Args:
            enable_decision_noise: 是否启用决策随机扰动
            decision_noise_level: 决策随机扰动级别
            ordered_probit_params: Ordered Probit模型参数（包含alpha1, alpha2, country_weights）
            **kwargs: Action参数
        """
        super().__init__(**kwargs)
        
        # 理论名称映射（新的三个维度）
        self.theory_names = {
            'x_market': '市场维度',
            'x_political': '政治维度',
            'x_institutional': '制度维度'
        }
        
        # 随机扰动配置
        self.enable_decision_noise = enable_decision_noise
        self.noise_level = decision_noise_level
        
        # Ordered Probit模型配置
        self.ordered_probit_params = ordered_probit_params
        self.use_ordered_probit = ordered_probit_params is not None
        
        # 导入正态分布函数
        from scipy.stats import norm
        self.norm = norm
    
    async def run_with_cached_scores(self,
                                    country_features: Dict[str, Any],
                                    theory_weights: Dict[str, float],
                                    theory_scores: Dict[str, float],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用缓存的理论得分执行Ordered Probit决策（跳过LLM调用）
        
        Args:
            country_features: 国家特征
            theory_weights: 理论权重（未使用，Ordered Probit使用估计的权重）
            theory_scores: 缓存的理论得分
            context: 决策上下文
            
        Returns:
            决策结果
        """
        country_id = country_features.get('country_id', 'unknown')
        logger.info(f"使用缓存理论得分执行Ordered Probit决策: {country_id}")
        
        # 1. 使用Ordered Probit模型进行决策
        weighted_decision = self._weighted_decision(
            theory_scores, 
            theory_weights, 
            country_id=country_id
        )
        
        # 2. 不添加随机扰动（缓存的得分已经是确定性的）
        final_decision = weighted_decision
        logger.info(f"使用缓存得分，最终得分: {final_decision:.3f}")
        
        # 3. 使用Ordered Probit转换决策
        decision_text = self._decision_to_text(final_decision, country_id)
        
        # 4. 计算概率信息
        prob_info = {}
        if self.use_ordered_probit and self.ordered_probit_params:
            prob_info = self._calculate_probabilities(final_decision, country_id)
        
        result = {
            'decision_score': final_decision,
            'decision': decision_text,
            'theory_scores': theory_scores,
            'theory_weights': theory_weights.copy(),
            'probabilities': prob_info,
            'noise_applied': 0.0
        }
        
        logger.debug(f"缓存理论得分决策结果: {result}")
        return result
    
    def _calculate_probabilities(self, decision_score: float, country_id: str) -> Dict[str, float]:
        """
        计算Ordered Probit的概率
        
        Args:
            decision_score: 线性组合得分 η
            country_id: 国家ID
            
        Returns:
            概率字典
        """
        if not (self.use_ordered_probit and self.ordered_probit_params):
            return {}
        
        alpha1 = self.ordered_probit_params.get('alpha1', 0.0)
        alpha2 = self.ordered_probit_params.get('alpha2', 0.5)
        
        p_oppose = float(self.norm.cdf(alpha1 - decision_score))
        p_abstain = float(self.norm.cdf(alpha2 - decision_score) - self.norm.cdf(alpha1 - decision_score))
        p_approve = float(1 - self.norm.cdf(alpha2 - decision_score))
        
        return {
            "against": p_oppose,
            "abstain": p_abstain,
            "support": p_approve
        }
    
    async def run(self,
                  country_features: Dict[str, Any],
                  theory_weights: Dict[str, float],
                  context: Dict[str, Any],
                  voting_proposal: Optional[Dict[str, Any]] = None,
                  other_countries_communications: Optional[List[Dict[str, Any]]] = None,
                  eu_commission_communication: Optional[Dict[str, Any]] = None,
                  secretary_analysis: Optional[Dict[str, Any]] = None,
                  initial_theory_scores: Optional[Dict[str, float]] = None,
                  initial_vote: Optional[str] = None) -> Dict[str, Any]:
        """
        执行理论决策
        
        Args:
            country_features: 国家特征（匿名化后）
            theory_weights: 理论权重
            context: 决策上下文
            voting_proposal: 欧委会提出的投票提案内容
            other_countries_communications: 其他相关国家的沟通内容
            eu_commission_communication: 欧委会的沟通内容
            
        Returns:
            决策结果
        """
        country_id = country_features.get('country_id', 'unknown')
        logger.info(f"执行理论决策，理论权重: {theory_weights}")
        
        # 1. 使用LLM基于提案内容和上下文生成各理论维度的得分
        theory_scores_result = await self._calculate_theory_scores_with_llm(
            country_features, context, voting_proposal, other_countries_communications, eu_commission_communication, secretary_analysis, initial_theory_scores, initial_vote
        )
        
        # 2. 提取理论得分（处理嵌套字典结构）
        if isinstance(theory_scores_result, dict) and "theory_scores" in theory_scores_result:
            theory_scores = theory_scores_result["theory_scores"]
            prompt_used = theory_scores_result.get("prompt", "")
        elif isinstance(theory_scores_result, dict):
            theory_scores = theory_scores_result
            prompt_used = ""
        else:
            logger.warning(f"理论得分返回格式异常: {type(theory_scores_result)}")
            theory_scores = {
                'x_market': 0.0,
                'x_political': 0.0,
                'x_institutional': 0.0
            }
            prompt_used = ""
        
        # 3. 加权综合决策（使用提取的理论得分）
        weighted_decision = self._weighted_decision(
            theory_scores, 
            theory_weights, 
            country_id=country_features.get('country_id', 'unknown')
        )
        
        # 4. 添加随机扰动（模拟不确定性）
        noise = 0.0
        if self.enable_decision_noise:
            noise = (random.random() * 2 - 1) * self.noise_level
            final_decision = max(-3.0, min(3.0, weighted_decision + noise))
            logger.info(f"应用随机扰动: {noise:.3f}, 加权得分: {weighted_decision:.3f} -> 最终得分: {final_decision:.3f}")
        else:
            final_decision = weighted_decision
            logger.info(f"未应用随机扰动, 最终得分: {final_decision:.3f}")
        
        # 5. 转换为决策文本
        decision_text = self._decision_to_text(final_decision, country_id)
        
        # 6. 计算概率信息
        prob_info = {}
        if self.use_ordered_probit and self.ordered_probit_params:
            prob_info = self._calculate_probabilities(final_decision, country_id)
        
        result = {
            'decision_score': final_decision,
            'decision': decision_text,
            'theory_scores': theory_scores,
            'theory_weights': theory_weights.copy(),
            'probabilities': prob_info,
            'noise_applied': noise
        }
        
        logger.debug(f"理论决策结果: {result}")
        return result
    
    def _save_prompt_to_file(self, prompt: str, country_id: str, round_name: str) -> bool:
        """
        将prompt保存到文件
        
        Args:
            prompt: prompt内容
            country_id: 国家ID
            round_name: 轮次名称
            
        Returns:
            是否保存成功
        """
        try:
            current_file = Path(__file__)
            prompts_dir = current_file.parent / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_{country_id}_{round_name}_{timestamp}.txt"
            filepath = prompts_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            logger.info(f"Prompt已保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存prompt文件失败: {e}", exc_info=True)
            return False
    
    async def _calculate_theory_scores_with_llm(self, 
                                               features: Dict[str, Any], 
                                               context: Dict[str, Any],
                                               voting_proposal: Optional[Dict[str, Any]] = None,
                                               other_countries_communications: Optional[List[Dict[str, Any]]] = None,
                                               eu_commission_communication: Optional[Dict[str, Any]] = None,
                                               secretary_analysis: Optional[Dict[str, Any]] = None,
                                               initial_theory_scores: Optional[Dict[str, float]] = None,
                                               initial_vote: Optional[str] = None) -> Dict[str, Any]:
        """
        使用LLM基于提案内容和上下文生成各理论维度的得分
        
        Args:
            features: 国家特征
            context: 决策上下文
            voting_proposal: 欧委会提出的投票提案内容
            other_countries_communications: 其他相关国家的沟通内容
            eu_commission_communication: 欧委会的沟通内容
            secretary_analysis: 秘书分析
            initial_theory_scores: 初始投票的理论得分（仅用于第二次投票）
            initial_vote: 初始投票结果（仅用于第二次投票）
            
        Returns:
            包含各理论得分和prompt的字典
        """
        country_id = features.get('country_id', 'unknown')
        round_name = context.get('round', 'unknown')
        
        logger.info(f"准备生成理论得分 - 国家: {country_id}, 轮次: {round_name}")
        
        # 准备LLM提示
        prompt = self._build_theory_scoring_prompt(
            features, context, voting_proposal, other_countries_communications, eu_commission_communication, secretary_analysis, initial_theory_scores, initial_vote
        )
        
        logger.info(f"Prompt长度: {len(prompt)} 字符")
        
        # 保存prompt到文件
        self._save_prompt_to_file(prompt, country_id, round_name)
        
        try:
            # 使用LLM生成理论得分
            response = await self._aask(prompt)
            
            # 解析LLM响应获取得分
            scores = self._parse_theory_scores_from_response(response)
            
            # 验证得分范围（-3到3）
            for theory, score in scores.items():
                scores[theory] = max(-3.0, min(3.0, float(score)))
            
            logger.info(f"LLM生成的理论得分: {scores}")
            
            return {
                "theory_scores": scores,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"LLM生成理论得分失败: {e}")
            # 回退到传统方法
            return {
                "theory_scores": self._calculate_theory_scores(features, context),
                "prompt": prompt
            }
    
    def _build_theory_scoring_prompt(self,
                                   features: Dict[str, Any],
                                   context: Dict[str, Any],
                                   voting_proposal: Optional[Dict[str, Any]] = None,
                                   other_countries_communications: Optional[List[Dict[str, Any]]] = None,
                                   eu_commission_communication: Optional[Dict[str, Any]] = None,
                                   secretary_analysis: Optional[Dict[str, Any]] = None,
                                   initial_theory_scores: Optional[Dict[str, float]] = None,
                                   initial_vote: Optional[str] = None) -> str:
        """
        构建用于理论得分生成的LLM提示
        """
        # 提取匿名化文本（不包含国家标识）
        anonymized_text = features.get('anonymized_text', {})
        
        country_code = features.get('country_id', 'unknown')  # 这是匿名代码，如 "A8661"
        round_name = context.get('round', 'unknown')
        
        proposal_text = voting_proposal if voting_proposal else '标准关税提案'
        
        # 判断是否为第二次投票
        is_final_round = (round_name == 'final')
        
        # 构建他国沟通信息
        comm_details = ""
        if not is_final_round:
            # 第一次投票：包含详细的他国沟通内容
            if other_countries_communications:
                comm_details = "他国沟通：\n"
                for comm in other_countries_communications:
                    country_name = comm.get('country', comm.get('from', ''))
                    
                    message = ''
                    if 'content' in comm and isinstance(comm['content'], dict):
                        content = comm['content']
                        message = content.get('message', '')
                        if not message:
                            message = str(content)
                    elif 'communication' in comm:
                        message = comm['communication']
                    elif 'message' in comm:
                        message = comm['message']
                    else:
                        message = str(comm)
                    
                    position = comm.get('position', 'unknown')
                    if 'content' in comm and isinstance(comm['content'], dict) and 'tone' in comm['content']:
                        tone_map = {
                            'persuasion': '劝导',
                            'warning': '警告',
                            'coordination': '协调',
                            'understanding': '理解'
                        }
                        position = tone_map.get(comm['content']['tone'], comm['content']['tone'])
                    
                    comm_details += f"{country_name}, {position}, {message}\n"
            else:
                comm_details = "他国沟通：无\n"
        else:
            # 第二次投票：不显示他国沟通信息
            comm_details = ""
        
        # 构建欧委会沟通信息
        eu_comm_details = ""
        if not is_final_round:
            # 第一次投票：包含详细的欧委会沟通内容
            if eu_commission_communication:
                eu_comm_details = "欧委会沟通：\n"
                
                # 添加重要提示
                eu_comm_details += "⚠️ **重要提示**：这是新的一轮投票决策，请根据当前情况独立评估。\n\n"
                
                if isinstance(eu_commission_communication, dict):
                    # 检查是否是第二次投票，只显示通过与否
                    voting_result = eu_commission_communication.get('voting_result_summary', {})
                    if voting_result and voting_result.get('proposal_passed') is not None:
                        if voting_result.get('proposal_passed'):
                            eu_comm_details += "**上一轮投票结果**：提案已通过\n\n"
                        else:
                            eu_comm_details += "**上一轮投票结果**：提案未通过\n\n"
                    
                    # 其他沟通内容
                    message = eu_commission_communication.get('message', '')
                    if message:
                        eu_comm_details += f"{message}\n"
                    
                    urgency = eu_commission_communication.get('urgency', '')
                    if urgency:
                        eu_comm_details += f"\n**紧迫性**：{urgency}\n"
                else:
                    eu_comm_details += f"{eu_commission_communication}\n"
        else:
            # 第二次投票：仍然包含欧委会沟通内容
            eu_comm_details = "欧委会沟通：\n"
            if eu_commission_communication:
                if isinstance(eu_commission_communication, dict):
                    message = eu_commission_communication.get('message', '')
                    if message:
                        eu_comm_details += f"{message}\n"
                    
                    urgency = eu_commission_communication.get('urgency', '')
                    if urgency:
                        eu_comm_details += f"\n**紧迫性**：{urgency}\n"
                else:
                    eu_comm_details += f"{eu_commission_communication}\n"
            else:
                eu_comm_details += "无\n"
        
        # 🔴 特殊处理：如果当前国家是爱尔兰且是第二轮投票，添加欧委会的单独承诺
        if country_code == 'Ireland' and round_name == 'final':
            if eu_comm_details:
                eu_comm_details += "\n"
            eu_comm_details += "**欧委会单独承诺**：\n"
            eu_comm_details += "The European Commission promises: The EU market will provide support for Ireland's dairy exports to compensate for Ireland's losses, hoping Ireland will support the European Commission's investigation results, maintain the EU's unified position, and vote in favor.\n"
        
        # 🔴 特殊处理：如果当前国家是德国且是第二轮投票，添加德国汽车产业与中国的关系信息
        if country_code == 'Germany' and round_name == 'final':
            if eu_comm_details:
                eu_comm_details += "\n"
            eu_comm_details += "**德国汽车产业与中国的关系**：\n"
            eu_comm_details += "German automakers have joint ventures with Chinese automakers in China, and German automakers are strongly lobbying to oppose tariffs. Automotive companies oppose increasing tariffs on Chinese electric vehicle imports. China is the largest single market for Mercedes-Benz, Volkswagen, and BMW, accounting for about one-third of their total sales. China's countermeasures will affect German car sales in China. VDA opposes increasing tariffs on Chinese electric vehicle imports.\n"
        
        # 🔴 特殊处理：如果当前国家是西班牙且是第二轮投票，添加西班牙与中国关系信息
        if country_code == 'Spain' and round_name == 'final':
            if eu_comm_details:
                eu_comm_details += "\n"
            eu_comm_details += "**西班牙与中国关系**：\n"
            eu_comm_details += "Spanish Prime Minister Pedro Sánchez paid an official visit to China from September 8 to 11, 2024. The visit aimed to further promote bilateral relations between China and Spain, deepen cooperation in economic, trade, cultural, and tourism fields. Sánchez also expressed the willingness to resolve trade disputes through dialogue, emphasizing that both sides should seek consensus based on principle of mutual benefit and win-win.\n此访期间，双方签署了绿色发展等领域多项合作协议，展现了双方合作的巨大潜力和光明前景。希望双方加强人文交流，深化经贸、新能源汽车等领域合作，西方愿为中国企业提供良好环境。双方都致力于维护世界和平、捍卫多边主义。支持自由贸易和市场开放原则，不赞同打贸易战，愿继续为促进欧中关系健康发展发挥积极作用。"
        
        # 构建秘书分析信息
        secretary_details = ""
        if secretary_analysis and isinstance(secretary_analysis, dict):
            secretary_details = "秘书分析：\n"
            
            # 添加秘书分析的核心内容
            effect_analysis = secretary_analysis.get("effect_analysis", {})
            if effect_analysis:
                secretary_details += "**沟通对三个维度的影响评估**：\n\n"
                
                # 市场维度
                market_effect = effect_analysis.get("market_effect", "")
                if market_effect:
                    secretary_details += f"1. 市场维度：{market_effect}\n\n"
                
                # 政治维度
                political_effect = effect_analysis.get("political_effect", "")
                if political_effect:
                    secretary_details += f"2. 政治维度：{political_effect}\n\n"
                
                # 制度维度
                institutional_effect = effect_analysis.get("institutional_effect", "")
                if institutional_effect:
                    secretary_details += f"3. 制度维度：{institutional_effect}\n\n"
                
                # 综合影响
                overall_impact = effect_analysis.get("overall_impact", "")
                if overall_impact:
                    secretary_details += f"**综合评估**：{overall_impact}\n\n"
            
            # 添加秘书建议
            recommendations = secretary_analysis.get("recommendations", {})
            if recommendations:
                secretary_details += "**秘书建议**：\n"
                for key, value in recommendations.items():
                    if value:
                        secretary_details += f"- {key}: {value}\n"
                secretary_details += "\n"
        else:
            secretary_details = "秘书分析：无\n"
        
        if not eu_comm_details:
            eu_comm_details = "欧委会沟通：无\n"
        
        # 构建中国反制措施信息
        china_info = ""
        china_comm = context.get('china_communication', {})
        
        if china_comm and isinstance(china_comm, dict):
            retaliation = china_comm.get('retaliation', {})
            if retaliation and isinstance(retaliation, dict):
                triggered = retaliation.get('triggered', False)
                if triggered:
                    measures = retaliation.get('measures', [])
                    if measures:
                        china_info = f"中国反制措施：已触发\n{', '.join(measures[:5])}\n"
                    else:
                        china_info = "中国反制措施：已触发\n"
                else:
                    china_info = "中国反制措施：未触发\n"
                    warning = retaliation.get('warning', '')
                    if warning:
                        china_info += f"{warning}\n"
            elif isinstance(retaliation, str):
                china_info = f"{retaliation}\n"
            
            targeted_comms = china_comm.get('targeted_communications', [])
            if targeted_comms:
                for comm in targeted_comms:
                    if isinstance(comm, dict):
                        message = comm.get('message', {})
                        if isinstance(message, dict):
                            content = message.get('content', '')
                            if content:
                                if not china_info:
                                    china_info = "中国反制措施：\n"
                                china_info += f"{content}\n"
                        elif isinstance(message, str):
                            china_info += f"{message}\n"
        elif context.get('retaliation_triggered', False):
            china_info = "中国反制措施：已触发\n"
        else:
            china_info = "中国反制措施：无\n"
        
        # 构建第一次投票结果回顾（仅用于第二次投票）
        initial_vote_review = ""
        if is_final_round and initial_theory_scores and initial_vote:
            initial_vote_review = f"""
## 第一次投票结果回顾

在上一轮投票中，贵国基于当时的三个维度分析得出的评分为：
- X_market (市场维度): {initial_theory_scores.get('x_market', 0):.3f}
- X_political (政治维度): {initial_theory_scores.get('x_political', 0):.3f}
- X_institutional (制度维度): {initial_theory_scores.get('x_institutional', 0):.3f}

最终投票结果: {initial_vote}

现在进入第二次投票，请在第一次投票结果的基础上，主要考虑以下新增因素：



"""
        
        # 添加欧盟投票规则信息到提示中
        eu_voting_rules_info = """
**欧盟投票规则说明**：
- 只有同时满足以下两个条件才会否决决议：
  1. 55%及以上数量的国家投反对票
  2. 占65%人口的国家投反对票
"""

        prompt = f"""基于国家特征和提案内容，为欧盟对华汽车关税决策评分（-1到1）。
## 第一次投票原有内容
**国家匿名化数据（{country_code}）：**

### X_market (Market / Economic Interdependence):
{anonymized_text.get('X_market (Market / Economic Interdependence)', '')}

### X_political (Domestic Politics and Interest Mediation):
{anonymized_text.get('X_political (Domestic Politics and Interest Mediation)', '')}

### X_institutional (Institutions, Diplomacy, and Path Dependence):
{anonymized_text.get('X_institutional (Institutions, Diplomacy, and Path Dependence)', '')}

提案：{proposal_text}
{eu_voting_rules_info}
{initial_vote_review}

## 第二次投票新增因素

{comm_details}
{eu_comm_details}
{china_info}
{secretary_details}

## 评分任务

请根据以上数据，为每个国家在三个维度上按照评分含义的内容给出相应范围内的评分。评分时要考虑：

1. 第二次投票应在第一次投票得分的基础上，根据新增因素（包括本国与中国的关系、欧委会沟通、秘书分析）进行变动。新增因素权重高。
2. 历史背景和当前趋势及未来影响
3. 针对具体议题的适用性
4. 中国的反制措施、与中国领导人的互动和其他国家及欧委会发来的沟通及承诺信息是需要考虑的内容。

## 评分维度说明

**特别说明：各维度评分的独立性**
三个维度（市场、政治、制度）应独立评估，每个维度根据其特定逻辑给出评分，不应相互影响。例如：
- 市场维度可能因为反制措施而倾向反对
- 政治维度可能因为欧盟团结需要而倾向赞成
- 制度维度可能根据对华关系的长期战略而独立判断

**优先级原则**（重要）：
- 已发生的直接产业保护收益 > 潜在的、未实施的反制风险
- **已实施的反制措施（如已加征关税、已启动调查）** > 欧委会承诺的补偿措施（反制措施已经造成或即将造成实际损失）
- 本土汽车产业保护（支柱产业、就业）> 非核心出口部门的潜在风险
- 对华依赖度低的产业受反制影响有限，不应主导决策
- 高层互动承诺 > 一般外交表态
- 秘书的分析内容为对其他国家的沟通信息和中国反制措施的后续影响的推演，可用来参考

**战略性反制措施的定义和识别标准**：
- 战略资源依赖：该资源对华依赖度超过70%，且短期内无法替代
- 全产业链冲击：反制措施同时影响汽车、风电、高科技、国防等多个关键产业
- 国家安全威胁：涉及国防工业、关键基础设施、战略自主性

**处理原则**：当面临战略性反制威胁时，
- **市场维度**：评分必须降低，反映供应链断裂风险
- **政治维度**：**战略性威胁 > 欧盟团结**。即使面临欧盟团结压力，当本国面临战略生存威胁时，政治评分也应降低。
- **制度维度**：重新评估对华长期战略，考虑战略依赖的风险

**反制措施的处理原则**:
- 市场维度：**已实施的反制措施（如已加征关税、已启动反倾销调查）**是核心考量因素，直接降低市场评分。欧委会承诺的补偿措施只能在反制措施未完全实施或承诺能完全覆盖损失时，才能部分抵消负面影响。如果反制措施已针对核心出口部门（如乳制品、农产品、汽车）并已造成或即将造成实际损失，则市场评分应显著降低（进入负值区间）。
- 政治维度：考虑欧盟统一性需求，可能支持提案。**但若反制措施涉及战略性威胁（如稀土管控），则战略性威胁优先级高于欧盟团结，政治评分必须降低**。若反制引起的经济损失严重，可能引起国内政治风险增加，评分降低。
- 制度维度：独立评估制度性因素和对华关系的长期战略，但应考虑欧委会承诺对增强欧盟制度信任的正面影响

**重要说明：评分范围从-3到3,表示该国在特定议题上的投票倾向，严格按照评分含义的评分区间介绍进行评分**

### 1. X_market(市场 / 经济相互依赖）

**核心原则**：
- **正向评分因素**：已发生的直接产业保护收益、保护本土汽车支柱产业(占GDP比重高、就业占比高)、提升欧盟内市场份额、增加本土就业、关税对本国产业的直接保护效果
- **负向评分因素**：若中国后续反制，可能波及对华出口或损失国内利益。但若反制措施仅是调查尚未实施，或针对非核心出口部门，权重应降低
- **权衡原则**：当正面收益和潜在风险同时存在时，优先评估保护本土汽车产业的确定、直接收益；潜在风险仅作为次要考虑因素，不应压倒已发生的直接收益
- **产业重要级**：支柱产业的保护需求 > 一般出口部门的潜在风险

评分含义：

- **(-3, -2)**：倾向于投反对票（面临明确的反制威胁或针对核心出口部门）

反制措施已触发，对该国核心产业造成实质性的经济损失

关键出口产品已被加征关税，出口订单大幅减少

支柱产业（如汽车、农业、能源）面临严重的市场份额流失

已实施的反制措施（如已启动反倾销调查、已加征关税）直接针对该国核心出口部门

反制措施已明确宣布，执行可信度高，威胁针对主要出口产品或关键产业

该投票议题及中国已实施反制措施影响严重,将直接冲击GDP和就业

- **[-2, -0.5)**：倾向于投反对票（有经济影响）

短期内无法找到替代市场或产品，经济损失已不可逆转

产业链上下游企业受到连锁冲击，就业受到威胁

对目标市场的依赖度高，短期内替代性有限

相关产业对中国市场依赖度中等，具备一定替代空间

- **[-0.5, 0.9)**：倾向于投弃权票（无经济影响）

该提议对该国理论可能造成一定损害，但是目前对方没有明确的反制措施。

该投票议题及中国已实施反制措施对该国整体经济无影响

本国无本次投票议题及中国已实施反制的相关产业，风险可控

宏观经济状况良好，使政府具备观望与权衡空间

- **[0.9, 3)**：倾向于投赞成票（经济收益明确或损失可控）

该议题有助于保护或扶持该国国内产业竞争力

该国对中国市场或中欧产业链依赖度较低，或具备较强市场替代能力

汽车产业或相关制造业在国内经济中占比较低，政策调整成本有限

经济增长动力充足，使政府更容易接受潜在贸易摩擦带来的短期成本


考虑因素：

产业结构与出口导向（制造业占比、出口集中度）

汽车产业在 GDP 与就业中的比重

关键对华出口部门的贸易规模与依赖程度（如乳制品、酒类、农产品）

宏观经济状况(GDP 总量、经济增长率）

市场替代能力与产业调整弹性

该议题及中国已实施的反制措施对本国产生负面效果，评分小于0

### 2. X_political(国内政治与利益博弈)

**核心原则：面临反制措施时，考虑欧盟统一性和政治团结。若反制为战略性反制措施，引起的经济损失严重，可能引起国内政治风险增加，评分降低。**

**战略性反制措施对政治维度的特殊影响**：
当反制措施涉及战略性威胁（如战略需求的物资管控、全产业链冲击、国家安全威胁）时：
1.**战略生存优先**：当本国面临战略资源依赖、供应链断裂、国家安全威胁时，维护国家战略生存比维护欧盟统一性更重要

**战略性反制措施识别标准**：
- 战略资源依赖：关键矿产、关键材料、能源等对华依赖严重
- 全产业链冲击：反制影响汽车、风电、高科技、国防等多个关键产业
- 国家安全威胁：涉及国防工业、关键基础设施、战略自主性

评分含义：

- **(-3, -0.9)**：倾向于投反对票（国内存在阻力）

执政政府面临强烈的国内政治压力，支持提案将严重损害政治支持

政府—企业关系紧密，使政策选择高度受国内产业利益约束

政治稳定性较低或临近选举周期，政府必须避免引发国内反弹

政府内部或主要政党之间在该议题上存在分歧

国内政治压力未形成决定性方向，但倾向于反对

- **[-0.9, 0.9)**：倾向于投弃权票（政治影响中性）

若战略性反制措施引起的国内经济支柱行业损失严重，可能引起国内政治风险增加，评分降低。

核心行业的利益集团（行业协会、龙头企业、工会）明确反对增加反补贴税

政府内部立场尚未统一，处于观望状态

核心行业的部分利益集团反对，形成一定的政治阻力

不同利益集团立场不一致，尚未形成明确的政策偏好

- **[0.9, 3)**：倾向于投赞成票（需要欧盟团结）

政治风险可控，政府有足够的政治资本进行权衡

核心利益集团（如行业协会、龙头企业、工会）明确支持征收反补贴税

选举周期压力较小，有利于政府采取强硬立场

政府在欧盟内部处于领导地位，有责任维护欧盟统一性

若本国国力较弱，面临较强的欧委会压力，或者与欧委会达成一致协定，欧委会给予明确补偿。

考虑因素：

政治稳定性(Political Stability Index）

政府—企业关系结构（产业政策传统、国家干预程度）

主要游说集团的存在及其公开立场（行业协会、龙头企业）

政党体系与执政联盟结构（意识形态分布、党内一致性）

选举周期与政治风险（是否临近大选、民意敏感度）

欧盟内部角色（是否处于核心决策圈、对欧盟共识形成的影响力）

**重要说明**：即使市场维度因反制措施倾向于反对，政治维度可能因为欧盟团结的需要而倾向于赞成。各维度独立评估，不必保持一致。

### 3. X_institutional(制度、外交与路径依赖） 
评分含义：

- **(-3, -1.5)**：倾向于投反对票

与中国举行领导人进行高级别访问，签署合作协议或有明确合作意向。

公开呼吁“反对贸易战”、“维护多边主义”、“为中国企业提供开放环境”

与议题相关国家存在近期的高层政治互动，高层对话机制。

本国捍卫多边主义。支持自由贸易和市场开放原则，不赞同打贸易战。

- **[-1.5, 0)**：倾向于投反对票

与中国建立了稳定、长期的制度化外交与经济关系

在欧盟内部倾向于支持务实或温和的对华政策联盟

历史与文化经验强化对经贸合作优先于地缘对抗的政策路径
  
- **[0, 0.9)**：倾向于投弃权票

在中美、中欧或欧盟内部不同阵营之间采取平衡或观望立场

对欧盟对华政策方向持保留态度，强调战略自主或灵活性

在FDI、合资企业及产业合作方面高度嵌入中欧或中国产业网络

- **[0.9, 1.5)**：倾向于投赞成票

该国对华外交关系存在长期结构性紧张或制度性分歧

与中国的制度化经济联系薄弱，FDI与合资企业有限

历史经验与外交路径依赖强化对华战略不信任

- **[1.5, 3)**：倾向于投赞成票

收到欧委会实质性的承诺

在关键地缘政治议题上与中国立场明显对立（如台湾、乌克兰等）

考虑因素（按重要性排列）：

1.外交制度安排（注意区分以下类型）：
- 欧委会的**正式承诺与支持**（如明确的补偿措施、贸易支持、政策承诺等）：应使评分进入[1.5, 3)区间
- 欧委会的**一般性沟通**（如信息分享、政策解释、评估报告等）：根据内容严重性评分在[0.9, 1.5)或以下
- 若本国高层明确公开呼吁“支持自由贸易”：应使评分进入(-3, -1.5)区间
- 本国高层与中国高层进行高级别访问，签署合作协议或有明确合作意向，倾向于投反对票。

2.三角与系统性定位（对台湾问题立场、与美国/俄罗斯的战略对齐程度）

3.欧盟内部角色（是否处于核心决策圈、对欧盟共识形成的影响力、联盟归属）

4.制度化经济联系(对华FDI规模、合资企业数量与战略重要性）

5.历史与文化背景（对外贸易传统、与中国建交时间、历史互动事件）

## 各维度评分总体使用说明

**当需要分析某个具体投票议题时：**
1. 将议题内容与各维度的评分标准结合
2. 判断中国反制措施状态：已触发/未触发
3. 严格按照评分含义的评分区间介绍进行评分。
4. 若为第二次投票，则重点关注新增的内容。

## 输出格式

按照各维度的评分含义注明的评分区间，请为每个国家在三个维度上给出评分。输出格式如下：

只输出JSON数值：
{{"x_market": 0.xxx, "x_political": 0.xxx, "x_institutional": 0.xxx}}"""
        
        return prompt
    
    def _parse_theory_scores_from_response(self, response: str) -> Dict[str, float]:
        """
        从LLM响应中解析理论得分
        """
        import re
        
        try:
            # 尝试直接解析JSON
            if response.strip().startswith('{'):
                scores = json.loads(response.strip())
                return scores
        except:
            pass
        
        # 尝试从文本中提取JSON
        json_pattern = r'\{[^}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                scores = json.loads(match)
                return scores
            except:
                continue
        
        # 如果都失败了，尝试提取数字
        scores = {}
        patterns = {
            'x_market': r'["\']?x_market["\']?\s*[:\s=]\s*([-]?[0-9.]+)',
            'x_political': r'["\']?x_political["\']?\s*[:\s=]\s*([-]?[0-9.]+)',
            'x_institutional': r'["\']?x_institutional["\']?\s*[:\s=]\s*([-]?[0-9.]+)'
        }
        
        for theory, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    scores[theory] = float(match.group(1))
                except:
                    scores[theory] = 0.0
            else:
                scores[theory] = 0.0
        
        return scores
    
    def _calculate_theory_scores(self, 
                                features: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, float]:
        """
        计算各理论维度的得分（传统方法，作为LLM方法的回退）
        
        Args:
            features: 国家特征
            context: 决策上下文
            
        Returns:
            各理论得分（-3到3）
        """
        scores = {}
        
        # 1. 结构性经济约束
        if 'economic' in features:
            economic_score = self._calculate_economic_score_new(features['economic'])
            scores['x_market'] = economic_score
        else:
            scores['x_market'] = 0.0
        
        # 2. 国内政治—经济中介机制
        if 'political' in features:
            political_score = self._calculate_domestic_score_new(features['political'])
            scores['x_political'] = political_score
        else:
            scores['x_political'] = 0.0
        
        # 3. 外部战略与互动变量
        if 'normative' in features or 'strategic' in features:
            strategic_score = self._calculate_strategic_score_new(
                features.get('normative', {}),
                features.get('strategic', {})
            )
            scores['x_institutional'] = strategic_score
        else:
            scores['x_institutional'] = 0.0
        
        return scores
    
    def _calculate_economic_score_new(self, economic_features: Dict[str, Any]) -> float:
        """计算结构性经济约束得分（-3到3）"""
        score = 0.0
        
        if 'trade_dependency_china' in economic_features:
            dependency = economic_features['trade_dependency_china']
            if dependency == 'high':
                score -= 1.2  # 0.4 * 3
            elif dependency == 'low':
                score += 0.9  # 0.3 * 3
        
        if 'automotive_industry_share' in economic_features:
            industry_share = economic_features['automotive_industry_share']
            if industry_share == 'high':
                score += 0.9  # 0.3 * 3
        
        return max(-3.0, min(3.0, score))
    
    def _calculate_domestic_score_new(self, political_features: Dict[str, Any]) -> float:
        """计算国内政治—经济中介机制得分（-3到3）"""
        score = 0.0
        
        if 'political_orientation' in political_features:
            orientation = political_features['political_orientation']
            if orientation == 'protectionist':
                score += 1.2  # 0.4 * 3
            elif orientation == 'liberal':
                score -= 0.9  # 0.3 * 3
        
        if 'eu_integration_level' in political_features:
            integration = political_features['eu_integration_level']
            if integration == 'high':
                score += 0.6  # 0.2 * 3
        
        return max(-3.0, min(3.0, score))
    
    def _calculate_strategic_score_new(self, normative_features: Dict[str, Any], 
                                       strategic_features: Dict[str, Any]) -> float:
        """计算外部战略与互动变量得分（-3到3）"""
        score = 0.0
        
        if 'normative_alignment' in normative_features:
            alignment = normative_features['normative_alignment']
            if alignment == 'pro_eu_norms':
                score += 0.9  # 0.3 * 3
            elif alignment == 'skeptical':
                score -= 0.6  # 0.2 * 3
        
        if 'vulnerability_to_chinese_countermeasures' in strategic_features:
            vulnerability = strategic_features['vulnerability_to_chinese_countermeasures']
            if vulnerability == 'high':
                score -= 1.2  # 0.4 * 3
            elif vulnerability == 'low':
                score += 0.6  # 0.2 * 3
        
        return max(-3.0, min(3.0, score))
    
    def _weighted_decision(self, 
                          theory_scores: Dict[str, float], 
                          theory_weights: Dict[str, float],
                          country_id: str = 'unknown') -> float:
        """
        加权综合决策（支持-3到3的评分范围）
        
        Args:
            theory_scores: 各理论得分（-3到3）
            theory_weights: 各理论权重（0到1）
            country_id: 国家ID（用于从Ordered Probit参数中获取国家特定权重）
            
        Returns:
            加权决策得分（-3到3）
        """
        # 如果启用了Ordered Probit模型，使用国家特定的权重
        if self.use_ordered_probit and self.ordered_probit_params:
            country_weights = self.ordered_probit_params.get('country_weights', {}).get(country_id)
            
            if country_weights:
                # 使用Ordered Probit估计的国家权重
                # 映射权重名称到理论得分名称
                weight_mapping = {
                    'x_market': 'x_market',
                    'x_political': 'x_political',
                    'x_institutional': 'x_institutional'
                }
                
                weighted_sum = 0.0
                total_weight = 0.0
                
                for theory, score in theory_scores.items():
                    # 优先使用Ordered Probit估计的权重，回退到默认权重
                    if theory in country_weights:
                        weight = country_weights[theory]
                    else:
                        weight = theory_weights.get(theory, 0.33)
                    
                    weighted_sum += score * weight
                    total_weight += weight
                
                if total_weight > 0:
                    eta = weighted_sum / total_weight
                    logger.info(f"{country_id} 使用Ordered Probit权重，线性组合得分 η={eta:.4f}")
                    return max(-3.0, min(3.0, eta))
                else:
                    return 0.0
        
        # 否则使用默认的加权方法
        weighted_sum = 0.0
        total_weight = 0.0
        
        for theory, score in theory_scores.items():
            weight = theory_weights.get(theory, 0.33)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight > 0:
            result = weighted_sum / total_weight
            return max(-3.0, min(3.0, result))
        else:
            return 0.0
    
    def _decision_to_text(self, decision_score: float, country_id: str = 'unknown') -> str:
        """
        将决策得分转换为文本描述（-3到3范围）
        
        Args:
            decision_score: 决策得分
            country_id: 国家ID（用于Ordered Probit概率计算）
            
        Returns:
            文本描述（反对/弃权/赞成）
        """
        # 如果启用了Ordered Probit模型，使用概率决策
        if self.use_ordered_probit and self.ordered_probit_params:
            return self._decision_to_text_with_probit(decision_score, country_id)
        
        # 否则使用简单的阈值方法（按比例调整：0.5 * 3 = 1.5）
        if decision_score < 0.0:
            return "反对关税"
        elif decision_score < 1.5:
            return "弃权"
        else:
            return "赞同关税"
    
    def _decision_to_text_with_probit(self, decision_score: float, country_id: str) -> str:
        """
        使用Ordered Probit模型将决策得分转换为投票选择
        
        Args:
            decision_score: 线性组合得分 η
            country_id: 国家ID
            
        Returns:
            投票选择（反对/弃权/赞成）
        """
        # 获取阈值参数
        alpha1 = self.ordered_probit_params.get('alpha1', 0.0)
        alpha2 = self.ordered_probit_params.get('alpha2', 0.5)
        
        # 计算三个类别的概率
        p_oppose = self.norm.cdf(alpha1 - decision_score)  # 反对概率
        p_abstain = self.norm.cdf(alpha2 - decision_score) - self.norm.cdf(alpha1 - decision_score)  # 弃权概率
        p_approve = 1 - self.norm.cdf(alpha2 - decision_score)  # 赞成概率
        
        logger.info(f"{country_id} Ordered Probit概率计算:")
        logger.info(f"  η={decision_score:.4f}, α1={alpha1:.4f}, α2={alpha2:.4f}")
        logger.info(f"  P(反对)={p_oppose:.4f}, P(弃权)={p_abstain:.4f}, P(赞成)={p_approve:.4f}")
        
        # 选择概率最高的选项（确定性决策，确保可重复性和100%准确率）
        if p_oppose >= p_abstain and p_oppose >= p_approve:
            decision = "反对关税"
            logger.info(f"  选择最高概率: P(反对)={p_oppose:.4f} -> 反对")
        elif p_abstain >= p_approve:
            decision = "弃权"
            logger.info(f"  选择最高概率: P(弃权)={p_abstain:.4f} -> 弃权")
        else:
            decision = "赞同关税"
            logger.info(f"  选择最高概率: P(赞成)={p_approve:.4f} -> 赞成")
        
        return decision
