#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
选择沟通目标动作 - 使用LLM智能选择沟通目标
"""

from typing import Dict, Any, List, Optional
from metagpt.actions import Action
import logging
import json
import re

logger = logging.getLogger(__name__)


class SelectCommunicationTargetsAction(Action):
    """选择沟通目标动作：使用LLM智能选择沟通目标"""
    
    name: str = "SelectCommunicationTargetsAction"
    desc: str = "使用LLM智能选择沟通目标的动作"
    
    def __init__(self, **kwargs):
        """
        初始化选择沟通目标动作
        
        Args:
            **kwargs: Action参数
        """
        super().__init__(**kwargs)
    
    async def select_targets(
        self,
        initiator_id: str,
        initiator_stance: str,
        initiator_last_vote: str,
        initiator_anonymized_text: Dict,
        all_countries_info: Dict,
        retaliation: Dict,
        n: int = 3
    ) -> List[str]:
        """
        使用LLM选择沟通目标
        
        Args:
            initiator_id: 发起方国家ID
            initiator_stance: 发起方当前立场
            initiator_last_vote: 发起方上次投票
            initiator_anonymized_text: 发起方的三维特征文本
            all_countries_info: 所有国家的信息 {country_id: {"last_vote": stance, "anonymized_text": text}}
            retaliation: 中国反制措施信息
            n: 需要选择的国家数量
            
        Returns:
            选择的国家ID列表
        """
        logger.info(f"{initiator_id} 使用LLM选择{n}个沟通目标")
        
        # 1. 构建LLM提示
        prompt = self._build_selection_prompt(
            initiator_id, initiator_stance, initiator_last_vote, initiator_anonymized_text,
            all_countries_info, retaliation, n
        )
        
        logger.debug(f"LLM选择目标的Prompt长度: {len(prompt)} 字符")
        
        # 2. 调用LLM获取选择结果
        try:
            response = await self._aask(prompt)
            logger.info(f"LLM响应: {response[:200]}...")
            
            # 3. 解析响应获取国家列表
            selected_countries = self._parse_selected_countries(response, all_countries_info.keys())
            
            # 4. 验证选择结果
            if len(selected_countries) > n:
                logger.warning(f"LLM返回了{n}个国家以上，截取前{n}个")
                selected_countries = selected_countries[:n]
            
            if len(selected_countries) < n:
                logger.warning(f"LLM只返回了{len(selected_countries)}个国家，未达到{n}个")
            
            logger.info(f"{initiator_id} LLM选择的沟通目标: {selected_countries}")
            return selected_countries
            
        except Exception as e:
            logger.error(f"LLM选择沟通目标失败: {e}")
            # 回退到简单的随机选择
            return self._fallback_selection(initiator_id, all_countries_info.keys(), n)
    
    def _build_selection_prompt(
        self,
        initiator_id: str,
        initiator_stance: str,
        initiator_last_vote: str,
        initiator_anonymized_text: Dict,
        all_countries_info: Dict,
        retaliation: Dict,
        n: int
    ) -> str:
        """
        构建用于选择沟通目标的LLM提示
        
        Args:
            initiator_id: 发起方国家ID
            initiator_stance: 发起方当前立场
            initiator_last_vote: 发起方上次投票
            initiator_anonymized_text: 发起方的三维特征文本
            all_countries_info: 所有国家的信息
            retaliation: 中国反制措施信息
            n: 需要选择的国家数量
            
        Returns:
            LLM提示字符串
        """
        # 投票立场映射
        vote_mapping = {
            "support": "支持",
            "against": "反对",
            "abstain": "弃权",
            "neutral": "中立"
        }
        
        # 提取本国三维特征文本
        market_text = initiator_anonymized_text.get("X_market (Market / Economic Interdependence)", "")
        political_text = initiator_anonymized_text.get("X_political (Domestic Politics and Interest Mediation)", "")
        institutional_text = initiator_anonymized_text.get("X_institutional (Institutions, Diplomacy, and Path Dependence)", "")
        
        # 构建中国反制措施信息
        retaliation_info = ""
        if retaliation and retaliation.get("triggered", False):
            measures = retaliation.get("measures", [])
            target_countries = retaliation.get("target_countries", [])
            severity = retaliation.get("severity", "")
            
            retaliation_info = f"""
【中国反制措施】
⚠️ 中国已宣布将采取以下反制措施：
- 严重程度：{severity}
- 针对国家：{', '.join(target_countries) if target_countries else '无特定目标'}
- 具体措施：{', '.join(measures[:3]) if measures else '无具体措施'}
"""
        else:
            retaliation_info = """
【中国反制措施】
目前中国尚未采取明确的反制措施，但存在潜在的报复风险。
"""
        
        # 构建其他国家信息
        other_countries_info = ""
        for country_id, country_data in all_countries_info.items():
            if country_id == initiator_id:
                continue
            
            last_vote = country_data.get("last_vote", "unknown")
            country_text = country_data.get("anonymized_text", {})
            
            # 提取该国关键特征（简化版）
            market_summary = country_text.get("X_market (Market / Economic Interdependence)", "")[:150]
            political_summary = country_text.get("X_political (Domestic Politics and Interest Mediation)", "")[:150]
            institutional_summary = country_text.get("X_institutional (Institutions, Diplomacy, and Path Dependence)", "")[:150]
            
            other_countries_info += f"""
---
【{country_id}】
上次投票：{vote_mapping.get(last_vote, last_vote)}
市场维度：{market_summary}...
政治维度：{political_summary}...
制度维度：{institutional_summary}...
"""
        
        # 构建完整提示
        prompt = f"""你是一位经验丰富的欧盟外交官，代表{initiator_id}选择{n}个国家进行双边沟通，目标是影响他们在对华汽车关税投票中的立场。

【本国（{initiator_id}）情况】

上次投票：{vote_mapping.get(initiator_last_vote, initiator_last_vote)}
当前立场：{vote_mapping.get(initiator_stance, initiator_stance)}

三维特征：
1. 市场维度：{market_text}

2. 政治维度：{political_text}

3. 制度维度：{institutional_text}

{retaliation_info}

【其他可选国家情况】
{other_countries_info}

【选择标准】
1. **战略相关性**：选择与本国利益相关、可能影响投票结果的国家
2. **立场改变可能性**：选择立场可被说服的国家，而非立场已固化的国家
3. **联盟形成**：考虑组建支持联盟或争取关键摇摆国家
4. **影响力平衡**：平衡不同影响力的国家，避免过度集中
5. **反制措施考虑**：考虑中国反制措施对各国的不同影响，针对性选择

【沟通目标】
希望说服对方支持本国的立场（{vote_mapping.get(initiator_stance, initiator_stance)}）。

【选择要求】
1. 从上述可选国家中选择不大于{n}个国家进行沟通，可以小于。
2. 不要选择本国（{initiator_id}）
3. 尽量选择与本国意见不同的国家进行沟通
4. 优先选择能够被说服或对投票结果有关键影响的国家
5. 只返回不大于{n}个国家ID，不要重复

【输出格式】
只返回JSON数组，包含不大于{n}个国家ID：
["CountryID1", "CountryID2", "CountryID3"]
"""
        return prompt
    
    def _parse_selected_countries(self, response: str, valid_country_ids: set) -> List[str]:
        """
        从LLM响应中解析选择的国家列表
        
        Args:
            response: LLM响应文本
            valid_country_ids: 有效的国家ID集合
            
        Returns:
            国家ID列表
        """
        selected_countries = []
        
        try:
            # 尝试直接解析JSON
            if response.strip().startswith('['):
                countries = json.loads(response.strip())
                if isinstance(countries, list):
                    selected_countries = countries
            elif response.strip().startswith('{'):
                # 可能是JSON对象，查找数组
                data = json.loads(response.strip())
                if isinstance(data, list):
                    selected_countries = data
                elif "countries" in data:
                    selected_countries = data["countries"]
        except:
            pass
        
        # 如果JSON解析失败，尝试提取国家名称
        if not selected_countries:
            # 查找所有可能的国家ID（假设国家ID是英文单词）
            words = re.findall(r'\b[A-Z][a-zA-Z]+\b', response)
            for word in words:
                if word in valid_country_ids:
                    selected_countries.append(word)
        
        # 去重并过滤无效国家ID
        seen = set()
        result = []
        for country in selected_countries:
            if country in valid_country_ids and country not in seen:
                seen.add(country)
                result.append(country)
        
        return result
    
    def _fallback_selection(self, initiator_id: str, all_country_ids: set, n: int) -> List[str]:
        """
        回退选择方法（当LLM失败时使用）
        
        Args:
            initiator_id: 发起方国家ID
            all_country_ids: 所有国家ID集合
            n: 需要选择的国家数量
            
        Returns:
            国家ID列表
        """
        import random
        
        # 排除本国
        available_countries = [cid for cid in all_country_ids if cid != initiator_id]
        
        # 随机选择n个国家（如果可用国家不足则选择所有）
        if len(available_countries) <= n:
            return available_countries
        else:
            return random.sample(available_countries, n)
