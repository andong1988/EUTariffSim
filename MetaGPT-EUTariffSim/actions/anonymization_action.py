#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据匿名化动作 - 使用LLM对国家数据进行匿名化处理
"""

from typing import Dict, Any
from metagpt.actions import Action
import logging
import random
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AnonymizationAction(Action):
    """数据匿名化动作：使用LLM对国家数据进行匿名化"""
    
    name: str = "AnonymizationAction"
    desc: str = "使用LLM对国家数据进行匿名化处理的动作"
    
    def __init__(self, **kwargs):
        """
        初始化匿名化动作
        
        Args:
            **kwargs: Action参数
        """
        super().__init__(**kwargs)
        
        # 用于生成匿名代码
        self._used_codes = set()
    
    def _generate_anonymous_code(self, country_id: str) -> str:
        """
        生成匿名代码
        
        Args:
            country_id: 国家ID
            
        Returns:
            匿名代码，格式如 "A1234"
        """
        # 使用哈希值生成固定但唯一的代码
        hash_val = hash(country_id) % 10000
        code = f"A{hash_val:04d}"
        
        # 确保唯一性
        while code in self._used_codes:
            hash_val = (hash_val + 1) % 10000
            code = f"A{hash_val:04d}"
        
        self._used_codes.add(code)
        return code
    
    async def anonymize_country(self, 
                               country_id: str,
                               country_data_raw: str,
                               population: int) -> Dict[str, Any]:
        """
        对单个国家进行匿名化
        
        Args:
            country_id: 国家ID
            country_data_raw: 原始国家数据文本
            population: 国家人口
            
        Returns:
            匿名化后的数据字典
        """
        logger.info(f"开始匿名化处理: {country_id}")
        
        try:
            # 生成匿名代码
            anonymous_code = self._generate_anonymous_code(country_id)
            
            # 构建LLM提示
            prompt = self._build_anonymization_prompt(
                country_id, country_data_raw, anonymous_code
            )
            
            # 保存prompt到文件（用于调试）
            self._save_prompt_to_file(prompt, country_id)
            
            # 调用LLM生成匿名化文本
            response = await self._aask(prompt)
            
            # 解析LLM响应
            anonymized_sections = self._parse_anonymized_text(response)
            
            # 验证结果
            if not anonymized_sections:
                raise ValueError("LLM返回的匿名化文本为空或格式不正确")
            
            # 构建结果（包含原始数据结构）
            result = {
                "country_id": country_id,
                "country_name": "",  # 原始国家名称（用于记录）
                "anonymous_code": anonymous_code,
                "population": population,
                "original_dimensions": {
                    "X_market (Market / Economic Interdependence)": "",
                    "X_political (Domestic Politics and Interest Mediation)": "",
                    "X_institutional (Institutions, Diplomacy, and Path Dependence)": ""
                },
                "anonymized_dimensions": anonymized_sections,
                "anonymization_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"成功匿名化: {country_id} -> {anonymous_code}")
            return result
            
        except Exception as e:
            logger.error(f"匿名化失败 [{country_id}]: {e}", exc_info=True)
            raise
    
    def _build_anonymization_prompt(self,
                                   country_id: str,
                                   country_data_raw: str,
                                   anonymous_code: str) -> str:
        """
        构建轻度匿名化提示（三个维度版本）
        
        Args:
            country_id: 国家ID
            country_data_raw: 原始国家数据文本
            anonymous_code: 生成的匿名代码
            
        Returns:
            LLM提示文本
        """
        prompt = f"""你是一位专业的数据匿名化专家。请对以下国家数据进行轻度匿名化处理。

【国家ID】{country_id}
【匿名代码】{anonymous_code}

【原始数据】
{country_data_raw}

【核心任务】
对数据进行轻度匿名化处理：删除具体数值和识别符，但保留准确的定性描述和特征强度。

【处理规则】：

1. **必须删除的内容**：
   - 具体数字（年份、人口数值、GDP数值、贸易额）
   - 国家名称、首都城市名、著名城市名
   - 品牌名称、公司名称、产品名称（如标致、雷诺、宝马、奥迪等汽车品牌）
   - 具体地理名称（河流、山脉、半岛、具体地理位置等）
   - 具体组织名称（将"NATO"改为"跨大西洋军事联盟"，"EU"改为"区域经济组织"等）
   - 具体历史事件的时间点和具体细节

2. **保留并强化的内容**（使用准确的定性描述）：
   - 产业重要性的准确描述
     * 不是简单说"汽车产业重要"，而是描述为"汽车产业是国民经济的支柱产业，对GDP贡献显著，提供大量就业机会"
     * 描述产业规模时使用"规模庞大"、"具有重要地位"、"占主导地位"、"有一定规模但不是主要产业"等准确表述
     * 描述产业影响力时说明"对出口结构影响重大"、"在国内经济中起核心作用"等
   
   - 经济特征的准确描述
     * 将"汽车产量450万辆"改为"汽车产业规模庞大，是重要的生产基地"
     * 将"对华出口200亿欧元"改为"对中国市场出口规模较大，贸易依存度较高"
     * 将"增长率2.8%"改为"经济保持稳定增长态势"
     * 保留各产业之间的相对重要性对比
   
   - 政治特征的准确描述
     * 描述政治稳定性程度："政治稳定性非常高"、"政治稳定性中等"、"政治环境相对稳定"等
     * 描述政府-产业关系："政府与产业界关系紧密，政策协调性强"、"政府干预较多，产业政策活跃"等
     * 描述政治倾向："倾向于保护主义"、"奉行开放贸易政策"、"在经济干预和自由贸易之间寻求平衡"等
   
   - 制度和外交特征的准确描述
     * 描述对华关系："与中国建立了稳定长期的双边关系"、"对华关系务实但谨慎"、"对华关系存在一定紧张"等
     * 描述外交立场："支持务实合作"、"强调规则和制度"、"倾向于多边主义"等
     * 描述战略考量："注重战略自主"、"与盟友关系紧密"、"在地缘政治中寻求平衡"等

3. **三个维度分段输出**：
   - X_market (Market / Economic Interdependence)：准确描述产业结构、产业重要性、经济特征等（不提及具体数字）
   - X_political (Domestic Politics and Interest Mediation)：准确描述政治稳定性、政府-产业关系、政治倾向等（不提及具体政党和指数值）
   - X_institutional (Institutions, Diplomacy, and Path Dependence)：准确描述外交关系、制度安排、战略定位等（不提及具体组织名称和时间点）

【输出格式】
请严格按照以下格式输出，每个维度用【】标记：

【X_market (Market / Economic Interdependence)】
这里写市场/经济维度的轻度匿名化描述，使用准确但不含具体数字的定性描述...

【X_political (Domestic Politics and Interest Mediation)】
这里写政治维度的轻度匿名化描述...

【X_institutional (Institutions, Diplomacy, and Path Dependence)】
这里写制度/外交维度的轻度匿名化描述...

【重要注意事项】：
- 绝对不要出现原国家名称或其别称
- 尽量不要出现具体数字（包括年份）
- 绝对不要出现具体城市、河流、山脉等地理名称
- 绝对不要出现具体品牌、公司、产品名称
- 将具体组织名称用通用描述替代
- **关键是：描述要准确详细，充分体现各维度的特征和强度，不能过于模糊，要能体现国家特性**
- 只输出上述三个维度的描述，不要输出其他任何内容
- 确保每个维度都有内容，如果原始数据缺失，使用通用知识补充但保持轻度匿名化
"""
        return prompt
    
    def _parse_anonymized_text(self, response: str) -> Dict[str, str]:
        """
        解析LLM返回的匿名化文本（三个维度版本）
        
        Args:
            response: LLM响应文本
            
        Returns:
            按维度分段的匿名化文本字典
        """
        sections = {
            "X_market (Market / Economic Interdependence)": "",
            "X_political (Domestic Politics and Interest Mediation)": "",
            "X_institutional (Institutions, Diplomacy, and Path Dependence)": ""
        }
        
        import re
        
        # 尝试匹配三个维度
        patterns = {
            "X_market": r"【X_market.*?】\s*\n(.*?)(?=\n【|$)",
            "X_political": r"【X_political.*?】\s*\n(.*?)(?=\n【|$)",
            "X_institutional": r"【X_institutional.*?】\s*\n(.*?)(?=\n【|$)"
        }
        
        for section_name, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                # 找到完整的键名
                full_key = next((key for key in sections.keys() if key.startswith(section_name)), section_name)
                sections[full_key] = match.group(1).strip()
                logger.debug(f"解析到{full_key}维度: {len(sections[full_key])}字符")
            else:
                logger.warning(f"未找到{section_name}维度的内容")
        
        # 检查是否有空维度，如果有则尝试备用方法
        if all(len(content) == 0 for content in sections.values()):
            logger.warning("所有维度都为空，尝试使用整个响应")
            # 尝试将整个响应按段落分割
            paragraphs = response.split('\n\n')
            if len(paragraphs) >= 3:
                sections["X_market (Market / Economic Interdependence)"] = paragraphs[0].strip()
                sections["X_political (Domestic Politics and Interest Mediation)"] = paragraphs[1].strip()
                sections["X_institutional (Institutions, Diplomacy, and Path Dependence)"] = paragraphs[2].strip()
            else:
                # 最后的回退方案
                sections["X_market (Market / Economic Interdependence)"] = response
        
        # 验证至少有一个维度有内容
        if all(len(content) == 0 for content in sections.values()):
            raise ValueError(f"无法解析匿名化文本。原始响应:\n{response}")
        
        return sections
    
    def _save_prompt_to_file(self, prompt: str, country_id: str) -> bool:
        """
        将prompt保存到文件
        
        Args:
            prompt: prompt内容
            country_id: 国家ID
            
        Returns:
            是否保存成功
        """
        try:
            current_file = Path(__file__)
            prompts_dir = current_file.parent / "prompts" / "anonymization"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anonymization_prompt_{country_id}_{timestamp}.txt"
            filepath = prompts_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            logger.info(f"匿名化Prompt已保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存匿名化prompt文件失败: {e}", exc_info=True)
            return False
