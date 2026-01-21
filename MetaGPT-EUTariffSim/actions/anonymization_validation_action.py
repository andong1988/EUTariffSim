#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
匿名化验证动作 - 使用LLM检验匿名化效果
"""

from typing import Dict, Any, List, Optional, ClassVar
from metagpt.actions import Action
import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class AnonymizationValidationAction(Action):
    """匿名化验证动作：使用LLM猜测国家以检验匿名化效果"""
    
    name: str = "AnonymizationValidationAction"
    desc: str = "使用LLM猜测国家以检验匿名化效果的动作"
    
    # 欧洲国家列表，用于标准化国家名称
    EUROPEAN_COUNTRIES: ClassVar[List[str]] = [
        "Germany", "France", "Italy", "Spain", "Netherlands", 
        "Belgium", "Denmark", "Ireland", "Lithuania", "Poland",
        "Portugal", "Sweden", "Finland", "Norway", "Switzerland",
        "Austria", "Czech Republic", "Hungary", "Greece"
    ]
    
    def __init__(self, **kwargs):
        """
        初始化验证动作
        
        Args:
            **kwargs: Action参数
        """
        super().__init__(**kwargs)
    
    async def validate_anonymization(self, 
                                    anonymized_data_path: str = "anonymized_data.json") -> Dict[str, Any]:
        """
        验证匿名化效果
        
        Args:
            anonymized_data_path: 匿名化数据文件路径
            
        Returns:
            验证结果字典
        """
        logger.info(f"开始验证匿名化效果: {anonymized_data_path}")
        
        # 读取匿名化数据
        anonymized_data = self._load_anonymized_data(anonymized_data_path)
        
        if not anonymized_data:
            raise ValueError(f"无法读取匿名化数据文件: {anonymized_data_path}")
        
        countries = anonymized_data.get("countries", {})
        total_countries = len(countries)
        
        logger.info(f"共 {total_countries} 个国家需要验证")
        
        # 验证结果存储
        validation_results = []
        correct_count = 0
        uncertain_count = 0
        
        # 逐个国家验证
        for country_id, country_data in countries.items():
            logger.info(f"正在验证国家: {country_id}")
            
            try:
                # 构建猜测提示
                prompt = self._build_guessing_prompt(
                    country_data["anonymized_text"]
                )
                
                # 调用LLM猜测
                llm_response = await self._aask(prompt)
                
                # 解析LLM猜测
                guessed_country = self._parse_llm_guess(llm_response)
                
                # 比较猜测结果
                is_correct = self._compare_guess(guessed_country, country_id)
                
                # 记录结果
                result = {
                    "country_id": country_id,
                    "anonymous_code": country_data.get("anonymous_code"),
                    "llm_response": llm_response,
                    "guessed_country": guessed_country,
                    "is_correct": is_correct,
                    "is_uncertain": guessed_country.lower() in ["不确定", "uncertain", "unknown", "i don't know"],
                    "validation_timestamp": datetime.now().isoformat()
                }
                
                validation_results.append(result)
                
                # 统计
                if is_correct:
                    correct_count += 1
                    logger.info(f"✓ 正确识别: {country_id}")
                elif result["is_uncertain"]:
                    uncertain_count += 1
                    logger.info(f"? 不确定: {country_id} -> {guessed_country}")
                else:
                    logger.info(f"✗ 错误猜测: {country_id} -> {guessed_country}")
                
                # 添加延迟避免API限流
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"验证失败 [{country_id}]: {e}", exc_info=True)
                validation_results.append({
                    "country_id": country_id,
                    "error": str(e),
                    "validation_timestamp": datetime.now().isoformat()
                })
        
        # 生成报告
        report = self._generate_report(
            validation_results,
            total_countries,
            correct_count,
            uncertain_count
        )
        
        # 保存报告
        report_path = self._save_report(report, anonymized_data_path)
        
        logger.info(f"验证完成！报告已保存到: {report_path}")
        
        return report
    
    def _load_anonymized_data(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        加载匿名化数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            匿名化数据字典
        """
        # 尝试多个可能的路径位置
        possible_paths = [
            Path(filepath),  # 原始路径
            Path(__file__).parent.parent / filepath,  # 基于脚本位置的相对路径
            Path(__file__).parent.parent.parent / filepath,  # 项目根目录
            Path.cwd() / filepath,  # 当前工作目录
        ]
        
        for path in possible_paths:
            try:
                if path.exists() and path.is_file():
                    logger.info(f"从路径加载数据: {path}")
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    return data
            except Exception as e:
                logger.debug(f"尝试路径 {path} 失败: {e}")
                continue
        
        # 所有路径都失败
        logger.error(f"无法在以下路径找到文件: {filepath}")
        logger.error(f"已尝试的路径: {[str(p) for p in possible_paths]}")
        return None
    
    def _build_guessing_prompt(self, anonymized_text: Dict[str, str]) -> str:
        """
        构建让LLM猜国家的提示
        
        Args:
            anonymized_text: 匿名化文本（包含四个维度）
            
        Returns:
            LLM提示文本
        """
        economic = anonymized_text.get("economic", "")
        political = anonymized_text.get("political", "")
        normative = anonymized_text.get("normative", "")
        strategic = anonymized_text.get("strategic", "")
        
        prompt = """你是一位国际政治经济分析专家。请根据以下匿名化描述，推测这是哪个国家。

【经济特征】
{}

【政治特征】
{}

【规范特征】
{}

【战略特征】
{}

请严格按照以下要求回答：
1. 只回答国家名称（英文全称），可以是世界上的任何国家
2. 不要添加任何解释、理由或额外说明
3. 如果根据描述无法确定具体国家，请回答："不确定"
4. 请仅基于上述描述中的信息进行判断，不要使用其他外部知识或假设
5. 只回答国家名称或"不确定"，不要输出其他任何内容

国家名称（英文）："""
        
        return prompt.format(economic, political, normative, strategic)
    
    def _parse_llm_guess(self, llm_response: str) -> str:
        """
        解析LLM的猜测结果
        
        Args:
            llm_response: LLM原始响应
            
        Returns:
            解析后的国家名称
        """
        # 去除前后空白
        guess = llm_response.strip()
        
        # 移除引号
        guess = guess.strip('"\'')
        
        # 移除可能的前缀
        if "国家名称：" in guess:
            guess = guess.split("国家名称：")[-1].strip()
        elif "Country:" in guess:
            guess = guess.split("Country:")[-1].strip()
        elif "国家名称" in guess:
            guess = guess.split("国家名称")[-1].strip()
        
        # 移除可能的句号和其他标点
        guess = guess.rstrip('.,，。！!？?')
        
        return guess
    
    def _compare_guess(self, guess: str, actual: str) -> bool:
        """
        比较猜测结果与真实国家名
        
        Args:
            guess: LLM猜测的国家名
            actual: 真实国家名
            
        Returns:
            是否匹配
        """
        # 标准化处理
        guess_normalized = guess.strip().lower()
        actual_normalized = actual.strip().lower()
        
        # 完全匹配
        if guess_normalized == actual_normalized:
            return True
        
        # 处理常见变体
        country_aliases = {
            "netherland": ["netherlands", "the netherlands"],
            "germany": ["germany", "federal republic of germany", "deutschland"],
            "france": ["france", "french republic"],
            "italy": ["italy", "italian republic", "italia"],
            "spain": ["spain", "kingdom of spain", "españa"],
            "denmark": ["denmark", "kingdom of denmark"],
            "ireland": ["ireland", "republic of ireland"],
            "lithuania": ["lithuania", "republic of lithuania"]
        }
        
        # 检查别名
        if actual_normalized in country_aliases:
            if guess_normalized in country_aliases[actual_normalized]:
                return True
        
        # 如果guess在别名列表中，反向检查
        for actual_name, aliases in country_aliases.items():
            if guess_normalized in aliases:
                if actual_normalized == actual_name:
                    return True
        
        # 使用字符串相似度作为最后的手段
        similarity = SequenceMatcher(None, guess_normalized, actual_normalized).ratio()
        if similarity > 0.8:
            logger.info(f"使用相似度匹配: {guess} vs {actual} (相似度: {similarity:.2f})")
            return True
        
        return False
    
    def _generate_report(self, 
                        validation_results: List[Dict[str, Any]],
                        total_countries: int,
                        correct_count: int,
                        uncertain_count: int) -> Dict[str, Any]:
        """
        生成验证报告
        
        Args:
            validation_results: 验证结果列表
            total_countries: 总国家数
            correct_count: 正确识别数
            uncertain_count: 不确定数
            
        Returns:
            验证报告字典
        """
        # 计算统计指标
        error_count = total_countries - correct_count - uncertain_count
        accuracy_rate = (correct_count / total_countries * 100) if total_countries > 0 else 0
        uncertain_rate = (uncertain_count / total_countries * 100) if total_countries > 0 else 0
        error_rate = (error_count / total_countries * 100) if total_countries > 0 else 0
        
        # 分析正确识别的国家
        correct_countries = [
            r["country_id"] for r in validation_results 
            if r.get("is_correct", False)
        ]
        
        # 分析错误识别的国家
        wrong_countries = []
        for r in validation_results:
            if not r.get("is_correct", False) and not r.get("is_uncertain", False):
                wrong_countries.append({
                    "country_id": r["country_id"],
                    "guessed_country": r.get("guessed_country", "N/A"),
                    "llm_response": r.get("llm_response", "N/A")
                })
        
        # 分析不确定的国家
        uncertain_countries = [
            r["country_id"] for r in validation_results 
            if r.get("is_uncertain", False)
        ]
        
        # 构建报告
        report = {
            "summary": {
                "total_countries": total_countries,
                "correct_count": correct_count,
                "uncertain_count": uncertain_count,
                "error_count": error_count,
                "accuracy_rate": round(accuracy_rate, 2),
                "uncertain_rate": round(uncertain_rate, 2),
                "error_rate": round(error_rate, 2)
            },
            "detailed_results": validation_results,
            "analysis": {
                "correctly_identified": correct_countries,
                "incorrectly_identified": wrong_countries,
                "uncertain": uncertain_countries
            },
            "conclusion": self._generate_conclusion(
                accuracy_rate, uncertain_rate, error_rate,
                correct_countries, wrong_countries, uncertain_countries
            ),
            "report_timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_conclusion(self,
                            accuracy_rate: float,
                            uncertain_rate: float,
                            error_rate: float,
                            correct_countries: List[str],
                            wrong_countries: List[Dict],
                            uncertain_countries: List[str]) -> str:
        """
        生成结论文本
        
        Args:
            accuracy_rate: 准确率
            uncertain_rate: 不确定率
            error_rate: 错误率
            correct_countries: 正确识别的国家列表
            wrong_countries: 错误识别的国家列表
            uncertain_countries: 不确定的国家列表
            
        Returns:
            结论文本
        """
        conclusion_parts = []
        
        # 总体评估
        if accuracy_rate < 20:
            evaluation = "匿名化效果非常优秀"
        elif accuracy_rate < 40:
            evaluation = "匿名化效果优秀"
        elif accuracy_rate < 60:
            evaluation = "匿名化效果良好"
        elif accuracy_rate < 80:
            evaluation = "匿名化效果一般"
        else:
            evaluation = "匿名化效果较差，存在隐私泄露风险"
        
        conclusion_parts.append(f"【总体评估】{evaluation}")
        conclusion_parts.append(f"匿名化数据的识别准确率为 {accuracy_rate}%，")
        conclusion_parts.append(f"不确定率为 {uncertain_rate}%，错误率为 {error_rate}%。")
        
        # 正确识别分析
        if correct_countries:
            conclusion_parts.append(f"\n【成功匿名化】以下 {len(correct_countries)} 个国家成功通过匿名化测试：")
            conclusion_parts.append(f"{', '.join(correct_countries)}")
        
        # 错误识别分析
        if wrong_countries:
            conclusion_parts.append(f"\n【匿名化失败】以下 {len(wrong_countries)} 个国家被成功识别，需要加强匿名化：")
            for wc in wrong_countries:
                conclusion_parts.append(
                    f"- {wc['country_id']} 被识别为 {wc['guessed_country']}"
                )
        
        # 不确定分析
        if uncertain_countries:
            conclusion_parts.append(f"\n【模糊匿名化】以下 {len(uncertain_countries)} 个国家无法被明确识别，匿名化效果较好：")
            conclusion_parts.append(f"{', '.join(uncertain_countries)}")
        
        return "\n".join(conclusion_parts)
    
    def _save_report(self, report: Dict[str, Any], source_file: str) -> str:
        """
        保存验证报告
        
        Args:
            report: 验证报告
            source_file: 源文件路径
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建reports目录
            current_file = Path(__file__)
            reports_dir = current_file.parent.parent / "reports" / "anonymization"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
            filepath = reports_dir / filename
            
            # 保存报告
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 同时保存一个人类可读的文本报告
            txt_filename = f"validation_report_{timestamp}.txt"
            txt_filepath = reports_dir / txt_filename
            
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("匿名化效果验证报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"生成时间: {report['report_timestamp']}\n")
                f.write(f"数据源: {source_file}\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("【统计摘要】\n")
                f.write("-" * 80 + "\n")
                summary = report["summary"]
                f.write(f"总国家数: {summary['total_countries']}\n")
                f.write(f"正确识别: {summary['correct_count']} ({summary['accuracy_rate']}%)\n")
                f.write(f"不确定: {summary['uncertain_count']} ({summary['uncertain_rate']}%)\n")
                f.write(f"错误识别: {summary['error_count']} ({summary['error_rate']}%)\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("【结论】\n")
                f.write("-" * 80 + "\n")
                f.write(report["conclusion"])
                f.write("\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("【详细结果】\n")
                f.write("-" * 80 + "\n")
                for result in report["detailed_results"]:
                    if "error" in result:
                        f.write(f"国家: {result['country_id']} - 验证失败: {result['error']}\n")
                    else:
                        status = "✓" if result["is_correct"] else ("?" if result["is_uncertain"] else "✗")
                        f.write(f"{status} 国家: {result['country_id']}\n")
                        f.write(f"  LLM响应: {result['llm_response']}\n")
                        f.write(f"  猜测: {result['guessed_country']}\n\n")
            
            logger.info(f"文本报告已保存到: {txt_filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存验证报告失败: {e}", exc_info=True)
            return ""
