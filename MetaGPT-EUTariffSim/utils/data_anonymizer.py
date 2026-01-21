"""数据匿名化处理器"""

import json
import hashlib
import logging
from typing import Dict


class DataAnonymizer:
    """数据匿名化处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataAnonymizer")
        self.anonymization_cache = {}
    
    def anonymize_country_data(self, country_data: Dict) -> Dict:
        """匿名化国家数据"""
        country_id = country_data.get("country_name", "unknown")
        
        # 检查缓存
        data_hash = self._calculate_data_hash(country_data)
        if data_hash in self.anonymization_cache:
            return self.anonymization_cache[data_hash]
        
        anonymized = {
            "country_id": self._generate_anonymous_id(country_id),
            "economic_features": self._anonymize_economic_data(country_data),
            "political_features": self._anonymize_political_data(country_data),
            "normative_features": self._anonymize_normative_data(country_data),
            "strategic_features": self._anonymize_strategic_data(country_data),
            "original_hash": data_hash
        }
        
        # 缓存结果
        self.anonymization_cache[data_hash] = anonymized
        
        self.logger.info(f"国家数据匿名化完成: {country_id} -> {anonymized['country_id']}")
        return anonymized
    
    def _calculate_data_hash(self, data: Dict) -> str:
        """计算数据哈希值"""
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()[:8]
    
    def _generate_anonymous_id(self, country_id: str) -> str:
        """生成匿名ID"""
        return f"Country_{hash(country_id) % 10000:04d}"
    
    def _anonymize_economic_data(self, data: Dict) -> Dict:
        """匿名化经济数据 - 增强版本，包含更细致的行业影响分析"""
        economic_text = data.get("government_interests", "") + data.get("domestic_Economic_Pressures", "")
        
        # 提取关键经济特征
        features = {
            "trade_dependency_china": self._extract_trade_dependency(economic_text),
            "automotive_industry_share": self._extract_automotive_share(economic_text),
            "gdp_level": self._extract_gdp_level(economic_text),
            "economic_stability": self._extract_economic_stability(economic_text),
            # 新增：对提案行业的具体影响
            "automotive_industry_exposure": self._extract_automotive_exposure(economic_text),
            "supply_chain_vulnerability": self._extract_supply_chain_vulnerability(economic_text),
            "employment_in_automotive": self._extract_automotive_employment(economic_text),
            "r_d_investment_dependency": self._extract_rd_dependency(economic_text),
            "market_access_importance": self._extract_market_access_importance(economic_text)
        }
        
        return features
    
    def _anonymize_political_data(self, data: Dict) -> Dict:
        """匿名化政治数据"""
        political_text = data.get("political_and_Public_Opinion", "") + data.get("european_relations", "")
        
        features = {
            "political_orientation": self._extract_political_orientation(political_text),
            "eu_integration_level": self._extract_eu_integration(political_text),
            "government_stability": self._extract_government_stability(political_text)
        }
        
        return features
    
    def _anonymize_normative_data(self, data: Dict) -> Dict:
        """匿名化规范数据"""
        normative_text = data.get("historical_and_Cultural_Background", "") + data.get("diplomatic_Relations", "")
        
        features = {
            "normative_alignment": self._extract_normative_alignment(normative_text),
            "historical_ally_alignment": self._extract_historical_alignment(normative_text),
            "values_priority": self._extract_values_priority(normative_text)
        }
        
        return features
    
    def _anonymize_strategic_data(self, data: Dict) -> Dict:
        """匿名化战略数据"""
        strategic_text = data.get("international_situation", "") + data.get("government_interests", "")
        
        features = {
            "vulnerability_to_chinese_countermeasures": self._extract_vulnerability(strategic_text),
            "strategic_autonomy_priority": self._extract_strategic_autonomy(strategic_text),
            "technological_competition_stance": self._extract_tech_competition_stance(strategic_text)
        }
        
        return features
    
    def _extract_trade_dependency(self, text: str) -> str:
        """提取对华贸易依赖度"""
        text_lower = text.lower()
        if "高度依赖" in text_lower or "high dependency" in text_lower:
            return "high"
        elif "低依赖" in text_lower or "low dependency" in text_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_automotive_share(self, text: str) -> str:
        """提取汽车产业份额"""
        text_lower = text.lower()
        if "支柱产业" in text_lower or "重要支柱" in text_lower or "24%" in text_lower:
            return "high"
        elif "较小" in text_lower or "minor" in text_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_gdp_level(self, text: str) -> str:
        """提取GDP水平"""
        text_lower = text.lower()
        if "万亿" in text_lower or "trillion" in text_lower:
            return "high"
        elif "千亿" in text_lower or "billion" in text_lower:
            return "medium"
        else:
            return "low"
    
    def _extract_economic_stability(self, text: str) -> str:
        """提取经济稳定性"""
        text_lower = text.lower()
        if "稳定" in text_lower or "stable" in text_lower:
            return "high"
        elif "波动" in text_lower or "volatile" in text_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_political_orientation(self, text: str) -> str:
        """提取政治倾向"""
        text_lower = text.lower()
        if "左" in text_lower or "left" in text_lower:
            return "left"
        elif "右" in text_lower or "right" in text_lower:
            return "right"
        else:
            return "centrist"
    
    def _extract_eu_integration(self, text: str) -> str:
        """提取欧盟一体化程度"""
        text_lower = text.lower()
        if "高度一体化" in text_lower or "high integration" in text_lower:
            return "high"
        elif "怀疑" in text_lower or "skeptical" in text_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_government_stability(self, text: str) -> str:
        """提取政府稳定性"""
        text_lower = text.lower()
        if "高稳定性" in text_lower or "high stability" in text_lower:
            return "high"
        elif "动荡" in text_lower or "unstable" in text_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_normative_alignment(self, text: str) -> str:
        """提取规范对齐"""
        text_lower = text.lower()
        if "亲欧盟" in text_lower or "pro_eu" in text_lower:
            return "pro_eu_norms"
        elif "怀疑" in text_lower or "skeptical" in text_lower:
            return "skeptical"
        else:
            return "neutral"
    
    def _extract_historical_alignment(self, text: str) -> str:
        """提取历史盟友对齐"""
        text_lower = text.lower()
        if "西方" in text_lower or "western" in text_lower:
            return "western"
        elif "中立" in text_lower or "neutral" in text_lower:
            return "neutral"
        else:
            return "independent"
    
    def _extract_values_priority(self, text: str) -> str:
        """提取价值观优先级"""
        text_lower = text.lower()
        if "人权" in text_lower or "human rights" in text_lower:
            return "human_rights"
        elif "经济" in text_lower or "economic" in text_lower:
            return "economic"
        else:
            return "balanced"
    
    def _extract_vulnerability(self, text: str) -> str:
        """提取脆弱性"""
        text_lower = text.lower()
        if "高脆弱" in text_lower or "high vulnerability" in text_lower:
            return "high"
        elif "低脆弱" in text_lower or "low vulnerability" in text_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_strategic_autonomy(self, text: str) -> str:
        """提取战略自主性"""
        text_lower = text.lower()
        if "高战略自主" in text_lower or "high autonomy" in text_lower:
            return "high"
        elif "低战略自主" in text_lower or "low autonomy" in text_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_tech_competition_stance(self, text: str) -> str:
        """提取技术竞争立场"""
        text_lower = text.lower()
        if "竞争" in text_lower or "competitive" in text_lower:
            return "competitive"
        elif "合作" in text_lower or "cooperative" in text_lower:
            return "cooperative"
        else:
            return "neutral"
    
    # 新增：对提案行业的具体影响分析方法
    def _extract_automotive_exposure(self, text: str) -> str:
        """提取汽车产业暴露度"""
        text_lower = text.lower()
        if "汽车" in text_lower or "automotive" in text_lower:
            if "支柱" in text_lower or "pillar" in text_lower or "核心" in text_lower:
                return "critical"
            elif "重要" in text_lower or "important" in text_lower or "major" in text_lower:
                return "high"
            elif "一定" in text_lower or "certain" in text_lower or "moderate" in text_lower:
                return "moderate"
            else:
                return "low"
        else:
            return "minimal"
    
    def _extract_supply_chain_vulnerability(self, text: str) -> str:
        """提取供应链脆弱性"""
        text_lower = text.lower()
        if "供应链" in text_lower or "supply chain" in text_lower:
            if "高度依赖" in text_lower or "highly dependent" in text_lower or "严重依赖" in text_lower:
                return "severe"
            elif "依赖" in text_lower or "dependent" in text_lower:
                return "moderate"
            elif "部分依赖" in text_lower or "partially dependent" in text_lower:
                return "limited"
            else:
                return "minimal"
        else:
            return "unknown"
    
    def _extract_automotive_employment(self, text: str) -> str:
        """提取汽车产业就业影响"""
        text_lower = text.lower()
        if "就业" in text_lower or "employment" in text_lower or "工作" in text_lower:
            if "大量" in text_lower or "significant" in text_lower or "major" in text_lower:
                return "high"
            elif "一定" in text_lower or "moderate" in text_lower or "certain" in text_lower:
                return "moderate"
            elif "少量" in text_lower or "minor" in text_lower or "limited" in text_lower:
                return "low"
            else:
                return "minimal"
        else:
            return "unknown"
    
    def _extract_rd_dependency(self, text: str) -> str:
        """提取研发依赖度"""
        text_lower = text.lower()
        if "研发" in text_lower or "r&d" in text_lower or "research" in text_lower:
            if "高度依赖" in text_lower or "highly dependent" in text_lower:
                return "high"
            elif "合作" in text_lower or "cooperation" in text_lower or "collaborative" in text_lower:
                return "moderate"
            elif "独立" in text_lower or "independent" in text_lower:
                return "low"
            else:
                return "minimal"
        else:
            return "unknown"
    
    def _extract_market_access_importance(self, text: str) -> str:
        """提取市场准入重要性"""
        text_lower = text.lower()
        if "市场" in text_lower or "market" in text_lower:
            if "关键" in text_lower or "critical" in text_lower or "vital" in text_lower:
                return "critical"
            elif "重要" in text_lower or "important" in text_lower or "significant" in text_lower:
                return "high"
            elif "一定" in text_lower or "moderate" in text_lower:
                return "moderate"
            else:
                return "low"
        else:
            return "minimal"
