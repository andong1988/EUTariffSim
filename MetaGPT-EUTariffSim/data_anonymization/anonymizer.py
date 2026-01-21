"""数据匿名化管理器"""

import json
import logging
import asyncio
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from .data_loader import DataLoader
from actions.anonymization_action import AnonymizationAction

logger = logging.getLogger(__name__)


class DataAnonymizer:
    """数据匿名化管理器"""
    
    def __init__(self, eu_data_path: str = None, output_file: str = None):
        """
        初始化数据匿名化管理器
        
        Args:
            eu_data_path: eu_data.py文件路径，如果为None则使用默认路径
            output_file: 匿名化数据输出文件路径
        """
        self.data_loader = DataLoader(eu_data_path)
        
        if output_file is None:
            # 默认输出路径：data/anonymized_countries.json
            self.output_file = Path(__file__).parent.parent / "data" / "anonymized_countries.json"
        else:
            self.output_file = Path(output_file)
        
        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 匿名化动作
        self.anonymization_action = AnonymizationAction()
        
        # 匿名化结果
        self.anonymized_results = {}
    
    async def anonymize_all_countries(self, 
                                     country_list: List[str] = None,
                                     show_progress: bool = True) -> Dict[str, Any]:
        """
        对所有国家进行匿名化
        
        Args:
            country_list: 要匿名化的国家列表，如果为None则处理所有国家
            show_progress: 是否显示进度
            
        Returns:
            匿名化结果字典
        """
        # 获取要处理的国家列表
        if country_list is None:
            country_list = self.data_loader.get_country_list()
        
        logger.info(f"开始匿名化处理，共{len(country_list)}个国家")
        
        if show_progress:
            print(f"\n{'='*60}")
            print(f"开始数据匿名化处理")
            print(f"{'='*60}")
            print(f"总国家数: {len(country_list)}")
            print(f"{'='*60}\n")
        
        # 处理每个国家
        success_count = 0
        failed_countries = []
        
        for idx, country_id in enumerate(country_list, 1):
            try:
                if show_progress:
                    print(f"[{idx}/{len(country_list)}] 正在处理: {country_id}...", end=" ", flush=True)
                
                # 获取原始数据
                country_data_raw = self.data_loader.format_country_data_for_anonymization(country_id)
                population = self.data_loader.get_country_population(country_id)
                
                # 执行匿名化
                result = await self.anonymization_action.anonymize_country(
                    country_id, country_data_raw, population
                )
                
                self.anonymized_results[country_id] = result
                success_count += 1
                
                if show_progress:
                    print(f"✓ 完成 ({result['anonymous_code']})")
                
            except Exception as e:
                logger.error(f"匿名化失败 [{country_id}]: {e}")
                failed_countries.append({
                    "country_id": country_id,
                    "error": str(e)
                })
                
                if show_progress:
                    print(f"✗ 失败: {e}")
        
        # 构建最终结果
        final_result = {
            "metadata": {
                "anonymization_timestamp": datetime.now().isoformat(),
                "total_countries": len(country_list),
                "success_count": success_count,
                "failed_count": len(failed_countries),
                "source_file": str(self.data_loader.eu_data_path)
            },
            "countries": self.anonymized_results,
            "failed_countries": failed_countries
        }
        
        if show_progress:
            print(f"\n{'='*60}")
            print(f"匿名化处理完成")
            print(f"{'='*60}")
            print(f"成功: {success_count} 个国家")
            if failed_countries:
                print(f"失败: {len(failed_countries)} 个国家")
                for failed in failed_countries:
                    print(f"  - {failed['country_id']}: {failed['error']}")
            print(f"{'='*60}\n")
        
        return final_result
    
    async def anonymize_country(self, country_id: str) -> Dict[str, Any]:
        """
        对单个国家进行匿名化
        
        Args:
            country_id: 国家ID
            
        Returns:
            匿名化结果字典
        """
        logger.info(f"开始匿名化单个国家: {country_id}")
        
        # 获取原始数据
        country_data_raw = self.data_loader.format_country_data_for_anonymization(country_id)
        population = self.data_loader.get_country_population(country_id)
        
        # 执行匿名化
        result = await self.anonymization_action.anonymize_country(
            country_id, country_data_raw, population
        )
        
        self.anonymized_results[country_id] = result
        
        logger.info(f"成功匿名化: {country_id} -> {result['anonymous_code']}")
        return result
    
    def save_to_file(self, anonymization_result: Dict[str, Any] = None) -> bool:
        """
        将匿名化结果保存到文件
        
        Args:
            anonymization_result: 匿名化结果字典，如果为None则使用self.anonymized_results
            
        Returns:
            是否保存成功
        """
        try:
            if anonymization_result is None:
                # 构建默认结果格式
                anonymization_result = {
                    "metadata": {
                        "anonymization_timestamp": datetime.now().isoformat(),
                        "total_countries": len(self.anonymized_results),
                        "source_file": str(self.data_loader.eu_data_path)
                    },
                    "countries": self.anonymized_results
                }
            
            # 保存到文件
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(anonymization_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"匿名化数据已保存到: {self.output_file}")
            print(f"匿名化数据已保存到: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存匿名化数据失败: {e}")
            print(f"保存失败: {e}")
            return False
    
    def load_from_file(self) -> Dict[str, Any]:
        """
        从文件加载匿名化数据
        
        Returns:
            匿名化数据字典
        """
        try:
            if not self.output_file.exists():
                raise FileNotFoundError(f"匿名化数据文件不存在: {self.output_file}")
            
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"成功加载匿名化数据: {len(data.get('countries', {}))}个国家")
            return data
            
        except Exception as e:
            logger.error(f"加载匿名化数据失败: {e}")
            raise
    
    def get_country_anonymized_data(self, country_id: str) -> Dict[str, Any]:
        """
        获取特定国家的匿名化数据
        
        Args:
            country_id: 国家ID
            
        Returns:
            匿名化数据字典，如果未找到则返回None
        """
        if country_id in self.anonymized_results:
            return self.anonymized_results[country_id]
        return None
    
    def get_anonymization_map(self) -> Dict[str, str]:
        """
        获取国家到匿名代码的映射
        
        Returns:
            映射字典 {country_id: anonymous_code}
        """
        mapping = {}
        for country_id, data in self.anonymized_results.items():
            mapping[country_id] = data.get("anonymous_code", f"Unknown_{country_id}")
        return mapping
    
    def get_population_map(self) -> Dict[str, int]:
        """
        获取国家到人口的映射
        
        Returns:
            映射字典 {country_id: population}
        """
        mapping = {}
        for country_id, data in self.anonymized_results.items():
            mapping[country_id] = data.get("population", 0)
        return mapping
    
    def get_features_map(self) -> Dict[str, Dict[str, str]]:
        """
        获取国家到特征的映射
        
        Returns:
            映射字典 {country_id: {"economic": "...", "political": "...", ...}}
        """
        mapping = {}
        for country_id, data in self.anonymized_results.items():
            mapping[country_id] = data.get("anonymized_text", {})
        return mapping
