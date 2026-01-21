"""数据加载器 - 从eu_data.py加载原始国家数据"""

import importlib.util
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器"""
    
    def __init__(self, eu_data_path: str = None):
        """
        初始化数据加载器
        
        Args:
            eu_data_path: eu_data.py文件的路径，如果为None则使用默认路径
        """
        if eu_data_path is None:
            # 默认路径：项目根目录下的eu_data.py
            self.eu_data_path = Path(__file__).parent.parent.parent / "eu_data.py"
        else:
            self.eu_data_path = Path(eu_data_path)
        
        self.eu_data = {}
        self._load_data()
    
    def _load_data(self):
        """从eu_data.py加载数据"""
        try:
            logger.info(f"正在加载数据文件: {self.eu_data_path}")
            
            # 动态导入eu_data模块
            spec = importlib.util.spec_from_file_location("eu_data", self.eu_data_path)
            eu_data_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eu_data_module)
            
            # 获取countrys_EU字典
            if hasattr(eu_data_module, 'countrys_EU'):
                self.eu_data = eu_data_module.countrys_EU
                logger.info(f"成功加载{len(self.eu_data)}个国家的数据")
            else:
                logger.error("eu_data.py中未找到countrys_EU字典")
                raise ValueError("eu_data.py中未找到countrys_EU字典")
                
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            raise
    
    def get_all_countries(self) -> Dict[str, Any]:
        """获取所有国家数据"""
        return self.eu_data
    
    def get_country_data(self, country_id: str) -> Dict[str, Any]:
        """
        获取特定国家的数据
        
        Args:
            country_id: 国家ID（如"Germany"）
            
        Returns:
            国家数据字典
        """
        if country_id not in self.eu_data:
            logger.warning(f"未找到国家数据: {country_id}")
            return {}
        
        country_data = self.eu_data[country_id]
        
        # 添加调试日志
        logger.info(f"加载国家数据: {country_id}")
        logger.info(f"  可用的键: {list(country_data.keys())}")
        logger.info(f"  X_market 长度: {len(country_data.get('X_market (Market / Economic Interdependence)', ''))}")
        logger.info(f"  X_political 长度: {len(country_data.get('X_political (Domestic Politics and Interest Mediation)', ''))}")
        logger.info(f"  X_institutional 长度: {len(country_data.get('X_institutional (Institutions, Diplomacy, and Path Dependence)', ''))}")
        
        return country_data
    
    def get_country_list(self) -> list:
        """获取国家ID列表"""
        return list(self.eu_data.keys())
    
    def get_country_population(self, country_id: str) -> int:
        """
        获取国家人口
        
        Args:
            country_id: 国家ID
            
        Returns:
            人口数量，如果未找到则返回0
        """
        country_data = self.get_country_data(country_id)
        return country_data.get("population", 0)
    
    def format_country_data_for_anonymization(self, country_id: str) -> str:
        """
        格式化国家数据以便匿名化处理（新格式：三个维度）
        
        Args:
            country_id: 国家ID
            
        Returns:
            格式化的文本
        """
        country_data = self.get_country_data(country_id)
        
        if not country_data:
            return ""
        
        # 直接提取三个维度
        x_market = country_data.get("X_market (Market / Economic Interdependence)", "")
        x_political = country_data.get("X_political (Domestic Politics and Interest Mediation)", "")
        x_institutional = country_data.get("X_institutional (Institutions, Diplomacy, and Path Dependence)", "")
        
        text = f"""【X_market (Market / Economic Interdependence)】
{x_market}

【X_political (Domestic Politics and Interest Mediation)】
{x_political}

【X_institutional (Institutions, Diplomacy, and Path Dependence)】
{x_institutional}
"""
        return text.strip()
