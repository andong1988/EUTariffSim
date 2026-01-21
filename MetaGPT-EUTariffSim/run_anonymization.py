#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据匿名化运行脚本 - 独立运行数据匿名化模块
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from data_anonymization import DataAnonymizer
from utils.config import SimulationConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("欧盟对华汽车关税模拟系统 - 数据匿名化模块")
    print("="*60 + "\n")
    
    # 加载配置
    try:
        config = SimulationConfig()
        logger.info("成功加载配置")
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        print(f"错误：加载配置失败 - {e}")
        return
    
    # 检查匿名化是否启用
    try:
        import yaml
        config_file = Path(__file__).parent / "config" / "config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            anonymization_config = config_data.get('data_anonymization', {})
            is_enabled = anonymization_config.get('enabled', False)
            
            if not is_enabled:
                print("⚠️  匿名化功能未启用")
                print("请在 config/config.yaml 中将 data_anonymization.enabled 设置为 true")
                print("\n当前配置:")
                print(f"  enabled: {is_enabled}")
                return
            
            output_file = anonymization_config.get('output_file', 'data/anonymized_countries.json')
            country_list = config.selected_countries
            
            print(f"✓ 匿名化功能已启用")
            print(f"  输出文件: {output_file}")
            print(f"  处理国家: {country_list if country_list else '所有国家'}")
            print()
            
        else:
            print("⚠️  配置文件不存在: config/config.yaml")
            print("使用默认配置...")
            output_file = None
            country_list = None
            
    except Exception as e:
        logger.warning(f"读取配置文件失败，使用默认配置: {e}")
        output_file = None
        country_list = None
    
    # 创建匿名化器
    try:
        anonymizer = DataAnonymizer(
            eu_data_path=None,  # 使用默认路径
            output_file=output_file
        )
        print(f"✓ 初始化匿名化器")
        print()
    except Exception as e:
        logger.error(f"初始化匿名化器失败: {e}")
        print(f"错误：初始化失败 - {e}")
        return
    
    # 执行匿名化
    try:
        print("开始执行数据匿名化...")
        print("="*60)
        
        result = await anonymizer.anonymize_all_countries(
            country_list=country_list,
            show_progress=True
        )
        
        print("="*60)
        print()
        
        # 保存结果
        if anonymizer.save_to_file(result):
            print("✓ 匿名化数据已成功保存")
        else:
            print("✗ 保存匿名化数据失败")
        
        # 显示摘要
        print("\n" + "="*60)
        print("匿名化摘要")
        print("="*60)
        print(f"总国家数: {result['metadata']['total_countries']}")
        print(f"成功: {result['metadata']['success_count']}")
        print(f"失败: {result['metadata']['failed_count']}")
        
        if result['failed_countries']:
            print("\n失败的国家:")
            for failed in result['failed_countries']:
                print(f"  - {failed['country_id']}: {failed['error']}")
        
        # 显示匿名化映射
        print("\n匿名化映射:")
        for country_id, data in result['countries'].items():
            print(f"  {country_id} -> {data['anonymous_code']}")
        
        print("="*60)
        
        if result['metadata']['failed_count'] > 0:
            print("\n⚠️  部分国家匿名化失败，请检查错误信息")
            print("您可以手动检查失败的国家，或修正问题后重新运行")
        else:
            print("\n✓ 所有国家匿名化成功！")
            print("现在可以运行主模拟系统了")
        
    except Exception as e:
        logger.error(f"匿名化过程失败: {e}", exc_info=True)
        print(f"\n错误：匿名化失败 - {e}")
        print("\n请检查:")
        print("  1. LLM配置是否正确")
        print("  2. 网络连接是否正常")
        print("  3. eu_data.py文件是否存在且格式正确")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
