#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
匿名化验证运行脚本 - 运行匿名化效果检验
"""

import asyncio
import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from metagpt.logs import logger
from actions.anonymization_validation_action import AnonymizationValidationAction


async def main():
    """
    主函数：运行匿名化验证
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("开始运行匿名化效果验证")
    logger.info("=" * 80)
    
    try:
        # 创建验证Action
        validation_action = AnonymizationValidationAction()
        
        # 执行验证
        # 使用相对于项目根目录的路径
        data_path = "anonymized_data.json"
        
        logger.info(f"正在读取匿名化数据: {data_path}")
        report = await validation_action.validate_anonymization(data_path)
        
        # 输出摘要
        logger.info("\n" + "=" * 80)
        logger.info("验证完成！摘要：")
        logger.info("=" * 80)
        
        summary = report["summary"]
        logger.info(f"总国家数: {summary['total_countries']}")
        logger.info(f"正确识别: {summary['correct_count']} ({summary['accuracy_rate']}%)")
        logger.info(f"不确定: {summary['uncertain_count']} ({summary['uncertain_rate']}%)")
        logger.info(f"错误识别: {summary['error_count']} ({summary['error_rate']}%)")
        
        logger.info("\n" + report["conclusion"])
        
        logger.info("\n详细报告已保存到 reports/anonymization/ 目录")
        
    except Exception as e:
        logger.error(f"验证过程出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
