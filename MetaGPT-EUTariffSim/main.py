#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
欧盟对华汽车关税投票模拟系统 - 主程序入口

基于MetaGPT框架的多智能体模拟系统，用于分析欧盟成员国对华汽车关税投票的决策机制。
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from metagpt.logs import logger
from config_loader import load_config
from workflows.main_simulation_workflow import MainSimulationWorkflow


async def main():
    """主函数：运行欧盟对华汽车关税投票模拟"""
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("欧盟对华汽车关税投票模拟系统")
    logger.info("=" * 60)
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"配置加载成功: {len(config.keys())} 个配置项")
    
    try:
        # 创建主模拟工作流
        simulation_workflow = MainSimulationWorkflow(config)
        
        # 运行模拟
        logger.info("开始模拟运行...")
        results = await simulation_workflow.run()
        
        # 输出结果
        logger.info("模拟完成!")
        if 'output_directory' in results:
            logger.info(f"结果保存到: {results['output_directory']}")
        else:
            logger.info("结果已在内存中处理")
        
        # 显示关键结果
        display_key_results(results)
        
    except Exception as e:
        logger.error(f"模拟运行失败: {e}")
        logger.exception(e)
        return 1
    
    return 0


def setup_logging(config):
    """设置日志配置"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 配置根日志记录器而不是MetaGPT的logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 添加文件处理器
    log_file = log_config.get('file', './logs/simulation.log')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def display_key_results(results):
    """显示关键模拟结果"""
    logger.info("\n" + "=" * 60)
    logger.info("模拟关键结果")
    logger.info("=" * 60)
    
    if 'voting_result' in results:
        voting = results['voting_result']
        logger.info(f"投票结果: {'通过' if voting['passed'] else '未通过'}")
        logger.info(f"支持票: {voting.get('yes_votes', 0)} / 总票数: {voting.get('total_votes', 0)}")
        logger.info(f"支持率: {voting.get('yes_percentage', 0):.2%}")
        logger.info(f"阈值: {voting.get('threshold', 0):.2%}")
    
    if 'theory_analysis' in results:
        theory = results['theory_analysis']
        logger.info("\n理论解释力分析:")
        for theory_name, score in theory.get('explanatory_power', {}).items():
            logger.info(f"  {theory_name}: {score:.3f}")
    
    if 'countries' in results:
        countries = results['countries']
        logger.info(f"\n参与国家数量: {len(countries)}")
        
        # 显示立场分布
        positions = [c.get('final_position', 0) for c in countries.values()]
        avg_position = sum(positions) / len(positions) if positions else 0
        logger.info(f"平均最终立场: {avg_position:.3f}")
        
        # 显示支持/反对国家
        support_count = sum(1 for p in positions if p > 0.5)
        oppose_count = sum(1 for p in positions if p < 0.5)
        neutral_count = len(positions) - support_count - oppose_count
        logger.info(f"支持国家: {support_count}, 反对国家: {oppose_count}, 中立国家: {neutral_count}")


if __name__ == "__main__":
    # Windows系统需要设置事件循环策略
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
