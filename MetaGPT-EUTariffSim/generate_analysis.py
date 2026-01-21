#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析可视化生成脚本
基于最新的模拟结果生成详细的统计图表和分析报告
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from analysis_visualization import SimulationAnalyzer

def load_latest_simulation_report():
    """加载最新的模拟报告"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return None
    
    # 查找最新的simulation_report文件
    simulation_files = [f for f in os.listdir(results_dir) if f.startswith("simulation_report_") and f.endswith(".json")]
    if not simulation_files:
        print("未找到模拟报告文件")
        return None
    
    # 按时间戳排序，获取最新的
    simulation_files.sort(reverse=True)
    latest_file = simulation_files[0]
    file_path = os.path.join(results_dir, latest_file)
    
    print(f"加载最新模拟报告: {latest_file}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_comprehensive_analysis(simulation_data):
    """生成综合分析"""
    if not simulation_data:
        print("模拟数据为空")
        return
    
    # 创建分析器
    analyzer = SimulationAnalyzer(simulation_data)
    
    # 生成所有可视化图表
    print("开始生成可视化图表...")
    
    try:
        # 1. 投票分析图表
        analyzer.create_voting_analysis_charts()
        print("✓ 投票分析图表生成完成")
        
        # 2. 理论分析图表
        analyzer.create_theory_analysis_charts()
        print("✓ 理论分析图表生成完成")
        
        # 3. 沟通分析图表
        analyzer.create_communication_analysis_charts()
        print("✓ 沟通分析图表生成完成")
        
        # 4. 准确率分析图表
        analyzer.create_accuracy_analysis_charts()
        print("✓ 准确率分析图表生成完成")
        
        # 5. 权重分析图表
        analyzer.create_weight_analysis_charts()
        print("✓ 权重分析图表生成完成")
        
        # 6. 国家对比图表
        analyzer.create_country_comparison_charts()
        print("✓ 国家对比图表生成完成")
        
        # 7. 时间序列图表
        analyzer.create_time_series_charts()
        print("✓ 时间序列图表生成完成")
        
        # 8. 综合仪表板
        analyzer.create_comprehensive_dashboard()
        print("✓ 综合仪表板生成完成")
        
        # 生成详细分析报告
        analyzer.generate_detailed_text_report()
        print("✓ 详细分析报告生成完成")
        
        print("\n🎉 所有分析图表和报告生成完成！")
        print(f"📁 结果保存在: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"生成分析图表时出错: {e}")
        import traceback
        traceback.print_exc()

def print_summary_statistics(simulation_data):
    """打印摘要统计信息"""
    if not simulation_data:
        return
    
    print("\n" + "="*60)
    print("📊 模拟结果摘要统计")
    print("="*60)
    
    # 基本信息
    metadata = simulation_data.get("simulation_metadata", {})
    print(f"🕐 模拟时间: {metadata.get('timestamp', 'N/A')}")
    print(f"⏱️  模拟时长: {metadata.get('simulation_duration', 'N/A')}")
    print(f"🌍 参与国家: {len(metadata.get('countries_participated', []))}")
    
    # 投票结果
    analysis = simulation_data.get("analysis", {})
    voting_analysis = analysis.get("voting_pattern_analysis", {})
    
    print(f"\n📈 投票结果:")
    print(f"  初始投票: 支持{voting_analysis.get('initial_distribution', {}).get('support', 0)}票, "
          f"反对{voting_analysis.get('initial_distribution', {}).get('against', 0)}票, "
          f"弃权{voting_analysis.get('initial_distribution', {}).get('abstain', 0)}票")
    print(f"  最终投票: 支持{voting_analysis.get('final_distribution', {}).get('support', 0)}票, "
          f"反对{voting_analysis.get('final_distribution', {}).get('against', 0)}票, "
          f"弃权{voting_analysis.get('final_distribution', {}).get('abstain', 0)}票")
    print(f"  立场变化率: {voting_analysis.get('change_rate', 0):.1%}")
    
    # 沟通分析
    comm_analysis = analysis.get("communication_analysis", {})
    print(f"\n💬 沟通分析:")
    print(f"  总沟通次数: {comm_analysis.get('total_communications', 0)}")
    print(f"  国家间沟通: {comm_analysis.get('country_to_country', 0)}")
    print(f"  欧委会沟通: {comm_analysis.get('eu_commission', 0)}")
    print(f"  中国反制: {'触发' if comm_analysis.get('retaliation_triggered', False) else '未触发'}")
    
    # 准确率分析
    accuracy_analysis = analysis.get("accuracy_analysis", {})
    print(f"\n🎯 准确率分析:")
    print(f"  整体准确率: {accuracy_analysis.get('overall_accuracy', 0):.1%}")
    
    # 权重优化
    weight_analysis = analysis.get("weight_optimization_analysis", {})
    print(f"\n⚖️  权重优化:")
    print(f"  优化国家数: {weight_analysis.get('countries_optimized', 0)}")
    print(f"  平均改进: {weight_analysis.get('average_improvement', 0):.3f}")
    
    print("="*60)

def main():
    """主函数"""
    print("🚀 开始生成欧盟关税模拟数据分析报告")
    print("="*60)
    
    # 加载最新模拟数据
    simulation_data = load_latest_simulation_report()
    
    if simulation_data:
        # 打印摘要统计
        print_summary_statistics(simulation_data)
        
        # 生成综合分析
        generate_comprehensive_analysis(simulation_data)
        
        print("\n✨ 分析完成！请查看生成的图表和报告文件。")
    else:
        print("❌ 无法加载模拟数据，请先运行模拟系统。")

if __name__ == "__main__":
    main()
