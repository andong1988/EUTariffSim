"""
手动阈值配置示例

演示如何使用手动阈值控制功能进行Ordered Probit模型优化
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.ordered_probit_analysis import OrderedProbitAnalysis
from utils.ordered_probit_mle import OrderedProbitMLE


def example_auto_thresholds():
    """
    示例1: 使用自动优化阈值（默认行为）
    """
    print("="*80)
    print("示例1: 自动优化阈值模式")
    print("="*80)
    
    # 创建分析器（默认使用自动阈值优化）
    analyzer = OrderedProbitAnalysis(
        cache_dir="../cache",
        vote_data_path="../real_vote.xlsx",
        results_dir="results/probit_auto"
    )
    
    # 运行完整分析
    analyzer.run_complete_analysis()
    
    print("\n✓ 自动阈值模式分析完成")


def example_manual_thresholds():
    """
    示例2: 使用手动配置阈值
    """
    print("\n" + "="*80)
    print("示例2: 手动阈值控制模式")
    print("="*80)
    
    # 创建分析器，使用手动阈值
    analyzer = OrderedProbitAnalysis(
        cache_dir="../cache",
        vote_data_path="../real_vote.xlsx",
        results_dir="results/probit_manual"
    )
    
    # 修改MLE模型的初始化，使用手动阈值
    n_countries = len(analyzer.countries)
    analyzer.mle_model = OrderedProbitMLE(
        n_countries=n_countries,
        n_theories=3,
        manual_thresholds=True,
        manual_alpha1=-1.5,  # 手动配置α₁
        manual_alpha2=1.5    # 手动配置α₂
    )
    
    # 运行完整分析
    analyzer.run_complete_analysis()
    
    print("\n✓ 手动阈值模式分析完成")


def example_invalid_thresholds():
    """
    示例3: 演示无效的手动阈值配置
    """
    print("\n" + "="*80)
    print("示例3: 无效的手动阈值配置")
    print("="*80)
    
    try:
        # 尝试创建一个无效的手动阈值配置（α₂ <= α₁）
        mle = OrderedProbitMLE(
            n_countries=5,
            n_theories=3,
            manual_thresholds=True,
            manual_alpha1=0.5,
            manual_alpha2=0.3  # 无效：α₂必须大于α₁
        )
        print("❌ 错误：应该抛出异常但没有")
    except ValueError as e:
        print(f"✓ 正确捕获到异常: {e}")
    
    try:
        # 尝试创建一个无效的手动阈值配置（α₁超出范围）
        mle = OrderedProbitMLE(
            n_countries=5,
            n_theories=3,
            manual_thresholds=True,
            manual_alpha1=3.5,  # 无效：超出[-3, 3]范围
            manual_alpha2=1.5
        )
        print("❌ 错误：应该抛出异常但没有")
    except ValueError as e:
        print(f"✓ 正确捕获到异常: {e}")


def compare_thresholds():
    """
    示例4: 比较自动阈值和手动阈值的结果
    """
    print("\n" + "="*80)
    print("示例4: 比较自动阈值和手动阈值的结果")
    print("="*80)
    
    # 这里可以添加代码来比较两种模式的性能差异
    # 例如：准确率、收敛速度等
    print("\n提示：运行示例1和示例2后，可以比较results目录下的结果文件")


def main():
    """主函数：运行所有示例"""
    print("\n" + "="*80)
    print("手动阈值配置示例程序")
    print("="*80)
    print("\n本程序演示如何使用手动阈值控制功能")
    print("\n可用示例：")
    print("  1. 自动优化阈值模式（默认）")
    print("  2. 手动阈值控制模式")
    print("  3. 无效的手动阈值配置")
    print("  4. 比较两种模式的结果")
    print("\n选择要运行的示例（输入数字1-4，或输入'all'运行所有示例）: ")
    
    choice = input().strip()
    
    if choice == '1':
        example_auto_thresholds()
    elif choice == '2':
        example_manual_thresholds()
    elif choice == '3':
        example_invalid_thresholds()
    elif choice == '4':
        compare_thresholds()
    elif choice.lower() == 'all':
        example_auto_thresholds()
        example_manual_thresholds()
        example_invalid_thresholds()
        compare_thresholds()
        print("\n" + "="*80)
        print("所有示例运行完成！")
        print("="*80)
    else:
        print("无效的选择，程序退出")


if __name__ == "__main__":
    main()
