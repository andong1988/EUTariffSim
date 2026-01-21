"""
只使用已有权重数据生成雷达图
不进行模型优化，直接读取estimated_parameters.json生成可视化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.probit_visualizer import ProbitVisualizer


def main():
    """主函数：读取权重并生成雷达图"""
    
    # 1. 读取权重文件
    params_file = os.path.join(os.path.dirname(__file__), "results", "probit", "estimated_parameters.json")
    
    print(f"正在读取权重文件: {params_file}")
    
    with open(params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 提取国家名称和权重
    country_weights = data['country_weights']
    
    # 将Netherland改为The Netherlands（用于显示）
    country_names = []
    weights_list = []
    
    name_mapping = {
        'Netherland': 'The Netherlands'
    }
    
    for country, weights in country_weights.items():
        # 使用映射后的名称（如果有）
        display_name = name_mapping.get(country, country)
        country_names.append(display_name)
        weights_list.append([
            weights['x_market'],
            weights['x_political'],
            weights['x_institutional']
        ])
    
    # 转换为numpy数组
    weights_array = np.array(weights_list)
    
    print(f"\n读取到 {len(country_names)} 个国家的权重数据:")
    for i, (country, weights) in enumerate(zip(country_names, weights_list)):
        print(f"  {i+1}. {country}: market={weights[0]:.4f}, political={weights[1]:.4f}, institutional={weights[2]:.4f}")
    
    # 3. 创建可视化器并生成雷达图
    print("\n正在生成雷达图...")
    
    visualizer = ProbitVisualizer(output_dir="results/probit")
    
    # 创建单独的雷达图
    n_countries = len(country_names)
    n_theories = 3
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # 设置角度
    angles = np.linspace(0, 2 * np.pi, n_theories, endpoint=False).tolist()
    angles += angles[:1]
    
    # 为每个国家绘制雷达图
    radar_colors = plt.cm.tab10(np.linspace(0, 1, n_countries))
    
    for idx in range(n_countries):
        values = weights_array[idx].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2,
               label=country_names[idx], color=radar_colors[idx])
        ax.fill(angles, values, alpha=0.15, color=radar_colors[idx])
    
    # 设置标签和标题 - 不显示默认的x轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # 清空默认标签

    # 手动添加标签，可以精确控制位置
    label_distance = 1.15  # 标签距离中心的距离，可以调整
    label_names = ['w_market', 'w_political', 'w_institutional']

    for angle, name in zip(angles[:-1], label_names):
        # 转换角度为弧度
        ax.text(angle, label_distance, name,
                ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_title('All Countries Weight Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    # 保存图片
    output_file = os.path.join(os.path.dirname(__file__), "results", "probit", "radar_chart_all_countries.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n雷达图已保存到: {output_file}")
    plt.close()
    
    print("\n完成！")


if __name__ == "__main__":
    main()
