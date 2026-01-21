# 手动阈值配置功能说明

## 功能概述

Ordered Probit模型权重优化工具现已支持手动阈值配置功能。用户可以选择：
- **自动模式**（默认）：模型自动优化阈值α₁和α₂，以及各国的权重
- **手动模式**：用户手动指定阈值α₁和α₂，模型只优化各国的权重

## 主要特性

1. **向后兼容**：默认行为保持不变，不影响现有代码
2. **参数验证**：自动验证手动阈值的有效性
3. **灵活控制**：根据需要选择优化策略
4. **清晰输出**：明确显示使用的模式和结果

## 使用方法

### 1. 自动优化阈值（默认模式）

```python
from analysis.ordered_probit_analysis import OrderedProbitAnalysis

# 创建分析器（默认使用自动阈值优化）
analyzer = OrderedProbitAnalysis(
    cache_dir="../cache",
    vote_data_path="../real_vote.xlsx",
    results_dir="results/probit_auto"
)

# 运行完整分析
analyzer.run_complete_analysis()
```

### 2. 手动配置阈值

```python
from analysis.ordered_probit_analysis import OrderedProbitAnalysis
from utils.ordered_probit_mle import OrderedProbitMLE

# 创建分析器
analyzer = OrderedProbitAnalysis(
    cache_dir="../cache",
    vote_data_path="../real_vote.xlsx",
    results_dir="results/probit_manual"
)

# 创建使用手动阈值的MLE模型
n_countries = len(analyzer.countries)
analyzer.mle_model = OrderedProbitMLE(
    n_countries=n_countries,
    n_theories=3,
    manual_thresholds=True,    # 启用手动阈值控制
    manual_alpha1=-0.5,       # 手动配置α₁
    manual_alpha2=0.5         # 手动配置α₂
)

# 运行完整分析
analyzer.run_complete_analysis()
```

### 3. 参数说明

#### OrderedProbitMLE构造函数参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `n_countries` | int | 必需 | 国家数量 |
| `n_theories` | int | 3 | 理论维度数量 |
| `manual_thresholds` | bool | False | 是否手动控制阈值 |
| `manual_alpha1` | float | None | 手动配置的α₁阈值（反对/弃权边界） |
| `manual_alpha2` | float | None | 手动配置的α₂阈值（弃权/赞成边界） |

#### 阈值约束

1. **顺序约束**：必须满足 α₂ > α₁
2. **范围约束**：α₁和α₂都必须在[-3, 3]范围内
3. **必需参数**：启用`manual_thresholds=True`时，必须同时提供`manual_alpha1`和`manual_alpha2`

### 4. 错误处理

```python
from utils.ordered_probit_mle import OrderedProbitMLE

# 示例1：α₂ <= α₁（无效）
try:
    mle = OrderedProbitMLE(
        n_countries=5,
        manual_thresholds=True,
        manual_alpha1=0.5,
        manual_alpha2=0.3  # 错误：α₂必须大于α₁
    )
except ValueError as e:
    print(f"错误: {e}")
    # 输出: 错误: 手动阈值必须满足 α₂ > α₁，当前: α₁=0.5, α₂=0.3

# 示例2：阈值超出范围（无效）
try:
    mle = OrderedProbitMLE(
        n_countries=5,
        manual_thresholds=True,
        manual_alpha1=3.5,  # 错误：超出[-3, 3]范围
        manual_alpha2=1.5
    )
except ValueError as e:
    print(f"错误: {e}")
    # 输出: 错误: α₁必须在[-3, 3]范围内，当前值: 3.5

# 示例3：缺少必需参数（无效）
try:
    mle = OrderedProbitMLE(
        n_countries=5,
        manual_thresholds=True,
        manual_alpha1=-0.5  # 错误：缺少manual_alpha2
    )
except ValueError as e:
    print(f"错误: {e}")
    # 输出: 错误: 启用手动阈值时，必须提供manual_alpha1和manual_alpha2参数
```

## 运行示例程序

项目提供了完整的示例程序，位于：
`MetaGPT-EUTariffSim/analysis/manual_threshold_example.py`

### 运行方法

```bash
cd MetaGPT-EUTariffSim/analysis
python manual_threshold_example.py
```

### 可用示例

1. **示例1**：自动优化阈值模式（默认）
2. **示例2**：手动阈值控制模式
3. **示例3**：演示无效的手动阈值配置
4. **示例4**：比较两种模式的结果

## 输出结果

### 自动模式输出

```
================================================================================
开始迭代优化流程（自动阈值模式）
================================================================================

数据分布:
  反对 (0): X
  弃权 (1): X
  赞成 (2): X

使用自动阈值优化流程...

================================================================================
【阶段1】优先优化反对票准确率
================================================================================
...
```

### 手动模式输出

```
================================================================================
开始迭代优化流程（手动阈值模式）
================================================================================

数据分布:
  反对 (0): X
  弃权 (1): X
  赞成 (2): X

================================================================================
【手动阈值模式】使用用户配置的阈值
================================================================================

手动配置的阈值:
  α₁ = -0.500000
  α₂ = 0.500000

类别权重设置:
  反对票权重: 10.0
  弃权票权重: 5.0
  赞成票权重: 1.0

开始优化权重（α₁和α₂固定）...

优化完成！
最终阈值:
  α₁ = -0.500000 (手动配置)
  α₂ = 0.500000 (手动配置)

最终准确率:
  反对: XX.XX%
  弃权: XX.XX%
  赞成: XX.XX%
  整体: XX.XX%

================================================================================
手动阈值优化流程完成
================================================================================
```

## 保存的参数文件

优化完成后，参数会保存到以下文件：

1. **JSON格式**：`results/probit_*/estimated_parameters.json`
2. **YAML格式**：`results/probit_*/estimated_parameters.yaml`

### 参数文件结构

```json
{
  "thresholds": {
    "alpha1": -0.5,
    "alpha2": 0.5
  },
  "country_weights": {
    "Germany": {
      "x_market": 0.3333,
      "x_political": 0.3333,
      "x_institutional": 0.3333
    },
    ...
  },
  "theory_names": [
    "x_market",
    "x_political",
    "x_institutional"
  ],
  "convergence_info": {
    "success": true,
    "method": "manual_thresholds",
    ...
  }
}
```

## 两种模式的对比

| 特性 | 自动模式 | 手动模式 |
|------|----------|----------|
| 阈值优化 | ✅ 自动优化α₁和α₂ | ❌ 使用用户指定的值 |
| 权重优化 | ✅ 两阶段优化 | ✅ 单阶段优化 |
| 适用场景 | 探索性分析、无先验知识 | 有明确阈值要求、对比实验 |
| 计算复杂度 | 较高 | 较低 |
| 收敛速度 | 较慢 | 较快 |

## 最佳实践

### 何时使用自动模式

1. 没有先验知识，希望模型自动学习最佳阈值
2. 进行探索性分析，想了解数据驱动的阈值
3. 需要最大化的预测准确率

### 何时使用手动模式

1. 有明确的阈值要求（如基于理论或实验）
2. 需要对比不同阈值下的权重变化
3. 控制变量实验，固定阈值观察权重变化
4. 阈值已知且可信，只需优化权重

### 阈值选择建议

- **α₁（反对/弃权边界）**：通常在[-2.4, -0.6]范围内
- **α₂（弃权/赞成边界）**：通常在[0.6, 2.4]范围内
- 建议：可以先运行自动模式，然后使用其结果作为手动模式的初始值

## 常见问题

### Q1: 手动阈值会影响权重的优化吗？

**A**: 不会。手动模式只固定阈值α₁和α₂，各国的权重仍然会根据数据进行优化。

### Q2: 如何判断手动阈值是否合理？

**A**: 可以通过以下方式：
1. 运行自动模式，查看优化得到的阈值作为参考
2. 查看手动模式的准确率，确保不低于自动模式太多
3. 检查权重分布是否合理（每个维度不应过于极端）

### Q3: 可以中途切换模式吗？

**A**: 不可以。模式必须在创建MLE模型时确定，之后无法更改。如果需要切换，需要重新创建模型。

### Q4: 手动阈值会影响收敛速度吗？

**A**: 通常会提高收敛速度。因为手动模式跳过了阈值优化阶段，只优化权重，计算量更小。

## 技术细节

### 优化算法

- **自动模式**：使用两阶段迭代优化
  - 阶段1：优先优化反对票准确率
  - 阶段2：优化弃权票和赞成票准确率
- **手动模式**：单阶段优化
  - 阈值固定，只优化权重参数

### 参数约束

1. **权重归一化**：每个国家的权重经过softmax归一化，确保和为1
2. **最小权重约束**：每个维度至少占10%，防止极端化
3. **L2正则化**：鼓励权重接近均匀分布
4. **熵正则化**：进一步防止权重过度集中

## 更新日志

### 版本 1.1 (当前版本)

- ✅ 新增手动阈值配置功能
- ✅ 添加参数验证机制
- ✅ 改进输出信息，明确显示使用的模式
- ✅ 提供完整的示例程序和文档

### 版本 1.0

- 基础的自动阈值优化功能

## 联系与反馈

如有问题或建议，请通过以下方式联系：
- 项目仓库：[GitHub链接]
- 问题反馈：[Issues链接]

---

**最后更新**：2026年1月5日
