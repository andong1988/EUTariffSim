# 数据匿名化模块

## 概述

该模块负责从原始数据文件中读取国家数据，使用大语言模型进行匿名化处理，并将匿名化后的数据保存到独立文件中。主程序从匿名化文件读取数据进行模拟。

## 模块结构

```
data_anonymization/
├── __init__.py              # 模块初始化文件
├── data_loader.py           # 从eu_data.py加载原始数据
├── anonymizer.py            # 数据匿名化管理器
├── README.md                # 本文档
```

相关文件：
```
actions/
├── anonymization_action.py  # LLM匿名化Action

根目录/
├── run_anonymization.py     # 独立运行的匿名化脚本
```

## 使用方法

### 1. 配置匿名化

编辑 `config/config.yaml` 文件，配置 `data_anonymization` 部分：

```yaml
# 数据匿名化配置（LLM驱动）
data_anonymization:
  # 是否启用数据匿名化（默认关闭，需要手工控制）
  enabled: false
  
  # 匿名化数据文件路径
  output_file: "./anonymized_data.json"
  
  # LLM配置（用于数据匿名化）
  llm:
    provider: "openai"  # 使用的LLM提供商
    model: "gpt-4-turbo-preview"
    temperature: 0.3  # 较低的温度以获得更一致的输出
    max_tokens: 2000
```

### 2. 运行匿名化

在项目根目录运行：

```bash
cd MetaGPT-EUTariffSim
python run_anonymization.py
```

该脚本会：
1. 从 `eu_data.py` 读取原始国家数据
2. 使用LLM对每个国家进行匿名化，生成匿名代码和四个维度的匿名化文本
3. 将结果保存到配置的 `output_file` 路径

### 3. 运行主程序

匿名化完成后，可以正常运行主程序：

```bash
python simulation_system.py
```

主程序会自动从匿名化文件加载数据。

## 数据格式

### 输出文件格式 (anonymized_data.json)

```json
{
  "metadata": {
    "generated_at": "2024-01-01T00:00:00",
    "total_countries": 8
  },
  "countries": {
    "Germany": {
      "country_id": "Germany",
      "anonymous_code": "Country_A1",
      "population": 83200000,
      "anonymized_text": {
        "economic": "经济维度匿名化文本...",
        "political": "政治维度匿名化文本...",
        "normative": "规范维度匿名化文本...",
        "strategic": "战略维度匿名化文本..."
      }
    },
    ...
  }
}
```

## 匿名化维度

每个国家的匿名化数据包含四个维度，每个维度不超过200字：

1. **经济维度** (economic)：描述国家的经济特征、产业结构、贸易关系等
2. **政治维度** (political)：描述国家的政治体制、治理能力、政策倾向等
3. **规范维度** (normative)：描述国家的价值观、规范认同、道德考量等
4. **战略维度** (strategic)：描述国家的战略利益、安全关切、外交立场等

## 配置说明

### 主要配置项

- `enabled`: 是否启用数据匿名化（默认false）
- `output_file`: 匿名化数据输出文件路径
- `llm.provider`: 使用的LLM提供商（openai/deepseek/zhipuai）
- `llm.model`: 使用的LLM模型名称
- `llm.temperature`: 温度参数（0.1-1.0，越低输出越稳定）

### 重试配置

```yaml
retry:
  max_attempts: 3      # 最大重试次数
  delay_seconds: 2     # 重试间隔（秒）
```

## 注意事项

1. **LLM API密钥**：确保已配置正确的LLM API密钥（环境变量或配置文件）
2. **匿名化频率**：匿名化是一个耗时操作，建议在数据更新时才重新运行
3. **文件路径**：确保 `output_file` 指定的目录存在且有写入权限
4. **网络连接**：匿名化过程需要调用LLM API，确保网络连接正常

## 故障排除

### 问题：匿名化数据文件不存在

**错误信息**：
```
匿名化数据文件不存在: ./anonymized_data.json
```

**解决方案**：
1. 运行 `python run_anonymization.py` 生成匿名化数据
2. 或在配置中检查 `data_anonymization.output_file` 路径是否正确

### 问题：LLM调用失败

**错误信息**：
```
LLM调用失败: API密钥无效
```

**解决方案**：
1. 检查环境变量是否正确设置（如 `OPENAI_API_KEY`）
2. 检查配置文件中的LLM配置是否正确
3. 检查网络连接是否正常

### 问题：匿名化文本质量不佳

**解决方案**：
1. 调整 `llm.temperature` 参数（降低以提高稳定性，或提高以增加多样性）
2. 修改 `anonymization_action.py` 中的提示词
3. 使用更强大的LLM模型

## 开发说明

### 扩展匿名化逻辑

如需修改匿名化逻辑，主要修改以下文件：

1. `actions/anonymization_action.py`：修改LLM提示词和响应解析
2. `data_anonymization/anonymizer.py`：修改匿名化流程
3. `config/config.yaml`：调整配置参数

### 添加新的维度

在 `config/config.yaml` 中添加新维度：

```yaml
dimensions:
  - name: "economic"
    description: "经济维度"
    max_words: 200
  - name: "new_dimension"
    description: "新维度描述"
    max_words: 200
```

同时在 `actions/anonymization_action.py` 中更新提示词以包含新维度。

## 版本历史

- v1.0 (2024-01-01): 初始版本，支持基于LLM的四维度数据匿名化
