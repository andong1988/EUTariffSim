MetaGPT-EUTariffSim/
├── simulation_system.py          # 新的主程序（重构后）
├── run_anonymization.py    # 匿名化脚本（重构后）
├── analysis
│   ├── order_probit_analysis.py    #进行probit权重优化
├──run_anonymization_validation.py    #匿名化验证脚本（重构后）

---cache    #文件夹内为各国的三维理论得分缓存
---analysis/results/probit/estimated_parameters.json    #存放probit权重优化结果的各种参数
---actions
    ---prompts  #进行三维理论得分计算的prompts
../results/probit      #存放probit权重优化结果的各种图片
/results/charts/         #存放三维理论得分计算的图片
/anonymized_data.json    #国家数据匿名化后的数据