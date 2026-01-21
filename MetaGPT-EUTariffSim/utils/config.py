"""配置和枚举类"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum


class SimulationPhase(Enum):
    """模拟阶段枚举"""
    INITIALIZATION = "initialization"
    DATA_ANONYMIZATION = "data_anonymization"
    PROPOSAL = "proposal"
    INITIAL_VOTING = "initial_voting"
    COMMUNICATION = "communication"
    FINAL_VOTING = "final_voting"
    WEIGHT_OPTIMIZATION = "weight_optimization"
    ANALYSIS = "analysis"


class VotingStance(Enum):
    """投票立场枚举"""
    SUPPORT = "support"
    AGAINST = "against"
    ABSTAIN = "abstain"
    NEUTRAL = "neutral"


@dataclass
class SimulationConfig:
    """模拟配置"""
    # 基础配置
    num_countries: int = 2  # 默认3个国家
    selected_countries: List[str] = None  # 具体选择的国家列表，如果为None则按顺序选择前num_countries个国家
    enable_weight_optimization: bool = True
    enable_communication: bool = True
    enable_visualization: bool = True
    
    # 模拟参数
    tariff_rate: float = 0.35  # 35%关税
    communication_urgency_threshold: float = 0.3
    max_communications_per_round: int = 3
    weight_optimization_iterations: int = 50
    
    # 理论权重初始化
    initial_theory_weights: Dict[str, float] = None
    load_saved_weights: bool = True  # 是否读取之前保存的理论权重，默认读取
    
    # 理论得分缓存配置
    enable_theory_scores_cache: bool = True  # 是否启用理论得分缓存，默认开启（保存LLM生成的理论得分到文件）
    use_cached_scores_for_voting: bool = True  # 投票时是否从缓存加载理论得分（已废弃，使用 use_cache_for_round 替代）
    use_cached_scores_for_optimization: bool = True  # 权重优化时是否使用本轮运行生成的缓存理论得分，默认开启
    use_cache_for_round: Dict[str, bool] = None  # 各轮次是否使用缓存的配置 {"initial": True, "final": False}
    
    # 随机扰动配置
    enable_decision_noise: bool = False  # 是否启用决策随机扰动，默认关闭
    decision_noise_level: float = 0.05  # 决策随机扰动级别，默认0.05
    
    # 真实投票数据
    actual_voting_data: Dict[str, List[str]] = None
    
    # 第二次投票前中国针对性沟通配置
    enable_china_targeted_communication: bool = False  # 是否启用中国在第二次投票前的针对性沟通
    china_targeted_communications: Dict[str, str] = None  # 中国在第二次投票前向特定国家发送的手动指定信息 {国家名: 信息内容}
    
    # 输出配置
    save_intermediate_results: bool = True
    generate_detailed_report: bool = True
    create_visualizations: bool = True
    
    def __post_init__(self):
        if self.initial_theory_weights is None:
            self.initial_theory_weights = {
                "x_market": 0.333,
                "x_political": 0.333,
                "x_institutional": 0.334
            }
        
        # 初始化各轮次缓存配置的默认值
        if self.use_cache_for_round is None:
            self.use_cache_for_round = {
                "initial": True,  # 初始投票可以使用缓存
                "final": False    # 最终投票不使用缓存
            }
        
        # 注意：actual_voting_data 现在在 EUTariffSimulation 初始化时从 Excel 文件加载
        # 这里保留默认值作为备用
        if self.actual_voting_data is None:
            self.actual_voting_data = {
                "Germany": ["abstain", "against"],
                "France": ["support", "support"],
                "Italy": ["support", "support"],
                "Netherlands": ["support", "support"],
                "Ireland": ["abstain", "support"],
                "Denmark": ["support", "support"],
                "Spain": ["support", "abstain"]
            }
