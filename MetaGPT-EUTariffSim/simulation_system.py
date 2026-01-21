"""
重构后的欧盟对华汽车关税投票模拟系统
基于MetaGPT框架的完整多智能体系统
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent))

from utils.config import SimulationConfig, SimulationPhase, VotingStance
from utils.weight_optimizer import WeightOptimizer
from data_anonymization import DataLoader
from roles import EUCountryRole, EUCommissionRole, ChinaRole
from roles.secretary_role import SecretaryRole
from analysis_visualization import SimulationAnalyzer

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EUTariffSimulation:
    """欧盟关税模拟系统"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.weight_optimizer = WeightOptimizer(self.logger)
        
        # 加载匿名化数据文件路径
        self.anonymized_data_file = self._get_anonymized_data_file_path()
        self._load_anonymized_data()
        
        # 智能体
        self.country_roles = {}
        self.eu_commission = None
        self.china = None
        self.secretary_roles = {}  # 秘书智能体，每个国家一个
        
        # 模拟数据
        self.simulation_phases = {}
        self.simulation_results = {}
        
        # 结果目录
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 模拟状态
        self.current_phase = None
        self.simulation_start_time = None
        self.simulation_end_time = None
        
        # 加载真实投票数据
        self.real_voting_data = self._load_real_voting_data()
        
        # 加载国家理论权重
        self.country_theory_weights = self._load_country_theory_weights()
        
        # 加载Ordered Probit参数
        self.ordered_probit_params = self._load_ordered_probit_parameters()
        
        # 欧盟投票规则
        self.eu_voting_rules = {
            "against_threshold_countries": 0.55,  # 55%及以上数量的国家投反对票才会否决
            "against_threshold_population": 0.65,  # 占65%人口的国家投反对票才会否决
        }
        
        # 存储国家人口数据
        self.country_populations = {}
        
        # 国家匿名化映射（在数据匿名化阶段创建）
        self.country_anonymization_map = {}
        
        # 用于调用LLM的Action实例
        from actions.theoretical_decision_action import TheoreticalDecisionAction
        self.theoretical_decision_action = TheoreticalDecisionAction()
        
        # 用于选择沟通目标的Action实例
        from actions.select_communication_targets_action import SelectCommunicationTargetsAction
        self.select_communication_targets_action = SelectCommunicationTargetsAction()
    
    def _get_anonymized_data_file_path(self) -> Path:
        """从配置文件获取匿名化数据文件路径"""
        try:
            import yaml
            config_file = Path(__file__).parent / "config" / "config.yaml"
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                data_anonymization_config = config.get('data_anonymization', {})
                output_file = data_anonymization_config.get('output_file', './anonymized_data.json')
                
                # 如果是相对路径，转换为绝对路径
                if not Path(output_file).is_absolute():
                    output_file = Path(__file__).parent / output_file
                
                return Path(output_file)
            else:
                # 默认路径
                return Path(__file__).parent / "anonymized_data.json"
        except Exception as e:
            self.logger.warning(f"读取配置文件失败，使用默认路径: {e}")
            return Path(__file__).parent / "anonymized_data.json"
    
    def _load_anonymized_data(self):
        """根据配置加载匿名化数据或原始数据"""
        # 初始化字典
        self.country_anonymization_map = {}
        self.country_populations = {}
        
        # 读取配置文件
        try:
            import yaml
            config_file = Path(__file__).parent / "config" / "config.yaml"
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                self.use_anonymized_data = config.get('data_anonymization', {}).get('enabled', True)
            else:
                self.use_anonymized_data = True
        except Exception as e:
            self.logger.warning(f"读取配置文件失败，使用默认值（启用匿名化）: {e}")
            self.use_anonymized_data = True
        
        # 根据配置选择数据源
        if self.use_anonymized_data:
            self.logger.info("配置：使用匿名化数据")
            self._load_from_anonymized_file()
        else:
            self.logger.info("配置：使用原始数据")
            self._load_from_raw_file()
    
    def _load_from_anonymized_file(self):
        """从匿名化文件加载数据"""
        try:
            if not self.anonymized_data_file.exists():
                self.logger.warning(f"匿名化数据文件不存在: {self.anonymized_data_file}")
                self.logger.warning("尝试回退到原始数据")
                self._load_from_raw_file()
                return
            
            with open(self.anonymized_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取国家数据
            countries = data.get("countries", {})
            
            # 构建匿名化映射
            for country_id, country_data in countries.items():
                self.country_anonymization_map[country_id] = country_data.get("anonymous_code", f"Unknown_{country_id}")
                self.country_populations[country_id] = country_data.get("population", 0)
            
            # 存储完整的匿名化数据
            self.anonymized_data = countries
            
            self.logger.info(f"成功加载匿名化数据: {len(countries)}个国家")
            self.logger.info(f"匿名化映射: {self.country_anonymization_map}")
            
        except Exception as e:
            self.logger.error(f"加载匿名化数据失败: {e}")
            self.logger.warning("尝试回退到原始数据")
            self._load_from_raw_file()
    
    def _load_from_raw_file(self):
        """直接从原始文件加载数据"""
        try:
            from data_anonymization import DataLoader
            
            data_loader = DataLoader()
            eu_data = data_loader.get_all_countries()
            
            # 构建数据映射
            for country_id, country_data in eu_data.items():
                self.country_anonymization_map[country_id] = country_id  # 原始数据不使用匿名代码
                self.country_populations[country_id] = country_data.get("population", 0)
            
            # 存储原始数据
            self.anonymized_data = eu_data
            
            self.logger.info(f"成功加载原始数据: {len(eu_data)}个国家")
            
        except Exception as e:
            self.logger.error(f"加载原始数据失败: {e}")
            raise
    
    def _load_country_populations(self) -> Dict[str, int]:
        """从eu_data加载国家人口数据"""
        try:
            eu_data_path = Path(__file__).parent.parent / "eu_data.py"
            eu_data = {}
            
            import importlib.util
            spec = importlib.util.spec_from_file_location("eu_data", eu_data_path)
            eu_data_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eu_data_module)
            
            if hasattr(eu_data_module, 'countrys_EU'):
                eu_data = eu_data_module.countrys_EU
            
            # 提取人口数据
            populations = {}
            for country_id, country_data in eu_data.items():
                if "population" in country_data:
                    populations[country_id] = country_data["population"]
            
            self.logger.info(f"成功加载{len(populations)}个国家的人口数据")
            return populations
            
        except Exception as e:
            self.logger.error(f"加载国家人口数据失败: {e}")
            return {}
    
    def _calculate_eu_voting_result(self, votes: Dict[str, str]) -> Dict[str, Any]:
        """根据欧盟投票规则计算投票结果"""
        if not self.country_populations:
            self.country_populations = self._load_country_populations()
        
        total_countries = len(votes)
        if total_countries == 0:
            return {
                "total_countries": 0,
                "support_count": 0,
                "against_count": 0,
                "abstain_count": 0,
                "against_country_ratio": 0.0,
                "against_population_ratio": 0.0,
                "passed": True,
                "reason": "无投票，默认通过"
            }
        
        # 统计投票数量
        support_count = sum(1 for vote in votes.values() if vote == "support")
        against_count = sum(1 for vote in votes.values() if vote == "against")
        abstain_count = sum(1 for vote in votes.values() if vote == "abstain")
        
        # 计算反对票比例
        against_country_ratio = against_count / total_countries
        
        # 计算反对票人口比例
        total_population = 0
        against_population = 0
        
        for country_id, vote in votes.items():
            if country_id in self.country_populations:
                population = self.country_populations[country_id]
                total_population += population
                
                if vote == "against":
                    against_population += population
        
        against_population_ratio = against_population / total_population if total_population > 0 else 0.0
        
        # 判断是否通过
        veto_by_countries = against_country_ratio >= self.eu_voting_rules["against_threshold_countries"]
        veto_by_population = against_population_ratio >= self.eu_voting_rules["against_threshold_population"]
        veto_triggered = veto_by_countries and veto_by_population
        passed = not veto_triggered
        
        if veto_triggered:
            reason = f"否决：反对国家比例{against_country_ratio:.1%}≥{self.eu_voting_rules['against_threshold_countries']:.0%} 且 反对人口比例{against_population_ratio:.1%}≥{self.eu_voting_rules['against_threshold_population']:.0%}"
        else:
            reason = f"通过：不满足否决条件（反对国家比例{against_country_ratio:.1%}，反对人口比例{against_population_ratio:.1%}）"
        
        return {
            "total_countries": total_countries,
            "support_count": support_count,
            "against_count": against_count,
            "abstain_count": abstain_count,
            "against_country_ratio": against_country_ratio,
            "against_population_ratio": against_population_ratio,
            "passed": passed,
            "reason": reason
        }
    
    def _load_country_theory_weights(self) -> Dict[str, Dict[str, float]]:
        """从文件加载国家理论权重"""
        try:
            weights_file = Path(__file__).parent / "country_theory_weights.json"
            
            if not weights_file.exists():
                self.logger.info(f"权重文件不存在，将创建新文件: {weights_file}")
                return {}
            
            with open(weights_file, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
            
            self.logger.info(f"成功加载{len(weights_data)}个国家的理论权重")
            return weights_data
            
        except Exception as e:
            self.logger.error(f"加载国家理论权重失败: {e}")
            return {}
    
    def _load_ordered_probit_parameters(self) -> Dict[str, Any]:
        """从文件加载Ordered Probit模型的参数"""
        try:
            # 检查配置是否启用Ordered Probit模型
            import yaml
            config_file = Path(__file__).parent / "config" / "config.yaml"
            
            if not config_file.exists():
                self.logger.info("配置文件不存在，不使用Ordered Probit模型")
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            ordered_probit_config = config.get('ordered_probit', {})
            
            # 检查是否启用模型
            if not ordered_probit_config.get('use_model', False):
                self.logger.info("Ordered Probit模型未启用")
                return None
            
            # 获取参数文件路径
            params_path_str = ordered_probit_config.get('params_path', 'results/probit/estimated_parameters.json')
            params_path = Path(__file__).parent / params_path_str
            
            if not params_path.exists():
                self.logger.warning(f"Ordered Probit参数文件不存在: {params_path}")
                return None
            
            # 加载参数文件
            with open(params_path, 'r', encoding='utf-8') as f:
                params_data = json.load(f)
            
            # 提取阈值参数
            thresholds = params_data.get('thresholds', {})
            alpha1 = thresholds.get('alpha1', 0.0)
            alpha2 = thresholds.get('alpha2', 0.5)
            
            # 提取国家权重
            country_weights = params_data.get('country_weights', {})
            
            result = {
                'alpha1': alpha1,
                'alpha2': alpha2,
                'country_weights': country_weights,
                'params_file': str(params_path)
            }
            
            self.logger.info(f"成功加载Ordered Probit参数: α1={alpha1:.4f}, α2={alpha2:.4f}, {len(country_weights)}个国家权重")
            return result
            
        except Exception as e:
            self.logger.error(f"加载Ordered Probit参数失败: {e}")
            return None
    
    def _save_country_theory_weights(self):
        """保存优化后的国家理论权重到文件"""
        try:
            weights_file = Path(__file__).parent / "country_theory_weights.json"
            
            current_weights = {}
            for country_id, role in self.country_roles.items():
                current_weights[country_id] = role.theory_weights.copy()
            
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump(current_weights, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"已保存{len(current_weights)}个国家的优化权重")
            
        except Exception as e:
            self.logger.error(f"保存国家理论权重失败: {e}")
    
    def _save_theoretical_scores_to_cache(self, initial_voting: Dict, final_voting: Dict):
        """保存理论得分到cache目录供Ordered Probit分析使用（使用独立的probit子目录，避免与EUCountryRole的缓存冲突）
        
        Args:
            initial_voting: 初始投票结果
            final_voting: 最终投票结果
        """
        try:
            # 创建独立的probit缓存目录，避免覆盖EUCountryRole的缓存文件
            probit_cache_dir = Path(__file__).parent / "cache" / "probit"
            probit_cache_dir.mkdir(exist_ok=True)
            
            self.logger.info("=== 保存理论得分到cache目录 ===")
            
            saved_count = 0
            
            for country_id in self.country_roles.keys():
                # 获取initial理论得分
                initial_details = initial_voting.get('voting_details', {}).get(country_id, {})
                initial_theory = initial_details.get('theoretical_factors', {})
                
                # 获取final理论得分
                final_details = final_voting.get('voting_details', {}).get(country_id, {})
                final_theory = final_details.get('theoretical_factors', {})
                
                # 构建保存格式（符合ordered_probit_analysis.py期望的格式）
                scores_data = {
                    "initial": {
                        "theory_scores": {
                            "x_market": float(initial_theory.get('x_market', 0.0)),
                            "x_political": float(initial_theory.get('x_political', 0.0)),
                            "x_institutional": float(initial_theory.get('x_institutional', 0.0))
                        }
                    },
                    "final": {
                        "theory_scores": {
                            "x_market": float(final_theory.get('x_market', 0.0)),
                            "x_political": float(final_theory.get('x_political', 0.0)),
                            "x_institutional": float(final_theory.get('x_institutional', 0.0))
                        }
                    }
                }
                
                # 保存文件到probit子目录，文件名加上_probit后缀
                cache_file = probit_cache_dir / f"theory_scores_{country_id}_probit.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(scores_data, f, ensure_ascii=False, indent=2)
                
                saved_count += 1
                self.logger.info(f"  已保存: {country_id} -> {cache_file.name}")
            
            self.logger.info(f"成功保存{saved_count}个国家的理论得分到cache/probit目录")
            
        except Exception as e:
            self.logger.error(f"保存理论得分到cache目录失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_real_voting_data(self) -> Dict[str, List[str]]:
        """从Excel文件加载真实投票数据"""
        try:
            excel_path = Path(__file__).parent.parent / "real_vote.xlsx"
            
            if not excel_path.exists():
                self.logger.warning(f"真实投票数据文件不存在: {excel_path}")
                return {}
            
            df = pd.read_excel(excel_path)
            
            country_name_mapping = {
                "丹麦": "Denmark",
                "意大利": "Italy", 
                "荷兰": "Netherlands",
                "法国": "France",
                "德国": "Germany",
                "西班牙": "Spain",
                "立陶宛": "Lithuania",
                "爱尔兰": "Ireland"
            }
            
            voting_data = {}
            
            for index, row in df.iterrows():
                chinese_name = row.iloc[1]
                english_name = row.iloc[0]
                
                if pd.isna(chinese_name) and pd.isna(english_name):
                    continue
                
                if pd.notna(english_name):
                    country_key = english_name
                elif pd.notna(chinese_name) and chinese_name in country_name_mapping:
                    country_key = country_name_mapping[chinese_name]
                else:
                    continue
                
                first_vote = row.iloc[2]
                second_vote = row.iloc[3]
                
                def convert_vote(vote):
                    if pd.isna(vote):
                        return "support"
                    vote_str = str(vote).strip()
                    if "支持" in vote_str:
                        return "support"
                    elif "反对" in vote_str:
                        return "against"
                    elif "弃权" in vote_str:
                        return "abstain"
                    else:
                        return "support"
                
                first_vote_stance = convert_vote(first_vote)
                second_vote_stance = convert_vote(second_vote)
                
                voting_data[country_key] = [first_vote_stance, second_vote_stance]
            
            self.logger.info(f"成功加载真实投票数据: {len(voting_data)}个国家")
            return voting_data
            
        except Exception as e:
            self.logger.error(f"加载真实投票数据失败: {e}")
            return {}
    
    async def run_complete_simulation(self) -> Dict:
        """运行完整模拟"""
        self.simulation_start_time = datetime.now()
        self.logger.info("开始欧盟关税模拟")
        
        try:
            # 阶段1: 初始化智能体
            await self._phase_initialization()
            
            # 阶段2: 数据匿名化
            anonymized_data = await self._phase_data_anonymization()
            
            # 阶段3: 欧委会提案
            proposal = await self._phase_proposal()
            
            # 阶段4: 初始投票
            initial_voting = await self._phase_initial_voting(proposal)
            
            # 阶段5: 沟通协商
            communications = await self._phase_communication(initial_voting)
            
            # 阶段6: 秘书分析
            secretary_analysis = await self._phase_secretary_analysis(communications)
            
            # 阶段7: 最终投票
            final_voting = await self._phase_final_voting(proposal, communications, secretary_analysis)
            
            # 阶段7: 权重优化
            weight_optimization = await self._phase_weight_optimization(initial_voting, final_voting)
            
            # 阶段8: 分析总结
            analysis = await self._phase_analysis(proposal, initial_voting, communications, 
                                             final_voting, weight_optimization)
            
            self.simulation_end_time = datetime.now()
            
            # 保存理论得分到cache目录供Ordered Probit分析使用
            self._save_theoretical_scores_to_cache(initial_voting, final_voting)
            
            # 生成最终报告
            final_report = await self._generate_comprehensive_report(
                proposal, initial_voting, communications, final_voting, weight_optimization, analysis
            )
            
            # 自动进行数据分析和图表生成
            self.logger.info("开始自动数据分析和图表生成")
            try:
                charts_dir = self.results_dir / "charts"
                charts_dir.mkdir(exist_ok=True)
                
                analyzer = SimulationAnalyzer(final_report, str(self.results_dir))
                visualization_results = analyzer.analyze_and_visualize(final_report)
                final_report["visualization"] = visualization_results
                self.logger.info("数据分析和图表生成完成")
            except Exception as e:
                self.logger.error(f"数据分析和图表生成失败: {e}")
                final_report["visualization"] = {"error": str(e)}
            
            self.logger.info("模拟完成")
            return final_report
            
        except Exception as e:
            self.logger.error(f"模拟过程中发生错误: {e}")
            raise
    
    async def _phase_initialization(self):
        """初始化阶段"""
        self.current_phase = SimulationPhase.INITIALIZATION
        self.logger.info("=== 阶段1: 智能体初始化 ===")
        
        # 初始化欧委会智能体
        self.eu_commission = EUCommissionRole()
        
        # 初始化中国智能体
        self.china = ChinaRole()
        
        # 加载欧盟国家数据
        eu_data_path = Path(__file__).parent.parent / "eu_data.py"
        eu_data = {}
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("eu_data", eu_data_path)
            eu_data_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eu_data_module)
            
            if hasattr(eu_data_module, 'countrys_EU'):
                eu_data = eu_data_module.countrys_EU
        except Exception as e:
            self.logger.warning(f"加载eu_data.py失败: {e}")
        
        # 选择指定的国家
        if self.config.selected_countries:
            selected_countries = []
            for country in self.config.selected_countries:
                if country in eu_data:
                    selected_countries.append(country)
            
            self.config.num_countries = len(selected_countries)
        else:
            selected_countries = list(eu_data.keys())[:self.config.num_countries]
        
        self.logger.info(f"选择参与模拟的国家: {selected_countries}")
        
        # 创建国家智能体
        for country_id in selected_countries:
            country_data = eu_data[country_id]
            
            # 优先使用Ordered Probit模型的国家权重
            if self.ordered_probit_params and country_id in self.ordered_probit_params['country_weights']:
                theory_weights = self.ordered_probit_params['country_weights'][country_id]
                self.logger.info(f"{country_id} 使用Ordered Probit优化权重: {theory_weights}")
            elif self.config.load_saved_weights and country_id in self.country_theory_weights:
                theory_weights = self.country_theory_weights[country_id]
                self.logger.info(f"{country_id} 使用保存的传统优化权重: {theory_weights}")
            else:
                theory_weights = self.config.initial_theory_weights.copy()
                self.logger.info(f"{country_id} 使用默认权重: {theory_weights}")
            
            role = EUCountryRole(
                country_id=country_id,
                anonymized_data={},
                theory_weights=theory_weights,
                enable_theory_scores_cache=self.config.enable_theory_scores_cache,
                enable_decision_noise=self.config.enable_decision_noise,
                decision_noise_level=self.config.decision_noise_level,
                ordered_probit_params=self.ordered_probit_params,
                use_cache_for_round=self.config.use_cache_for_round
            )
            
            # 如果启用了理论得分缓存，且在初始化阶段就清除旧缓存
            # 这样确保每次运行都生成新的理论得分
            if self.config.enable_theory_scores_cache and not self.config.use_cached_scores_for_voting:
                role._clear_theory_scores_cache()
            self.country_roles[country_id] = role
        
        self.simulation_phases["initialization"] = {
            "phase": "initialization",
            "countries_initialized": len(self.country_roles),
            "eu_commission_initialized": self.eu_commission is not None,
            "china_initialized": self.china is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"初始化完成: {len(self.country_roles)}个国家")
    
    async def _phase_data_anonymization(self) -> Dict:
        """数据加载阶段（根据配置选择数据源）"""
        self.current_phase = SimulationPhase.DATA_ANONYMIZATION
        
        if self.use_anonymized_data:
            self.logger.info("=== 阶段2: 加载匿名化数据 ===")
            return await self._load_anonymized_dimensions()
        else:
            self.logger.info("=== 阶段2: 加载原始数据 ===")
            return await self._load_original_dimensions()
    
    async def _load_anonymized_dimensions(self) -> Dict:
        """加载匿名化后的三个维度数据"""
        anonymized_data = {}
        for country_id, role in self.country_roles.items():
            if country_id in self.anonymized_data:
                country_data = self.anonymized_data[country_id]
                
                # 直接使用三个维度
                anonymized = {
                    "country_id": country_data.get("anonymous_code", f"Unknown_{country_id}"),
                    "anonymized_text": country_data.get("anonymized_dimensions", {})
                }
                
                anonymized_data[country_id] = anonymized
                role.anonymized_data = anonymized
                
                # 创建该国家的秘书智能体
                self.secretary_roles[country_id] = SecretaryRole(
                    country_id=country_id,
                    anonymized_data=anonymized
                )
                
                self.logger.debug(f"加载匿名化数据: {country_id} -> {anonymized['country_id']}")
        
        self.logger.info(f"已加载{len(anonymized_data)}个国家的匿名化数据")
        self.logger.info(f"已创建{len(self.secretary_roles)}个国家的秘书智能体")
        
        self.simulation_phases["data_anonymization"] = {
            "phase": "data_anonymization",
            "data_source": "anonymized",
            "countries_loaded": len(anonymized_data),
            "secretaries_initialized": len(self.secretary_roles),
            "timestamp": datetime.now().isoformat()
        }
        
        return anonymized_data
    
    async def _load_original_dimensions(self) -> Dict:
        """直接从原始文件加载三个维度数据"""
        from data_anonymization import DataLoader
        
        data_loader = DataLoader()
        original_data = {}
        
        for country_id, role in self.country_roles.items():
            country_data = data_loader.get_country_data(country_id)
            
            # 添加调试日志
            self.logger.info(f"===== {country_id} 数据加载调试 =====")
            self.logger.info(f"country_data 类型: {type(country_data)}")
            self.logger.info(f"country_data 键: {list(country_data.keys()) if country_data else []}")
            
            # 直接使用原始的三个维度
            x_market = country_data.get("X_market (Market / Economic Interdependence)", "")
            x_political = country_data.get("X_political (Domestic Politics and Interest Mediation)", "")
            x_institutional = country_data.get("X_institutional (Institutions, Diplomacy, and Path Dependence)", "")
            
            self.logger.info(f"X_market 长度: {len(x_market)}")
            self.logger.info(f"X_political 长度: {len(x_political)}")
            self.logger.info(f"X_institutional 长度: {len(x_institutional)}")
            self.logger.info(f"X_institutional 前100字符: {x_institutional[:100] if x_institutional else 'EMPTY'}")
            
            original = {
                "country_id": country_id,
                "anonymized_text": {
                    "X_market (Market / Economic Interdependence)": x_market,
                    "X_political (Domestic Politics and Interest Mediation)": x_political,
                    "X_institutional (Institutions, Diplomacy, and Path Dependence)": x_institutional
                }
            }
            
            original_data[country_id] = original
            role.anonymized_data = original
            
            # 创建该国家的秘书智能体
            self.secretary_roles[country_id] = SecretaryRole(
                country_id=country_id,
                anonymized_data=original
            )
            
            self.logger.debug(f"加载原始数据: {country_id}")
        
        self.logger.info(f"已加载{len(original_data)}个国家的原始数据")
        self.logger.info(f"已创建{len(self.secretary_roles)}个国家的秘书智能体")
        
        self.simulation_phases["data_anonymization"] = {
            "phase": "data_anonymization",
            "data_source": "original",
            "countries_loaded": len(original_data),
            "secretaries_initialized": len(self.secretary_roles),
            "timestamp": datetime.now().isoformat()
        }
        
        return original_data
    
    async def _phase_proposal(self) -> Dict:
        """欧委会提案阶段"""
        self.current_phase = SimulationPhase.PROPOSAL
        self.logger.info("=== 阶段3: 欧委会提案 ===")
        
        proposal = await self.eu_commission.publish_proposal(self.config.tariff_rate)
        
        self.simulation_phases["proposal"] = {
            "phase": "proposal",
            "proposal": proposal,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"欧委会提出{proposal['tariff_rate']*100}%关税提案")
        return proposal
    
    async def _phase_initial_voting(self, proposal: Dict) -> Dict:
        """初始投票阶段"""
        self.current_phase = SimulationPhase.INITIAL_VOTING
        self.logger.info("=== 阶段4: 初始投票 ===")
        
        votes = {}
        voting_details = {}
        
        context = {
            "proposal": proposal,
            "round": "initial",
            "phase": "initial_voting"
        }
        
        for country_id, role in self.country_roles.items():
            decision = await role.make_theoretical_decision(context, voting_proposal=proposal)
            votes[country_id] = decision["stance"]
            voting_details[country_id] = decision
        
        voting_result = {
            "votes": votes,
            "voting_details": voting_details,
            "proposal": proposal,
            "round": "initial",
            "timestamp": datetime.now().isoformat()
        }
        
        self.simulation_phases["initial_voting"] = voting_result
        
        support_count = sum(1 for vote in votes.values() if vote == "support")
        against_count = sum(1 for vote in votes.values() if vote == "against")
        abstain_count = sum(1 for vote in votes.values() if vote == "abstain")
        
        # 按投票类型分组国家
        support_countries = [country_id for country_id, vote in votes.items() if vote == "support"]
        against_countries = [country_id for country_id, vote in votes.items() if vote == "against"]
        abstain_countries = [country_id for country_id, vote in votes.items() if vote == "abstain"]
        
        # 输出详细的投票统计
        self.logger.info("\n" + "="*80)
        self.logger.info("【初始投票统计结果】")
        self.logger.info("="*80)
        self.logger.info(f"总票数: {len(votes)}票")
        self.logger.info(f"\n✅ 支持 - {support_count}票:")
        if support_countries:
            for country in support_countries:
                self.logger.info(f"    • {country}")
        else:
            self.logger.info("    （无）")
        
        self.logger.info(f"\n❌ 反对 - {against_count}票:")
        if against_countries:
            for country in against_countries:
                self.logger.info(f"    • {country}")
        else:
            self.logger.info("    （无）")
        
        self.logger.info(f"\n⚪ 弃权 - {abstain_count}票:")
        if abstain_countries:
            for country in abstain_countries:
                self.logger.info(f"    • {country}")
        else:
            self.logger.info("    （无）")
        self.logger.info("="*80 + "\n")
        
        # 输出详细的初始投票信息
        self.logger.info("\n" + "="*80)
        self.logger.info("【初始投票详细数据】")
        self.logger.info("="*80)
        for country_id, decision in voting_details.items():
            self.logger.info(f"\n【{country_id} - 初始投票】")
            self.logger.info(f"  读取的权重: {decision.get('weights_used', {})}")
            self.logger.info(f"  LLM生成的理论得分: {decision.get('theoretical_factors', {})}")
            self.logger.info(f"  加权计算后的得分: {decision.get('decision_score', 0):.4f}")
            self.logger.info(f"  投票结果: {decision.get('stance', 'unknown')}")
            self.logger.info(f"  推理: {decision.get('reasoning', '')}")
        self.logger.info("="*80 + "\n")
        
        # 欧委会分析并通知投票结果
        if self.eu_commission and hasattr(self.eu_commission, 'analyze_and_communicate_voting_results'):
            voting_analysis = await self.eu_commission.analyze_and_communicate_voting_results(
                votes=votes,
                country_populations=self.country_populations,
                voting_rules=self.eu_voting_rules,
                country_anonymization_map=self.country_anonymization_map
            )
            voting_result["eu_commission_analysis"] = voting_analysis
            
            if voting_analysis.get("communications"):
                for comm in voting_analysis["communications"]:
                    country_id = comm["to"]
                    if country_id in self.country_roles:
                        self.logger.info(f"{country_id} 接收到欧委会初始投票结果通知")
        
        return voting_result
    
    async def _phase_communication(self, initial_voting: Dict) -> Dict:
        """沟通协商阶段"""
        self.current_phase = SimulationPhase.COMMUNICATION
        self.logger.info("=== 阶段5: 沟通协商 ===")
        
        communications = {
            "country_to_country": [],
            "eu_commission": [],
            "china": [],
            "china_targeted_before_final_vote": [],
            "retaliation": None
        }
        
        if self.config.enable_communication:
            # 1. 国家间沟通
            country_communications = await self._execute_country_communications(initial_voting)
            communications["country_to_country"] = country_communications
            
            # 2. 欧委会沟通
            eu_communications = await self.eu_commission.communicate_with_countries(
                list(self.country_roles.keys()), initial_voting["votes"]
            )
            communications["eu_commission"] = eu_communications
            
            # 3. 中国评估反制措施
            eu_voting_result = initial_voting.get("eu_commission_analysis", {}).get("voting_analysis", {}).get("voting_result", {})
            proposal_passed = eu_voting_result.get("passed", True)
            
            retaliation = await self.china.assess_retaliation(initial_voting["votes"], proposal_passed)
            communications["retaliation"] = retaliation
            
            # 4. 中国沟通
            all_countries = list(self.country_roles.keys())
            china_communications = await self.china.communicate_with_countries(
                all_countries, retaliation
            )
            communications["china"] = china_communications
            
            # 5. 中国针对性沟通
            if self.config.enable_china_targeted_communication and self.config.china_targeted_communications:
                china_targeted_communications = await self._execute_china_targeted_communications(initial_voting)
                communications["china_targeted_before_final_vote"] = china_targeted_communications
        
        communications["timestamp"] = datetime.now().isoformat()
        self.simulation_phases["communication"] = communications
        
        self.logger.info(f"沟通协商完成")
        return communications
    
    async def _execute_country_communications(self, initial_voting: Dict, retaliation: Dict = None) -> List[Dict]:
        """执行国家间沟通 - 使用LLM智能选择沟通目标
        
        Args:
            initial_voting: 初始投票结果
            retaliation: 中国反制措施（可选，如果未提供则生成初步的用于沟通）
        """
        communications = []
        votes = initial_voting["votes"]
        voting_details = initial_voting.get("voting_details", {})
        
        # 如果未提供反制措施，先生成初步的用于沟通
        if retaliation is None:
            eu_voting_result = initial_voting.get("eu_commission_analysis", {}).get("voting_analysis", {}).get("voting_result", {})
            proposal_passed = eu_voting_result.get("passed", True)
            retaliation = await self.china.assess_retaliation(initial_voting["votes"], proposal_passed)
            self.logger.info(f"生成了初步的中国反制措施用于国家间沟通: 触发={retaliation.get('triggered', False)}")
        
        # 记录每个国家应该收到的沟通数量
        expected_communications = {}
        for country_id in self.country_roles.keys():
            expected_communications[country_id] = 0
        
        # 准备所有国家的信息供LLM选择使用
        all_countries_info = {}
        for country_id in self.country_roles.keys():
            role = self.country_roles[country_id]
            all_countries_info[country_id] = {
                "last_vote": votes.get(country_id, "neutral"),
                "anonymized_text": role.anonymized_data.get("anonymized_text", {})
            }
        
        # 为每个国家选择沟通目标并发送沟通
        for country_id, role in self.country_roles.items():
            current_stance = votes[country_id]
            
            # 使用LLM选择沟通目标
            selected_targets = await self.select_communication_targets_action.select_targets(
                initiator_id=country_id,
                initiator_stance=current_stance,
                initiator_last_vote=votes.get(country_id, "neutral"),
                initiator_anonymized_text=role.anonymized_data.get("anonymized_text", {}),
                all_countries_info=all_countries_info,
                retaliation=retaliation,
                n=self.config.max_communications_per_round
            )
            
            if not selected_targets:
                self.logger.debug(f"{country_id} LLM未选择沟通目标，跳过")
                continue
            
            # 记录每个目标国家应该收到的沟通数
            for target_id in selected_targets:
                expected_communications[target_id] += 1
            
            # 向每个目标国家发送沟通
            for target_id in selected_targets:
                target_stance = votes[target_id]
                content = await self._generate_communication_content(
                    country_id, target_id, current_stance, target_stance, votes, retaliation
                )
                
                communication = {
                    "from": country_id,
                    "to": target_id,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
                communications.append(communication)
        
        # 记录日志：每个国家应该收到多少沟通
        for country_id, count in expected_communications.items():
            self.logger.debug(f"{country_id} 应该收到 {count} 条来自其他国家的沟通")
        
        self.logger.info(f"国家间沟通生成完成: 共{len(communications)}条沟通")
        
        return communications
    
    def _calculate_communication_urgency(self, role: EUCountryRole, votes: Dict) -> float:
        """
        计算沟通紧急程度（根据第一轮投票结果调整）
        
        Args:
            role: 国家角色
            votes: 投票结果
            
        Returns:
            沟通紧急程度 (0-1)
        """
        base_urgency = role.stance_strength
        
        features = role.anonymized_data
        political_features = features.get("political_features", {})
        
        if political_features.get("government_stability") == "high":
            base_urgency *= 1.2
        
        current_stance = votes.get(role.country_id, "neutral")
        opposition_count = sum(1 for stance in votes.values() 
                            if stance != current_stance and stance != "neutral")
        
        if opposition_count > 0:
            base_urgency *= (1 + opposition_count * 0.1)
        
        # 根据第一轮投票结果调整沟通欲望
        # 获取第一轮投票结果
        eu_voting_result = {}
        if hasattr(self, 'simulation_phases') and "initial_voting" in self.simulation_phases:
            eu_voting_result = self.simulation_phases["initial_voting"].get(
                "eu_commission_analysis", {}
            ).get("voting_analysis", {}).get("voting_result", {})
        
        proposal_passed = eu_voting_result.get("passed", True)
        
        if proposal_passed:
            # 法案通过：投反对票的国家沟通欲望更大
            if current_stance == "against":
                base_urgency *= 1.5
        else:
            # 法案否决：投赞同票的国家沟通欲望更大
            if current_stance == "support":
                base_urgency *= 1.5
        
        return min(base_urgency, 1.0)
    
    async def _generate_communication_content(self, initiator_id: str, target_id: str,
                                             initiator_stance: str, target_stance: str,
                                             initial_votes: Dict, retaliation: Dict = None) -> Dict:
        """生成沟通内容 - 使用大语言模型生成详细沟通内容
        
        Args:
            initiator_id: 发起方国家ID
            target_id: 目标方国家ID
            initiator_stance: 发起方立场
            target_stance: 目标方立场
            initial_votes: 初始投票结果
            retaliation: 中国反制措施（用于在沟通中考虑）
        """
        content_type = "persuasion" if initiator_stance != target_stance else "coordination"
        
        # 获取发起方和目标方的国家数据
        initiator_role = self.country_roles.get(initiator_id)
        target_role = self.country_roles.get(target_id)
        
        if not initiator_role or not target_role:
            # 回退到简单消息
            return {
                "type": content_type,
                "message": "无法获取国家数据，使用默认沟通内容。",
                "tone": "neutral",
                "urgency": 0.5,
                "quality": 0.5,
                "llm_generated": False
            }
        
        # 获取国家匿名化数据
        initiator_data = initiator_role.anonymized_data
        target_data = target_role.anonymized_data
        
        # 获取匿名化文本（三个维度）
        initiator_anonymized_text = initiator_data.get("anonymized_text", {})
        target_anonymized_text = target_data.get("anonymized_text", {})
        
        # 提取本国三个维度的文本内容
        initiator_market_text = initiator_anonymized_text.get("X_market (Market / Economic Interdependence)", "未知")
        initiator_political_text = initiator_anonymized_text.get("X_political (Domestic Politics and Interest Mediation)", "未知")
        initiator_institutional_text = initiator_anonymized_text.get("X_institutional (Institutions, Diplomacy, and Path Dependence)", "未知")
        
        # 提取对方国家三个维度的文本内容
        target_market_text = target_anonymized_text.get("X_market (Market / Economic Interdependence)", "未知")
        target_political_text = target_anonymized_text.get("X_political (Domestic Politics and Interest Mediation)", "未知")
        target_institutional_text = target_anonymized_text.get("X_institutional (Institutions, Diplomacy, and Path Dependence)", "未知")
        
        # 获取上次投票情况（使用初始投票）
        initiator_last_vote = initial_votes.get(initiator_id, "neutral")
        target_last_vote = initial_votes.get(target_id, "neutral")
        
        # 期望对方投的票
        desired_vote = initiator_stance
        desired_vote_text = {
            "support": "支持",
            "against": "反对",
            "abstain": "弃权",
            "neutral": "保持中立"
        }.get(initiator_stance, initiator_stance)
        
        # 生成详细的大语言模型提示词
        llm_prompt = self._generate_communication_prompt(
            initiator_id, target_id,
            initiator_market_text, initiator_political_text, initiator_institutional_text,
            target_market_text, target_political_text, target_institutional_text,
            initiator_last_vote, target_last_vote,
            initiator_stance, target_stance,
            desired_vote, desired_vote_text,
            retaliation
        )
        
        # 调用大语言模型生成沟通内容
        try:
            message = await self._call_llm_for_communication(llm_prompt)
            llm_generated = True
            quality = 0.9
        except Exception as e:
            self.logger.warning(f"LLM生成沟通内容失败: {e}，使用备用方法")
            message = self._generate_fallback_communication(
                initiator_id, target_id,
                initiator_stance, target_stance
            )
            llm_generated = False
            quality = 0.6
        
        # 确定语气
        tone = "persuasive"
        if initiator_stance == "support" and target_stance == "against":
            tone = "persuasive"
        elif initiator_stance == "against" and target_stance == "support":
            tone = "warning"
        else:
            tone = "diplomatic"
        
        # 计算紧急程度
        urgency = self._calculate_communication_urgency(initiator_role, initial_votes)
        
        return {
            "type": content_type,
            "message": message,
            "tone": tone,
            "urgency": urgency,
            "quality": quality,
            "llm_generated": llm_generated,
            "initiator_stance": initiator_stance,
            "target_stance": target_stance,
            "desired_vote": desired_vote,
            "initiator_data_summary": self._summarize_country_data(initiator_data),
            "target_data_summary": self._summarize_country_data(target_data)
        }
    
    def _generate_communication_prompt(self, initiator_id: str, target_id: str,
                                      initiator_market_text: str, initiator_political_text: str, initiator_institutional_text: str,
                                      target_market_text: str, target_political_text: str, target_institutional_text: str,
                                      initiator_last_vote: str, target_last_vote: str,
                                      initiator_stance: str, target_stance: str,
                                      desired_vote: str, desired_vote_text: str,
                                      retaliation: Dict = None) -> str:
        """生成用于LLM的沟通内容生成提示词
        
        Args:
            initiator_id: 发起方国家ID
            target_id: 目标方国家ID
            initiator_market_text: 发起方市场维度文本
            initiator_political_text: 发起方政治维度文本
            initiator_institutional_text: 发起方制度维度文本
            target_market_text: 目标方市场维度文本
            target_political_text: 目标方政治维度文本
            target_institutional_text: 目标方制度维度文本
            initiator_last_vote: 发起方上次投票
            target_last_vote: 目标方上次投票
            initiator_stance: 发起方当前立场
            target_stance: 目标方当前立场
            desired_vote: 期望对方投的票
            desired_vote_text: 期望对方投的票的文本
            retaliation: 中国反制措施（可选）
        """
        
        # 投票立场映射
        vote_mapping = {
            "support": "支持",
            "against": "反对",
            "abstain": "弃权",
            "neutral": "中立"
        }
        
        prompt = f"""
你是一位经验丰富的当前国家的外交及经济官员，代表{initiator_id}与{target_id}进行沟通。
你的目标是说服{target_id}在对华电动汽车加征关税投票中{desired_vote_text}（投{vote_mapping.get(desired_vote, desired_vote)}票）。

【背景信息】
欧盟委员会正在考虑对中国汽车征收关税。各成员国需要进行投票决定。你正在进行国家间的双边沟通。
"""
        
        # 添加中国反制措施信息
        if retaliation and retaliation.get("triggered", False):
            measures = retaliation.get("measures", [])
            measures_text = "\n".join([f"  • {m}" for m in measures]) if measures else "  无具体措施"
            target_countries = retaliation.get("target_countries", [])
            targets_text = ", ".join(target_countries) if target_countries else "无特定目标"
            severity = retaliation.get("severity", "未知")
            
            prompt += f"""
【中国反制措施】
⚠️ 中国已宣布将采取以下反制措施：
- 严重程度：{severity}
- 针对国家：{targets_text}
- 具体措施：
{measures_text}

请在沟通中充分考虑这些反制措施的影响。
"""
        elif retaliation:
            prompt += f"""
【中国反制措施】
目前中国尚未采取明确的反制措施，但存在潜在的报复风险。
请在沟通中考虑可能的经济和外交后果。
"""
        
        prompt += f"""
【本国（{initiator_id}）情况】

1. 市场与经济相互依赖维度：
{initiator_market_text}...

2. 国内政治与利益调解维度：
{initiator_political_text}...

3. 制度、外交与路径依赖维度：
{initiator_institutional_text}...

【投票情况】
- 我方上次投票：{vote_mapping.get(initiator_last_vote, initiator_last_vote)}
- 对方上次投票：{vote_mapping.get(target_last_vote, target_last_vote)}
- 我方当前立场：{vote_mapping.get(initiator_stance, initiator_stance)}
- 对方当前立场：{vote_mapping.get(target_stance, target_stance)}

【沟通目标】
希望对方投符合自己利益的票的类型。

【沟通要求】
请生成一段正式、外交且具有说服力的沟通内容，要求总字数不超过80字，尽量简短：

1. **分析对方国家利益**：根据对方国家的经济、政治、规范和战略特征，分析其在该次投票议题上的核心关切和利益所在。

2. **提出充分的理由**：
   - 如果希望对方支持关税：强调保护欧盟汽车工业、维护公平竞争、应对中国产能过剩等理由
   - 如果希望对方反对关税：强调避免贸易战、保护供应链稳定、维护消费者利益、避免反制措施等理由
0.
  3. **提供实质性利益交换或合作提议**：
   - 国家利益不能随便损失，需要衡量得失，当其投票非常关键时才会有更实质性的内容。
   - 按照本国及目标国家综合实力，权衡得失，必要时提出具体的合作项目或政策支持
   - 按照本国及目标国家综合实力，权衡得失，必要时提供技术转让、市场准入、投资合作等具体利益
   - 按照本国及目标国家综合实力，权衡得失，必要时承诺加强双边关系和在欧盟内部的协调

4. **考虑中国反制措施的影响**：
   - 根据对方国家的特征定制沟通内容
   - 回应对方可能的具体关切
   - 提出对方可能接受的解决方案

请以简洁直白的语言输出沟通内容，不要有无关的内容。总字数不超过80字。最后结尾加上希望对方投什么票
"""
        return prompt
    
    async def _call_llm_for_communication(self, prompt: str) -> str:
        """调用大语言模型生成沟通内容"""
        try:
            # 使用TheoreticalDecisionAction的_aask方法
            if hasattr(self.theoretical_decision_action, '_aask'):
                message = await self.theoretical_decision_action._aask(prompt)
                return message.strip()
            else:
                # 如果没有aask方法，创建临时Action来调用
                from metagpt.actions import Action
                temp_action = Action()
                if hasattr(temp_action, '_aask'):
                    message = await temp_action._aask(prompt)
                    return message.strip()
        except Exception as e:
            self.logger.error(f"调用LLM生成沟通内容时出错: {e}")
            raise
        return ""
    
    def _generate_fallback_communication(self, initiator_id: str, target_id: str,
                                        initiator_stance: str, target_stance: str) -> str:
        """生成备用的沟通内容（当LLM失败时使用）"""
        if initiator_stance == "support" and target_stance == "against":
            return (f"尊敬的{target_id}代表，我们理解贵国对自由贸易的坚持。然而，"
                   f"关税措施对保护欧盟汽车工业、维护公平竞争环境至关重要。"
                   f"尽管中国可能采取反制措施，欧盟将通过联合应对和市场多元化降低风险。"
                   f"我们愿意在其他贸易领域提供更多合作机会，期待您的支持。")
        elif initiator_stance == "against" and target_stance == "support":
            return (f"尊敬的{target_id}代表，关税措施可能引发贸易战，对双方经济造成严重损害。"
                   f"我们特别担心中国可能采取的反制措施，这将影响欧盟企业对华出口和全球供应链。"
                   f"建议通过对话协商解决争端，避免激怒中国，维护欧盟整体利益。")
        
    
    def _summarize_country_data(self, country_data: Dict) -> Dict:
        """总结国家关键数据"""
        economic = country_data.get("economic_features", {})
        political = country_data.get("political_features", {})
        
        return {
            "trade_dependency": economic.get("trade_dependency_china"),
            "automotive_share": economic.get("automotive_industry_share"),
            "political_orientation": political.get("political_orientation"),
            "eu_integration": political.get("eu_integration_level")
        }
    
    async def _execute_china_targeted_communications(self, initial_voting: Dict) -> List[Dict]:
        """执行中国在第二次投票前的针对性沟通"""
        communications = []
        initial_votes = initial_voting.get("votes", {})
        
        targeted_communications = self.config.china_targeted_communications
        
        if not targeted_communications:
            return communications
        
        for target_country, message_content in targeted_communications.items():
            if target_country not in self.country_roles:
                continue
            
            initial_stance = initial_votes.get(target_country, "neutral")
            
            communication = {
                "from": "China",
                "to": target_country,
                "type": "targeted_communication",
                "timestamp": datetime.now().isoformat(),
                "message": {
                    "content": message_content,
                    "initial_stance": initial_stance,
                    "communication_round": "before_final_vote",
                    "urgency": 0.9,
                    "quality": 1.0
                }
            }
            communications.append(communication)
        
        return communications
    
    async def _phase_secretary_analysis(self, communications: Dict) -> Dict:
        """秘书分析阶段 - 秘书智能体分析沟通信息并生成后续效应"""
        self.current_phase = SimulationPhase.COMMUNICATION
        self.logger.info("=== 阶段6: 秘书分析 ===")
        
        secretary_analysis = {}
        
        # 获取初始投票结果
        initial_voting = self.simulation_phases.get("initial_voting", {})
        initial_votes = initial_voting.get("votes", {})
        
        # 为每个国家的秘书智能体调用分析方法
        for country_id, secretary in self.secretary_roles.items():
            try:
                # 构建该国家的完整沟通信息
                country_specific_communications = {
                    "country_to_country": [
                        comm for comm in communications.get("country_to_country", [])
                        if comm.get("to") == country_id
                    ],
                    "eu_commission": communications.get("eu_commission", []),
                    "china_targeted": [
                        comm for comm in communications.get("china_targeted_before_final_vote", [])
                        if comm.get("to") == country_id
                    ],
                    "china_general": communications.get("china", []),
                    "retaliation": communications.get("retaliation", {})
                }
                
                # 调用秘书分析方法（传入round_name为'initial'，表示这是初始投票后的分析）
                analysis_result = await secretary.analyze_communications(
                    country_specific_communications,
                    round_name='initial'
                )
                
                secretary_analysis[country_id] = {
                    "country_id": country_id,
                    "secretary_analysis": analysis_result,
                    "initial_vote": initial_votes.get(country_id, "neutral"),
                    "communications_received": {
                        "country_to_country": len(country_specific_communications["country_to_country"]),
                        "eu_commission": len(country_specific_communications["eu_commission"]),
                        "china_targeted": len(country_specific_communications["china_targeted"]),
                        "china_general": len(country_specific_communications["china_general"])
                    }
                }
                
                # 记录日志
                effect = analysis_result.get("effect_analysis", {})
                overall = effect.get("overall_impact", "")
                if overall:
                    self.logger.info(f"{country_id} 秘书分析: {overall}")
                
            except Exception as e:
                self.logger.error(f"{country_id} 秘书分析失败: {e}")
                secretary_analysis[country_id] = {
                    "country_id": country_id,
                    "error": str(e),
                    "secretary_analysis": None,
                    "initial_vote": initial_votes.get(country_id, "neutral")
                }
        
        self.simulation_phases["secretary_analysis"] = {
            "phase": "secretary_analysis",
            "analysis_results": secretary_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"秘书分析完成: {len(secretary_analysis)}个国家")
        
        # 输出详细分析结果
        self.logger.info("\n" + "="*80)
        self.logger.info("【秘书分析详细结果】")
        self.logger.info("="*80)
        for country_id, analysis in secretary_analysis.items():
            if "error" in analysis:
                self.logger.info(f"\n【{country_id} - 分析失败】")
                self.logger.info(f"  错误: {analysis['error']}")
                continue
            
            secretary_result = analysis.get("secretary_analysis", {})
            effect = secretary_result.get("effect_analysis", {})
            initial_vote = analysis.get("initial_vote", "neutral")
            comms_received = analysis.get("communications_received", {})
            
            self.logger.info(f"\n【{country_id} - 秘书分析】")
            self.logger.info(f"  初始投票: {initial_vote}")
            self.logger.info(f"  收到沟通:")
            self.logger.info(f"    - 国家间: {comms_received.get('country_to_country', 0)}条")
            self.logger.info(f"    - 欧委会: {comms_received.get('eu_commission', 0)}条")
            self.logger.info(f"    - 中国(针对性): {comms_received.get('china_targeted', 0)}条")
            self.logger.info(f"    - 中国(一般): {comms_received.get('china_general', 0)}条")
            self.logger.info(f"  后续效应分析:")
            self.logger.info(f"    - 市场维度: {effect.get('market_effect', '')[:100]}...")
            self.logger.info(f"    - 政治维度: {effect.get('political_effect', '')[:100]}...")
            self.logger.info(f"    - 制度维度: {effect.get('institutional_effect', '')[:100]}...")
            self.logger.info(f"    - 综合影响: {effect.get('overall_impact', '')[:100]}...")
        self.logger.info("="*80 + "\n")
        
        return secretary_analysis
    
    async def _phase_final_voting(self, proposal: Dict, communications: Dict, secretary_analysis: Dict) -> Dict:
        """最终投票阶段"""
        self.current_phase = SimulationPhase.FINAL_VOTING
        self.logger.info("=== 阶段6: 最终投票 ===")
        
        votes = {}
        voting_details = {}
        
        # 提取初始投票阶段的欧委会沟通信息
        initial_voting = self.simulation_phases.get("initial_voting", {})
        initial_eu_commission_communications = initial_voting.get("eu_commission_analysis", {}).get("communications", [])
        
        # 合并初始和最终投票前的欧委会沟通
        final_eu_commission_communications = communications.get("eu_commission", [])
        merged_eu_commission_communications = initial_eu_commission_communications + final_eu_commission_communications
        
        # 合并初始和最终投票前的国家间沟通
        initial_country_communications = initial_voting.get("country_communications", [])
        final_country_communications = communications.get("country_to_country", [])
        merged_country_communications = initial_country_communications + final_country_communications
        
        # 获取中国反制措施和针对性沟通信息
        china_retaliation = communications.get("retaliation", {})
        china_targeted_communications = communications.get("china_targeted_before_final_vote", [])
        china_general_communications = communications.get("china", [])
        
        # 构建中国沟通信息（包含反制措施和针对性沟通）
        china_communication = {
            "retaliation": china_retaliation,
            "targeted_communications": china_targeted_communications,
            "general_communications": china_general_communications
        }
        
        context = {
            "proposal": proposal,
            "round": "final",
            "phase": "final_voting",
            "communications": communications
        }
        
        for country_id, role in self.country_roles.items():
            # 构建欧委会沟通信息（从合并的沟通中筛选出针对当前国家的）
            eu_commission_communication = None
            if merged_eu_commission_communications:
                # 从合并的沟通列表中筛选出针对当前国家的沟通
                country_specific_eu_communications = [
                    comm for comm in merged_eu_commission_communications
                    if comm.get("to") == country_id
                ]
                
                if country_specific_eu_communications:
                    # 使用最新的针对该国家的沟通
                    eu_commission_communication = country_specific_eu_communications[-1]
                    # 添加统计信息
                    eu_commission_communication["communications"] = merged_eu_commission_communications
                    eu_commission_communication["total_count"] = len(merged_eu_commission_communications)
            
            # 构建该国家接收到的特定沟通
            # 从合并的国家间沟通中筛选该国家作为接收者的沟通
            country_specific_communications = [
                comm for comm in merged_country_communications 
                if comm.get("to") == country_id or comm.get("target_country") == country_id
            ]
            
            # 从中国针对性沟通中筛选该国家的沟通
            china_specific_communications = [
                comm for comm in china_targeted_communications 
                if comm.get("to") == country_id
            ]
            
            # 构建完整的沟通上下文
            decision_communications = {
                "other_countries_communications": country_specific_communications,
                "eu_commission_communication": eu_commission_communication,
                "china_communication": {
                    "retaliation": china_retaliation,
                    "targeted_communications": china_specific_communications,
                    "general_communications": china_general_communications
                }
            }
            
            # 获取第一次投票的得分和结果（用于第二次投票）
            initial_details = initial_voting.get('voting_details', {}).get(country_id, {})
            initial_theory_scores = initial_details.get('theoretical_factors', {})
            initial_vote = initial_details.get('stance', '')  # 修复：使用stance而不是decision
            
            # 添加调试日志
            self.logger.info(f"{country_id} 第二次投票 - 初始数据获取:")
            self.logger.info(f"  initial_details keys: {list(initial_details.keys())}")
            self.logger.info(f"  initial_theory_scores: {initial_theory_scores}")
            self.logger.info(f"  initial_vote: {initial_vote}")
            
            decision = await role.make_theoretical_decision(
                context=context,
                voting_proposal=proposal,
                other_countries_communications=country_specific_communications,
                eu_commission_communication=eu_commission_communication,
                china_communication=decision_communications["china_communication"],
                secretary_analysis=secretary_analysis.get(country_id, {}).get("secretary_analysis"),
                initial_theory_scores=initial_theory_scores,
                initial_vote=initial_vote
            )
            votes[country_id] = decision["stance"]
            voting_details[country_id] = decision
        
        voting_result = {
            "votes": votes,
            "voting_details": voting_details,
            "proposal": proposal,
            "round": "final",
            "timestamp": datetime.now().isoformat()
        }
        
        self.simulation_phases["final_voting"] = voting_result
        
        support_count = sum(1 for vote in votes.values() if vote == "support")
        against_count = sum(1 for vote in votes.values() if vote == "against")
        abstain_count = sum(1 for vote in votes.values() if vote == "abstain")
        
        # 按投票类型分组国家
        support_countries = [country_id for country_id, vote in votes.items() if vote == "support"]
        against_countries = [country_id for country_id, vote in votes.items() if vote == "against"]
        abstain_countries = [country_id for country_id, vote in votes.items() if vote == "abstain"]
        
        # 输出详细的投票统计
        self.logger.info("\n" + "="*80)
        self.logger.info("【最终投票统计结果】")
        self.logger.info("="*80)
        self.logger.info(f"总票数: {len(votes)}票")
        self.logger.info(f"\n✅ 支持 - {support_count}票:")
        if support_countries:
            for country in support_countries:
                self.logger.info(f"    • {country}")
        else:
            self.logger.info("    （无）")
        
        self.logger.info(f"\n❌ 反对 - {against_count}票:")
        if against_countries:
            for country in against_countries:
                self.logger.info(f"    • {country}")
        else:
            self.logger.info("    （无）")
        
        self.logger.info(f"\n⚪ 弃权 - {abstain_count}票:")
        if abstain_countries:
            for country in abstain_countries:
                self.logger.info(f"    • {country}")
        else:
            self.logger.info("    （无）")
        self.logger.info("="*80 + "\n")
        
        # 输出详细的最终投票信息
        self.logger.info("\n" + "="*80)
        self.logger.info("【最终投票详细数据】")
        self.logger.info("="*80)
        for country_id, decision in voting_details.items():
            self.logger.info(f"\n【{country_id} - 最终投票】")
            self.logger.info(f"  读取的权重: {decision.get('weights_used', {})}")
            self.logger.info(f"  LLM生成的理论得分: {decision.get('theoretical_factors', {})}")
            self.logger.info(f"  加权计算后的得分: {decision.get('decision_score', 0):.4f}")
            self.logger.info(f"  投票结果: {decision.get('stance', 'unknown')}")
            self.logger.info(f"  推理: {decision.get('reasoning', '')}")
        self.logger.info("="*80 + "\n")
        
        return voting_result
    
    async def _phase_weight_optimization(self, initial_voting: Dict, final_voting: Dict) -> Dict:
        """权重优化阶段"""
        self.current_phase = SimulationPhase.WEIGHT_OPTIMIZATION
        
        # 检查是否启用了Ordered Probit模型
        if self.ordered_probit_params:
            self.logger.info("=== 阶段7: 权重优化（已跳过）===")
            self.logger.info("Ordered Probit模型已启用，使用MLE估计的权重，跳过传统权重优化")
            
            optimization = {
                "enabled": False,
                "skipped": True,
                "reason": "Ordered Probit模型已启用，使用MLE估计的权重",
                "timestamp": datetime.now().isoformat()
            }
            
            self.simulation_phases["weight_optimization"] = optimization
            return optimization
        
        # 传统权重优化（仅在未启用Ordered Probit时执行）
        self.logger.info("=== 阶段7: 权重优化 ===")
        
        optimization_results = {}
        
        if self.config.enable_weight_optimization:
            real_voting_data = self.real_voting_data
            
            for country_id in self.country_roles.keys():
                if country_id not in real_voting_data:
                    continue
                
                actual_votes = real_voting_data[country_id]
                simulated_votes = [
                    initial_voting["votes"].get(country_id, "neutral"),
                    final_voting["votes"].get(country_id, "neutral")
                ]
                
                # 根据配置决定是否使用缓存的理论得分
                if self.config.use_cached_scores_for_optimization:
                    # 直接从缓存文件读取理论得分（不依赖上下文哈希匹配）
                    import json
                    from pathlib import Path
                    
                    cache_path = Path(__file__).parent / "cache" / f"theory_scores_{country_id}.json"
                    
                    initial_theory_scores = None
                    final_theory_scores = None
                    
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached_data = json.load(f)
                            
                            # 直接读取初始轮和最终轮的理论得分
                            if "initial" in cached_data and "theory_scores" in cached_data["initial"]:
                                initial_theory_scores = cached_data["initial"]["theory_scores"]
                                self.logger.info(f"{country_id} 权重优化：从缓存文件读取初始轮理论得分: {initial_theory_scores}")
                            else:
                                self.logger.warning(f"{country_id} 权重优化：缓存文件中无初始轮数据")
                            
                            if "final" in cached_data and "theory_scores" in cached_data["final"]:
                                final_theory_scores = cached_data["final"]["theory_scores"]
                                self.logger.info(f"{country_id} 权重优化：从缓存文件读取最终轮理论得分: {final_theory_scores}")
                            else:
                                self.logger.warning(f"{country_id} 权重优化：缓存文件中无最终轮数据")
                            
                    except FileNotFoundError:
                        self.logger.warning(f"{country_id} 权重优化：缓存文件不存在: {cache_path}")
                    except Exception as e:
                        self.logger.error(f"{country_id} 权重优化：读取缓存文件失败: {e}")
                    
                    # 如果缓存读取失败，回退到使用投票详情中的理论得分
                    if not initial_theory_scores:
                        initial_theory_scores = initial_voting["voting_details"].get(country_id, {}).get("theoretical_factors", {})
                        self.logger.warning(f"{country_id} 权重优化：使用投票详情中的初始轮理论得分")
                    
                    if not final_theory_scores:
                        final_theory_scores = final_voting["voting_details"].get(country_id, {}).get("theoretical_factors", {})
                        self.logger.warning(f"{country_id} 权重优化：使用投票详情中的最终轮理论得分")
                else:
                    # 不使用缓存，直接使用投票详情中的理论得分
                    initial_theory_scores = initial_voting["voting_details"].get(country_id, {}).get("theoretical_factors", {})
                    final_theory_scores = final_voting["voting_details"].get(country_id, {}).get("theoretical_factors", {})
                    self.logger.info(f"{country_id} 权重优化：使用投票详情中的理论得分（未启用缓存）")
                
                current_accuracy = sum(1 for sim, actual in zip(simulated_votes, actual_votes) 
                                     if sim == actual) / len(actual_votes)
                
                if current_accuracy < 1.0:
                    optimized_weights = self._optimize_weights_for_country(
                        country_id, actual_votes, simulated_votes, 
                        initial_theory_scores, final_theory_scores
                    )
                    
                    optimized_accuracy = self._validate_optimized_weights(
                        country_id, initial_theory_scores, final_theory_scores, 
                        optimized_weights, actual_votes
                    )
                    
                    optimization_results[country_id] = {
                        "original_weights": self.country_roles[country_id].theory_weights.copy(),
                        "optimized_weights": optimized_weights,
                        "actual_votes": actual_votes,
                        "simulated_votes": simulated_votes,
                        "original_accuracy": current_accuracy,
                        "optimized_accuracy": optimized_accuracy,
                        "improvement": optimized_accuracy - current_accuracy
                    }
                    
                    self.country_roles[country_id].update_theory_weights(optimized_weights)
        
        optimization = {
            "enabled": self.config.enable_weight_optimization,
            "results": optimization_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.simulation_phases["weight_optimization"] = optimization
        
        # 输出详细的权重优化信息
        self.logger.info("\n" + "="*80)
        self.logger.info("【权重优化详细数据】")
        self.logger.info("="*80)
        for country_id, result in optimization_results.items():
            self.logger.info(f"\n【{country_id} - 权重优化】")
            self.logger.info(f"  真实投票结果: {result['actual_votes']}")
            self.logger.info(f"  模拟投票结果: {result['simulated_votes']}")
            self.logger.info(f"  初始权重: {result['original_weights']}")
            self.logger.info(f"  优化后权重: {result['optimized_weights']}")
            self.logger.info(f"  初始准确率: {result['original_accuracy']:.2%}")
            self.logger.info(f"  优化后准确率: {result['optimized_accuracy']:.2%}")
            self.logger.info(f"  准确率提升: {result['improvement']:.2%}")
            self.logger.info(f"  初始轮理论得分: {initial_theory_scores}")
            self.logger.info(f"  最终轮理论得分: {final_theory_scores}")
        self.logger.info("="*80 + "\n")
        
        if self.config.enable_weight_optimization and optimization_results:
            self._save_country_theory_weights()
        
        return optimization
    
    def _optimize_weights_for_country(self, country_id: str, actual_votes: List[str],
                                    simulated_votes: List[str],
                                    initial_theory_scores: Dict, final_theory_scores: Dict) -> Dict[str, float]:
        """为特定国家优化权重 - 新的评分范围(-1到1)"""
        vote_targets = []
        for vote in actual_votes:
            if vote == "support":
                vote_targets.append(0.8)  # 支持对应正分
            elif vote == "against":
                vote_targets.append(-0.8)  # 反对对应负分
            else:
                vote_targets.append(0.0)  # 弃权对应中性值
        
        # 首先尝试蒙特卡洛优化
        best_weights = self.weight_optimizer.monte_carlo_weight_optimization(
            country_id, initial_theory_scores, final_theory_scores, vote_targets, actual_votes
        )
        
        # 如果蒙特卡洛优化失败，尝试贝叶斯优化
        if not best_weights:
            self.logger.warning(f"{country_id} 蒙特卡洛优化未找到完美解，尝试贝叶斯优化")
            best_weights = self.weight_optimizer.bayesian_weight_optimization(
                country_id, initial_theory_scores, final_theory_scores, vote_targets, actual_votes
            )
        
        # 如果贝叶斯优化失败，回退到梯度下降方法
        if not best_weights:
            self.logger.warning(f"{country_id} 贝叶斯优化失败，使用梯度下降方法")
            best_weights = self.weight_optimizer.solve_weight_optimization(
                initial_theory_scores, final_theory_scores, vote_targets, actual_votes
            )
        
        # 确保权重在合理范围内并归一化
        if best_weights:
            best_weights = self.weight_optimizer.normalize_and_constrain_weights(best_weights)
        
        return best_weights
    
    def _validate_optimized_weights(self, country_id: str, initial_theory_scores: Dict,
                                  final_theory_scores: Dict, optimized_weights: Dict[str, float],
                                  actual_votes: List[str]) -> float:
        """验证优化后的权重"""
        validated_accuracy = self.weight_optimizer.calculate_accuracy_with_weights(
            initial_theory_scores, final_theory_scores, optimized_weights, actual_votes
        )
        self.logger.info(f"{country_id} 权重验证: 实际准确率{validated_accuracy:.3f}")
        return validated_accuracy
    
    async def _phase_analysis(self, proposal: Dict, initial_voting: Dict,
                            communications: Dict, final_voting: Dict,
                            weight_optimization: Dict) -> Dict:
        """分析总结阶段"""
        self.current_phase = SimulationPhase.ANALYSIS
        self.logger.info("=== 阶段8: 分析总结 ===")
        
        analysis = {
            "voting_pattern_analysis": self._analyze_voting_patterns(initial_voting, final_voting),
            "communication_analysis": self._analyze_communications(communications),
            "weight_optimization_analysis": self._analyze_weight_optimization(weight_optimization),
            "accuracy_analysis": self._calculate_accuracy(initial_voting, final_voting),
            "simulation_effectiveness": self._evaluate_simulation_effectiveness()
        }
        
        analysis["timestamp"] = datetime.now().isoformat()
        self.simulation_phases["analysis"] = analysis
        
        return analysis
    
    def _analyze_voting_patterns(self, initial_voting: Dict, final_voting: Dict) -> Dict:
        """分析投票模式"""
        initial_votes = initial_voting["votes"]
        final_votes = final_voting["votes"]
        
        stance_changes = {}
        for country_id in self.country_roles.keys():
            initial_stance = initial_votes.get(country_id, "neutral")
            final_stance = final_votes.get(country_id, "neutral")
            
            if initial_stance != final_stance:
                stance_changes[country_id] = {
                    "from": initial_stance,
                    "to": final_stance
                }
        
        return {
            "stance_changes": stance_changes,
            "change_rate": len(stance_changes) / len(self.country_roles),
            "stability_index": 1 - (len(stance_changes) / len(self.country_roles))
        }
    
    def _analyze_communications(self, communications: Dict) -> Dict:
        """分析沟通模式"""
        total_comms = (
            len(communications.get("country_to_country", [])) +
            len(communications.get("eu_commission", [])) +
            len(communications.get("china", []))
        )
        
        return {
            "total_communications": total_comms,
            "country_to_country": len(communications.get("country_to_country", [])),
            "eu_commission": len(communications.get("eu_commission", [])),
            "china": len(communications.get("china", [])),
            "retaliation_triggered": communications.get("retaliation", {}).get("triggered", False)
        }
    
    def _analyze_weight_optimization(self, weight_optimization: Dict) -> Dict:
        """分析权重优化效果"""
        if not weight_optimization.get("enabled", False):
            return {"enabled": False, "reason": "权重优化未启用"}
        
        results = weight_optimization["results"]
        
        if not results:
            return {"enabled": True, "no_results": True}
        
        total_improvement = sum(r["improvement"] for r in results.values())
        avg_improvement = total_improvement / len(results)
        
        return {
            "enabled": True,
            "countries_optimized": len(results),
            "average_improvement": avg_improvement,
            "total_improvement": total_improvement
        }
    
    def _calculate_accuracy(self, initial_voting: Dict, final_voting: Dict) -> Dict:
        """计算预测准确率"""
        accuracy_results = {}
        
        real_voting_data = self.real_voting_data
        
        for country_id in self.country_roles.keys():
            if country_id in real_voting_data:
                actual_votes = real_voting_data[country_id]
            elif country_id in self.config.actual_voting_data:
                actual_votes = self.config.actual_voting_data[country_id]
            else:
                actual_votes = ["support", "support"]
            
            simulated_votes = [
                initial_voting["votes"].get(country_id, "neutral"),
                final_voting["votes"].get(country_id, "neutral")
            ]
            
            correct_predictions = sum(1 for sim, actual in zip(simulated_votes, actual_votes) 
                                    if sim == actual)
            accuracy = correct_predictions / len(actual_votes)
            
            accuracy_results[country_id] = {
                "actual_votes": actual_votes,
                "simulated_votes": simulated_votes,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy
            }
        
        if accuracy_results:
            total_correct = sum(r["correct_predictions"] for r in accuracy_results.values())
            total_predictions = sum(len(r["actual_votes"]) for r in accuracy_results.values())
            overall_accuracy = total_correct / total_predictions
        else:
            overall_accuracy = 0.0
        
        return {
            "country_accuracy": accuracy_results,
            "overall_accuracy": overall_accuracy,
            "countries_evaluated": len(accuracy_results)
        }
    
    def _evaluate_simulation_effectiveness(self) -> Dict:
        """评估模拟效果"""
        duration = (self.simulation_end_time or datetime.now()) - self.simulation_start_time
        
        return {
            "simulation_duration": str(duration),
            "phases_completed": len(self.simulation_phases),
            "countries_simulated": len(self.country_roles),
            "communication_enabled": self.config.enable_communication,
            "weight_optimization_enabled": self.config.enable_weight_optimization
        }
    
    async def _generate_comprehensive_report(self, proposal: Dict, initial_voting: Dict,
                                         communications: Dict, final_voting: Dict,
                                         weight_optimization: Dict, analysis: Dict) -> Dict:
        """生成综合报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 使用序列化方法避免循环引用
        serialized_phases = self._serialize_simulation_phases()
        
        report = {
            "simulation_metadata": {
                "timestamp": timestamp,
                "simulation_id": f"EU_Tariff_Sim_{timestamp}",
                "config": self.config.__dict__,
                "countries_participated": list(self.country_roles.keys()),
                "simulation_duration": str(self.simulation_end_time - self.simulation_start_time),
                "phases_completed": list(self.simulation_phases.keys())
            },
            "simulation_phases": serialized_phases,
            "analysis": analysis,
            "key_findings": self._generate_key_findings(analysis),
            "recommendations": self._generate_recommendations(analysis)
        }
        
        if self.config.save_intermediate_results:
            report_path = self.results_dir / f"simulation_report_{timestamp}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"综合报告已保存到: {report_path}")
        
        # 生成投票结果表格和CSV文件
        self._generate_voting_table_and_csv(initial_voting, final_voting)
        
        return report
    
    def _generate_key_findings(self, analysis: Dict) -> List[str]:
        """生成关键发现"""
        findings = []
        
        accuracy = analysis.get("accuracy_analysis", {})
        overall_accuracy = accuracy.get("overall_accuracy", 0)
        findings.append(f"整体预测准确率: {overall_accuracy:.1%}")
        
        voting_patterns = analysis.get("voting_pattern_analysis", {})
        change_rate = voting_patterns.get("change_rate", 0)
        findings.append(f"立场变化率: {change_rate:.1%}")
        
        comm_analysis = analysis.get("communication_analysis", {})
        total_comms = comm_analysis.get("total_communications", 0)
        findings.append(f"总沟通次数: {total_comms}")
        
        return findings
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        accuracy = analysis.get("accuracy_analysis", {})
        overall_accuracy = accuracy.get("overall_accuracy", 0)
        
        if overall_accuracy < 0.8:
            recommendations.append("建议进一步优化理论权重以提高预测准确率")
        
        return recommendations
    
    def _serialize_simulation_phases(self) -> Dict:
        """序列化模拟阶段数据，移除不可序列化的对象和循环引用
        
        Returns:
            可序列化的模拟阶段数据字典
        """
        serialized_phases = {}
        
        for phase_name, phase_data in self.simulation_phases.items():
            try:
                # 尝试序列化当前阶段数据
                # 如果失败，则提取可序列化的部分
                serialized_data = self._serialize_data_safely(phase_data)
                serialized_phases[phase_name] = serialized_data
            except Exception as e:
                self.logger.warning(f"序列化阶段 {phase_name} 失败: {e}")
                # 提取基本信息
                serialized_phases[phase_name] = {
                    "phase": phase_name,
                    "error": f"序列化失败: {str(e)}",
                    "timestamp": phase_data.get("timestamp", "")
                }
        
        return serialized_phases
    
    def _serialize_data_safely(self, data: Any) -> Any:
        """安全地序列化数据，移除不可序列化的对象
        
        Args:
            data: 要序列化的数据
            
        Returns:
            序列化后的数据
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # 跳过某些已知不可序列化的键
                if key in ['__dict__', '__weakref__', '__module__', '__doc__']:
                    continue
                
                # 尝试序列化值
                try:
                    # 尝试JSON序列化测试
                    json.dumps({key: value})
                    result[key] = self._serialize_data_safely(value)
                except (TypeError, ValueError):
                    # 如果无法序列化，尝试转换为字符串
                    if isinstance(value, (str, int, float, bool, type(None))):
                        result[key] = value
                    elif hasattr(value, '__dict__'):
                        # 对于对象，提取其__dict__
                        try:
                            result[key] = str(value)
                        except:
                            result[key] = f"<unserializable object: {type(value).__name__}>"
                    else:
                        result[key] = str(value)
            return result
        
        elif isinstance(data, (list, tuple)):
            result = []
            for item in data:
                try:
                    # 尝试序列化测试
                    json.dumps([item])
                    result.append(self._serialize_data_safely(item))
                except (TypeError, ValueError):
                    if isinstance(item, (str, int, float, bool, type(None))):
                        result.append(item)
                    elif hasattr(item, '__dict__'):
                        result.append(str(item))
                    else:
                        result.append(f"<unserializable: {type(item).__name__}>")
            return result
        
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        
        else:
            # 对于其他类型，转换为字符串
            return str(data)
    
    def _generate_voting_table_and_csv(self, initial_voting: Dict, final_voting: Dict):
        """生成投票结果表格并保存为CSV文件
        
        Args:
            initial_voting: 初始投票结果
            final_voting: 最终投票结果
        """
        self.logger.info("=== 生成投票结果表格 ===")
        
        # 准备表格数据
        table_data = []
        csv_data = []
        
        # 获取所有参与的国家
        countries = list(self.country_roles.keys())
        
        # 获取Ordered Probit阈值（如果存在）
        alpha1 = self.ordered_probit_params.get('alpha1', 'N/A') if self.ordered_probit_params else 'N/A'
        alpha2 = self.ordered_probit_params.get('alpha2', 'N/A') if self.ordered_probit_params else 'N/A'
        
        for country_id in countries:
            # 获取初始投票数据
            initial_details = initial_voting.get('voting_details', {}).get(country_id, {})
            initial_theory = initial_details.get('theoretical_factors', {})
            initial_score = initial_details.get('decision_score', 0)
            initial_stance = initial_details.get('stance', 'N/A')
            initial_real = self.real_voting_data.get(country_id, ['N/A', 'N/A'])[0]
            
            # 获取最终投票数据
            final_details = final_voting.get('voting_details', {}).get(country_id, {})
            final_theory = final_details.get('theoretical_factors', {})
            final_score = final_details.get('decision_score', 0)
            final_stance = final_details.get('stance', 'N/A')
            final_real = self.real_voting_data.get(country_id, ['N/A', 'N/A'])[1]
            
            # 获取权重
            if self.ordered_probit_params and country_id in self.ordered_probit_params.get('country_weights', {}):
                weights = self.ordered_probit_params['country_weights'][country_id]
            else:
                weights = self.country_roles[country_id].theory_weights
            
            # 表格行数据（为了显示美观，使用较短的数值格式）
            table_row = [
                country_id,  # 国家
                f"{initial_theory.get('x_market', 0):.3f}",  # 第一次-理论市场
                f"{initial_theory.get('x_political', 0):.3f}",  # 第一次-理论政治
                f"{initial_theory.get('x_institutional', 0):.3f}",  # 第一次-理论制度
                f"{initial_score:.4f}",  # 第一次-加权得分
                initial_stance[:3],  # 第一次-模拟结果（缩写）
                initial_real[:3],  # 第一次-真实结果（缩写）
                f"{final_theory.get('x_market', 0):.3f}",  # 第二次-理论市场
                f"{final_theory.get('x_political', 0):.3f}",  # 第二次-理论政治
                f"{final_theory.get('x_institutional', 0):.3f}",  # 第二次-理论制度
                f"{final_score:.4f}",  # 第二次-加权得分
                final_stance[:3],  # 第二次-模拟结果（缩写）
                final_real[:3],  # 第二次-真实结果（缩写）
                f"{weights.get('x_market', 0):.3f}",  # 权重-市场
                f"{weights.get('x_political', 0):.3f}",  # 权重-政治
                f"{weights.get('x_institutional', 0):.3f}",  # 权重-制度
                f"{alpha1:.4f}" if isinstance(alpha1, (int, float)) else alpha1,  # 阈值α1
                f"{alpha2:.4f}" if isinstance(alpha2, (int, float)) else alpha2  # 阈值α2
            ]
            
            table_data.append(table_row)
            
            # CSV行数据（使用完整数值）
            # 获取初始和最终投票的概率信息
            initial_probs = initial_details.get('probit_probabilities', {})
            final_probs = final_details.get('probit_probabilities', {})
            
            csv_row = [
                country_id,
                initial_theory.get('x_market', 0),
                initial_theory.get('x_political', 0),
                initial_theory.get('x_institutional', 0),
                initial_score,
                initial_probs.get('support', 0),  # 第一次赞成概率
                initial_probs.get('abstain', 0),  # 第一次弃权概率
                initial_probs.get('against', 0),   # 第一次反对概率
                initial_stance,
                initial_real,
                final_theory.get('x_market', 0),
                final_theory.get('x_political', 0),
                final_theory.get('x_institutional', 0),
                final_score,
                final_probs.get('support', 0),   # 第二次赞成概率
                final_probs.get('abstain', 0),   # 第二次弃权概率
                final_probs.get('against', 0),    # 第二次反对概率
                final_stance,
                final_real,
                weights.get('x_market', 0),
                weights.get('x_political', 0),
                weights.get('x_institutional', 0),
                alpha1,
                alpha2
            ]
            
            csv_data.append(csv_row)
        
        # 定义表格列标题
        table_headers = [
            "国家",
            "第一次\n市场", "第一次\n政治", "第一次\n制度",
            "第一次\n加权", "第一次\n模拟", "第一次\n真实",
            "第二次\n市场", "第二次\n政治", "第二次\n制度",
            "第二次\n加权", "第二次\n模拟", "第二次\n真实",
            "权重\n市场", "权重\n政治", "权重\n制度",
            "阈值\nα1", "阈值\nα2"
        ]
        
        # 定义CSV列标题
        csv_headers = [
            "国家", 
            "第一次-理论市场", "第一次-理论政治", "第一次-理论制度",
            "第一次-加权得分", "第一次赞成概率", "第一次弃权概率", "第一次反对概率",
            "第一次-模拟结果", "第一次-真实结果",
            "第二次-理论市场", "第二次-理论政治", "第二次-理论制度",
            "第二次-加权得分", "第二次赞成概率", "第二次弃权概率", "第二次反对概率",
            "第二次-模拟结果", "第二次-真实结果",
            "权重-市场", "权重-政治", "权重-制度",
            "阈值α1", "阈值α2"
        ]
        
        # 在终端输出表格
        self.logger.info("\n" + "="*120)
        self.logger.info("【投票结果汇总表格】")
        self.logger.info("="*120)
        
        table_str = tabulate(
            table_data,
            headers=table_headers,
            tablefmt="grid",
            numalign="right",
            stralign="center"
        )
        print(table_str)
        
        # 保存为CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"voting_table_{timestamp}.csv"
        
        # 创建DataFrame并保存
        df = pd.DataFrame(csv_data, columns=csv_headers)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"\n投票结果表格已保存到: {csv_path}")
        self.logger.info("="*120 + "\n")


async def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = SimulationConfig(
        selected_countries=["Germany", "France", "Denmark", "Italy", "Netherland", "Spain", "Ireland", "Lithuania"],
        enable_weight_optimization=True,
        enable_communication=True,
        enable_visualization=True,             
        save_intermediate_results=True,
        generate_detailed_report=True,
        load_saved_weights=True,
        enable_china_targeted_communication=True,
        use_cached_scores_for_voting=True,  # 启用带哈希验证的缓存，避免重复调用LLM
        china_targeted_communications={
            "Germany": "",
            "Spain": "Spanish Prime Minister Pedro Sánchez paid an official visit to China from September 8 to 11, 2024. The visit aimed to further promote bilateral relations between China and Spain, deepen cooperation in economic, trade, cultural, and tourism fields. Sánchez also expressed the willingness to resolve trade disputes through dialogue, emphasizing that both sides should seek consensus based on the principle of mutual benefit and win-win.此访期间，双方签署了绿色发展等领域多项合作协议，展现了双方合作的巨大潜力和光明前景。希望双方加强人文交流，深化经贸、新能源汽车等领域合作，愿为中国企业提供良好环境。西中双方都致力于维护世界和平、捍卫多边主义。支持自由贸易和市场开放原则，不赞同打贸易战，愿继续为促进欧中关系健康发展发挥积极作用。",
            "Ireland": """ """
        }
    )
    
    simulation = EUTariffSimulation(config)
    final_report = await simulation.run_complete_simulation()
    
    print("\n" + "="*60)
    print("🎯 欧盟关税模拟系统")
    print("="*60)
    
    metadata = final_report["simulation_metadata"]
    analysis = final_report["analysis"]
    
    print(f"🌍 参与国家: {metadata['countries_participated']}")
    print(f"⏱️  模拟时长: {metadata['simulation_duration']}")
    print(f"🎯 整体准确率: {analysis['accuracy_analysis']['overall_accuracy']:.1%}")
    
    initial_votes = final_report["simulation_phases"]["initial_voting"]["votes"]
    final_votes = final_report["simulation_phases"]["final_voting"]["votes"]
    
    initial_support = sum(1 for v in initial_votes.values() if v == "support")
    final_support = sum(1 for v in final_votes.values() if v == "support")
    
    print(f"\n📊 投票结果:")
    print(f"  初始投票: 支持{initial_support}票")
    print(f"  最终投票: 支持{final_support}票")
    
    print(f"\n🔍 关键发现:")
    for finding in final_report["key_findings"]:
        print(f"  • {finding}")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
