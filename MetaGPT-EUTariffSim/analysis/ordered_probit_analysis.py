"""
Ordered Probit模型完整分析流程

功能：
1. 从cache文件读取LLM理论得分
2. 从real_vote.xlsx读取实际投票数据
3. 使用MLE估计国家权重和全局阈值
4. 生成预测并评估模型
5. 创建6类可视化图表
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tabulate import tabulate

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ordered_probit_mle import OrderedProbitMLE
from utils.probit_visualizer import ProbitVisualizer


class OrderedProbitAnalysis:
    """Ordered Probit模型分析器"""
    
    def __init__(self, cache_dir: str = "../cache", 
                 vote_data_path: str = None,
                 results_dir: str = "results/probit"):
        """
        初始化分析器
        
        参数:
            cache_dir: 缓存目录，存放LLM理论得分
            vote_data_path: 实际投票数据文件路径（如果为None，则使用默认路径）
            results_dir: 结果输出目录
        """
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置缓存目录的绝对路径
        if not os.path.isabs(cache_dir):
            self.cache_dir = os.path.join(script_dir, cache_dir)
        else:
            self.cache_dir = cache_dir
        
        # 设置投票数据文件路径的绝对路径
        if vote_data_path is None:
            # 默认路径：脚本所在目录的上级目录
            self.vote_data_path = os.path.join(script_dir, "..", "real_vote.xlsx")
        elif not os.path.isabs(vote_data_path):
            self.vote_data_path = os.path.join(script_dir, vote_data_path)
        else:
            self.vote_data_path = vote_data_path
        
        # 设置结果目录的绝对路径
        if not os.path.isabs(results_dir):
            self.results_dir = os.path.join(script_dir, results_dir)
        else:
            self.results_dir = results_dir
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 初始化模型和可视化器
        self.mle_model = None
        self.visualizer = ProbitVisualizer(output_dir=results_dir)
        
        # 国家名称映射（英文 <-> 中文）
        self.country_name_mapping = {
            'Germany': '德国',
            'France': '法国',
            'Italy': '意大利',
            'Spain': '西班牙',
            'Poland': '波兰',
            'The Netherlands': '荷兰',  # Excel中写的是Netherland
            'Netherlands': '荷兰',  # 兼容旧数据
            'Netherland': '荷兰',
            'Denmark': '丹麦',
            'Ireland': '爱尔兰',
            'Lithuania': '立陶宛',
            'Hungary': '匈牙利',
            'Sweden': '瑞典',
            'Greece': '希腊',
            'Finland': '芬兰',
            'Belgium': '比利时',
            'Czechia': '捷克',
            'Austria': '奥地利',
            'Portugal': '葡萄牙',
            'Romania': '罗马尼亚',
            'Bulgaria': '保加利亚',
            'Croatia': '克罗地亚',
            'Slovakia': '斯洛伐克',
            'Slovenia': '斯洛文尼亚',
            'Latvia': '拉脱维亚',
            'Estonia': '爱沙尼亚',
            'Cyprus': '塞浦路斯',
            'Luxembourg': '卢森堡',
            'Malta': '马耳他'
        }
        
        # 反向映射（中文 -> 英文）
        self.cn_to_en_mapping = {v: k for k, v in self.country_name_mapping.items()}
        
        # 完整的国家列表
        self.all_countries = list(self.country_name_mapping.keys())
        
        # 动态确定分析的国家列表（只分析有理论得分文件的国家）
        self.countries = self._get_countries_with_theory_scores()
        self.country_to_idx = {country: idx for idx, country in enumerate(self.countries)}
        
        print(f"初始化完成，共{len(self.countries)}个国家（仅分析有理论得分的国家）")
        print(f"分析国家列表: {', '.join(self.countries)}")
    
    def _get_countries_with_theory_scores(self) -> List[str]:
        """
        获取有理论得分文件的国家列表
        
        返回:
            有理论得分的国家列表（英文名）
        """
        countries_with_scores = []
        
        # 首先扫描缓存目录中所有的理论得分文件
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) 
                          if f.startswith('theory_scores_') and f.endswith('.json')]
            
            # 从文件名中提取国家名
            for cache_file in cache_files:
                # 移除前缀和后缀
                country_name = cache_file.replace('theory_scores_', '').replace('.json', '')
                
                # 如果文件名就是标准国家名
                if country_name in self.all_countries:
                    countries_with_scores.append(country_name)
                # 如果文件名是 Netherland，映射到 Netherlands
                elif country_name == 'Netherland':
                    countries_with_scores.append('Netherlands')
                # 否则，尝试通过反向映射找到标准名
                elif country_name in self.cn_to_en_mapping:
                    countries_with_scores.append(self.cn_to_en_mapping[country_name])
        
        return sorted(countries_with_scores)
    
    def load_theory_scores(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        从cache目录读取LLM理论得分（包含initial和final）
        
        返回:
            元组 (initial_scores, final_scores)
            initial_scores: 第一轮理论得分字典 {country_name: np.array([t1, t2, t3])}
            final_scores: 第二轮理论得分字典 {country_name: np.array([t1, t2, t3])}
        """
        print("\n正在加载LLM理论得分...")
        
        initial_scores = {}
        final_scores = {}
        loaded_count = 0
        
        for country in self.countries:
            # 尝试查找缓存文件（处理 Netherland -> Netherlands 的特殊情况）
            cache_file = os.path.join(self.cache_dir, f"theory_scores_{country}.json")
            
            # 如果标准文件不存在，尝试查找 Netherland（针对 Netherlands）
            if not os.path.exists(cache_file) and country == 'Netherlands':
                cache_file_alt = os.path.join(self.cache_dir, "theory_scores_Netherland.json")
                if os.path.exists(cache_file_alt):
                    cache_file = cache_file_alt
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取initial理论得分
                initial_data = data.get('initial', {}).get('theory_scores', {})
                t1_init = initial_data.get('x_market', 0.0)
                t2_init = initial_data.get('x_political', 0.0)
                t3_init = initial_data.get('x_institutional', 0.0)
                
                # 提取final理论得分
                final_data = data.get('final', {}).get('theory_scores', {})
                t1_final = final_data.get('x_market', 0.0)
                t2_final = final_data.get('x_political', 0.0)
                t3_final = final_data.get('x_institutional', 0.0)
                
                # 验证得分范围（应该在-3到3之间）
                for phase_name, scores_list in [('initial', [t1_init, t2_init, t3_init]), 
                                               ('final', [t1_final, t2_final, t3_final])]:
                    for i, score in enumerate(scores_list):
                        if not isinstance(score, (int, float)):
                            print(f"警告: {country} 的{phase_name}理论得分{i}不是数值类型，使用默认值0.0")
                            scores_list[i] = 0.0
                        elif score < -3.0 or score > 3.0:
                            print(f"警告: {country} 的{phase_name}理论得分{i}={score:.4f}超出范围[-3, 3]，裁剪到范围内")
                            scores_list[i] = max(-3.0, min(3.0, score))
                
                initial_scores[country] = np.array([t1_init, t2_init, t3_init])
                final_scores[country] = np.array([t1_final, t2_final, t3_final])
                
                loaded_count += 1
                print(f"  {country}: initial={initial_scores[country]}, final={final_scores[country]}")
            else:
                print(f"警告: 未找到{country}的缓存文件，使用默认值0.0")
                initial_scores[country] = np.array([0.0, 0.0, 0.0])
                final_scores[country] = np.array([0.0, 0.0, 0.0])
        
        print(f"成功加载{loaded_count}/{len(self.countries)}个国家的理论得分")
        return initial_scores, final_scores
    
    def load_actual_votes(self) -> pd.DataFrame:
        """
        从Excel文件读取实际投票数据
        
        返回:
            DataFrame包含投票数据
        """
        print("\n正在加载实际投票数据...")
        
        if not os.path.exists(self.vote_data_path):
            raise FileNotFoundError(f"投票数据文件不存在: {self.vote_data_path}")
        
        df = pd.read_excel(self.vote_data_path)
        print(f"投票数据包含{len(df)}条记录")
        
        # 显示数据结构
        print("\n投票数据列名:", df.columns.tolist())
        print("\n前5行数据:")
        print(df.head())
        
        return df
    
    def prepare_training_data(self, initial_scores: Dict[str, np.ndarray],
                              final_scores: Dict[str, np.ndarray],
                              vote_df: pd.DataFrame) -> List[Dict]:
        """
        准备训练数据（使用不同轮次的理论得分）
        
        参数:
            initial_scores: 第一轮理论得分字典
            final_scores: 第二轮理论得分字典
            vote_df: 投票数据DataFrame
        
        返回:
            训练数据列表
        """
        print("\n正在准备训练数据...")
        
        training_data = []
        
        # 投票映射
        vote_mapping = {
            'oppose': 0, '反对': 0, 'no': 0, 'against': 0, '不支持': 0,
            'abstain': 1, '弃权': 1, 'abs': 1,
            'approve': 2, '赞成': 2, 'yes': 2, 'for': 2, '支持': 2,
            '0': 0, '1': 1, '2': 2
        }
        
        # 处理Excel文件：第一列是英文国家名，第二列是中文名，后面列是各次投票
        for col_idx in range(2, len(vote_df.columns)):
            vote_round = vote_df.columns[col_idx]
            print(f"\n处理投票轮次: {vote_round}")
            
            # 根据投票轮次选择使用哪个理论得分
            # 第一列（索引2）是第一次投票，使用 initial
            # 第二列（索引3）是第二次投票，使用 final
            if col_idx == 2:
                current_scores = initial_scores
                print(f"使用 INITIAL 理论得分")
            elif col_idx == 3:
                current_scores = final_scores
                print(f"使用 FINAL 理论得分")
            else:
                # 如果有更多次投票，默认使用 final
                current_scores = final_scores
                print(f"使用 FINAL 理论得分（默认）")
            
            for row_idx, row in vote_df.iterrows():
                # 获取国家名（优先使用第一列的英文，如果为空则使用第二列的中文）
                country_raw = row.iloc[0]
                country_cn_raw = row.iloc[1] if len(row) > 1 else None
                
                country = None
                
                # 尝试从英文列获取国家名
                if not pd.isna(country_raw) and isinstance(country_raw, str):
                    country_en = country_raw.strip().title()
                    # 处理Netherland -> Netherlands的特殊情况
                    if country_en == 'Netherland':
                        country_en = 'Netherlands'
                    # 检查是否在分析列表中
                    if country_en in self.country_to_idx:
                        country = country_en
                
                # 如果英文列为空或不在列表中，尝试使用中文列
                if country is None and not pd.isna(country_cn_raw) and isinstance(country_cn_raw, str):
                    country_cn = country_cn_raw.strip()
                    # 通过中文名获取英文名
                    country = self.cn_to_en_mapping.get(country_cn)
                
                # 跳过无法识别的国家
                if country is None:
                    if not pd.isna(country_raw) or not pd.isna(country_cn_raw):
                        print(f"警告: 行{row_idx} 无法识别国家 (英文: {country_raw}, 中文: {country_cn_raw})，跳过")
                    continue
                
                # 检查国家是否有理论得分
                if country not in current_scores:
                    print(f"警告: {country} 没有理论得分数据，跳过")
                    continue
                
                # 获取投票
                vote_raw = row.iloc[col_idx]
                
                # 跳过NaN值
                if pd.isna(vote_raw):
                    continue
                
                # 转换投票
                if isinstance(vote_raw, str):
                    vote_lower = vote_raw.lower().strip()
                    vote = vote_mapping.get(vote_lower)
                elif isinstance(vote_raw, (int, float)):
                    vote = int(vote_raw) if vote_raw in [0, 1, 2] else None
                else:
                    vote = None
                
                if vote is None:
                    print(f"警告: {country} 无法解析投票 '{vote_raw}'，跳过")
                    continue
                
                # 获取当前轮次的理论得分
                scores = current_scores[country]
                
                # 添加到训练数据
                training_data.append({
                    'country_idx': self.country_to_idx[country],
                    'theory_scores': scores,
                    'actual_vote': vote,
                    'country_name': country,
                    'vote_round': str(vote_round)
                })
        
        print(f"准备了{len(training_data)}条训练数据")
        
        # 显示投票分布
        vote_counts = {0: 0, 1: 0, 2: 0}
        for item in training_data:
            vote_counts[item['actual_vote']] += 1
        
        print("\n投票分布:")
        print(f"  反对 (0): {vote_counts[0]}")
        print(f"  弃权 (1): {vote_counts[1]}")
        print(f"  赞成 (2): {vote_counts[2]}")
        
        return training_data
    
    def fit_model(self, training_data: List[Dict]) -> Dict:
        """
        拟合Ordered Probit模型（使用迭代优化方法）
        
        参数:
            training_data: 训练数据
        
        返回:
            收敛信息
        """
        print("\n正在拟合Ordered Probit模型（使用迭代优化方法）...")
        
        # 初始化MLE模型（如果尚未创建或需要重新创建）
        n_countries = len(self.countries)
        
        # 检查是否已存在模型，如果存在且国家数量匹配，则复用
        # 这样可以保留用户设置的手动阈值配置
        if self.mle_model is None:
            self.mle_model = OrderedProbitMLE(n_countries=n_countries, n_theories=3)
            print("  创建新的MLE模型（自动阈值模式）")
        elif isinstance(self.mle_model, OrderedProbitMLE):
            if self.mle_model.n_countries != n_countries:
                # 国家数量不匹配，需要重新创建
                self.mle_model = OrderedProbitMLE(n_countries=n_countries, n_theories=3)
                print(f"  国家数量变化，重新创建MLE模型（自动阈值模式）")
            else:
                # 复用现有模型
                if self.mle_model.manual_thresholds:
                    print(f"  复用现有MLE模型（手动阈值模式）")
                    print(f"    手动阈值: α₁={self.mle_model.manual_alpha1}, α₂={self.mle_model.manual_alpha2}")
                else:
                    print(f"  复用现有MLE模型（自动阈值模式）")
        else:
            # mle_model存在但类型不正确，重新创建
            self.mle_model = OrderedProbitMLE(n_countries=n_countries, n_theories=3)
            print("  MLE模型类型不正确，重新创建（自动阈值模式）")
        
        # 使用迭代优化方法拟合模型
        convergence_info = self.mle_model.fit_iterative(training_data)
        
        print(f"\n模型拟合完成!")
        if convergence_info.get('method') == 'iterative':
            # 迭代优化方法的结果
            print(f"优化方法: 迭代优化（两阶段）")
            phase_results = convergence_info.get('phase_results', {})
            if 'phase1' in phase_results:
                print(f"\n阶段1结果:")
                print(f"  阈值: α₁={phase_results['phase1']['alpha1']:.4f}, α₂={phase_results['phase1']['alpha2']:.4f}")
                print(f"  准确率: 反对={phase_results['phase1']['accuracy'][0]:.2%}, 弃权={phase_results['phase1']['accuracy'][1]:.2%}, 赞成={phase_results['phase1']['accuracy'][2]:.2%}")
            if 'phase2' in phase_results:
                print(f"\n阶段2结果（最终）:")
                print(f"  阈值: α₁={phase_results['phase2']['alpha1']:.4f}, α₂={phase_results['phase2']['alpha2']:.4f}")
                print(f"  准确率: 反对={phase_results['phase2']['accuracy'][0]:.2%}, 弃权={phase_results['phase2']['accuracy'][1]:.2%}, 赞成={phase_results['phase2']['accuracy'][2]:.2%}")
        else:
            # 原始优化方法的结果
            print(f"收敛状态: {'成功' if convergence_info['success'] else '失败'}")
            print(f"迭代次数: {convergence_info['nit']}")
            print(f"目标函数值: {convergence_info['fun']:.4f}")
            print(f"消息: {convergence_info['message']}")
        
        return convergence_info
    
    def evaluate_model(self, training_data: List[Dict]) -> Dict:
        """
        评估模型性能
        
        参数:
            training_data: 训练数据
        
        返回:
            评估结果
        """
        print("\n正在评估模型性能...")
        
        # 生成预测
        actual_votes = []
        predicted_votes = []
        
        for item in training_data:
            country_idx = item['country_idx']
            theory_scores = item['theory_scores']
            actual_vote = item['actual_vote']
            
            # 预测概率
            proba = self.mle_model.predict_proba(theory_scores, country_idx)
            
            # 选择概率最大的类别
            predicted_vote = np.argmax(proba)
            
            actual_votes.append(actual_vote)
            predicted_votes.append(predicted_vote)
        
        # 计算准确率
        accuracy = np.mean(np.array(actual_votes) == np.array(predicted_votes))
        
        # 计算各类别准确率
        class_accuracy = {}
        for vote_type in [0, 1, 2]:
            mask = np.array(actual_votes) == vote_type
            if np.sum(mask) > 0:
                class_accuracy[vote_type] = np.mean(
                    np.array(actual_votes)[mask] == np.array(predicted_votes)[mask]
                )
            else:
                class_accuracy[vote_type] = 0.0
        
        results = {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'actual': actual_votes,
            'predicted': predicted_votes
        }
        
        print(f"\n模型评估结果:")
        print(f"  整体准确率: {accuracy:.2%}")
        print(f"  反对准确率: {class_accuracy[0]:.2%}")
        print(f"  弃权准确率: {class_accuracy[1]:.2%}")
        print(f"  赞成准确率: {class_accuracy[2]:.2%}")
        
        return results
    
    def generate_visualizations(self, training_data: List[Dict], evaluation_results: Dict):
        """
        生成所有可视化图表
        
        参数:
            training_data: 训练数据
            evaluation_results: 评估结果
        """
        print("\n正在生成可视化图表...")
        
        # 提取参数
        alpha1 = self.mle_model.alpha1
        alpha2 = self.mle_model.alpha2
        weights = self.mle_model.weights
        
        # 生成预测和残差
        residuals = self.mle_model.get_residuals(training_data)
        
        # 1. 拟合优度
        self.visualizer.plot_goodness_of_fit(
            evaluation_results['actual'],
            evaluation_results['predicted'],
            self.countries,
            save_name="goodness_of_fit.png"
        )
        
        # 2. 阈值曲线
        self.visualizer.plot_threshold_curves(alpha1, alpha2)
        
        # 3. 国家权重
        self.visualizer.plot_country_weights(weights, self.countries)
        
        # 4. 残差诊断
        self.visualizer.plot_residuals(residuals)
        
        # 5. 边际效应（为每个国家生成）
        print("\n生成边际效应图表...")
        for theory_idx in range(3):
            for country_idx in range(len(self.countries)):  # 遍历所有国家
                country_name = self.countries[country_idx]
                save_name = f"marginal_effects_{country_name}_theory{theory_idx}.png"
                self.visualizer.plot_marginal_effects(
                    self.mle_model,
                    country_idx,
                    country_name,
                    theory_idx,
                    save_name=save_name
                )
        
        # 6. 模型比较（如果有其他模型）
        # 这里可以添加与其他模型的比较
        
        print(f"\n所有可视化图表已保存到 {self.results_dir}")
    
    def save_parameters(self):
        """保存估计的参数"""
        print("\n正在保存参数...")
        
        params_file = os.path.join(self.results_dir, "estimated_parameters.json")
        self.mle_model.save_parameters(params_file, self.countries)
        
        # 同时保存为YAML格式以便配置文件使用
        yaml_file = os.path.join(self.results_dir, "estimated_parameters.yaml")
        
        import yaml
        params_dict = {
            'ordered_probit': {
                'alpha1': float(self.mle_model.alpha1),
                'alpha2': float(self.mle_model.alpha2)
            },
            'theory_weights': {}
        }
        
        theory_names = [
            'x_market',
            'x_political',
            'x_institutional'
        ]
        
        for i, country in enumerate(self.countries):
            params_dict['theory_weights'][country] = {
                theory_names[0]: float(self.mle_model.weights[i, 0]),
                theory_names[1]: float(self.mle_model.weights[i, 1]),
                theory_names[2]: float(self.mle_model.weights[i, 2])
            }
        
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(params_dict, f, allow_unicode=True, default_flow_style=False)
        
        print(f"参数已保存到 {params_file} 和 {yaml_file}")
    
    def print_detailed_results(self, initial_scores: Dict[str, np.ndarray],
                                final_scores: Dict[str, np.ndarray],
                                training_data: List[Dict]):
        """
        在终端输出详细结果，包括各国2次的理论评分、权重、加权值、投票概率和预测结果
        
        参数:
            initial_scores: 第一轮理论得分字典
            final_scores: 第二轮理论得分字典
            training_data: 训练数据
        """
        print("\n" + "="*100)
        print("详细结果输出 - 各国理论评分、权重、投票概率与预测对比")
        print("="*100)
        
        theory_names = ['市场因素', '政治因素', '制度因素']
        vote_names = ['反对', '弃权', '赞成']
        
        # 输出估计的全局参数
        print("\n【全局参数估计结果】")
        print("-"*100)
        print(f"阈值 α₁ (反对/弃权边界): {self.mle_model.alpha1:.6f}")
        print(f"阈值 α₂ (弃权/赞成边界): {self.mle_model.alpha2:.6f}")
        
        # 按国家输出详细结果
        for country_idx, country in enumerate(self.countries):
            print("\n" + "="*100)
            print(f"【{country}】")
            print("="*100)
            
            # 输出该国家的权重
            print(f"\n三维权重值:")
            weights = self.mle_model.weights[country_idx]
            for i, name in enumerate(theory_names):
                print(f"  {name} (w{i+1}): {weights[i]:.6f}")
            print(f"  权重总和: {np.sum(weights):.6f}")
            
            # 处理两次投票
            for vote_round in ['initial', 'final']:
                print(f"\n{'─'*100}")
                if vote_round == 'initial':
                    print(f"第一次投票 (使用 INITIAL 理论评分)")
                    scores = initial_scores[country]
                    round_num = 1
                else:
                    print(f"第二次投票 (使用 FINAL 理论评分)")
                    scores = final_scores[country]
                    round_num = 2
                
                # 输出理论评分
                print(f"\n理论评分值:")
                for i, name in enumerate(theory_names):
                    print(f"  {name} (t{i+1}): {scores[i]:.6f}")
                
                # 计算加权值 η = Σ(tᵢ × wᵢ)
                weighted_value = np.dot(scores, weights)
                print(f"\n加权值 η = Σ(tᵢ × wᵢ): {weighted_value:.6f}")
                
                # 计算投票概率
                proba = self.mle_model.predict_proba(scores, country_idx)
                print(f"\n投票概率 (根据新权重和阈值计算):")
                for i, name in enumerate(vote_names):
                    print(f"  P({name}): {proba[i]:.6f}")
                
                # 找到实际投票数据
                actual_vote = None
                predicted_vote = int(np.argmax(proba))
                
                for item in training_data:
                    if item['country_idx'] == country_idx:
                        # 检查是否是当前轮次
                        # 第一次投票：列名包含"第一次"或"1月"
                        # 第二次投票：列名包含"第二次"或"10月"
                        if vote_round == 'initial' and ('第一次' in item['vote_round'] or '1月' in item['vote_round']):
                            actual_vote = item['actual_vote']
                            break
                        elif vote_round == 'final' and ('第二次' in item['vote_round'] or '10月' in item['vote_round']):
                            actual_vote = item['actual_vote']
                            break
                
                # 输出预测和对比
                print(f"\n最终投票选择:")
                print(f"  预测投票: {vote_names[predicted_vote]} ({predicted_vote})")
                if actual_vote is not None:
                    print(f"  实际投票: {vote_names[actual_vote]} ({actual_vote})")
                    match = "✓ 正确" if predicted_vote == actual_vote else "✗ 错误"
                    print(f"  对比结果: {match}")
                else:
                    print(f"  实际投票: 数据缺失")
        
        # 汇总统计
        print("\n" + "="*100)
        print("【汇总统计】")
        print("="*100)
        
        # 计算整体准确率
        correct = 0
        total = 0
        country_correct = {country: 0 for country in self.countries}
        country_total = {country: 0 for country in self.countries}
        vote_correct = [0, 0, 0]
        vote_total = [0, 0, 0]
        
        for item in training_data:
            country_idx = item['country_idx']
            theory_scores = item['theory_scores']
            actual_vote = item['actual_vote']
            country = self.countries[country_idx]
            
            pred_proba = self.mle_model.predict_proba(theory_scores, country_idx)
            predicted_vote = int(np.argmax(pred_proba))
            
            total += 1
            country_total[country] += 1
            vote_total[actual_vote] += 1
            
            if predicted_vote == actual_vote:
                correct += 1
                country_correct[country] += 1
                vote_correct[actual_vote] += 1
        
        print(f"\n整体预测准确率: {correct}/{total} = {correct/total:.2%}")
        
        print(f"\n各投票类别准确率:")
        for i, name in enumerate(vote_names):
            if vote_total[i] > 0:
                print(f"  {name}: {vote_correct[i]}/{vote_total[i]} = {vote_correct[i]/vote_total[i]:.2%}")
        
        print(f"\n各国预测准确率:")
        for country in self.countries:
            if country_total[country] > 0:
                acc = country_correct[country] / country_total[country]
                print(f"  {country}: {country_correct[country]}/{country_total[country]} = {acc:.2%}")
        
        print("\n" + "="*100)
        print("详细结果输出完成")
        print("="*100 + "\n")
    
    def generate_comprehensive_table(self, initial_scores: Dict[str, np.ndarray],
                                      final_scores: Dict[str, np.ndarray],
                                      training_data: List[Dict]):
        """
        生成综合表格并在终端输出
        
        横坐标：第一次的各理论评分，第一次加权值，第一次模拟投票结果，第一次真实投票结果，
               第二次各理论评分，第二次加权值，第二次模拟投票结果，第二次真实投票结果，
               各维度权重值，阈值
        纵坐标：各个欧盟成员国
        
        参数:
            initial_scores: 第一轮理论得分字典
            final_scores: 第二轮理论得分字典
            training_data: 训练数据
        """
        print("\n" + "="*140)
        print("综合结果表格")
        print("="*140)
        
        # 投票名称映射
        vote_names_cn = {0: '反对', 1: '弃权', 2: '赞成'}
        vote_names_en = {0: 'against', 1: 'abstain', 2: 'support'}
        
        # 获取全局阈值
        alpha1 = self.mle_model.alpha1
        alpha2 = self.mle_model.alpha2
        
        # 准备表格数据
        table_data = []
        
        # 遍历每个国家
        for country_idx, country in enumerate(self.countries):
            # 获取该国家的权重
            weights = self.mle_model.weights[country_idx]
            
            # 处理第一次投票
            initial_scores_array = initial_scores[country]
            initial_weighted_value = np.dot(initial_scores_array, weights)
            initial_proba = self.mle_model.predict_proba(initial_scores_array, country_idx)
            initial_predicted_vote = int(np.argmax(initial_proba))
            
            # 查找第一次的真实投票
            initial_actual_vote = None
            for item in training_data:
                if item['country_idx'] == country_idx:
                    if '第一次' in item['vote_round'] or '1月' in item['vote_round']:
                        initial_actual_vote = item['actual_vote']
                        break
            
            # 处理第二次投票
            final_scores_array = final_scores[country]
            final_weighted_value = np.dot(final_scores_array, weights)
            final_proba = self.mle_model.predict_proba(final_scores_array, country_idx)
            final_predicted_vote = int(np.argmax(final_proba))
            
            # 查找第二次的真实投票
            final_actual_vote = None
            for item in training_data:
                if item['country_idx'] == country_idx:
                    if '第二次' in item['vote_round'] or '10月' in item['vote_round']:
                        final_actual_vote = item['actual_vote']
                        break
            
            # 构建表格行
            row = [
                country,  # 国家
                # 第一次理论评分
                f"{initial_scores_array[0]:.4f}",  # 市场
                f"{initial_scores_array[1]:.4f}",  # 政治
                f"{initial_scores_array[2]:.4f}",  # 制度
                # 第一次
                f"{initial_weighted_value:.4f}",  # 加权值
                vote_names_cn.get(initial_predicted_vote, '未知'),  # 模拟结果
                vote_names_cn.get(initial_actual_vote, '缺失') if initial_actual_vote is not None else '缺失',  # 真实结果
                # 第二次理论评分
                f"{final_scores_array[0]:.4f}",  # 市场
                f"{final_scores_array[1]:.4f}",  # 政治
                f"{final_scores_array[2]:.4f}",  # 制度
                # 第二次
                f"{final_weighted_value:.4f}",  # 加权值
                vote_names_cn.get(final_predicted_vote, '未知'),  # 模拟结果
                vote_names_cn.get(final_actual_vote, '缺失') if final_actual_vote is not None else '缺失',  # 真实结果
                # 权重值
                f"{weights[0]:.4f}",  # 市场权重
                f"{weights[1]:.4f}",  # 政治权重
                f"{weights[2]:.4f}",  # 制度权重
                # 阈值
                f"{alpha1:.4f}",  # α1
                f"{alpha2:.4f}"   # α2
            ]
            
            table_data.append(row)
        
        # 定义表格列标题
        headers = [
            "国家",
            # 第一次
            "第一次\n市场", "第一次\n政治", "第一次\n制度",
            "第一次\n加权值", "第一次\n模拟", "第一次\n真实",
            # 第二次
            "第二次\n市场", "第二次\n政治", "第二次\n制度",
            "第二次\n加权值", "第二次\n模拟", "第二次\n真实",
            # 权重
            "权重\n市场", "权重\n政治", "权重\n制度",
            # 阈值
            "阈值\nα1", "阈值\nα2"
        ]
        
        # 使用tabulate生成表格
        table_str = tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            numalign="right",
            stralign="center"
        )
        
        print(table_str)
        print("="*140 + "\n")
    
    def generate_summary_report(self, evaluation_results: Dict):
        """生成分析摘要报告"""
        print("\n正在生成摘要报告...")
        
        report_file = os.path.join(self.results_dir, "analysis_summary.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Ordered Probit模型分析报告\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. 模型参数估计结果\n")
            f.write("-"*80 + "\n")
            f.write(f"阈值参数:\n")
            f.write(f"  α₁ (反对/弃权边界): {self.mle_model.alpha1:.4f}\n")
            f.write(f"  α₂ (弃权/赞成边界): {self.mle_model.alpha2:.4f}\n\n")
            
            f.write("国家权重估计:\n")
            theory_names = [
                '市场因素',
                '政治因素',
                '制度因素'
            ]
            
            for i, country in enumerate(self.countries):
                f.write(f"  {country}:\n")
                for j, name in enumerate(theory_names):
                    f.write(f"    {name}: {self.mle_model.weights[i, j]:.4f}\n")
                f.write("\n")
            
            f.write("\n2. 模型性能评估\n")
            f.write("-"*80 + "\n")
            f.write(f"整体准确率: {evaluation_results['accuracy']:.2%}\n")
            f.write(f"反对准确率: {evaluation_results['class_accuracy'][0]:.2%}\n")
            f.write(f"弃权准确率: {evaluation_results['class_accuracy'][1]:.2%}\n")
            f.write(f"赞成准确率: {evaluation_results['class_accuracy'][2]:.2%}\n\n")
            
            f.write("3. 收敛信息\n")
            f.write("-"*80 + "\n")
            info = self.mle_model.convergence_info
            f.write(f"收敛状态: {'成功' if info['success'] else '失败'}\n")
            
            # 检查是否是迭代优化方法
            if info.get('method') == 'iterative':
                f.write(f"优化方法: 迭代优化（两阶段）\n")
                phase_results = info.get('phase_results', {})
                if 'phase1' in phase_results:
                    p1 = phase_results['phase1']
                    f.write(f"\n阶段1结果:\n")
                    f.write(f"  阈值: α₁={p1['alpha1']:.4f}, α₂={p1['alpha2']:.4f}\n")
                    f.write(f"  准确率: 反对={p1['accuracy'][0]:.2%}, 弃权={p1['accuracy'][1]:.2%}, 赞成={p1['accuracy'][2]:.2%}\n")
                if 'phase2' in phase_results:
                    p2 = phase_results['phase2']
                    f.write(f"\n阶段2结果（最终）:\n")
                    f.write(f"  阈值: α₁={p2['alpha1']:.4f}, α₂={p2['alpha2']:.4f}\n")
                    f.write(f"  准确率: 反对={p2['accuracy'][0]:.2%}, 弃权={p2['accuracy'][1]:.2%}, 赞成={p2['accuracy'][2]:.2%}\n")
            else:
                # 原始优化方法
                f.write(f"迭代次数: {info.get('nit', 'N/A')}\n")
                f.write(f"函数评估次数: {info.get('nfev', 'N/A')}\n")
                f.write(f"目标函数值: {info.get('fun', 'N/A')}\n")
                f.write(f"消息: {info.get('message', 'N/A')}\n\n")
            
            f.write("4. 输出文件\n")
            f.write("-"*80 + "\n")
            f.write(f"参数文件: {self.results_dir}/estimated_parameters.json\n")
            f.write(f"参数文件(YAML): {self.results_dir}/estimated_parameters.yaml\n")
            f.write(f"可视化图表: {self.results_dir}/*.png\n")
        
        print(f"摘要报告已保存到 {report_file}")
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("="*80)
        print("开始Ordered Probit模型完整分析")
        print("="*80)
        
        try:
            # 1. 加载理论得分（包含initial和final）
            initial_scores, final_scores = self.load_theory_scores()
            
            # 2. 加载实际投票数据
            vote_df = self.load_actual_votes()
            
            # 3. 准备训练数据（使用不同轮次的理论得分）
            training_data = self.prepare_training_data(initial_scores, final_scores, vote_df)
            
            if len(training_data) == 0:
                print("错误: 没有可用的训练数据!")
                return
            
            # 4. 拟合模型
            convergence_info = self.fit_model(training_data)
            
            if not convergence_info['success']:
                print("警告: 模型未完全收敛，结果可能不准确")
            
            # 5. 评估模型
            evaluation_results = self.evaluate_model(training_data)
            
            # 6. 生成可视化
            self.generate_visualizations(training_data, evaluation_results)
            
            # 7. 保存参数
            self.save_parameters()
            
            # 8. 生成摘要报告
            self.generate_summary_report(evaluation_results)
            
            # 9. 输出详细结果到终端
            self.print_detailed_results(initial_scores, final_scores, training_data)
            
            # 10. 生成综合结果表格
            self.generate_comprehensive_table(initial_scores, final_scores, training_data)
            
            print("\n" + "="*80)
            print("分析完成!")
            print("="*80)
            
        except Exception as e:
            print(f"\n错误: 分析过程中出现异常")
            print(f"异常信息: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 创建分析器并运行完整分析
    analyzer = OrderedProbitAnalysis()
    analyzer.mle_model = OrderedProbitMLE(
    n_countries=len(analyzer.countries),
    manual_thresholds=True,
    manual_alpha1=-0.6,
    manual_alpha2=0.6,
)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
