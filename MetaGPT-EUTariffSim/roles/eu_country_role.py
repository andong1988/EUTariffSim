"""欧盟国家角色智能体"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from metagpt.roles.role import Role

from actions.theoretical_decision_action import TheoreticalDecisionAction
from utils.config import VotingStance


class EUCountryRole(Role):
    """欧盟国家角色"""
    
    def __init__(self, country_id: str, anonymized_data: Dict, theory_weights: Dict[str, float], 
                 enable_theory_scores_cache: bool = False,
                 enable_decision_noise: bool = False,
                 decision_noise_level: float = 0.05,
                 ordered_probit_params: Dict = None,
                 use_cache_for_round: Dict[str, bool] = None):
        super().__init__(name=f"EU_Country_{country_id}", profile=f"EU Member State {country_id}")
        
        self.country_id = country_id
        self.anonymized_data = anonymized_data
        self.theory_weights = theory_weights.copy()
        
        # 决策历史
        self.decision_history = []
        self.communication_history = []
        
        # 动态属性
        self.stance_strength = 0.5
        self.communication_urgency = 0.0
        
        self.logger = logging.getLogger(f"{__name__}.{country_id}")
        
        # 初始化理论决策动作（传递随机扰动配置和Ordered Probit参数）
        self.theoretical_decision_action = TheoreticalDecisionAction(
            enable_decision_noise=enable_decision_noise,
            decision_noise_level=decision_noise_level,
            ordered_probit_params=ordered_probit_params
        )
        
        # 理论得分缓存配置
        self.enable_theory_scores_cache = enable_theory_scores_cache
        self.use_cache_for_round = use_cache_for_round if use_cache_for_round else {"initial": True, "final": False}
        self.theory_scores_cache_file = f"theory_scores_{self.country_id}.json"
    
    def _load_theory_scores_cache(self, decision_context: Dict = None) -> Optional[Dict]:
        """加载理论得分缓存（考虑决策上下文和投票轮次）"""
        try:
            cache_path = Path(__file__).parent.parent / "cache" / self.theory_scores_cache_file
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # 检查数据哈希是否匹配
                    if cached_data.get("data_hash") == self.anonymized_data.get("original_hash"):
                        # 检查决策上下文是否匹配
                        if decision_context is None:
                            # 如果没有提供决策上下文，只检查数据哈希，返回初始投票的理论得分
                            return cached_data.get("initial", {})
                        else:
                            # 获取投票轮次
                            round = decision_context.get("round", "initial")
                            
                            # 计算当前决策上下文的哈希
                            context_hash = self._calculate_context_hash(decision_context)
                            
                            # 根据轮次查找对应的理论得分
                            if round in cached_data:
                                round_data = cached_data[round]
                                cached_context_hash = round_data.get("context_hash")
                                
                                if context_hash == cached_context_hash:
                                    self.logger.debug(f"决策上下文匹配（{round}轮），使用缓存: {self.country_id}")
                                    # 返回完整的轮次数据（包含theory_scores和prompt）
                                    return round_data
                                else:
                                    self.logger.debug(f"决策上下文不匹配（{round}轮），重新生成: {self.country_id}")
                            else:
                                self.logger.debug(f"缓存中不存在{round}轮的数据，重新生成: {self.country_id}")
                            return None
            return None
        except Exception as e:
            self.logger.warning(f"加载理论得分缓存失败: {e}")
            return None
    
    def _save_theory_scores_cache(self, theory_scores: Dict, decision_context: Dict = None, prompt: str = ""):
        """保存理论得分缓存（考虑决策上下文和投票轮次）"""
        try:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / self.theory_scores_cache_file
            
            # 获取投票轮次
            round = decision_context.get("round", "initial") if decision_context else "initial"
            
            # 计算决策上下文哈希
            context_hash = self._calculate_context_hash(decision_context) if decision_context else None
            
            # 尝试加载现有缓存数据
            cache_data = {}
            if cache_path.exists():
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                except:
                    pass
            
            # 更新缓存数据结构
            cache_data["data_hash"] = self.anonymized_data.get("original_hash")
            cache_data["timestamp"] = datetime.now().isoformat()
            
            # 根据轮次保存理论得分和prompt
            cache_data[round] = {
                "theory_scores": theory_scores,
                "prompt": prompt,
                "context_hash": context_hash,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.debug(f"保存决策上下文哈希和prompt（{round}轮）: {context_hash} for {self.country_id}")
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"理论得分缓存和prompt已保存（{round}轮）: {self.country_id}")
        except Exception as e:
            self.logger.warning(f"保存理论得分缓存失败: {e}")
    
    async def make_theoretical_decision(self, context: Dict, voting_proposal: Dict = None,
                                      other_countries_communications: List[Dict] = None,
                                      eu_commission_communication: Dict = None,
                                      china_communication: Dict = None,
                                      secretary_analysis: Dict = None,
                                      initial_theory_scores: Dict = None,
                                      initial_vote: str = None) -> Dict:
        """基于理论框架和Ordered Probit模型做出决策"""
        try:
            # 准备决策上下文
            decision_context = {
                **context,
                "voting_proposal": voting_proposal,
                "other_countries_communications": other_countries_communications,
                "eu_commission_communication": eu_commission_communication,
                "china_communication": china_communication
            }
            
            # 检查当前轮次是否允许使用缓存
            round_name = decision_context.get("round", "initial")
            use_cache_for_current_round = self.use_cache_for_round.get(round_name, True)
            
            # 根据配置决定是否尝试从缓存加载理论得分
            cached_data = None
            if self.enable_theory_scores_cache and use_cache_for_current_round:
                cached_data = self._load_theory_scores_cache(decision_context)
            
            # 如果缓存命中，需要使用缓存的理论得分重新调用TheoreticalDecisionAction
            # 因为Ordered Probit的计算在TheoreticalDecisionAction中完成
            if cached_data:
                self.logger.info(f"使用缓存的理论得分: {self.country_id}")
                theory_scores = cached_data.get("theory_scores", {})
                # 验证缓存数据的完整性
                if self._validate_theory_scores(theory_scores):
                    # 直接使用缓存的理论得分，跳过LLM调用
                    action_result = await self.theoretical_decision_action.run_with_cached_scores(
                        country_features=self.anonymized_data,
                        theory_weights=self.theory_weights,
                        theory_scores=theory_scores,
                        context=decision_context
                    )
                else:
                    self.logger.warning(f"缓存数据无效，重新生成")
                    cached_data = None
                    action_result = None
            
            # 如果缓存未命中或无效，使用LLM生成
            if not cached_data:
                self.logger.info(f"使用LLM生成理论得分和决策: {self.country_id}")
                action_result = await self.theoretical_decision_action.run(
                    country_features=self.anonymized_data,
                    theory_weights=self.theory_weights,
                    context=decision_context,
                    voting_proposal=voting_proposal,
                    other_countries_communications=other_countries_communications,
                    eu_commission_communication=eu_commission_communication,
                    secretary_analysis=secretary_analysis,
                    initial_theory_scores=initial_theory_scores,
                    initial_vote=initial_vote
                )
                
                # 保存理论得分到缓存
                theory_scores = action_result.get('theory_scores', {})
                if self._validate_theory_scores(theory_scores):
                    self._save_theory_scores_cache(theory_scores, decision_context, "")
            
            # 构建标准化的决策返回格式
            decision = self._build_decision_result(action_result, decision_context)
            
            # 更新动态属性
            self.stance_strength = decision["stance_strength"]
            
            # 记录决策历史
            self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"决策过程出错: {e}", exc_info=True)
            raise
    
    
    def _build_decision_result(self, action_result: Dict, decision_context: Dict) -> Dict:
        """构建标准化的决策结果"""
        
        # 从action_result中提取关键信息
        theory_scores = action_result.get('theory_scores', {})
        decision_text = action_result.get('decision', '')
        decision_score = action_result.get('decision_score', 0.0)
        
        # 将决策文本映射到stance枚举值
        stance = self._map_decision_to_stance(decision_text)
        
        # 计算立场强度（基于Ordered Probit概率或简单计算）
        prob_info = action_result.get('probabilities', {})
        if prob_info:
            # 如果有概率信息，使用最大概率作为强度
            stance_strength = max(prob_info.values())
        else:
            # 否则基于decision_score计算
            stance_strength = min(1.0, abs(decision_score))
        
        decision = {
            "country_id": self.country_id,
            "stance": stance,
            "stance_strength": stance_strength,
            "decision_score": decision_score,
            "theoretical_factors": theory_scores,
            "weights_used": self.theory_weights.copy(),
            "reasoning": decision_text,
            "llm_used": True,
            # Ordered Probit相关信息
            "probit_probabilities": prob_info,
            "latent_variable": decision_score,
            "decision_context": {
                "round": decision_context.get("round", ""),
                "communication_count": len(decision_context.get("other_countries_communications") or []),
                "eu_communication_present": decision_context.get("eu_commission_communication") is not None,
                "china_communication_present": decision_context.get("china_communication") is not None
            }
        }
        
        return decision
    
    def _map_decision_to_stance(self, decision_text: str) -> str:
        """将决策文本映射到投票立场枚举值"""
        decision_lower = decision_text.lower()
        
        if "反对" in decision_lower or "against" in decision_lower or "no" in decision_lower:
            return VotingStance.AGAINST.value
        elif "弃权" in decision_lower or "abstain" in decision_lower or "abs" in decision_lower:
            return VotingStance.ABSTAIN.value
        elif "赞同" in decision_lower or "赞成" in decision_lower or "support" in decision_lower or "yes" in decision_lower or "for" in decision_lower:
            return VotingStance.SUPPORT.value
        else:
            # 默认返回弃权
            self.logger.warning(f"无法识别的决策文本: {decision_text}，默认返回弃权")
            return VotingStance.ABSTAIN.value
    
    def _validate_theory_scores(self, theory_scores: Dict[str, float]) -> bool:
        """验证理论得分结构的完整性和有效性
        
        Args:
            theory_scores: 待验证的理论得分字典
            
        Returns:
            是否有效
        """
        # 定义必需的理论键
        required_keys = [
            "x_market",
            "x_political",
            "x_institutional"
        ]
        
        # 检查是否为字典
        if not isinstance(theory_scores, dict):
            self.logger.warning(f"理论得分不是字典类型: {type(theory_scores)}")
            return False
        
        # 检查是否包含所有必需的键
        missing_keys = [key for key in required_keys if key not in theory_scores]
        if missing_keys:
            self.logger.warning(f"理论得分缺少必需的键: {missing_keys}")
            return False
        
        # 检查得分是否为数值且在合理范围内（-3到3）
        for key, value in theory_scores.items():
            if not isinstance(value, (int, float)):
                self.logger.warning(f"理论得分 '{key}' 不是数值类型: {type(value)}")
                return False
            if not (-3.0 <= value <= 3.0):
                self.logger.warning(f"理论得分 '{key}' 超出范围 [-3,3]: {value}")
                return False
        
        self.logger.debug(f"理论得分验证通过: {theory_scores}")
        return True
    
    def _clear_theory_scores_cache(self):
        """清除理论得分缓存文件"""
        try:
            cache_path = Path(__file__).parent.parent / "cache" / self.theory_scores_cache_file
            if cache_path.exists():
                cache_path.unlink()
                self.logger.info(f"{self.country_id} 已清除理论得分缓存: {cache_path}")
            else:
                self.logger.debug(f"{self.country_id} 理论得分缓存文件不存在: {cache_path}")
        except Exception as e:
            self.logger.warning(f"{self.country_id} 清除理论得分缓存失败: {e}")
    
    def _calculate_context_hash(self, decision_context: Dict) -> str:
        """计算决策上下文的哈希值"""
        try:
            # 安全地获取通信列表长度
            other_comms = decision_context.get("other_countries_communications", [])
            communication_count = len(other_comms) if other_comms is not None else 0
            
            # 安全地获取提案关税率
            voting_proposal = decision_context.get("voting_proposal", {})
            proposal_tariff = 0.0
            if voting_proposal and isinstance(voting_proposal, dict):
                proposal_tariff = voting_proposal.get("tariff_rate", 0.0)
            
            # 提取关键的上下文信息用于哈希计算
            context_key = {
                "round": decision_context.get("round", ""),
                "phase": decision_context.get("phase", ""),
                "proposal_tariff": proposal_tariff,
                "communication_count": communication_count,
                "eu_communication_present": decision_context.get("eu_commission_communication") is not None,
                "china_communication_present": decision_context.get("china_communication") is not None
            }
            
            # 转换为JSON字符串并计算哈希
            context_str = json.dumps(context_key, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(context_str.encode('utf-8')).hexdigest()[:8]
            
        except Exception as e:
            self.logger.warning(f"计算决策上下文哈希失败: {e}")
            # 如果计算失败，返回一个基于时间戳的简单哈希
            return hashlib.md5(str(datetime.now().timestamp()).encode('utf-8')).hexdigest()[:8]
    
    def _clear_theory_scores_cache(self):
        """清除理论得分缓存文件，确保每次运行都生成新的理论得分"""
        try:
            cache_path = Path(__file__).parent.parent / "cache" / self.theory_scores_cache_file
            if cache_path.exists():
                cache_path.unlink()
                self.logger.info(f"{self.country_id} 已清除理论得分缓存: {cache_path}")
            else:
                self.logger.debug(f"{self.country_id} 理论得分缓存文件不存在: {cache_path}")
        except Exception as e:
            self.logger.warning(f"{self.country_id} 清除理论得分缓存失败: {e}")
    
    def update_theory_weights(self, new_weights: Dict[str, float]):
        """更新理论权重"""
        self.theory_weights = new_weights
        self.logger.info(f"{self.country_id} 理论权重已更新: {new_weights}")
    
    def get_decision_statistics(self) -> Dict:
        """获取决策统计信息"""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "support_count": 0,
                "abstain_count": 0,
                "against_count": 0,
                "average_stance_strength": 0.0
            }
        
        support_count = sum(1 for d in self.decision_history if d["stance"] == VotingStance.SUPPORT.value)
        abstain_count = sum(1 for d in self.decision_history if d["stance"] == VotingStance.ABSTAIN.value)
        against_count = sum(1 for d in self.decision_history if d["stance"] == VotingStance.AGAINST.value)
        
        return {
            "total_decisions": len(self.decision_history),
            "support_count": support_count,
            "abstain_count": abstain_count,
            "against_count": against_count,
            "average_stance_strength": sum(d["stance_strength"] for d in self.decision_history) / len(self.decision_history)
        }
