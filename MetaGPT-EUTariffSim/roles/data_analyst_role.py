"""
数据分析师角色模块

定义数据分析师的智能体角色，负责实时数据收集与分析。
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from metagpt.roles.role import Role
from metagpt.actions.action import Action
from metagpt.schema import Message

from theory.theory_encoder import TheoryEncoder, TheoryContext, TheoryType, TheoryWeight


class AnalysisType(Enum):
    """分析类型枚举"""
    DESCRIPTIVE = "descriptive"  # 描述性分析
    PREDICTIVE = "predictive"    # 预测性分析
    PRESCRIPTIVE = "prescriptive"  # 规范性分析
    DIAGNOSTIC = "diagnostic"    # 诊断性分析


class MetricType(Enum):
    """指标类型枚举"""
    DECISION_PROCESS = "decision_process"  # 决策过程指标
    GAME_DYNAMICS = "game_dynamics"      # 博弈动态指标
    ECONOMIC_IMPACT = "economic_impact"   # 经济影响指标
    THEORY_VALIDATION = "theory_validation"  # 理论验证指标


@dataclass
class AnalysisMetric:
    """分析指标"""
    metric_id: str
    metric_type: MetricType
    name: str
    description: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class AnalysisResult:
    """分析结果"""
    analysis_id: str
    analysis_type: AnalysisType
    metrics: List[AnalysisMetric]
    insights: List[str]
    recommendations: List[str]
    data_quality_score: float
    methodology: str
    limitations: List[str]


@dataclass
class DataAnalystState:
    """数据分析师状态"""
    collected_data: Dict[str, Any]
    analysis_history: List[AnalysisResult]
    real_time_metrics: Dict[str, AnalysisMetric]
    data_sources: List[str]
    analysis_queue: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]


class DataAnalystRole(Role):
    """数据分析师角色智能体"""
    
    def __init__(
        self,
        name: str = "Data Analyst",
        profile: str = "Quantitative Analysis Expert",
        goal: str = "Collect, analyze, and interpret simulation data for insights",
        constraints: str = "Must ensure data quality and methodological rigor"
    ):
        super().__init__(name=name, profile=profile, goal=goal, constraints=constraints)
        
        self.theory_encoder = TheoryEncoder()
        
        # 初始化数据分析师状态
        self.state = DataAnalystState(
            collected_data={},
            analysis_history=[],
            real_time_metrics={},
            data_sources=[
                "country_positions",
                "negotiation_records",
                "voting_results",
                "communication_networks",
                "theory_consistency_scores"
            ],
            analysis_queue=[],
            alerts=[]
        )
        
        # 设置监听的动作
        self._watch([DataCollectionAction, RealTimeAnalysisAction, ReportGenerationAction])
    
    async def _collect_data(self, collection_request: Dict[str, Any]) -> Dict[str, Any]:
        """收集数据"""
        data_type = collection_request.get("data_type", "all")
        time_range = collection_request.get("time_range", "current")
        sources = collection_request.get("sources", self.state.data_sources)
        
        # 从各个数据源收集数据
        collected_data = {}
        
        for source in sources:
            try:
                source_data = await self._collect_from_source(source, time_range)
                collected_data[source] = source_data
            except Exception as e:
                # 记录收集错误但继续
                self.state.alerts.append({
                    "timestamp": datetime.now(),
                    "type": "data_collection_error",
                    "source": source,
                    "error": str(e)
                })
        
        # 数据质量检查
        quality_assessment = await self._assess_data_quality(collected_data)
        
        # 更新状态
        self.state.collected_data.update(collected_data)
        
        return {
            "analyst_id": self.name,
            "collection_request": collection_request,
            "collected_data": collected_data,
            "data_quality": quality_assessment,
            "collection_summary": await self._generate_collection_summary(collected_data),
            "alerts": self.state.alerts[-5:] if self.state.alerts else []  # 最近5个警报
        }
    
    async def _collect_from_source(self, source: str, time_range: str) -> Dict[str, Any]:
        """从特定数据源收集数据"""
        if source == "country_positions":
            return await self._collect_country_positions(time_range)
        elif source == "negotiation_records":
            return await self._collect_negotiation_records(time_range)
        elif source == "voting_results":
            return await self._collect_voting_results(time_range)
        elif source == "communication_networks":
            return await self._collect_communication_networks(time_range)
        elif source == "theory_consistency_scores":
            return await self._collect_theory_consistency_scores(time_range)
        else:
            return {"error": f"Unknown data source: {source}"}
    
    async def _collect_country_positions(self, time_range: str) -> Dict[str, Any]:
        """收集国家立场数据"""
        # 模拟国家立场数据收集
        positions = {
            "Country_A": {"position": "support", "confidence": 0.8, "timestamp": datetime.now()},
            "Country_B": {"position": "oppose", "confidence": 0.7, "timestamp": datetime.now()},
            "Country_C": {"position": "abstain", "confidence": 0.6, "timestamp": datetime.now()},
            "Country_D": {"position": "support", "confidence": 0.9, "timestamp": datetime.now()},
            "Country_E": {"position": "oppose", "confidence": 0.5, "timestamp": datetime.now()}
        }
        
        return {
            "positions": positions,
            "total_countries": len(positions),
            "position_distribution": self._calculate_position_distribution(positions),
            "average_confidence": np.mean([p["confidence"] for p in positions.values()]),
            "collection_time": datetime.now()
        }
    
    async def _collect_negotiation_records(self, time_range: str) -> Dict[str, Any]:
        """收集谈判记录数据"""
        # 模拟谈判记录数据收集
        negotiations = [
            {
                "round": 1,
                "participants": ["Country_A", "Country_B"],
                "outcome": "partial_agreement",
                "duration_minutes": 45,
                "messages_exchanged": 12,
                "timestamp": datetime.now()
            },
            {
                "round": 2,
                "participants": ["Country_C", "Country_D", "Country_E"],
                "outcome": "no_agreement",
                "duration_minutes": 60,
                "messages_exchanged": 18,
                "timestamp": datetime.now()
            }
        ]
        
        return {
            "negotiations": negotiations,
            "total_rounds": len(negotiations),
            "success_rate": sum(1 for n in negotiations if n["outcome"] != "no_agreement") / len(negotiations),
            "average_duration": np.mean([n["duration_minutes"] for n in negotiations]),
            "total_messages": sum(n["messages_exchanged"] for n in negotiations),
            "collection_time": datetime.now()
        }
    
    async def _collect_voting_results(self, time_range: str) -> Dict[str, Any]:
        """收集投票结果数据"""
        # 模拟投票结果数据收集
        voting_results = {
            "support": 4,
            "oppose": 3,
            "abstain": 3,
            "total": 10,
            "qualified_majority_achieved": True,
            "voting_timestamp": datetime.now()
        }
        
        return {
            "results": voting_results,
            "support_percentage": voting_results["support"] / voting_results["total"],
            "opposition_percentage": voting_results["oppose"] / voting_results["total"],
            "abstention_percentage": voting_results["abstain"] / voting_results["total"],
            "margin": voting_results["support"] - voting_results["oppose"],
            "collection_time": datetime.now()
        }
    
    async def _collect_communication_networks(self, time_range: str) -> Dict[str, Any]:
        """收集通信网络数据"""
        # 模拟通信网络数据收集
        communications = [
            {"from": "Country_A", "to": "Country_B", "messages": 5, "timestamp": datetime.now()},
            {"from": "Country_B", "to": "Country_A", "messages": 4, "timestamp": datetime.now()},
            {"from": "Country_C", "to": "Country_D", "messages": 3, "timestamp": datetime.now()},
            {"from": "Country_D", "to": "Country_E", "messages": 2, "timestamp": datetime.now()}
        ]
        
        # 计算网络指标
        network_metrics = self._calculate_network_metrics(communications)
        
        return {
            "communications": communications,
            "total_messages": sum(c["messages"] for c in communications),
            "unique_participants": len(set([c["from"] for c in communications] + [c["to"] for c in communications])),
            "network_metrics": network_metrics,
            "collection_time": datetime.now()
        }
    
    async def _collect_theory_consistency_scores(self, time_range: str) -> Dict[str, Any]:
        """收集理论一致性分数数据"""
        # 模拟理论一致性分数数据收集
        consistency_scores = {
            "rational_choice": {"average": 0.75, "std": 0.12, "min": 0.45, "max": 0.95},
            "two_level_game": {"average": 0.68, "std": 0.15, "min": 0.35, "max": 0.92},
            "constructivism": {"average": 0.72, "std": 0.10, "min": 0.50, "max": 0.88},
            "interdependence_weaponization": {"average": 0.65, "std": 0.18, "min": 0.30, "max": 0.90}
        }
        
        return {
            "consistency_scores": consistency_scores,
            "overall_consistency": np.mean([scores["average"] for scores in consistency_scores.values()]),
            "theory_ranking": sorted(consistency_scores.items(), key=lambda x: x[1]["average"], reverse=True),
            "collection_time": datetime.now()
        }
    
    def _calculate_position_distribution(self, positions: Dict[str, Any]) -> Dict[str, int]:
        """计算立场分布"""
        distribution = {"support": 0, "oppose": 0, "abstain": 0}
        for pos_data in positions.values():
            distribution[pos_data["position"]] += 1
        return distribution
    
    def _calculate_network_metrics(self, communications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算网络指标"""
        # 简化的网络指标计算
        participants = set()
        total_messages = 0
        
        for comm in communications:
            participants.add(comm["from"])
            participants.add(comm["to"])
            total_messages += comm["messages"]
        
        # 计算度中心度（简化）
        degree_centrality = {}
        for participant in participants:
            degree_centrality[participant] = sum(
                1 for c in communications 
                if c["from"] == participant or c["to"] == participant
            )
        
        return {
            "network_density": total_messages / (len(participants) * (len(participants) - 1)),
            "average_degree": np.mean(list(degree_centrality.values())),
            "max_degree": max(degree_centrality.values()),
            "degree_centrality": degree_centrality
        }
    
    async def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估数据质量"""
        quality_scores = {}
        
        for source, source_data in data.items():
            if isinstance(source_data, dict) and "error" not in source_data:
                # 简化的数据质量评估
                completeness = 0.9  # 假设完整性
                accuracy = 0.85     # 假设准确性
                timeliness = 0.95   # 假设及时性
                consistency = 0.8    # 假设一致性
                
                quality_scores[source] = {
                    "overall_score": (completeness + accuracy + timeliness + consistency) / 4,
                    "completeness": completeness,
                    "accuracy": accuracy,
                    "timeliness": timeliness,
                    "consistency": consistency
                }
            else:
                quality_scores[source] = {
                    "overall_score": 0.0,
                    "error": "Data collection failed"
                }
        
        overall_quality = np.mean([scores["overall_score"] for scores in quality_scores.values()])
        
        return {
            "source_quality": quality_scores,
            "overall_quality": overall_quality,
            "quality_level": "high" if overall_quality > 0.8 else "medium" if overall_quality > 0.6 else "low",
            "recommendations": self._generate_quality_recommendations(quality_scores)
        }
    
    def _generate_quality_recommendations(self, quality_scores: Dict[str, Any]) -> List[str]:
        """生成数据质量改进建议"""
        recommendations = []
        
        for source, scores in quality_scores.items():
            if scores["overall_score"] < 0.7:
                recommendations.append(f"Improve data collection for {source}")
            
            if scores.get("completeness", 1.0) < 0.8:
                recommendations.append(f"Enhance completeness for {source}")
            
            if scores.get("accuracy", 1.0) < 0.8:
                recommendations.append(f"Verify accuracy for {source}")
        
        return recommendations
    
    async def _generate_collection_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据收集摘要"""
        total_sources = len(data)
        successful_sources = len([d for d in data.values() if isinstance(d, dict) and "error" not in d])
        
        return {
            "total_sources_attempted": total_sources,
            "successful_collections": successful_sources,
            "success_rate": successful_sources / total_sources if total_sources > 0 else 0,
            "data_volume": self._estimate_data_volume(data),
            "collection_timestamp": datetime.now()
        }
    
    def _estimate_data_volume(self, data: Dict[str, Any]) -> str:
        """估算数据量"""
        # 简化的数据量估算
        total_items = sum(len(d) if isinstance(d, dict) else 1 for d in data.values())
        
        if total_items < 10:
            return "small"
        elif total_items < 50:
            return "medium"
        else:
            return "large"
    
    async def _perform_real_time_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """执行实时分析"""
        analysis_type = analysis_request.get("analysis_type", "descriptive")
        focus_areas = analysis_request.get("focus_areas", ["all"])
        time_window = analysis_request.get("time_window", "current")
        
        # 执行不同类型的分析
        results = {}
        
        if analysis_type == "descriptive" or "all" in focus_areas:
            results["descriptive"] = await self._perform_descriptive_analysis()
        
        if analysis_type == "predictive" or "all" in focus_areas:
            results["predictive"] = await self._perform_predictive_analysis()
        
        if analysis_type == "diagnostic" or "all" in focus_areas:
            results["diagnostic"] = await self._perform_diagnostic_analysis()
        
        if analysis_type == "prescriptive" or "all" in focus_areas:
            results["prescriptive"] = await self._perform_prescriptive_analysis()
        
        # 生成综合分析结果
        comprehensive_result = await self._generate_comprehensive_analysis(results)
        
        # 更新实时指标
        await self._update_real_time_metrics(comprehensive_result)
        
        return {
            "analyst_id": self.name,
            "analysis_request": analysis_request,
            "analysis_results": results,
            "comprehensive_result": comprehensive_result,
            "real_time_metrics": dict(self.state.real_time_metrics),
            "analysis_timestamp": datetime.now()
        }
    
    async def _perform_descriptive_analysis(self) -> Dict[str, Any]:
        """执行描述性分析"""
        # 基于当前收集的数据进行描述性分析
        current_data = self.state.collected_data
        
        descriptive_metrics = {}
        
        # 分析国家立场分布
        if "country_positions" in current_data:
            positions_data = current_data["country_positions"]
            descriptive_metrics["position_analysis"] = {
                "distribution": positions_data.get("position_distribution", {}),
                "average_confidence": positions_data.get("average_confidence", 0),
                "total_countries": positions_data.get("total_countries", 0)
            }
        
        # 分析谈判动态
        if "negotiation_records" in current_data:
            negotiations_data = current_data["negotiation_records"]
            descriptive_metrics["negotiation_analysis"] = {
                "success_rate": negotiations_data.get("success_rate", 0),
                "average_duration": negotiations_data.get("average_duration", 0),
                "total_rounds": negotiations_data.get("total_rounds", 0)
            }
        
        # 分析投票结果
        if "voting_results" in current_data:
            voting_data = current_data["voting_results"]
            descriptive_metrics["voting_analysis"] = {
                "support_percentage": voting_data.get("support_percentage", 0),
                "qualified_majority": voting_data.get("results", {}).get("qualified_majority_achieved", False),
                "margin": voting_data.get("margin", 0)
            }
        
        return {
            "analysis_type": "descriptive",
            "metrics": descriptive_metrics,
            "key_findings": self._extract_descriptive_insights(descriptive_metrics),
            "data_coverage": len(current_data)
        }
    
    async def _perform_predictive_analysis(self) -> Dict[str, Any]:
        """执行预测性分析"""
        # 基于历史数据预测未来趋势
        current_data = self.state.collected_data
        
        predictions = {}
        
        # 预测投票结果
        if "country_positions" in current_data:
            positions_data = current_data["country_positions"]
            distribution = positions_data.get("position_distribution", {})
            
            # 简化的预测逻辑
            support_prediction = distribution.get("support", 0) / 10  # 转换为比例
            predictions["voting_outcome"] = {
                "success_probability": support_prediction,
                "predicted_margin": distribution.get("support", 0) - distribution.get("oppose", 0),
                "confidence": 0.75
            }
        
        # 预测谈判成功概率
        if "negotiation_records" in current_data:
            negotiations_data = current_data["negotiation_records"]
            current_success_rate = negotiations_data.get("success_rate", 0.5)
            
            predictions["negotiation_success"] = {
                "success_probability": current_success_rate * 1.1,  # 简化的趋势预测
                "expected_rounds": int(negotiations_data.get("average_duration", 50) / 10),
                "confidence": 0.65
            }
        
        return {
            "analysis_type": "predictive",
            "predictions": predictions,
            "methodology": "trend_extrapolation_and_pattern_recognition",
            "confidence_level": 0.7,
            "limitations": ["limited_historical_data", "simplified_assumptions"]
        }
    
    async def _perform_diagnostic_analysis(self) -> Dict[str, Any]:
        """执行诊断性分析"""
        # 分析问题的根本原因
        current_data = self.state.collected_data
        
        diagnostics = {}
        
        # 诊断立场分歧原因
        if "country_positions" in current_data and "theory_consistency_scores" in current_data:
            positions_data = current_data["country_positions"]
            theory_data = current_data["theory_consistency_scores"]
            
            diagnostics["position_divergence"] = {
                "primary_factors": ["economic_interests", "political_alignment", "theoretical_preferences"],
                "theory_correlation": theory_data.get("consistency_scores", {}),
                "confidence_level": 0.8
            }
        
        # 诊断谈判障碍
        if "negotiation_records" in current_data:
            negotiations_data = current_data["negotiation_records"]
            success_rate = negotiations_data.get("success_rate", 0.5)
            
            diagnostics["negotiation_barriers"] = {
                "identified_barriers": ["position_rigidity", "communication_gaps", "trust_deficit"],
                "impact_assessment": "high" if success_rate < 0.5 else "medium",
                "recommended_interventions": ["mediation", "confidence_building", "issue_framing"]
            }
        
        return {
            "analysis_type": "diagnostic",
            "diagnostics": diagnostics,
            "root_cause_analysis": "multi_factorial_analysis",
            "confidence_level": 0.75
        }
    
    async def _perform_prescriptive_analysis(self) -> Dict[str, Any]:
        """执行规范性分析"""
        # 提供行动建议
        current_data = self.state.collected_data
        
        recommendations = {}
        
        # 基于立场分析提供建议
        if "country_positions" in current_data:
            positions_data = current_data["country_positions"]
            distribution = positions_data.get("position_distribution", {})
            
            if distribution.get("support", 0) < distribution.get("oppose", 0):
                recommendations["strategy"] = "increase_outreach_to_opposing_countries"
            else:
                recommendations["strategy"] = "maintain_current_momentum"
        
        # 基于谈判分析提供建议
        if "negotiation_records" in current_data:
            negotiations_data = current_data["negotiation_records"]
            success_rate = negotiations_data.get("success_rate", 0.5)
            
            if success_rate < 0.6:
                recommendations["negotiation_improvement"] = [
                    "enhance_preparation",
                    "improve_communication_channels",
                    "consider_third_party_mediation"
                ]
        
        return {
            "analysis_type": "prescriptive",
            "recommendations": recommendations,
            "action_priorities": ["improve_negotiation_outcomes", "build_consensus", "monitor_trends"],
            "implementation_timeline": "immediate_to_medium_term",
            "expected_impact": "moderate_to_high"
        }
    
    async def _generate_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合分析结果"""
        # 整合所有分析结果
        comprehensive = {
            "overall_assessment": "moderate_progress_with_challenges",
            "key_metrics": self._extract_key_metrics(results),
            "trend_analysis": self._analyze_trends(results),
            "risk_assessment": self._assess_risks(results),
            "opportunity_identification": self._identify_opportunities(results)
        }
        
        return comprehensive
    
    def _extract_descriptive_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """提取描述性洞察"""
        insights = []
        
        if "position_analysis" in metrics:
            pos_analysis = metrics["position_analysis"]
            if pos_analysis.get("average_confidence", 0) > 0.7:
                insights.append("High confidence in country positions indicates clear preferences")
            else:
                insights.append("Moderate confidence suggests potential for position changes")
        
        if "negotiation_analysis" in metrics:
            neg_analysis = metrics["negotiation_analysis"]
            if neg_analysis.get("success_rate", 0) > 0.6:
                insights.append("Negotiation process showing positive outcomes")
            else:
                insights.append("Negotiation challenges require attention")
        
        return insights
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """提取关键指标"""
        key_metrics = {}
        
        if "descriptive" in results:
            desc_metrics = results["descriptive"].get("metrics", {})
            if "position_analysis" in desc_metrics:
                key_metrics["position_confidence"] = desc_metrics["position_analysis"].get("average_confidence", 0)
            if "negotiation_analysis" in desc_metrics:
                key_metrics["negotiation_success_rate"] = desc_metrics["negotiation_analysis"].get("success_rate", 0)
        
        if "predictive" in results:
            pred_metrics = results["predictive"].get("predictions", {})
            if "voting_outcome" in pred_metrics:
                key_metrics["voting_success_probability"] = pred_metrics["voting_outcome"].get("success_probability", 0)
        
        return key_metrics
    
    def _analyze_trends(self, results: Dict[str, Any]) -> Dict[str, str]:
        """分析趋势"""
        trends = {}
        
        # 基于预测分析确定趋势
        if "predictive" in results:
            predictions = results["predictive"].get("predictions", {})
            if "voting_outcome" in predictions:
                success_prob = predictions["voting_outcome"].get("success_probability", 0.5)
                trends["voting_trend"] = "positive" if success_prob > 0.6 else "negative" if success_prob < 0.4 else "stable"
            
            if "negotiation_success" in predictions:
                success_prob = predictions["negotiation_success"].get("success_probability", 0.5)
                trends["negotiation_trend"] = "improving" if success_prob > 0.6 else "declining" if success_prob < 0.4 else "stable"
        
        return trends
    
    def _assess_risks(self, results: Dict[str, Any]) -> List[str]:
        """评估风险"""
        risks = []
        
        if "diagnostic" in results:
            diagnostics = results["diagnostic"].get("diagnostics", {})
            if "position_divergence" in diagnostics:
                risks.append("Continued position divergence may impede consensus")
            if "negotiation_barriers" in diagnostics:
                risks.append("Negotiation barriers could lead to process breakdown")
        
        if "predictive" in results:
            predictions = results["predictive"].get("predictions", {})
            if "voting_outcome" in predictions:
                success_prob = predictions["voting_outcome"].get("success_probability", 0.5)
                if success_prob < 0.5:
                    risks.append("Low probability of achieving voting success")
        
        return risks
    
    def _identify_opportunities(self, results: Dict[str, Any]) -> List[str]:
        """识别机会"""
        opportunities = []
        
        if "descriptive" in results:
            desc_metrics = results["descriptive"].get("metrics", {})
            if "position_analysis" in desc_metrics:
                avg_confidence = desc_metrics["position_analysis"].get("average_confidence", 0)
                if avg_confidence < 0.7:
                    opportunities.append("Moderate confidence levels suggest room for persuasion")
        
        if "prescriptive" in results:
            recommendations = results["prescriptive"].get("recommendations", {})
            if "strategy" in recommendations:
                opportunities.append("Clear strategic direction available for implementation")
        
        return opportunities
    
    async def _update_real_time_metrics(self, comprehensive_result: Dict[str, Any]) -> None:
        """更新实时指标"""
        timestamp = datetime.now()
        
        # 更新关键指标
        key_metrics = comprehensive_result.get("key_metrics", {})
        
        for metric_name, metric_value in key_metrics.items():
            metric = AnalysisMetric(
                metric_id=f"real_time_{metric_name}",
                metric_type=MetricType.DECISION_PROCESS,
                name=metric_name.replace("_", " ").title(),
                description=f"Real-time {metric_name} metric",
                value=metric_value,
                unit="score",
                timestamp=timestamp,
                source="real_time_analysis"
            )
            
            self.state.real_time_metrics[metric_name] = metric
    
    async def _generate_report(self, report_request: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析报告"""
        report_type = report_request.get("report_type", "comprehensive")
        audience = report_request.get("audience", "analysts")
        format_type = report_request.get("format", "json")
        
        # 收集所有相关数据
        report_data = {
            "executive_summary": await self._generate_executive_summary(),
            "detailed_analysis": await self._generate_detailed_analysis(),
            "methodology": self._get_methodology_description(),
            "data_sources": self.state.data_sources,
            "limitations": self._get_analysis_limitations(),
            "recommendations": await self._generate_strategic_recommendations(),
            "appendices": await self._generate_appendices()
        }
        
        # 根据受众调整报告
        if audience == "executives":
            report_data = self._tailor_for_executives(report_data)
        elif audience == "policy_makers":
            report_data = self._tailor_for_policy_makers(report_data)
        
        # 格式化报告
        formatted_report = self._format_report(report_data, format_type)
        
        return {
            "analyst_id": self.name,
            "report_request": report_request,
            "report": formatted_report,
            "metadata": {
                "generation_timestamp": datetime.now(),
                "report_type": report_type,
                "audience": audience,
                "format": format_type,
                "data_freshness": "real_time"
            }
        }
    
    async def _generate_executive_summary(self) -> Dict[str, Any]:
        """生成执行摘要"""
        return {
            "current_status": "moderate_progress",
            "key_findings": [
                "Negotiation process showing mixed results",
                "Position convergence potential identified",
                "Theory frameworks providing useful insights"
            ],
            "critical_issues": [
                "Position divergence remains significant",
                "Communication effectiveness needs improvement"
            ],
            "recommendations": [
                "Enhance mediation efforts",
                "Focus on common ground identification",
                "Leverage theoretical insights for strategy"
            ],
            "success_probability": 0.65
        }
    
    async def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """生成详细分析"""
        return {
            "quantitative_analysis": {
                "position_dynamics": self.state.collected_data.get("country_positions", {}),
                "negotiation_metrics": self.state.collected_data.get("negotiation_records", {}),
                "voting_projections": self.state.collected_data.get("voting_results", {})
            },
            "qualitative_analysis": {
                "theoretical_insights": self.state.collected_data.get("theory_consistency_scores", {}),
                "communication_patterns": self.state.collected_data.get("communication_networks", {}),
                "stakeholder_perspectives": "analyzed_through_theoretical_frameworks"
            },
            "comparative_analysis": {
                "historical_comparisons": "limited_by_data_availability",
                "cross_theory_validation": "ongoing",
                "scenario_testing": "preliminary_results_available"
            }
        }
    
    def _get_methodology_description(self) -> Dict[str, str]:
        """获取方法论描述"""
        return {
            "data_collection": "Real-time data collection from multiple sources",
            "analysis_approach": "Mixed methods combining quantitative and qualitative analysis",
            "theoretical_framework": "Multi-theory approach with consistency validation",
            "statistical_methods": "Descriptive statistics, trend analysis, predictive modeling",
            "quality_assurance": "Multi-level data quality validation and cross-checking"
        }
    
    def _get_analysis_limitations(self) -> List[str]:
        """获取分析局限性"""
        return [
            "Limited historical data for trend analysis",
            "Simplified assumptions in predictive models",
            "Real-time data may contain noise",
            "Theoretical framework validation ongoing",
            "Sample size limitations for statistical significance"
        ]
    
    async def _generate_strategic_recommendations(self) -> List[Dict[str, Any]]:
        """生成战略建议"""
        return [
            {
                "recommendation": "Enhance negotiation preparation",
                "priority": "high",
                "timeline": "immediate",
                "expected_impact": "moderate_to_high",
                "implementation_complexity": "medium"
            },
            {
                "recommendation": "Improve communication channels",
                "priority": "medium",
                "timeline": "short_term",
                "expected_impact": "moderate",
                "implementation_complexity": "low"
            },
            {
                "recommendation": "Leverage theoretical insights",
                "priority": "medium",
                "timeline": "ongoing",
                "expected_impact": "moderate",
                "implementation_complexity": "low"
            }
        ]
    
    async def _generate_appendices(self) -> Dict[str, Any]:
        """生成附录"""
        return {
            "raw_data_summary": self.state.collected_data,
            "detailed_metrics": {k: v.__dict__ for k, v in self.state.real_time_metrics.items()},
            "analysis_history": [r.__dict__ for r in self.state.analysis_history[-5:]],  # 最近5个分析
            "data_quality_reports": self.state.alerts[-10:] if self.state.alerts else []  # 最近10个警报
        }
    
    def _tailor_for_executives(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """为高管定制报告"""
        # 简化技术细节，强调业务影响
        return {
            "executive_summary": report_data["executive_summary"],
            "key_metrics": self._extract_key_metrics({"descriptive": {"metrics": {}}}),
            "business_impact": "moderate_with_improvement_potential",
            "action_items": ["enhance_negotiation", "improve_communication", "monitor_trends"],
            "roi_projection": "positive_if_recommendations_implemented"
        }
    
    def _tailor_for_policy_makers(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """为政策制定者定制报告"""
        # 强调政策含义和实施建议
        return {
            "policy_implications": "significant_for_trade_policy",
            "stakeholder_analysis": "diverse_interests_require_balanced_approach",
            "implementation_roadmap": "phased_approach_recommended",
            "risk_mitigation": "multiple_strategies_available",
            "success_criteria": "measurable_outcomes_defined"
        }
    
    def _format_report(self, report_data: Dict[str, Any], format_type: str) -> Any:
        """格式化报告"""
        if format_type == "json":
            return report_data
        elif format_type == "summary":
            return {
                "summary": report_data["executive_summary"],
                "key_recommendations": report_data["recommendations"][:3]
            }
        else:
            return report_data
    
    async def _act(self) -> Message:
        """主要行动方法"""
        # 获取最新消息
        msg = self._rc.msgs[-1] if self._rc.msgs else None
        
        if not msg:
            return Message(content="No message received", role=self.name, send_to=self.name)
        
        # 根据消息类型执行相应动作
        if hasattr(msg, 'action_type'):
            action_type = msg.action_type
            content = msg.content
            
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    content = {"data": content}
            
            if action_type == "data_collection":
                result = await self._collect_data(content)
            elif action_type == "real_time_analysis":
                result = await self._perform_real_time_analysis(content)
            elif action_type == "report_generation":
                result = await self._generate_report(content)
            else:
                result = {"error": f"Unknown action type: {action_type}"}
        else:
            # 默认执行实时分析
            result = await self._perform_real_time_analysis({})
        
        return Message(
            content=json.dumps(result, ensure_ascii=False, default=str),
            role=self.name,
            send_to=self.name
        )


# 定义动作类
class DataCollectionAction(Action):
    """数据收集动作"""
    
    async def run(self, collection_request: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "data_collection", "data": collection_request}


class RealTimeAnalysisAction(Action):
    """实时分析动作"""
    
    async def run(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "real_time_analysis", "data": analysis_request}


class ReportGenerationAction(Action):
    """报告生成动作"""
    
    async def run(self, report_request: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "report_generation", "data": report_request}
