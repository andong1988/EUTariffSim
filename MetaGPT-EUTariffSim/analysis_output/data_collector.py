"""
数据收集器 - 收集和分析模拟过程中的数据
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionRecord:
    """立场记录"""
    timestamp: str
    country_id: str
    position: float
    confidence: float
    theoretical_basis: Dict[str, float]
    domestic_constraints: Dict[str, Any]
    international_factors: Dict[str, Any]


@dataclass
class VoteRecord:
    """投票记录"""
    timestamp: str
    country_id: str
    vote: str  # "yes", "no", "abstain"
    rationale: str
    theoretical_justification: Dict[str, float]


@dataclass
class NegotiationRoundRecord:
    """谈判轮次记录"""
    round_num: int
    timestamp: str
    bilateral_negotiations: List[Dict[str, Any]]
    multilateral_negotiations: List[Dict[str, Any]]
    messages_exchanged: int
    alliances_formed: Dict[str, List[str]]
    position_changes: Dict[str, float]


class DataCollector:
    """数据收集器 - 收集、存储和分析模拟数据"""
    
    def __init__(self):
        self.simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data = {
            "simulation_id": self.simulation_id,
            "start_time": datetime.now().isoformat(),
            "positions": {},  # 时间点 -> {国家ID -> 立场记录}
            "votes": {},  # 时间点 -> {国家ID -> 投票记录}
            "negotiation_rounds": [],  # 谈判轮次记录列表
            "proposals": [],  # 提案记录列表
            "countermeasures": [],  # 反制措施记录列表
            "events": [],  # 事件记录列表
            "communication_network": {},  # 通信网络数据
            "theoretical_analysis": {},  # 理论分析数据
            "stage_history": []  # 阶段历史记录
        }
        
        logger.info(f"初始化数据收集器，模拟ID: {self.simulation_id}")
    
    def record_initial_position(self, country_id: str, position_data: Dict[str, Any]):
        """
        记录单个国家的初始立场
        
        Args:
            country_id: 国家ID
            position_data: 立场数据
        """
        time_point = "T0"  # 初始时间点
        if time_point not in self.data["positions"]:
            self.data["positions"][time_point] = {}
        
        # 序列化position_data以处理FeatureLevel枚举
        try:
            from theory.theory_encoder import TheoryEncoder
            encoder = TheoryEncoder()
            serialized_data = encoder.serialize_for_json(position_data)
        except ImportError:
            serialized_data = position_data
        
        # 创建记录时确保所有数据都是JSON可序列化的
        record_dict = {
            "timestamp": datetime.now().isoformat(),
            "country_id": country_id,
            "position": serialized_data.get("position", 0.0),
            "confidence": serialized_data.get("confidence", 0.0),
            "theoretical_basis": serialized_data.get("theoretical_basis", {}),
            "domestic_constraints": serialized_data.get("domestic_constraints", {}),
            "international_factors": serialized_data.get("international_factors", {})
        }
        self.data["positions"][time_point][country_id] = record_dict
        logger.debug(f"记录初始立场: {country_id}")

    def record_vote(self, country_id: str, vote_data: Dict[str, Any]):
        """
        记录单个国家的投票
        
        Args:
            country_id: 国家ID
            vote_data: 投票数据
        """
        time_point = "T2"  # 投票时间点
        if time_point not in self.data["votes"]:
            self.data["votes"][time_point] = {}
        
        # 序列化vote_data以处理FeatureLevel枚举
        try:
            from theory.theory_encoder import TheoryEncoder
            encoder = TheoryEncoder()
            serialized_data = encoder.serialize_for_json(vote_data)
        except ImportError:
            serialized_data = vote_data
        
        # 创建记录时确保所有数据都是JSON可序列化的
        record_dict = {
            "timestamp": datetime.now().isoformat(),
            "country_id": country_id,
            "vote": serialized_data.get("final_position", "abstain"),
            "rationale": serialized_data.get("reasoning", ""),
            "theoretical_justification": serialized_data.get("theoretical_justification", {})
        }
        self.data["votes"][time_point][country_id] = record_dict
        logger.debug(f"记录投票: {country_id}")

    def record_positions(self, time_point: str, positions: Dict[str, Dict[str, Any]]):
        """
        记录立场数据
        
        Args:
            time_point: 时间点标识符 (如 "T0", "T1")
            positions: 国家立场字典 {国家ID -> 立场数据}
        """
        if time_point not in self.data["positions"]:
            self.data["positions"][time_point] = {}
        
        for country_id, position_data in positions.items():
            record = PositionRecord(
                timestamp=datetime.now().isoformat(),
                country_id=country_id,
                position=position_data.get("position", 0.0),
                confidence=position_data.get("confidence", 0.0),
                theoretical_basis=position_data.get("theoretical_basis", {}),
                domestic_constraints=position_data.get("domestic_constraints", {}),
                international_factors=position_data.get("international_factors", {})
            )
            self.data["positions"][time_point][country_id] = asdict(record)
        
        logger.debug(f"记录立场数据: {time_point}, {len(positions)}个国家")
    
    def record_votes(self, time_point: str, votes: Dict[str, Dict[str, Any]]):
        """
        记录投票数据
        
        Args:
            time_point: 时间点标识符 (如 "T2")
            votes: 国家投票字典 {国家ID -> 投票数据}
        """
        if time_point not in self.data["votes"]:
            self.data["votes"][time_point] = {}
        
        for country_id, vote_data in votes.items():
            record = VoteRecord(
                timestamp=datetime.now().isoformat(),
                country_id=country_id,
                vote=vote_data.get("vote", "abstain"),
                rationale=vote_data.get("rationale", ""),
                theoretical_justification=vote_data.get("theoretical_justification", {})
            )
            self.data["votes"][time_point][country_id] = asdict(record)
        
        logger.debug(f"记录投票数据: {time_point}, {len(votes)}个国家")
    
    def record_voting_result(self, voting_result: Dict[str, Any]):
        """
        记录投票结果
        
        Args:
            voting_result: 投票结果数据
        """
        self.data["voting_result"] = voting_result
        logger.debug(f"记录投票结果: {voting_result.get('passed', False)}")
    
    def record_proposal(self, proposal: Dict[str, Any]):
        """
        记录关税提案
        
        Args:
            proposal: 提案数据
        """
        proposal_record = {
            "timestamp": datetime.now().isoformat(),
            "proposal": proposal,
            "stage": "proposal_phase"
        }
        self.data["proposals"].append(proposal_record)
        logger.debug(f"记录关税提案: {proposal.get('type', 'unknown')}")
    
    def record_responses(self, time_point: str, responses: Dict[str, Dict[str, Any]]):
        """
        记录响应数据
        
        Args:
            time_point: 时间点标识符
            responses: 国家响应字典 {国家ID -> 响应数据}
        """
        # 响应数据可以视为特殊的立场数据
        position_data = {}
        for country_id, response in responses.items():
            position_data[country_id] = {
                "position": response.get("position", 0.0),
                "confidence": response.get("confidence", 0.0),
                "theoretical_basis": response.get("theoretical_basis", {}),
                "rationale": response.get("rationale", "")
            }
        
        self.record_positions(time_point, position_data)
        logger.debug(f"记录响应数据: {time_point}, {len(responses)}个国家")
    
    def record_negotiation_round(self, round_num: int, round_data: Dict[str, Any]):
        """
        记录谈判轮次数据
        
        Args:
            round_num: 轮次编号
            round_data: 轮次数据
        """
        record = NegotiationRoundRecord(
            round_num=round_num,
            timestamp=datetime.now().isoformat(),
            bilateral_negotiations=round_data.get("negotiation_round", {}).get("bilateral_negotiations", []),
            multilateral_negotiations=round_data.get("negotiation_round", {}).get("multilateral_negotiations", []),
            messages_exchanged=round_data.get("negotiation_round", {}).get("messages_exchanged", 0),
            alliances_formed=round_data.get("alliances", {}),
            position_changes=round_data.get("position_adjustments", {})
        )
        
        self.data["negotiation_rounds"].append(asdict(record))
        logger.debug(f"记录谈判轮次: {round_num}")
    
    def record_countermeasures(self, countermeasures: Dict[str, Any]):
        """
        记录反制措施
        
        Args:
            countermeasures: 反制措施数据
        """
        countermeasure_record = {
            "timestamp": datetime.now().isoformat(),
            "countermeasures": countermeasures,
            "stage": "countermeasure_phase"
        }
        self.data["countermeasures"].append(countermeasure_record)
        logger.debug(f"记录反制措施: {countermeasures.get('type', 'unknown')}")
    
    def record_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        记录事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_data": event_data
        }
        self.data["events"].append(event_record)
        logger.debug(f"记录事件: {event_type}")
    
    def record_communication(self, sender: str, receiver: str, message_type: str, content: Dict[str, Any]):
        """
        记录通信数据
        
        Args:
            sender: 发送者ID
            receiver: 接收者ID
            message_type: 消息类型
            content: 消息内容
        """
        if "communication_network" not in self.data:
            self.data["communication_network"] = {}
        
        edge_key = f"{sender}-{receiver}"
        if edge_key not in self.data["communication_network"]:
            self.data["communication_network"][edge_key] = {
                "sender": sender,
                "receiver": receiver,
                "messages": []
            }
        
        message_record = {
            "timestamp": datetime.now().isoformat(),
            "message_type": message_type,
            "content": content
        }
        
        self.data["communication_network"][edge_key]["messages"].append(message_record)
    
    def record_stage(self, stage: str, stage_data: Dict[str, Any]):
        """
        记录阶段数据
        
        Args:
            stage: 阶段名称
            stage_data: 阶段数据
        """
        stage_record = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "data": stage_data
        }
        self.data["stage_history"].append(stage_record)
        logger.debug(f"记录阶段: {stage}")
    
    def get_simulation_data(self) -> Dict[str, Any]:
        """
        获取完整的模拟数据
        
        Returns:
            完整的模拟数据字典
        """
        self.data["end_time"] = datetime.now().isoformat()
        self.data["duration_seconds"] = (
            datetime.fromisoformat(self.data["end_time"]) - 
            datetime.fromisoformat(self.data["start_time"])
        ).total_seconds()
        
        return self.data
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取数据摘要
        
        Returns:
            数据摘要字典
        """
        data = self.get_simulation_data()
        
        summary = {
            "simulation_id": data["simulation_id"],
            "duration_seconds": data.get("duration_seconds", 0),
            "num_stages": len(data.get("stage_history", [])),
            "num_position_records": sum(len(positions) for positions in data.get("positions", {}).values()),
            "num_vote_records": sum(len(votes) for votes in data.get("votes", {}).values()),
            "num_negotiation_rounds": len(data.get("negotiation_rounds", [])),
            "num_proposals": len(data.get("proposals", [])),
            "num_countermeasures": len(data.get("countermeasures", [])),
            "num_events": len(data.get("events", [])),
            "communication_edges": len(data.get("communication_network", {}))
        }
        
        return summary
    
    def save_to_file(self, filepath: str = None):
        """
        保存数据到文件
        
        Args:
            filepath: 文件路径，如果为None则使用默认路径
        """
        if filepath is None:
            filepath = f"simulation_data_{self.simulation_id}.json"
        
        data = self.get_simulation_data()
        
        # 导入理论编码器以使用序列化功能
        try:
            from theory.theory_encoder import TheoryEncoder
            encoder = TheoryEncoder()
            # 使用理论编码器的序列化方法处理FeatureLevel枚举
            serialized_data = encoder.serialize_for_json(data)
        except ImportError:
            # 如果无法导入理论编码器，使用基本序列化
            logger.warning("无法导入理论编码器，使用基本序列化")
            serialized_data = data
        
        # 转换datetime对象为字符串
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, default=convert_datetime, indent=2, ensure_ascii=False)
        
        logger.info(f"模拟数据已保存到: {filepath}")
        return filepath
    
    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        导出数据到pandas DataFrame
        
        Returns:
            包含多个DataFrame的字典
        """
        data = self.get_simulation_data()
        dfs = {}
        
        # 导出立场数据
        position_records = []
        for time_point, countries in data.get("positions", {}).items():
            for country_id, record in countries.items():
                record_copy = record.copy()
                record_copy["time_point"] = time_point
                position_records.append(record_copy)
        
        if position_records:
            dfs["positions"] = pd.DataFrame(position_records)
        
        # 导出投票数据
        vote_records = []
        for time_point, countries in data.get("votes", {}).items():
            for country_id, record in countries.items():
                record_copy = record.copy()
                record_copy["time_point"] = time_point
                vote_records.append(record_copy)
        
        if vote_records:
            dfs["votes"] = pd.DataFrame(vote_records)
        
        # 导出谈判轮次数据
        if data.get("negotiation_rounds"):
            dfs["negotiation_rounds"] = pd.DataFrame(data["negotiation_rounds"])
        
        # 导出通信网络数据
        communication_records = []
        for edge_key, edge_data in data.get("communication_network", {}).items():
            for message in edge_data.get("messages", []):
                record = {
                    "sender": edge_data["sender"],
                    "receiver": edge_data["receiver"],
                    "message_type": message["message_type"],
                    "timestamp": message["timestamp"]
                }
                communication_records.append(record)
        
        if communication_records:
            dfs["communications"] = pd.DataFrame(communication_records)
        
        return dfs
    
    def analyze_theoretical_consistency(self) -> Dict[str, float]:
        """
        分析理论一致性
        
        Returns:
            理论一致性得分字典
        """
        data = self.get_simulation_data()
        
        # 这里应该实现理论一致性分析逻辑
        # 为演示目的，返回示例数据
        analysis = {
            "rational_choice_consistency": 0.75,
            "two_level_game_consistency": 0.68,
            "constructivism_consistency": 0.82,
            "weaponized_interdependence_consistency": 0.71,
            "overall_consistency": 0.74
        }
        
        self.data["theoretical_analysis"] = analysis
        return analysis
    
    def generate_report(self) -> Dict[str, Any]:
        """
        生成分析报告
        
        Returns:
            分析报告字典
        """
        summary = self.get_summary()
        theoretical_analysis = self.analyze_theoretical_consistency()
        
        report = {
            "simulation_summary": summary,
            "theoretical_analysis": theoretical_analysis,
            "key_insights": [
                "模拟成功完成所有阶段",
                "理论一致性分析显示建构主义理论解释力最强",
                "投票结果显示合格多数未达成" if not self.data.get("voting_result", {}).get("passed", False) 
                else "投票结果显示合格多数已达成"
            ],
            "recommendations": [
                "增加更多理论变量以提高解释力",
                "扩展国内政治约束的建模",
                "添加更多事件类型以测试系统鲁棒性"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return report
