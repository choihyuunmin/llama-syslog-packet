from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PacketAnalysisRequest(BaseModel):
    """패킷 분석 요청 모델"""
    file_id: str
    filename: str
    question: str

class PacketAnalysisResponse(BaseModel):
    """패킷 분석 응답 모델"""
    answer: str
    analysis: Dict[str, Any]

class BasicStats(BaseModel):
    """기본 통계 정보 모델"""
    total_packets: int
    total_sessions: int
    start_time: str
    end_time: str
    total_bytes: int

class ProtocolDistribution(BaseModel):
    """프로토콜 분포 모델"""
    distribution: Dict[int, int]
    most_common: Optional[int]

class TrafficPattern(BaseModel):
    """트래픽 패턴 모델"""
    hourly_distribution: Dict[int, int]
    busiest_hour: Optional[int]

class SecurityAnalysis(BaseModel):
    """보안 분석 모델"""
    suspicious_patterns: List[str]
    security_level: str

class Visualizations(BaseModel):
    """시각화 모델"""
    timeline: str
    protocol_distribution: str

class AnalysisResult(BaseModel):
    """분석 결과 모델"""
    basic_stats: BasicStats
    protocol_dist: ProtocolDistribution
    traffic_pattern: TrafficPattern
    security_analysis: SecurityAnalysis
    visualizations: Visualizations 