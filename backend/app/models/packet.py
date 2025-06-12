from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PacketAnalysisRequest(BaseModel):
    file_id: str
    filename: str
    question: str

class PacketAnalysisResponse(BaseModel):
    answer: str
    analysis: Dict[str, Any]

class BasicStats(BaseModel):
    total_packets: int
    total_sessions: int
    start_time: str
    end_time: str
    total_bytes: int

class ProtocolDistribution(BaseModel):
    distribution: Dict[int, int]
    most_common: Optional[int]

class TrafficPattern(BaseModel):
    hourly_distribution: Dict[int, int]
    busiest_hour: Optional[int]

class SecurityAnalysis(BaseModel):
    suspicious_patterns: List[str]
    security_level: str

class Visualizations(BaseModel):
    timeline: str
    protocol_distribution: str

class AnalysisResult(BaseModel):
    basic_stats: BasicStats
    protocol_dist: ProtocolDistribution
    traffic_pattern: TrafficPattern
    security_analysis: SecurityAnalysis
    visualizations: Visualizations 