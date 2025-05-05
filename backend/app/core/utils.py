from pathlib import Path
import pyshark
import re
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_pcap(file_path: Path) -> Dict[str, Any]:
    """
    PCAP 파일을 분석하여 네트워크 트래픽 정보를 추출합니다.
    """
    try:
        capture = pyshark.FileCapture(str(file_path))
        stats = {
            "total_packets": 0,
            "protocols": {},
            "source_ips": {},
            "destination_ips": {},
            "top_conversations": []
        }
        
        for packet in capture:
            stats["total_packets"] += 1
            
            # 프로토콜 통계
            if hasattr(packet, 'highest_layer'):
                protocol = packet.highest_layer
                stats["protocols"][protocol] = stats["protocols"].get(protocol, 0) + 1
            
            # IP 주소 통계
            if hasattr(packet, 'ip'):
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                stats["source_ips"][src_ip] = stats["source_ips"].get(src_ip, 0) + 1
                stats["destination_ips"][dst_ip] = stats["destination_ips"].get(dst_ip, 0) + 1
        
        # 상위 대화 추출
        conversations = []
        for src_ip in stats["source_ips"]:
            for dst_ip in stats["destination_ips"]:
                count = min(stats["source_ips"][src_ip], stats["destination_ips"][dst_ip])
                if count > 0:
                    conversations.append({
                        "source": src_ip,
                        "destination": dst_ip,
                        "count": count
                    })
        
        stats["top_conversations"] = sorted(
            conversations,
            key=lambda x: x["count"],
            reverse=True
        )[:10]
        
        return stats
    except Exception as e:
        logger.error(f"PCAP 분석 중 오류 발생: {str(e)}")
        raise

def analyze_log(file_path: Path) -> Dict[str, Any]:
    """
    로그 파일을 분석하여 주요 패턴과 통계를 추출합니다.
    """
    try:
        with open(file_path, 'r') as f:
            logs = f.readlines()
        
        stats = {
            "total_lines": len(logs),
            "error_count": 0,
            "warning_count": 0,
            "ip_addresses": set(),
            "timestamps": [],
            "common_patterns": {}
        }
        
        # 로그 패턴 분석
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        
        for line in logs:
            # IP 주소 추출
            ips = re.findall(ip_pattern, line)
            stats["ip_addresses"].update(ips)
            
            # 타임스탬프 추출
            timestamps = re.findall(timestamp_pattern, line)
            stats["timestamps"].extend(timestamps)
            
            # 에러 및 경고 카운트
            if 'ERROR' in line.upper():
                stats["error_count"] += 1
            elif 'WARNING' in line.upper():
                stats["warning_count"] += 1
            
            # 공통 패턴 분석
            words = line.split()
            for word in words:
                if len(word) > 3:  # 너무 짧은 단어는 무시
                    stats["common_patterns"][word] = stats["common_patterns"].get(word, 0) + 1
        
        # IP 주소를 리스트로 변환
        stats["ip_addresses"] = list(stats["ip_addresses"])
        
        # 상위 10개 패턴 추출
        stats["top_patterns"] = dict(sorted(
            stats["common_patterns"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return stats
    except Exception as e:
        logger.error(f"로그 분석 중 오류 발생: {str(e)}")
        raise

def get_file_type(file_path: Path) -> str:
    """
    파일 확장자를 기반으로 파일 타입을 반환합니다.
    """
    extension = file_path.suffix.lower()
    if extension == '.pcap':
        return 'pcap'
    elif extension in ['.log', '.txt']:
        return 'log'
    else:
        return 'unknown' 