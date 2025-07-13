import os
from pathlib import Path
import pyshark
import re
from typing import List, Dict, Any, Optional
import logging
from web.app.core.config import settings

# logger 설정
logger = logging.getLogger(__name__)

class FileAnalysisError(Exception):
    pass

class InvalidFileTypeError(Exception):
    pass


def analyze_pcap(file_path: Path) -> Dict[str, Any]:
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File Not Found: {file_path}")
            
        logger.info(f"PCAP File Analysis Start: {file_path}")
        capture = pyshark.FileCapture(str(file_path))
        
        stats: Dict[str, Any] = {
            "total_packets": 0,
            "protocols": {},
            "source_ips": {},
            "destination_ips": {},
            "top_conversations": [],
            "analysis_time": None
        }
        
        packet_count = 0
        for packet in capture:
            packet_count += 1
            stats["total_packets"] = packet_count
            
            if hasattr(packet, 'highest_layer'):
                protocol = packet.highest_layer
                stats["protocols"][protocol] = stats["protocols"].get(protocol, 0) + 1
            
            if hasattr(packet, 'ip'):
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                stats["source_ips"][src_ip] = stats["source_ips"].get(src_ip, 0) + 1
                stats["destination_ips"][dst_ip] = stats["destination_ips"].get(dst_ip, 0) + 1
        
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
        
        logger.info(f"PCAP File Analysis Complete: {packet_count} packets analyzed")
        return stats
        
    except FileNotFoundError:
        logger.error(f"PCAP File Not Found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"PCAP File Analysis Error: {str(e)}")
        raise FileAnalysisError(f"PCAP File Analysis Failed: {str(e)}")


def analyze_log(file_path: Path) -> Dict[str, Any]:
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File Not Found: {file_path}")
            
        logger.info(f"Log File Analysis Start: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            logs = f.readlines()
        
        stats: Dict[str, Any] = {
            "total_lines": len(logs),
            "error_count": 0,
            "warning_count": 0,
            "ip_addresses": set(),
            "timestamps": [],
            "common_patterns": {},
            "analysis_time": None
        }
        
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        
        for line in logs:
            ips = re.findall(ip_pattern, line)
            stats["ip_addresses"].update(ips)
            
            timestamps = re.findall(timestamp_pattern, line)
            stats["timestamps"].extend(timestamps)
            
            if 'ERROR' in line.upper():
                stats["error_count"] += 1
            elif 'WARNING' in line.upper():
                stats["warning_count"] += 1
            
            words = line.split()
            for word in words:
                if len(word) > 3:
                    stats["common_patterns"][word] = stats["common_patterns"].get(word, 0) + 1
        
        stats["ip_addresses"] = list(stats["ip_addresses"])
        stats["top_patterns"] = dict(sorted(
            stats["common_patterns"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        logger.info(f"Log File Analysis Complete: {len(logs)} lines analyzed")
        return stats
        
    except FileNotFoundError:
        logger.error(f"Log File Not Found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Log File Analysis Error: {str(e)}")
        raise FileAnalysisError(f"Log File Analysis Failed: {str(e)}")


def get_file_type(file_path: Path) -> str:
    extension = file_path.suffix.lower()
    
    if extension == '.pcap':
        return 'pcap'
    elif extension in ['.log', '.txt']:
        return 'log'
    else:
        return 'unknown'


def validate_file_upload(filename: str, file_size: int) -> None:
    file_path = Path(filename)
    file_type = get_file_type(file_path)
    
    if file_type == 'unknown':
        raise InvalidFileTypeError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported extensions: {', '.join(settings.allowed_extensions)}"
        )
    
    if file_size > settings.max_file_size:
        max_size_mb = settings.max_file_size / (1024 * 1024)
        raise ValueError(f"File size exceeds limit. Maximum size: {max_size_mb}MB") 