from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pyshark

class PacketAnalyzer:
    def __init__(self):
        self.packets = []
        self.sessions = {}
    
    def analyze_pcap(self, file_path: str) -> Dict[str, Any]:
        try:
            # pyshark를 사용하여 패킷 분석
            capture = pyshark.FileCapture(file_path)
            self.packets = []
            self.sessions = {}
            
            for packet in capture:
                packet_info = {
                    "timestamp": packet.sniff_time,
                    "protocol": packet.highest_layer,
                    "length": int(packet.length),
                    "source_ip": packet.ip.src if hasattr(packet, 'ip') else None,
                    "destination_ip": packet.ip.dst if hasattr(packet, 'ip') else None,
                    "source_port": packet.tcp.srcport if hasattr(packet, 'tcp') else None,
                    "destination_port": packet.tcp.dstport if hasattr(packet, 'tcp') else None
                }
                self.packets.append(packet_info)
                
                # 세션 정보 업데이트
                if packet_info["source_ip"] and packet_info["destination_ip"]:
                    session_key = f"{packet_info['source_ip']}-{packet_info['destination_ip']}"
                    if session_key not in self.sessions:
                        self.sessions[session_key] = []
                    self.sessions[session_key].append(packet_info)
            
            # 분석 결과 생성
            return {
                "basic_stats": self._get_basic_stats(),
                "protocol_dist": self._get_protocol_distribution(),
                "traffic_pattern": self._analyze_traffic_pattern(),
                "security_analysis": self._analyze_security(),
                "visualizations": self._generate_visualizations()
            }
        except Exception as e:
            raise Exception(f"Error analyzing PCAP file: {str(e)}")
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        return {
            "total_packets": len(self.packets),
            "total_sessions": len(self.sessions),
            "start_time": min(pkt["timestamp"] for pkt in self.packets),
            "end_time": max(pkt["timestamp"] for pkt in self.packets),
        }
    
    def _get_protocol_distribution(self) -> Dict[str, Any]:
        protocol_counts = {}
        for packet in self.packets:
            proto = packet["protocol"]
            protocol_counts[proto] = protocol_counts.get(proto, 0) + 1
        
        return {
            "distribution": protocol_counts,
            "most_common": max(protocol_counts.items(), key=lambda x: x[1])[0] if protocol_counts else None
        }
    
    def _analyze_traffic_pattern(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.packets)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        hourly_counts = df.groupby(df["timestamp"].dt.hour).size()
        
        return {
            "hourly_distribution": hourly_counts.to_dict(),
            "busiest_hour": hourly_counts.idxmax() if not hourly_counts.empty else None
        }
    
    def _analyze_security(self) -> Dict[str, Any]:
        suspicious_patterns = []
        
        port_scan_threshold = 20
        for session in self.sessions.values():
            if len(session) > port_scan_threshold:
                suspicious_patterns.append("Possible port scan detected")
        
        common_ports = {80, 443, 22, 53}
        for packet in self.packets:
            if packet["destination_port"] and int(packet["destination_port"]) not in common_ports:
                suspicious_patterns.append(f"Uncommon port usage: {packet['destination_port']}")
        
        return {
            "suspicious_patterns": suspicious_patterns,
            "security_level": "high" if suspicious_patterns else "normal"
        }
    
    def _generate_visualizations(self) -> Dict[str, str]:
        df = pd.DataFrame(self.packets)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # 트래픽 타임라인
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["length"], "b.")
        plt.title("Packet Size Over Time")
        plt.xlabel("Time")
        plt.ylabel("Packet Size (bytes)")
        timeline_plot = self._plot_to_base64()
        plt.close()
        
        # 프로토콜 분포
        plt.figure(figsize=(10, 6))
        protocol_counts = df["protocol"].value_counts()
        sns.barplot(x=protocol_counts.index, y=protocol_counts.values)
        plt.title("Protocol Distribution")
        plt.xlabel("Protocol")
        plt.ylabel("Count")
        protocol_plot = self._plot_to_base64()
        plt.close()
        
        return {
            "timeline": timeline_plot,
            "protocol_distribution": protocol_plot
        }
    
    def _plot_to_base64(self) -> str:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}" 