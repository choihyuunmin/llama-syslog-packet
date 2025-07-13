from scapy.all import *
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

class PcapProcessor:
    def __init__(self, pcap_file: str):
        self.pcap_file = Path(pcap_file)
        self.packets: List[Dict] = []
        self.sessions: Dict[str, List[Dict]] = {}
        self.analysis_results: Dict[str, Any] = {}
        
    def _extract_packet_info(self, packet: Packet) -> Dict:
        try:
            timestamp = float(packet.time)
            # 주요 IP 프로토콜 번호→이름 매핑 (RFC 5237 기준)
            proto_map = {
                1: 'ICMP',
                2: 'IGMP',
                6: 'TCP',
                17: 'UDP',
                41: 'IPv6',
                47: 'GRE',
                50: 'ESP',
                51: 'AH',
                58: 'ICMPv6',
                89: 'OSPF',
                132: 'SCTP',
                88: 'EIGRP',
                103: 'PIM',
                115: 'L2TP',
                27: 'RDP',
                46: 'RSVP',
                137: 'MPLS-in-IP',
                112: 'VRRP',
                4: 'IP-in-IP',
                36: 'XTP',
                94: 'IPIP',
                108: 'IPComp',
                109: 'SNP',
                124: 'ISIS',
                131: 'CFTP',
                143: 'Ethernet',
            }
            proto_num = packet[IP].proto if IP in packet else None
            proto_str = proto_map.get(proto_num, str(proto_num) if proto_num is not None else None)
            info = {
                'timestamp'     : datetime.fromtimestamp(timestamp).isoformat(),
                'src_ip'        : packet[IP].src if IP in packet else None,
                'dst_ip'        : packet[IP].dst if IP in packet else None,
                'src_port'      : packet[TCP].sport if TCP in packet else None,
                'dst_port'      : packet[TCP].dport if TCP in packet else None,
                'protocol'      : proto_str,
                'length'        : len(packet),
                'window'        : packet[TCP].window if TCP in packet else None,
                'payload'       : str(packet[TCP].payload) if TCP in packet and packet[TCP].payload else None
            }

            return info
        except Exception as e:
            logger.error(f"Error extracting packet info: {e}")
            return None
    
    def process_pcap(self):
        try:
            logger.info(f"Pcap file name : {self.pcap_file}")
            packets = rdpcap(str(self.pcap_file))
            
            for packet in packets:
                if IP in packet:
                    info = self._extract_packet_info(packet)
                    if info:
                        self.packets.append(info)
            
            self.generate_dataset()
            logger.info(f"PCAP processing end : {len(self.packets)} packets processed")

            return self.packets  # IP 주소별로 그룹화된 패킷 정보 반환
        except Exception as e:
            logger.error(f"Error processing PCAP: {str(e)}")
            raise

    def generate_dataset(self) -> List[Dict[str, str]]:
        dataset = []
        
        # 1. Basic Analysis Questions
        dataset.extend(self._generate_basic_analysis_questions())
        
        # 2. Advanced Analysis Questions
        dataset.extend(self._generate_advanced_analysis_questions())
        
        # 3. Visualization Questions
        dataset.extend(self._generate_visualization_questions())
        
        # 4. Security Config Questions
        dataset.extend(self._generate_security_config_questions())
        
        return dataset

    def _generate_basic_analysis_questions(self) -> List[Dict[str, str]]:
        """Generate basic analysis questions and answers."""
        questions = []
        
        # Packet count analysis
        questions.append({
            "instruction": "What is the total number of packets captured and what is their distribution by protocol?",
            "output": self._get_protocol_distribution_answer()
        })
        
        # Top talkers analysis
        questions.append({
            "instruction": "Who are the top 5 source and destination IP addresses by packet count?",
            "output": self._get_top_talkers_answer()
        })
        
        # Basic traffic pattern
        questions.append({
            "instruction": "What is the average packet size and how does it vary over time?",
            "output": self._get_packet_size_analysis_answer()
        })
        
        return questions

    def _generate_advanced_analysis_questions(self) -> List[Dict[str, str]]:
        """Generate advanced analysis questions and answers."""
        questions = []
        
        # Protocol anomaly detection
        questions.append({
            "instruction": "Are there any unusual protocol patterns that might indicate security concerns?",
            "output": self._get_protocol_anomaly_answer()
        })
        
        # Session analysis
        questions.append({
            "instruction": "Analyze the TCP session patterns. Are there any signs of scanning or suspicious behavior?",
            "output": self._analyze_tcp_sessions()
        })
        
        return questions

    def _generate_visualization_questions(self) -> List[Dict[str, str]]:
        """Generate visualization-related questions and answers."""
        questions = []
        
        # Protocol distribution pie chart
        questions.append({
            "instruction": "Create a pie chart showing the distribution of network protocols.",
            "output": """
                ```python
                import matplotlib.pyplot as plt

                # Get protocol distribution
                protocol_counts = defaultdict(int)
                for packet in packets:
                    protocol_counts[packet['protocol']] += 1

                # Create pie chart
                plt.figure(figsize=(10, 8))
                plt.pie(protocol_counts.values(), labels=protocol_counts.keys(), autopct='%1.1f%%')
                plt.title('Protocol Distribution')
                plt.show()
                ```
                """
        })
        
        # Traffic volume time series
        questions.append({
            "instruction": "Create a time series plot showing network traffic volume over time.",
            "output": """
                ```python
                import pandas as pd
                import seaborn as sns

                # Create time series data
                time_data = pd.DataFrame([{'timestamp': p['timestamp'], 'size': p['length']} for p in packets])
                time_data['timestamp'] = pd.to_datetime(time_data['timestamp'])
                time_data = time_data.set_index('timestamp').resample('1min').sum()

                # Plot time series
                plt.figure(figsize=(15, 6))
                sns.lineplot(data=time_data, x=time_data.index, y='size')
                plt.title('Network Traffic Volume Over Time')
                plt.xlabel('Time')
                plt.ylabel('Bytes')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                ```
            """
        })
        
        return questions

    def _generate_security_config_questions(self) -> List[Dict[str, str]]:
        """Generate security configuration questions and answers."""
        questions = []
        
        # Firewall rules
        questions.append({
            "instruction": "What firewall rules should be implemented based on the observed suspicious traffic patterns?",
            "output": self._get_firewall_recommendations()
        })
        
        # IDS/IPS settings
        questions.append({
            "instruction": "What IDS/IPS rules should be configured to detect and prevent the observed suspicious activities?",
            "output": self._get_ids_recommendations()
        })
        
        # Network segmentation
        questions.append({
            "instruction": "Based on the traffic analysis, what network segmentation recommendations can be made?",
            "output": self._get_network_segmentation_recommendations()
        })
        
        return questions

    def _get_firewall_recommendations(self) -> str:
        """Generate firewall configuration recommendations."""
        suspicious_patterns = self._analyze_suspicious_patterns()
        
        recommendations = ["Based on the traffic analysis, the following firewall rules are recommended:"]
        
        if suspicious_patterns.get('port_scan', False):
            recommendations.append("""
                # iptables rules to prevent port scanning
                iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
                iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP
                iptables -A INPUT -m state --state INVALID -j DROP
                """)
        
        if suspicious_patterns.get('ddos', False):
            recommendations.append("""
                # Rate limiting rules
                iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
                """)
        
        return "\n".join(recommendations)

    def _get_ids_recommendations(self) -> str:
        """Generate IDS/IPS configuration recommendations."""
        suspicious_patterns = self._analyze_suspicious_patterns()
        
        recommendations = ["Recommended Snort/Suricata rules based on the analysis:"]
        
        if suspicious_patterns.get('port_scan', False):
            recommendations.append("""
                # Port scan detection
                alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:"Potential Port Scan"; flow:stateless; flags:S; threshold:type both,track by_src,count 50,seconds 60; classtype:attempted-recon; sid:1000001; rev:1;)
                """)
        
        if suspicious_patterns.get('data_exfil', False):
            recommendations.append("""
                # Data exfiltration detection
                alert tcp $HOME_NET any -> $EXTERNAL_NET any (msg:"Potential Data Exfiltration"; flow:established; threshold:type both,track by_src,count 1000,seconds 60; byte_test:4,>,1000000,12,relative; classtype:data-loss; sid:1000002; rev:1;)
                """)
        
        return "\n".join(recommendations)

    def _analyze_suspicious_patterns(self) -> Dict[str, bool]:
        if not self.packets:
            return {
                'port_scan': False,
                'ddos': False,
                'data_exfil': False,
                'suspicious_protocols': False
            }
        
        patterns = {
            'port_scan': False,
            'ddos': False,
            'data_exfil': False,
            'suspicious_protocols': False
        }
        
        # 포트 스캔 패턴 분석
        port_scan_patterns = defaultdict(lambda: defaultdict(set))
        for packet in self.packets:
            if 'src_ip' in packet and 'dst_port' in packet:
                timestamp = packet['timestamp'][:13]
                port_scan_patterns[packet['src_ip']][timestamp].add(packet['dst_port'])
        
        for src_ip, time_ports in port_scan_patterns.items():
            for timestamp, ports in time_ports.items():
                if len(ports) > 10:
                    patterns['port_scan'] = True
                    break
        
        # DDoS 패턴 분석
        time_traffic = defaultdict(int)
        for packet in self.packets:
            if 'timestamp' in packet and 'length' in packet:
                time_key = packet['timestamp'][:13]
                time_traffic[time_key] += packet['length']
        
        avg_traffic = sum(time_traffic.values()) / len(time_traffic) if time_traffic else 0
        for traffic in time_traffic.values():
            if traffic > avg_traffic * 5:
                patterns['ddos'] = True
                break
        
        # 데이터 유출 패턴 분석
        src_dst_flows = defaultdict(int)
        for packet in self.packets:
            if 'src_ip' in packet and 'dst_ip' in packet and 'length' in packet:
                key = (packet['src_ip'], packet['dst_ip'])
                src_dst_flows[key] += packet['length']
        
        for flow_size in src_dst_flows.values():
            if flow_size > 1000000:
                patterns['data_exfil'] = True
                break
        
        # 의심스러운 프로토콜 패턴 분석
        protocol_ips = defaultdict(set)
        for packet in self.packets:
            if 'src_ip' in packet and 'protocol' in packet:
                protocol_ips[packet['protocol']].add(packet['src_ip'])
        
        for protocol, ips in protocol_ips.items():
            if len(ips) > 5:
                patterns['suspicious_protocols'] = True
                break
        
        return patterns

    def _analyze_port_scan_activity(self) -> str:
        """Analyze potential port scanning activity."""
        if not self.packets:
            return "No packets to analyze for port scanning activity."
        
        # 포트 스캔 패턴 분석
        port_scan_patterns = defaultdict(lambda: defaultdict(set))
        for packet in self.packets:
            if 'src_ip' in packet and 'dst_port' in packet:
                timestamp = packet['timestamp'][:13]  # 시간단위로 그룹화
                port_scan_patterns[packet['src_ip']][timestamp].add(packet['dst_port'])
        
        # 의심스러운 포트 스캔 탐지
        suspicious_scans = []
        for src_ip, time_ports in port_scan_patterns.items():
            for timestamp, ports in time_ports.items():
                if len(ports) > 10:  # 임계값: 10개 이상의 서로 다른 포트
                    suspicious_scans.append({
                        'src_ip': src_ip,
                        'timestamp': timestamp,
                        'port_count': len(ports)
                    })
        
        if not suspicious_scans:
            return "No significant port scanning activity detected."
        
        result = "Potential port scanning activity detected:\n"
        for scan in suspicious_scans:
            result += f"- Source IP {scan['src_ip']} scanned {scan['port_count']} different ports at {scan['timestamp']}\n"
        
        return result

    def _analyze_tcp_sessions(self) -> str:
        if not self.sessions:
            return "No TCP sessions found in the capture."
        
        session_stats = []
        for session_key, packets in self.sessions.items():
            # 세션당 패킷 수
            packet_count = len(packets)
            # 평균 패킷 크기
            avg_size = sum(p['length'] for p in packets) / packet_count if packet_count > 0 else 0
            # TCP 플래그 분포
            flags = defaultdict(int)
            for p in packets:
                if 'flags' in p:
                    flags[p['flags']] += 1
            
            session_stats.append({
                'session': session_key,
                'packet_count': packet_count,
                'avg_size': avg_size,
                'flags': dict(flags)
            })
        
        # 비정상 세션 탐지
        anomalies = []
        for stats in session_stats:
            # 단방향 통신 (SYN만 있는 경우)
            if stats['flags'].get('S', 0) > 5 and not stats['flags'].get('SA', 0):
                anomalies.append(f"Potential SYN scan detected in session {stats['session']}")
            # 비정상적으로 큰 세션
            if stats['packet_count'] > 1000:
                anomalies.append(f"Large session detected: {stats['session']} with {stats['packet_count']} packets")
        
        if not anomalies:
            return "No significant TCP session anomalies detected."
        
        return "TCP session analysis results:\n" + "\n".join(f"- {anomaly}" for anomaly in anomalies)

    def _get_packet_size_analysis_answer(self) -> str:
        if not self.packets:
            return "No packets to analyze for size patterns."
        
        # 패킷 크기 통계 계산
        sizes = [p['length'] for p in self.packets if 'length' in p]
        if not sizes:
            return "No packet size information available."
        
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        time_buckets = defaultdict(list)
        for packet in self.packets:
            if 'timestamp' in packet and 'length' in packet:
                time_key = packet['timestamp'][:13]
                time_buckets[time_key].append(packet['length'])
        
        result = f"Packet size statistics:\n"
        result += f"- Average size: {avg_size:.2f} bytes\n"
        result += f"- Minimum size: {min_size} bytes\n"
        result += f"- Maximum size: {max_size} bytes\n\n"
        
        result += "Size variation over time:\n"
        for time_key, sizes in sorted(time_buckets.items()):
            avg = sum(sizes) / len(sizes)
            result += f"- {time_key}: Average size = {avg:.2f} bytes\n"
        
        return result

    def _get_protocol_distribution_answer(self) -> str:
        if not self.packets:
            return "No packets to analyze for protocol distribution."
        
        protocol_counts = defaultdict(int)
        for packet in self.packets:
            protocol = packet.get('protocol', 'Unknown')
            protocol_counts[protocol] += 1
        
        total_packets = len(self.packets)
        result = f"Total packets captured: {total_packets}\n\nProtocol distribution:\n"
        
        for protocol, count in sorted(protocol_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_packets) * 100
            result += f"- {protocol}: {count} packets ({percentage:.1f}%)\n"
        
        return result

    def _get_top_talkers_answer(self) -> str:
        if not self.packets:
            return "No packets to analyze for top talkers."
        
        # IP 주소별 패킷 수 계산
        src_ip_counts = defaultdict(int)
        dst_ip_counts = defaultdict(int)
        
        for packet in self.packets:
            if 'src_ip' in packet:
                src_ip_counts[packet['src_ip']] += 1
            if 'dst_ip' in packet:
                dst_ip_counts[packet['dst_ip']] += 1
        
        # 결과 문자열 생성
        result = "Top 5 source IP addresses by packet count:\n"
        for ip, count in sorted(src_ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            result += f"- {ip}: {count} packets\n"
        
        result += "\nTop 5 destination IP addresses by packet count:\n"
        for ip, count in sorted(dst_ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            result += f"- {ip}: {count} packets\n"
        
        return result

    def _get_protocol_anomaly_answer(self) -> str:
        if not self.packets:
            return "No packets to analyze for protocol anomalies."
        
        anomalies = []
        
        protocol_combinations = defaultdict(int)
        for packet in self.packets:
            if 'src_ip' in packet and 'protocol' in packet:
                key = (packet['src_ip'], packet['protocol'])
                protocol_combinations[key] += 1
        
        for (ip, protocol), count in protocol_combinations.items():
            if count > 1000:
                anomalies.append(f"Unusual protocol usage: IP {ip} sent {count} {protocol} packets")
        
        port_protocols = defaultdict(set)
        for packet in self.packets:
            if 'dst_port' in packet and 'protocol' in packet:
                port_protocols[packet['dst_port']].add(packet['protocol'])
        
        for port, protocols in port_protocols.items():
            if len(protocols) > 3: 
                anomalies.append(f"Multiple protocols on port {port}: {', '.join(protocols)}")
        
        if not anomalies:
            return "No significant protocol anomalies detected."
        
        return "Protocol anomalies detected:\n" + "\n".join(f"- {anomaly}" for anomaly in anomalies)

    def _get_network_segmentation_recommendations(self) -> str:
        if not self.packets:
            return "No traffic data available for network segmentation recommendations."
        
        recommendations = ["Based on the traffic analysis, the following network segmentation recommendations are made:"]
        
        ip_groups = defaultdict(set)
        for packet in self.packets:
            if 'src_ip' in packet and 'dst_ip' in packet:
                ip_groups[packet['src_ip']].add(packet['dst_ip'])
        
        for src_ip, dst_ips in ip_groups.items():
            if len(dst_ips) > 10: 
                recommendations.append(f"- Consider isolating {src_ip} in a separate segment due to high connectivity")
        
        protocol_ips = defaultdict(set)
        for packet in self.packets:
            if 'src_ip' in packet and 'protocol' in packet:
                protocol_ips[packet['protocol']].add(packet['src_ip'])
        
        for protocol, ips in protocol_ips.items():
            if len(ips) > 5:  
                recommendations.append(f"- Consider creating a dedicated segment for {protocol} traffic")
        
        return "\n".join(recommendations)
