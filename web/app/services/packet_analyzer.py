from scapy.all import rdpcap, IP, TCP, UDP
from datetime import datetime
from typing import Dict, Any, List

class PacketAnalyzer:
    def analyze_pcap(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyzes a pcap file and returns a list of packet information.
        """
        packets_info = []
        try:
            packets = rdpcap(file_path)
            for packet in packets:
                timestamp = packet.time
                
                protocol = None
                if packet.haslayer(TCP):
                    protocol = 'TCP'
                elif packet.haslayer(UDP):
                    protocol = 'UDP'
                elif packet.haslayer(IP):
                    protocol = 'IP'

                flags = None
                if TCP in packet:
                    flags = str(packet[TCP].flags)

                info = {
                    'timestamp': datetime.fromtimestamp(float(timestamp)).isoformat(),
                    'type': 'network_packet',
                    'src_ip': packet[IP].src if IP in packet else None,
                    'src_port': packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else None),
                    'dst_ip': packet[IP].dst if IP in packet else None,
                    'dst_port': packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else None),
                    'protocol': protocol,
                    'flags': flags,
                    'length': len(packet)
                }
                packets_info.append(info)
        except Exception as e:
            raise ValueError(f"Error reading pcap file {file_path}: {e}")
            
        return packets_info