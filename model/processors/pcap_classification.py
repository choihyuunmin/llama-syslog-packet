from scapy.all import *
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class AIT_LDS_PacketClassifier:
    """
    Enhanced packet processor for AIT-LDS dataset compatibility
    Implements hierarchical attack classification based on cyber kill chain
    """
    
    def __init__(self, pcap_file: str):
        self.pcap_file = Path(pcap_file)
        self.packets: List[Dict] = []
        self.sessions: Dict[str, List[Dict]] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        # AIT-LDS Attack Categories
        self.attack_categories = {
            'RECONNAISSANCE': {
                'traceroute': {'mitre_id': 'T1018', 'severity': 'LOW'},
                'network_scan': {'mitre_id': 'T1018', 'severity': 'MEDIUM'},
                'dns_scan': {'mitre_id': 'T1018', 'severity': 'MEDIUM'},
                'service_scan': {'mitre_id': 'T1046', 'severity': 'MEDIUM'},
                'wordpress_scan': {'mitre_id': 'T1046', 'severity': 'HIGH'},
                'directory_scan': {'mitre_id': 'T1083', 'severity': 'MEDIUM'}
            },
            'INITIAL_INTRUSION': {
                'webshell_upload': {'mitre_id': 'T1505.003', 'severity': 'HIGH'},
                'webshell_execution': {'mitre_id': 'T1505.003', 'severity': 'HIGH'},
                'backdoor_establishment': {'mitre_id': 'T1505', 'severity': 'CRITICAL'}
            },
            'CREDENTIAL_ACCESS': {
                'database_dump': {'mitre_id': 'T1003', 'severity': 'HIGH'},
                'password_cracking': {'mitre_id': 'T1110.002', 'severity': 'HIGH'},
                'credential_use': {'mitre_id': 'T1078', 'severity': 'MEDIUM'}
            },
            'LATERAL_MOVEMENT': {
                'reverse_shell': {'mitre_id': 'T1059', 'severity': 'HIGH'},
                'privilege_escalation': {'mitre_id': 'T1078', 'severity': 'HIGH'},
                'root_execution': {'mitre_id': 'T1078.003', 'severity': 'CRITICAL'}
            },
            'EXFILTRATION': {
                'dns_exfiltration': {'mitre_id': 'T1048.003', 'severity': 'HIGH'},
                'data_extraction': {'mitre_id': 'T1041', 'severity': 'HIGH'}
            }
        }
        
        # Hierarchical meta-labels
        self.meta_labels = {
            'foothold': ['network_scan', 'dns_scan', 'service_scan', 'webshell_upload'],
            'escalation': ['password_cracking', 'privilege_escalation', 'root_execution'],
            'persistence': ['backdoor_establishment', 'reverse_shell']
        }
        
    def _extract_packet_info(self, packet: Packet) -> Dict:
        """Extract comprehensive packet information for analysis"""
        try:
            timestamp = float(packet.time)
            info = {
                'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                'src_ip': packet[IP].src if IP in packet else None,
                'dst_ip': packet[IP].dst if IP in packet else None,
                'src_port': packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else None),
                'dst_port': packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else None),
                'protocol': packet[IP].proto if IP in packet else None,
                'length': len(packet),
                'flags': packet[TCP].flags if TCP in packet else None,
                'window': packet[TCP].window if TCP in packet else None,
                'payload': str(packet[TCP].payload) if TCP in packet and packet[TCP].payload else None,
                'ttl': packet[IP].ttl if IP in packet else None,
                'fragment_offset': packet[IP].frag if IP in packet else None
            }
            print
            return info
        except Exception as e:
            logger.error(f"Error extracting packet info: {e}")
            return None
    
    def process_pcap(self) -> List[Dict]:
        """Process PCAP file and extract packet information"""
        try:
            logger.info(f"Processing PCAP file: {self.pcap_file}")
            packets = rdpcap(str(self.pcap_file))
            
            for packet in packets:
                if IP in packet:
                    info = self._extract_packet_info(packet)
                    if info:
                        self.packets.append(info)
            
            self._group_sessions()
            self.generate_dataset()
            logger.info(f"PCAP processing completed: {len(self.packets)} packets processed")
            
            return self.packets
        except Exception as e:
            logger.error(f"Error processing PCAP: {str(e)}")
            raise
    
    def _group_sessions(self):
        """Group packets into TCP sessions"""
        for packet in self.packets:
            if packet.get('src_ip') and packet.get('dst_ip') and packet.get('protocol') == 6:  # TCP
                session_key = f"{packet['src_ip']}:{packet['src_port']}-{packet['dst_ip']}:{packet['dst_port']}"
                if session_key not in self.sessions:
                    self.sessions[session_key] = []
                self.sessions[session_key].append(packet)
    
    def generate_dataset(self) -> List[Dict[str, str]]:
        """Generate AIT-LDS compatible dataset with hierarchical classification"""
        dataset = []
        
        # 1. Attack Classification Questions
        dataset.extend(self._generate_attack_classification_questions())
        
        # 2. Hierarchical Analysis Questions
        dataset.extend(self._generate_hierarchical_analysis_questions())
        
        # 3. MITRE ATT&CK Mapping Questions
        dataset.extend(self._generate_mitre_mapping_questions())
        
        # 4. Threat Intelligence Questions
        dataset.extend(self._generate_threat_intelligence_questions())
        
        # 5. Security Response Questions
        dataset.extend(self._generate_security_response_questions())
        
        return dataset
    
    def _generate_attack_classification_questions(self) -> List[Dict[str, str]]:
        """Generate attack classification questions based on AIT-LDS categories"""
        questions = []
        
        # Reconnaissance Detection
        questions.append({
            "instruction": "Classify the network scanning activity observed in the packet capture. What type of reconnaissance attack is being performed?",
            "output": self._classify_reconnaissance_activity()
        })
        
        # Initial Intrusion Analysis
        questions.append({
            "instruction": "Analyze the packet data for signs of initial intrusion attempts. What attack vector is being used to gain initial access?",
            "output": self._analyze_initial_intrusion()
        })
        
        # Credential Access Detection
        questions.append({
            "instruction": "Examine the traffic patterns for credential access attempts. What methods are being used to obtain user credentials?",
            "output": self._detect_credential_access()
        })
        
        # Lateral Movement Analysis
        questions.append({
            "instruction": "Identify lateral movement activities in the network traffic. What techniques are being used for privilege escalation?",
            "output": self._analyze_lateral_movement()
        })
        
        # Exfiltration Detection
        questions.append({
            "instruction": "Detect data exfiltration activities in the packet capture. What covert channels are being used for data extraction?",
            "output": self._detect_exfiltration()
        })
        
        return questions
    
    def _generate_hierarchical_analysis_questions(self) -> List[Dict[str, str]]:
        """Generate questions for hierarchical attack analysis"""
        questions = []
        
        # Foothold Analysis
        questions.append({
            "instruction": "Analyze the initial foothold establishment phase. What combination of reconnaissance and intrusion techniques are observed?",
            "output": self._analyze_foothold_establishment()
        })
        
        # Escalation Chain Analysis
        questions.append({
            "instruction": "Examine the privilege escalation chain. What sequence of techniques leads to system-level access?",
            "output": self._analyze_escalation_chain()
        })
        
        # Persistence Mechanisms
        questions.append({
            "instruction": "Identify persistence mechanisms deployed by the attacker. What methods ensure continued access to the system?",
            "output": self._analyze_persistence_mechanisms()
        })
        
        return questions
    
    def _generate_mitre_mapping_questions(self) -> List[Dict[str, str]]:
        """Generate MITRE ATT&CK framework mapping questions"""
        questions = []
        
        # Tactic Identification
        questions.append({
            "instruction": "Map the observed attack activities to MITRE ATT&CK tactics. What are the primary tactics employed in this attack sequence?",
            "output": self._map_mitre_tactics()
        })
        
        # Technique Analysis
        questions.append({
            "instruction": "Identify specific MITRE ATT&CK techniques used in the attack. Provide technique IDs and descriptions for each observed activity.",
            "output": self._identify_mitre_techniques()
        })
        
        return questions
    
    def _generate_threat_intelligence_questions(self) -> List[Dict[str, str]]:
        """Generate threat intelligence and IOC questions"""
        questions = []
        
        # IOC Extraction
        questions.append({
            "instruction": "Extract indicators of compromise (IOCs) from the packet capture. What IP addresses, domains, and signatures indicate malicious activity?",
            "output": self._extract_iocs()
        })
        
        # Threat Actor Profiling
        questions.append({
            "instruction": "Based on the attack patterns and techniques observed, what can be inferred about the threat actor's capabilities and objectives?",
            "output": self._profile_threat_actor()
        })
        
        return questions
    
    def _generate_security_response_questions(self) -> List[Dict[str, str]]:
        """Generate security response and mitigation questions"""
        questions = []
        
        # Detection Rules
        questions.append({
            "instruction": "Generate detection rules for the observed attack patterns. What signatures should be implemented to detect similar attacks?",
            "output": self._generate_detection_rules()
        })
        
        # Mitigation Strategies
        questions.append({
            "instruction": "Recommend mitigation strategies for the identified attack vectors. What security controls should be implemented to prevent similar attacks?",
            "output": self._recommend_mitigations()
        })
        
        return questions
    
    def _classify_reconnaissance_activity(self) -> str:
        """Classify reconnaissance activities based on packet patterns"""
        if not self.packets:
            return json.dumps({
                "category": "NORMAL",
                "attack_type": "NONE",
                "mitre_technique": "N/A"
            })
        
        # Analyze scanning patterns
        port_scan_indicators = self._detect_port_scanning()
        dns_scan_indicators = self._detect_dns_scanning()
        service_scan_indicators = self._detect_service_scanning()
        
        if port_scan_indicators['detected']:
            return json.dumps({
                "category": "RECONNAISSANCE",
                "attack_type": "network_scan",
                "mitre_technique": "T1018",
                "severity": "MEDIUM",
                "recommendation": "Implement port scan detection and rate limiting"
            })
        elif dns_scan_indicators['detected']:
            return json.dumps({
                "category": "RECONNAISSANCE", 
                "attack_type": "dns_scan",
                "mitre_technique": "T1018",
                "severity": "MEDIUM",
                "recommendation": "Monitor DNS query patterns and implement DNS filtering"
            })
        elif service_scan_indicators['detected']:
            return json.dumps({
                "category": "RECONNAISSANCE",
                "attack_type": "service_scan", 
                "mitre_technique": "T1046",
                "severity": "MEDIUM",
                "recommendation": "Deploy service enumeration detection mechanisms"
            })
        else:
            return json.dumps({
                "category": "NORMAL",
                "attack_type": "NONE",
                "mitre_technique": "N/A"
            })
    
    def _analyze_initial_intrusion(self) -> str:
        """Analyze initial intrusion attempts"""
        webshell_indicators = self._detect_webshell_activity()
        exploit_indicators = self._detect_exploit_attempts()
        
        if webshell_indicators['detected']:
            return json.dumps({
                "category": "INITIAL_INTRUSION",
                "attack_type": "webshell_upload",
                "mitre_technique": "T1505.003",
                "severity": "HIGH",
                "recommendation": "Implement web application firewall and file upload restrictions"
            })
        elif exploit_indicators['detected']:
            return json.dumps({
                "category": "INITIAL_INTRUSION",
                "attack_type": "exploit_attempt",
                "mitre_technique": "T1190",
                "severity": "HIGH",
                "recommendation": "Apply security patches and implement intrusion prevention systems"
            })
        else:
            return json.dumps({
                "category": "NORMAL",
                "attack_type": "NONE",
                "mitre_technique": "N/A"
            })
    
    def _detect_credential_access(self) -> str:
        """Detect credential access attempts"""
        brute_force_indicators = self._detect_brute_force()
        credential_dump_indicators = self._detect_credential_dumping()
        
        if brute_force_indicators['detected']:
            return json.dumps({
                "category": "CREDENTIAL_ACCESS",
                "attack_type": "password_cracking",
                "mitre_technique": "T1110.002",
                "severity": "HIGH",
                "recommendation": "Implement account lockout policies and multi-factor authentication"
            })
        elif credential_dump_indicators['detected']:
            return json.dumps({
                "category": "CREDENTIAL_ACCESS",
                "attack_type": "database_dump",
                "mitre_technique": "T1003",
                "severity": "HIGH",
                "recommendation": "Enhance database security and implement credential monitoring"
            })
        else:
            return json.dumps({
                "category": "NORMAL",
                "attack_type": "NONE",
                "mitre_technique": "N/A"
            })
    
    def _analyze_lateral_movement(self) -> str:
        """Analyze lateral movement activities"""
        shell_indicators = self._detect_reverse_shell()
        escalation_indicators = self._detect_privilege_escalation()
        
        if shell_indicators['detected']:
            return json.dumps({
                "category": "LATERAL_MOVEMENT",
                "attack_type": "reverse_shell",
                "mitre_technique": "T1059",
                "severity": "HIGH",
                "recommendation": "Monitor outbound connections and implement endpoint detection"
            })
        elif escalation_indicators['detected']:
            return json.dumps({
                "category": "LATERAL_MOVEMENT",
                "attack_type": "privilege_escalation",
                "mitre_technique": "T1078",
                "severity": "HIGH",
                "recommendation": "Implement privileged access management and monitoring"
            })
        else:
            return json.dumps({
                "category": "NORMAL",
                "attack_type": "NONE",
                "mitre_technique": "N/A"
            })
    
    def _detect_exfiltration(self) -> str:
        """Detect data exfiltration activities"""
        dns_exfil_indicators = self._detect_dns_exfiltration()
        data_transfer_indicators = self._detect_large_data_transfers()
        
        if dns_exfil_indicators['detected']:
            return json.dumps({
                "category": "EXFILTRATION",
                "attack_type": "dns_exfiltration",
                "mitre_technique": "T1048.003",
                "severity": "HIGH",
                "recommendation": "Implement DNS monitoring and data loss prevention"
            })
        elif data_transfer_indicators['detected']:
            return json.dumps({
                "category": "EXFILTRATION",
                "attack_type": "data_extraction",
                "mitre_technique": "T1041",
                "severity": "HIGH",
                "recommendation": "Monitor network traffic for unusual data patterns"
            })
        else:
            return json.dumps({
                "category": "NORMAL",
                "attack_type": "NONE",
                "mitre_technique": "N/A"
            })
    
    def _analyze_foothold_establishment(self) -> str:
        """Analyze foothold establishment phase"""
        reconnaissance_score = self._calculate_reconnaissance_score()
        intrusion_score = self._calculate_intrusion_score()
        combined_score = (reconnaissance_score + intrusion_score) / 2
        
        if combined_score > 0.7:
            return json.dumps({
                "meta_label": "foothold",
                "phase": "ESTABLISHMENT",
                "components": {
                    "reconnaissance": reconnaissance_score,
                    "initial_intrusion": intrusion_score
                },
                "attack_chain": self._get_attack_chain_sequence(),
                "recommendation": "Immediate incident response required - foothold established"
            })
        else:
            return json.dumps({
                "meta_label": "normal",
                "phase": "NONE",
                "components": {
                    "reconnaissance": reconnaissance_score,
                    "initial_intrusion": intrusion_score
                },
                "attack_chain": [],
                "recommendation": "Continue monitoring for potential threats"
            })
    
    def _analyze_escalation_chain(self) -> str:
        """Analyze privilege escalation chain"""
        escalation_indicators = []
        
        # Check for credential access
        if self._has_credential_access():
            escalation_indicators.append("credential_access")
        
        # Check for privilege escalation
        if self._has_privilege_escalation():
            escalation_indicators.append("privilege_escalation")
        
        # Check for root access
        if self._has_root_access():
            escalation_indicators.append("root_access")
        
        if len(escalation_indicators) >= 2:
            return json.dumps({
                "meta_label": "escalation",
                "phase": "PRIVILEGE_ESCALATION",
                "escalation_path": escalation_indicators,
                "severity": "CRITICAL" if "root_access" in escalation_indicators else "HIGH",
                "recommendation": "Immediate containment and forensic analysis required"
            })
        else:
            return json.dumps({
                "meta_label": "normal",
                "phase": "NONE",
                "escalation_path": [],
                "severity": "LOW",
                "recommendation": "Continue monitoring for escalation attempts"
            })
    
    def _analyze_persistence_mechanisms(self) -> str:
        """Analyze persistence mechanisms"""
        backdoor_indicators = self._detect_backdoor_installation()
        shell_persistence = self._detect_shell_persistence()
        
        if backdoor_indicators['detected'] or shell_persistence['detected']:
            return json.dumps({
                "meta_label": "persistence",
                "phase": "PERSISTENCE_ESTABLISHMENT",
                "mechanisms": {
                    "backdoor": backdoor_indicators['detected'],
                    "reverse_shell": shell_persistence['detected']
                },
                "severity": "CRITICAL",
                "recommendation": "System isolation and complete forensic analysis required"
            })
        else:
            return json.dumps({
                "meta_label": "normal",
                "phase": "NONE",
                "mechanisms": {
                    "backdoor": False,
                    "reverse_shell": False
                },
                "severity": "LOW",
                "recommendation": "Continue monitoring for persistence indicators"
            })
    
    def _map_mitre_tactics(self) -> str:
        """Map observed activities to MITRE ATT&CK tactics"""
        tactics = []
        
        if self._has_reconnaissance():
            tactics.append("TA0043")  # Reconnaissance
        if self._has_initial_access():
            tactics.append("TA0001")  # Initial Access
        if self._has_credential_access():
            tactics.append("TA0006")  # Credential Access
        if self._has_privilege_escalation():
            tactics.append("TA0004")  # Privilege Escalation
        if self._has_lateral_movement():
            tactics.append("TA0008")  # Lateral Movement
        if self._has_exfiltration():
            tactics.append("TA0010")  # Exfiltration
        
        return json.dumps({
            "mitre_tactics": tactics,
            "attack_phases": len(tactics),
            "kill_chain_completion": len(tactics) / 6.0 * 100,
            "threat_level": "CRITICAL" if len(tactics) >= 4 else "HIGH" if len(tactics) >= 2 else "MEDIUM",
            "recommendation": f"Multi-phase attack detected spanning {len(tactics)} tactics"
        })
    
    def _identify_mitre_techniques(self) -> str:
        """Identify specific MITRE ATT&CK techniques"""
        techniques = []
        
        # Map detected activities to specific techniques
        for category, attacks in self.attack_categories.items():
            for attack_type, details in attacks.items():
                if self._is_attack_detected(attack_type):
                    techniques.append({
                        "technique_id": details['mitre_id'],
                        "technique_name": attack_type.replace('_', ' ').title(),
                        "category": category,
                        "severity": details['severity']
                    })
        
        return json.dumps({
            "detected_techniques": techniques,
            "technique_count": len(techniques),
            "severity_distribution": self._calculate_severity_distribution(techniques),
            "recommendation": "Implement specific countermeasures for identified techniques"
        })
    
    def _extract_iocs(self) -> str:
        """Extract indicators of compromise"""
        iocs = {
            "ip_addresses": self._extract_suspicious_ips(),
            "domains": self._extract_suspicious_domains(),
            "ports": self._extract_suspicious_ports(),
            "signatures": self._extract_attack_signatures()
        }
        
        return json.dumps({
            "indicators_of_compromise": iocs,
            "total_iocs": sum(len(v) for v in iocs.values()),
            "threat_score": self._calculate_threat_score(iocs),
            "recommendation": "Add IOCs to threat intelligence feeds and security controls"
        })
    
    def _profile_threat_actor(self) -> str:
        """Profile threat actor based on observed techniques"""
        capabilities = self._assess_attacker_capabilities()
        objectives = self._infer_attacker_objectives()
        sophistication = self._calculate_sophistication_level()
        
        return json.dumps({
            "threat_actor_profile": {
                "capabilities": capabilities,
                "objectives": objectives,
                "sophistication_level": sophistication,
                "attack_pattern": self._identify_attack_pattern()
            },
            "threat_assessment": "ADVANCED" if sophistication > 0.7 else "INTERMEDIATE" if sophistication > 0.4 else "BASIC",
            "recommendation": "Implement enhanced monitoring and response procedures"
        })
    
    def _generate_detection_rules(self) -> str:
        """Generate detection rules for observed attack patterns"""
        rules = []
        
        # Generate Snort/Suricata rules
        if self._has_port_scanning():
            rules.append({
                "type": "snort",
                "rule": 'alert tcp any any -> $HOME_NET any (msg:"Potential Port Scan"; flow:stateless; flags:S; threshold:type both,track by_src,count 20,seconds 60; sid:1000001;)',
                "description": "Detects potential port scanning activity"
            })
        
        if self._detect_dns_exfiltration():
            rules.append({
                "type": "snort",
                "rule": 'alert udp any any -> any 53 (msg:"Potential DNS Exfiltration"; content:"|00 01 00 00 00 00 00 00|"; offset:4; depth:8; threshold:type both,track by_src,count 50,seconds 300; sid:1000002;)',
                "description": "Detects potential DNS exfiltration"
            })
        
        return json.dumps({
            "detection_rules": rules,
            "rule_count": len(rules),
            "coverage": self._calculate_rule_coverage(),
            "recommendation": "Deploy rules in monitoring mode first, then enable blocking"
        })
    
    def _recommend_mitigations(self) -> str:
        """Recommend mitigation strategies"""
        mitigations = []
        
        if self._has_reconnaissance():
            mitigations.append({
                "category": "RECONNAISSANCE",
                "mitigation": "Implement network segmentation and port scan detection",
                "priority": "HIGH"
            })
        
        if self._has_initial_access():
            mitigations.append({
                "category": "INITIAL_ACCESS",
                "mitigation": "Deploy web application firewall and patch management",
                "priority": "CRITICAL"
            })
        
        if self._has_credential_access():
            mitigations.append({
                "category": "CREDENTIAL_ACCESS",
                "mitigation": "Implement multi-factor authentication and credential monitoring",
                "priority": "HIGH"
            })
        
        if self._has_exfiltration():
            mitigations.append({
                "category": "EXFILTRATION",
                "mitigation": "Deploy data loss prevention and network monitoring",
                "priority": "CRITICAL"
            })
        
        return json.dumps({
            "mitigation_strategies": mitigations,
            "total_mitigations": len(mitigations),
            "implementation_priority": "IMMEDIATE" if any(m['priority'] == 'CRITICAL' for m in mitigations) else "HIGH",
            "recommendation": "Implement critical mitigations within 24 hours"
        })
    
    # Helper methods for detection logic
    def _detect_port_scanning(self) -> Dict:
        """Detect port scanning patterns"""
        if not self.packets:
            return {'detected': False}
        
        # Track connections per source IP
        src_ports = defaultdict(set)
        for packet in self.packets:
            if packet.get('src_ip') and packet.get('dst_port'):
                src_ports[packet['src_ip']].add(packet['dst_port'])
        
        # Check for scanning patterns
        for src_ip, ports in src_ports.items():
            if len(ports) > 10:  # Threshold for port scanning
                return {
                    'detected': True,
                }
        
        return {'detected': False}
    
    def _detect_dns_scanning(self) -> Dict:
        """Detect DNS scanning patterns"""
        dns_queries = defaultdict(int)
        for packet in self.packets:
            if packet.get('dst_port') == 53:  # DNS port
                dns_queries[packet.get('src_ip')] += 1
        
        for src_ip, count in dns_queries.items():
            if count > 50:  # Threshold for DNS scanning
                return {
                    'detected': True,
                }

        return {'detected': False}
    
    def _detect_service_scanning(self) -> Dict:
        """Detect service scanning patterns"""
        service_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
        service_scans = defaultdict(set)
        
        for packet in self.packets:
            if packet.get('dst_port') in service_ports:
                service_scans[packet.get('src_ip')].add(packet.get('dst_port'))
        
        for src_ip, ports in service_scans.items():
            if len(ports) >= 5:  # Threshold for service scanning
                return {
                    'detected': True,
                }
        
        return {'detected': False}
    
    def _detect_webshell_activity(self) -> Dict:
        """Detect webshell activity patterns"""
        web_requests = []
        for packet in self.packets:
            if packet.get('dst_port') in [80, 443, 8080]:  # Web ports
                if packet.get('payload'):
                    payload = packet['payload'].lower()
                    if any(keyword in payload for keyword in ['cmd=', 'eval(', 'system(', 'exec(']):
                        web_requests.append(packet)
        
        if len(web_requests) > 3:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_exploit_attempts(self) -> Dict:
        """Detect exploit attempts"""
        exploit_patterns = [
            '../../../', 'union select', 'script>', 'alert(', 
            'javascript:', 'eval(', 'system(', 'exec(', 'passthru(', 
            'shell_exec(', 'popen(', 'proc_open(', 'pcntl_exec('
        ]
        exploit_count = 0
        
        for packet in self.packets:
            if packet.get('payload'):
                payload = packet['payload'].lower()
                if any(pattern in payload for pattern in exploit_patterns):
                    exploit_count += 1
        
        if exploit_count > 2:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_brute_force(self) -> Dict:
        """Detect brute force attempts"""
        auth_attempts = defaultdict(int)
        for packet in self.packets:
            if packet.get('dst_port') in [21, 22, 23, 25, 110, 143, 443, 993, 995]:
                auth_attempts[packet.get('src_ip')] += 1
        
        for src_ip, count in auth_attempts.items():
            if count > 20:  # Threshold for brute force
                return {
                    'detected': True,
                }
        
        return {'detected': False}
    
    def _detect_credential_dumping(self) -> Dict:
        """Detect credential dumping activities"""
        # Look for large data transfers from authentication services
        large_transfers = []
        for packet in self.packets:
            if packet.get('length', 0) > 1000 and packet.get('src_port') in [389, 636, 1433, 1521, 3306, 5432]:
                large_transfers.append(packet)
        
        if len(large_transfers) > 5:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_reverse_shell(self) -> Dict:
        """Detect reverse shell connections"""
        outbound_connections = []
        for packet in self.packets:
            # Look for outbound connections on unusual ports
            if packet.get('dst_port') and packet.get('dst_port') > 1024 and packet.get('dst_port') not in [8080, 8443]:
                outbound_connections.append(packet)
        
        if len(outbound_connections) > 10:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_privilege_escalation(self) -> Dict:
        """Detect privilege escalation attempts"""
        # Look for privilege escalation indicators in network traffic
        privilege_indicators = 0
        for packet in self.packets:
            if packet.get('payload'):
                payload = packet['payload'].lower()
                if any(keyword in payload for keyword in ['sudo', 'su -', 'passwd', 'shadow', '/etc/passwd']):
                    privilege_indicators += 1
        
        if privilege_indicators > 3:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_dns_exfiltration(self) -> Dict:
        """Detect DNS exfiltration patterns"""
        dns_queries = []
        for packet in self.packets:
            if packet.get('dst_port') == 53 and packet.get('length', 0) > 100:  # Large DNS queries
                dns_queries.append(packet)
        
        if len(dns_queries) > 20:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_large_data_transfers(self) -> Dict:
        """Detect large data transfers"""
        large_transfers = []
        for packet in self.packets:
            if packet.get('length', 0) > 1500:  # Large packets
                large_transfers.append(packet)
        
        if len(large_transfers) > 50:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_backdoor_installation(self) -> Dict:
        """Detect backdoor installation"""
        backdoor_ports = [4444, 5555, 6666, 7777, 31337, 12345]
        backdoor_activity = []
        
        for packet in self.packets:
            if packet.get('dst_port') in backdoor_ports or packet.get('src_port') in backdoor_ports:
                backdoor_activity.append(packet)
        
        if len(backdoor_activity) > 2:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    def _detect_shell_persistence(self) -> Dict:
        """Detect shell persistence mechanisms"""
        # Look for persistent connections
        persistent_connections = defaultdict(int)
        for session_key, packets in self.sessions.items():
            if len(packets) > 100:  # Long-running sessions
                persistent_connections[session_key] = len(packets)
        
        if persistent_connections:
            return {
                'detected': True,
            }
        
        return {'detected': False}
    
    # Additional helper methods for scoring and analysis
    def _calculate_reconnaissance_score(self) -> float:
        """Calculate reconnaissance activity score"""
        score = 0.0
        if self._detect_port_scanning()['detected']:
            score += 0.3
        if self._detect_dns_scanning()['detected']:
            score += 0.3
        if self._detect_service_scanning()['detected']:
            score += 0.4
        return min(score, 1.0)
    
    def _calculate_intrusion_score(self) -> float:
        """Calculate intrusion activity score"""
        score = 0.0
        if self._detect_webshell_activity()['detected']:
            score += 0.5
        if self._detect_exploit_attempts()['detected']:
            score += 0.5
        return min(score, 1.0)
    
    def _get_attack_chain_sequence(self) -> List[str]:
        """Get the sequence of attack steps"""
        sequence = []
        if self._detect_port_scanning()['detected']:
            sequence.append("port_scan")
        if self._detect_dns_scanning()['detected']:
            sequence.append("dns_scan")
        if self._detect_service_scanning()['detected']:
            sequence.append("service_scan")
        if self._detect_webshell_activity()['detected']:
            sequence.append("webshell_upload")
        if self._detect_exploit_attempts()['detected']:
            sequence.append("exploit_attempt")
        return sequence
    
    def _has_reconnaissance(self) -> bool:
        """Check if reconnaissance activity is present"""
        return any([
            self._detect_port_scanning()['detected'],
            self._detect_dns_scanning()['detected'],
            self._detect_service_scanning()['detected']
        ])
    
    def _has_initial_access(self) -> bool:
        """Check if initial access activity is present"""
        return any([
            self._detect_webshell_activity()['detected'],
            self._detect_exploit_attempts()['detected']
        ])
    
    def _has_credential_access(self) -> bool:
        """Check if credential access activity is present"""
        return any([
            self._detect_brute_force()['detected'],
            self._detect_credential_dumping()['detected']
        ])
    
    def _has_privilege_escalation(self) -> bool:
        """Check if privilege escalation activity is present"""
        return self._detect_privilege_escalation()['detected']
    
    def _has_lateral_movement(self) -> bool:
        """Check if lateral movement activity is present"""
        return self._detect_reverse_shell()['detected']
    
    def _has_exfiltration(self) -> bool:
        """Check if exfiltration activity is present"""
        return any([
            self._detect_dns_exfiltration()['detected'],
            self._detect_large_data_transfers()['detected']
        ])
    
    def _has_root_access(self) -> bool:
        """Check if root access indicators are present"""
        return self._detect_privilege_escalation()['detected']
    
    def _has_port_scanning(self) -> bool:
        """Check if port scanning is present"""
        return self._detect_port_scanning()['detected']
    
    def _is_attack_detected(self, attack_type: str) -> bool:
        """Check if specific attack type is detected"""
        detection_methods = {
            'network_scan': self._detect_port_scanning,
            'dns_scan': self._detect_dns_scanning,
            'service_scan': self._detect_service_scanning,
            'webshell_upload': self._detect_webshell_activity,
            'password_cracking': self._detect_brute_force,
            'reverse_shell': self._detect_reverse_shell,
            'privilege_escalation': self._detect_privilege_escalation,
            'dns_exfiltration': self._detect_dns_exfiltration
        }
        
        if attack_type in detection_methods:
            return detection_methods[attack_type]()['detected']
        return False
    
    def _calculate_severity_distribution(self, techniques: List[Dict]) -> Dict:
        """Calculate severity distribution of techniques"""
        severity_count = defaultdict(int)
        for technique in techniques:
            severity_count[technique['severity']] += 1
        return dict(severity_count)
    
    def _extract_suspicious_ips(self) -> List[str]:
        """Extract suspicious IP addresses"""
        suspicious_ips = set()
        for packet in self.packets:
            if self._is_suspicious_ip_behavior(packet.get('src_ip')):
                suspicious_ips.add(packet.get('src_ip'))
        return list(suspicious_ips)
    
    def _extract_suspicious_domains(self) -> List[str]:
        """Extract suspicious domains"""
        # This would be implemented based on DNS query analysis
        return []
    
    def _extract_suspicious_ports(self) -> List[int]:
        """Extract suspicious ports"""
        suspicious_ports = set()
        for packet in self.packets:
            if packet.get('dst_port') and packet.get('dst_port') > 1024:
                suspicious_ports.add(packet.get('dst_port'))
        return list(suspicious_ports)
    
    def _extract_attack_signatures(self) -> List[str]:
        """Extract attack signatures"""
        signatures = []
        for packet in self.packets:
            if packet.get('payload'):
                payload = packet['payload'].lower()
                if any(sig in payload for sig in ['cmd=', 'eval(', 'union select']):
                    signatures.append(payload[:100])  # First 100 chars
        return signatures
    
    def _calculate_threat_score(self, iocs: Dict) -> float:
        """Calculate overall threat score"""
        total_iocs = sum(len(v) for v in iocs.values())
        return min(total_iocs / 50.0, 1.0)
    
    def _assess_attacker_capabilities(self) -> List[str]:
        """Assess attacker capabilities"""
        capabilities = []
        if self._has_reconnaissance():
            capabilities.append("Network reconnaissance")
        if self._has_initial_access():
            capabilities.append("Vulnerability exploitation")
        if self._has_credential_access():
            capabilities.append("Credential harvesting")
        if self._has_lateral_movement():
            capabilities.append("Lateral movement")
        if self._has_exfiltration():
            capabilities.append("Data exfiltration")
        return capabilities
    
    def _infer_attacker_objectives(self) -> List[str]:
        """Infer attacker objectives"""
        objectives = []
        if self._has_credential_access():
            objectives.append("Credential theft")
        if self._has_exfiltration():
            objectives.append("Data theft")
        if self._has_lateral_movement():
            objectives.append("Network compromise")
        return objectives
    
    def _calculate_sophistication_level(self) -> float:
        """Calculate attacker sophistication level"""
        techniques_used = sum([
            self._has_reconnaissance(),
            self._has_initial_access(),
            self._has_credential_access(),
            self._has_privilege_escalation(),
            self._has_lateral_movement(),
            self._has_exfiltration()
        ])
        return techniques_used / 6.0
    
    def _identify_attack_pattern(self) -> str:
        """Identify attack pattern"""
        if self._has_exfiltration():
            return "Data theft campaign"
        elif self._has_lateral_movement():
            return "Network compromise"
        elif self._has_initial_access():
            return "System intrusion"
        elif self._has_reconnaissance():
            return "Reconnaissance probe"
        else:
            return "Unknown pattern"
    
    def _calculate_rule_coverage(self) -> float:
        """Calculate detection rule coverage"""
        # This would calculate how much of the attack surface is covered
        return 0.85  # Placeholder
    
    def _is_suspicious_ip_behavior(self, ip: str) -> bool:
        """Check if IP exhibits suspicious behavior"""
        if not ip:
            return False
        
        # Count activities from this IP
        activity_count = sum(1 for packet in self.packets if packet.get('src_ip') == ip)
        return activity_count > 50  # Threshold for suspicious activity
    
if __name__ == "__main__":
    # Initialize the classifier
    classifier = AIT_LDS_PacketClassifier("../datasets/packet/002482_facebook_audio4.pcapng")
    
    # Process the PCAP file
    packets = classifier.process_pcap()
    
    # Generate the dataset
    dataset = classifier.generate_dataset()
    
    # Save the dataset
    with open("ait_lds_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} question-answer pairs")
    print(f"Processed {len(packets)} packets")
    
    # Example of a generated question-answer pair
    for item in dataset[:3]:
        print(f"\nInstruction: {item['instruction']}")
        print(f"Output: {item['output']}")
        print("-" * 80)