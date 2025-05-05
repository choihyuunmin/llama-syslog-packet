import json
from pathlib import Path
from typing import Dict, List
import random
from datetime import datetime, timedelta

class BenchmarkGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_pcap_benchmark(self, num_questions: int = 100) -> None:
        """Generate benchmark questions for PCAP analysis."""
        questions = []
        
        # Protocol analysis questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Analyze the protocol distribution in this network traffic. What are the primary protocols used and their percentages?",
                'answer': f"Protocol distribution: TCP: {random.randint(40, 70)}%, UDP: {random.randint(20, 40)}%, ICMP: {random.randint(5, 15)}%"
            })
            
        # Traffic pattern questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Analyze the traffic pattern in this session. What is the average packet size and does it indicate any anomalies?",
                'answer': f"Average packet size: {random.randint(500, 1500)} bytes. {'Normal' if random.random() > 0.3 else 'Anomalous - unusual packet size distribution'}"
            })
            
        # Anomaly detection questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Detect any anomalies in this network traffic. What type of anomalies are present and what is their severity?",
                'answer': f"{'No anomalies detected' if random.random() > 0.4 else 'Anomaly detected: ' + random.choice(['SYN flood', 'Port scan', 'DDoS', 'Malware communication'])}"
            })
            
        # Correlation analysis questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Analyze the correlation between different sessions. Are there any patterns that suggest coordinated activity?",
                'answer': f"{'No coordinated activity detected' if random.random() > 0.5 else 'Potential coordinated activity detected across ' + str(random.randint(2, 5)) + ' sessions'}"
            })
            
        # Save to file
        with open(self.output_dir / 'pcap_benchmark.json', 'w') as f:
            json.dump(questions, f, indent=2)
            
    def generate_syslog_benchmark(self, num_questions: int = 100) -> None:
        """Generate benchmark questions for syslog analysis."""
        questions = []
        
        # Log pattern analysis questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Analyze the severity distribution in these logs. What patterns can be observed?",
                'answer': f"Severity distribution: Critical: {random.randint(5, 15)}%, Error: {random.randint(20, 40)}%, Warning: {random.randint(30, 50)}%, Info: {random.randint(10, 30)}%"
            })
            
        # Process correlation questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Analyze the process correlation in these logs. Are there any suspicious process interactions?",
                'answer': f"Process correlation: {'Normal' if random.random() > 0.4 else 'Suspicious - multiple process interactions detected'} with {random.randint(3, 8)} distinct processes"
            })
            
        # Anomaly detection questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Detect any anomalies in these system logs. What type of anomalies are present and what is their impact?",
                'answer': f"{'No anomalies detected' if random.random() > 0.4 else 'Anomaly detected: ' + random.choice(['Log burst', 'Process chain', 'Critical errors', 'Security events'])}"
            })
            
        # System health questions
        for _ in range(num_questions // 4):
            questions.append({
                'question': "Analyze the overall system health based on these logs. What is the current state and are there any concerns?",
                'answer': f"System health: {'Stable' if random.random() > 0.3 else 'Degrading'} with {random.randint(0, 5)} critical issues and {random.randint(0, 10)} warnings"
            })
            
        # Save to file
        with open(self.output_dir / 'syslog_benchmark.json', 'w') as f:
            json.dump(questions, f, indent=2)
            
    def generate_combined_benchmark(self, num_questions: int = 200) -> None:
        """Generate combined benchmark questions for both PCAP and syslog analysis."""
        questions = []
        
        # Cross-domain correlation questions
        for _ in range(num_questions // 2):
            questions.append({
                'question': "Correlate the network traffic patterns with system log events. Are there any relationships that suggest security incidents?",
                'answer': f"{'No significant correlations detected' if random.random() > 0.4 else 'Correlation detected: Network anomaly at ' + str(random.randint(1, 24)) + ':00 corresponds to system log events'}"
            })
            
        # Root cause analysis questions
        for _ in range(num_questions // 2):
            questions.append({
                'question': "Perform a root cause analysis combining network and system log data. What is the likely cause of the observed anomalies?",
                'answer': f"Root cause: {'No root cause identified' if random.random() > 0.4 else random.choice(['External attack', 'System misconfiguration', 'Resource exhaustion', 'Malware activity'])}"
            })
            
        # Save to file
        with open(self.output_dir / 'combined_benchmark.json', 'w') as f:
            json.dump(questions, f, indent=2)
            
    def generate_all_benchmarks(self):
        """Generate all benchmark datasets."""
        self.generate_pcap_benchmark()
        self.generate_syslog_benchmark()
        self.generate_combined_benchmark() 