import json
import logging
from pathlib import Path
import random

class BenchmarkGenerator:
    def __init__(self, output_dir):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_test_cases(self):
        """Generate test cases for model evaluation"""
        test_cases = [
            {
                "id": "test_1",
                "input": "Analyze the following log: [2024-03-20 10:15:23] WARNING: Failed login attempt from 192.168.1.100",
                "expected_output": "Log Analysis: Failed login attempt detected from IP 192.168.1.100. This could indicate a potential security threat. Consider monitoring or blocking access from this IP address."
            },
            {
                "id": "test_2",
                "input": "Explain the following firewall rule: iptables -A INPUT -p tcp --dport 22 -j DROP",
                "expected_output": "This firewall rule blocks all TCP connections to port 22 (SSH). It is a security rule that prevents any access to the SSH service."
            },
            {
                "id": "test_3",
                "input": "Analyze the following network packet: TCP SYN packet, Source: 10.0.0.1:12345, Destination: 192.168.1.1:80",
                "expected_output": "This packet is a TCP SYN packet attempting to establish a connection from 10.0.0.1 port 12345 to 192.168.1.1 port 80 (HTTP). This appears to be a normal web service access attempt."
            },
            {
                "id": "test_4",
                "input": "Interpret the following security alert: [CRITICAL] Multiple failed authentication attempts detected from 172.16.0.50",
                "expected_output": "Critical Security Alert: Multiple failed authentication attempts detected from IP 172.16.0.50. This could indicate a brute force attack attempt. Immediate action is required."
            },
            {
                "id": "test_5",
                "input": "Analyze the following system log: [ERROR] Disk usage exceeded 90% on /dev/sda1",
                "expected_output": "System Log Analysis: Disk usage on /dev/sda1 has exceeded 90%. This could lead to system issues due to insufficient disk space. Consider cleaning up unnecessary files or expanding disk capacity."
            }
        ]
        return test_cases

    def generate_all_benchmarks(self):
        """Generate benchmark dataset"""
        test_cases = self.generate_test_cases()
        
        # Save to JSON file
        with open(self.output_dir / 'benchmark.json', 'w') as f:
            json.dump(test_cases, f, indent=2)
        
        self.logger.info(f"Benchmark dataset has been generated in {self.output_dir}") 