from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
from collections import defaultdict

class SyslogAnalyzer:
    def __init__(self):
        self.logs = []
        self.patterns = {
            'timestamp': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
            'hostname': r'(\S+)',
            'process': r'(\w+)(?:\[\d+\])?',
            'message': r':\s*(.*)'
        }
    
    def analyze_syslog(self, file_path: str) -> Dict[str, Any]:
        """Syslog 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.logs = []
                for line in f:
                    log_entry = self._parse_log_line(line.strip())
                    if log_entry:
                        self.logs.append(log_entry)
            
            # 분석 결과 생성
            return {
                "basic_stats": self._get_basic_stats(),
                "process_dist": self._get_process_distribution(),
                "severity_dist": self._get_severity_distribution(),
                "security_analysis": self._analyze_security(),
                "visualizations": self._generate_visualizations()
            }
        except Exception as e:
            raise Exception(f"Error analyzing syslog file: {str(e)}")
    
    def _parse_log_line(self, line: str) -> Dict[str, Any]:
        """로그 라인 파싱"""
        try:
            # 기본 syslog 형식 파싱
            timestamp_match = re.search(self.patterns['timestamp'], line)
            hostname_match = re.search(self.patterns['hostname'], line[timestamp_match.end():] if timestamp_match else line)
            process_match = re.search(self.patterns['process'], line[hostname_match.end():] if hostname_match else line)
            message_match = re.search(self.patterns['message'], line[process_match.end():] if process_match else line)
            
            if not all([timestamp_match, hostname_match, process_match, message_match]):
                return None
            
            # 심각도 레벨 추출
            severity = self._extract_severity(message_match.group(1))
            
            return {
                "timestamp": datetime.strptime(timestamp_match.group(1), "%b %d %H:%M:%S"),
                "hostname": hostname_match.group(1),
                "process": process_match.group(1),
                "message": message_match.group(1),
                "severity": severity
            }
        except Exception:
            return None
    
    def _extract_severity(self, message: str) -> str:
        """메시지에서 심각도 레벨 추출"""
        severity_keywords = {
            'emerg': 'emergency',
            'alert': 'alert',
            'crit': 'critical',
            'err': 'error',
            'warning': 'warning',
            'notice': 'notice',
            'info': 'info',
            'debug': 'debug'
        }
        
        for keyword, severity in severity_keywords.items():
            if keyword in message.lower():
                return severity
        return 'info'
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        """기본 통계 정보"""
        return {
            "total_logs": len(self.logs),
            "start_time": min(log["timestamp"] for log in self.logs),
            "end_time": max(log["timestamp"] for log in self.logs),
            "unique_hosts": len(set(log["hostname"] for log in self.logs)),
            "unique_processes": len(set(log["process"] for log in self.logs))
        }
    
    def _get_process_distribution(self) -> Dict[str, Any]:
        """프로세스 분포 분석"""
        process_counts = defaultdict(int)
        for log in self.logs:
            process_counts[log["process"]] += 1
        
        return {
            "distribution": dict(process_counts),
            "most_common": max(process_counts.items(), key=lambda x: x[1])[0] if process_counts else None
        }
    
    def _get_severity_distribution(self) -> Dict[str, Any]:
        """심각도 분포 분석"""
        severity_counts = defaultdict(int)
        for log in self.logs:
            severity_counts[log["severity"]] += 1
        
        return {
            "distribution": dict(severity_counts),
            "most_common": max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None
        }
    
    def _analyze_security(self) -> Dict[str, Any]:
        """보안 분석"""
        suspicious_patterns = []
        security_keywords = [
            'failed', 'error', 'denied', 'unauthorized',
            'attack', 'intrusion', 'malware', 'virus',
            'breach', 'compromise', 'exploit'
        ]
        
        for log in self.logs:
            message = log["message"].lower()
            for keyword in security_keywords:
                if keyword in message:
                    suspicious_patterns.append(f"Security keyword detected: {keyword} in {log['process']}")
        
        return {
            "suspicious_patterns": suspicious_patterns,
            "security_level": "high" if suspicious_patterns else "normal"
        }
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """시각화 생성"""
        df = pd.DataFrame(self.logs)
        
        # 시간대별 로그 수
        plt.figure(figsize=(12, 6))
        hourly_counts = df.groupby(df["timestamp"].dt.hour).size()
        sns.lineplot(x=hourly_counts.index, y=hourly_counts.values)
        plt.title("Log Count Over Time")
        plt.xlabel("Hour")
        plt.ylabel("Log Count")
        timeline_plot = self._plot_to_base64()
        plt.close()
        
        # 심각도 분포
        plt.figure(figsize=(10, 6))
        severity_counts = df["severity"].value_counts()
        sns.barplot(x=severity_counts.index, y=severity_counts.values)
        plt.title("Severity Distribution")
        plt.xlabel("Severity")
        plt.ylabel("Count")
        severity_plot = self._plot_to_base64()
        plt.close()
        
        return {
            "timeline": timeline_plot,
            "severity_distribution": severity_plot
        }
    
    def _plot_to_base64(self) -> str:
        """플롯을 base64로 변환"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}" 