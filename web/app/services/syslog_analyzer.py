import re
from datetime import datetime
from typing import Dict, Any, List, Optional

class SyslogAnalyzer:
    def __init__(self):
        self.syslog_pattern = re.compile(
            r'^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+'
            r'(?P<hostname>\S+)\s+'
            r'(?P<program>\S+?)(?:\[(?P<pid>\d+)\])?:\s+'
            r'(?P<message>.+)'
        )

        self.access_log_pattern = re.compile(
            r'^(?P<ip>\S+) \S+ \S+ '
            r'\[(?P<timestamp>[^\]]+)] '
            r'"(?P<request>[^"]+)" '
            r'(?P<status>\d{3}) '
            r'(?P<size>\d+|-) '
            r'"(?P<referrer>[^"]*)" '
            r'"(?P<user_agent>[^"]+)"'
        )

    def analyze_syslog(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyzes a log file (supporting syslog and access_log formats) and returns a list of log information.
        """
        logs_info = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                info = self._parse_line(line.strip())
                if info:
                    logs_info.append(info)
        return logs_info

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parses a single log line, trying different formats."""
        # Try access log format first
        access_match = self.access_log_pattern.match(line)
        if access_match:
            log_data = access_match.groupdict()
            try:
                timestamp = datetime.strptime(log_data['timestamp'], '%d/%b/%Y:%H:%M:%S %z')
            except ValueError:
                timestamp = datetime.now()
            
            size_str = log_data.get('size', '0')
            size = int(size_str) if size_str.isdigit() else 0

            return {
                'timestamp': timestamp.isoformat(),
                'type': 'log',
                'log_type': 'access_log',
                'ip': log_data.get('ip'),
                'request': log_data.get('request'),
                'status_code': int(log_data.get('status', 0)),
                'size': size,
                'referrer': log_data.get('referrer'),
                'user_agent': log_data.get('user_agent'),
                'message': line
            }

        # Try syslog format
        syslog_match = self.syslog_pattern.match(line)
        if syslog_match:
            log_data = syslog_match.groupdict()
            message = log_data.get('message', '')
            
            try:
                ts_str = f"{datetime.now().year} {log_data['timestamp']}"
                timestamp = datetime.strptime(ts_str, '%Y %b %d %H:%M:%S')
            except ValueError:
                timestamp = datetime.now()

            return {
                'timestamp': timestamp.isoformat(),
                'type': 'log',
                'log_type': 'syslog',
                'log_level': self._get_log_level(message),
                'component': log_data.get('program', 'N/A'),
                'message': message.strip()
            }
            
        return None

    def _get_log_level(self, message: str) -> str:
        """Extracts log level from the message."""
        message_lower = message.lower()
        if 'error' in message_lower or 'failed' in message_lower:
            return 'Error'
        if 'warn' in message_lower or 'warning' in message_lower:
            return 'Warning'
        if 'info' in message_lower:
            return 'Info'
        if 'debug' in message_lower:
            return 'Debug'
        return 'Unknown'