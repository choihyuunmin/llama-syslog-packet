import re
from datetime import datetime
from typing import Dict, Any, List, Optional

class SyslogAnalyzer:
    def __init__(self):
        # More robust regex to handle different syslog formats
        self.log_pattern = re.compile(
            r'^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+' 
            r'(?P<hostname>\S+)\s+' 
            r'(?P<program>\S+?)(?:\[(?P<pid>\d+)\])?:\s+' 
            r'(?P<message>.+)$'
        )

    def analyze_syslog(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyzes a syslog file and returns a list of log information.
        """
        logs_info = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                info = self._parse_line(line.strip())
                if info:
                    logs_info.append(info)
        return logs_info

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parses a single log line."""
        match = self.log_pattern.match(line)
        if not match:
            return None

        log_data = match.groupdict()
        message = log_data.get('message', '')
        
        # Attempt to parse timestamp
        try:
            ts_str = f"{datetime.now().year} {log_data['timestamp']}"
            timestamp = datetime.strptime(ts_str, '%Y %b %d %H:%M:%S')
        except ValueError:
            timestamp = datetime.now()

        return {
            'timestamp': timestamp.isoformat(),
            'type': 'syslog',
            'log_level': self._get_log_level(message),
            'component': log_data.get('program', 'N/A'),
            'message': message.strip()
        }

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