import re
from typing import Dict, List, Any
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

class SyslogProcessor:
    def __init__(self, syslog_file: str):
        """Initialize SyslogProcessor.
        
        Args:
            syslog_file (str): Path to the syslog file
        """
        self.syslog_file = Path(syslog_file)
        self.logs: List[Dict] = []
        self.df: pd.DataFrame = pd.DataFrame()
        
    def process_logs(self) -> None:
        """Process syslog file and convert to structured format."""
        try:
            parsed_logs = []
            
            # encoding problems
            encodings = ['utf-8', 'ascii', 'latin-1', 'cp949']
            
            for encoding in encodings:
                try:
                    with open(self.syslog_file, 'r', encoding=encoding) as f:
                        for line in f:
                            log_info = self._parse_log_line(line.strip())
                            if log_info:
                                parsed_logs.append(log_info)
                    logger.info(f"Successfully processed syslog file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode with {encoding} encoding, trying next...")
                    continue
                except Exception as e:
                    logger.error(f"Error processing syslog with {encoding} encoding: {e}")
                    raise
            
            self.logs = parsed_logs
            self.df = pd.DataFrame(parsed_logs)
            if not self.df.empty:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            logger.info(f"Processed {len(self.logs)} log entries")
        except Exception as e:
            logger.error(f"Error processing syslog: {e}")
            raise
    
    def _parse_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a single syslog line into structured format.
        
        Args:
            line (str): Raw syslog line
            
        Returns:
            Dict[str, Any]: Structured log entry
        """
        try:
            # Common syslog format pattern
            pattern = r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(.*)'
            match = re.match(pattern, line)
            
            if not match:
                return None
                
            timestamp, host, program, message = match.groups()
            
            # Add current year as syslog doesn't include year
            current_year = datetime.now().year
            timestamp = f"{timestamp} {current_year}"
            dt = datetime.strptime(timestamp, '%b %d %H:%M:%S %Y')
            
            return {
                'timestamp': dt.isoformat(),
                'host': host,
                'program': program,
                'message': message,
                'severity': self._get_severity(message),
                'category': self._get_category(message)
            }
        except Exception as e:
            logger.error(f"Error parsing log line: {e}")
            return None
    
    def _get_severity(self, message: str) -> str:
        """Determine log severity based on message content."""
        severity_patterns = {
            'critical': r'crit|fatal|emerg|panic',
            'error': r'error|err|fail',
            'warning': r'warn|alert',
            'info': r'info|notice',
            'debug': r'debug|dbg'
        }
        
        for severity, pattern in severity_patterns.items():
            if re.search(pattern, message.lower()):
                return severity
        return 'info'
    
    def _get_category(self, message: str) -> str:
        """Categorize log message based on content."""
        categories = {
            'auth': r'auth|login|user|sudo|su\b|ssh|password|credential',
            'security': r'secur|firewall|iptables|attack|hack|malicious|threat|violation|deny',
            'system': r'system|kernel|daemon|service|process|cpu|memory|disk',
            'network': r'network|interface|connection|tcp|udp|ip|port|packet',
            'application': r'app|web|api|database|sql|cache|queue'
        }
        
        for category, pattern in categories.items():
            if re.search(pattern, message.lower()):
                return category
        return 'other'
    
    def generate_dataset(self) -> List[Dict[str, str]]:
        """Generate Q&A dataset from processed logs.
        
        Returns:
            List[Dict[str, str]]: List of question-answer pairs
        """
        if self.df.empty:
            logger.warning("No logs processed yet. Call process_logs() first.")
            return []
        
        dataset = []
        
        # Template for dataset generation
        templates = {
            'basic_analysis': [
                {
                    'question': "What is the distribution of log entries by severity level in the system logs?",
                    'answer': lambda: self._get_severity_distribution()
                },
                {
                    'question': "What are the most common types of security events in the logs?",
                    'answer': lambda: self._get_security_events_summary()
                },
                {
                    'question': "How many authentication-related events are there in the logs?",
                    'answer': lambda: self._get_auth_events_count()
                }
            ],
            'advanced_analysis': [
                {
                    'question': "Analyze the authentication failure patterns in the logs. Are there any signs of brute force attempts?",
                    'answer': lambda: self._analyze_auth_failures()
                },
                {
                    'question': "What are the most critical security events observed in the logs and their potential impact?",
                    'answer': lambda: self._analyze_critical_security_events()
                }
            ],
            'visualization': [
                {
                    'question': "Create a time series visualization of security events over time.",
                    'answer': lambda: self._get_security_visualization_code()
                },
                {
                    'question': "Generate a visualization showing the distribution of different types of security events.",
                    'answer': lambda: self._get_event_distribution_visualization()
                }
            ],
            'security_config': [
                {
                    'question': "Based on the observed authentication failures, what SSH configuration changes would you recommend?",
                    'answer': lambda: self._get_ssh_config_recommendation()
                },
                {
                    'question': "What system logging configuration would you recommend based on the observed security events?",
                    'answer': lambda: self._get_logging_config_recommendation()
                }
            ]
        }
        
        # Generate dataset from templates
        for category, questions in templates.items():
            for template in questions:
                try:
                    answer = template['answer']()
                    if answer:  # Only add if we got a valid answer
                        dataset.append({
                            'question': template['question'],
                            'answer': answer,
                            'category': category
                        })
                except Exception as e:
                    logger.error(f"Error generating answer for question: {template['question']}, error: {e}")
        
        return dataset
    
    def _get_severity_distribution(self) -> str:
        if self.df.empty:
            return "No logs available for severity distribution analysis."
        
        severity_counts = self.df['severity'].value_counts()
        total_logs = len(self.df)
        
        result = f"Total log entries: {total_logs}\n\nSeverity distribution:\n"
        for severity, count in severity_counts.items():
            percentage = (count / total_logs) * 100
            result += f"- {severity}: {count} entries ({percentage:.1f}%)\n"
        
        return result
    
    def _get_security_events_summary(self) -> str:
        if self.df.empty:
            return "No logs available for security events analysis."
        
        security_keywords = {
            'authentication': ['auth', 'login', 'password', 'ssh'],
            'access_control': ['access', 'permission', 'denied', 'forbidden'],
            'system_security': ['firewall', 'iptables', 'security', 'attack'],
            'network_security': ['connection', 'port', 'scan', 'ddos']
        }
        
        event_counts = defaultdict(int)
        for _, row in self.df.iterrows():
            message = row['message'].lower()
            for category, keywords in security_keywords.items():
                if any(keyword in message for keyword in keywords):
                    event_counts[category] += 1
        
        result = "Security events summary:\n"
        for category, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
            result += f"- {category.replace('_', ' ').title()}: {count} events\n"
        
        return result
    
    def _get_auth_events_count(self) -> str:
        """인증 관련 이벤트 수를 계산합니다."""
        if self.df.empty:
            return "No logs available for authentication events analysis."
        
        auth_keywords = ['auth', 'login', 'password', 'ssh', 'su', 'sudo']
        
        auth_events = self.df[self.df['message'].str.lower().str.contains('|'.join(auth_keywords))]
        
        success_count = len(auth_events[auth_events['message'].str.contains('success|accepted', case=False)])
        failure_count = len(auth_events[auth_events['message'].str.contains('fail|denied|rejected', case=False)])
        
        return f"Authentication events summary:\n- Total: {len(auth_events)} events\n- Successful: {success_count}\n- Failed: {failure_count}"
    
    def _analyze_auth_failures(self) -> str:
        """인증 실패 패턴을 분석합니다."""
        if self.df.empty:
            return "No logs available for authentication failure analysis."
        
        failure_events = self.df[
            self.df['message'].str.contains('fail|denied|rejected', case=False) &
            self.df['message'].str.contains('auth|login|ssh|su|sudo', case=False)
        ]
        
        if failure_events.empty:
            return "No authentication failures found in the logs."
        
        ip_failures = defaultdict(int)
        for _, row in failure_events.iterrows():
            # IP 주소 추출 (정규식 사용)
            ip_match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', row['message'])
            if ip_match:
                ip_failures[ip_match.group()] += 1
        
        result = "Authentication failure analysis:\n"
        result += f"Total authentication failures: {len(failure_events)}\n\n"
        
        # IP 주소별 실패 횟수 정렬
        for ip, count in sorted(ip_failures.items(), key=lambda x: x[1], reverse=True):
            if count > 5:
                result += f"- IP {ip}: {count} failed attempts (Potential brute force attack)\n"
        
        return result
    
    def _get_security_visualization_code(self) -> str:
        return """```python
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Prepare the data
                security_events = df[df['category'] == 'security'].copy()
                security_events['hour'] = security_events['timestamp'].dt.hour

                # Create the visualization
                plt.figure(figsize=(12, 6))
                sns.histplot(data=security_events, x='hour', bins=24)
                plt.title('Security Events Distribution by Hour')
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Events')
                plt.tight_layout()
                plt.show()
                ```"""
    
    def _get_ssh_config_recommendation(self) -> str:
        auth_failures = self.df[
            (self.df['category'] == 'auth') & 
            (self.df['program'] == 'sshd') &
            self.df['message'].str.contains('fail|invalid|bad|wrong', case=False)
        ]
        
        if len(auth_failures) > 10:
            return """Based on the high number of SSH authentication failures, recommended sshd_config settings:

                # /etc/ssh/sshd_config
                PermitRootLogin no
                MaxAuthTries 3
                LoginGraceTime 60
                PasswordAuthentication no
                UsePAM yes

                # Also recommended to implement fail2ban:
                [sshd]
                enabled = true
                bantime = 3600
                findtime = 600
                maxretry = 3"""
        
        return "Current SSH configuration appears adequate based on log patterns."

    def _get_event_distribution_visualization(self) -> str:
        return """
            ```python
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns

            security_categories = {
                'authentication': ['auth', 'login', 'password'],
                'access_control': ['access', 'permission', 'denied'],
                'system_security': ['firewall', 'security', 'attack'],
                'network_security': ['connection', 'port', 'scan']
            }

            category_counts = {}
            for category, keywords in security_categories.items():
                count = len(df[df['message'].str.contains('|'.join(keywords), case=False)])
                category_counts[category] = count

            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
            plt.title('Distribution of Security Event Types')
            plt.xlabel('Event Category')
            plt.ylabel('Number of Events')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            ```
            """

    def _analyze_critical_security_events(self) -> str:
        """중요한 보안 이벤트를 분석하고 잠재적 영향을 평가합니다."""
        if self.df.empty:
            return "No logs available for critical security events analysis."
        
        critical_events = {
            'intrusion': {
                'keywords': ['intrusion', 'breach', 'compromise', 'hack'],
                'impact': 'Potential system compromise and data exposure'
            },
            'malware': {
                'keywords': ['malware', 'virus', 'trojan', 'ransomware'],
                'impact': 'System infection and potential data encryption'
            },
            'exploit': {
                'keywords': ['exploit', 'vulnerability', 'CVE-', 'zero-day'],
                'impact': 'System vulnerability exploitation'
            },
            'data_leak': {
                'keywords': ['leak', 'exfil', 'data breach', 'sensitive'],
                'impact': 'Unauthorized data access and potential data loss'
            }
        }
        
        detected_events = defaultdict(list)
        for _, row in self.df.iterrows():
            message = row['message'].lower()
            for category, info in critical_events.items():
                if any(keyword in message for keyword in info['keywords']):
                    detected_events[category].append({
                        'timestamp': row['timestamp'],
                        'message': row['message'],
                        'impact': info['impact']
                    })
        
        result = "Critical Security Events Analysis:\n\n"
        
        if not detected_events:
            return result + "No critical security events detected in the logs."
        
        for category, events in detected_events.items():
            if events:
                result += f"{category.replace('_', ' ').title()} Events:\n"
                result += f"Potential Impact: {events[0]['impact']}\n"
                result += f"Number of Events: {len(events)}\n"
                result += "Recent Events:\n"
                for event in events[:3]:  # 최근 3개 이벤트만 표시
                    result += f"- {event['timestamp']}: {event['message']}\n"
                result += "\n"
        
        total_critical = sum(len(events) for events in detected_events.values())
        if total_critical > 10:
            result += "Overall Risk Assessment: HIGH - Multiple critical security events detected\n"
        elif total_critical > 5:
            result += "Overall Risk Assessment: MEDIUM - Several critical security events detected\n"
        else:
            result += "Overall Risk Assessment: LOW - Few critical security events detected\n"
        
        return result

    def _get_logging_config_recommendation(self) -> str:
        if self.df.empty:
            return "No logs available for logging configuration recommendations."
        
        recommendations = ["System logging configuration recommendations based on log analysis:"]
        
        severity_counts = self.df['severity'].value_counts()
        if 'error' in severity_counts and severity_counts['error'] > 100:
            recommendations.append("""
            # Increase error logging verbosity
            *.error /var/log/error.log
            *.warning /var/log/warning.log
            """)
        
        security_events = self.df[self.df['message'].str.contains('security|attack|intrusion', case=False)]
        if not security_events.empty:
            recommendations.append("""
            # Enhanced security event logging
            auth.* /var/log/auth.log
            kern.* /var/log/kern.log
            local7.* /var/log/boot.log
            """)
        
        recommendations.append("""
        # Log rotation settings
        /var/log/*.log {
            daily
            rotate 7
            compress
            delaycompress
            missingok
            notifempty
            create 640 root adm
        }
        """)
        
        if len(self.df) > 10000:
            recommendations.append("""
            # Extended log retention for high-volume systems
            /var/log/*.log {
                weekly
                rotate 4
                compress
                delaycompress
                missingok
                notifempty
                create 640 root adm
            }
            """)
        
        if not security_events.empty:
            recommendations.append("""
            # Security event log aggregation
            $ModLoad imtcp
            $InputTCPServerRun 514
            $template RemoteLogs,"/var/log/remote/%HOSTNAME%/%PROGRAMNAME%.log"
            *.* ?RemoteLogs
            """)
        
        return "\n".join(recommendations)