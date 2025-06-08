import hashlib
import re
from typing import Dict, List, Any
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class DrainLogParser:
    def __init__(self, depth: int = 4, max_children: int = 100, similarity_threshold: float = 0.5):
        """Drain 파서 초기화.
        
        Args:
            depth (int): 트리의 최대 깊이
            max_children (int): 노드당 최대 자식 수
            similarity_threshold (float): 템플릿 매칭을 위한 유사도 임계값
        """
        self.depth = depth
        self.max_children = max_children
        self.similarity_threshold = similarity_threshold
        self.root = self._create_node()
        self.log_templates: Dict[str, str] = {}
        self.template_count: Dict[str, int] = {}
        
    def _create_node(self) -> Dict:
        """새로운 트리 노드 생성."""
        return {
            'children': {},
            'templates': {}
        }
    
    def _get_log_tokens(self, log_line: str) -> List[str]:
        """로그 라인을 토큰으로 분리.
        
        Args:
            log_line (str): 원본 로그 라인
            
        Returns:
            List[str]: 토큰화된 로그 라인
        """
        # 공백으로 분리하고 특수문자 처리
        tokens = re.split(r'[\s=:,]', log_line)
        return [t for t in tokens if t]
    
    def _get_token_length(self, token: str) -> int:
        """토큰의 길이를 반환. 숫자는 0으로 처리."""
        if token.isdigit():
            return 0
        return len(token)
    
    def _calculate_similarity(self, template: str, log_line: str) -> float:
        """두 로그 라인 간의 유사도 계산.
        
        Args:
            template (str): 템플릿 로그 라인
            log_line (str): 비교할 로그 라인
            
        Returns:
            float: 유사도 점수 (0-1)
        """
        template_tokens = self._get_log_tokens(template)
        log_tokens = self._get_log_tokens(log_line)
        
        if len(template_tokens) != len(log_tokens):
            return 0.0
            
        matches = sum(1 for t1, t2 in zip(template_tokens, log_tokens) if t1 == t2)
        return matches / len(template_tokens)
    
    def _get_template_id(self, template):
        """템플릿의 고유 ID 생성."""
        return hashlib.md5(template.encode()).hexdigest()
    
    def parse(self, log_line):
        """로그 라인을 파싱하여 템플릿과 파라미터를 추출.
        
        Args:
            log_line (str): 파싱할 로그 라인
            
        Returns:
            Tuple[str, str]: (템플릿 ID, 파라미터화된 로그 라인)
        """
        tokens = self._get_log_tokens(log_line)
        current_node = self.root
        
        # 트리 순회
        for i in range(min(self.depth, len(tokens))):
            token = tokens[i]
            token_length = self._get_token_length(token)
            
            if token_length not in current_node['children']:
                current_node['children'][token_length] = self._create_node()
            
            current_node = current_node['children'][token_length]
            
            if token not in current_node['children']:
                if len(current_node['children']) >= self.max_children:
                    # 가장 유사한 템플릿 찾기
                    best_template = None
                    best_similarity = 0
                    
                    for template in current_node['templates'].values():
                        similarity = self._calculate_similarity(template, log_line)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_template = template
                    
                    if best_similarity >= self.similarity_threshold:
                        template_id = self._get_template_id(best_template)
                        self.template_count[template_id] += 1
                        return template_id, best_template
                
                current_node['children'][token] = self._create_node()
            
            current_node = current_node['children'][token]
        
        # 템플릿 매칭
        best_template = None
        best_similarity = 0
        
        for template in current_node['templates'].values():
            similarity = self._calculate_similarity(template, log_line)
            if similarity > best_similarity:
                best_similarity = similarity
                best_template = template
        
        if best_similarity >= self.similarity_threshold:
            template_id = self._get_template_id(best_template)
            self.template_count[template_id] += 1
            return template_id, best_template
        
        # 새로운 템플릿 생성
        template_id = self._get_template_id(log_line)
        current_node['templates'][template_id] = log_line
        self.log_templates[template_id] = log_line
        self.template_count[template_id] = 1
        
        return template_id, log_line

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

            return self.logs
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
                    'instruction': "What is the distribution of log entries by severity level in the system logs?",
                    'output': lambda: self._get_severity_distribution()
                },
                {
                    'instruction': "What are the most common types of security events in the logs?",
                    'output': lambda: self._get_security_events_summary()
                },
                {
                    'instruction': "How many authentication-related events are there in the logs?",
                    'output': lambda: self._get_auth_events_count()
                }
            ],
            'advanced_analysis': [
                {
                    'instruction': "Analyze the authentication failure patterns in the logs. Are there any signs of brute force attempts?",
                    'output': lambda: self._analyze_auth_failures()
                },
                {
                    'instruction': "What are the most critical security events observed in the logs and their potential impact?",
                    'output': lambda: self._analyze_critical_security_events()
                }
            ],
            'visualization': [
                {
                    'instruction': "Create a time series visualization of security events over time.",
                    'output': lambda: self._get_security_visualization_code()
                },
                {
                    'instruction': "Generate a visualization showing the distribution of different types of security events.",
                    'output': lambda: self._get_event_distribution_visualization()
                }
            ],
            'security_config': [
                {
                    'instruction': "Based on the observed authentication failures, what SSH configuration changes would you recommend?",
                    'output': lambda: self._get_ssh_config_recommendation()
                },
                {
                    'instruction': "What system logging configuration would you recommend based on the observed security events?",
                    'output': lambda: self._get_logging_config_recommendation()
                }
            ]
        }
        
        # Generate dataset from templates
        for category, questions in templates.items():
            for template in questions:
                try:
                    answer = template['output']()
                    if answer:  # Only add if we got a valid answer
                        dataset.append({
                            'instruction': template['instruction'],
                            'output': answer,
                            'category': category
                        })
                except Exception as e:
                    logger.error(f"Error generating answer for question: {template['instruction']}, error: {e}")
        
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
