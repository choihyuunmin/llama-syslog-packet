import json
import logging
import time
import re
import numpy as np
from pathlib import Path
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Evaluator:
    """
    Domain-specific Cybersecurity Evaluator
    - Attack type classification accuracy
    - Information extraction performance
    - Threat detection accuracy
    - Natural language generation quality
    """
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.smoothing = SmoothingFunction().method1
        
        # Define attack types
        self.attack_types = [
            "Brute-force / Login Attacks",
            "Lateral Movement", 
            "Malware / Backdoor Installation",
            "DoS / DDoS",
            "Port Scanning / Reconnaissance",
            "Privilege Escalation",
            "Command & Control (C2) Communication",
            "Data Exfiltration",
            "Advanced Persistent Threat (APT)",
            "Insider Threat"
        ]
        
    def extract_attack_type_from_text(self, text: str) -> str:
        """Extract attack type from text"""
        text_lower = text.lower()
        
        # Keyword mapping for each attack type
        attack_keywords = {
            "Brute-force / Login Attacks": ["brute", "login", "password", "authentication", "failed"],
            "Lateral Movement": ["lateral", "movement", "pivot", "spread", "network"],
            "Malware / Backdoor Installation": ["malware", "backdoor", "trojan", "virus", "infection"],
            "DoS / DDoS": ["dos", "ddos", "denial", "service", "flood"],
            "Port Scanning / Reconnaissance": ["scan", "reconnaissance", "probe", "discovery", "enumeration"],
            "Privilege Escalation": ["privilege", "escalation", "elevation", "root", "admin"],
            "Command & Control (C2) Communication": ["c2", "command", "control", "beacon", "communication"],
            "Data Exfiltration": ["exfiltration", "data", "steal", "extract", "leak"],
            "Advanced Persistent Threat (APT)": ["apt", "persistent", "advanced", "sophisticated"],
            "Insider Threat": ["insider", "internal", "employee", "user"]
        }
        
        # Score-based matching
        scores = {}
        for attack_type, keywords in attack_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[attack_type] = score
        
        # Return attack type with highest score
        if scores:
            return max(scores, key=scores.get)
        
        # Try direct matching
        for attack_type in self.attack_types:
            if attack_type.lower() in text_lower:
                return attack_type
                
        return "Unknown"
    
    def calculate_attack_classification_accuracy(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate attack type classification accuracy"""
        y_true = []
        y_pred = []
        
        for pred in predictions:
            true_attack = pred['expected_output'].strip()
            pred_attack = self.extract_attack_type_from_text(pred['output'])
            
            y_true.append(true_attack)
            y_pred.append(pred_attack)
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(y_true) if y_true else 0
        
        # Calculate F1 score (macro average)
        try:
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        except:
            f1_macro = precision_macro = recall_macro = 0
        
        return {
            'attack_classification_accuracy': accuracy,
            'attack_classification_f1': f1_macro,
            'attack_classification_precision': precision_macro,
            'attack_classification_recall': recall_macro,
            'predictions_detail': list(zip(y_true, y_pred))
        }
    
    def extract_ip_addresses(self, text: str) -> List[str]:
        """Extract IP addresses from text"""
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        return re.findall(ip_pattern, text)
    
    def extract_ports(self, text: str) -> List[str]:
        """Extract port numbers from text"""
        port_pattern = r'\b(?:port\s+)?(\d{1,5})\b'
        ports = re.findall(port_pattern, text.lower())
        return [p for p in ports if 1 <= int(p) <= 65535]
    
    def extract_protocols(self, text: str) -> List[str]:
        """Extract protocols from text"""
        protocols = ['tcp', 'udp', 'http', 'https', 'ftp', 'ssh', 'dns', 'smtp']
        found_protocols = []
        text_lower = text.lower()
        for protocol in protocols:
            if protocol in text_lower:
                found_protocols.append(protocol.upper())
        return found_protocols
    
    def calculate_information_extraction_f1(self, predictions: List[Dict], data_source: List[Dict]) -> Dict[str, float]:
        """Calculate information extraction performance (IP, ports, protocols, etc.)"""
        ip_f1_scores = []
        port_f1_scores = []
        protocol_f1_scores = []
        
        for i, pred in enumerate(predictions):
            if i < len(data_source):
                true_ips = set()
                true_ports = set()
                true_protocols = set()
                
                for item in data_source[i].get('input', []):
                    if item.get('type') == 'network_packet':
                        true_ips.update([item.get('src_ip', ''), item.get('dst_ip', '')])
                        true_ports.update([str(item.get('src_port', '')), str(item.get('dst_port', ''))])
                        true_protocols.add(item.get('protocol', '').upper())
                    elif item.get('type') == 'syslog':
                        if item.get('related_ip'):
                            true_ips.add(item.get('related_ip'))
                
                pred_ips = set(self.extract_ip_addresses(pred['output']))
                pred_ports = set(self.extract_ports(pred['output']))
                pred_protocols = set(self.extract_protocols(pred['output']))
                
                def calculate_f1(true_set, pred_set):
                    if not true_set and not pred_set:
                        return 1.0
                    if not true_set or not pred_set:
                        return 0.0
                    
                    true_set = {x for x in true_set if x}  # 빈 문자열 제거
                    pred_set = {x for x in pred_set if x}
                    
                    if not true_set and not pred_set:
                        return 1.0
                    if not true_set or not pred_set:
                        return 0.0
                    
                    intersection = len(true_set.intersection(pred_set))
                    precision = intersection / len(pred_set) if pred_set else 0
                    recall = intersection / len(true_set) if true_set else 0
                    
                    if precision + recall == 0:
                        return 0.0
                    return 2 * (precision * recall) / (precision + recall)
                
                ip_f1_scores.append(calculate_f1(true_ips, pred_ips))
                port_f1_scores.append(calculate_f1(true_ports, pred_ports))
                protocol_f1_scores.append(calculate_f1(true_protocols, pred_protocols))
        
        return {
            'ip_extraction_f1': np.mean(ip_f1_scores) if ip_f1_scores else 0,
            'port_extraction_f1': np.mean(port_f1_scores) if port_f1_scores else 0,
            'protocol_extraction_f1': np.mean(protocol_f1_scores) if protocol_f1_scores else 0,
            'overall_extraction_f1': np.mean([
                np.mean(ip_f1_scores) if ip_f1_scores else 0,
                np.mean(port_f1_scores) if port_f1_scores else 0,
                np.mean(protocol_f1_scores) if protocol_f1_scores else 0
            ])
        }
    
    def calculate_threat_detection_accuracy(self, predictions: List[Dict]) -> Dict[str, float]:
        correct_detections = 0
        total_detections = len(predictions)
        
        threat_indicators = ['attack', 'threat', 'malicious', 'suspicious', 'intrusion', 'breach', 'compromised']
        
        for pred in predictions:
            pred_text = pred['output'].lower()
            expected_text = pred['expected_output'].lower()
            
            threat_mentioned = any(indicator in pred_text for indicator in threat_indicators)
            expected_threat = any(attack_type.lower() in expected_text for attack_type in self.attack_types)
            
            if threat_mentioned and expected_threat:
                correct_detections += 1
            elif not threat_mentioned and not expected_threat:
                correct_detections += 1
        
        return {
            'threat_detection_accuracy': correct_detections / total_detections if total_detections > 0 else 0,
            'threat_detection_rate': correct_detections / total_detections if total_detections > 0 else 0
        }
    
    def calculate_response_quality(self, predictions: List[Dict]) -> Dict[str, float]:
        quality_scores = []
        
        for pred in predictions:
            response = pred['output']
            score = 0
            
            if 50 <= len(response) <= 500:
                score += 0.2
            
            if any(self.extract_ip_addresses(response)):
                score += 0.2
            if any(self.extract_ports(response)):
                score += 0.1
            if any(self.extract_protocols(response)):
                score += 0.1
            
            if any(attack_type.lower() in response.lower() for attack_type in self.attack_types):
                score += 0.2
            
            recommendation_keywords = ['recommend', 'suggest', 'should', 'action', 'mitigate', 'prevent']
            if any(keyword in response.lower() for keyword in recommendation_keywords):
                score += 0.2
            
            quality_scores.append(min(score, 1.0))
        
        return {
            'response_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'response_completeness': np.mean(quality_scores) if quality_scores else 0
        }
    
    def calculate_traditional_nlg_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        bleu_scores = []
        rouge_scores = []
        
        for pred in predictions:
            reference = [word_tokenize(pred['expected_output'].lower())]
            candidate = word_tokenize(pred['output'].lower())
            
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=self.smoothing)
            bleu_scores.append(bleu_score)
            
            rouge_result = self.scorer.score(pred['expected_output'], pred['output'])
            rouge_scores.append(rouge_result['rougeL'].fmeasure)
        
        return {
            'bleu': np.mean(bleu_scores) if bleu_scores else 0,
            'rouge_l': np.mean(rouge_scores) if rouge_scores else 0
        }
    
    def calculate_semantic_similarity(self, predictions: List[Dict]) -> Dict[str, float]:
        similarities = []
        
        for pred in predictions:
            pred_emb = self.similarity_model.encode(pred['output'], convert_to_tensor=True)
            gt_emb = self.similarity_model.encode(pred['expected_output'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_emb, gt_emb).item()
            similarities.append(similarity)
        
        return {
            'semantic_similarity': np.mean(similarities) if similarities else 0
        }
    
    def evaluate(self, original_data: List[Dict], predictions: Dict[str, List[Dict]], dataset_type: str) -> Dict[str, Dict]:
        metrics_data = {}
        
        for model_name, model_predictions in predictions.items():
            try:
                self.logger.info(f"Evaluating {model_name}...")
                
                attack_metrics = self.calculate_attack_classification_accuracy(model_predictions)
                extraction_metrics = self.calculate_information_extraction_f1(model_predictions, original_data)
                detection_metrics = self.calculate_threat_detection_accuracy(model_predictions)
                quality_metrics = self.calculate_response_quality(model_predictions)
                nlg_metrics = self.calculate_traditional_nlg_metrics(model_predictions)
                similarity_metrics = self.calculate_semantic_similarity(model_predictions)
                
                all_metrics = {
                    **attack_metrics,
                    **extraction_metrics,
                    **detection_metrics,
                    **quality_metrics,
                    **nlg_metrics,
                    **similarity_metrics
                }
                
                # 종합 점수 계산
                domain_score = (
                    all_metrics['attack_classification_accuracy'] * 0.3 +
                    all_metrics['overall_extraction_f1'] * 0.2 +
                    all_metrics['threat_detection_accuracy'] * 0.2 +
                    all_metrics['response_quality_score'] * 0.3
                )
                
                all_metrics['domain_specific_score'] = domain_score
                all_metrics['overall_score'] = (domain_score * 0.7 + all_metrics['semantic_similarity'] * 0.3)
                
                metrics_data[model_name] = all_metrics
                
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {e}")
                continue
        
        return metrics_data
    
    def generate_detailed_report(self, metrics_data: Dict[str, Dict], output_path: str):
        report = []
        report.append("# 사이버보안 도메인 특화 평가 리포트\n")
        report.append("## 평가 지표 설명\n")
        report.append("- **공격 분류 정확도**: 공격 유형을 정확히 분류한 비율")
        report.append("- **정보 추출 F1**: IP, 포트, 프로토콜 등 기술적 정보 추출 성능")
        report.append("- **위협 탐지 정확도**: 보안 위협을 정확히 탐지한 비율")
        report.append("- **응답 품질**: 응답의 완성도 및 유용성")
        report.append("- **도메인 특화 점수**: 위 지표들의 가중 평균\n")
        
        report.append("## 모델별 성능 비교\n")
        report.append("| 모델 | 공격분류 | 정보추출 | 위협탐지 | 응답품질 | 도메인점수 | 종합점수 |")
        report.append("|------|----------|----------|----------|----------|------------|----------|")
        
        for model_name, metrics in metrics_data.items():
            report.append(
                    f"| {model_name} | "
                    f"{metrics.get('attack_classification_accuracy', 0):.3f} | "
                    f"{metrics.get('overall_extraction_f1', 0):.3f} | "
                    f"{metrics.get('threat_detection_accuracy', 0):.3f} | "
                    f"{metrics.get('response_quality_score', 0):.3f} | "
                    f"{metrics.get('domain_specific_score', 0):.3f} | "
                    f"{metrics.get('overall_score', 0):.3f} |"
                )
        
        report.append("\n## 세부 메트릭\n")
        for model_name, metrics in metrics_data.items():
            report.append(f"### {model_name}\n")
            for metric_name, value in metrics.items():
                if not metric_name.endswith('_detail') and isinstance(value, (int, float)):
                    report.append(f"- {metric_name}: {value:.4f}")
            report.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))


class BenchmarkEvaluator(Evaluator):    
    def __init__(self):
        super().__init__()
    
    def calculate_accuracy(self, predictions):
        """기존 정확도 계산 메서드 호환"""
        attack_metrics = self.calculate_attack_classification_accuracy(predictions)
        return {
            'accuracy': attack_metrics['attack_classification_accuracy'],
            'exact_accuracy': attack_metrics['attack_classification_accuracy'],
            'semantic_accuracy': attack_metrics['attack_classification_accuracy']
        } 