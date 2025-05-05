import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging

class BenchmarkEvaluator:
    def __init__(self, benchmark_path: str):
        self.benchmark_path = Path(benchmark_path)
        self.logger = logging.getLogger(__name__)
        
    def load_benchmark(self) -> Dict:
        """Load benchmark data from JSON file."""
        with open(self.benchmark_path, 'r') as f:
            return json.load(f)
            
    def evaluate_accuracy(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate prediction accuracy."""
        results = {
            'total_questions': len(ground_truth),
            'correct_answers': 0,
            'incorrect_answers': 0,
            'accuracy': 0.0
        }
        
        for pred, truth in zip(predictions, ground_truth):
            if self._compare_answers(pred['answer'], truth['answer']):
                results['correct_answers'] += 1
            else:
                results['incorrect_answers'] += 1
                
        results['accuracy'] = results['correct_answers'] / results['total_questions']
        return results
        
    def evaluate_complexity(self, predictions: List[Dict]) -> Dict:
        """Evaluate answer complexity and depth."""
        complexity_scores = []
        depth_scores = []
        
        for pred in predictions:
            # Complexity: length and technical terms
            complexity = len(pred['answer'].split())
            complexity_scores.append(complexity)
            
            # Depth: number of analysis aspects
            depth = len(pred['answer'].split('.'))
            depth_scores.append(depth)
            
        return {
            'average_complexity': np.mean(complexity_scores),
            'max_complexity': np.max(complexity_scores),
            'average_depth': np.mean(depth_scores),
            'max_depth': np.max(depth_scores)
        }
        
    def evaluate_anomaly_detection(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate anomaly detection performance."""
        y_true = []
        y_pred = []
        
        for pred, truth in zip(predictions, ground_truth):
            if 'anomaly' in pred['question'].lower():
                y_true.append(1 if 'detected' in truth['answer'].lower() else 0)
                y_pred.append(1 if 'detected' in pred['answer'].lower() else 0)
                
        if not y_true:  # No anomaly questions
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def evaluate_pattern_recognition(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate pattern recognition capabilities."""
        pattern_scores = []
        
        for pred, truth in zip(predictions, ground_truth):
            if 'pattern' in pred['question'].lower():
                pred_patterns = self._extract_patterns(pred['answer'])
                truth_patterns = self._extract_patterns(truth['answer'])
                
                if truth_patterns:
                    score = len(set(pred_patterns) & set(truth_patterns)) / len(truth_patterns)
                    pattern_scores.append(score)
                    
        return {
            'average_pattern_score': np.mean(pattern_scores) if pattern_scores else 0.0,
            'pattern_recognition_rate': len(pattern_scores) / len(predictions)
        }
        
    def evaluate_correlation_analysis(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate correlation analysis capabilities."""
        correlation_scores = []
        
        for pred, truth in zip(predictions, ground_truth):
            if 'correlation' in pred['question'].lower():
                pred_correlations = self._extract_correlations(pred['answer'])
                truth_correlations = self._extract_correlations(truth['answer'])
                
                if truth_correlations:
                    score = len(set(pred_correlations) & set(truth_correlations)) / len(truth_correlations)
                    correlation_scores.append(score)
                    
        return {
            'average_correlation_score': np.mean(correlation_scores) if correlation_scores else 0.0,
            'correlation_analysis_rate': len(correlation_scores) / len(predictions)
        }
        
    def _compare_answers(self, pred_answer: str, truth_answer: str) -> bool:
        """Compare predicted and ground truth answers."""
        # Normalize answers for comparison
        pred = pred_answer.lower().strip()
        truth = truth_answer.lower().strip()
        
        # Exact match
        if pred == truth:
            return True
            
        # Partial match with key information
        key_terms = set(truth.split())
        pred_terms = set(pred.split())
        return len(key_terms & pred_terms) / len(key_terms) > 0.7
        
    def _extract_patterns(self, answer: str) -> List[str]:
        """Extract patterns from answer text."""
        patterns = []
        for line in answer.split('.'):
            if 'pattern' in line.lower() or 'trend' in line.lower():
                patterns.append(line.strip())
        return patterns
        
    def _extract_correlations(self, answer: str) -> List[str]:
        """Extract correlations from answer text."""
        correlations = []
        for line in answer.split('.'):
            if 'correlation' in line.lower() or 'relationship' in line.lower():
                correlations.append(line.strip())
        return correlations
        
    def run_evaluation(self, predictions: List[Dict]) -> Dict:
        """Run complete evaluation."""
        ground_truth = self.load_benchmark()
        
        results = {
            'accuracy': self.evaluate_accuracy(predictions, ground_truth),
            'complexity': self.evaluate_complexity(predictions),
            'anomaly_detection': self.evaluate_anomaly_detection(predictions, ground_truth),
            'pattern_recognition': self.evaluate_pattern_recognition(predictions, ground_truth),
            'correlation_analysis': self.evaluate_correlation_analysis(predictions, ground_truth)
        }
        
        # Calculate overall score
        weights = {
            'accuracy': 0.3,
            'complexity': 0.2,
            'anomaly_detection': 0.2,
            'pattern_recognition': 0.15,
            'correlation_analysis': 0.15
        }
        
        overall_score = (
            results['accuracy']['accuracy'] * weights['accuracy'] +
            (results['complexity']['average_complexity'] / 100) * weights['complexity'] +
            results['anomaly_detection']['f1_score'] * weights['anomaly_detection'] +
            results['pattern_recognition']['average_pattern_score'] * weights['pattern_recognition'] +
            results['correlation_analysis']['average_correlation_score'] * weights['correlation_analysis']
        )
        
        results['overall_score'] = overall_score
        return results 