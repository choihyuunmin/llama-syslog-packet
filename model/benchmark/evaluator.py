import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class BenchmarkEvaluator:
    def __init__(self, benchmark_path, batch_size=100):
        self.benchmark_path = Path(benchmark_path)
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_benchmark(self):
        """Load benchmark data from JSON file."""
        with open(self.benchmark_path, 'r') as f:
            return json.load(f)
            
    def evaluate_accuracy(self, predictions, ground_truth):
        """Evaluate prediction accuracy using semantic similarity."""
        results = {
            'total_questions': len(ground_truth),
            'correct_answers': 0,
            'incorrect_answers': 0,
            'accuracy': 0.0,
            'similarity_scores': []
        }
        
        # Batch processing for embeddings
        for i in tqdm(range(0, len(predictions), self.batch_size), desc="Evaluating accuracy"):
            batch_preds = predictions[i:i + self.batch_size]
            batch_truth = ground_truth[i:i + self.batch_size]
            
            pred_embeddings = self.model.encode([p['answer'] for p in batch_preds])
            truth_embeddings = self.model.encode([t['answer'] for t in batch_truth])
            
            for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
                similarity = np.dot(pred_emb, truth_emb) / (
                    np.linalg.norm(pred_emb) * np.linalg.norm(truth_emb)
                )
                results['similarity_scores'].append(similarity)
                
                if similarity > 0.8:  # Threshold for considering as correct
                    results['correct_answers'] += 1
                else:
                    results['incorrect_answers'] += 1
                
        results['accuracy'] = results['correct_answers'] / results['total_questions']
        results['average_similarity'] = np.mean(results['similarity_scores'])
        return results
        
    def evaluate_complexity(self, predictions):
        """Evaluate answer complexity and depth."""
        complexity_scores = []
        depth_scores = []
        technical_terms = set([
            'protocol', 'packet', 'session', 'anomaly', 'correlation',
            'security', 'attack', 'vulnerability', 'exploit', 'malware'
        ])
        
        for pred in tqdm(predictions, desc="Evaluating complexity"):
            # Complexity: length, technical terms, and structure
            answer = pred['answer'].lower()
            complexity = len(answer.split())
            technical_term_count = sum(1 for term in technical_terms if term in answer)
            complexity_scores.append(complexity * (1 + 0.1 * technical_term_count))
            
            # Depth: number of analysis aspects and hierarchical structure
            depth = len(answer.split('.'))
            depth_scores.append(depth)
            
        return {
            'average_complexity': np.mean(complexity_scores),
            'max_complexity': np.max(complexity_scores),
            'average_depth': np.mean(depth_scores),
            'max_depth': np.max(depth_scores),
            'technical_term_usage': technical_term_count / len(predictions)
        }
        
    def evaluate_anomaly_detection(self, predictions, ground_truth):
        """Evaluate anomaly detection performance."""
        y_true = []
        y_pred = []
        confidence_scores = []
        
        for pred, truth in tqdm(zip(predictions, ground_truth), desc="Evaluating anomaly detection"):
            if 'anomaly' in pred['question'].lower():
                # Use semantic similarity for anomaly detection
                pred_emb = self.model.encode(pred['answer'])
                truth_emb = self.model.encode(truth['answer'])
                similarity = np.dot(pred_emb, truth_emb) / (
                    np.linalg.norm(pred_emb) * np.linalg.norm(truth_emb)
                )
                
                y_true.append(1 if 'detected' in truth['answer'].lower() else 0)
                y_pred.append(1 if similarity > 0.8 else 0)
                confidence_scores.append(similarity)
                
        if not y_true:  # No anomaly questions
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'average_confidence': 0.0
            }
            
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_confidence': np.mean(confidence_scores)
        }
        
    def evaluate_pattern_recognition(self, predictions, ground_truth):
        """Evaluate pattern recognition capabilities."""
        pattern_scores = []
        pattern_details = []
        
        for pred, truth in tqdm(zip(predictions, ground_truth), desc="Evaluating pattern recognition"):
            if 'pattern' in pred['question'].lower():
                pred_patterns = self._extract_patterns(pred['answer'])
                truth_patterns = self._extract_patterns(truth['answer'])
                
                if truth_patterns:
                    # Calculate semantic similarity for patterns
                    pred_emb = self.model.encode(pred_patterns)
                    truth_emb = self.model.encode(truth_patterns)
                    similarities = np.dot(pred_emb, truth_emb.T)
                    score = np.mean(np.max(similarities, axis=1))
                    pattern_scores.append(score)
                    
                    pattern_details.append({
                        'predicted': pred_patterns,
                        'ground_truth': truth_patterns,
                        'similarity': score
                    })
                    
        return {
            'average_pattern_score': np.mean(pattern_scores) if pattern_scores else 0.0,
            'pattern_recognition_rate': len(pattern_scores) / len(predictions),
            'pattern_details': pattern_details
        }
        
    def evaluate_correlation_analysis(self, predictions, ground_truth):
        """Evaluate correlation analysis capabilities."""
        correlation_scores = []
        correlation_details = []
        
        for pred, truth in tqdm(zip(predictions, ground_truth), desc="Evaluating correlation analysis"):
            if 'correlation' in pred['question'].lower():
                pred_correlations = self._extract_correlations(pred['answer'])
                truth_correlations = self._extract_correlations(truth['answer'])
                
                if truth_correlations:
                    # Calculate semantic similarity for correlations
                    pred_emb = self.model.encode(pred_correlations)
                    truth_emb = self.model.encode(truth_correlations)
                    similarities = np.dot(pred_emb, truth_emb.T)
                    score = np.mean(np.max(similarities, axis=1))
                    correlation_scores.append(score)
                    
                    correlation_details.append({
                        'predicted': pred_correlations,
                        'ground_truth': truth_correlations,
                        'similarity': score
                    })
                    
        return {
            'average_correlation_score': np.mean(correlation_scores) if correlation_scores else 0.0,
            'correlation_analysis_rate': len(correlation_scores) / len(predictions),
            'correlation_details': correlation_details
        }
        
    def _extract_patterns(self, answer):
        """Extract patterns from answer text."""
        patterns = []
        for line in answer.split('.'):
            if any(term in line.lower() for term in ['pattern', 'trend', 'sequence', 'recurring']):
                patterns.append(line.strip())
        return patterns
        
    def _extract_correlations(self, answer):
        """Extract correlations from answer text."""
        correlations = []
        for line in answer.split('.'):
            if any(term in line.lower() for term in ['correlation', 'relationship', 'connection', 'association']):
                correlations.append(line.strip())
        return correlations
        
    def run_evaluation(self, predictions):
        """Run complete evaluation."""
        ground_truth = self.load_benchmark()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': self.evaluate_accuracy(predictions, ground_truth),
            'complexity': self.evaluate_complexity(predictions),
            'anomaly_detection': self.evaluate_anomaly_detection(predictions, ground_truth),
            'pattern_recognition': self.evaluate_pattern_recognition(predictions, ground_truth),
            'correlation_analysis': self.evaluate_correlation_analysis(predictions, ground_truth)
        }
        
        # Calculate overall score with dynamic weights
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
        
    def visualize_results(self, results, output_dir):
        """Visualize evaluation results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Overall Performance Metrics
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'complexity', 'anomaly_detection', 'pattern_recognition', 'correlation_analysis']
        scores = [
            results['accuracy']['accuracy'],
            results['complexity']['average_complexity'] / 100,
            results['anomaly_detection']['f1_score'],
            results['pattern_recognition']['average_pattern_score'],
            results['correlation_analysis']['average_correlation_score']
        ]
        
        plt.bar(metrics, scores)
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_metrics.png')
        plt.close()
        
        # 2. Similarity Score Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results['accuracy']['similarity_scores'], bins=20)
        plt.title('Answer Similarity Score Distribution')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'similarity_distribution.png')
        plt.close()
        
        # 3. Pattern Recognition Details
        if results['pattern_recognition']['pattern_details']:
            plt.figure(figsize=(10, 6))
            similarities = [d['similarity'] for d in results['pattern_recognition']['pattern_details']]
            sns.histplot(similarities, bins=20)
            plt.title('Pattern Recognition Similarity Distribution')
            plt.xlabel('Similarity Score')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'pattern_similarity.png')
            plt.close()
        
        # 4. Correlation Analysis Details
        if results['correlation_analysis']['correlation_details']:
            plt.figure(figsize=(10, 6))
            similarities = [d['similarity'] for d in results['correlation_analysis']['correlation_details']]
            sns.histplot(similarities, bins=20)
            plt.title('Correlation Analysis Similarity Distribution')
            plt.xlabel('Similarity Score')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'correlation_similarity.png')
            plt.close() 