import json
import logging
import time
import psutil
import numpy as np
from pathlib import Path
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

class BenchmarkEvaluator:
    def __init__(self, benchmark_path, batch_size=100):
        self.logger = logging.getLogger(__name__)
        self.benchmark_path = Path(benchmark_path)
        self.batch_size = batch_size
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # 문장 유사도 계산을 위한 모델 로드
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        with open(self.benchmark_path, 'r') as f:
            self.benchmark_data = json.load(f)

    def calculate_semantic_similarity(self, predictions, ground_truth):
        """의미적 유사도 계산"""
        pred_embeddings = self.similarity_model.encode(predictions, convert_to_tensor=True)
        gt_embeddings = self.similarity_model.encode(ground_truth, convert_to_tensor=True)
        
        # 코사인 유사도 계산
        similarities = util.pytorch_cos_sim(pred_embeddings, gt_embeddings)
        return similarities.diagonal().mean().item()

    def calculate_accuracy_metrics(self, predictions, ground_truth):
        """정확도 관련 메트릭 계산"""
        return {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, average='weighted'),
            'recall': recall_score(ground_truth, predictions, average='weighted'),
            'f1_score': f1_score(ground_truth, predictions, average='weighted')
        }

    def calculate_rouge_scores(self, predictions, ground_truth):
        """ROUGE 점수 계산"""
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, gt in zip(predictions, ground_truth):
            score = self.scorer.score(pred, gt)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
            
        return {k: np.mean(v) for k, v in scores.items()}

    def calculate_hallucination_rate(self, predictions, ground_truth):
        """환각률 계산 (의미적 유사도 기반)"""
        similarities = []
        for pred, gt in zip(predictions, ground_truth):
            pred_emb = self.similarity_model.encode(pred, convert_to_tensor=True)
            gt_emb = self.similarity_model.encode(gt, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_emb, gt_emb).item()
            similarities.append(similarity)
        
        # 유사도가 0.5 미만인 경우를 환각으로 간주
        hallucination_rate = sum(1 for s in similarities if s < 0.5) / len(similarities)
        return hallucination_rate

    def run_evaluation(self, predictions):
        """벤치마크 평가 실행"""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        ground_truth = [item['expected_output'] for item in self.benchmark_data]
        pred_outputs = [pred['output'] for pred in predictions]

        # 기본 메트릭 계산
        accuracy_metrics = self.calculate_accuracy_metrics(pred_outputs, ground_truth)
        rouge_scores = self.calculate_rouge_scores(pred_outputs, ground_truth)
        semantic_similarity = self.calculate_semantic_similarity(pred_outputs, ground_truth)
        hallucination_rate = self.calculate_hallucination_rate(pred_outputs, ground_truth)

        results = {
            'accuracy_metrics': accuracy_metrics,
            'rouge_scores': rouge_scores,
            'semantic_similarity': semantic_similarity,
            'hallucination_rate': hallucination_rate,
            'resource_metrics': {
                'execution_time_ms': (time.time() - start_time) * 1000,
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024 - start_memory
            }
        }

        return results

    def visualize_results(self, results, output_dir):
        """결과 시각화"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 정확도 메트릭 시각화
        plt.figure(figsize=(10, 6))
        accuracy_metrics = results['accuracy_metrics']
        metrics = list(accuracy_metrics.keys())
        values = list(accuracy_metrics.values())
        
        sns.barplot(x=metrics, y=values)
        plt.title('Accuracy Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_metrics.png')
        plt.close()

        # 2. ROUGE 점수 시각화
        plt.figure(figsize=(10, 6))
        rouge_scores = results['rouge_scores']
        metrics = list(rouge_scores.keys())
        values = list(rouge_scores.values())
        
        sns.barplot(x=metrics, y=values)
        plt.title('ROUGE Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'rouge_scores.png')
        plt.close()

        # 3. 의미적 유사도와 환각률 시각화
        plt.figure(figsize=(8, 6))
        metrics = ['Semantic Similarity', 'Hallucination Rate']
        values = [results['semantic_similarity'], results['hallucination_rate']]
        
        sns.barplot(x=metrics, y=values)
        plt.title('Semantic Analysis')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'semantic_analysis.png')
        plt.close()

        # 4. 리소스 사용량 시각화
        plt.figure(figsize=(8, 6))
        resource_metrics = results['resource_metrics']
        metrics = list(resource_metrics.keys())
        values = list(resource_metrics.values())
        
        sns.barplot(x=metrics, y=values)
        plt.title('Resource Usage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_usage.png')
        plt.close() 