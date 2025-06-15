import json
import logging
import time
import psutil
import numpy as np
import re
from pathlib import Path
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'jpg',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class BenchmarkEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def calculate_regex_matching(self, predictions, ground_truth_data):
        matches = []
        for pred, gt_data in zip(predictions, ground_truth_data):
            if 'regex_pattern' in gt_data:
                pattern = gt_data['regex_pattern']
                match = bool(re.search(pattern, pred['output']))
                matches.append(1 if match else 0)
        return np.mean(matches) if matches else 0.0

    def calculate_numeric_metrics(self, predictions, ground_truth_data):
        numeric_pairs = []
        for pred, gt_data in zip(predictions, ground_truth_data):
            if 'numeric_value' in gt_data:
                pred_numbers = re.findall(r'-?\d*\.?\d+', pred['output'])
                if pred_numbers:
                    try:
                        pred_value = float(pred_numbers[0])
                        gt_value = float(gt_data['numeric_value'])
                        numeric_pairs.append((pred_value, gt_value))
                    except ValueError:
                        continue

        if not numeric_pairs:
            return {'mae': 0.0, 'mse': 0.0}

        pred_values, gt_values = zip(*numeric_pairs)
        return {
            'mae': mean_absolute_error(gt_values, pred_values),
            'mse': mean_squared_error(gt_values, pred_values)
        }

    def calculate_semantic_similarity(self, predictions, ground_truth):
        pred_embeddings = self.similarity_model.encode([p['output'] for p in predictions], convert_to_tensor=True)
        gt_embeddings = self.similarity_model.encode([g['output'] for g in ground_truth], convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(pred_embeddings, gt_embeddings)
        return similarities.diagonal().mean().item()

    def calculate_accuracy_metrics(self, predictions, ground_truth):
        pred_tokens = [set(p['output'].lower().split()) for p in predictions]
        gt_tokens = [set(g['output'].lower().split()) for g in ground_truth]
        
        exact_matches = [1 if p == g else 0 for p, g in zip(pred_tokens, gt_tokens)]
        return {
            'accuracy': np.mean(exact_matches),
            'precision': precision_score(exact_matches, [1] * len(exact_matches), average='weighted'),
            'recall': recall_score(exact_matches, [1] * len(exact_matches), average='weighted'),
            'f1_score': f1_score(exact_matches, [1] * len(exact_matches), average='weighted')
        }

    def calculate_rouge_scores(self, predictions, ground_truth):
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, gt in zip(predictions, ground_truth):
            score = self.scorer.score(pred['output'], gt['output'])
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
            
        return {k: np.mean(v) for k, v in scores.items()}

    def calculate_hallucination_rate(self, predictions, ground_truth):
        similarities = []
        for pred, gt in zip(predictions, ground_truth):
            pred_emb = self.similarity_model.encode(pred['output'], convert_to_tensor=True)
            gt_emb = self.similarity_model.encode(gt['output'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_emb, gt_emb).item()
            similarities.append(similarity)
        
        hallucination_rate = sum(1 for s in similarities if s < 0.5) / len(similarities)
        return hallucination_rate

    def visualize_results(self, results, output_dir, dataset_type):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 정확도 메트릭 시각화
        plt.figure(figsize=(10, 6))
        accuracy_metrics = results['accuracy_metrics']
        metrics = list(accuracy_metrics.keys())
        values = list(accuracy_metrics.values())
        
        ax = sns.barplot(x=metrics, y=values, palette='viridis')
        plt.title(f'Accuracy Metrics Comparison - {dataset_type}', pad=20)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_type}_accuracy_metrics.jpg')
        plt.close()

        # 2. ROUGE 점수 시각화
        plt.figure(figsize=(10, 6))
        rouge_scores = results['rouge_scores']
        metrics = list(rouge_scores.keys())
        values = list(rouge_scores.values())
        
        ax = sns.barplot(x=metrics, y=values, palette='muted')
        plt.title(f'ROUGE Scores Comparison - {dataset_type}', pad=20)
        plt.xlabel('ROUGE Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_type}_rouge_scores.jpg')
        plt.close()

        # 3. 의미적 유사도와 환각률 시각화
        plt.figure(figsize=(10, 6))
        metrics = ['Semantic Similarity', 'Hallucination Rate', 'Regex Matching']
        values = [
            results['semantic_similarity'],
            results['hallucination_rate'],
            results['regex_matching_score']
        ]
        
        ax = sns.barplot(x=metrics, y=values, palette='Set2')
        plt.title(f'Semantic Analysis Metrics - {dataset_type}', pad=20)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_type}_semantic_analysis.jpg')
        plt.close()

        # 4. 수치 예측 메트릭 시각화
        plt.figure(figsize=(10, 6))
        numeric_metrics = results['numeric_metrics']
        metrics = list(numeric_metrics.keys())
        values = list(numeric_metrics.values())
        
        ax = sns.barplot(x=metrics, y=values, palette='coolwarm')
        plt.title(f'Numeric Prediction Error Metrics - {dataset_type}', pad=20)
        plt.xlabel('Error Metrics')
        plt.ylabel('Error Value')
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_type}_numeric_metrics.jpg')
        plt.close()

        # 5. 리소스 사용량 시각화
        plt.figure(figsize=(10, 6))
        resource_metrics = results['resource_metrics']
        metrics = list(resource_metrics.keys())
        values = list(resource_metrics.values())
        
        ax = sns.barplot(x=metrics, y=values, palette='YlOrRd')
        plt.title(f'Resource Usage Metrics - {dataset_type}', pad=20)
        plt.xlabel('Resource Metrics')
        plt.ylabel('Value')
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.1f}', ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_type}_resource_usage.jpg')
        plt.close()

        # 6. 통합 메트릭 시각화
        plt.figure(figsize=(12, 8))
        all_metrics = {
            'Accuracy': results['accuracy_metrics']['accuracy'],
            'F1-Score': results['accuracy_metrics']['f1_score'],
            'ROUGE-L': results['rouge_scores']['rougeL'],
            'Semantic Similarity': results['semantic_similarity'],
            'Regex Matching': results['regex_matching_score']
        }
        
        metrics = list(all_metrics.keys())
        values = list(all_metrics.values())
        
        ax = sns.barplot(x=metrics, y=values, palette='viridis')
        plt.title(f'Overall Performance Metrics - {dataset_type}', pad=20)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_type}_overall_metrics.jpg')
        plt.close()

    def visualize_metrics(self, metrics_data, dataset_type):
        """Visualize metrics for all models in a single graph"""
        plt.figure(figsize=(15, 10))
        
        # Prepare data for each model
        models = list(metrics_data.keys())
        metrics = ['f1_score', 'rouge1', 'rouge2', 'rougeL', 'cosine_similarity']
        
        # Compare model performance for each metric
        x = np.arange(len(metrics))
        width = 0.8 / len(models)  # Adjust bar width
        
        for i, model in enumerate(models):
            values = [metrics_data[model][metric] for metric in metrics]
            plt.bar(x + i*width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Model Performance Comparison - {dataset_type} Dataset')
        plt.xticks(x + width*(len(models)-1)/2, metrics, rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save results
        plt.savefig(f'results/{dataset_type}_model_comparison.jpg', bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_regex_metrics(self, metrics_data, dataset_type):
        """Visualize regex matching results for all models in a single graph"""
        plt.figure(figsize=(15, 10))
        
        # Prepare data for each model
        models = list(metrics_data.keys())
        metrics = ['regex_match', 'numeric_mae', 'numeric_mse']
        
        # Compare model performance for each metric
        x = np.arange(len(metrics))
        width = 0.8 / len(models)  # Adjust bar width
        
        for i, model in enumerate(models):
            values = [metrics_data[model][metric] for metric in metrics]
            plt.bar(x + i*width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Regex/Numeric Matching Performance - {dataset_type} Dataset')
        plt.xticks(x + width*(len(models)-1)/2, metrics, rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save results
        plt.savefig(f'results/{dataset_type}_regex_comparison.jpg', bbox_inches='tight', dpi=300)
        plt.close()

    def calculate_f1_score(self, predictions):
        """Calculate F1 score between predicted and expected outputs"""
        f1_scores = []
        
        for pred in predictions:
            # Tokenize predicted and expected outputs
            pred_tokens = set(pred['output'].lower().split())
            gt_tokens = set(pred['expected_output'].lower().split())
            
            # Calculate true positives, false positives, and false negatives
            true_positives = len(pred_tokens.intersection(gt_tokens))
            false_positives = len(pred_tokens - gt_tokens)
            false_negatives = len(gt_tokens - pred_tokens)
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        # Return average F1 score
        return np.mean(f1_scores)

    def calculate_rouge_score(self, predictions, rouge_type):
        """ROUGE 점수 계산"""
        scores = []
        for pred in predictions:
            score = self.scorer.score(pred['output'], pred['expected_output'])
            scores.append(score[rouge_type].fmeasure)
        return np.mean(scores)

    def calculate_cosine_similarity(self, predictions):
        """코사인 유사도 계산"""
        similarities = []
        for pred in predictions:
            pred_emb = self.similarity_model.encode(pred['output'], convert_to_tensor=True)
            gt_emb = self.similarity_model.encode(pred['expected_output'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_emb, gt_emb).item()
            similarities.append(similarity)
        return np.mean(similarities)

    def calculate_regex_match(self, predictions):
        """정규식 매칭 점수 계산"""
        matches = []
        for pred in predictions:
            # 숫자 패턴 매칭
            pred_numbers = re.findall(r'-?\d*\.?\d+', pred['output'])
            gt_numbers = re.findall(r'-?\d*\.?\d+', pred['expected_output'])
            
            if pred_numbers and gt_numbers:
                try:
                    pred_value = float(pred_numbers[0])
                    gt_value = float(gt_numbers[0])
                    matches.append(1 if abs(pred_value - gt_value) < 0.01 else 0)
                except ValueError:
                    matches.append(0)
            else:
                # 일반 텍스트 매칭
                pred_text = pred['output'].lower()
                gt_text = pred['expected_output'].lower()
                matches.append(1 if pred_text == gt_text else 0)
        
        return np.mean(matches)

    def calculate_numeric_metrics(self, predictions, metric_type):
        """수치 예측 메트릭 계산 (MAE/MSE)"""
        numeric_pairs = []
        for pred in predictions:
            pred_numbers = re.findall(r'-?\d*\.?\d+', pred['output'])
            gt_numbers = re.findall(r'-?\d*\.?\d+', pred['expected_output'])
            
            if pred_numbers and gt_numbers:
                try:
                    pred_value = float(pred_numbers[0])
                    gt_value = float(gt_numbers[0])
                    numeric_pairs.append((pred_value, gt_value))
                except ValueError:
                    continue

        if not numeric_pairs:
            return 0.0

        pred_values, gt_values = zip(*numeric_pairs)
        if metric_type == 'mae':
            return mean_absolute_error(gt_values, pred_values)
        else:  # mse
            return mean_squared_error(gt_values, pred_values)

    def calculate_code_execution_score(self, predictions):
        """Calculate pass@1 score for code generation tasks"""
        pass_count = 0
        total_count = 0
        
        for pred in predictions:
            try:
                # Check if the output is valid Python code
                code = pred['output']
                if not code.strip():
                    continue
                
                # Try to compile the code
                compile(code, '<string>', 'exec')
                
                # If compilation succeeds, increment pass count
                pass_count += 1
            except (SyntaxError, ValueError, TypeError):
                pass
            finally:
                total_count += 1
        
        # Calculate pass@1 score
        return pass_count / total_count if total_count > 0 else 0

    def evaluate(self, original_data, predictions, dataset_type):
        """평가 실행 및 결과 저장"""
        metrics_data = {}
        
        for model_name, model_predictions in predictions.items():
            metrics = {
                'f1_score': self.calculate_f1_score(model_predictions),
                'rouge1': self.calculate_rouge_score(model_predictions, 'rouge1'),
                'rouge2': self.calculate_rouge_score(model_predictions, 'rouge2'),
                'rougeL': self.calculate_rouge_score(model_predictions, 'rougeL'),
                'cosine_similarity': self.calculate_cosine_similarity(model_predictions),
                'regex_match': self.calculate_regex_match(model_predictions),
                'numeric_mae': self.calculate_numeric_metrics(model_predictions, 'mae'),
                'numeric_mse': self.calculate_numeric_metrics(model_predictions, 'mse'),
                'code_pass@1': self.calculate_code_execution_score(model_predictions)
            }
            metrics_data[model_name] = metrics
        
        # 메트릭 데이터를 JSON 파일로 저장
        metrics_file = f'results/{dataset_type}_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        # 시각화
        self.visualize_metrics(metrics_data, dataset_type)
        self.visualize_regex_metrics(metrics_data, dataset_type)
        self.visualize_code_metrics(metrics_data, dataset_type)
        
        return metrics_data

    def visualize_code_metrics(self, metrics_data, dataset_type):
        """Visualize code generation metrics for all models"""
        plt.figure(figsize=(15, 10))
        
        # Prepare data for each model
        models = list(metrics_data.keys())
        metrics = ['code_pass@1']
        
        # Compare model performance for each metric
        x = np.arange(len(metrics))
        width = 0.8 / len(models)  # Adjust bar width
        
        for i, model in enumerate(models):
            values = [metrics_data[model][metric] for metric in metrics]
            plt.bar(x + i*width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Code Generation Performance - {dataset_type} Dataset')
        plt.xticks(x + width*(len(models)-1)/2, metrics, rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save results
        plt.savefig(f'results/{dataset_type}_code_comparison.jpg', bbox_inches='tight', dpi=300)
        plt.close() 