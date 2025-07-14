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
import csv

class BenchmarkEvaluator:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
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
        """Run evaluation and save results as CSV only"""
        metrics_data = {}
        for model_name, model_predictions in predictions.items():
            try:
                metrics = {
                    'f1_score': self.calculate_f1_score(model_predictions),
                    'rouge1': self.calculate_rouge_scores(model_predictions, original_data)[ 'rouge1'],
                    'rouge2': self.calculate_rouge_scores(model_predictions, original_data)[ 'rouge2'],
                    'rougeL': self.calculate_rouge_scores(model_predictions, original_data)[ 'rougeL'],
                    'cosine_similarity': self.calculate_cosine_similarity(model_predictions),
                    'numeric_mae': self.calculate_numeric_metrics(model_predictions, 'mae'),
                    'numeric_mse': self.calculate_numeric_metrics(model_predictions, 'mse')
                }
                metrics_data[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {e}")
                continue

        # Save metrics data as CSV file
        metrics_file = f'results/{dataset_type}_metrics.csv'
        Path('results').mkdir(parents=True, exist_ok=True)
        try:
            with open(metrics_file, 'w', encoding='utf-8', newline='') as csvfile:
                fieldnames = ['model'] + list(next(iter(metrics_data.values())).keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for model, metrics in metrics_data.items():
                    row = {'model': model}
                    row.update({k: round(v, 4) for k, v in metrics.items()})  # 소수점 4자리 반올림
                    writer.writerow(row)
        except IOError as e:
            self.logger.error(f"Failed to write CSV file {metrics_file}: {e}")
            raise

        return metrics_data