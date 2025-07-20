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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import csv
import ast

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BenchmarkEvaluator:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.smoothing = SmoothingFunction().method1
        
    def calculate_accuracy(self, predictions):
        """Calculate accuracy based on exact matches and semantic similarity"""
        exact_matches = []
        semantic_scores = []
        
        for pred in predictions:
            # Exact match
            pred_text = pred['output'].lower().strip()
            gt_text = pred['expected_output'].lower().strip()
            exact_matches.append(1 if pred_text == gt_text else 0)
            
            # Semantic similarity
            pred_emb = self.similarity_model.encode(pred['output'], convert_to_tensor=True)
            gt_emb = self.similarity_model.encode(pred['expected_output'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_emb, gt_emb).item()
            semantic_scores.append(similarity)
        
        exact_accuracy = np.mean(exact_matches)
        semantic_accuracy = np.mean(semantic_scores)
        combined_accuracy = 0.3 * exact_accuracy + 0.7 * semantic_accuracy
        
        return {
            'exact_accuracy': exact_accuracy,
            'semantic_accuracy': semantic_accuracy,
            'accuracy': combined_accuracy
        }

    def _check_technical_appropriateness(self, prediction, ground_truth):
        """Check if technical terms are used appropriately"""
        # Simple heuristic: check if prediction contains similar technical patterns as ground truth
        pred_tech_patterns = re.findall(r'\b(?:0x[0-9a-fA-F]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|\d+:\d+|[A-Z]{2,})\b', prediction)
        gt_tech_patterns = re.findall(r'\b(?:0x[0-9a-fA-F]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|\d+:\d+|[A-Z]{2,})\b', ground_truth)
        
        if not gt_tech_patterns:
            return 1.0 if not pred_tech_patterns else 0.8
        
        pattern_overlap = len(set(pred_tech_patterns).intersection(set(gt_tech_patterns))) / len(set(gt_tech_patterns))
        return pattern_overlap

    def calculate_relevance(self, predictions):
        """Calculate response relevance using BLEU and ROUGE scores"""
        bleu_scores = []
        rouge_scores = []
        
        for pred in predictions:
            # Calculate BLEU score
            reference = [word_tokenize(pred['expected_output'].lower())]
            candidate = word_tokenize(pred['output'].lower())
            
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=self.smoothing)
            bleu_scores.append(bleu_score)
            
            # Calculate ROUGE score
            rouge_result = self.scorer.score(pred['expected_output'], pred['output'])
            rouge_scores.append(rouge_result['rougeL'].fmeasure)
        
        return {
            'bleu': np.mean(bleu_scores),
            'rouge_l': np.mean(rouge_scores),
            'relevance_score': 0.5 * np.mean(bleu_scores) + 0.5 * np.mean(rouge_scores)
        }

    def calculate_consistency(self, predictions):
        """Calculate consistency of responses (requires multiple runs)"""
        # For single run, calculate internal consistency (answer coherence)
        consistency_scores = []
        
        for pred in predictions:
            # Check internal consistency by analyzing answer structure
            response = pred['output']
            
            # Simple consistency check: look for contradictory statements
            sentences = response.split('.')
            if len(sentences) <= 1:
                consistency_scores.append(1.0)
                continue
            
            # Check semantic consistency between sentences
            sentence_embeddings = self.similarity_model.encode(sentences, convert_to_tensor=True)
            if len(sentence_embeddings) > 1:
                similarities = []
                for i in range(len(sentence_embeddings) - 1):
                    sim = util.pytorch_cos_sim(sentence_embeddings[i], sentence_embeddings[i+1]).item()
                    similarities.append(sim)
                consistency_scores.append(np.mean(similarities))
            else:
                consistency_scores.append(1.0)
        
        return np.mean(consistency_scores)

    def calculate_robustness(self, predictions):
        """Calculate robustness against noisy or unusual inputs"""
        robustness_scores = []
        
        for pred in predictions:
            response = pred['output']
            
            # Check if model provided a reasonable response (not empty, not error message)
            if not response.strip():
                robustness_scores.append(0.0)
                continue
            
            # Check for error indicators
            error_indicators = ['error', 'cannot', 'unable', 'fail', 'exception', 'invalid']
            error_count = sum(1 for indicator in error_indicators if indicator in response.lower())
            
            # Calculate robustness score (lower error count = higher robustness)
            if error_count == 0:
                robustness_score = 1.0
            else:
                robustness_score = max(0.0, 1.0 - (error_count * 0.2))
            
            robustness_scores.append(robustness_score)
        
        return np.mean(robustness_scores)

    def calculate_natural_language_quality(self, predictions):
        """Calculate natural language quality (fluency, grammar, readability)"""
        quality_scores = []
        
        for pred in predictions:
            response = pred['output']
            
            # Basic quality indicators
            score = 1.0
            
            # Check for basic grammar issues (simple heuristics)
            if not response.strip():
                quality_scores.append(0.0)
                continue
            
            # Check sentence structure
            sentences = response.split('.')
            valid_sentences = [s.strip() for s in sentences if s.strip()]
            
            if not valid_sentences:
                quality_scores.append(0.0)
                continue
            
            # Check for proper capitalization
            properly_capitalized = sum(1 for s in valid_sentences if s[0].isupper()) / len(valid_sentences)
            
            # Check for reasonable sentence length
            avg_sentence_length = np.mean([len(s.split()) for s in valid_sentences])
            length_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
            
            # Check for repetition
            words = response.lower().split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            repetition_score = min(1.0, unique_ratio + 0.3)
            
            quality_score = 0.4 * properly_capitalized + 0.3 * length_score + 0.3 * repetition_score
            quality_scores.append(quality_score)
        
        return np.mean(quality_scores)

    def calculate_bleu_score(self, predictions):
        """Calculate BLEU score for all predictions"""
        bleu_scores = []
        
        for pred in predictions:
            reference = [word_tokenize(pred['expected_output'].lower())]
            candidate = word_tokenize(pred['output'].lower())
            
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=self.smoothing)
            bleu_scores.append(bleu_score)
        
        return np.mean(bleu_scores)

    def calculate_f1_score(self, predictions):
        """Calculate F1 score between predicted and expected outputs"""
        f1_scores = []
        
        for pred in predictions:
            pred_tokens = set(pred['output'].lower().split())
            gt_tokens = set(pred['expected_output'].lower().split())
            
            true_positives = len(pred_tokens.intersection(gt_tokens))
            false_positives = len(pred_tokens - gt_tokens)
            false_negatives = len(gt_tokens - pred_tokens)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores)

    def calculate_rouge_scores(self, predictions):
        """Calculate ROUGE scores"""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred in predictions:
            score = self.scorer.score(pred['expected_output'], pred['output'])
            rouge1_scores.append(score['rouge1'].fmeasure)
            rouge2_scores.append(score['rouge2'].fmeasure)
            rougeL_scores.append(score['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rouge_l': np.mean(rougeL_scores)
        }

    def calculate_cosine_similarity(self, predictions):
        """Calculate cosine similarity between predictions and ground truth"""
        similarities = []
        for pred in predictions:
            pred_emb = self.similarity_model.encode(pred['output'], convert_to_tensor=True)
            gt_emb = self.similarity_model.encode(pred['expected_output'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(pred_emb, gt_emb).item()
            similarities.append(similarity)
        return np.mean(similarities)

    def evaluate(self, original_data, predictions, dataset_type):
        """Run comprehensive evaluation with all 8 metrics"""
        metrics_data = {}
        
        for model_name, model_predictions in predictions.items():
            try:
                # 1. Accuracy metrics
                accuracy_metrics = self.calculate_accuracy(model_predictions)
                
                # 2. Relevance (BLEU and ROUGE)
                relevance_metrics = self.calculate_relevance(model_predictions)
                
                # 4. Consistency
                consistency_score = self.calculate_consistency(model_predictions)
                
                # 5. Robustness
                robustness_score = self.calculate_robustness(model_predictions)
                
                # 6. Natural language quality
                nlq_score = self.calculate_natural_language_quality(model_predictions)
                
                # Additional metrics
                f1_score = self.calculate_f1_score(model_predictions)
                rouge_scores = self.calculate_rouge_scores(model_predictions)
                cosine_sim = self.calculate_cosine_similarity(model_predictions)
                
                # Combine all metrics
                metrics = {
                    # Core 8 metrics
                    'accuracy': accuracy_metrics['accuracy'],
                    'exact_accuracy': accuracy_metrics['exact_accuracy'],
                    'semantic_accuracy': accuracy_metrics['semantic_accuracy'],
                    'relevance_score': relevance_metrics['relevance_score'],
                    'consistency': consistency_score,
                    'robustness': robustness_score,
                    'natural_language_quality': nlq_score,
                    
                    # Additional detailed metrics
                    'bleu': relevance_metrics['bleu'],
                    'rouge1': rouge_scores['rouge1'],
                    'rouge2': rouge_scores['rouge2'],
                    'rouge_l': rouge_scores['rouge_l'],
                    'f1_score': f1_score,
                    'cosine_similarity': cosine_sim
                }
                
                metrics_data[model_name] = metrics
                
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {e}")
                continue

        return metrics_data