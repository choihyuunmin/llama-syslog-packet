import argparse
import json
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from generator import BenchmarkGenerator
from evaluator import BenchmarkEvaluator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_predictions(predictions_path):
    """Load model predictions from JSON file."""
    with open(predictions_path, 'r') as f:
        return json.load(f)

def save_results(results, output_path):
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def run_benchmark(
    benchmark_dir,
    predictions_path,
    num_runs=5,
    batch_size=100
):
    """Run benchmark evaluation multiple times."""
    all_results = []
    
    for run in range(num_runs):
        logging.info(f"Starting benchmark run {run + 1}/{num_runs}")
        
        # Load predictions
        predictions = load_predictions(predictions_path)
        
        # Initialize evaluator
        evaluator = BenchmarkEvaluator(
            benchmark_path=benchmark_dir / 'benchmark.json',
            batch_size=batch_size
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(predictions)
        all_results.append(results)
        
        # Save individual run results
        run_dir = benchmark_dir / f"run_{run + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        save_results(results, run_dir / 'results.json')
        evaluator.visualize_results(results, run_dir)
        
        logging.info(f"Completed benchmark run {run + 1}/{num_runs}")
    
    return all_results

def analyze_runs(all_results, output_dir):
    """Analyze results from multiple runs."""
    metrics = [
        'accuracy',
        'complexity',
        'anomaly_detection',
        'pattern_recognition',
        'correlation_analysis'
    ]
    
    # 1. Overall Score Trend
    plt.figure(figsize=(12, 6))
    scores = [r['overall_score'] for r in all_results]
    plt.plot(range(1, len(scores) + 1), scores, marker='o')
    plt.title('Overall Score Trend Across Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Overall Score')
    plt.grid(True)
    plt.savefig(output_dir / 'overall_score_trend.png')
    plt.close()
    
    # 2. Metric Stability
    plt.figure(figsize=(15, 8))
    metric_data = {
        'accuracy': [r['accuracy']['accuracy'] for r in all_results],
        'complexity': [r['complexity']['average_complexity'] / 100 for r in all_results],
        'anomaly_detection': [r['anomaly_detection']['f1_score'] for r in all_results],
        'pattern_recognition': [r['pattern_recognition']['average_pattern_score'] for r in all_results],
        'correlation_analysis': [r['correlation_analysis']['average_correlation_score'] for r in all_results]
    }
    
    for metric, values in metric_data.items():
        plt.plot(range(1, len(values) + 1), values, marker='o', label=metric)
    
    plt.title('Metric Stability Across Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'metric_stability.png')
    plt.close()
    
    # 3. Statistical Summary
    summary = {
        'overall_score': {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    }
    
    for metric in metrics:
        values = [r[metric]['average_score'] for r in all_results]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Save summary
    with open(output_dir / 'run_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Run network security analysis benchmark')
    parser.add_argument('--generate', action='store_true', help='Generate benchmark dataset')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model predictions')
    parser.add_argument('--benchmark-dir', type=str, default='benchmark_data',
                      help='Directory for benchmark data')
    parser.add_argument('--predictions', type=str, help='Path to model predictions file')
    parser.add_argument('--num-runs', type=int, default=5,
                      help='Number of benchmark runs')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for evaluation')
    
    args = parser.parse_args()
    setup_logging()
    
    benchmark_dir = Path(args.benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    if args.generate:
        logging.info("Generating benchmark dataset...")
        generator = BenchmarkGenerator()
        generator.generate_all_benchmarks(benchmark_dir)
        logging.info("Benchmark dataset generation completed")
    
    if args.evaluate:
        if not args.predictions:
            logging.error("Predictions file path is required for evaluation")
            return
            
        logging.info("Starting benchmark evaluation...")
        all_results = run_benchmark(
            benchmark_dir=benchmark_dir,
            predictions_path=args.predictions,
            num_runs=args.num_runs,
            batch_size=args.batch_size
        )
        
        # Analyze results from all runs
        analysis_dir = benchmark_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analyze_runs(all_results, analysis_dir)
        
        logging.info("Benchmark evaluation completed")

if __name__ == '__main__':
    main() 