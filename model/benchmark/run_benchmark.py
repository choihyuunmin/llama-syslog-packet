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
    model_name,
    num_runs=5,
    batch_size=100
):
    """Run benchmark evaluation multiple times."""
    all_results = []
    
    for run in range(num_runs):
        logging.info(f"Starting benchmark run {run + 1}/{num_runs} for model {model_name}")
        
        # Load predictions
        predictions = load_predictions(predictions_path)
        
        # Initialize evaluator
        evaluator = BenchmarkEvaluator(
            benchmark_path=benchmark_dir / 'benchmark.json',
            batch_size=batch_size
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(predictions)
        results['model_name'] = model_name
        all_results.append(results)
        
        # Save individual run results
        run_dir = benchmark_dir / f"run_{run + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        save_results(results, run_dir / f'{model_name}_results.json')
        evaluator.visualize_results(results, run_dir)
        
        logging.info(f"Completed benchmark run {run + 1}/{num_runs} for model {model_name}")
    
    return all_results

def evaluate_all_models(benchmark_dir, predictions_dir, num_runs=5, batch_size=100):
    """Evaluate all models in the predictions directory."""
    all_results = []
    predictions_dir = Path(predictions_dir)
    
    # Get all prediction files
    prediction_files = list(predictions_dir.glob('*_predictions.json'))
    
    for pred_file in prediction_files:
        model_name = pred_file.stem.replace('_predictions', '')
        logging.info(f"Evaluating model: {model_name}")
        
        model_results = run_benchmark(
            benchmark_dir=benchmark_dir,
            predictions_path=pred_file,
            model_name=model_name,
            num_runs=num_runs,
            batch_size=batch_size
        )
        all_results.extend(model_results)
    
    return all_results

def compare_models(all_results, output_dir):
    """Compare results from multiple models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by model
    model_results = {}
    for run_results in all_results:
        model_name = run_results['model_name']
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(run_results)
    
    # Calculate average metrics for each model
    model_metrics = {}
    for model_name, results in model_results.items():
        metrics = {
            'accuracy': np.mean([r['accuracy_metrics']['accuracy'] for r in results]),
            'f1_score': np.mean([r['accuracy_metrics']['f1_score'] for r in results]),
            'rouge_l': np.mean([r['rouge_scores']['rougeL'] for r in results]),
            'semantic_similarity': np.mean([r['semantic_similarity'] for r in results]),
            'hallucination_rate': np.mean([r['hallucination_rate'] for r in results]),
            'execution_time': np.mean([r['resource_metrics']['execution_time_ms'] for r in results]),
            'memory_usage': np.mean([r['resource_metrics']['memory_usage_mb'] for r in results])
        }
        model_metrics[model_name] = metrics
    
    # 1. Performance Metrics Comparison
    plt.figure(figsize=(15, 8))
    metrics = ['accuracy', 'f1_score', 'rouge_l', 'semantic_similarity', 'hallucination_rate']
    x = np.arange(len(metrics))
    width = 0.8 / len(model_metrics)
    
    for i, (model_name, metrics_dict) in enumerate(model_metrics.items()):
        values = [metrics_dict[m] for m in metrics]
        plt.bar(x + i * width, values, width, label=model_name)
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x + width * (len(model_metrics) - 1) / 2, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png')
    plt.close()
    
    # 2. Resource Usage Comparison
    plt.figure(figsize=(12, 6))
    resource_metrics = ['execution_time', 'memory_usage']
    x = np.arange(len(resource_metrics))
    width = 0.8 / len(model_metrics)
    
    for i, (model_name, metrics_dict) in enumerate(model_metrics.items()):
        values = [metrics_dict[m] for m in resource_metrics]
        plt.bar(x + i * width, values, width, label=model_name)
    
    plt.title('Resource Usage Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.xticks(x + width * (len(model_metrics) - 1) / 2, resource_metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'resource_comparison.png')
    plt.close()
    
    # Save comparison results
    with open(output_dir / 'model_comparison.json', 'w') as f:
        json.dump(model_metrics, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Run network security analysis benchmark')
    parser.add_argument('--generate', action='store_true', help='Generate benchmark dataset')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model predictions')
    parser.add_argument('--benchmark-dir', type=str, default='benchmark_data',
                      help='Directory for benchmark data')
    parser.add_argument('--predictions-dir', type=str, default='model_predictions',
                      help='Directory containing model predictions')
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
        generator = BenchmarkGenerator(benchmark_dir)
        generator.generate_all_benchmarks()
        logging.info("Benchmark dataset generation completed")
    
    if args.evaluate:
        logging.info("Starting benchmark evaluation for all models...")
        all_results = evaluate_all_models(
            benchmark_dir=benchmark_dir,
            predictions_dir=args.predictions_dir,
            num_runs=args.num_runs,
            batch_size=args.batch_size
        )
        
        # Analyze results from all runs
        analysis_dir = benchmark_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        compare_models(all_results, analysis_dir)
        
        logging.info("Benchmark evaluation completed for all models")

if __name__ == '__main__':
    main() 