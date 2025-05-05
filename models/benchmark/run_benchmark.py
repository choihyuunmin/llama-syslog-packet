import argparse
import json
from pathlib import Path
from generator import BenchmarkGenerator
from evaluator import BenchmarkEvaluator
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run network and system log analysis benchmarks')
    parser.add_argument('--generate', action='store_true', help='Generate benchmark datasets')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model predictions')
    parser.add_argument('--benchmark-dir', type=str, default='benchmark_data', help='Directory for benchmark data')
    parser.add_argument('--predictions', type=str, help='Path to model predictions file')
    args = parser.parse_args()
    
    logger = setup_logging()
    benchmark_dir = Path(args.benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    if args.generate:
        logger.info("Generating benchmark datasets...")
        generator = BenchmarkGenerator(str(benchmark_dir))
        generator.generate_all_benchmarks()
        logger.info("Benchmark datasets generated successfully")
        
    if args.evaluate:
        if not args.predictions:
            logger.error("Predictions file path is required for evaluation")
            return
            
        logger.info("Evaluating model predictions...")
        evaluator = BenchmarkEvaluator(str(benchmark_dir / 'combined_benchmark.json'))
        
        with open(args.predictions, 'r') as f:
            predictions = json.load(f)
            
        results = evaluator.run_evaluation(predictions)
        
        # Print evaluation results
        logger.info("\nEvaluation Results:")
        logger.info(f"Overall Score: {results['overall_score']:.2f}")
        logger.info(f"Accuracy: {results['accuracy']['accuracy']:.2f}")
        logger.info(f"Average Complexity: {results['complexity']['average_complexity']:.2f}")
        logger.info(f"Anomaly Detection F1: {results['anomaly_detection']['f1_score']:.2f}")
        logger.info(f"Pattern Recognition Score: {results['pattern_recognition']['average_pattern_score']:.2f}")
        logger.info(f"Correlation Analysis Score: {results['correlation_analysis']['average_correlation_score']:.2f}")
        
        # Save detailed results
        with open(benchmark_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {benchmark_dir / 'evaluation_results.json'}")

if __name__ == '__main__':
    main() 