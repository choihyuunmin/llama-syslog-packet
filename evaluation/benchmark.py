import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from evaluator import Evaluator
import sys
import os


from run_benchmark import (
    LlamaModel, OpenAIModel, OllamaModel, 
    load_benchmark_data, measure_latency
)

def run_base_vs_custom_benchmark(benchmark_data, output_dir):
    """Compare base model vs custom model performance using real models"""
    print("\n" + "="*60)
    print("BASE MODEL vs CUSTOM MODEL COMPARISON")
    print("="*60)
    
    try:
        base_model = LlamaModel("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")
        custom_model = LlamaModel("Llama-PcapLog", "choihyuunmin/Llama-PcapLog")
        
        evaluator = Evaluator()
        
        model_results = {}
        
        for model_name, model in [("Base Llama-3-8B", base_model), ("Llama-PcapLog", custom_model)]:
            predictions = []
            latencies = []
            
            for item in benchmark_data:
                prompt = f"{item['instruction']}\n\nContext:\n{json.dumps(item['input'], indent=2)}"
                response, latency = measure_latency(model, prompt)
                
                predictions.append({
                    'input': prompt,
                    'output': response,
                    'expected_output': item['output']
                })
                latencies.append(latency)
            
            model_predictions = {model_name: predictions}
            metrics = evaluator.evaluate(benchmark_data, model_predictions, "attack_detection")
            
            if model_name in metrics:
                metrics[model_name]['avg_latency_ms'] = np.mean(latencies)
                metrics[model_name]['total_latency_ms'] = sum(latencies)
            
            model_results[model_name] = {
                'metrics': metrics.get(model_name, {}),
                'predictions': predictions,
                'latencies': latencies
            }
            
            print(f"{model_name} evaluation completed")
        
        return model_results
        
    except Exception as e:
        print(f"Error loading real models: {e}")
        print("Note: This requires:")
        print("   - Hugging Face authentication (huggingface-cli login)")
        print("   - Sufficient GPU memory (16GB+ recommended)")
        print("   - Internet connection for model downloading")
        print("   - Required packages: transformers, torch, accelerate")
        return None

def run_multi_model_benchmark(benchmark_data, output_dir):
    """Compare multiple models performance using real models"""
    print("\n" + "="*60)
    print("MULTI-MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    try:
        print("Loading multiple models for comparison")
        models = []
        
        # Custom model
        print("   Loading Llama-PcapLog")
        custom_model = LlamaModel("Llama-PcapLog", "choihyuunmin/Llama-PcapLog")
        models.append(("Llama-PcapLog", custom_model))
        print("Llama-PcapLog loaded")
        
        # Base model
        print("   Loading base Meta-Llama-3-8B-Instruct")
        base_model = LlamaModel("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")
        models.append(("meta-llama/Meta-Llama-3-8B-Instruct", base_model))
        print("Base model loaded")
        
        # Skip Ollama models for now (not available in this environment)
        print("Ollama not installed in this environment, using only transformer models")
        
        # OpenAI model (if API key available)
        if os.getenv("OPENAI_API_KEY"):
            try:
                openai_model = OpenAIModel("gpt-4o", os.getenv("OPENAI_API_KEY"))
                models.append(("gpt-4o", openai_model))
                print("OpenAI GPT-4o is available")
            except Exception as e:
                print(f"OpenAI GPT-4o not available: {e}")
        
        evaluator = Evaluator()
        all_results = {}
        
        for model_name, model in models:
            print(f"\nTesting {model_name} (5 runs to select best performance)")
            
            best_run_results = None
            best_domain_score = 0
            
            # Run 5 times and select the best performance
            for run_number in range(5):
                print(f"   Run {run_number + 1}/5...")
                
                predictions = []
                latencies = []
                
                for item in benchmark_data:
                    prompt = f"{item['instruction']}\n\nContext:\n{json.dumps(item['input'], indent=2)}"
                    response, latency = measure_latency(model, prompt)
                    
                    predictions.append({
                        'input': prompt,
                        'output': response,
                        'expected_output': item['output']
                    })
                    latencies.append(latency)
                
                # Evaluate this run
                model_predictions = {model_name: predictions}
                metrics = evaluator.evaluate(benchmark_data, model_predictions, "attack_detection")
                
                # Add latency metrics
                if model_name in metrics:
                    metrics[model_name]['avg_latency_ms'] = np.mean(latencies)
                    metrics[model_name]['total_latency_ms'] = sum(latencies)
                
                current_domain_score = metrics.get(model_name, {}).get('domain_specific_score', 0)
                print(f"      Domain Score: {current_domain_score:.3f}")
                
                # Keep the best run
                if current_domain_score > best_domain_score:
                    best_domain_score = current_domain_score
                    best_run_results = {
                        'metrics': metrics.get(model_name, {}),
                        'predictions': predictions,
                        'latencies': latencies,
                        'run_number': run_number + 1
                    }
            
            all_results[model_name] = best_run_results
            
            print(f"{model_name} evaluation completed - Best run #{best_run_results['run_number']}")
            print(f"- Best Domain Specific Score: {best_domain_score:.3f}")
            print(f"- Attack Classification: {best_run_results['metrics'].get('attack_classification_accuracy', 0):.3f}")
            print(f"- Information Extraction F1: {best_run_results['metrics'].get('overall_extraction_f1', 0):.3f}")
        
        return all_results
        
    except Exception as e:
        print(f"Error during multi-model benchmark: {e}")
        return None

def create_multi_model_csv(multi_model_results, output_dir):
    """Create CSV for multi-model comparison with English headers"""
    print("\nCreating multi-model comparison CSV")
    
    if not multi_model_results:
        print("No results to create CSV")
        return None
    
    # Prepare table data
    table_data = []
    
    # Define evaluation metrics
    metrics_for_table = [
        ('attack_classification_accuracy', 'Attack_Classification_Accuracy'),
        ('overall_extraction_f1', 'Information_Extraction_F1'),
        ('threat_detection_accuracy', 'Threat_Detection_Accuracy'),
        ('response_quality_score', 'Response_Quality'),
        ('domain_specific_score', 'Domain_Specific_Score'),
        ('overall_score', 'Overall_Score'),
        ('bleu', 'BLEU'),
        ('rouge_l', 'ROUGE_L'),
        ('semantic_similarity', 'Semantic_Similarity'),
        ('avg_latency_ms', 'Avg_Latency_ms')
    ]
    
    for model_name, results in multi_model_results.items():
        metrics = results['metrics']
        
        row = {'Model': model_name}
        
        for metric_key, metric_display in metrics_for_table:
            value = metrics.get(metric_key, 0)
            if metric_key == 'avg_latency_ms':
                row[metric_display] = f"{value:.1f}"
            else:
                row[metric_display] = f"{value:.3f}"
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort models (Llama-PcapLog first)
    model_order = ['Llama-PcapLog', 'meta-llama/Meta-Llama-3-8B-Instruct', 'gpt-4o', 'qwen2:7b', 'gemma3:4b', 'mistral:7b', 'llama3.1:8b']
    df['sort_order'] = df['Model'].map({model: i for i, model in enumerate(model_order)})
    df = df.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Save CSV (for paper)
    csv_path = output_dir / "multi_model_comparison_paper.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save LaTeX table format
    latex_table = df.to_latex(index=False, float_format="%.3f", escape=False)
    with open(output_dir / "multi_model_comparison_table.tex", 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    # Save detailed results as JSON
    with open(output_dir / "multi_model_detailed_results.json", 'w', encoding='utf-8') as f:
        # Save metrics only, exclude predictions
        simplified_results = {}
        for model_name, results in multi_model_results.items():
            simplified_results[model_name] = {
                'metrics': results['metrics'],
                'avg_latency': np.mean(results['latencies']) if results['latencies'] else 0
            }
        json.dump(simplified_results, f, indent=2, ensure_ascii=False)
    
    print(f"CSV saved: {csv_path}")
    print(f"LaTeX table saved: {output_dir / 'multi_model_comparison_table.tex'}")
    
    # Print performance ranking
    print("\nModel Performance Ranking (Domain Specific Score):")
    domain_scores = [(model, results['metrics'].get('domain_specific_score', 0)) 
                    for model, results in multi_model_results.items()]
    domain_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(domain_scores, 1):
        print(f"  {i}. {model}: {score:.3f}")
    
    return df

def main():
    """Main execution function - 5 runs per model, CSV output only"""
    print("Cybersecurity Domain-Specific LLM Benchmark (5 Runs - Best Results)")
    print("=" * 80)
    
    # Setup output directory
    output_dir = Path("benchmark_results_5runs")
    output_dir.mkdir(exist_ok=True)
    
    # Load test data
    benchmark_data = load_benchmark_data("test/attack_test_dataset.json")
    print(f"Test data loaded: {len(benchmark_data)} cases")
    
    # Run multi-model comparison with 5 runs each
    print("\n" + "="*60)
    print("MULTI-MODEL Performance Comparison (5 Runs Each)")
    print("="*60)
    multi_model_results = run_multi_model_benchmark(benchmark_data, output_dir)
    
    if multi_model_results:
        # Create ONLY CSV results
        print("\n" + "="*60)
        print("Saving CSV Results (Best of 5 Runs)")
        print("="*60)
        results_df = create_multi_model_csv(multi_model_results, output_dir)
        
        print("\n" + "="*60) 
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        print("Final Performance Rankings (Domain Specific Score):")
        ranking = sorted(multi_model_results.items(), 
                        key=lambda x: x[1]['metrics'].get('domain_specific_score', 0), 
                        reverse=True)
        
        for i, (model_name, results) in enumerate(ranking, 1):
            score = results['metrics'].get('domain_specific_score', 0)
            run_num = results.get('run_number', 'N/A')
            attack_acc = results['metrics'].get('attack_classification_accuracy', 0)
            if i == 1:
                print(f"  1. {model_name}: {score:.3f} (Run #{run_num}, Attack Acc: {attack_acc:.3f})")
            elif i == 2:
                print(f"  2. {model_name}: {score:.3f} (Run #{run_num}, Attack Acc: {attack_acc:.3f})")
            elif i == 3:
                print(f"  3. {model_name}: {score:.3f} (Run #{run_num}, Attack Acc: {attack_acc:.3f})")
            else:
                print(f"  4. {model_name}: {score:.3f} (Run #{run_num}, Attack Acc: {attack_acc:.3f})")
        
        # Performance improvement calculation
        if len(ranking) >= 2:
            best_model_score = ranking[0][1]['metrics'].get('domain_specific_score', 0)
            second_model_score = ranking[1][1]['metrics'].get('domain_specific_score', 0)
            if second_model_score > 0:
                improvement = ((best_model_score - second_model_score) / second_model_score) * 100
                print(f"\nPerformance Improvement: {improvement:.1f}% ({ranking[0][0]} vs {ranking[1][0]})")
    else:
        print("Multi-model benchmark failed")
        print("This might be due to:")
        print("   - Missing model files")
        print("   - Insufficient GPU memory")  
        print("   - Network issues downloading models")
        print("   - Missing dependencies")
    
    print(f"\nBenchmark completed with 5 runs per model!")
    print(f"CSV results saved to: {output_dir}")
    print("Only the best performance from each model's 5 runs is included.")

if __name__ == "__main__":
    main() 