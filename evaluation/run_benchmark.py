import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
from evaluator import BenchmarkEvaluator
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import ollama
import ast
from collections import defaultdict
import csv
import time

class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def predict(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement predict method")

class LlamaModel(BaseModel):
    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            device_map="auto"
        )
        if self.device == "mps":
            self.model = self.model.to("mps")
    
    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = response[len(prompt):].strip()
        return response

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def predict(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Openai Error: {str(e)}")
            return ""

class OllamaModel(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    def predict(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logging.error(f"Ollama Error: {str(e)}")
            return ""

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_benchmark_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def measure_latency(model, prompt):
    """Measure response latency for a single prediction"""
    start_time = time.time()
    response = model.predict(prompt)
    end_time = time.time()
    return response, (end_time - start_time) * 1000  # Convert to milliseconds

def run_base_vs_custom_comparison(benchmark_dir, output_dir):
    """Compare base llama3.1 model with custom Llama-PcapLog model"""
    print("\n" + "="*60)
    print("BASE MODEL vs CUSTOM MODEL COMPARISON")
    print("="*60)
    
    # Define models for comparison
    base_model = LlamaModel("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")
    custom_model = LlamaModel("Llama-PcapLog", "choihyuunmin/Llama-PcapLog")
    
    models = [base_model, custom_model]
    evaluator = BenchmarkEvaluator()
    
    comparison_results = {}
    
    # Test on both pcap and syslog datasets
    for dataset_type in ['pcap', 'syslog']:
        print(f"\n--- {dataset_type.upper()} Dataset Evaluation ---")
        
        benchmark_file = benchmark_dir / f'{dataset_type}_test.json'
        benchmark_data = load_benchmark_data(benchmark_file)
        
        # Group by type
        type_groups = {}
        for item in benchmark_data:
            t = item['type']
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(item)
        
        dataset_results = {}
        
        for type_name, items in type_groups.items():
            print(f"\nEvaluating {type_name} tasks...")
            
            type_results = {}
            
            # Collect predictions and latencies
            for model in models:
                print(f"Testing {model.model_name}...")
                
                predictions = []
                latencies = []
                
                for item in items:
                    prompt = f"{item['instruction']}\n\nContext:\n{item['input']}"
                    response, latency = measure_latency(model, prompt)
                    
                    predictions.append({
                        'input': prompt,
                        'output': response,
                        'expected_output': item['output']
                    })
                    latencies.append(latency)
                
                # Evaluate metrics
                model_predictions = {model.model_name: predictions}
                metrics = evaluator.evaluate(items, model_predictions, dataset_type)
                
                # Add latency metrics
                metrics[model.model_name]['avg_latency_ms'] = sum(latencies) / len(latencies)
                metrics[model.model_name]['total_latency_ms'] = sum(latencies)
                
                type_results[model.model_name] = {
                    'metrics': metrics[model.model_name],
                    'predictions': predictions,
                    'latencies': latencies
                }
            
            dataset_results[type_name] = type_results
        
        comparison_results[dataset_type] = dataset_results
    
    # Save comparison results
    save_comparison_results(comparison_results, output_dir, "base_vs_custom")
    
    # Generate comparison report
    generate_comparison_report(comparison_results, output_dir, "base_vs_custom", 
                             base_model.model_name, custom_model.model_name)
    
    return comparison_results

def run_custom_vs_general_comparison(benchmark_dir, output_dir):
    """Compare custom model with general LLMs"""
    print("\n" + "="*60)
    print("CUSTOM MODEL vs GENERAL LLMs COMPARISON")
    print("="*60)
    
    # Define models for comparison
    custom_model = LlamaModel("Llama-PcapLog", "choihyuunmin/Llama-PcapLog")
    general_models = [
        OllamaModel("qwen2:7b"),
        OllamaModel("gemma3:4b"),
        OllamaModel("mistral:7b"),
        OllamaModel("llama3.1:8b")
    ]
    
    # Add OpenAI model if API key is available
    if os.getenv("OPENAI_API_KEY"):
        general_models.append(OpenAIModel("gpt-4o", os.getenv("OPENAI_API_KEY")))
    
    models = [custom_model] + general_models
    evaluator = BenchmarkEvaluator()
    
    comparison_results = {}
    
    # Test on both pcap and syslog datasets
    for dataset_type in ['pcap', 'syslog']:
        print(f"\n--- {dataset_type.upper()} Dataset Evaluation ---")
        
        benchmark_file = benchmark_dir / f'{dataset_type}_test.json'
        benchmark_data = load_benchmark_data(benchmark_file)
        
        # Group by type
        type_groups = {}
        for item in benchmark_data:
            t = item['type']
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(item)
        
        dataset_results = {}
        
        for type_name, items in type_groups.items():
            print(f"\nEvaluating {type_name} tasks...")
            
            type_results = {}
            
            # Collect predictions and latencies
            for model in models:
                print(f"Testing {model.model_name}...")
                
                predictions = []
                latencies = []
                
                for item in items:
                    prompt = f"{item['instruction']}\n\nContext:\n{item['input']}"
                    response, latency = measure_latency(model, prompt)
                    
                    predictions.append({
                        'input': prompt,
                        'output': response,
                        'expected_output': item['output']
                    })
                    latencies.append(latency)
                
                # Evaluate metrics
                model_predictions = {model.model_name: predictions}
                metrics = evaluator.evaluate(items, model_predictions, dataset_type)
                
                # Add latency metrics
                if model.model_name in metrics:
                    metrics[model.model_name]['avg_latency_ms'] = sum(latencies) / len(latencies) if latencies else 0
                    metrics[model.model_name]['total_latency_ms'] = sum(latencies)
                
                type_results[model.model_name] = {
                    'metrics': metrics.get(model.model_name, {}),
                    'predictions': predictions,
                    'latencies': latencies
                }
            
            dataset_results[type_name] = type_results
        
        comparison_results[dataset_type] = dataset_results
    
    # Save comparison results
    save_comparison_results(comparison_results, output_dir, "custom_vs_general")
    
    # Generate comparison report
    generate_comparison_report(comparison_results, output_dir, "custom_vs_general", 
                             custom_model.model_name, "General LLMs")

    # Run code generation benchmark
    print("\n" + "="*60)
    print("CODE GENERATION BENCHMARK (pass@k)")
    print("="*60)
    run_code_generation_passk_benchmark(models, k=3, output_dir=str(output_dir))

    return comparison_results

def save_comparison_results(results, output_dir, comparison_type):
    """Save detailed comparison results to JSON and CSV files"""
    # Save detailed JSON results
    json_path = output_dir / f"{comparison_type}_detailed_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create summary CSV
    csv_data = []
    for dataset_type, dataset_results in results.items():
        for type_name, type_results in dataset_results.items():
            for model_name, model_data in type_results.items():
                metrics = model_data['metrics']
                row = {
                    'dataset': dataset_type,
                    'task_type': type_name,
                    'model': model_name,
                    **metrics
                }
                csv_data.append(row)
    
    csv_path = output_dir / f"{comparison_type}_summary.csv"
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

def generate_comparison_report(results, output_dir, comparison_type, model1_name, model2_name):
    """Generate a comprehensive comparison report"""
    report_path = output_dir / f"{comparison_type}_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {comparison_type.replace('_', ' ').title()} Report\n\n")
        f.write(f"Comparison between **{model1_name}** and **{model2_name}**\n\n")
        
        # Overall summary
        f.write("## Overall Performance Summary\n\n")
        
        for dataset_type, dataset_results in results.items():
            f.write(f"### {dataset_type.upper()} Dataset\n\n")
            
            for type_name, type_results in dataset_results.items():
                f.write(f"#### {type_name} Tasks\n\n")
                f.write("| Model | Accuracy | BLEU | ROUGE-L | Avg Latency (ms) |\n")
                f.write("|-------|----------|------|---------|------------------|\n")
                
                for model_name, model_data in type_results.items():
                    metrics = model_data['metrics']
                    f.write(f"| {model_name} | {metrics.get('accuracy', 'N/A'):.4f} | "
                           f"{metrics.get('bleu', 'N/A'):.4f} | {metrics.get('rouge_l', 'N/A'):.4f} | "
                           f"{metrics.get('avg_latency_ms', 'N/A'):.2f} |\n")
                
                f.write("\n")
        
        # Performance improvement analysis (for base vs custom)
        if comparison_type == "base_vs_custom":
            f.write("## Performance Improvement Analysis\n\n")
            
            for dataset_type, dataset_results in results.items():
                f.write(f"### {dataset_type.upper()} Dataset Improvements\n\n")
                
                for type_name, type_results in dataset_results.items():
                    models_list = list(type_results.keys())
                    if len(models_list) >= 2:
                        base_metrics = type_results[models_list[0]]['metrics']
                        custom_metrics = type_results[models_list[1]]['metrics']
                        
                        f.write(f"#### {type_name} Tasks\n\n")
                        
                        for metric in ['accuracy', 'bleu', 'rouge_l']:
                            if metric in base_metrics and metric in custom_metrics:
                                base_val = base_metrics[metric]
                                custom_val = custom_metrics[metric]
                                improvement = ((custom_val - base_val) / base_val * 100) if base_val > 0 else 0
                                f.write(f"- **{metric.upper()}**: {improvement:+.2f}% improvement "
                                       f"({base_val:.4f} → {custom_val:.4f})\n")
                        
                        # Latency comparison
                        if 'avg_latency_ms' in base_metrics and 'avg_latency_ms' in custom_metrics:
                            base_latency = base_metrics['avg_latency_ms']
                            custom_latency = custom_metrics['avg_latency_ms']
                            latency_change = ((custom_latency - base_latency) / base_latency * 100) if base_latency > 0 else 0
                            f.write(f"- **Average Latency**: {latency_change:+.2f}% change "
                                   f"({base_latency:.2f}ms → {custom_latency:.2f}ms)\n")
                        
                        f.write("\n")

def get_code_generation_benchmark_examples() -> list[dict]:
    return [
        {
            "type": "code_generation",
            "instruction": "Write Python code using matplotlib to plot the distribution of packet lengths from the following list.",
            "input": [
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Created slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Started Session 298515 of user root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Removed slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                }
            ],
        },
        {
            "type": "code_generation",
            "instruction": "Given the following configuration, generate a YAML config file.",
            "input": [
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Created slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Started Session 298515 of user root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Removed slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                }
            ]
        },
        {
            "type": "code_generation",
            "instruction": "Write Python code to extract only 'error' level messages from the following syslog data.",
            "input": [
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Created slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Started Session 298515 of user root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "",
                    "program": "systemd",
                    "message": "Removed slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                }
            ],
            "output": ""
        }
    ]

def is_python_code_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def run_code_generation_passk_benchmark(models, k: int = 3, output_dir: str = "results"):
    items = get_code_generation_benchmark_examples()
    dataset_type = "code_generation"
    results = defaultdict(list)
    csv_rows = []
    for model in models:
        for idx, item in enumerate(items):
            passed = False
            attempts = []
            for attempt in range(k):
                prompt = f"{item['instruction']}\n\nContext:\n{item['input']}"
                code = model.predict(prompt)
                syntax_ok = is_python_code_syntax_valid(code)
                attempts.append({
                    'code': code,
                    'syntax_valid': syntax_ok
                })
                if syntax_ok:
                    passed = True
                    break
            results[model.model_name].append(passed)
            print(f"  Q{idx+1}: {'PASS' if passed else 'FAIL'}")
            for i, att in enumerate(attempts):
                csv_rows.append({
                    'model': model.model_name,
                    'question_number': idx+1,
                    'instruction': item['instruction'],
                    'input': str(item['input']),
                    'generated_code': att['code'],
                    'syntax_valid': att['syntax_valid'],
                    'attempt': i+1,
                    'k': k,
                    'pass@k': passed
                })

    for model_name, passes in results.items():
        score = sum(passes) / len(passes) if passes else 0.0
        print(f"{model_name}: {score:.2f} ({sum(passes)}/{len(passes)})")

    import os
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "code_generation_passk_results.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'question_number', 'instruction', 'input', 'generated_code',
            'syntax_valid', 'attempt', 'k', 'pass@k'])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Run LLM benchmark comparisons')
    parser.add_argument('--comparison', choices=['base_vs_custom', 'custom_vs_general', 'both'], 
                       default='both', help='Type of comparison to run')
    parser.add_argument('--benchmark_dir', type=str, default='benchmark_data',
                       help='Directory containing benchmark data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    setup_logging()
    
    if args.comparison in ['base_vs_custom', 'both']:
        print("Starting Base vs Custom Model Comparison...")
        run_base_vs_custom_comparison(benchmark_dir, output_dir)
    
    if args.comparison in ['custom_vs_general', 'both']:
        print("Starting Custom vs General LLMs Comparison...")
        run_custom_vs_general_comparison(benchmark_dir, output_dir)
    
    print(f"\nAll comparisons completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 
