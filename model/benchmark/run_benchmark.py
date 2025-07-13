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
import ast
from collections import defaultdict
import csv

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
            logging.error(f"OpenAI API 호출 중 오류 발생: {str(e)}")
            return ""

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_benchmark_data(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)

def save_results(benchmark_data, predictions, metrics_data, output_dir, dataset_type, run_num=None):
    # CSV 파일명 설정
    if run_num is not None:
        csv_filename = f'{dataset_type}_run_{run_num}_metrics.csv'
        json_filename = f'{dataset_type}_run_{run_num}_results.json'
    else:
        csv_filename = f'{dataset_type}_avg_metrics.csv'
        json_filename = f'{dataset_type}_avg_results.json'
    
    # 메트릭 데이터를 DataFrame으로 변환
    metrics_df = pd.DataFrame(metrics_data).T
    metrics_df.index.name = 'model'
    
    # CSV 파일로 저장
    csv_path = output_dir / csv_filename
    metrics_df.to_csv(csv_path)
    
    # JSON 파일로 저장 (원본 데이터 + 예측 결과)
    json_path = output_dir / json_filename
    results = {
        'benchmark_data': benchmark_data,
        'predictions': predictions,
        'metrics': metrics_data
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def run_benchmark(models, benchmark_dir, output_dir):
    """벤치마크 실행"""
    evaluator = BenchmarkEvaluator()
    num_runs = 5 
    
    for dataset_type in ['pcap', 'syslog']:
        print(f"\n=== {dataset_type} 데이터셋 평가 시작 ===")
        
        benchmark_file = benchmark_dir / f'{dataset_type}_test.json'
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)

        type_groups = {}
        for item in benchmark_data:
            t = item['type']
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(item)

        # type별로 벤치마크 실행
        for type_name, items in type_groups.items():
            print(f"\n=== {dataset_type} {type_name} 데이터셋 평가 시작 ===")
            
            all_metrics_data = []
            all_predictions_list = []
            
            for run in range(num_runs):
                print(f"\n=== 실행 {run + 1}/{num_runs} ===")
                
                # 각 모델별 예측 결과 수집
                all_predictions = {}
                for model in models:
                    print(f"\n{model.model_name} 모델 예측 중...")
                    predictions = []
                    for item in items: # 현재 type의 문제들만 반복
                        prompt = f"{item['instruction']}\n\n Context: \n{item['input']}"
                        prediction = model.predict(prompt)
                        predictions.append({
                            'input': prompt,
                            'output': prediction,
                            'expected_output': item['output']
                        })
                    all_predictions[model.model_name] = predictions
                
                # 평가 실행
                metrics_data = evaluator.evaluate(items, all_predictions, dataset_type) # items를 직접 전달
                all_metrics_data.append(metrics_data)
                all_predictions_list.append(all_predictions)
                
                # 각 실행의 결과 저장
                save_results(items, all_predictions, metrics_data, output_dir, f"{dataset_type}_{type_name}", run + 1) # type_name 추가
            
            # 평균 메트릭 계산
            avg_metrics_data = {}
            for model_name in all_metrics_data[0].keys():
                avg_metrics = {}
                for metric in all_metrics_data[0][model_name].keys():
                    values = [run_data[model_name][metric] for run_data in all_metrics_data]
                    avg_metrics[metric] = sum(values) / len(values)
                avg_metrics_data[model_name] = avg_metrics
            
            # 평균 메트릭 저장
            save_results(items, all_predictions_list[-1], avg_metrics_data, output_dir, f"{dataset_type}_{type_name}") # type_name 추가
            
            # 평균 메트릭으로 시각화
            evaluator.visualize_metrics(avg_metrics_data, f"{dataset_type}_{type_name}_avg")
            evaluator.visualize_regex_metrics(avg_metrics_data, f"{dataset_type}_{type_name}_avg")
            evaluator.visualize_code_metrics(avg_metrics_data, f"{dataset_type}_{type_name}_avg")
            
            print(f"=== {dataset_type} {type_name} 데이터셋 평가 완료 ===\n")
        
        print(f"=== {dataset_type} 데이터셋 평가 완료 ===\n")

def get_code_generation_benchmark_examples() -> list[dict]:
    return [
        {
            "type": "code_generation",
            "instruction": "Write Python code using matplotlib to plot the distribution of packet lengths from the following list.",
            "input": [
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "krit-wwas-dev",
                    "program": "systemd",
                    "message": "Created slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "krit-wwas-dev",
                    "program": "systemd",
                    "message": "Started Session 298515 of user root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "krit-wwas-dev",
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
                    "host": "krit-wwas-dev",
                    "program": "systemd",
                    "message": "Created slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "krit-wwas-dev",
                    "program": "systemd",
                    "message": "Started Session 298515 of user root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "krit-wwas-dev",
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
                    "host": "krit-wwas-dev",
                    "program": "systemd",
                    "message": "Created slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "krit-wwas-dev",
                    "program": "systemd",
                    "message": "Started Session 298515 of user root.",
                    "severity": "info",
                    "category": "auth"
                },
                {
                    "timestamp": "2025-05-13T16:24:01",
                    "host": "krit-wwas-dev",
                    "program": "systemd",
                    "message": "Removed slice User Slice of root.",
                    "severity": "info",
                    "category": "auth"
                }
            ],
            "output": (
                "errors = [log for log in logs if log['severity'] == 'error']\n"
                "for error in errors:\n"
                "    print(error['timestamp'], error['message'])"
            )
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
    results = defaultdict(list)  # model_name -> [True/False ...]
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
            # 모든 시도 기록 (최대 k개)
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

    # 모델별 pass@k 집계
    for model_name, passes in results.items():
        score = sum(passes) / len(passes) if passes else 0.0
        print(f"{model_name}: {score:.2f} ({sum(passes)}/{len(passes)})")

    # CSV 저장
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
    # 모델 설정
    models = [
        LlamaModel("LlamaTrace", "choihyuunmin/LlamaTrace"),
        LlamaModel("Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
        OpenAIModel("gpt-4o", os.getenv("OPENAI_API_KEY"))
    ]
    
    benchmark_dir = Path('benchmark_data')
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    run_benchmark(models, benchmark_dir, output_dir)
    run_code_generation_passk_benchmark(models, k=3, output_dir=str(output_dir))

if __name__ == '__main__':
    main() 
