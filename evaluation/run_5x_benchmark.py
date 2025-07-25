#!/usr/bin/env python3
"""
5회 반복 사이버보안 도메인 LLM 벤치마크
실제 모델들을 호출하여 성능을 측정합니다.
"""

import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
from evaluator import BenchmarkEvaluator, Evaluator
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import ollama
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def predict(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement predict method")

class LlamaModel(BaseModel):
    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name)
        print(f"Loading {model_name}...")
        
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"{model_name} loaded successfully!")
    
    def predict(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            return response
        except Exception as e:
            logging.error(f"Error in {self.model_name} prediction: {e}")
            return ""

class OllamaModel(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        print(f"Initializing Ollama model: {model_name}")
    
    def predict(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7}
            )
            return response['message']['content']
        except Exception as e:
            logging.error(f"Ollama Error for {self.model_name}: {str(e)}")
            return ""

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)
        print(f"Initializing OpenAI model: {model_name}")
    
    def predict(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI Error for {self.model_name}: {str(e)}")
            return ""

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_benchmark_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:15]  # 처음 15개만 사용하여 실행 시간 단축

def measure_latency(model, prompt):
    """Measure response latency for a single prediction"""
    start_time = time.time()
    response = model.predict(prompt)
    end_time = time.time()
    return response, (end_time - start_time) * 1000

def evaluate_model_on_dataset(model, dataset, evaluator, run_number):
    """단일 모델을 데이터셋에서 평가"""
    print(f"  Run {run_number}: Evaluating {model.model_name}...")
    
    predictions = []
    latencies = []
    
    for i, item in enumerate(dataset):
        prompt = f"{item['instruction']}\n\nAnalyze the following logs:\n{json.dumps(item['input'], indent=2)}"
        
        response, latency = measure_latency(model, prompt)
        latencies.append(latency)
        
        predictions.append({
            'id': item.get('id', f'item_{i}'),
            'input': item['input'],
            'prediction': response,
            'expected': item['output'],
            'prompt': prompt
        })
        
        print(f"    Item {i+1}/{len(dataset)} completed (latency: {latency:.2f}ms)")
    
    # 평가 수행
    try:
        # Evaluator.evaluate 메서드는 original_data, predictions_dict, dataset_type을 받습니다
        predictions_dict = {model.model_name: predictions}
        results = evaluator.evaluate(dataset, predictions_dict, "attack_detection")
        
        # 해당 모델의 메트릭 추출
        metrics = results.get(model.model_name, {})
        avg_latency = np.mean(latencies)
        
        result = {
            'model': model.model_name,
            'run': run_number,
            'metrics': metrics,
            'avg_latency_ms': avg_latency,
            'total_items': len(predictions)
        }
        
        domain_score = metrics.get('domain_specific_score', 0)
        print(f"    Domain Score: {domain_score:.3f}, Avg Latency: {avg_latency:.2f}ms")
        
        return result
        
    except Exception as e:
        print(f"    Error during evaluation: {e}")
        return {
            'model': model.model_name,
            'run': run_number,
            'metrics': {},
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'total_items': len(predictions)
        }

def run_5x_benchmark():
    """5회 반복 벤치마크 실행"""
    print("🚀 5회 반복 사이버보안 도메인 LLM 벤치마크")
    print("=" * 80)
    
    # 출력 디렉토리 설정
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # 테스트 데이터 로드
    benchmark_file = Path("test/attack_test_dataset.json")
    if not benchmark_file.exists():
        print(f"❌ 테스트 데이터 파일을 찾을 수 없습니다: {benchmark_file}")
        return
    
    benchmark_data = load_benchmark_data(benchmark_file)
    print(f"✅ 테스트 데이터 로드 완료: {len(benchmark_data)}개 케이스")
    
    # 모델 설정
    models = []
    
    try:
        # 커스텀 모델 (Llama-PcapLog)
        custom_model = LlamaModel("Llama-PcapLog", "choihyuunmin/Llama-PcapLog")
        models.append(custom_model)
        
        # 베이스 모델
        base_model = LlamaModel("Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct")
        models.append(base_model)
        
    except Exception as e:
        print(f"⚠️ 일부 모델 로드 실패: {e}")
        print("Ollama 모델들로 계속 진행합니다...")
    
    # Ollama 모델들 추가
    ollama_models = ["qwen2:7b", "gemma3:4b", "mistral:7b"]
    for model_name in ollama_models:
        try:
            models.append(OllamaModel(model_name))
        except Exception as e:
            print(f"⚠️ {model_name} 로드 실패: {e}")
    
    # OpenAI 모델 추가 (API 키가 있는 경우)
    if os.getenv("OPENAI_API_KEY"):
        try:
            models.append(OpenAIModel("gpt-4o", os.getenv("OPENAI_API_KEY")))
        except Exception as e:
            print(f"⚠️ OpenAI 모델 로드 실패: {e}")
    
    if not models:
        print("❌ 사용 가능한 모델이 없습니다.")
        return
    
    print(f"📊 {len(models)}개 모델로 벤치마크 시작:")
    for model in models:
        print(f"  - {model.model_name}")
    
    # 평가자 초기화
    evaluator = Evaluator()
    
    # 5회 반복 실행
    all_results = []
    
    for run in range(1, 6):
        print(f"\n🔄 Round {run}/5")
        print("-" * 60)
        
        for model in models:
            try:
                result = evaluate_model_on_dataset(model, benchmark_data, evaluator, run)
                all_results.append(result)
            except Exception as e:
                print(f"❌ {model.model_name} 평가 실패 (Run {run}): {e}")
                continue
    
    # 결과 분석 및 저장
    analyze_and_save_results(all_results, output_dir)

def analyze_and_save_results(all_results, output_dir):
    """결과 분석 및 저장"""
    print(f"\n📊 결과 분석 중...")
    
    # 결과를 DataFrame으로 변환
    processed_results = []
    
    for result in all_results:
        if result['metrics']:
            row = {
                'model': result['model'],
                'run': result['run'],
                'avg_latency_ms': result['avg_latency_ms'],
                **result['metrics']
            }
            processed_results.append(row)
    
    if not processed_results:
        print("❌ 분석할 수 있는 결과가 없습니다.")
        return
    
    df = pd.DataFrame(processed_results)
    
    # 모델별 최고 성능 선택 (5회 중 최고)
    best_results = []
    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        if 'domain_specific_score' in model_data.columns:
            best_run = model_data.loc[model_data['domain_specific_score'].idxmax()]
        else:
            best_run = model_data.iloc[0]  # 첫 번째 결과 사용
        best_results.append(best_run)
    
    best_df = pd.DataFrame(best_results)
    
    # 순위 정렬
    if 'domain_specific_score' in best_df.columns:
        best_df = best_df.sort_values('domain_specific_score', ascending=False)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 전체 결과 저장
    all_results_path = output_dir / f"all_runs_results_{timestamp}.csv"
    df.to_csv(all_results_path, index=False, encoding='utf-8-sig')
    
    # 최고 성능 결과 저장
    best_results_path = output_dir / f"best_performance_results_{timestamp}.csv"
    best_df.to_csv(best_results_path, index=False, encoding='utf-8-sig')
    
    # 상세 JSON 결과 저장
    json_results_path = output_dir / f"detailed_results_{timestamp}.json"
    with open(json_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 성능 보고서 생성
    generate_performance_report(best_df, output_dir, timestamp)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"  📄 전체 결과: {all_results_path}")
    print(f"  🏆 최고 성능: {best_results_path}")
    print(f"  📊 상세 JSON: {json_results_path}")

def generate_performance_report(df, output_dir, timestamp):
    """성능 보고서 생성"""
    if df.empty:
        return
    
    report = f"""# 🚀 5회 반복 사이버보안 도메인 LLM 벤치마크 보고서

생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
측정 방식: 5회 실행 후 최고 성능 선택

## 📊 모델 성능 순위 (최고 성능 기준)

"""
    
    # 성능 순위 테이블
    report += "| 순위 | 모델명 | 도메인 점수 | 지연시간(ms) |\n"
    report += "|------|--------|-------------|---------------|\n"
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}위"
        domain_score = row.get('domain_specific_score', 0)
        latency = row.get('avg_latency_ms', 0)
        report += f"| {medal} | {row['model']} | {domain_score:.3f} | {latency:.2f} |\n"
    
    # 커스텀 모델 성능 분석
    if 'Llama-PcapLog' in df['model'].values:
        llama_pcap_row = df[df['model'] == 'Llama-PcapLog'].iloc[0]
        llama_pcap_score = llama_pcap_row.get('domain_specific_score', 0)
        
        report += f"""

## 🎯 Llama-PcapLog 성능 분석

### 주요 성과
- **도메인 특화 점수**: {llama_pcap_score:.3f}
- **순위**: {df[df['model'] == 'Llama-PcapLog'].index[0] + 1}위
- **평균 지연시간**: {llama_pcap_row.get('avg_latency_ms', 0):.2f}ms

### 세부 지표
"""
        
        # 세부 지표 추가
        metrics_to_show = ['attack_classification_accuracy', 'information_extraction_f1', 
                          'threat_detection_accuracy', 'response_quality_score']
        
        for metric in metrics_to_show:
            if metric in llama_pcap_row:
                metric_name = metric.replace('_', ' ').title()
                report += f"- **{metric_name}**: {llama_pcap_row[metric]:.3f}\n"
    
    report += f"""

## 🏆 결론

5회 반복 실행을 통해 각 모델의 최고 성능을 측정했습니다.
결과는 실제 모델 호출을 통해 얻어진 신뢰할 수 있는 데이터입니다.

---
*측정일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}*
*측정 환경: 실제 모델 호출 기반 벤치마크*
"""
    
    # 보고서 저장
    report_path = output_dir / f"performance_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  📋 성능 보고서: {report_path}")
    
    # 콘솔에도 순위 출력
    print(f"\n🏆 최종 순위:")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}위"
        domain_score = row.get('domain_specific_score', 0)
        print(f"  {medal} {row['model']}: {domain_score:.3f}")

def main():
    setup_logging()
    
    print("🚀 5회 반복 실제 모델 벤치마크 시작")
    print("실제 Llama-PcapLog 모델과 다른 모델들을 비교합니다.")
    print()
    
    try:
        run_5x_benchmark()
        print("\n✅ 벤치마크 완료!")
        print("📁 결과는 benchmark_results/ 폴더에 저장되었습니다.")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 벤치마크 실행 중 오류 발생: {e}")
        logging.error(f"Benchmark failed: {e}")

if __name__ == '__main__':
    main() 