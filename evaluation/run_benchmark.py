import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from evaluator import Evaluator
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import time
import gc
import ast
import re

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
        
        print(f"Loading {model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # pad_token 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # gemma 모델의 경우 특별한 설정 적용
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            # gemma 모델에 대한 특별한 처리
            if "gemma" in model_name.lower():
                print(f"  Gemma 모델 감지: 안전한 설정을 적용합니다...")
                if self.device == "cuda":
                    model_kwargs.update({
                        "device_map": None,  # auto 대신 None 사용
                        "torch_dtype": torch.float32,  # float16 대신 float32 사용
                    })
                else:
                    model_kwargs["device_map"] = None
            else:
                if self.device == "cuda":
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["device_map"] = None
            
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            # 디바이스로 수동 이동
            if self.device == "mps":
                self.model = self.model.to("mps")
            elif self.device == "cpu":
                self.model = self.model.to("cpu")
            elif self.device == "cuda" and "gemma" in model_name.lower():
                # gemma 모델의 경우 수동으로 CUDA로 이동
                self.model = self.model.to("cuda")
                
            print(f"  {model_name} 로딩 완료")
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                print(f"{model_name} CUDA 로딩 실패: {e}")
                print(f"  CPU 모드로 fallback...")
                self.device = "cpu"
                
                # CPU 모드로 재시도
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "device_map": None,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True
                }
                
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                self.model = self.model.to("cpu")
                print(f"  {model_name} CPU 모드로 로딩 완료")
            else:
                raise e
    
    def predict(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                try:
                    # 더 작은 토큰 길이로 재시도
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,  # 더 작은 출력 길이
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                    return response
                except Exception as retry_error:
                    print(f"  재시도도 실패: {retry_error}")
                    return "" 
            else:
                print(f"{self.model_name} 예측 중 오류: {e}")
                return "" 
        except Exception as e:
            print(f"{self.model_name} 예측 중 알 수 없는 오류: {e}")
            return "" 
    
    def cleanup(self):
        """메모리 정리"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            # CUDA 메모리 정리 시 예외 처리
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # GPU 작업 완료 대기
                except RuntimeError as e:
                    try:
                        # 대안적 메모리 정리
                        torch.cuda.ipc_collect()
                        gc.collect()
                    except Exception as alt_e:
                        print(f"  대안적 메모리 정리도 실패: {alt_e}")
            
            gc.collect()
            print(f"  {self.model_name} 메모리 정리 완료")
            
        except Exception as e:
            print(f"{self.model_name} 메모리 정리 중 오류 발생: {e}")

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        # 실제 OpenAI API 모델명으로 매핑
        self.model_name = "gpt-4o" if model_name == "gpt-4o" else model_name
        print(f"OpenAI 모델 초기화: {model_name} (API 모델명: {self.model_name})")
    
    def predict(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512
            )
            result = response.choices[0].message.content
            return result if result else ""
        except Exception as e:
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

def get_code_generation_test_data():
    """Code generation test data - 20 items"""
    return [
        {
            "instruction": "Write a Python function to filter network packet data for connections to a specific port.",
            "input": {"port": 80, "protocol": "tcp"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a Python function that includes regular expressions to extract IP addresses from log data.",
            "input": {"log": "192.168.1.1 - - [25/Dec/2023:10:00:00 +0000] GET /"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a Python class to analyze cyber attack logs and classify attack types.",
            "input": {"attack_types": ["brute_force", "ddos", "malware"]},
            "expected_type": "class"
        },
        {
            "instruction": "Write a Python function to parse firewall rules into JSON format.",
            "input": {"rule": "allow tcp from any to 192.168.1.0/24 port 22"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a function to parse syslog data and convert it into a structured dictionary.",
            "input": {"syslog": "Jan 1 00:00:01 server sshd: Failed password for user"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a packet capture data processing function for network traffic analysis.",
            "input": {"format": "pcap"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a Python class to connect to and query a security event database.",
            "input": {"db_type": "sqlite", "table": "security_events"},
            "expected_type": "class"
        },
        {
            "instruction": "Write a function to detect SQL injection attacks in web logs.",
            "input": {"log_entry": "GET /login.php?id=1' OR '1'='1"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a port scan analysis function for network scan detection.",
            "input": {"connections": [{"src": "192.168.1.100", "dst_port": 22}]},
            "expected_type": "function"
        },
        {
            "instruction": "Write a TLS/SSL connection monitoring function to analyze encrypted communications.",
            "input": {"protocol": "TLS1.3", "cipher": "AES256"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a function to calculate and compare hash values of malware samples.",
            "input": {"hash_type": "sha256", "file_path": "/tmp/sample.exe"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a class to parse Intrusion Detection System (IDS) alerts.",
            "input": {"alert_format": "snort", "severity": "high"},
            "expected_type": "class"
        },
        {
            "instruction": "Write a function to analyze DNS query logs and detect suspicious domains.",
            "input": {"dns_log": "query: evil-domain.com type A"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a function to establish network baselines and detect anomalous traffic.",
            "input": {"baseline_period": "7days", "threshold": 2.0},
            "expected_type": "function"
        },
        {
            "instruction": "Write a class to automatically generate security incident reports.",
            "input": {"template": "NIST", "format": "pdf"},
            "expected_type": "class"
        },
        {
            "instruction": "Write a function to perform container security scanning in virtualized environments.",
            "input": {"container_runtime": "docker", "scan_type": "vulnerability"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a function to collect and analyze cloud security logs.",
            "input": {"cloud_provider": "aws", "service": "cloudtrail"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a class to process real-time threat intelligence feeds.",
            "input": {"feed_type": "IOC", "format": "STIX"},
            "expected_type": "class"
        },
        {
            "instruction": "Write a function to validate network segmentation policies.",
            "input": {"policy": "zero-trust", "network": "192.168.0.0/16"},
            "expected_type": "function"
        },
        {
            "instruction": "Write a class to implement security automation workflows.",
            "input": {"trigger": "security_alert", "action": "isolate_host"},
            "expected_type": "class"
        }
    ]

def is_python_code_valid(code: str) -> bool:
    """Python 코드 구문 유효성 검사"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def extract_python_code(response: str) -> str:
    """응답에서 Python 코드 블록 추출"""
    # ```python 코드 블록 찾기
    python_pattern = r'```python\s*\n(.*?)\n```'
    match = re.search(python_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # ``` 일반 코드 블록 찾기
    code_pattern = r'```\s*\n(.*?)\n```'
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # def 또는 class로 시작하는 라인들 찾기
    lines = response.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
            in_code = True
        
        if in_code:
            code_lines.append(line)
            
        # 빈 줄이 연속으로 나오면 코드 끝
        if in_code and line.strip() == '' and len(code_lines) > 1:
            break
    
    return '\n'.join(code_lines).strip()

def evaluate_code_generation_passk(model, test_data, k=3):
    """코드 생성 pass@k 평가"""
    total_tasks = len(test_data)
    passed_tasks = 0
    detailed_results = []
    
    for i, task in enumerate(test_data):
        print(f"  코드 생성 테스트 {i+1}/{total_tasks}")
        
        prompt = f"{task['instruction']}\n\nInput: {json.dumps(task['input'], ensure_ascii=False)}\n\nPlease write Python code:"
        
        task_passed = False
        attempts = []
        
        # k번 시도
        for attempt in range(k):
            try:
                response, latency = measure_latency(model, prompt)
                code = extract_python_code(response)
                
                is_valid = is_python_code_valid(code) if code else False
                
                attempts.append({
                    'attempt': attempt + 1,
                    'response': response,
                    'extracted_code': code,
                    'is_valid': is_valid,
                    'latency_ms': latency
                })
                
                if is_valid:
                    task_passed = True
                    break
                    
            except Exception as e:
                print(f"    시도 {attempt + 1} 실패: {str(e)}")
                attempts.append({
                    'attempt': attempt + 1,
                    'response': "",  # 빈 문자열로 수정
                    'extracted_code': "",
                    'is_valid': False,
                    'latency_ms': 0
                })
        
        if task_passed:
            passed_tasks += 1
        
        detailed_results.append({
            'task_index': i + 1,
            'instruction': task['instruction'],
            'input_data': json.dumps(task['input'], ensure_ascii=False),
            'passed': task_passed,
            'attempts': attempts,
            'k': k
        })
    
    pass_at_k = passed_tasks / total_tasks if total_tasks > 0 else 0
    
    return {
        'pass_at_k': pass_at_k,
        'passed_tasks': passed_tasks,
        'total_tasks': total_tasks,
        'detailed_results': detailed_results
    }

def test_single_model(model_info, benchmark_data, code_gen_data, evaluator):
    """단일 모델 테스트 및 메모리 정리"""
    model_name, model_class, model_args = model_info
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        # 모델 로딩 시 CUDA 관련 예외 처리
        try:
            model = model_class(model_name, model_args)
        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error):
                print(f"{model_name} CUDA 로딩 실패: {cuda_error}")
                # CPU 모드로 fallback 시도
                original_device = torch.cuda.is_available()
                torch.cuda.is_available = lambda: False  # 임시로 CUDA 비활성화
                try:
                    model = model_class(model_name, model_args)
                    print(f"  {model_name} CPU 모드로 로딩 성공")
                except Exception as cpu_error:
                    print(f"{model_name} CPU 모드로도 로딩 실패: {cpu_error}")
                    return [], [], {}
                finally:
                    torch.cuda.is_available = lambda: original_device  # 원래 상태 복원
            else:
                raise cuda_error
        
        all_results = []
        detailed_results = []
        test_set_results = {}
        
        for test_type, test_items in benchmark_data.items():
            print(f"\n--- {test_type.upper()} 테스트 세트 평가 중 ---")
            print(f"테스트 항목 수: {len(test_items)}")
            
            predictions = []
            latencies = []
            test_set_detailed = []
            
            for item_idx, item in enumerate(test_items):
                try:
                    prompt = f"{item['instruction']}\n\nContext:\n{json.dumps(item['input'], indent=2, ensure_ascii=False)}"
                    
                    response, latency = measure_latency(model, prompt)
                    
                    predictions.append({
                        'input': prompt,
                        'output': response,
                        'expected_output': item['output']
                    })
                    latencies.append(latency)
                    
                    test_item_detail = {
                        'item_index': item_idx + 1,
                        'instruction': item['instruction'],
                        'input_data': json.dumps(item['input'], ensure_ascii=False),
                        'expected_output': item['output'],
                        'generated_output': response,
                        'latency_ms': latency,
                        'response_length': len(response) if response else 0
                    }
                    
                    test_set_detailed.append(test_item_detail)
                    
                    # 전체 결과에도 추가
                    detailed_results.append({
                        'test_type': test_type,
                        'model': model_name,
                        **test_item_detail
                    })
                    
                    if (item_idx + 1) % 5 == 0:
                        print(f"  진행률: {item_idx + 1}/{len(test_items)} 완료")
                        
                except Exception as e:
                    print(f"  오류 발생 (항목 {item_idx + 1}): {str(e)}")
                    latencies.append(0)
                    
                    error_detail = {
                        'item_index': item_idx + 1,
                        'instruction': item['instruction'],
                        'input_data': json.dumps(item['input'], ensure_ascii=False),
                        'expected_output': item['output'],
                        'generated_output': f"오류: {str(e)}",
                        'latency_ms': 0,
                        'response_length': 0
                    }
                    
                    test_set_detailed.append(error_detail)
                    detailed_results.append({
                        'test_type': test_type,
                        'model': model_name,
                        **error_detail
                    })
            
            # 평가 메트릭 계산
            model_predictions = {model_name: predictions}
            try:
                metrics = evaluator.evaluate(test_items, model_predictions, test_type)
                model_metrics = metrics.get(model_name, {})
                
                # 메트릭 값 검증 및 정리 (수치가 아닌 값들을 0으로 대체)
                cleaned_metrics = {}
                for key, value in model_metrics.items():
                    try:
                        # 숫자 형태로 변환 가능한지 확인
                        if isinstance(value, (int, float)):
                            cleaned_metrics[key] = float(value)
                        elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                            cleaned_metrics[key] = float(value)
                        else:
                            cleaned_metrics[key] = 0.0
                    except (ValueError, TypeError):
                        cleaned_metrics[key] = 0.0
                        
                model_metrics = cleaned_metrics
                
            except Exception as e:
                print(f"  메트릭 계산 오류: {str(e)}")
                model_metrics = {}
            
            # 기본 메트릭 추가
            # success_rate: 유효한 출력(빈 문자열이 아닌)을 생성한 비율
            valid_predictions = len([p for p in predictions if p['output'] and p['output'].strip()])
            model_metrics.update({
                'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
                'total_latency_ms': sum(latencies),
                'total_items': len(test_items),
                'success_rate': valid_predictions / len(predictions) if predictions else 0
            })
            
            # 결과 저장
            result = {
                'test_type': test_type,
                'model': model_name,
                **model_metrics
            }
            
            all_results.append(result)
            
            # 테스트 세트별 결과 저장
            test_set_results[test_type] = {
                'metrics': model_metrics,
                'detailed_results': test_set_detailed,
                'total_items': len(test_items)
            }
            
            # 주요 메트릭 출력
            accuracy = model_metrics.get('attack_classification_accuracy', 0)
            avg_latency = model_metrics.get('avg_latency_ms', 0)
            print(f"  정확도: {accuracy:.3f}, 평균 지연시간: {avg_latency:.2f}ms")
        
        # 코드 생성 테스트 수행
        print(f"\n--- CODE GENERATION 테스트 세트 평가 중 ---")
        print(f"테스트 항목 수: {len(code_gen_data)}")
        
        code_gen_results = evaluate_code_generation_passk(model, code_gen_data, k=3)
        
        # 코드 생성 결과를 전체 결과에 추가
        try:
            avg_latency_ms = sum([
                sum([attempt['latency_ms'] for attempt in detail['attempts']])
                for detail in code_gen_results['detailed_results']
            ]) / (len(code_gen_results['detailed_results']) * 3) if code_gen_results['detailed_results'] else 0
        except (ZeroDivisionError, TypeError):
            avg_latency_ms = 0.0
            
        code_gen_result = {
            'test_type': 'code_generation',
            'model': model_name,
            'pass_at_k': float(code_gen_results.get('pass_at_k', 0)),
            'passed_tasks': int(code_gen_results.get('passed_tasks', 0)),
            'total_tasks': int(code_gen_results.get('total_tasks', 0)),
            'avg_latency_ms': float(avg_latency_ms),
            'success_rate': float(code_gen_results.get('pass_at_k', 0))
        }
        
        all_results.append(code_gen_result)
        
        # 코드 생성 테스트 세트별 결과 저장
        test_set_results['code_generation'] = {
            'metrics': {
                'pass_at_k': code_gen_results['pass_at_k'],
                'passed_tasks': code_gen_results['passed_tasks'],
                'total_tasks': code_gen_results['total_tasks']
            },
            'detailed_results': code_gen_results['detailed_results'],
            'total_items': len(code_gen_data)
        }
        
        print(f"  Pass@3: {code_gen_results['pass_at_k']:.3f} ({code_gen_results['passed_tasks']}/{code_gen_results['total_tasks']})")
        
        return all_results, detailed_results, test_set_results
        
    except Exception as e:
        print(f"{model_name} 테스트 실패: {str(e)}")
        return [], [], {}

def run_comprehensive_benchmark(benchmark_dir, output_dir):
    """6개 모델에 대한 종합 벤치마크 수행 (순차 실행)"""
    print("\n" + "="*80)
    print("COMPREHENSIVE LLM BENCHMARK - 6 MODELS COMPARISON")
    print("="*80)
    
    # 6개 모델 정의 (공개 모델들로 수정)
    model_configs = [
        ("Llama-PcapLog", LlamaModel, "choihyuunmin/Llama-PcapLog"),
        ("Llama-3-8B", LlamaModel, "meta-llama/Meta-Llama-3-8B"),
        ("Qwen2-7B", LlamaModel, "Qwen/Qwen2-7B"),
        ("Gemma-3-4B-IT", LlamaModel, "google/gemma-3-4b-it"),
        ("Mistral-7B-Instruct", LlamaModel, "mistralai/Mistral-7B-Instruct-v0.1")
    ]
    
    # OpenAI 모델 추가 (API 키가 있는 경우)
    if os.getenv("OPENAI_API_KEY"):
        model_configs.append(("gpt-4o", OpenAIModel, os.getenv("OPENAI_API_KEY")))

    print(f"총 {len(model_configs)}개 모델로 벤치마크를 진행합니다.")
    
    evaluator = Evaluator()
    
    # test_dataset.json 파일 로드
    benchmark_file = benchmark_dir / 'test' / 'test_dataset.json'
    
    if not benchmark_file.exists():
        print(f"Error: {benchmark_file} 파일을 찾을 수 없습니다.")
        return
        
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    # 코드 생성 테스트 데이터 로드
    code_gen_data = get_code_generation_test_data()
    
    print(f"기본 테스트 세트: {len(benchmark_data)}개")
    print(f"코드 생성 테스트: {len(code_gen_data)}개")
    
    # 전체 결과 저장용
    all_results = []
    all_detailed_results = []
    all_test_set_results = {}  # 각 테스트 세트별 결과들
    
    # 각 모델을 순차적으로 테스트 (메모리 효율성)
    for i, model_config in enumerate(model_configs):
        model_name = model_config[0]
        print(f"\n모델 {i+1}/{len(model_configs)}: {model_name}")
        
        try:
            # 단일 모델 테스트
            model_results, model_detailed, model_test_sets = test_single_model(
                model_config, benchmark_data, code_gen_data, evaluator
            )
            
            # 결과 합치기
            all_results.extend(model_results)
            all_detailed_results.extend(model_detailed)
            
            for test_type, test_result in model_test_sets.items():
                if test_type not in all_test_set_results:
                    all_test_set_results[test_type] = {}
                all_test_set_results[test_type][model_name] = test_result
            
            print(f"{model_name} 테스트 완료")
            
        except Exception as model_error:
            
            failed_result = {
                'test_type': 'failed',
                'model': model_name,
                'error': str(model_error)[:200],
                'avg_latency_ms': 0.0,
                'success_rate': 0.0,
                'total_items': 0,
                'passed_tasks': 0,
                'total_tasks': 0
            }
            all_results.append(failed_result)
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            print(f"메모리 정리 완료")
        except RuntimeError as e:
            print(f"메모리 정리 중 CUDA 오류 발생: {e}")
            print("메모리 정리를 건너뛰고 다음 모델로 진행합니다.")
            try:
                gc.collect()
            except Exception:
                pass
    
    save_benchmark_results_to_csv(all_results, all_detailed_results, all_test_set_results, output_dir)
    
    print_overall_rankings(all_results)
    
    return all_results, all_detailed_results

def clean_data_for_csv(data):
    """CSV 저장을 위한 데이터 정리"""
    if isinstance(data, list):
        return [clean_data_for_csv(item) for item in data]
    elif isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            try:
                if isinstance(value, (int, float)):
                    cleaned[key] = float(value) if not pd.isna(value) else 0.0
                elif isinstance(value, str):
                    # 숫자 문자열인지 확인
                    if value.replace('.', '').replace('-', '').isdigit():
                        cleaned[key] = float(value)
                    else:
                        cleaned[key] = value  # 문자열은 그대로 유지
                else:
                    cleaned[key] = str(value) if value is not None else ""
            except (ValueError, TypeError):
                cleaned[key] = str(value) if value is not None else ""
        return cleaned
    else:
        return data

def save_benchmark_results_to_csv(results, detailed_results, test_set_results, output_dir):
    """벤치마크 결과를 CSV 파일로 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 데이터 정리
    cleaned_results = clean_data_for_csv(results)
    cleaned_detailed = clean_data_for_csv(detailed_results)
    
    try:
        # 요약 결과 저장
        summary_df = pd.DataFrame(cleaned_results)
        # NaN 값들을 0으로 대체
        numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_columns] = summary_df[numeric_columns].fillna(0)
        
        summary_path = output_dir / 'benchmark_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n요약 결과가 {summary_path}에 저장되었습니다.")
        
        # 상세 결과 저장
        detailed_df = pd.DataFrame(cleaned_detailed)
        # NaN 값들을 적절한 기본값으로 대체
        for col in detailed_df.columns:
            if detailed_df[col].dtype in ['float64', 'int64']:
                detailed_df[col] = detailed_df[col].fillna(0)
            else:
                detailed_df[col] = detailed_df[col].fillna("")
                
        detailed_path = output_dir / 'benchmark_detailed.csv'
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        print(f"상세 결과가 {detailed_path}에 저장되었습니다.")
        
    except Exception as e:
        print(f"CSV 저장 중 오류 발생: {e}")
        print("데이터 구조 확인:")
        if results:
            print(f"  results 샘플: {results[0] if results else 'None'}")
        if detailed_results:
            print(f"  detailed_results 샘플: {detailed_results[0] if detailed_results else 'None'}")
    
    # 통합 벤치마크 결과 테이블 생성
    create_unified_benchmark_table(test_set_results, output_dir)
    
    # 각 테스트 세트별 결과 저장
    for test_type, test_models in test_set_results.items():
        print(f"\n{test_type} 테스트 세트 결과 저장 중...")
        
        # 테스트 세트별 요약
        test_summary = []
        test_detailed = []
        
        for model_name, model_data in test_models.items():
            # 요약 데이터
            summary_row = {
                'model': model_name,
                'test_type': test_type,
                **model_data['metrics']
            }
            test_summary.append(summary_row)
            
            # 상세 데이터
            for detail in model_data['detailed_results']:
                detail_row = {
                    'model': model_name,
                    'test_type': test_type,
                    **detail
                }
                test_detailed.append(detail_row)
        
        # 테스트 세트별 요약 저장
        if test_summary:
            test_summary_df = pd.DataFrame(test_summary)
            test_summary_path = output_dir / f'{test_type}_summary.csv'
            test_summary_df.to_csv(test_summary_path, index=False, encoding='utf-8-sig')
            print(f"  {test_type} 요약: {test_summary_path}")
        
    
    # 모델별 평균 성능 계산 (모든 테스트 포함)
    if results:
        # 공격 분류 정확도가 있는 결과만 필터링
        attack_results = [r for r in results if 'attack_classification_accuracy' in r]
        if attack_results:
            attack_df = pd.DataFrame(attack_results)
            model_performance = attack_df.groupby('model').agg({
                'attack_classification_accuracy': 'mean',
                'avg_latency_ms': 'mean',
                'success_rate': 'mean'
            }).round(4)
            
            performance_path = output_dir / 'model_performance_comparison.csv'
            model_performance.to_csv(performance_path, encoding='utf-8-sig')
            print(f"모델 성능 비교가 {performance_path}에 저장되었습니다.")
        
        # Pass@k 결과만 따로 저장
        passk_results = [r for r in results if 'pass_at_k' in r]
        if passk_results:
            passk_df = pd.DataFrame(passk_results)
            passk_path = output_dir / 'code_generation_passk.csv'
            passk_df.to_csv(passk_path, index=False, encoding='utf-8-sig')
            print(f"코드 생성 Pass@k 결과가 {passk_path}에 저장되었습니다.")

def create_unified_benchmark_table(test_set_results, output_dir):
    """모든 벤치마크 결과를 하나의 통합된 테이블로 생성"""
    unified_results = []
    
    # 각 모델과 테스트 타입에 대해 통합 행 생성
    all_models = set()
    for test_type, test_models in test_set_results.items():
        all_models.update(test_models.keys())
    
    for model in all_models:
        for test_type, test_models in test_set_results.items():
            if model in test_models:
                metrics = test_models[model]['metrics']
                
                # 기본 정보
                row = {
                    'model': model,
                    'type': test_type,
                }
                
                # 모든 메트릭 추가
                metric_mapping = {
                    # 공격 분류 메트릭
                    'attack_classification_accuracy': 'attack_accuracy',
                    'attack_classification_f1': 'attack_f1',
                    'attack_classification_precision': 'attack_precision',
                    'attack_classification_recall': 'attack_recall',
                    
                    # 정보 추출 메트릭
                    'ip_extraction_f1': 'ip_f1',
                    'port_extraction_f1': 'port_f1',
                    'protocol_extraction_f1': 'protocol_f1',
                    'overall_extraction_f1': 'extraction_f1',
                    
                    # 위협 탐지 메트릭
                    'threat_detection_accuracy': 'threat_accuracy',
                    'threat_detection_f1': 'threat_f1',
                    'threat_detection_precision': 'threat_precision',
                    'threat_detection_recall': 'threat_recall',
                    
                    # 자연어 생성 품질 메트릭
                    'bleu': 'bleu_score',
                    'rouge_l': 'rouge_l_score',
                    'semantic_similarity': 'cosine_similarity',
                    'response_quality_score': 'response_quality',
                    
                    # 도메인 특화 메트릭
                    'domain_specific_score': 'domain_score',
                    'overall_score': 'overall_score',
                    
                    # 코드 생성 메트릭
                    'pass_at_k': 'pass_at_3',
                    'passed_tasks': 'passed_tasks',
                    'total_tasks': 'total_tasks',
                    
                    # 성능 메트릭
                    'avg_latency_ms': 'avg_latency_ms',
                    'total_latency_ms': 'total_latency_ms',
                    'success_rate': 'success_rate',
                    'total_items': 'total_items'
                }
                
                # 모든 가능한 메트릭을 기본값 0으로 초기화
                for original_key, new_key in metric_mapping.items():
                    row[new_key] = metrics.get(original_key, 0)
                
                # 특별 처리: 코드 생성이 아닌 경우 pass@k 관련 필드는 null로 설정
                if test_type != 'code_generation':
                    row['pass_at_3'] = None
                    row['passed_tasks'] = None
                    row['total_tasks'] = None
                
                # 특별 처리: 코드 생성인 경우 다른 메트릭들은 null로 설정
                elif test_type == 'code_generation':
                    non_code_metrics = [
                        'attack_accuracy', 'attack_f1', 'attack_precision', 'attack_recall',
                        'ip_f1', 'port_f1', 'protocol_f1', 'extraction_f1',
                        'threat_accuracy', 'threat_f1', 'threat_precision', 'threat_recall',
                        'bleu_score', 'rouge_l_score', 'cosine_similarity', 'response_quality',
                        'domain_score', 'overall_score'
                    ]
                    for metric in non_code_metrics:
                        row[metric] = None
                
                unified_results.append(row)
    
    # DataFrame 생성 및 저장
    if unified_results:
        unified_df = pd.DataFrame(unified_results)
        
        # 컬럼 순서 정리
        column_order = [
            'model', 'type',
            'attack_accuracy', 'attack_f1', 'attack_precision', 'attack_recall',
            'ip_f1', 'port_f1', 'protocol_f1', 'extraction_f1',
            'threat_accuracy', 'threat_f1', 'threat_precision', 'threat_recall',
            'bleu_score', 'rouge_l_score', 'cosine_similarity', 'response_quality',
            'domain_score', 'overall_score',
            'pass_at_3', 'passed_tasks', 'total_tasks',
            'avg_latency_ms', 'success_rate', 'total_items'
        ]
        
        # 존재하는 컬럼만 선택
        available_columns = [col for col in column_order if col in unified_df.columns]
        unified_df = unified_df[available_columns]
        
        # 모델명으로 정렬 (Llama-PcapLog가 맨 위에 오도록)
        unified_df['sort_key'] = unified_df['model'].apply(lambda x: 0 if x == 'Llama-PcapLog' else 1)
        unified_df = unified_df.sort_values(['sort_key', 'model', 'type']).drop('sort_key', axis=1)
        
        # 소수점 반올림
        numeric_columns = unified_df.select_dtypes(include=[float]).columns
        unified_df[numeric_columns] = unified_df[numeric_columns].round(4)
        
        # 저장
        unified_path = output_dir / 'unified_benchmark_results.csv'
        unified_df.to_csv(unified_path, index=False, encoding='utf-8-sig')

def print_overall_rankings(results):
    """종합 순위 출력"""
    print("\n" + "="*80)
    print("종합 성능 순위")
    print("="*80)
    
    if not results:
        print("결과가 없습니다.")
        return
    
    # 공격 분류 정확도 기준 순위
    attack_results = [r for r in results if 'attack_classification_accuracy' in r and r['test_type'] != 'code_generation']
    if attack_results:
        from collections import defaultdict
        model_stats = defaultdict(list)
        
        for result in attack_results:
            model_stats[result['model']].append(result['attack_classification_accuracy'])
        
        model_averages = {model: sum(scores)/len(scores) for model, scores in model_stats.items()}
        sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
        
    
    # 코드 생성 Pass@k 순위
    passk_results = [r for r in results if 'pass_at_k' in r]
    if passk_results:
        print(f"\n코드 생성 Pass@3 순위:")
        passk_sorted = sorted(passk_results, key=lambda x: x['pass_at_k'], reverse=True)
        
        for rank, result in enumerate(passk_sorted, 1):
            model = result['model']
            passk = result['pass_at_k']
            passed = result['passed_tasks']
            total = result['total_tasks']
            

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive LLM benchmark')
    parser.add_argument('--benchmark_dir', type=str, default='.',
                       help='Directory containing benchmark data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    setup_logging()
    
    print("6개 모델 종합 벤치마크를 시작합니다...")
    print("모델 목록:")
    print("1. Llama-PcapLog")
    print("2. Llama-3.1-8B-Instruct")
    print("3. Qwen2-7B") 
    print("4. Gemma-3-4B-IT")
    print("5. Mistral-7B-Instruct")
    print("6. GPT-4o (API 키가 있는 경우)")
    
    print("\n테스트 구성:")
    print("- 기본 분석 테스트: 20개")
    print("- 고급 분석 테스트: 20개") 
    print("- 위협 탐지 테스트: 20개")
    print("- 코드 생성 테스트: 20개 (Pass@3)")
    print("- 총 80개 테스트")
    
    # 종합 벤치마크 수행
    results, detailed_results = run_comprehensive_benchmark(benchmark_dir, output_dir)
    
    print(f"\n모든 벤치마크가 완료되었습니다. 결과는 {output_dir} 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main() 
