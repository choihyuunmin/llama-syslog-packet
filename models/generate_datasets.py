import argparse
import copy
import dataclasses
import json
import logging
import os
import random
import re
import string
import sys
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from openai import OpenAI
from rouge_score import rouge_scorer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from processors.pcap_processor import PcapProcessor
from processors.syslog_processor import SyslogProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

def openai_completion(
    prompts,
    decoding_args,
    model_name,
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
):
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(np.ceil(num_prompts / batch_size)))
    ]

    client = OpenAI()
    completions = []
    for prompt_batch in prompt_batches:
        batch_decoding_args = copy.deepcopy(decoding_args)

        while True:
            try:
                shared_kwargs = {
                    "model": model_name,
                    "max_tokens": batch_decoding_args.max_tokens,
                    "temperature": batch_decoding_args.temperature,
                    "top_p": batch_decoding_args.top_p,
                    "n": batch_decoding_args.n,
                    "stream": batch_decoding_args.stream,
                    "stop": batch_decoding_args.stop,
                    "presence_penalty": batch_decoding_args.presence_penalty,
                    "frequency_penalty": batch_decoding_args.frequency_penalty,
                }
                
                # Convert prompts to messages format for chat completion
                messages = [{"role": "user", "content": prompt} for prompt in prompt_batch]
                
                completion_batch = client.chat.completions.create(
                    messages=messages,
                    **shared_kwargs
                )
                
                choices = []
                for choice in completion_batch.choices:
                    choices.append({
                        "text": choice.message.content,
                        "finish_reason": choice.finish_reason,
                        "total_tokens": completion_batch.usage.total_tokens
                    })
                
                completions.extend(choices)
                break
            except Exception as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)

    if return_text:
        completions = [completion["text"] for completion in completions]
    if decoding_args.n > 1:
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        (completions,) = completions
    return completions

class DatasetGenerator:
    """학습 데이터셋 생성기
    
    이 클래스는 PCAP 및 Syslog 파일을 분석하여
    네트워크 보안 관련 질문-답변 데이터셋을 생성합니다.
    
    Attributes:
        output_dir: 출력 디렉토리
        model_name: 사용할 GPT 모델 이름
        num_instructions_to_generate: 생성할 추가 질문 수
        num_prompt_instructions: 프롬프트에 사용할 예시 질문 수
        request_batch_size: 한 번에 처리할 요청 수
        temperature: 생성 다양성 조절
        top_p: 토큰 선택 확률 임계값
        num_cpus: 사용할 CPU 코어 수
    """
    
    def __init__(
        self,
        output_dir="processed",
        model_name="gpt-4o-mini",
        num_instructions_to_generate=100,
        num_prompt_instructions=3,
        request_batch_size=2,
        temperature=1.0,
        top_p=1.0,
        num_cpus=16,
    ):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.num_instructions_to_generate = num_instructions_to_generate
        self.num_prompt_instructions = num_prompt_instructions
        self.request_batch_size = request_batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.num_cpus = num_cpus
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        
    def encode_prompt(self, prompt_instructions, context_data=None):
        """프롬프트 인코딩.
        
        Args:
            prompt_instructions: 예시 질문-답변 쌍
            context_data: 네트워크 패킷 및 로그 데이터
            
        Returns:
            인코딩된 프롬프트
        """
        prompt = """You are a network security expert. Generate question-answer pairs about network packet and system log analysis.
                    Each response should be in the following JSON format:
                    {
                        "question": "your question here",
                        "answer": "your answer here",
                        "category": "category in auth|security|system|network|application"
                    }

                    Here is the network packet and system log data to analyze:

                """
        
        # Add context data if available
        if context_data:
            if "pcap_data" in context_data:
                prompt += "\nNetwork Packet Data:\n"
                prompt += context_data["pcap_data"]
                prompt += "\n"
            
            if "syslog_data" in context_data:
                prompt += "\nSystem Log Data:\n"
                prompt += context_data["syslog_data"]
                prompt += "\n"
        
        prompt += "\nHere are some examples of question-answer pairs:\n\n"
        
        for idx, task_dict in enumerate(prompt_instructions):
            question = task_dict["question"]
            answer = task_dict["answer"]
            
            prompt += f"Example {idx + 1}:\n"
            prompt += f"Question: {question}\n"
            prompt += f"Answer: {answer}\n\n"
            
        prompt += "Now generate a new question-answer pair in the same JSON format based on the provided data:\n"
        return prompt
    
    def post_process_gpt3_response(self, num_prompt_instructions, response):
        if response is None:
            return []
            
        try:
            # Extract JSON from the response
            json_str = response["text"].strip()
            # Find the first occurrence of a JSON object
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return []
                
            json_str = json_str[start_idx:end_idx]
            instruction = json.loads(json_str)
            
            # Validate the structure
            if not isinstance(instruction, dict):
                return []
            if "question" not in instruction or "answer" not in instruction:
                return []
                
            question = instruction["question"].strip()
            answer = instruction["answer"].strip()
            
            # Basic filtering
            if len(question.split()) <= 3 or len(question.split()) > 150:
                return []
                
            if question[0] in string.punctuation or not question[0].isascii():
                return []
                
            return [{
                "question": question,
                "answer": answer
            }]
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse GPT response: {e}")
            return []
    
    def find_word_in_string(self, word, string):
        """문자열에서 단어 검색.
        
        Args:
            word: 검색할 단어
            string: 검색 대상 문자열
            
        Returns:
            단어 존재 여부
        """
        return re.compile(r"\b({0})\b".format(word), flags=re.IGNORECASE).search(string)
    
    def generate_additional_instructions(
        self,
        seed_instructions,
        context_data=None,
        existing_instructions=None
    ):
        if existing_instructions is None:
            existing_instructions = []
            
        all_instructions = [d["question"] for d in seed_instructions] + [
            d["question"] for d in existing_instructions
        ]
        all_instruction_tokens = [
            self.scorer._tokenizer.tokenize(inst) for inst in all_instructions
        ]
        
        request_idx = 0
        while len(existing_instructions) < self.num_instructions_to_generate:
            request_idx += 1
            batch_inputs = []
            
            prompt_instructions = random.sample(seed_instructions, self.num_prompt_instructions)
            prompt = self.encode_prompt(prompt_instructions, context_data)
            batch_inputs.append(prompt)
                
            decoding_args = OpenAIDecodingArguments(
                temperature=self.temperature,
                n=1,
                max_tokens=5072,
                top_p=self.top_p,
                stop=["\n20", "20.", "20."]
            )
            
            try:
                results = openai_completion(
                    prompts=batch_inputs,
                    model_name=self.model_name,
                    batch_size=self.request_batch_size,
                    decoding_args=decoding_args
                )
                
                instruction_data = []
                for result in results:
                    new_instructions = self.post_process_gpt3_response(
                        self.num_prompt_instructions, result
                    )
                    instruction_data += new_instructions
                    
                total = len(instruction_data)
                keep = 0
                
                for instruction_data_entry in instruction_data:
                    question = instruction_data_entry["question"]
                    
                    # Skip if question already exists
                    if question in all_instructions:
                        continue
                        
                    keep += 1
                    existing_instructions.append(instruction_data_entry)
                    all_instructions.append(question)
                    all_instruction_tokens.append(self.scorer._tokenizer.tokenize(question))
                    
                logger.info(f"Generated {total} instructions, kept {keep} instructions")
                
            except Exception as e:
                logger.error(f"추가 질문 생성 중 오류 발생: {e}")
                raise
                
        return existing_instructions
    
    def generate_pcap_dataset(self, pcap_file):
        try:
            logger.info(f"PCAP 파일 처리 시작: {pcap_file}")
            
            # PCAP 프로세서 초기화 및 처리
            processor = PcapProcessor(pcap_file)
            pcap_data = processor.process_pcap()
            
            # 초기 데이터셋 생성
            seed_instructions = processor.generate_dataset()
            if not seed_instructions:
                logger.warning("생성된 PCAP 데이터셋이 비어있습니다.")
                return None
            
            # 컨텍스트 데이터 준비
            context_data = {
                "pcap_data": str(pcap_data) if pcap_data is not None else "No PCAP data available"
            }
            
            # 추가 질문 생성
            new_instructions = self.generate_additional_instructions(
                seed_instructions,
                context_data=context_data
            )
            
            # 기존 질문과 새로 생성된 질문 합치기
            all_instructions = []
            
            # 기존 질문 추가
            for instruction in seed_instructions:
                all_instructions.append({
                    "question": instruction["question"],
                    "answer": instruction["answer"],
                    "category": instruction.get("category", "network"),
                })
            
            # 새로 생성된 질문 추가
            for instruction in new_instructions:
                all_instructions.append({
                    "question": instruction["question"],
                    "answer": instruction["answer"],
                    "category": instruction.get("category", "network"),
                })
            
            # 파일 저장
            output_path = self._get_output_path("pcap_dataset")
            self._save_dataset(all_instructions, output_path)
            
            logger.info(f"PCAP 데이터셋 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PCAP 데이터셋 생성 중 오류 발생: {e}")
            raise
    
    def generate_syslog_dataset(self, syslog_file):
        try:
            # Syslog 프로세서 초기화 및 처리
            processor = SyslogProcessor(syslog_file)
            syslog_data = processor.process_logs()
            
            # 초기 데이터셋 생성
            seed_instructions = processor.generate_dataset()
            if not seed_instructions:
                logger.warning("생성된 Syslog 데이터셋이 비어있습니다.")
                return None
            
            # 컨텍스트 데이터 준비
            context_data = {
                "syslog_data": str(syslog_data) if syslog_data is not None else "No Syslog data available"
            }
            
            # 추가 질문 생성
            new_instructions = self.generate_additional_instructions(
                seed_instructions,
                context_data=context_data
            )
            
            # 기존 질문과 새로 생성된 질문 합치기
            all_instructions = []
            
            # 기존 질문 추가
            for instruction in seed_instructions:
                all_instructions.append({
                    "question": instruction["question"],
                    "answer": instruction["answer"],
                    "category": instruction.get("category", "system"),
                })
            
            # 새로 생성된 질문 추가
            for instruction in new_instructions:
                all_instructions.append({
                    "question": instruction["question"],
                    "answer": instruction["answer"],
                    "category": instruction.get("category", "system"),
                })
            
            # 파일 저장
            output_path = self._get_output_path("syslog_dataset")
            self._save_dataset(all_instructions, output_path)
            
            logger.info(f"Syslog 데이터셋 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Syslog 데이터셋 생성 중 오류 발생: {e}")
            raise
    
    def _get_output_path(self, base_name):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}.json"
        return self.output_dir / filename
    
    def _save_dataset(self, dataset, output_path):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"데이터셋 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"데이터셋 저장 중 오류 발생: {e}")
            raise

def validate_input_file(file_path):
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

def find_target_files(root_path):
    """지정된 경로 아래의 모든 파일을 재귀적으로 찾습니다.
    
    Args:
        root_path: 검색을 시작할 루트 경로
        
    Yields:
        찾은 파일의 경로
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {root_path}")
        
    for file_path in root.rglob("*"):
        if file_path.is_file():
            yield file_path

def main():
    parser = argparse.ArgumentParser(description="packet/syslog 데이터셋 생성기")
    parser.add_argument("--pcap-dir", type=str, help="PCAP 파일이 있는 디렉토리 경로")
    parser.add_argument("--syslog-dir", type=str, help="Syslog 파일이 있는 디렉토리 경로")
    parser.add_argument("--output-dir", type=str, default="processed", help="출력 디렉토리")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini", help="사용할 GPT 모델 이름")
    parser.add_argument("--num-instructions", type=int, default=100, help="생성할 추가 질문 수")
    parser.add_argument("--num-prompt-instructions", type=int, default=3, help="프롬프트에 사용할 예시 질문 수")
    parser.add_argument("--request-batch-size", type=int, default=2, help="한 번에 처리할 요청 수")
    parser.add_argument("--temperature", type=float, default=1.0, help="생성 다양성 조절")
    
    args = parser.parse_args()
    
    if not args.pcap_dir and not args.syslog_dir:
        parser.error("최소한 하나의 입력 디렉토리(--pcap-dir 또는 --syslog-dir)를 지정해야 합니다.")
    
    try:
        # Initialize dataset generator
        generator = DatasetGenerator(
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_instructions_to_generate=args.num_instructions,
            num_prompt_instructions=args.num_prompt_instructions,
            request_batch_size=args.request_batch_size,
            temperature=args.temperature,
            top_p=1.0,
            num_cpus=16
        )
        
        # PCAP 파일 처리
        if args.pcap_dir:
            logger.info("PCAP 파일 처리 시작")
            pcap_files = [f for f in find_target_files(args.pcap_dir)]
            if not pcap_files:
                logger.warning(f"처리할 PCAP 파일을 찾을 수 없습니다: {args.pcap_dir}")
            else:
                logger.info(f"총 {len(pcap_files)}개의 PCAP 파일을 찾았습니다.")
                for file_path in pcap_files:
                    try:
                        logger.info(f"PCAP 파일 처리 시작: {file_path}")
                        generator.generate_pcap_dataset(str(file_path))
                    except Exception as e:
                        logger.error(f"PCAP 파일 처리 중 오류 발생 ({file_path}): {e}")
                        continue
                logger.info("모든 PCAP 파일 처리가 완료되었습니다.")
        
        # Syslog 파일 처리
        if args.syslog_dir:
            logger.info("Syslog 파일 처리 시작")
            syslog_files = [f for f in find_target_files(args.syslog_dir)]
            if not syslog_files:
                logger.warning(f"처리할 Syslog 파일을 찾을 수 없습니다: {args.syslog_dir}")
            else:
                logger.info(f"총 {len(syslog_files)}개의 Syslog 파일을 찾았습니다.")
                for file_path in syslog_files:
                    try:
                        generator.generate_syslog_dataset(str(file_path))
                    except Exception as e:
                        logger.error(f"Syslog 파일 처리 중 오류 발생 ({file_path}): {e}")
                        continue
                logger.info("모든 Syslog 파일 처리가 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"데이터셋 생성 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 