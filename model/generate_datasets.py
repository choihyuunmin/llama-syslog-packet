import argparse
import copy
import dataclasses
import json
import logging
import os
import random
import string
import sys
import time
from datetime import datetime
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
    return_text=False,
):
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    client = OpenAI()
    completions = []

    for prompt in prompts:

        shared_kwargs = {
            "model": model_name,
            "max_tokens": decoding_args.max_tokens,
            "temperature": decoding_args.temperature,
            "top_p": decoding_args.top_p,
            "n": decoding_args.n,
            "stream": decoding_args.stream,
            "stop": decoding_args.stop,
            "presence_penalty": decoding_args.presence_penalty,
            "frequency_penalty": decoding_args.frequency_penalty,
        }
        messages = [{"role": "user", "content": prompt}]                
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

    if return_text:
        completions = [completion["text"] for completion in completions]
    if decoding_args.n > 1:
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        (completions,) = completions
    return completions

class DatasetGenerator:
    def __init__(
        self,
        output_dir="processed",
        model_name="gpt-4o-mini",
        num_instructions_to_generate=10,
        num_prompt_instructions=3,
        temperature=1.0,
        top_p=1.0,
    ):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.num_instructions_to_generate = num_instructions_to_generate
        self.num_prompt_instructions = num_prompt_instructions
        self.temperature = temperature
        self.top_p = top_p
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        
    def encode_prompt(self, prompt_instructions, context_data=None):
        prompt = """You are a network security expert. Generate question-answer pairs about network packet or system log analysis.
                    Each response should be in the following JSON format:
                    {
                        "instruction": "your question here",
                        "input": "context_data",
                        "output": "your answer here"
                    }

                    Important: You must use the provided input data to generate relevant questions and answers.
                    The input field in your response should contain the actual data you used from the provided context.
                    Your questions should be specific to the patterns, anomalies, or security events visible in the input data.
                """
        
        if context_data:
            if "pcap_data" in context_data:
                sample_data = context_data["pcap_data"]
            
            if "syslog_data" in context_data:
                sample_data = context_data["syslog_data"]
        
        prompt += "\nHere are some examples of question-answer pairs:\n\n"
        
        for idx, task_dict in enumerate(prompt_instructions):
            question = task_dict["instruction"]
            answer = task_dict["output"]
            
            prompt += f"Example {idx + 1}:\n"
            prompt += f"Instruction: {question}\n"
            prompt += f"Input: {sample_data}\n"
            prompt += f"Output: {answer}\n\n"
            
        prompt += """Now generate a new question-answer pair in the same JSON format based on the provided data.
                    Remember to:
                    1. Use the actual input data in your response
                    2. Create questions that are specifically relevant to the patterns or events in the input data
                    3. Provide detailed answers that reference specific elements from the input data
                    4. Ensure your response follows the exact JSON format shown above
                """
        return prompt
    
    def post_process_gpt3_response(self, response):
        if response is None:
            return []
            
        try:
            json_str = response["text"].strip()
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return []
                
            json_str = json_str[start_idx:end_idx]
            instruction = json.loads(json_str)
            
            if not isinstance(instruction, dict):
                return []
            if "instruction" not in instruction or "output" not in instruction:
                return []
                
            question = instruction["instruction"].strip()
            input_data = instruction["input"].strip()
            answer = instruction["output"].strip()
            
            if len(question.split()) <= 3 or len(question.split()) > 100:
                return []
                
            if question[0] in string.punctuation or not question[0].isascii():
                return []
                
            return [{
                "instruction": question,
                "input": input_data,
                "output": answer
            }]
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse GPT response: {e}")
            return []
    
    def generate_additional_instructions(
        self,
        seed_instructions,
        context_data,
        existing_instructions=None
    ):
        if existing_instructions is None:
            existing_instructions = []
            
        all_instructions = [d["instruction"] for d in seed_instructions] + [
            d["instruction"] for d in existing_instructions
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
                    decoding_args=decoding_args
                )
                
                instruction_data = []
                for result in results:
                    new_instructions = self.post_process_gpt3_response(result)
                    instruction_data += new_instructions
                    
                total = len(instruction_data)
                keep = 0
                
                for instruction_data_entry in instruction_data:
                    question = instruction_data_entry["instruction"]
                    
                    # Skip if question already exists
                    if question in all_instructions:
                        continue
                        
                    keep += 1
                    existing_instructions.append(instruction_data_entry)
                    all_instructions.append(question)
                    all_instruction_tokens.append(self.scorer._tokenizer.tokenize(question))
                    
                logger.info(f"Generated {total} instructions, kept {keep} instructions")
                
            except Exception as e:
                logger.error(f"Error occurred while generating additional questions: {e}")
                raise
                
        return existing_instructions
    
    def generate_pcap_dataset(self, pcap_file):
        try:            
            # initialize and process PCAP processor
            processor = PcapProcessor(pcap_file)
            packets = processor.process_pcap()
            
            all_instructions = []
            
            context_data = {
                "pcap_data": str(packets)
            }
            
            seed_instructions = processor.generate_dataset()

            if not seed_instructions:
                logger.warning("Generated dataset is empty.")
                return None
            
            if len(seed_instructions[0]["input"]) < 3:
                return None

            # 추가 질문 생성
            new_instructions = self.generate_additional_instructions(
                seed_instructions,
                context_data=context_data
            )
            
            # 기존 질문 추가
            for instruction in seed_instructions:
                all_instructions.append({
                    "instruction": instruction["instruction"],
                    "input": packets,
                    "output": instruction["output"],
                })
            
            # 새로 생성된 질문 추가
            for instruction in new_instructions:
                all_instructions.append({
                    "instruction": instruction["instruction"],
                    "input": packets,
                    "output": instruction["output"],
                })
            
            if not all_instructions:
                logger.warning("Generated dataset is empty.")
                return None
            
            # 파일 저장
            output_path = self._get_output_path("pcap_dataset")
            self._save_dataset(all_instructions, output_path)
            
            logger.info(f"PCAP dataset generation completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error occurred while generating PCAP dataset: {e}")
            raise
    
    def generate_syslog_dataset(self, syslog_file):
        try:
            # Syslog 프로세서 초기화 및 처리
            processor = SyslogProcessor(syslog_file)
            syslog_data = processor.process_logs()
            
            # 초기 데이터셋 생성
            seed_instructions = processor.generate_dataset()
            if not seed_instructions:
                logger.warning("Generated Syslog dataset is empty.")
                return None
            
            if len(seed_instructions[0]["input"]) < 3:
                logger.warning("Generated Syslog dataset is empty.")
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
                    "instruction": instruction["instruction"],
                    "input": syslog_data,
                    "output": instruction["output"],
                })
            
            # 새로 생성된 질문 추가
            for instruction in new_instructions:
                all_instructions.append({
                    "instruction": instruction["instruction"],
                    "input": syslog_data,
                    "output": instruction["output"],
                })
            
            # 파일 저장
            output_path = self._get_output_path("syslog_dataset")
            self._save_dataset(all_instructions, output_path)
            
            logger.info(f"Syslog dataset generation completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error occurred while generating Syslog dataset: {e}")
            raise
    
    def _get_output_path(self, base_name):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}.json"
        return self.output_dir / filename
    
    def _save_dataset(self, dataset, output_path):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"Dataset saved successfully: {output_path}")
        except Exception as e:
            logger.error(f"Error occurred while saving dataset: {e}")
            raise

def validate_input_file(file_path):
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

def find_target_files(root_path):
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root_path}")
        
    for file_path in root.rglob("*"):
        if file_path.is_file():
            yield file_path

def main():
    parser = argparse.ArgumentParser(description="packet/syslog dataset generator")
    parser.add_argument("--pcap-dir", type=str, help="directory containing PCAP files")
    parser.add_argument("--syslog-dir", type=str, help="directory containing Syslog files")
    parser.add_argument("--output-dir", type=str, default="processed")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--num-instructions", type=int, default=100)
    parser.add_argument("--num-prompt-instructions", type=int, default=3)
    parser.add_argument("--request-batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    
    args = parser.parse_args()
    
    try:
        # Initialize dataset generator
        generator = DatasetGenerator(
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_instructions_to_generate=args.num_instructions,
            num_prompt_instructions=args.num_prompt_instructions,
            temperature=args.temperature,
            top_p=1.0,
        )
        
        # 모든 파일 경로 수집
        all_files = []
        
        # PCAP 파일 수집
        if args.pcap_dir:
            pcap_files = [f for f in find_target_files(args.pcap_dir)]
            if not pcap_files:
                logger.warning(f"No PCAP files found to process: {args.pcap_dir}")
            else:
                logger.info(f"Found {len(pcap_files)} PCAP files.")
                all_files.extend([("pcap", str(f)) for f in pcap_files])
        
        # Syslog 파일 수집
        if args.syslog_dir:
            syslog_files = [f for f in find_target_files(args.syslog_dir)]
            if not syslog_files:
                logger.warning(f"No Syslog files found to process: {args.syslog_dir}")
            else:
                logger.info(f"Found {len(syslog_files)} Syslog files.")
                all_files.extend([("syslog", str(f)) for f in syslog_files])
        
        random.shuffle(all_files)
        for file_type, file_path in all_files:
            try:
                if file_type == "pcap":
                    logger.info(f"Processing PCAP file: {file_path}")
                    generator.generate_pcap_dataset(file_path)
                else:  # syslog
                    logger.info(f"Processing Syslog file: {file_path}")
                    generator.generate_syslog_dataset(file_path)
            except Exception as e:
                logger.error(f"Error occurred while processing file ({file_path}): {e}")
                continue
        
        logger.info("All files processing completed.")
        
    except Exception as e:
        logger.error(f"Error occurred while generating dataset: {e}")
        raise

if __name__ == "__main__":
    main()