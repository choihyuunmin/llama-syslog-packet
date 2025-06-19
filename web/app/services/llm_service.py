from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import json

class LLMService:
    def __init__(self, model_name: str = "choihyuunmin/mobile-Llama-3-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate_response(self, question: str, context: Dict[str, Any]) -> str:
        """질문에 대한 응답 생성"""
        try:
            # 프롬프트 생성
            prompt = self._create_prompt(question, context)
            
            # 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 응답 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=500,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # 응답 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _create_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """컨텍스트를 포함한 프롬프트 생성"""
        prompt = f"""
        You are a network traffic analysis expert. Analyze the following network traffic data and answer the question.

        Context:
        - Basic Statistics:
          * Total Packets: {context['basic_stats']['total_packets']}
          * Total Sessions: {context['basic_stats']['total_sessions']}
          * Time Range: {context['basic_stats']['start_time']} to {context['basic_stats']['end_time']}
          * Total Bytes: {context['basic_stats']['total_bytes']}
        
        - Protocol Distribution:
          * Most Common Protocol: {context['protocol_dist']['most_common']}
          * Protocol Counts: {context['protocol_dist']['distribution']}
        
        - Traffic Pattern:
          * Busiest Hour: {context['traffic_pattern']['busiest_hour']}
          * Hourly Distribution: {context['traffic_pattern']['hourly_distribution']}
        
        - Security Analysis:
          * Security Level: {context['security_analysis']['security_level']}
          * Suspicious Patterns: {context['security_analysis']['suspicious_patterns']}

        Question: {question}

        Please provide a detailed analysis based on the above information.
        """
        return prompt 