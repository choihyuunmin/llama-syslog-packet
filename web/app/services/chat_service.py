import os
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from .rag_service import RAGService
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'choihyuunmin/LlamaTrace')
        self.offload_folder = os.getenv('OFFLOAD_FOLDER', 'offload')
        self.rag_service = RAGService()
        self.load_model()

    def load_model(self):
        try:
            # 토크나이저 설정
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # pad_token이 없으면 eos_token을 사용
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                pad_token_id=self.tokenizer.pad_token_id,
                offload_folder=self.offload_folder
            )
            
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_response_sync(self, message: str) -> Dict[str, Any]:
        try:
            # RAG를 통한 관련 컨텍스트 검색
            context = self.rag_service.get_relevant_context(message)
            
            # 프롬프트 구성
            prompt = f"""Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information, please answer the following question:
            {message}
            
            Answer:"""
            
            # 입력 토큰화 및 패딩
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # 응답 생성
            outputs = self.model.generate(
                **inputs,
                max_length=2048,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # 응답 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "response": response,
                "context_used": context
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def process_packet_data(self, packet_data: list) -> None:
        try:
            self.rag_service.process_packet_data(packet_data)
            logger.info("Packet data processed and stored successfully")
        except Exception as e:
            logger.error(f"Error processing packet data: {e}")
            raise 