from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, model_name: str = "choihyuunmin/mobile-Llama-3-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model(model_name)
    
    def load_model(self, model_name: str):
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_name = model_name
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    async def generate_response(self, message: str, model: str) -> Dict[str, str]:
        try:
            if not self.model or not self.tokenizer:
                raise Exception("Model not loaded")

            # 입력 텍스트를 토큰화
            inputs = self.tokenizer(message, return_tensors="pt")
            
            # 모델로부터 응답 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 생성된 텍스트 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {"response": response}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def generate_response_sync(self, message: str, model: str) -> Dict[str, str]:
        try:
            if not self.model or not self.tokenizer:
                raise Exception("Model not loaded")

            inputs = self.tokenizer(message, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"response": response}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise 