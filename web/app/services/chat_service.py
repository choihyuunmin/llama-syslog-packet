from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Optional, Any
import logging
from core.config import settings, AVAILABLE_MODELS

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


class ChatService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.default_model
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        
        # 모델 로딩
        self.load_model(self.model_name)
    
    def load_model(self, model_name: str) -> None:
        try:
            if model_name not in AVAILABLE_MODELS:
                raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
            
            model_info = AVAILABLE_MODELS[model_name]
            actual_model_name = model_info["name"]
            
            logger.info(f"모델 로딩 시작: {model_name} ({actual_model_name})")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                actual_model_name,
                cache_dir=settings.model_cache_dir
            )
            
            # Set pad_token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_name,
                cache_dir=settings.model_cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.model_name = model_name
            logger.info(f"모델 로딩 완료: {model_name}")
            
        except ValueError as e:
            logger.error(f"모델 로딩 실패 - 잘못된 모델명: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            raise ModelLoadError(f"모델 로딩 실패: {str(e)}")

    async def generate_response(self, message: str, model: Optional[str] = None) -> Dict[str, Any]:
        try:
            # 모델 변경이 요청된 경우
            if model and model != self.model_name:
                self.load_model(model)
            
            if not self.model or not self.tokenizer:
                raise ModelLoadError("모델이 로딩되지 않았습니다")

            logger.info(f"응답 생성 시작: {len(message)}자 메시지")
            
            # 입력 텍스트를 토큰화
            inputs = self.tokenizer(message, return_tensors="pt")
            
            # 모델로부터 응답 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # 생성된 텍스트 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 입력 메시지 제거하여 실제 응답만 추출
            if response.startswith(message):
                response = response[len(message):].strip()
            
            result = {
                "response": response,
                "model": self.model_name,
                "input_length": len(message),
                "output_length": len(response)
            }
            
            logger.info(f"응답 생성 완료: {len(response)}자 응답")
            return result
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {str(e)}")
            raise ModelLoadError(f"응답 생성 실패: {str(e)}")

    def generate_response_sync(self, message: str, model: Optional[str] = None) -> Dict[str, Any]:
        try:
            # 모델 변경이 요청된 경우
            if model and model != self.model_name:
                self.load_model(model)
            
            if not self.model or not self.tokenizer:
                raise ModelLoadError("모델이 로딩되지 않았습니다")

            logger.info(f"동기 응답 생성 시작: {len(message)}자 메시지")
            
            inputs = self.tokenizer(message, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 입력 메시지 제거하여 실제 응답만 추출
            if response.startswith(message):
                response = response[len(message):].strip()
            
            result = {
                "response": response,
                "model": self.model_name,
                "input_length": len(message),
                "output_length": len(response)
            }
            
            logger.info(f"동기 응답 생성 완료: {len(response)}자 응답")
            return result
            
        except Exception as e:
            logger.error(f"동기 응답 생성 실패: {str(e)}")
            raise ModelLoadError(f"동기 응답 생성 실패: {str(e)}")

    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        return AVAILABLE_MODELS

    def get_current_model(self) -> str:
        return self.model_name 