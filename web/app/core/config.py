from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    
    # 기본 설정
    app_name: str = Field(default="SysPacket Analysis API", description="애플리케이션 이름")
    app_version: str = Field(default="1.0.0", description="애플리케이션 버전")
    debug: bool = Field(default=False, description="디버그 모드")
    
    # 서버 설정
    host: str = Field(default="0.0.0.0", description="서버 호스트")
    port: int = Field(default=8000, description="서버 포트")
    
    # CORS 설정
    cors_origins: List[str] = Field(
        default=["*"], 
    )
    cors_credentials: bool = Field(default=True, description="CORS 자격 증명 허용")
    cors_methods: List[str] = Field(
        default=["*"], 
    )
    cors_headers: List[str] = Field(
        default=["*"], 
    )
    
    # 파일 업로드 설정
    upload_dir: Path = Field(default=Path("uploads"), description="업로드 디렉토리")
    max_file_size: int = Field(default=100 * 1024 * 1024, description="최대 파일 크기 (bytes)")
    allowed_extensions: List[str] = Field(
        default=[".pcap", ".pcapng", ".log", ".txt"], 
        description="허용된 파일 확장자"
    )
    
    # OpenAI 설정
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API 키")
    
    # 모델 설정
    default_model: str = Field(default="choihyuunmin/LlamaTrace", description="기본 모델")
    model_cache_dir: Optional[Path] = Field(default=None, description="모델 캐시 디렉토리")
    
    # 로깅 설정
    log_level: str = Field(default="INFO", description="로그 레벨")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="로그 포맷"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 전역 설정 인스턴스
settings = Settings()

# 사용 가능한 모델 정의
AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
    "LLaMa-PcapLog": {
        "name": "choihyuunmin/LLaMa-PcapLog",
        "description": "syslog, packet 분석 모델",
        "type": "llama"
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "description": "gpt-4o 모델",
        "type": "gpt"
    },
    "LlamaTrace": {
        "name": "choihyuunmin/LlamaTrace",
        "description": "syslog, packet 분석 모델",
        "type": "llama"
    }
} 