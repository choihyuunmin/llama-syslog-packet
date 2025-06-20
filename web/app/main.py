from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import logging
from web.app.core.config import settings
from web.app.api.file_routes import router as file_router
from web.app.api.chat_routes import router as chat_router
from web.app.api.model_routes import router as model_router
from web.app.api.log_routes import router as log_router

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.app_name,
    description="API for analyzing Syslog and PCAP files and answering questions about network traffic",
    version=settings.app_version,
    debug=settings.debug
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# 업로드 디렉토리 생성
settings.upload_dir.mkdir(exist_ok=True)
logger.info(f"업로드 디렉토리 생성: {settings.upload_dir}")

# 라우터 등록
app.include_router(file_router, prefix="/api", tags=["files"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(model_router, prefix="/api", tags=["models"])
app.include_router(log_router, prefix="/api", tags=["logs"])

@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version
    }


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.on_event("startup")
async def startup_event():
    logger.info(f"{settings.app_name} v{settings.app_version} 시작됨")
    logger.info(f"서버 설정: {settings.host}:{settings.port}")
    logger.info(f"디버그 모드: {settings.debug}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{settings.app_name} 종료됨") 