from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .api.file_routes import router as file_router
from .api.chat_routes import router as chat_router
from .api.model_routes import router as model_router
from .api.log_routes import router as log_router

app = FastAPI(
    title="Packet Analysis API",
    description="API for analyzing PCAP files and answering questions about network traffic",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드 디렉토리 설정
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 라우터 등록
app.include_router(file_router, prefix="/api", tags=["files"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(model_router, prefix="/api", tags=["models"])
app.include_router(log_router, prefix="/api", tags=["logs"])

@app.get("/health")
async def health_check() -> dict[str, str]:
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"} 