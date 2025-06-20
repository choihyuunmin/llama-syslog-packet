#!/usr/bin/env python3
"""
SysPacket Analysis API 실행 스크립트
"""
import uvicorn
from web.app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 