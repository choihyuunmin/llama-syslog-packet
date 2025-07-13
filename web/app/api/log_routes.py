from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from web.app.services.syslog_analyzer import SyslogAnalyzer

router = APIRouter()
syslog_analyzer = SyslogAnalyzer()

@router.post("/upload-syslog")
async def upload_syslog(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path("uploads") / f"{timestamp}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        analysis_result = syslog_analyzer.analyze_syslog(str(file_path))
        
        return {
            "message": "File uploaded and analyzed successfully",
            "file_id": timestamp,
            "analysis": analysis_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-syslog")
async def analyze_syslog(file_id: str, filename: str) -> Dict[str, Any]:
    try:
        file_path = Path("uploads") / f"{file_id}_{filename}"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        analysis_result = syslog_analyzer.analyze_syslog(str(file_path))
        
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 