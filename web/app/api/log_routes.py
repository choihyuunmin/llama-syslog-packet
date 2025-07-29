from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from services.file_service import FileService
from core.dependencies import get_file_service

router = APIRouter()

@router.post("/upload-syslog")
async def upload_syslog(
    file: UploadFile = File(...),
    file_service: FileService = Depends(get_file_service)
) -> Dict[str, Any]:
    try:
        # Save file using file service
        file_content = await file.read()
        file_path = await file_service.save_file(file_content, file.filename)
        
        # Analyze file
        analysis_result = await file_service.analyze_file(file_path)
        
        return {
            "message": "File uploaded and analyzed successfully",
            "file_id": file_path.stem,
            "analysis": analysis_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-syslog")
async def analyze_syslog(
    file_id: str, 
    filename: str,
    file_service: FileService = Depends(get_file_service)
) -> Dict[str, Any]:
    try:
        file_path = file_service.upload_dir / f"{file_id}_{filename}"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        analysis_result = await file_service.analyze_file(file_path)
        
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 