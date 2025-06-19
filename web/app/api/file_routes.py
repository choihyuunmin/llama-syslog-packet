from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from typing import List
from ..services.file_service import FileService

router = APIRouter()
file_service = FileService(Path("uploads"))

@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_path = await file_service.save_file(content, file.filename)
        analysis_result = await file_service.analyze_file(file_path)
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def list_files():
    try:
        files = file_service.list_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 