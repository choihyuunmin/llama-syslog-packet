from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
from typing import List, Dict, Any
import logging
from web.app.services.file_service import FileService
from web.app.core.utils import InvalidFileTypeError, FileAnalysisError
from web.app.core.dependencies import get_file_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_file(
    file: UploadFile = File(...),
    file_service: FileService = Depends(get_file_service)
) -> Dict[str, Any]:
    """
    파일을 업로드하고 분석합니다.
    
    Args:
        file: 업로드할 파일
        file_service: 파일 서비스 인스턴스
        
    Returns:
        Dict[str, Any]: 분석 결과
        
    Raises:
        HTTPException: 파일 업로드 또는 분석 실패 시
    """
    try:
        logger.info(f"파일 업로드 요청: {file.filename} ({file.size} bytes)")
        
        # 파일 내용 읽기
        content = await file.read()
        
        # 파일 저장
        file_path = await file_service.save_file(content, file.filename)
        
        # 파일 분석
        analysis_result = await file_service.analyze_file(file_path)
        
        logger.info(f"파일 분석 완료: {file.filename}")
        return analysis_result
        
    except InvalidFileTypeError as e:
        logger.warning(f"지원하지 않는 파일 타입: {file.filename} - {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        logger.warning(f"파일 유효성 검사 실패: {file.filename} - {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileAnalysisError as e:
        logger.error(f"파일 분석 실패: {file.filename} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"파일 분석 실패: {str(e)}")
    except Exception as e:
        logger.error(f"파일 처리 중 예상치 못한 오류: {file.filename} - {str(e)}")
        raise HTTPException(status_code=500, detail="파일 처리 중 오류가 발생했습니다")


@router.get("/files", response_model=Dict[str, List[Dict[str, Any]]])
async def list_files(
    file_service: FileService = Depends(get_file_service)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    업로드된 파일 목록을 반환합니다.
    
    Args:
        file_service: 파일 서비스 인스턴스
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: 파일 목록
    """
    try:
        files = file_service.list_files()
        logger.info(f"파일 목록 조회: {len(files)}개 파일")
        return {"files": files}
    except Exception as e:
        logger.error(f"파일 목록 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="파일 목록 조회 중 오류가 발생했습니다")


@router.delete("/files/{filename}")
async def delete_file(
    filename: str,
    file_service: FileService = Depends(get_file_service)
) -> Dict[str, str]:
    """
    파일을 삭제합니다.
    
    Args:
        filename: 삭제할 파일명
        file_service: 파일 서비스 인스턴스
        
    Returns:
        Dict[str, str]: 삭제 결과
    """
    try:
        success = file_service.delete_file(filename)
        if success:
            logger.info(f"파일 삭제 완료: {filename}")
            return {"message": f"파일 '{filename}'이(가) 삭제되었습니다"}
        else:
            logger.warning(f"파일 삭제 실패: {filename}")
            raise HTTPException(status_code=404, detail=f"파일 '{filename}'을(를) 찾을 수 없습니다")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 삭제 중 오류: {filename} - {str(e)}")
        raise HTTPException(status_code=500, detail="파일 삭제 중 오류가 발생했습니다")


@router.get("/files/{filename}/info")
async def get_file_info(
    filename: str,
    file_service: FileService = Depends(get_file_service)
) -> Dict[str, Any]:
    """
    특정 파일의 정보를 반환합니다.
    
    Args:
        filename: 파일명
        file_service: 파일 서비스 인스턴스
        
    Returns:
        Dict[str, Any]: 파일 정보
    """
    try:
        file_path = file_service.upload_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"파일 '{filename}'을(를) 찾을 수 없습니다")
        
        stat = file_path.stat()
        from web.app.core.utils import get_file_type
        file_type = get_file_type(file_path)
        
        file_info = {
            "name": filename,
            "type": file_type,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": stat.st_ctime,
            "modified": stat.st_mtime
        }
        
        logger.info(f"파일 정보 조회: {filename}")
        return file_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 정보 조회 실패: {filename} - {str(e)}")
        raise HTTPException(status_code=500, detail="파일 정보 조회 중 오류가 발생했습니다") 