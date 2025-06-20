from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from web.app.core.utils import analyze_pcap, analyze_log, get_file_type, validate_file_upload
from web.app.core.config import settings

logger = logging.getLogger(__name__)


class FileService:
    """파일 업로드 및 분석을 담당하는 서비스 클래스"""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        """
        FileService 초기화
        
        Args:
            upload_dir: 업로드 디렉토리 경로 (None인 경우 설정에서 가져옴)
        """
        self.upload_dir = upload_dir or settings.upload_dir
        self.upload_dir.mkdir(exist_ok=True)
        logger.info(f"FileService 초기화됨. 업로드 디렉토리: {self.upload_dir}")

    async def save_file(self, file_content: bytes, filename: str) -> Path:
        """
        파일을 업로드 디렉토리에 저장합니다.
        
        Args:
            file_content: 파일 내용 (bytes)
            filename: 파일명
            
        Returns:
            Path: 저장된 파일 경로
            
        Raises:
            ValueError: 파일 크기가 제한을 초과할 때
            InvalidFileTypeError: 지원하지 않는 파일 타입일 때
        """
        try:
            # 파일 유효성 검사
            validate_file_upload(filename, len(file_content))
            
            # 파일명 중복 방지
            file_path = self._get_unique_file_path(filename)
            
            # 파일 저장
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
            
            logger.info(f"파일 저장 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"파일 저장 실패: {str(e)}")
            raise

    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        파일을 분석합니다.
        
        Args:
            file_path: 분석할 파일 경로
            
        Returns:
            Dict[str, Any]: 분석 결과
            
        Raises:
            FileAnalysisError: 파일 분석 중 오류 발생 시
        """
        try:
            start_time = datetime.now()
            file_type = get_file_type(file_path)
            
            logger.info(f"파일 분석 시작: {file_path} (타입: {file_type})")
            
            analysis_result = {}
            if file_type == 'pcap':
                analysis_result = analyze_pcap(file_path)
            elif file_type == 'log':
                analysis_result = analyze_log(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
            
            end_time = datetime.now()
            analysis_time = (end_time - start_time).total_seconds()
            
            result = {
                "filename": file_path.name,
                "file_type": file_type,
                "file_size": file_path.stat().st_size,
                "analysis_time": analysis_time,
                "analysis": analysis_result
            }
            
            logger.info(f"파일 분석 완료: {file_path} (소요시간: {analysis_time:.2f}초)")
            return result
            
        except Exception as e:
            logger.error(f"파일 분석 실패: {file_path} - {str(e)}")
            raise

    def list_files(self) -> List[Dict[str, Any]]:
        """
        업로드 디렉토리의 파일 목록을 반환합니다.
        
        Returns:
            List[Dict[str, Any]]: 파일 정보 목록
        """
        try:
            files = []
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_type = get_file_type(file_path)
                    stat = file_path.stat()
                    
                    files.append({
                        "name": file_path.name,
                        "type": file_type,
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # 파일명으로 정렬
            files.sort(key=lambda x: x["name"])
            
            logger.info(f"파일 목록 조회: {len(files)}개 파일")
            return files
            
        except Exception as e:
            logger.error(f"파일 목록 조회 실패: {str(e)}")
            raise

    def delete_file(self, filename: str) -> bool:
        """
        파일을 삭제합니다.
        
        Args:
            filename: 삭제할 파일명
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            file_path = self.upload_dir / filename
            
            if not file_path.exists():
                logger.warning(f"삭제할 파일이 존재하지 않음: {filename}")
                return False
            
            file_path.unlink()
            logger.info(f"파일 삭제 완료: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"파일 삭제 실패: {filename} - {str(e)}")
            return False

    def _get_unique_file_path(self, filename: str) -> Path:
        """
        중복되지 않는 파일 경로를 생성합니다.
        
        Args:
            filename: 원본 파일명
            
        Returns:
            Path: 고유한 파일 경로
        """
        file_path = self.upload_dir / filename
        counter = 1
        
        while file_path.exists():
            name = Path(filename).stem
            suffix = Path(filename).suffix
            new_filename = f"{name}_{counter}{suffix}"
            file_path = self.upload_dir / new_filename
            counter += 1
        
        return file_path 