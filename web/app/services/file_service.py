from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from web.app.core.utils import analyze_pcap, analyze_log, get_file_type, validate_file_upload
from web.app.core.config import settings

logger = logging.getLogger(__name__)


class FileService:
    def __init__(self, upload_dir: Optional[Path] = None):    
        self.upload_dir = upload_dir or settings.upload_dir
        self.upload_dir.mkdir(exist_ok=True)
        logger.info(f"FileService initialized. Upload directory: {self.upload_dir}")

    async def save_file(self, file_content: bytes, filename: str) -> Path:
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
        file_path = self.upload_dir / filename
        counter = 1
        
        while file_path.exists():
            name = Path(filename).stem
            suffix = Path(filename).suffix
            new_filename = f"{name}_{counter}{suffix}"
            file_path = self.upload_dir / new_filename
            counter += 1
        
        return file_path 