import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from core.config import settings

# logger 설정
logger = logging.getLogger(__name__)

class FileAnalysisError(Exception):
    pass

class InvalidFileTypeError(Exception):
    pass

def get_file_type(file_path: Path) -> str:
    extension = file_path.suffix.lower()
    
    if extension in ['.pcap', '.pcapng']:
        return 'pcap'
    elif extension in ['.log', '.txt']:
        return 'log'
    else:
        return 'unknown'

def validate_file_upload(filename: str, file_size: int) -> None:
    file_path = Path(filename)
    file_type = get_file_type(file_path)
    
    if file_type == 'unknown':
        raise InvalidFileTypeError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported extensions: {', '.join(settings.allowed_extensions)}"
        )
    
    if file_size > settings.max_file_size:
        max_size_mb = settings.max_file_size / (1024 * 1024)
        raise ValueError(f"File size exceeds limit. Maximum size: {max_size_mb}MB") 