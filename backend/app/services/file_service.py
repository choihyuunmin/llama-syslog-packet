from pathlib import Path
from typing import Dict, Any
from ..core.utils import analyze_pcap, analyze_log, get_file_type

class FileService:
    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(exist_ok=True)

    async def save_file(self, file_content: bytes, filename: str) -> Path:
        file_path = self.upload_dir / filename
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        return file_path

    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        file_type = get_file_type(file_path)
        analysis_result = {}

        if file_type == 'pcap':
            analysis_result = analyze_pcap(file_path)
        elif file_type == 'log':
            analysis_result = analyze_log(file_path)

        return {
            "filename": file_path.name,
            "file_type": file_type,
            "analysis": analysis_result
        }

    def list_files(self) -> list:
        files = []
        for f in self.upload_dir.iterdir():
            if f.is_file():
                file_type = get_file_type(f)
                files.append({
                    "name": f.name,
                    "type": file_type,
                    "size": f.stat().st_size
                })
        return files 