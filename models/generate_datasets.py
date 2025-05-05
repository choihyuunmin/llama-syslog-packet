import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from processors.pcap_processor import PcapProcessor
from processors.syslog_processor import SyslogProcessor
from collections import defaultdict


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """학습 데이터셋 생성기
    
    이 클래스는 PCAP 및 Syslog 파일을 분석하여
    네트워크 보안 관련 질문-답변 데이터셋을 생성합니다.
    
    Attributes:
        output_dir (Path): 출력 디렉토리
    """
    
    def __init__(self, output_dir: str = "processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_pcap_dataset(self, pcap_file: str) -> Optional[Path]:
        try:
            logger.info(f"PCAP 파일 처리 시작: {pcap_file}")
            
            # PCAP 프로세서 초기화 및 처리
            processor = PcapProcessor(pcap_file)
            processor.process_pcap()
            
            # 데이터셋 생성
            dataset = processor.generate_dataset()
            if not dataset:
                logger.warning("생성된 PCAP 데이터셋이 비어있습니다.")
                return None
            
            # 파일 저장
            output_path = self._get_output_path("pcap_dataset")
            self._save_dataset(dataset, output_path)
            
            logger.info(f"PCAP 데이터셋 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PCAP 데이터셋 생성 중 오류 발생: {e}")
            raise
    
    def generate_syslog_dataset(self, syslog_file: str) -> Optional[Path]:
        try:
            logger.info(f"Syslog 파일 처리 시작: {syslog_file}")
            
            # Syslog 프로세서 초기화 및 처리
            processor = SyslogProcessor(syslog_file)
            processor.process_logs()
            
            # 데이터셋 생성
            dataset = processor.generate_dataset()
            if not dataset:
                logger.warning("생성된 Syslog 데이터셋이 비어있습니다.")
                return None
            
            # 파일 저장
            output_path = self._get_output_path("syslog_dataset")
            self._save_dataset(dataset, output_path)
            
            logger.info(f"Syslog 데이터셋 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Syslog 데이터셋 생성 중 오류 발생: {e}")
            raise
    
    def _get_output_path(self, base_name: str) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}.json"
        return self.output_dir / filename
    
    def _save_dataset(self, dataset: List[Dict[str, str]], output_path: Path) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"데이터셋 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"데이터셋 저장 중 오류 발생: {e}")
            raise

def validate_input_file(file_path: str) -> None:
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="packet/syslog 데이터셋 생성기")
    parser.add_argument("--pcap", type=str, help="PCAP 파일 경로")
    parser.add_argument("--syslog", type=str, help="Syslog 파일 경로")
    parser.add_argument("--output-dir", type=str, default="processed", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    try:
        # 입력 파일 검증
        if args.pcap:
            validate_input_file(args.pcap)
        if args.syslog:
            validate_input_file(args.syslog)
        
        # 데이터셋 생성기 초기화
        generator = DatasetGenerator(args.output_dir)
        
        # PCAP 데이터셋 생성
        if args.pcap:
            generator.generate_pcap_dataset(args.pcap)
        
        # Syslog 데이터셋 생성
        if args.syslog:
            generator.generate_syslog_dataset(args.syslog)
        
        logger.info("모든 데이터셋 생성이 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"데이터셋 생성 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 