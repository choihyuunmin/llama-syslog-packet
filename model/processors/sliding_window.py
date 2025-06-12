"""슬라이딩 윈도우 처리를 위한 프로세서 모듈.

이 모듈은 pcap과 log 파일을 시간 기반으로 슬라이딩 윈도우 처리하여
새로운 파일들로 분할하는 기능을 제공합니다.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import shutil
from scapy.all import rdpcap, wrpcap
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sliding_window.log')
    ]
)
logger = logging.getLogger(__name__)

class SlidingWindowProcessor:
    """슬라이딩 윈도우 처리를 위한 클래스."""

    def __init__(
        self,
        window_size: int = 300,  # 윈도우 크기 (초) - 5분
        step_size: int = 60,     # 스텝 사이즈 (초) - 1분
        output_dir: str = "model/datasets/output"
    ):
        """SlidingWindowProcessor 초기화.

        Args:
            window_size: 윈도우 크기 (초)
            step_size: 스텝 사이즈 (초)
            output_dir: 출력 디렉토리 경로
        """
        self.window_size = window_size
        self.step_size = step_size
        self.output_dir = Path(output_dir)
        self._setup_directories()

    def _setup_directories(self) -> None:
        """출력 디렉토리 구조를 설정합니다."""
        # 메인 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # pcap과 log를 위한 하위 디렉토리 생성
        (self.output_dir / "packet").mkdir(exist_ok=True)
        (self.output_dir / "syslog").mkdir(exist_ok=True)

    def process_file(self, file_path: Path) -> None:
        """파일을 슬라이딩 윈도우로 처리합니다.

        Args:
            file_path: 처리할 파일 경로
        """
        try:
            logger.info(f"파일 처리 시작: {file_path}")
            
            # 파일 확장자에 따라 처리 방식 결정
            if file_path.suffix.lower() in ['.pcap', '.pcapng']:
                self._process_pcap_file(file_path)
            else:
                self._process_log_file(file_path)
                
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {file_path} - {str(e)}", exc_info=True)

    def _process_pcap_file(self, file_path: Path) -> None:
        """pcap 파일을 슬라이딩 윈도우로 처리합니다.

        Args:
            file_path: 처리할 pcap 파일 경로
        """
        # pcap 파일 읽기
        packets = rdpcap(str(file_path))
        if not packets:
            logger.warning(f"패킷이 없는 파일: {file_path}")
            return

        logger.info(f"총 {len(packets)}개의 패킷을 읽었습니다.")

        # 첫 번째와 마지막 패킷의 시간 가져오기
        start_time = datetime.fromtimestamp(float(packets[0].time))
        end_time = datetime.fromtimestamp(float(packets[-1].time))
        logger.info(f"처리 시간 범위: {start_time} ~ {end_time}")

        # 윈도우 시작 시간들 생성
        window_starts = self._generate_window_starts(start_time, end_time)
        logger.info(f"생성된 윈도우 수: {len(window_starts)}")

        # 각 윈도우에 대해 처리
        for i, window_start in enumerate(window_starts):
            window_end = window_start + timedelta(seconds=self.window_size)
            logger.debug(f"윈도우 {i} 처리 중: {window_start} ~ {window_end}")
            
            # 현재 윈도우에 속하는 패킷들 선택
            window_packets = [
                p for p in packets
                if window_start <= datetime.fromtimestamp(float(p.time)) < window_end
            ]

            if window_packets:
                # 출력 파일명 생성
                output_filename = f"{file_path.stem}_window_{i:03d}.pcap"
                output_path = self.output_dir / "packet" / output_filename
                
                # 패킷 저장
                wrpcap(str(output_path), window_packets)
                logger.info(f"윈도우 {i} 저장 완료: {output_path} (패킷 수: {len(window_packets)})")
            else:
                logger.debug(f"윈도우 {i}에 패킷이 없습니다.")

        logger.info(f"pcap 파일 처리 완료: {file_path}")

    def _process_log_file(self, file_path: Path) -> None:
        """log 파일을 슬라이딩 윈도우로 처리합니다.

        Args:
            file_path: 처리할 log 파일 경로
        """
        # log 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            logger.warning(f"로그가 없는 파일: {file_path}")
            return

        logger.info(f"총 {len(lines)}개의 로그 라인을 읽었습니다.")

        # 첫 번째와 마지막 로그의 시간 가져오기
        start_time = self._parse_log_time(lines[0])
        end_time = self._parse_log_time(lines[-1])

        if not start_time or not end_time:
            logger.warning(f"시간 파싱 실패: {file_path}")
            return

        logger.info(f"처리 시간 범위: {start_time} ~ {end_time}")

        # 윈도우 시작 시간들 생성
        window_starts = self._generate_window_starts(start_time, end_time)
        logger.info(f"생성된 윈도우 수: {len(window_starts)}")

        # 각 윈도우에 대해 처리
        for i, window_start in enumerate(window_starts):
            window_end = window_start + timedelta(seconds=self.window_size)
            logger.debug(f"윈도우 {i} 처리 중: {window_start} ~ {window_end}")
            
            # 현재 윈도우에 속하는 로그들 선택
            window_lines = [
                line for line in lines
                if window_start <= self._parse_log_time(line) < window_end
            ]

            if window_lines:
                # 출력 파일명 생성
                output_filename = f"{file_path.stem}_window_{i:03d}.log"
                output_path = self.output_dir / "syslog" / output_filename
                
                # 로그 저장
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.writelines(window_lines)
                logger.info(f"윈도우 {i} 저장 완료: {output_path} (로그 수: {len(window_lines)})")
            else:
                logger.debug(f"윈도우 {i}에 로그가 없습니다.")

        logger.info(f"log 파일 처리 완료: {file_path}")

    def _generate_window_starts(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[datetime]:
        """윈도우 시작 시간들을 생성합니다.

        Args:
            start_time: 시작 시간
            end_time: 종료 시간

        Returns:
            윈도우 시작 시간들의 리스트
        """
        window_starts = []
        current = start_time
        while current < end_time:
            window_starts.append(current)
            current += timedelta(seconds=self.step_size)
        return window_starts

    def _parse_log_time(self, log_line: str) -> Optional[datetime]:
        """로그 라인에서 시간을 파싱합니다.

        Args:
            log_line: 로그 라인

        Returns:
            파싱된 시간 또는 None
        """
        try:
            # 일반적인 syslog 형식 (예: Jan 1 00:00:00)
            time_str = log_line.split()[0:3]
            return datetime.strptime(' '.join(time_str), '%b %d %H:%M:%S')
        except:
            try:
                # ISO 형식 (예: 2024-01-01T00:00:00)
                time_str = log_line.split()[0]
                return datetime.fromisoformat(time_str)
            except:
                return None

    def process_directory(self, input_dir: str) -> None:
        """디렉토리 내의 모든 파일을 처리합니다.

        Args:
            input_dir: 입력 디렉토리 경로
        """
        input_path = Path(input_dir)
        
        # packet 디렉토리 처리
        packet_dir = input_path / "packet"
        if packet_dir.exists():
            logger.info(f"패킷 디렉토리 처리 시작: {packet_dir}")
            for file_path in packet_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.pcap', '.pcapng']:
                    logger.info(f"패킷 파일 처리 중: {file_path}")
                    self._process_pcap_file(file_path)
        else:
            logger.warning(f"패킷 디렉토리가 존재하지 않습니다: {packet_dir}")

        # syslog 디렉토리 처리
        syslog_dir = input_path / "syslog"
        if syslog_dir.exists():
            logger.info(f"시스로그 디렉토리 처리 시작: {syslog_dir}")
            for file_path in syslog_dir.iterdir():
                if file_path.is_file():
                    logger.info(f"로그 파일 처리 중: {file_path}")
                    self._process_log_file(file_path)
        else:
            logger.warning(f"시스로그 디렉토리가 존재하지 않습니다: {syslog_dir}")

def main():
    """메인 실행 함수."""
    logger.info("슬라이딩 윈도우 처리 시작")
    
    # 입력 디렉토리 설정
    input_dir = "../datasets"
    
    logger.info(f"입력 디렉토리: {input_dir}")
    
    # 프로세서 초기화 (5분 윈도우, 1분 스텝)
    processor = SlidingWindowProcessor(
        window_size=30,  # 5분
        step_size=60,     # 1분
        output_dir="../output"
    )
    
    logger.info(f"윈도우 크기: {processor.window_size}초")
    logger.info(f"스텝 크기: {processor.step_size}초")
    logger.info(f"출력 디렉토리: {processor.output_dir}")
    
    # 디렉토리 처리
    processor.process_directory(input_dir)
    
    logger.info("슬라이딩 윈도우 처리 완료")

if __name__ == "__main__":
    main() 