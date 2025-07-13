"""Sliding window processor module for pcap and log files."""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import shutil
from scapy.all import rdpcap, wrpcap
import pandas as pd

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
    def __init__(
        self,
        window_size: int = 300,
        step_size: int = 60,
        output_dir: str = "model/datasets/output"
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.output_dir = Path(output_dir)
        self._setup_directories()

    def _setup_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "packet").mkdir(exist_ok=True)
        (self.output_dir / "syslog").mkdir(exist_ok=True)

    def process_file(self, file_path: Path) -> None:
        try:
            logger.info(f"Starting file processing: {file_path}")
            
            if file_path.suffix.lower() in ['.pcap', '.pcapng']:
                self._process_pcap_file(file_path)
            else:
                self._process_log_file(file_path)
                
        except Exception as e:
            logger.error(f"Error occurred during file processing: {file_path} - {str(e)}", exc_info=True)

    def _process_pcap_file(self, file_path: Path) -> None:
        packets = rdpcap(str(file_path))
        if not packets:
            logger.warning(f"No packets found in file: {file_path}")
            return

        logger.info(f"Read {len(packets)} packets total.")

        start_time = datetime.fromtimestamp(float(packets[0].time))
        end_time = datetime.fromtimestamp(float(packets[-1].time))
        logger.info(f"Processing time range: {start_time} ~ {end_time}")

        window_starts = self._generate_window_starts(start_time, end_time)
        logger.info(f"Generated {len(window_starts)} windows")

        for i, window_start in enumerate(window_starts):
            window_end = window_start + timedelta(seconds=self.window_size)
            logger.debug(f"Processing window {i}: {window_start} ~ {window_end}")
            
            window_packets = [
                p for p in packets
                if window_start <= datetime.fromtimestamp(float(p.time)) < window_end
            ]

            if window_packets:
                output_filename = f"{file_path.stem}_window_{i:03d}.pcap"
                output_path = self.output_dir / "packet" / output_filename
                
                wrpcap(str(output_path), window_packets)
                logger.info(f"Window {i} saved: {output_path} (packets: {len(window_packets)})")
            else:
                logger.debug(f"No packets in window {i}")

        logger.info(f"Pcap file processing completed: {file_path}")

    def _process_log_file(self, file_path: Path) -> None:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            logger.warning(f"No logs found in file: {file_path}")
            return

        logger.info(f"Read {len(lines)} log lines total.")

        start_time = self._parse_log_time(lines[0])
        end_time = self._parse_log_time(lines[-1])

        if not start_time or not end_time:
            logger.warning(f"Time parsing failed: {file_path}")
            return

        logger.info(f"Processing time range: {start_time} ~ {end_time}")

        window_starts = self._generate_window_starts(start_time, end_time)
        logger.info(f"Generated {len(window_starts)} windows")

        for i, window_start in enumerate(window_starts):
            window_end = window_start + timedelta(seconds=self.window_size)
            logger.debug(f"Processing window {i}: {window_start} ~ {window_end}")
            
            window_lines = [
                line for line in lines
                if window_start <= self._parse_log_time(line) < window_end
            ]

            if window_lines:
                output_filename = f"{file_path.stem}_window_{i:03d}.log"
                output_path = self.output_dir / "syslog" / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.writelines(window_lines)
                logger.info(f"Window {i} saved: {output_path} (logs: {len(window_lines)})")
            else:
                logger.debug(f"No logs in window {i}")

        logger.info(f"Log file processing completed: {file_path}")

    def _generate_window_starts(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[datetime]:
        window_starts = []
        current = start_time
        while current < end_time:
            window_starts.append(current)
            current += timedelta(seconds=self.step_size)
        return window_starts

    def _parse_log_time(self, log_line: str) -> Optional[datetime]:
        try:
            # Parse syslog format (e.g., Jan 1 00:00:00)
            time_str = log_line.split()[0:3]
            return datetime.strptime(' '.join(time_str), '%b %d %H:%M:%S')
        except:
            try:
                # Parse ISO format (e.g., 2024-01-01T00:00:00)
                time_str = log_line.split()[0]
                return datetime.fromisoformat(time_str)
            except:
                return None

    def process_directory(self, input_dir: str) -> None:
        input_path = Path(input_dir)
        
        packet_dir = input_path / "packet"
        if packet_dir.exists():
            logger.info(f"Starting packet directory processing: {packet_dir}")
            for file_path in packet_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.pcap', '.pcapng']:
                    logger.info(f"Processing packet file: {file_path}")
                    self._process_pcap_file(file_path)
        else:
            logger.warning(f"Packet directory does not exist: {packet_dir}")

        syslog_dir = input_path / "syslog"
        if syslog_dir.exists():
            logger.info(f"Starting syslog directory processing: {syslog_dir}")
            for file_path in syslog_dir.iterdir():
                if file_path.is_file():
                    logger.info(f"Processing log file: {file_path}")
                    self._process_log_file(file_path)
        else:
            logger.warning(f"Syslog directory does not exist: {syslog_dir}")

def main():
    logger.info("Starting sliding window processing")
    
    input_dir = "../datasets"
    
    logger.info(f"Input directory: {input_dir}")
    
    processor = SlidingWindowProcessor(
        window_size=30,
        step_size=60,
        output_dir="../output"
    )
    
    logger.info(f"Window size: {processor.window_size} seconds")
    logger.info(f"Step size: {processor.step_size} seconds")
    logger.info(f"Output directory: {processor.output_dir}")
    
    processor.process_directory(input_dir)
    
    logger.info("Sliding window processing completed")

if __name__ == "__main__":
    main() 