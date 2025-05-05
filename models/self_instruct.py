import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from processors.pcap_processor import PcapProcessor
from processors.syslog_processor import SyslogProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SelfInstructGenerator:
    """Self-Instruct 학습 데이터셋 생성기
    
    이 클래스는 PCAP와 Syslog 파일을 분석하여 LLM을 활용한
    네트워크 보안 관련 질문-답변 데이터셋을 생성합니다.
    
    Attributes:
        output_dir (Path): 출력 디렉토리
        model (str): 사용할 LLM 모델
        api_key (str): OpenAI API 키
        datasets_dir (Path): 데이터셋 디렉토리
    """
    
    def __init__(self, output_dir: str = "processed", model: str = "gpt-4", api_key: str = None, datasets_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key
        self.datasets_dir = Path(datasets_dir)
        if not self.datasets_dir.exists():
            raise ValueError(f"Datasets directory not found: {datasets_dir}")

    def _get_dataset_files(self) -> tuple[List[Path], List[Path]]:
        """데이터셋 디렉토리에서 PCAP와 Syslog 파일 목록을 가져옵니다.
        
        Returns:
            tuple[List[Path], List[Path]]: PCAP 파일 목록과 Syslog 파일 목록
        """
        pcap_files = list(self.datasets_dir.glob("**/*.pcap"))
        syslog_files = list(self.datasets_dir.glob("**/*.log"))
        return pcap_files, syslog_files

    def _process_pcap_file(self, pcap_file: Path) -> Optional[Dict[str, str]]:
        """PCAP 파일을 처리하여 예제를 생성합니다.
        
        Args:
            pcap_file (Path): PCAP 파일 경로
            
        Returns:
            Optional[Dict[str, str]]: 생성된 예제 또는 None
        """
        try:
            processor = PcapProcessor(str(pcap_file))
            processor.process_pcap()
            dataset = processor.generate_dataset()
            
            if not dataset:
                return None
                
            # 첫 번째 예제를 사용
            example = dataset[0]
            return {
                "instruction": f"Analyze this packet capture from {pcap_file.name} for security vulnerabilities.",
                "input": example["input"],
                "output": example["output"]
            }
        except Exception as e:
            logger.error(f"Error processing PCAP file {pcap_file}: {e}")
            return None

    def _process_syslog_file(self, syslog_file: Path) -> Optional[Dict[str, str]]:
        """Syslog 파일을 처리하여 예제를 생성합니다.
        
        Args:
            syslog_file (Path): Syslog 파일 경로
            
        Returns:
            Optional[Dict[str, str]]: 생성된 예제 또는 None
        """
        try:
            processor = SyslogProcessor(str(syslog_file))
            processor.process_logs()
            dataset = processor.generate_dataset()
            
            if not dataset:
                return None
                
            # 첫 번째 예제를 사용
            example = dataset[0]
            return {
                "instruction": f"Analyze this syslog from {syslog_file.name} for security vulnerabilities.",
                "input": example["input"],
                "output": example["output"]
            }
        except Exception as e:
            logger.error(f"Error processing Syslog file {syslog_file}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_with_llm(self, prompt: str) -> str:
        """LLM을 사용하여 응답을 생성합니다.
        
        Args:
            prompt (str): LLM에 전달할 프롬프트
            
        Returns:
            str: LLM의 응답
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a network security expert. Generate detailed security analysis and recommendations based on the given logs or packet data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            raise

    def _generate_new_example(self, seed_examples: List[Dict[str, str]]) -> Dict[str, str]:
        """LLM을 사용하여 새로운 예제를 생성합니다.
        
        Args:
            seed_examples (List[Dict[str, str]]): 시드 예제 목록
            
        Returns:
            Dict[str, str]: 생성된 새로운 예제
        """
        prompt = f"""Based on these examples:
{json.dumps(seed_examples, indent=2)}

Generate a new network security analysis example in the same format. Include:
1. A clear instruction for analyzing logs or packet data
2. Realistic input data (syslog entries or packet captures)
3. Detailed security analysis and recommendations

The example should cover different types of security scenarios like:
- Network scanning
- Authentication attacks
- Protocol anomalies
- Service exploitation
- Data exfiltration
- Access control violations

Format the response as a JSON object with 'instruction', 'input', and 'output' fields."""

        response = self._generate_with_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response}")
            raise

    def generate_dataset(self, num_examples: int = 10) -> Path:
        """Self-Instruct 학습 데이터셋을 생성합니다.
        
        Args:
            num_examples (int): 생성할 예제 수
            
        Returns:
            Path: 생성된 데이터셋 파일 경로
        """
        try:
            # 데이터셋 파일 목록 가져오기
            pcap_files, syslog_files = self._get_dataset_files()
            logger.info(f"Found {len(pcap_files)} PCAP files and {len(syslog_files)} Syslog files")
            
            # PCAP 파일 처리
            dataset = []
            for pcap_file in pcap_files:
                example = self._process_pcap_file(pcap_file)
                if example:
                    dataset.append(example)
                    logger.info(f"Processed PCAP file: {pcap_file.name}")
            
            # Syslog 파일 처리
            for syslog_file in syslog_files:
                example = self._process_syslog_file(syslog_file)
                if example:
                    dataset.append(example)
                    logger.info(f"Processed Syslog file: {syslog_file.name}")
            
            # LLM을 사용하여 추가 예제 생성
            remaining_examples = num_examples - len(dataset)
            if remaining_examples > 0:
                for i in range(remaining_examples):
                    new_example = self._generate_new_example(dataset)
                    dataset.append(new_example)
                    logger.info(f"Generated example {i+1}/{remaining_examples}")
            
            # JSONL 형식으로 저장
            output_path = self._get_output_path("self_instruct_dataset")
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Self-Instruct dataset generation completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during Self-Instruct dataset generation: {e}")
            raise
    
    def _get_output_path(self, base_name: str) -> Path:
        """출력 파일 경로를 생성합니다.
        
        Args:
            base_name (str): 기본 파일명
            
        Returns:
            Path: 타임스탬프가 포함된 출력 파일 경로
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}.jsonl"
        return self.output_dir / filename

def main():
    """Self-Instruct 데이터셋 생성 스크립트의 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Instruct 데이터셋 생성기")
    parser.add_argument("--output-dir", type=str, default="processed", help="출력 디렉토리")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="데이터셋 디렉토리")
    parser.add_argument("--model", type=str, default="gpt-4", help="사용할 LLM 모델")
    parser.add_argument("--api-key", type=str, help="OpenAI API 키")
    parser.add_argument("--num-examples", type=int, default=10, help="생성할 예제 수")
    
    args = parser.parse_args()
    
    try:
        generator = SelfInstructGenerator(
            output_dir=args.output_dir,
            model=args.model,
            api_key=args.api_key,
            datasets_dir=args.datasets_dir
        )
        generator.generate_dataset(num_examples=args.num_examples)
        logger.info("Self-Instruct dataset generation completed")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        raise

if __name__ == "__main__":
    main() 