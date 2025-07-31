from typing import List, Dict, Any
import logging
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_openai import ChatOpenAI

from core.config import settings
from services.packet_analyzer import PacketAnalyzer
from services.syslog_analyzer import SyslogAnalyzer

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, use_openai: bool = False):
        self.processed_data = None
        self.current_file = None
        self.llm = None
        self.use_openai = use_openai
        
        # Llama model attributes
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_llama_model(self):
        try:
            model_name = "CNU-CHOI/Llama-PcapLog"
            # model_name = "meta-llama/Llama-3.1-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad_token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            
            logger.info(f"Loaded Llama model: {model_name} on {device}")
            
        except Exception as e:
            logger.error(f"Error loading Llama model: {e}")
            raise
    
    def _load_openai_model(self):
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required for GPT-3.5-turbo")
            
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=settings.openai_api_key
            )
            logger.info("Loaded OpenAI GPT-3.5-turbo model")
            
        except Exception as e:
            logger.error(f"Error loading OpenAI model: {e}")
            raise
        
    def process_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        try:
            self.current_file = file_path
            
            if file_type == 'pcap':
                return self._process_pcap(file_path)
            elif file_type == 'log':
                return self._process_syslog(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise
    
    def _process_pcap(self, file_path: str) -> Dict[str, Any]:
        analyzer = PacketAnalyzer()
        packets = analyzer.analyze_pcap(file_path)
        
        packet_summary = self._create_packet_summary(packets)
        
        self.processed_data = {
            "file_type": "pcap",
            "file_path": file_path,
            "packet_count": len(packets),
            "summary": packet_summary,
            "packets": packets
        }
        
        # Load model if not already loaded
        if self.model is None and self.llm is None:
            if self.use_openai:
                self._load_openai_model()
            else:
                self._load_llama_model()
        
        return {
            "packets": packets,
            "summary": packet_summary
        }
    
    def _process_syslog(self, file_path: str) -> Dict[str, Any]:
        analyzer = SyslogAnalyzer()
        logs = analyzer.analyze_syslog(file_path)
        
        log_summary = self._create_log_summary(logs)
        
        self.processed_data = {
            "file_type": "syslog",
            "file_path": file_path,
            "log_count": len(logs),
            "summary": log_summary,
            "logs": logs
        }
        
        # Load model if not already loaded
        if self.model is None and self.llm is None:
            if self.use_openai:
                self._load_openai_model()
            else:
                self._load_llama_model()
        
        return {
            "logs": logs,
            "summary": log_summary
        }
    
    def _create_packet_summary(self, packets: List[Dict]) -> str:
        if not packets:
            return "No packets found in the file."
        
        total_packets = len(packets)
        protocols = {}
        src_ips = {}
        dst_ips = {}
        
        for packet in packets:
            protocol = packet.get('protocol', 'Unknown')
            protocols[protocol] = protocols.get(protocol, 0) + 1
            
            src_ip = packet.get('src_ip')
            if src_ip:
                src_ips[src_ip] = src_ips.get(src_ip, 0) + 1
                
            dst_ip = packet.get('dst_ip')
            if dst_ip:
                dst_ips[dst_ip] = dst_ips.get(dst_ip, 0) + 1
        
        summary = f"""
        Packet Analysis Summary:
        - Total packets: {total_packets}
        - Protocols: {dict(list(protocols.items())[:5])}
        - Top source IPs: {dict(list(src_ips.items())[:5])}
        - Top destination IPs: {dict(list(dst_ips.items())[:5])}
        """
        
        return summary
    
    def _create_log_summary(self, logs: List[Dict]) -> str:
        if not logs:
            return "No logs found in the file."
        
        total_logs = len(logs)
        severities = {}
        programs = {}
        hosts = {}
        
        for log in logs:
            severity = log.get('log_level', 'unknown')
            severities[severity] = severities.get(severity, 0) + 1
            
            program = log.get('component', 'unknown')
            programs[program] = programs.get(program, 0) + 1
            
            host = log.get('hostname', 'unknown')
            hosts[host] = hosts.get(host, 0) + 1
        
        summary = f"""
        Log Analysis Summary:
        - Total logs: {total_logs}
        - Severity levels: {dict(list(severities.items())[:5])}
        - Programs: {dict(list(programs.items())[:5])}
        - Hosts: {dict(list(hosts.items())[:5])}
        """
        
        return summary
    
    def set_processed_data(self, processed_data: List[Dict[str, Any]]) -> None:
        """Set processed data from external source (e.g., Streamlit)"""
        if not processed_data:
            self.processed_data = None
            return
            
        # Determine if data contains pcap or syslog entries
        pcap_data = [item for item in processed_data if item.get('type') == 'network_packet']
        syslog_data = [item for item in processed_data if item.get('type') == 'log']
        
        if pcap_data and syslog_data:
            # Mixed data
            self.processed_data = {
                "file_type": "mixed",
                "packet_count": len(pcap_data),
                "log_count": len(syslog_data),
                "summary": self._create_mixed_summary(pcap_data, syslog_data),
                "packets": pcap_data,
                "logs": syslog_data
            }
        elif pcap_data:
            # Only packet data
            self.processed_data = {
                "file_type": "pcap",
                "packet_count": len(pcap_data),
                "summary": self._create_packet_summary(pcap_data),
                "packets": pcap_data
            }
        elif syslog_data:
            # Only log data
            self.processed_data = {
                "file_type": "syslog",
                "log_count": len(syslog_data),
                "summary": self._create_log_summary(syslog_data),
                "logs": syslog_data
            }
        else:
            self.processed_data = None
            
        # Load model if not already loaded
        if self.processed_data and self.model is None and self.llm is None:
            if self.use_openai:
                self._load_openai_model()
            else:
                self._load_llama_model()
    
    def _create_mixed_summary(self, packets: List[Dict], logs: List[Dict]) -> str:
        packet_summary = self._create_packet_summary(packets)
        log_summary = self._create_log_summary(logs)
        
        return f"""
        Mixed Data Analysis Summary:
        
        {packet_summary}
        
        {log_summary}
        """
    
    def query(self, question: str) -> str:
        if not self.processed_data:
            return "No file has been processed yet. Please upload a file first."
        
        try:
            # Create context from processed data
            context = self._create_context()
            
            if self.use_openai:
                prompt = f"""You are a network and system analysis expert. Use the following context to answer the question.

Context: {context}

Question: {question}

Answer based on the context provided. If the question asks for code, provide executable Python code."""
                
                response = self.llm.invoke(prompt)
                return response.content
            else:
                prompt = f"""<s>[INST] You are a network and system analysis expert specializing in packet analysis and log analysis. 
You have access to the following context information about a specific file that has been analyzed.

Context Information:
{context}

User Question: {question}

Please provide a detailed and accurate answer based on the context information provided above. 
If the question asks for code or visualization, provide executable Python code.
If the context doesn't contain enough information to answer the question, say so clearly.

Answer: [/INST]"""
                
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                logger.info(f"Outputs: {outputs}")
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Response: {response}")
                
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                
                return response
            
        except Exception as e:
            logger.error(f"Error querying system: {e}")
            return f"Error processing query: {str(e)}"
    
    def _create_context(self) -> str:
        if not self.processed_data:
            return ""
        
        context = f"""File Type: {self.processed_data['file_type']}

{self.processed_data['summary']}

"""
        
        if self.processed_data['file_type'] == 'pcap':
            context += f"Total Packets: {self.processed_data['packet_count']}\n"
            if self.processed_data['packets']:
                context += "Sample Packet Data:\n"
                for i, packet in enumerate(self.processed_data['packets'][:3]):
                    context += f"Packet {i+1}: {json.dumps(packet, indent=2)}\n"
        
        elif self.processed_data['file_type'] == 'syslog':
            context += f"Total Logs: {self.processed_data['log_count']}\n"
            if self.processed_data['logs']:
                context += "Sample Log Data:\n"
                for i, log in enumerate(self.processed_data['logs'][:3]):
                    context += f"Log {i+1}: {json.dumps(log, indent=2)}\n"
        
        elif self.processed_data['file_type'] == 'mixed':
            context += f"Total Packets: {self.processed_data['packet_count']}\n"
            context += f"Total Logs: {self.processed_data['log_count']}\n"
            
            if self.processed_data.get('packets'):
                context += "Sample Packet Data:\n"
                for i, packet in enumerate(self.processed_data['packets'][:2]):
                    context += f"Packet {i+1}: {json.dumps(packet, indent=2)}\n"
                    
            if self.processed_data.get('logs'):
                context += "Sample Log Data:\n"
                for i, log in enumerate(self.processed_data['logs'][:2]):
                    context += f"Log {i+1}: {json.dumps(log, indent=2)}\n"
        
        return context
    
    def get_context(self) -> Dict[str, Any]:
        if not self.current_file:
            return {"message": "No file processed"}
        
        return {
            "current_file": self.current_file,
            "file_type": "pcap" if self.current_file.endswith(".pcap") or self.current_file.endswith(".pcapng") else "log",
            "data_ready": self.processed_data is not None
        }