from typing import List, Dict, Any
import logging
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

from web.app.core.config import settings

logger = logging.getLogger(__name__)
accelerator = Accelerator()
class RAGService:
    def __init__(self, use_openai: bool = False):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.qa_chain = None
        self.current_file = None
        self.llm = None
        self.use_openai = use_openai
        
        # Llama model attributes
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_llama_model(self):
        try:
            model_name = "choihyuunmin/LlamaTrace"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            
            # GPU 사용 가능 시 활용
            device = accelerator.device
            model = model.to(device)
            
            # Store model and tokenizer for direct use
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
        from model.processors.pcap_processor import PcapProcessor
        
        processor = PcapProcessor(file_path)
        packets = processor.process_pcap()
        dataset = processor.generate_dataset()
        
        # Create documents for RAG
        documents = []
        
        # Add file metadata document
        file_info_doc = Document(
            page_content=f"File Information: PCAP file with {len(packets)} packets analyzed. File path: {file_path}",
            metadata={"type": "file_info", "source": "pcap", "packet_count": len(packets)}
        )
        documents.append(file_info_doc)
        
        # Add packet summary
        packet_summary = self._create_packet_summary(packets)
        summary_doc = Document(
            page_content=f"Packet Analysis Summary:\n{packet_summary}",
            metadata={"type": "summary", "source": "pcap"}
        )
        documents.append(summary_doc)
        
        # Add Q&A pairs
        for item in dataset:
            doc = Document(
                page_content=f"Question: {item['instruction']}\nDetailed Answer: {item['output']}",
                metadata={"type": "qa_pair", "source": "pcap", "question_type": "analysis"}
            )
            documents.append(doc)
        
        # Add raw packet data sample
        if packets:
            sample_packets = packets[:10]
            sample_text = "Sample Packet Data:\n"
            for i, packet in enumerate(sample_packets):
                sample_text += f"Packet {i+1}: {packet.get('src_ip', 'N/A')} -> {packet.get('dst_ip', 'N/A')} (Protocol: {packet.get('protocol', 'N/A')}, Length: {packet.get('length', 'N/A')})\n"
            
            sample_doc = Document(
                page_content=sample_text,
                metadata={"type": "sample_data", "source": "pcap", "sample_size": len(sample_packets)}
            )
            documents.append(sample_doc)
        
        self._create_vector_store(documents)
        
        return {
            "packets": packets,
            "dataset": dataset,
            "summary": packet_summary
        }
    
    def _process_syslog(self, file_path: str) -> Dict[str, Any]:
        from model.processors.syslog_processor import SyslogProcessor
        
        processor = SyslogProcessor(file_path)
        logs = processor.process_logs()
        dataset = processor.generate_dataset()
        
        # Create documents for RAG
        documents = []
        
        # Add file metadata document
        file_info_doc = Document(
            page_content=f"File Information: Syslog file with {len(logs)} log entries analyzed. File path: {file_path}",
            metadata={"type": "file_info", "source": "syslog", "log_count": len(logs)}
        )
        documents.append(file_info_doc)
        
        # Add log summary
        log_summary = self._create_log_summary(logs)
        summary_doc = Document(
            page_content=f"Log Analysis Summary:\n{log_summary}",
            metadata={"type": "summary", "source": "syslog"}
        )
        documents.append(summary_doc)
        
        # Add Q&A pairs
        for item in dataset:
            doc = Document(
                page_content=f"Question: {item['instruction']}\nDetailed Answer: {item['output']}",
                metadata={"type": "qa_pair", "source": "syslog", "question_type": "analysis"}
            )
            documents.append(doc)
        
        # Add raw log data sample
        if logs:
            sample_logs = logs[:10]
            sample_text = "Sample Log Data:\n"
            for i, log in enumerate(sample_logs):
                sample_text += f"Log {i+1}: [{log.get('timestamp', 'N/A')}] {log.get('host', 'N/A')} {log.get('program', 'N/A')}: {log.get('message', 'N/A')}\n"
            
            sample_doc = Document(
                page_content=sample_text,
                metadata={"type": "sample_data", "source": "syslog", "sample_size": len(sample_logs)}
            )
            documents.append(sample_doc)
        
        self._create_vector_store(documents)
        
        return {
            "logs": logs,
            "dataset": dataset,
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
            severity = log.get('severity', 'unknown')
            severities[severity] = severities.get(severity, 0) + 1
            
            program = log.get('program', 'unknown')
            programs[program] = programs.get(program, 0) + 1
            
            host = log.get('host', 'unknown')
            hosts[host] = hosts.get(host, 0) + 1
        
        summary = f"""
        Log Analysis Summary:
        - Total logs: {total_logs}
        - Severity levels: {dict(list(severities.items())[:5])}
        - Programs: {dict(list(programs.items())[:5])}
        - Hosts: {dict(list(hosts.items())[:5])}
        """
        
        return summary
    
    def _create_vector_store(self, documents: List[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        if self.llm is None:
            if self.use_openai:
                self._load_openai_model()
                prompt_template = """
                You are a network and system analysis expert. Use the following context to answer the question.
                
                Context: {context}
                
                Question: {question}
                
                Answer based on the context provided. If the question asks for code, provide executable Python code.
                """
                
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={
                            "k": 5,
                            "score_threshold": 0.7
                        }
                    ),
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
            else:
                self._load_llama_model()
        
        logger.info(f"Vector store created with {len(split_docs)} documents")
    
    def query(self, question: str) -> str:
        if not self.vector_store:
            return "No file has been processed yet. Please upload a file first."
        
        try:
            docs = self.vector_store.similarity_search(question, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            if self.use_openai:
                result = self.qa_chain({"query": question})
                response = result["result"]
            else:
                system_prompt = "You are a network and system analysis expert specializing in packet analysis and log analysis."
                
                user_prompt = f"""Based on the following context information about a specific file that has been analyzed, please answer the user's question.

                                Context Information:
                                {context}

                                User Question: {question}

                                Please provide a detailed and accurate answer based on the context information provided above. If the question asks for code or visualization, provide executable Python code. If the context doesn't contain enough information to answer the question, say so clearly."""
                
                prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
                
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
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the generated part (remove input prompt)
                if "[/INST]" in response:
                    response = response.split("[/INST]")[-1].strip()
                else:
                    # If no clear marker, try to extract after the last instruction
                    response = response[len(prompt):].strip()
                
                # Clean up any remaining special tokens or formatting
                response = response.replace("<s>", "").replace("</s>", "").strip()
            
            # Log retrieved context
            logger.info(f"Question: {question}")
            logger.info(f"Retrieved {len(docs)} source documents")
            for i, doc in enumerate(docs[:2]):
                logger.info(f"Source doc {i+1}: {doc.page_content[:200]}...")
            
            print(f"Response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return f"Error processing query: {str(e)}"
    
    def get_context(self) -> Dict[str, Any]:
        if not self.current_file:
            return {"message": "No file processed"}
        
        return {
            "current_file": self.current_file,
            "file_type": "pcap" if self.current_file.endswith(".pcap") or self.current_file.endswith(".pcapng") else "log",
            "vector_store_ready": self.vector_store is not None
        } 