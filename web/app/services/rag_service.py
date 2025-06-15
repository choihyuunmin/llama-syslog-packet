import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store_path = os.getenv('VECTOR_STORE_PATH', 'vector_store')
        self.initialize_vector_store()

    def initialize_vector_store(self):
        try:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings
                )
                logger.info("Loaded existing vector store")
            else:
                self.vector_store = FAISS.from_texts(
                    ["Initial vector store"],
                    self.embeddings
                )
                logger.info("Created new vector store")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def process_packet_data(self, packet_data: List[Dict[str, Any]]) -> None:
        try:
            documents = []
            for packet in packet_data:
                packet_text = f"""
                Timestamp: {packet.get('timestamp', 'N/A')}
                Source IP: {packet.get('src_ip', 'N/A')}
                Destination IP: {packet.get('dst_ip', 'N/A')}
                Protocol: {packet.get('protocol', 'N/A')}
                Length: {packet.get('length', 'N/A')}
                Additional Info: {json.dumps(packet.get('additional_info', {}), ensure_ascii=False)}
                """
                documents.append(Document(
                    page_content=packet_text,
                    metadata={
                        "timestamp": packet.get('timestamp', ''),
                        "protocol": packet.get('protocol', ''),
                        "src_ip": packet.get('src_ip', ''),
                        "dst_ip": packet.get('dst_ip', '')
                    }
                ))

            split_docs = self.text_splitter.split_documents(documents)
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    split_docs,
                    self.embeddings
                )
            else:
                self.vector_store.add_documents(split_docs)
            
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"{len(split_docs)} documents added to vector store")
            
        except Exception as e:
            logger.error(f"Error processing packet data: {e}")
            raise

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        try:
            if self.vector_store is None:
                return "No relevant context"
            
            docs = self.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
            
        except Exception as e:
            logger.error(f"Error searching context: {e}")
            return "Error searching context" 