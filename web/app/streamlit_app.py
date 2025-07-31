import streamlit as st
import json
from pathlib import Path
from services.packet_analyzer import PacketAnalyzer
from services.syslog_analyzer import SyslogAnalyzer
from services.rag_service import RAGService

st.set_page_config(
    page_title="Llama-PcapLog Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="chat-message-container"] div[data-testid="stMarkdownContainer"] p {
        overflow-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    st.session_state.messages = []
    st.session_state.rag_service = RAGService(use_openai=False)
    st.session_state.processed_data = None
    st.session_state.files_processed = False
    st.session_state.pcap_file = None
    st.session_state.log_file = None

if 'messages' not in st.session_state:
    init_session_state()

def clear_conversation():
    st.session_state.messages = []

def reset_files_and_data():
    init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.title("Llama-PcapLog")
    st.markdown("---")

    if st.session_state.files_processed:
        st.info("Files have been processed.")
        if st.button("Upload New Files", use_container_width=True):
            reset_files_and_data()
            st.rerun()
    else:
        st.header("File Upload")
        pcap_file = st.file_uploader(
            "PCAP File",
            type=["pcap", "pcapng"],
            help="Upload a PCAP file for network traffic analysis."
        )
        log_file = st.file_uploader(
            "Log File",
            type=None,
            help="Upload a log file for security and event analysis."
        )

        if pcap_file or log_file:
            if st.button("Process Files", use_container_width=True):
                st.session_state.pcap_file = pcap_file
                st.session_state.log_file = log_file
                st.session_state.processed_data = []
                with st.spinner("Processing files..."):
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    
                    files_to_process = []
                    if st.session_state.pcap_file:
                        files_to_process.append((st.session_state.pcap_file, PacketAnalyzer, 'pcap'))
                    if st.session_state.log_file:
                        files_to_process.append((st.session_state.log_file, SyslogAnalyzer, 'log'))

                    for uploaded_file, analyzer_class, file_type in files_to_process:
                        file_path = temp_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        try:
                            analyzer = analyzer_class()
                            if file_type == 'pcap':
                                data = analyzer.analyze_pcap(str(file_path))
                            else:
                                data = analyzer.analyze_syslog(str(file_path))
                            st.session_state.processed_data.extend(data)
                            st.success(f"Processed {file_type.upper()} file: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    if st.session_state.processed_data:
                        st.session_state.rag_service.set_processed_data(st.session_state.processed_data)
                        st.success(f"Processed {len(st.session_state.processed_data)} total entries")
                        st.session_state.files_processed = True
                        st.rerun()

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        clear_conversation()

st.title("Llama-PcapLog Chat")
st.markdown("Ask questions about your network traffic and log data.")

if st.session_state.processed_data:
    with st.container():
        st.header("Data Summary")
        pcap_count = len([item for item in st.session_state.processed_data if item.get('type') == 'network_packet'])
        log_count = len([item for item in st.session_state.processed_data if item.get('type') == 'log'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", len(st.session_state.processed_data))
        with col2:
            st.metric("Network Packets", pcap_count)
        with col3:
            st.metric("Log Entries", log_count)
        st.markdown("---")

# Chat Interface
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the uploaded files..."):
    if not st.session_state.files_processed:
        st.warning("Please upload and process files before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_service.query(prompt)
                
                parts = response.split('```')
                for i, part in enumerate(parts):
                    if not part:
                        continue
                    if i % 2 == 1:
                        lines = part.split('\n', 1)
                        lang = lines[0].strip()
                        code = lines[1] if len(lines) > 1 else ""
                        st.code(code, language=lang if lang else None)
                    else:
                        st.markdown(part)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

