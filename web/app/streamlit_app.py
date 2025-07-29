import streamlit as st
import json
from pathlib import Path
from services.packet_analyzer import PacketAnalyzer
from services.syslog_analyzer import SyslogAnalyzer
from services.rag_service import RAGService

st.set_page_config(
    page_title="Llama-PcapLog Chat",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_service' not in st.session_state:
    st.session_state.rag_service = RAGService()
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar for file uploads
with st.sidebar:
    st.header("File Upload")

    # PCAP file uploader
    st.subheader("PCAP Files")
    pcap_files = st.file_uploader(
        "Upload .pcap or .pcapng files",
        accept_multiple_files=True,
        type=["pcap", "pcapng"]
    )

    # Log file uploader
    st.subheader("Log Files")
    log_files = st.file_uploader(
        "Upload log files (including files without extension)",
        accept_multiple_files=True,
        type=None
    )

    if pcap_files or log_files:
        if st.button("Process Files"):
            st.session_state.processed_data = []
            with st.spinner("Processing files..."):
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)

                # Process PCAP files
                for uploaded_file in pcap_files:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        analyzer = PacketAnalyzer()
                        data = analyzer.analyze_pcap(str(file_path))
                        st.session_state.processed_data.extend(data)
                        st.success(f"Processed PCAP file: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

                # Process Log files
                for uploaded_file in log_files:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        analyzer = SyslogAnalyzer()
                        data = analyzer.analyze_syslog(str(file_path))
                        st.session_state.processed_data.extend(data)
                        st.success(f"Processed log file: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                # Update RAG service with processed data
                if st.session_state.processed_data:
                    st.session_state.rag_service.set_processed_data(st.session_state.processed_data)
                    st.success(f"Processed {len(st.session_state.processed_data)} total entries")

# Main chat interface
st.title("Llama-PcapLog Chat")

# Display data summary if available
if st.session_state.processed_data:
    with st.expander("Data Summary", expanded=False):
        pcap_count = len([item for item in st.session_state.processed_data if item.get('type') == 'network_packet'])
        syslog_count = len([item for item in st.session_state.processed_data if item.get('type') == 'syslog'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", len(st.session_state.processed_data))
        with col2:
            st.metric("Network Packets", pcap_count)
        with col3:
            st.metric("Syslog Entries", syslog_count)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the uploaded files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_service.query(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
