import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

from web.app.services.rag_service import RAGService
from web.app.services.code_executor import CodeExecutor

st.set_page_config(
    page_title="SysPacket Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'rag_service' not in st.session_state:
    st.session_state.rag_service = RAGService()
if 'code_executor' not in st.session_state:
    st.session_state.code_executor = CodeExecutor()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'file_data' not in st.session_state:
    st.session_state.file_data = None

def main():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f2937;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #6b7280;
            text-align: center;
            margin-bottom: 3rem;
        }
        .code-output {
            background: #1f2937;
            color: #f9fafb;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .file-info {
            background: #e0f2fe;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">SysPacket Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload PCAP or Syslog files and chat with AI for analysis</p>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 File Upload")
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        use_openai = st.checkbox("Use GPT-3.5-turbo (requires OpenAI API key)", value=False)
        
        if use_openai:
            st.info("Make sure to set OPENAI_API_KEY in your environment variables")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            help="Upload PCAP or Syslog files for analysis"
        )
        
        if uploaded_file is not None:
            if st.button("Process File", type="primary"):
                with st.spinner("Processing file..."):
                    try:
                        # Save uploaded file
                        file_path = Path("temp") / uploaded_file.name
                        file_path.parent.mkdir(exist_ok=True)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Determine file type
                        file_type = 'pcap' if uploaded_file.name.endswith('.pcap') or uploaded_file.name.endswith('.pcapng') else 'log'
                        
                        # Initialize RAG service with model choice
                        if 'rag_service' not in st.session_state or st.session_state.get('use_openai') != use_openai:
                            st.session_state.rag_service = RAGService(use_openai=use_openai)
                            st.session_state.use_openai = use_openai
                        
                        # Process file with RAG service
                        file_data = st.session_state.rag_service.process_file(str(file_path), file_type)
                        
                        st.session_state.current_file = str(file_path)
                        st.session_state.file_data = file_data
                        
                        # Reset code executor with new data
                        st.session_state.code_executor.reset_environment()
                        if file_type == 'pcap' or file_type == 'pcapng':
                            st.session_state.code_executor.global_vars['packets'] = file_data['packets']
                        else:
                            st.session_state.code_executor.global_vars['logs'] = file_data['logs']
                        
                        st.success(f"File processed successfully! {len(file_data.get('packets', file_data.get('logs', [])))} records loaded.")
                        
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        
        # File info
        if st.session_state.current_file:
            st.markdown("### 📊 File Information")
            context = st.session_state.rag_service.get_context()
            st.write(f"**File:** {Path(context['current_file']).name}")
            st.write(f"**Type:** {context['file_type']}")
            st.write(f"**Status:** {'Ready' if context['vector_store_ready'] else 'Processing'}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Chat Interface")
        
        # Custom CSS
        st.markdown("""
            <style>
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #fafafa;
            }
            .chat-message {
                margin-bottom: 15px;
                padding: 12px;
                border-radius: 12px;
                max-width: 85%;
                display: flex;
                align-items: flex-start;
                gap: 10px;
                word-wrap: break-word;
            }
            .chat-message-user {
                background-color: #e3f2fd;
                margin-left: auto;
                flex-direction: row-reverse;
            }
            .chat-message-assistant {
                background-color: #f5f5f5;
                margin-right: auto;
            }
            .chat-emoji {
                font-size: 1.5rem;
                flex-shrink: 0;
            }
            .chat-content {
                flex: 1;
                line-height: 1.4;
            }
            .chat-input-area {
                border-top: 1px solid #e0e0e0;
                padding-top: 15px;
                margin-top: 15px;
            }
            </style>
        """, unsafe_allow_html=True)
          
        # Display existing messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                emoji = "🙋"
                role_class = "chat-message-user"
            else:
                emoji = "🤖"
                role_class = "chat-message-assistant"

            # Chat message output
            st.markdown(f'''
                <div class="chat-message {role_class}">
                    <div class="chat-emoji">{emoji}</div>
                    <div class="chat-content">{message["content"]}</div>
                </div>
            ''', unsafe_allow_html=True)

            # Code execution results output
            if message.get("code_results"):
                st.markdown("**Code Execution Results:**")
                for result in message["code_results"]["results"]:
                    if result["success"]:
                        if result["stdout"]:
                            st.code(result["stdout"], language="text")
                        if result["figures"]:
                            for fig in result["figures"]:
                                st.image(BytesIO(fig["image_data"]), caption=f"Figure {fig['figure_number']}")
                    else:
                        st.error(f"Code execution failed: {result['error']['message']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input area
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        prompt = st.chat_input("Ask about your file...")
        
        if prompt:
            if not st.session_state.current_file:
                st.error("Please upload and process a file first.")
                st.stop()

            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                response = st.session_state.rag_service.query(prompt)

                # 코드 실행
                code_results = st.session_state.code_executor.execute_from_response(
                    response,
                    st.session_state.code_executor.global_vars
                )

                # 어시스턴트 응답 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "code_results": code_results if code_results["has_code"] else None
                })

            # 화면 새로고침
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Quick Analysis")
        
        if st.session_state.file_data:
            file_type = "pcap" if "packets" in st.session_state.file_data else "log"
            
            if file_type == "pcap":
                packets = st.session_state.file_data["packets"]
                if packets:
                    # Protocol distribution
                    protocols = {}
                    for packet in packets:
                        protocol = packet.get('protocol', 'Unknown')
                        protocols[protocol] = protocols.get(protocol, 0) + 1
                    
                    st.markdown("**Protocol Distribution**")
                    protocol_df = pd.DataFrame(list(protocols.items()), columns=['Protocol', 'Count'])
                    st.bar_chart(protocol_df.set_index('Protocol'))
                    
                    # Packet size distribution
                    sizes = [p.get('length', 0) for p in packets]
                    st.markdown("**Packet Size Distribution**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(sizes, bins=30, alpha=0.7)
                    ax.set_xlabel('Packet Size (bytes)')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    plt.close()
            
            else:  # log files
                logs = st.session_state.file_data["logs"]
                if logs:
                    # Severity distribution
                    severities = {}
                    for log in logs:
                        severity = log.get('severity', 'unknown')
                        severities[severity] = severities.get(severity, 0) + 1
                    
                    st.markdown("**Log Severity Distribution**")
                    severity_df = pd.DataFrame(list(severities.items()), columns=['Severity', 'Count'])
                    st.bar_chart(severity_df.set_index('Severity'))
        
        # Available variables
        if st.session_state.code_executor.global_vars:
            st.markdown("### 🔧 Available Variables")
            for var_name in st.session_state.code_executor.global_vars.keys():
                if not var_name.startswith('_'):
                    st.write(f"• `{var_name}`")
    
    # Clear chat button
    if st.session_state.messages and st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()