import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import re
import torch
from dotenv import load_dotenv
from services.chat_service import ChatService
from services.packet_analyzer import PacketAnalyzer

# 환경 변수 로드
load_dotenv()

# PyTorch 설정
torch.set_grad_enabled(False)

# 경로 설정
sys.path.append(str(Path(__file__).parent.parent))

# Streamlit 설정
st.set_page_config(
    page_title="SysPacket Analysis Tool",
    page_icon="📂",
    layout="wide"
)

# 세션 상태 초기화
if 'chat_service' not in st.session_state:
    try:
        st.session_state.chat_service = ChatService()
    except Exception as e:
        st.error(f"Error initializing chat service: {e}")
        st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'file_profile' not in st.session_state:
    st.session_state.file_profile = None
if 'code_snippets' not in st.session_state:
    st.session_state.code_snippets = []

def process_uploaded_file(file_path: str) -> None:
    """업로드된 파일 처리 및 RAG에 저장"""
    try:
        analyzer = PacketAnalyzer()
        if file_path.endswith('.pcap'):
            # PCAP 파일 분석
            result = analyzer.analyze_pcap(file_path)
            # 패킷 데이터를 RAG에 저장
            st.session_state.chat_service.process_packet_data(analyzer.packets)
        elif file_path.endswith('.log'):
            # Syslog 파일 분석
            result = analyzer.analyze_syslog(file_path)
            # 로그 데이터를 RAG에 저장
            st.session_state.chat_service.process_packet_data(analyzer.logs)
        
        # 파일 프로필 업데이트
        st.session_state.file_profile = {
            "file_name": os.path.basename(file_path),
            "file_type": "pcap" if file_path.endswith(".pcap") else "log",
            "file_size": os.path.getsize(file_path),
            "packet_count": len(analyzer.packets) if hasattr(analyzer, 'packets') else len(analyzer.logs),
            "protocols": list(result["protocol_dist"]["distribution"].keys()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.success("파일이 성공적으로 처리되었습니다.")
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

def get_llm_response_sync(message: str) -> str:
    try:
        result = st.session_state.chat_service.generate_response_sync(message)
        # 컨텍스트 정보 표시
        if result.get("context_used"):
            st.info("사용된 컨텍스트 정보:")
            st.text(result["context_used"])
        return result["response"]
    except Exception as e:
        st.error(f"Error getting LLM response: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

def extract_code_blocks(text: str) -> List[str]:
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    return [block.strip() for block in code_blocks]

def display_file_profile(profile: Dict[str, Any]):
    st.markdown("**File Information**", unsafe_allow_html=True)
    st.write(f"**Filename:** {profile['file_name']}")
    st.write(f"**File Type:** {profile['file_type']}")
    st.write(f"**File Size:** {profile['file_size'] / 1024:.2f} KB")
    st.write(f"**Packet Count:** {profile['packet_count']}")
    st.write("**Protocols:**")
    for protocol in profile['protocols']:
        st.write(f"- {protocol}")
    st.write(f"**Timestamp:** {profile['timestamp']}")

def display_chat_interface():
    st.markdown("<div style='font-size:1.1rem;font-weight:600;margin-bottom:8px;'>Chat</div>", unsafe_allow_html=True)
    st.markdown("""
        <style>
        .chat-box {
            height: 320px;
            overflow-y: auto;
            border: none;
            border-radius: 10px;
            padding: 18px 12px 12px 12px;
            background-color: #fff;
            margin-bottom: 8px;
            box-shadow: 0 1px 6px 0 rgba(0,0,0,0.04);
        }
        .chat-message-card {
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
            max-width: 80%;
            box-shadow: 0 1px 6px 0 rgba(0,0,0,0.04);
            font-size: 1.01rem;
            word-break: break-word;
        }
        .chat-message-user {
            background: #e6f0ff;
            margin-left: auto;
            text-align: right;
        }
        .chat-message-assistant {
            background: #f4f6fa;
            margin-right: auto;
            text-align: left;
        }
        .chat-input-custom {
            width: 100%;
            background: #f4f6fa;
            border-radius: 8px;
            border: none;
            padding: 12px 16px;
            font-size: 1rem;
            margin-top: 8px;
        }
        .send-btn {
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 8px 18px;
            margin-left: 8px;
            font-size: 1.1rem;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)
    
    chat_box_style = '<div class="chat-box">'
    chat_box_end = '</div>'
    st.markdown(chat_box_style, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        role_class = "chat-message-user" if message["role"] == "user" else "chat-message-assistant"
        card_html = f'<div class="chat-message-card {role_class}">{message["content"]}</div>'
        st.markdown(card_html, unsafe_allow_html=True)
        if message["role"] == "assistant":
            code_blocks = extract_code_blocks(message["content"])
            if code_blocks:
                st.session_state.code_snippets.extend(code_blocks)
    
    st.markdown(chat_box_end, unsafe_allow_html=True)
    
    # 입력창
    prompt = st.chat_input("Type your message here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = get_llm_response_sync(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

def display_visualization():
    st.markdown("<div style='font-size:1.1rem;font-weight:600;margin-bottom:8px;'>Visualization</div>", unsafe_allow_html=True)
    if st.session_state.current_file:
        analyzer = PacketAnalyzer()
        result = analyzer.analyze_pcap(st.session_state.current_file)
        
        # 프로토콜 분포 bar chart
        proto_dist = result["protocol_dist"]["distribution"]
        proto_df = pd.DataFrame({
            'Protocol': list(proto_dist.keys()),
            'Count': list(proto_dist.values())
        })
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.barplot(data=proto_df, x='Protocol', y='Count', ax=ax)
        st.pyplot(fig)
        
        # 패킷 크기 분포 히스토그램
        df = pd.DataFrame(analyzer.packets)
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        sns.histplot(df['length'], bins=30, kde=True, ax=ax2, color='skyblue')
        ax2.set_title('Packet Size Distribution')
        ax2.set_xlabel('Packet Size (bytes)')
        ax2.set_ylabel('Count')
        st.pyplot(fig2)
    else:
        st.info("Analysis results will be displayed here.")

def main():
    # 전체 배경 및 상단바/푸터 스타일
    st.markdown("""
        <style>
        .main-bg {background-color: #f6f7fa;}
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 10px 6px 10px;
            background: #fff;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 12px;
        }
        .logo-title {
            font-size: 1.15rem;
            font-weight: 700;
            display: flex;
            align-items: center;
        }
        .settings-btn {
            background: #f4f6fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 6px 16px;
            font-size: 0.98rem;
            cursor: pointer;
        }
        .footer {
            text-align: left;
            color: #888;
            font-size: 0.93rem;
            margin-top: 24px;
            margin-bottom: 6px;
        }
        </style>
        <div class='main-bg'>
            <div class='top-bar'>
                <div class='logo-title'>
                    <span style='font-size:1.15rem;margin-right:7px;'>📂</span> SysPacket Analysis Tool
                </div>
                <button class='settings-btn'>Settings</button>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3분할 레이아웃
    left_col, middle_col, right_col = st.columns([1.15, 2.3, 1.15], gap="large")

    with left_col:
        uploaded_file = st.file_uploader("Select File", type=['pcap', 'log'])
        if st.button("Upload", use_container_width=True) and uploaded_file is not None:
            file_path = os.path.join("uploads", uploaded_file.name)
            os.makedirs("uploads", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.current_file = file_path
            process_uploaded_file(file_path)
        
        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        if st.session_state.file_profile:
            display_file_profile(st.session_state.file_profile)
        else:
            st.info("No file selected")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with middle_col:
        display_chat_interface()
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        display_visualization()
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class='footer'>
            © 2025 SysPacket. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()