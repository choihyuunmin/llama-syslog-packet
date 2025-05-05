import streamlit as st
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.train_model import train_model

def main():
    st.set_page_config(
        page_title="syslog-packet analyzer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("syslog-packet analyzer")
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("학습 설정")
        
        # 데이터셋 경로 선택
        dataset_path = st.text_input(
            "데이터셋 경로",
            value="datasets",
            help="학습할 데이터셋이 있는 디렉토리 경로"
        )
        
        # 모델 선택
        model_name = st.selectbox(
            "모델 선택",
            options=["meta-llama/Llama-3.2-3B-Instruct", "openai-community/gpt2-medium", "deepseek-ai/DeepSeek-V3"],
            help="학습할 모델을 선택하세요"
        )
        
        # 출력 디렉토리 설정
        output_dir = st.text_input(
            "출력 디렉토리",
            value="output",
            help="학습된 모델이 저장될 디렉토리"
        )
        
        # 학습 파라미터 설정
        st.subheader("학습 파라미터")
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input(
                "batch size",
                min_value=1,
                max_value=32,
                value=4,
                step=1
            )
            
            epochs = st.number_input(
                "epoch count",
                min_value=1,
                max_value=100,
                value=3,
                step=1
            )
            
            learning_rate = st.number_input(
                "learning rate",
                min_value=1e-6,
                max_value=1e-2,
                value=2e-5,
                format="%.6f"
            )
            
        with col2:
            max_grad_norm = st.number_input(
                "max gradient norm",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            
            warmup_ratio = st.number_input(
                "warmup ratio",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.01
            )
            
            gradient_accumulation_steps = st.number_input(
                "gradient accumulation step",
                min_value=1,
                max_value=16,
                value=4,
                step=1
            )
        
        # 고급 설정
        st.subheader("고급 설정")
        col3, col4 = st.columns(2)
        
        with col3:
            fp16 = st.checkbox("FP16 학습", value=True)
            group_by_length = st.checkbox("시퀀스 길이별 그룹화", value=True)
            load_best_model_at_end = st.checkbox("학습 종료 시 최적 모델 로드", value=True)
            
        with col4:
            lr_scheduler_type = st.selectbox(
                "learn scheduler",
                options=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                index=1
            )
            
            optim = st.selectbox(
                "optimizer",
                options=["adamw_torch", "adamw_hf", "adamw_apex_fused", "adafactor"],
                index=0
            )
    
    # 메인 콘텐츠
    st.header("학습 시작")
    
    if st.button("학습 시작"):
        with st.spinner("모델 학습 중..."):
            try:
                # 학습 시작
                train_model(
                    dataset_path=dataset_path,
                    model_name=model_name,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    max_grad_norm=max_grad_norm,
                    warmup_ratio=warmup_ratio,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    fp16=fp16,
                    group_by_length=group_by_length,
                    load_best_model_at_end=load_best_model_at_end,
                    lr_scheduler_type=lr_scheduler_type,
                    optim=optim
                )
                
                st.success("학습이 완료되었습니다!")
                st.info(f"학습된 모델이 {output_dir} 디렉토리에 저장되었습니다.")
                
            except Exception as e:
                st.error(f"학습 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 