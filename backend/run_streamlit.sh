#!/bin/bash

# 가상환경이 활성화되어 있는지 확인
if [ -z "$VIRTUAL_ENV" ]; then
    echo "가상환경을 활성화해주세요."
    exit 1
fi

# Streamlit 앱 실행
streamlit run app/streamlit_app.py 