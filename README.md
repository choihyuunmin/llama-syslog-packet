# SysPacket

Syslog 및 PCAP 파일을 분석하고 네트워크 트래픽에 대한 질문에 답변하는 AI 기반 시스템입니다.

## 🏗️ 프로젝트 구조

```
syspacket/
├── model/                    # 데이터 처리 및 모델 학습
│   ├── benchmark/           # 모델 벤치마크
│   ├── datasets/            # 데이터셋
│   ├── processors/          # 데이터 처리기
│   ├── training/            # 모델 학습
│   └── ...
├── web/                     # 웹 애플리케이션
│   ├── app/                # FastAPI 애플리케이션
│   │   ├── api/           # API 라우터
│   │   ├── core/          # 핵심 설정 및 유틸리티
│   │   ├── services/      # 비즈니스 로직 서비스
│   │   └── main.py        # 애플리케이션 진입점
│   ├── Dockerfile         # Docker 설정
│   └── run_streamlit.sh   # Streamlit 실행 스크립트
├── pyproject.toml          # Poetry 프로젝트 설정
├── poetry.lock            # 의존성 잠금 파일
└── README.md              # 프로젝트 문서
```

## 🛠️ 기술 스택

- **Backend**: FastAPI (Python 3.10+)
- **AI/ML**: Transformers, PyTorch, Scikit-learn
- **네트워크 분석**: PyShark, Scapy, dpkt
- **데이터 처리**: Pandas, NumPy, Matplotlib, Seaborn
- **설정 관리**: Pydantic Settings
- **로깅**: Python logging
- **컨테이너화**: Docker
- **프로젝트 관리**: Poetry

## 📋 요구사항

- Python 3.10+
- Poetry 2.0+
- Docker (선택사항)

## 🚀 설치 및 실행

### 1. Poetry 설치

```bash
# Poetry 설치 (이미 설치되어 있다면 생략)
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. 프로젝트 설정

```bash
# 저장소 클론
git clone <repository-url>
cd syspacket

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

### 3. 환경 변수 설정

```bash
# 웹 애플리케이션 환경 변수 설정
cp web/env.example web/.env
# .env 파일 편집
```

### 4. 애플리케이션 실행

#### 웹 API 서버 실행
```bash
# 개발 모드
poetry run python web/run.py

# 또는 직접 uvicorn 사용
poetry run uvicorn web.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Streamlit 대시보드 실행
```bash
poetry run streamlit run web/app/streamlit_app.py
```

#### Docker 사용
```bash
# 웹 애플리케이션 빌드 및 실행
cd web
docker build -t syspacket-web .
docker run -p 8000:8000 syspacket-web
```
