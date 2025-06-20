# SysPacket Analysis API

Syslog 및 PCAP 파일을 분석하고 네트워크 트래픽에 대한 질문에 답변하는 FastAPI 기반 웹 애플리케이션입니다.

## Structure

```
web/
├── app/
│   ├── api/                    # API 라우터
│   │   ├── chat_routes.py     # 채팅 관련 엔드포인트
│   │   ├── file_routes.py     # 파일 관리 엔드포인트
│   │   ├── log_routes.py      # 로그 관련 엔드포인트
│   │   └── model_routes.py    # 모델 관리 엔드포인트
│   ├── core/                  # 핵심 설정 및 유틸리티
│   │   ├── config.py         # 애플리케이션 설정
│   │   ├── dependencies.py   # 의존성 주입
│   │   └── utils.py          # 유틸리티 함수
│   ├── services/             # 비즈니스 로직 서비스
│   │   ├── chat_service.py   # 채팅 서비스
│   │   ├── file_service.py   # 파일 관리 서비스
│   └── main.py              # FastAPI 애플리케이션 진입점
├── pyproject.toml            # 프로젝트 설정 및 의존성
├── Dockerfile               # Docker 설정
└── env.example              # 환경 변수 예제
```

- **Backend**: FastAPI (Python 3.10+)
- **AI/ML**: Transformers, PyTorch
- **네트워크 분석**: PyShark
- **설정 관리**: Pydantic Settings
- **로깅**: Python logging
- **컨테이너화**: Docker
- **프로젝트 관리**: pyproject.toml (PEP 621)

- Python 3.10+
- Docker (선택사항)

## Installation and Start

### 1. environment settings

```bash
git clone <repository-url>
cd web

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[dev]"

# 또는 프로덕션 환경
pip install -e .
```

### 2. set env

```bash
cp env.example .env

```

### 3. application start

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# or
docker build -t syspacket-api .
docker run -p 8000:8000 syspacket-api
```
