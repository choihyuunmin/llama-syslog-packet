# Llama-PcapLog

An AI-based system that analyzes syslog and PCAP files and answers questions about network traffic.

## project structure

```
Llama-PcapLog/
├── model/                   # Data processing and model training
│   ├── benchmark/           # Model benchmarking
│   ├── datasets/            # Datasets
│   ├── processors/          # Data processors
│   ├── training/            # Model training
│   └── ...
├── web/                     # Web application
│   ├── app/                 # FastAPI application
│   │   ├── api/             # API routers
│   │   ├── core/            # Core settings and utilities
│   │   ├── services/        # Business logic services
│   │   └── main.py          # Main entry point
│   ├── Dockerfile           # Docker configuration
│   └── run_streamlit.sh     # Streamlit launch script
├── pyproject.toml           # Poetry project settings
├── poetry.lock              # Dependency lock file
└── README.md                # Project documentation
```

## requirements

- Python 3.10+
- Poetry 2.0+
- Docker (optional)

## Install and Start

### 1. poetry install

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. project settings

```bash
git clone https://github.com/choihyuunmin/Llama-PcapLog
cd Llama-PcapLog
poetry install
poetry env activate
```

### 3. env

```bash
cp web/env.example web/.env
```

### 4. Start the Application

#### Run the Web API Server
```bash
cd web
./run_streamlit.sh

# Or use uvicorn directly
poetry run uvicorn web.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Run the Streamlit Dashboard
```bash
poetry run streamlit run web/app/streamlit_app.py
```

#### Using Docker
```bash
# Build and run the web application using Docker
cd web
docker build -t syspacket-web .
docker run -p 8000:8000 syspacket-web
```
