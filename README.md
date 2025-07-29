# LLaMa-PcapLog

For integrated analysis of heterogeneous security data, we present Llama-PcapLog, a fine-tuned LLM framework that jointly interprets network packets (pcap) and system logs (syslog), which automates threat detection and scenario analysis. 
Unlike prior methods that treat pcap and syslog analysis modalities in isolation, Llama-PcapLog captures their temporal and contextual dependencies to detect complex attack patterns.
To bridge structural gaps and preserve cross-layer semantics, we fine-tune the open-source LLM by considering pcap and syslog data.
For fine-tuning open-source LLM, we preprocess raw pcap and syslog data into an instruction-following format. 
We also employ a self-instruct strategy to generate diverse, domain-specific tasks across Q\&A and code generation. 
As many institutions or companies need on-premise or private LLM to protect the their customer privacy and data,
we fine-tune the Meta-Llama-3-8B model on the training dataset and demonstrate a lightweight web interface for interactive analysis.
We show that Llama-PcapLog achieves substantial improvements over the base Llama-3-8B with an overall extraction F1 score increase from 0.28 to 0.68. 
Experiments also show that Lllama-PcapLog attains a perfect Pass@k score of 1.00 (vs. 0.45) in the code generation benchmark. 
These results highlight its effectiveness and potential deployability in real-world cybersecurity workflows.

## project structure

```
LLaMa-PcapLog/
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
git clone https://github.com/choihyuunmin/llama-syslog-packet
cd llama-syslog-packet
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


## Demno

