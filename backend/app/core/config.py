from typing import Dict, List

AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
    "mobile-llama": {
        "name": "choihyuunmin/mobile-Llama-3-Instruct",
        "description": "모바일 환경에 최적화된 가벼운 LLaMA 모델",
        "type": "llama"
    },
    "ko-alpaca": {
        "name": "beomi/KoAlpaca-Polyglot-5.8B",
        "description": "한국어에 특화된 Alpaca 모델",
        "type": "alpaca"
    },
    "polyglot": {
        "name": "EleutherAI/polyglot-ko-5.8b",
        "description": "한국어 자연어 처리에 특화된 Polyglot 모델",
        "type": "polyglot"
    }
} 