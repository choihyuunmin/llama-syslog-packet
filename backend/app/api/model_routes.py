from fastapi import APIRouter, HTTPException
from ..core.config import AVAILABLE_MODELS
from ..services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.get("/models")
async def list_models():
    """사용 가능한 모든 모델 목록을 반환합니다."""
    return {"models": AVAILABLE_MODELS}

@router.post("/models/select")
async def select_model(model_id: str):
    """특정 모델을 선택하여 설정합니다."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        await chat_service.load_model(AVAILABLE_MODELS[model_id]["name"])
        return {"status": "success", "message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 