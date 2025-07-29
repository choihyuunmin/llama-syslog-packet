from fastapi import APIRouter, HTTPException, Depends
from core.config import AVAILABLE_MODELS
from services.chat_service import ChatService
from core.dependencies import get_chat_service

router = APIRouter()

@router.get("/models")
async def list_models():
    return {"models": AVAILABLE_MODELS}

@router.post("/models/select")
async def select_model(
    model_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        chat_service.load_model(model_id)
        return {"status": "success", "message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 