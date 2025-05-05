from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

class ChatRequest(BaseModel):
    message: str
    model: str

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await chat_service.generate_response(request.message, request.model)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 