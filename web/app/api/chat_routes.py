from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
from web.app.services.chat_service import ChatService, ModelLoadError
from web.app.core.dependencies import get_chat_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str
    model: Optional[str] = None


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str
    model: str
    input_length: int
    output_length: int


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """
    채팅 메시지에 대한 응답을 생성합니다.
    
    Args:
        request: 채팅 요청
        chat_service: 채팅 서비스 인스턴스
        
    Returns:
        ChatResponse: 생성된 응답
        
    Raises:
        HTTPException: 응답 생성 실패 시
    """
    try:
        logger.info(f"채팅 요청: {len(request.message)}자 메시지, 모델: {request.model or '기본'}")
        
        result = await chat_service.generate_response(
            message=request.message,
            model=request.model
        )
        
        response = ChatResponse(**result)
        logger.info(f"채팅 응답 생성 완료: {response.output_length}자 응답")
        return response
        
    except ModelLoadError as e:
        logger.error(f"모델 로딩 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {str(e)}")
    except Exception as e:
        logger.error(f"채팅 응답 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="응답 생성 중 오류가 발생했습니다")


@router.get("/models", response_model=Dict[str, Dict[str, str]])
async def get_available_models(
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, Dict[str, str]]:
    """
    사용 가능한 모델 목록을 반환합니다.
    
    Args:
        chat_service: 채팅 서비스 인스턴스
        
    Returns:
        Dict[str, Dict[str, str]]: 사용 가능한 모델 정보
    """
    try:
        models = chat_service.get_available_models()
        logger.info(f"사용 가능한 모델 목록 조회: {len(models)}개 모델")
        return models
    except Exception as e:
        logger.error(f"모델 목록 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="모델 목록 조회 중 오류가 발생했습니다")


@router.get("/models/current", response_model=Dict[str, str])
async def get_current_model(
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, str]:
    """
    현재 로딩된 모델 정보를 반환합니다.
    
    Args:
        chat_service: 채팅 서비스 인스턴스
        
    Returns:
        Dict[str, str]: 현재 모델 정보
    """
    try:
        current_model = chat_service.get_current_model()
        logger.info(f"현재 모델 조회: {current_model}")
        return {"current_model": current_model}
    except Exception as e:
        logger.error(f"현재 모델 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="현재 모델 조회 중 오류가 발생했습니다")


@router.post("/models/switch", response_model=Dict[str, str])
async def switch_model(
    model: str,
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, str]:
    """
    모델을 변경합니다.
    
    Args:
        model: 변경할 모델명
        chat_service: 채팅 서비스 인스턴스
        
    Returns:
        Dict[str, str]: 모델 변경 결과
        
    Raises:
        HTTPException: 모델 변경 실패 시
    """
    try:
        logger.info(f"모델 변경 요청: {model}")
        
        # 모델 변경을 위해 새 모델로 응답 생성 시도
        test_message = "테스트"
        await chat_service.generate_response(test_message, model)
        
        logger.info(f"모델 변경 완료: {model}")
        return {"message": f"모델이 '{model}'로 변경되었습니다", "current_model": model}
        
    except ModelLoadError as e:
        logger.error(f"모델 변경 실패: {str(e)}")
        raise HTTPException(status_code=400, detail=f"모델 변경 실패: {str(e)}")
    except Exception as e:
        logger.error(f"모델 변경 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="모델 변경 중 오류가 발생했습니다") 