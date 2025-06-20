"""
의존성 주입을 위한 공통 의존성 모듈
"""
from typing import Generator
from fastapi import Depends
from web.app.core.config import settings
from web.app.services.file_service import FileService
from web.app.services.chat_service import ChatService


def get_file_service() -> FileService:
    """
    FileService 인스턴스를 반환하는 의존성 함수
    
    Returns:
        FileService: 파일 서비스 인스턴스
    """
    return FileService()


def get_chat_service() -> ChatService:
    """
    ChatService 인스턴스를 반환하는 의존성 함수
    
    Returns:
        ChatService: 채팅 서비스 인스턴스
    """
    return ChatService()


def get_settings():
    """
    애플리케이션 설정을 반환하는 의존성 함수
    
    Returns:
        Settings: 애플리케이션 설정
    """
    return settings 