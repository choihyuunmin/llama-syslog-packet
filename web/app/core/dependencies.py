from core.config import settings
from services.file_service import FileService
from services.chat_service import ChatService

# Global service instances
_file_service_instance = None
_chat_service_instance = None

def get_file_service() -> FileService:
    global _file_service_instance
    if _file_service_instance is None:
        _file_service_instance = FileService()
    return _file_service_instance

def get_chat_service() -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance

def get_settings():
    return settings 