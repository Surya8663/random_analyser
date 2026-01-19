import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from app.config import settings

__all__ = ['settings']
class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Vision-Fusion Document Intelligence"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # File Processing
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".txt"]
    TEMP_DIR: str = "temp"
    
    # OCR Settings (Optional)
    OCR_DPI: int = 300
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Agent Settings
    AGENT_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Monitoring
    LOG_LEVEL: str = "INFO"
    
    # Make all other fields optional
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars

# Create settings instance
try:
    settings = Settings()
    print(f"✅ Configuration loaded: {settings.APP_NAME} v{settings.VERSION}")
    print(f"📁 Upload directory: {settings.UPLOAD_DIR}")
    print(f"🌐 Environment: {settings.ENVIRONMENT}")
except Exception as e:
    print(f"⚠️ Configuration warning: {e}")
    print("Using default configuration")
    
    # Fallback
    class MinimalSettings:
        APP_NAME = "Vision-Fusion Document Intelligence"
        VERSION = "1.0.0"
        DEBUG = True
        ENVIRONMENT = "development"
        HOST = "0.0.0.0"
        PORT = 8000
        WORKERS = 1
        UPLOAD_DIR = "uploads"
        MAX_FILE_SIZE = 50 * 1024 * 1024
        ALLOWED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".txt"]
        TEMP_DIR = "temp"
        LOG_LEVEL = "INFO"
    
    settings = MinimalSettings()