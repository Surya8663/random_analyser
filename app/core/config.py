# app/core/config.py
import os
from typing import Optional, List
from pydantic_settings import BaseSettings

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
    TESSERACT_PATH: Optional[str] = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    
    # Agent Settings
    AGENT_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Embeddings
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Qdrant Vector Database - ADD THESE
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "document_embeddings"
    QDRANT_TIMEOUT: int = 30
    
    # Monitoring
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars

# Create settings instance
try:
    settings = Settings()
    print(f"🔧 Configuration loaded:")
    print(f"   App: {settings.APP_NAME} v{settings.VERSION}")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   Server: {settings.HOST}:{settings.PORT}")
    print(f"   Upload Dir: {settings.UPLOAD_DIR}")
    print(f"   Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"   OCR Threshold: {settings.OCR_CONFIDENCE_THRESHOLD}")
    print(f"   Tesseract Path: {settings.TESSERACT_PATH}")
    print(f"   Text Model: {settings.TEXT_EMBEDDING_MODEL}")
    print(f"   Qdrant Collection: {settings.QDRANT_COLLECTION}")
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
except Exception as e:
    print(f"⚠️ Configuration warning: {e}")
    print("Using default configuration")
    
    # Fallback with all required attributes
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
        OCR_DPI = 300
        OCR_CONFIDENCE_THRESHOLD = 0.6
        TESSERACT_PATH = "C:/Program Files/Tesseract-OCR/tesseract.exe"
        AGENT_CONFIDENCE_THRESHOLD = 0.5
        TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        EMBEDDING_DIMENSION = 384
        QDRANT_HOST = "localhost"
        QDRANT_PORT = 6333
        QDRANT_COLLECTION = "document_embeddings"
        QDRANT_TIMEOUT = 30
        LOG_LEVEL = "INFO"
    
    settings = MinimalSettings()
    
    # Create directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)