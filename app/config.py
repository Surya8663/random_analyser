import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Vision-Fusion Document Intelligence"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # File Processing
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 104857600  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    TEMP_DIR: str = "temp"
    
    # OCR Settings
    TESSERACT_PATH: Optional[str] = None
    OCR_DPI: int = 300
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    
    # Vision / YOLO Settings
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    
    # Vector Database (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "document_embeddings"
    
    # Embeddings
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    
    # Agent Settings
    AGENT_CONFIDENCE_THRESHOLD: float = 0.6
    CONTRADICTION_SEVERITY_THRESHOLD: float = 0.7
    RISK_SCORE_THRESHOLD: float = 0.8
    
    # Cache / Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_TTL: int = 3600
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields
        case_sensitive = False

# Create settings instance
settings = Settings()

# Print configuration for debugging
if settings.DEBUG:
    print(f"🔧 Configuration loaded:")
    print(f"   App: {settings.APP_NAME} v{settings.VERSION}")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   Server: {settings.HOST}:{settings.PORT}")
    print(f"   Upload Dir: {settings.UPLOAD_DIR}")
    print(f"   Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"   OCR Threshold: {settings.OCR_CONFIDENCE_THRESHOLD}")
    print(f"   Tesseract Path: {settings.TESSERACT_PATH or 'Not set'}")
    print(f"   Text Model: {settings.TEXT_EMBEDDING_MODEL}")
    print(f"   Qdrant Collection: {settings.QDRANT_COLLECTION}")