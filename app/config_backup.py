from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Vision-Fusion Document Intelligence"
    VERSION: str = "1.0.0"
    ENV: str = "development"
    DEBUG: bool = True

    # API
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # Uploads
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg"]

    # OCR
    OCR_DPI: int = 300
    TESSERACT_PATH: str | None = None

    # Vision / YOLO
    YOLO_MODEL: str = "yolov8n.pt"

    # Vector DB
    VECTOR_SIZE: int = 384
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # LLM / LangChain
    LLM_TIMEOUT: int = 120
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = ""

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid"  # keep strict (GOOD)
    )

settings = Settings()
