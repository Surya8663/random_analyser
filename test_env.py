import os
from dotenv import load_dotenv

load_dotenv()

print("TESSERACT:", os.getenv("TESSERACT_PATH"))
print("YOLO MODEL:", os.getenv("YOLO_MODEL"))
print("OLLAMA:", os.getenv("OLLAMA_BASE_URL"))
print("QDRANT:", os.getenv("QDRANT_HOST"), os.getenv("QDRANT_PORT"))
print("REDIS:", os.getenv("REDIS_HOST"), os.getenv("REDIS_PORT"))
