# app/main.py
from fastapi import FastAPI, HTTPException
from app.core.models import MultiModalDocument
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime
import logging
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import settings
from app.core.config import settings
logger.info(f"✅ Configuration loaded: {settings.APP_NAME} v{settings.VERSION}")

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
logger.info(f"📁 Upload directory: {settings.UPLOAD_DIR}")
logger.info(f"🌐 Environment: {settings.ENVIRONMENT}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"📁 Environment: {settings.ENVIRONMENT}")
    
    try:
        # Try to initialize services
        try:
            from app.services.document_processor import DocumentProcessor
            app.state.document_processor = DocumentProcessor()
            logger.info("✅ DocumentProcessor service initialized")
        except ImportError as e:
            logger.warning(f"⚠️ DocumentProcessor not available: {e}")
            app.state.document_processor = None
        except Exception as e:
            logger.error(f"❌ DocumentProcessor initialization failed: {e}")
            app.state.document_processor = None
        
        try:
            from app.rag.retriever import MultiModalRetriever
            app.state.retriever = MultiModalRetriever()
            logger.info("✅ MultiModalRetriever service initialized")
            # No need to call .initialize() - it's done in __init__
            
        except ImportError as e:
            logger.warning(f"⚠️ MultiModalRetriever not available: {e}")
            app.state.retriever = None
        except Exception as e:
            logger.error(f"❌ MultiModalRetriever initialization failed: {e}")
            app.state.retriever = None
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        # Continue without services
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multi-Modal Document Intelligence System with AI Document Auditing",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import and include API router
try:
    # Try different import paths for LangGraph
    try:
        from langgraph.checkpoint import MemorySaver
    except ImportError:
        # Try alternative import
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError:
            # Create a simple MemorySaver replacement
            class MemorySaver:
                def __init__(self):
                    self.memory = {}
                
                def get(self, config):
                    return self.memory.get(str(config), {})
                
                def put(self, config, value):
                    self.memory[str(config)] = value
    
    from app.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1")
    logger.info("✅ API router mounted successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import API router dependencies: {e}")
    logger.warning("⚠️ Creating minimal API endpoints")
    
    # Create minimal API endpoints
    from fastapi import APIRouter, UploadFile, File, BackgroundTasks
    from typing import List
    import uuid
    import shutil
    import os
    
    router = APIRouter()
    
    @router.post("/upload")
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
    ):
        """Upload a document for processing"""
        try:
            # Validate file type
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in settings.ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not supported"
                )
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Save file
            upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, f"original{file_ext}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            return {
                "success": True,
                "document_id": document_id,
                "message": "Document uploaded successfully",
                "filename": file.filename
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/status/{document_id}")
    async def get_status(document_id: str):
        """Get processing status"""
        return {
            "document_id": document_id,
            "status": "uploaded",
            "message": "Document uploaded, processing not implemented"
        }
    
    @router.get("/results/{document_id}")
    async def get_results(document_id: str):
        """Get processing results"""
        return {
            "document_id": document_id,
            "success": False,
            "message": "Processing not implemented in minimal mode",
            "results": {}
        }
    
    @router.post("/query")
    async def query_document(query_request: dict):
        """Query document"""
        return {
            "success": False,
            "message": "Query system not implemented in minimal mode",
            "answer": "Please install full system for query capabilities"
        }
    
    app.include_router(router, prefix="/api/v1")
    logger.info("✅ Minimal API router mounted")

# Health check endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    try:
        services_status = {
            "document_processor": hasattr(app.state, 'document_processor') and app.state.document_processor is not None,
            "retriever": hasattr(app.state, 'retriever') and app.state.retriever is not None,
            "mode": "full" if 'api_router' in locals() else "minimal"
        }
        
        return {
            "service": settings.APP_NAME,
            "version": settings.VERSION,
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "services": services_status,
            "endpoints": {
                "upload": "/api/v1/upload",
                "status": "/api/v1/status/{document_id}",
                "results": "/api/v1/results/{document_id}",
                "query": "/api/v1/query",
                "health": "/health",
                "docs": "/docs"
            }
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.VERSION,
            "timestamp": datetime.now().isoformat(),
            "environment": settings.ENVIRONMENT,
            "upload_dir_exists": os.path.exists(settings.UPLOAD_DIR)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "success": False,
        "error": str(exc),
        "detail": "An internal server error occurred"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )