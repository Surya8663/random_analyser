from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

# Import from app modules - FAIL FAST ON IMPORT ERRORS
try:
    from app.api.routes import router
    from app.core.config import settings
    from app.utils.logger import setup_logger
    
    # Test critical imports
    from app.agents.orchestrator import AgentOrchestrator
    from app.core.models import ProcessingState
    
    HAS_MODULES = True
    logger = setup_logger(__name__)
    
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import required modules: {e}")
    print("Please ensure all dependencies are installed and the app structure is correct.")
    raise  # FAIL FAST - DO NOT CONTINUE WITH BROKEN IMPORTS

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize services - log failures but continue
    try:
        from app.services.document_processor import DocumentProcessor
        app.state.document_processor = DocumentProcessor()
        logger.info("DocumentProcessor service initialized")
    except ImportError as e:
        logger.error(f"Failed to initialize DocumentProcessor: {e}")
        app.state.document_processor = None
        # Don't raise - continue without this service
    
    # Initialize orchestrator
    try:
        app.state.orchestrator = AgentOrchestrator()
        logger.info("AgentOrchestrator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize AgentOrchestrator: {e}")
        raise  # CRITICAL - cannot continue without orchestrator
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

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
    allow_origins=["*"] if settings.DEBUG else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main router
app.include_router(router, prefix="/api/v1")

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "upload": "/api/v1/upload",
            "process": "/api/v1/process",
            "status": "/api/v1/status/{job_id}",
            "results": "/api/v1/results/{document_id}",
            "search": "/api/v1/search",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )