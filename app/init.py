"""
Vision-Fusion Document Intelligence System
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "gomathi"

# Export main components for easy access
__all__ = [
    "AgentOrchestrator",
    "ProcessingState",
    "settings"
]

# Import critical components to ensure they're available
try:
    from app.core.config import settings
    from app.core.models import ProcessingState
    from app.agents.orchestrator import AgentOrchestrator
    
    # Test that imports work
    _ = settings.APP_NAME
    
    print(f"✅ {settings.APP_NAME} v{settings.VERSION} initialized successfully")
    
except ImportError as e:
    print(f"❌ Failed to initialize app package: {e}")
    raise
except Exception as e:
    print(f"⚠️ Warning during app initialization: {e}")