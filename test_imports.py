# test_imports.py
import sys
import os

# Add app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing all imports...")

try:
    from app.core.config import settings
    print("✓ Settings imported")
    
    from app.rag.embeddings import EmbeddingEngine
    print("✓ EmbeddingEngine imported")
    
    from app.rag.retriever import MultiModalRetriever
    print("✓ MultiModalRetriever imported")
    
    from app.services.document_processor import DocumentProcessor
    print("✓ DocumentProcessor imported")
    
    from app.models.ocr_engine import HybridOCREngine
    print("✓ HybridOCREngine imported")
    
    from app.agents.orchestrator import AgentOrchestrator
    print("✓ AgentOrchestrator imported")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()