"""
Vision-Fusion Document Intelligence System

A multi-modal document analysis system combining:
- Computer Vision for layout and element detection
- OCR for text extraction
- Multi-Agent system for comprehensive analysis
- RAG for semantic search and querying
"""

__version__ = "1.0.0"
__author__ = "AI Agents Builder System"
__description__ = "Multi-Modal Document Intelligence Platform"

# Export main components
from app.core.config import settings
from app.agents.orchestrator import AgentOrchestrator
from app.rag.retriever import MultiModalRetriever

__all__ = [
    'settings',
    'AgentOrchestrator',
    'MultiModalRetriever'
]