"""
Multi-Agent System for Document Intelligence

This package contains all agents for document processing:
- Quality assessment
- Document classification
- Visual element detection
- OCR and text analysis
- Entity extraction
- Semantic reasoning
- Cross-modal alignment
- Confidence arbitration
- Consistency checking
- Contradiction detection
- Risk assessment
- Explanation generation
- Review recommendations
"""

from app.agents.orchestrator import AgentOrchestrator
from app.agents.fusion_agent import ValidationAgent

# Agent implementations (imported in orchestrator)
__all__ = [
    'AgentOrchestrator',
    'ValidationAgent'
]