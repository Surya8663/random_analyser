"""
Agents package for multi-agent document intelligence system
"""

__all__ = [
    "AgentOrchestrator",
    "ValidationAgent",
    "DocumentQualityAgent",
    "DocumentTypeClassifier",
    "VisualElementDetector"
]

# Define agent interfaces for type checking
class BaseAgent:
    """Base class for all agents"""
    async def __call__(self, state):
        raise NotImplementedError("Agents must implement __call__ method")

# Note: Actual agent implementations are imported dynamically by the orchestrator
# This file provides type hints and documentation