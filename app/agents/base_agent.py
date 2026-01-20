# app/agents/base_agent.py - NEW FILE
from abc import ABC, abstractmethod
from typing import Dict, Any
from app.core.models import MultiModalDocument

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    @abstractmethod
    async def process(self, document: MultiModalDocument) -> MultiModalDocument:
        """Process document and return updated document"""
        pass
    
    def get_name(self) -> str:
        """Get agent name for logging and identification"""
        return self.__class__.__name__