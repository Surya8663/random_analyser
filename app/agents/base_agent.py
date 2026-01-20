# app/agents/base_agent.py - CORRECTED
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from app.core.models import MultiModalDocument, ProvenanceRecord
from app.explain.provenance import ProvenanceTracker

class BaseAgent(ABC):
    """Base class for all agents with explainability support"""
    
    def __init__(self):
        self._provenance_tracker = None
        self._accepts_multi_modal = True
    
    @abstractmethod
    async def process(self, document: MultiModalDocument) -> MultiModalDocument:
        """Process document and return updated document"""
        pass
    
    def get_name(self) -> str:
        """Get agent name for logging and identification"""
        return self.__class__.__name__
    
    def set_provenance_tracker(self, tracker: ProvenanceTracker):
        """Set provenance tracker for this agent"""
        self._provenance_tracker = tracker
    
    def _record_provenance(self, 
                          field_name: str,
                          extraction_method: str,
                          source_modality: str,
                          confidence: float,
                          source_bbox: Optional[Any] = None,
                          source_page: Optional[int] = None,
                          reasoning_notes: Optional[str] = None) -> Optional[ProvenanceRecord]:
        """Record provenance for a field extraction"""
        if self._provenance_tracker:
            return self._provenance_tracker.create_provenance_record(
                agent_name=self.get_name(),
                extraction_method=extraction_method,
                source_modality=source_modality,
                field_name=field_name,
                confidence=confidence,
                source_bbox=source_bbox,
                source_page=source_page,
                reasoning_notes=reasoning_notes
            )
        return None