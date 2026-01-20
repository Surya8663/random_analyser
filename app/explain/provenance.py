# app/explain/provenance.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.core.models import ProvenanceRecord, BoundingBox, MultiModalDocument
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ProvenanceTracker:
    """Track provenance across the processing pipeline"""
    
    def __init__(self):
        self._agent_trace = {}
        self._field_provenance = {}
        self._current_document = None
    
    def start_document(self, document: MultiModalDocument):
        """Start tracking for a new document"""
        self._current_document = document
        self._agent_trace = {}
        self._field_provenance = {}
        logger.info(f"📝 Started provenance tracking for document: {document.document_id}")
    
    def record_agent_start(self, agent_name: str, timestamp: datetime = None):
        """Record when an agent starts processing"""
        if agent_name not in self._agent_trace:
            self._agent_trace[agent_name] = {
                "start_time": timestamp or datetime.now(),
                "end_time": None,
                "fields_extracted": [],
                "errors": []
            }
    
    def record_agent_end(self, agent_name: str, fields_extracted: List[str], 
                        errors: List[str] = None, timestamp: datetime = None):
        """Record when an agent finishes processing"""
        if agent_name in self._agent_trace:
            self._agent_trace[agent_name]["end_time"] = timestamp or datetime.now()
            self._agent_trace[agent_name]["fields_extracted"] = fields_extracted
            if errors:
                self._agent_trace[agent_name]["errors"] = errors
    
    def create_provenance_record(self,
                                agent_name: str,
                                extraction_method: str,
                                source_modality: str,
                                field_name: str,
                                confidence: float,
                                source_bbox: Optional[BoundingBox] = None,
                                source_page: Optional[int] = None,
                                reasoning_notes: Optional[str] = None) -> ProvenanceRecord:
        """Create a provenance record for a field extraction"""
        # Generate source region ID
        source_region_id = None
        if source_bbox:
            source_region_id = f"{agent_name}_{field_name}_{source_page}_{hash(str(source_bbox.to_list()))}"
        
        record = ProvenanceRecord(
            agent_name=agent_name,
            extraction_method=extraction_method,
            source_modality=source_modality,
            source_region_id=source_region_id,
            source_bbox=source_bbox,
            source_page=source_page,
            confidence=confidence,
            reasoning_notes=reasoning_notes
        )
        
        # Store in field provenance
        if field_name not in self._field_provenance:
            self._field_provenance[field_name] = []
        self._field_provenance[field_name].append(record)
        
        return record
    
    def get_field_provenance(self, field_name: str) -> List[ProvenanceRecord]:
        """Get provenance records for a specific field"""
        return self._field_provenance.get(field_name, [])
    
    def get_agent_trace(self, agent_name: str) -> Dict[str, Any]:
        """Get trace information for a specific agent"""
        return self._agent_trace.get(agent_name, {})
    
    def get_processing_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of agent processing"""
        timeline = []
        for agent_name, trace in self._agent_trace.items():
            if trace.get("start_time") and trace.get("end_time"):
                duration = (trace["end_time"] - trace["start_time"]).total_seconds()
                timeline.append({
                    "agent": agent_name,
                    "start": trace["start_time"].isoformat(),
                    "end": trace["end_time"].isoformat(),
                    "duration": duration,
                    "fields_extracted": trace.get("fields_extracted", []),
                    "errors": trace.get("errors", [])
                })
        return sorted(timeline, key=lambda x: x["start"])
    
    def save_to_document(self):
        """Save provenance data to the current document"""
        if self._current_document:
            self._current_document.provenance_graph = self._field_provenance
            self._current_document.agent_trace = self._agent_trace
    
    def generate_provenance_report(self) -> Dict[str, Any]:
        """Generate a provenance report for the current document"""
        if not self._current_document:
            return {"error": "No document currently being tracked"}
        
        report = {
            "document_id": self._current_document.document_id,
            "total_fields": len(self._field_provenance),
            "agent_timeline": self.get_processing_timeline(),
            "field_provenance_summary": {},
            "cross_modal_analysis": self._analyze_cross_modality()
        }
        
        # Summarize field provenance
        for field_name, records in self._field_provenance.items():
            report["field_provenance_summary"][field_name] = {
                "total_sources": len(records),
                "modalities": list(set(r.source_modality for r in records)),
                "agents": list(set(r.agent_name for r in records)),
                "highest_confidence": max((r.confidence for r in records), default=0.0)
            }
        
        return report
    
    def _analyze_cross_modality(self) -> Dict[str, Any]:
        """Analyze cross-modality provenance"""
        modality_counts = {}
        agent_modality_counts = {}
        
        for field_name, records in self._field_provenance.items():
            for record in records:
                # Count modalities
                modality = record.source_modality
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
                
                # Count agent-modality combinations
                key = f"{record.agent_name}_{modality}"
                agent_modality_counts[key] = agent_modality_counts.get(key, 0) + 1
        
        return {
            "modality_distribution": modality_counts,
            "agent_modality_distribution": agent_modality_counts,
            "fields_with_multiple_modalities": sum(
                1 for records in self._field_provenance.values() 
                if len(set(r.source_modality for r in records)) > 1
            )
        }