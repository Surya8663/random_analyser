# app/explain/explainability.py
import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from app.core.models import MultiModalDocument, ExplainableField
from app.explain.provenance import ProvenanceTracker
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ExplainabilityGenerator:
    """Generate explainability reports for document processing"""
    
    def __init__(self, output_dir: str = "explainability_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.provenance_tracker = ProvenanceTracker()
    
    def attach_to_document(self, document: MultiModalDocument):
        """Attach explainability tracking to a document"""
        self.provenance_tracker.start_document(document)
    
    def generate_explainability_report(self, 
                                     document: MultiModalDocument) -> Dict[str, Any]:
        """Generate comprehensive explainability report"""
        try:
            logger.info(f"💡 Generating explainability report for {document.document_id}")
            
            # Save provenance to document
            self.provenance_tracker.save_to_document()
            
            # Generate report
            provenance_report = self.provenance_tracker.generate_provenance_report()
            
            # Create complete explainability report
            report = {
                "document_id": document.document_id,
                "generated_at": datetime.now().isoformat(),
                "processing_summary": self._generate_processing_summary(document),
                "field_explanations": self._generate_field_explanations(document),
                "agent_contributions": self._generate_agent_contributions(document),
                "modality_analysis": self._analyze_modalities(document),
                "decision_points": self._identify_decision_points(document),
                "confidence_analysis": self._analyze_confidences(document),
                "provenance_timeline": provenance_report.get("agent_timeline", []),
                "cross_modality_insights": provenance_report.get("cross_modal_analysis", {})
            }
            
            # Save to file
            self._save_report(report, document.document_id)
            
            logger.info(f"✅ Explainability report generated for {document.document_id}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Explainability report generation failed: {e}")
            return {
                "document_id": document.document_id,
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def _generate_processing_summary(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Generate processing summary"""
        processing_time = document.get_processing_time()
        
        return {
            "total_pages": len(document.images) if hasattr(document, 'images') else 0,
            "text_length": len(document.raw_text) if hasattr(document, 'raw_text') else 0,
            "visual_elements": len(document.visual_elements) if hasattr(document, 'visual_elements') else 0,
            "extracted_fields": len(document.extracted_fields) if hasattr(document, 'extracted_fields') else 0,
            "processing_time_seconds": processing_time,
            "errors_count": len(document.errors) if hasattr(document, 'errors') else 0,
            "contradictions_count": len(document.contradictions) if hasattr(document, 'contradictions') else 0
        }
    
    def _generate_field_explanations(self, document: MultiModalDocument) -> Dict[str, Dict[str, Any]]:
        """Generate explanations for each extracted field"""
        field_explanations = {}
        
        if not hasattr(document, 'extracted_fields') or not document.extracted_fields:
            return field_explanations
        
        for field_name, field in document.extracted_fields.items():
            if isinstance(field, ExplainableField):
                primary_provenance = field.get_primary_provenance()
                
                explanation = {
                    "value": field.value,
                    "type": field.field_type,
                    "confidence": field.confidence,
                    "final_source": field.final_source,
                    "modality_sources": field.modality_sources,
                    "provenance_count": len(field.provenance)
                }
                
                if primary_provenance:
                    explanation.update({
                        "extraction_method": primary_provenance.extraction_method,
                        "source_modality": primary_provenance.source_modality,
                        "reasoning": primary_provenance.reasoning_notes,
                        "source_page": primary_provenance.source_page
                    })
                
                field_explanations[field_name] = explanation
        
        return field_explanations
    
    def _generate_agent_contributions(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Generate agent contribution analysis"""
        agent_contributions = {}
        
        if not hasattr(document, 'extracted_fields') or not document.extracted_fields:
            return agent_contributions
        
        # Count fields by agent
        for field in document.extracted_fields.values():
            if isinstance(field, ExplainableField):
                for provenance in field.provenance:
                    agent = provenance.agent_name
                    if agent not in agent_contributions:
                        agent_contributions[agent] = {
                            "field_count": 0,
                            "total_confidence": 0.0,
                            "modalities": set(),
                            "fields": []
                        }
                    
                    agent_contributions[agent]["field_count"] += 1
                    agent_contributions[agent]["total_confidence"] += provenance.confidence
                    agent_contributions[agent]["modalities"].add(provenance.source_modality)
                    agent_contributions[agent]["fields"].append(field.field_name)
        
        # Calculate averages and convert sets to lists
        for agent, data in agent_contributions.items():
            data["average_confidence"] = data["total_confidence"] / data["field_count"] if data["field_count"] > 0 else 0.0
            data["modalities"] = list(data["modalities"])
            data["unique_fields"] = list(set(data["fields"]))
            del data["total_confidence"]
            del data["fields"]
        
        return agent_contributions
    
    def _analyze_modalities(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Analyze modality contributions"""
        modality_analysis = {
            "text": {"field_count": 0, "average_confidence": 0.0},
            "visual": {"field_count": 0, "average_confidence": 0.0},
            "fusion": {"field_count": 0, "average_confidence": 0.0},
            "metadata": {"field_count": 0, "average_confidence": 0.0}
        }
        
        if not hasattr(document, 'extracted_fields') or not document.extracted_fields:
            return modality_analysis
        
        # Collect modality statistics
        modality_stats = {}
        for field in document.extracted_fields.values():
            if isinstance(field, ExplainableField):
                for provenance in field.provenance:
                    modality = provenance.source_modality
                    if modality not in modality_stats:
                        modality_stats[modality] = {"count": 0, "confidence_sum": 0.0}
                    
                    modality_stats[modality]["count"] += 1
                    modality_stats[modality]["confidence_sum"] += provenance.confidence
        
        # Update modality analysis
        for modality, stats in modality_stats.items():
            if modality in modality_analysis:
                modality_analysis[modality]["field_count"] = stats["count"]
                modality_analysis[modality]["average_confidence"] = stats["confidence_sum"] / stats["count"] if stats["count"] > 0 else 0.0
        
        return modality_analysis
    
    def _identify_decision_points(self, document: MultiModalDocument) -> List[Dict[str, Any]]:
        """Identify key decision points in processing"""
        decision_points = []
        
        # Check for multi-source fields
        if hasattr(document, 'extracted_fields') and document.extracted_fields:
            for field_name, field in document.extracted_fields.items():
                if isinstance(field, ExplainableField) and len(field.provenance) > 1:
                    # Field has multiple sources - this was a decision point
                    sources = [(p.agent_name, p.source_modality, p.confidence) 
                              for p in field.provenance]
                    
                    # Find which source was selected
                    selected_source = next((p for p in field.provenance 
                                          if p.agent_name == field.final_source), None)
                    
                    decision_points.append({
                        "field": field_name,
                        "decision_type": "source_selection",
                        "sources": sources,
                        "selected_source": selected_source.agent_name if selected_source else None,
                        "selection_criteria": "highest_confidence",
                        "confidence_delta": max(p.confidence for p in field.provenance) - 
                                          min(p.confidence for p in field.provenance) 
                                          if len(field.provenance) > 1 else 0.0
                    })
        
        # Check for contradictions
        if hasattr(document, 'contradictions') and document.contradictions:
            for contradiction in document.contradictions:
                decision_points.append({
                    "field": f"{contradiction.field_a} vs {contradiction.field_b}",
                    "decision_type": "contradiction_resolution",
                    "contradiction_type": contradiction.contradiction_type.value,
                    "severity": contradiction.severity.value,
                    "explanation": contradiction.explanation,
                    "resolution": contradiction.recommendation
                })
        
        return decision_points
    
    def _analyze_confidences(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Analyze confidence scores across the document"""
        confidences = []
        
        if hasattr(document, 'extracted_fields') and document.extracted_fields:
            for field in document.extracted_fields.values():
                if isinstance(field, ExplainableField):
                    confidences.append(field.confidence)
        
        if not confidences:
            return {
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "distribution": "no_data"
            }
        
        import numpy as np
        from collections import Counter
        
        confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        bin_counts = Counter()
        
        for conf in confidences:
            for i in range(len(confidence_bins) - 1):
                if confidence_bins[i] <= conf < confidence_bins[i + 1]:
                    bin_label = f"{confidence_bins[i]:.1f}-{confidence_bins[i + 1]:.1f}"
                    bin_counts[bin_label] += 1
                    break
        
        return {
            "average": float(np.mean(confidences)),
            "min": float(min(confidences)),
            "max": float(max(confidences)),
            "std": float(np.std(confidences)),
            "distribution": dict(bin_counts)
        }
    
    def _save_report(self, report: Dict[str, Any], document_id: str):
        """Save explainability report to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"explain_{document_id}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Explainability report saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save explainability report: {e}")