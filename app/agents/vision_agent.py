# app/agents/vision_agent.py - CORRECTED
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.core.models import MultiModalDocument, EnhancedVisualElement, LayoutRegion
from app.utils.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)

class VisionAgent(BaseAgent):
    """Real Vision Agent for semantic analysis of visual elements"""
    
    def __init__(self):
        super().__init__()
        self._accepts_multi_modal = True
    
    async def process(self, document: MultiModalDocument) -> MultiModalDocument:
        """Perform semantic analysis on visual elements"""
        try:
            logger.info("🔍 Running Vision Agent (Semantic Analysis)")
            
            # Record agent start
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_start(self.get_name())
            
            fields_extracted = []
            
            # Analyze each visual element semantically
            for element in document.visual_elements:
                # Ensure metadata exists
                if not hasattr(element, 'metadata') or element.metadata is None:
                    element.metadata = {}
                
                # Classify element
                semantic_label = self._classify_element_semantically(element)
                importance_score = self._calculate_importance_score(element)
                
                element.metadata["semantic_label"] = semantic_label
                element.metadata["importance_score"] = importance_score
                
                # Record provenance for visual element classification
                if element.element_type in ["signature", "table", "logo"]:
                    field_name = f"visual_{element.element_type}_{len(fields_extracted)}"
                    self._record_provenance(
                        field_name=field_name,
                        extraction_method="semantic_classification",
                        source_modality="visual",
                        confidence=element.confidence * importance_score,
                        source_bbox=element.bbox,
                        source_page=element.page_num,
                        reasoning_notes=f"Classified as {semantic_label} with importance {importance_score:.2f}"
                    )
                    fields_extracted.append(field_name)
            
            # Analyze layout regions
            for region in document.layout_regions:
                # Ensure metadata exists
                if not hasattr(region, 'metadata') or region.metadata is None:
                    region.metadata = {}
                
                region.metadata["structural_role"] = self._determine_structural_role(region)
            
            # Calculate visual statistics
            visual_stats = self._calculate_comprehensive_stats(document)
            if not hasattr(document, 'processing_metadata'):
                document.processing_metadata = {}
            document.processing_metadata["vision_analysis"] = visual_stats
            
            # Record agent end
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_end(
                    agent_name=self.get_name(),
                    fields_extracted=fields_extracted
                )
            
            logger.info(f"✅ Vision analysis completed: {len(document.visual_elements)} elements classified")
            return document
            
        except Exception as e:
            logger.error(f"❌ Vision analysis failed: {e}")
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_end(
                    agent_name=self.get_name(),
                    fields_extracted=[],
                    errors=[str(e)]
                )
            
            if not hasattr(document, 'errors'):
                document.errors = []
            document.errors.append(f"Vision agent error: {str(e)}")
            return document
    
    def _classify_element_semantically(self, element: EnhancedVisualElement) -> str:
        """Classify visual element based on type, position, and context"""
        element_type = element.element_type
        
        # Semantic mapping based on element type and position
        if element_type == "table":
            # Determine table type based on position
            if element.bbox.y1 < 0.3:
                return "header_table"
            elif element.bbox.y1 > 0.7:
                return "summary_table"
            else:
                return "data_table"
        
        elif element_type == "signature":
            return "authority_marker"
        
        elif element_type in ["logo", "stamp"]:
            return "branding_authority"
        
        elif element_type == "text_block":
            # Analyze text block position and size
            if element.bbox.x1 < 0.3:
                return "sidebar_content"
            elif (element.bbox.x2 - element.bbox.x1) > 0.6:
                return "main_content"
            else:
                return "supporting_content"
        
        else:
            return "misc_element"
    
    def _calculate_importance_score(self, element: EnhancedVisualElement) -> float:
        """Calculate importance score (0-1) for visual element"""
        base_score = element.confidence
        
        # Position-based importance (center is more important)
        center_x = (element.bbox.x1 + element.bbox.x2) / 2
        center_y = (element.bbox.y1 + element.bbox.y2) / 2
        
        # Distance from center (normalized)
        distance_from_center = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2) / 0.707
        position_score = 1.0 - distance_from_center
        
        # Size-based importance
        area = (element.bbox.x2 - element.bbox.x1) * (element.bbox.y2 - element.bbox.y1)
        size_score = min(area * 10, 1.0)  # Normalize
        
        # Type-based weight
        type_weights = {
            "signature": 1.2,
            "table": 1.1,
            "logo": 1.0,
            "stamp": 0.9,
            "text_block": 0.8,
            "object": 0.5
        }
        type_weight = type_weights.get(element.element_type, 0.5)
        
        # Combined score
        importance = (base_score * 0.4 + position_score * 0.3 + size_score * 0.3) * type_weight
        return min(importance, 1.0)
    
    def _determine_structural_role(self, region: LayoutRegion) -> str:
        """Determine structural role of layout region"""
        label = region.label.lower()
        
        if "header" in label:
            return "document_header"
        elif "footer" in label:
            return "document_footer"
        elif "title" in label:
            return "title_section"
        elif "table" in label:
            return "tabular_data"
        elif "text" in label or "paragraph" in label:
            return "text_content"
        elif "signature" in label:
            return "signature_area"
        else:
            return "structural_region"
    
    def _calculate_comprehensive_stats(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Calculate comprehensive visual statistics"""
        if not document.visual_elements:
            return {"error": "No visual elements found"}
        
        # Element type distribution
        type_distribution = {}
        for element in document.visual_elements:
            elem_type = element.element_type
            type_distribution[elem_type] = type_distribution.get(elem_type, 0) + 1
        
        # Confidence statistics
        confidences = [e.confidence for e in document.visual_elements]
        
        # Position analysis
        horizontal_distribution = {"left": 0, "center": 0, "right": 0}
        vertical_distribution = {"top": 0, "middle": 0, "bottom": 0}
        
        for element in document.visual_elements:
            center_x = (element.bbox.x1 + element.bbox.x2) / 2
            center_y = (element.bbox.y1 + element.bbox.y2) / 2
            
            if center_x < 0.33:
                horizontal_distribution["left"] += 1
            elif center_x < 0.66:
                horizontal_distribution["center"] += 1
            else:
                horizontal_distribution["right"] += 1
            
            if center_y < 0.33:
                vertical_distribution["top"] += 1
            elif center_y < 0.66:
                vertical_distribution["middle"] += 1
            else:
                vertical_distribution["bottom"] += 1
        
        # Size statistics
        areas = [(e.bbox.x2 - e.bbox.x1) * (e.bbox.y2 - e.bbox.y1) for e in document.visual_elements]
        
        return {
            "total_elements": len(document.visual_elements),
            "type_distribution": type_distribution,
            "confidence_stats": {
                "mean": float(np.mean(confidences)) if confidences else 0,
                "std": float(np.std(confidences)) if confidences else 0,
                "min": float(min(confidences)) if confidences else 0,
                "max": float(max(confidences)) if confidences else 0
            },
            "position_analysis": {
                "horizontal": horizontal_distribution,
                "vertical": vertical_distribution
            },
            "size_analysis": {
                "mean_area": float(np.mean(areas)) if areas else 0,
                "total_coverage": float(sum(areas)) if areas else 0
            },
            "layout_regions_count": len(document.layout_regions)
        }