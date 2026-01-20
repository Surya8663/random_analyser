# app/agents/vision_agent.py - UPDATED FOR MULTIMODALDOCUMENT
from typing import Dict, Any, List
from app.core.models import MultiModalDocument
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VisionAgent:
    """Vision Agent for analyzing visual elements - UPDATED for MultiModalDocument"""
    
    def __init__(self):
        # Mark that this agent accepts MultiModalDocument
        self._accepts_multi_modal = True
    
    async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
        """Analyze visual elements in document"""
        try:
            logger.info("👁️ Running Vision Agent (Multi-Modal)")
            
            if not hasattr(doc, 'agent_outputs'):
                doc.agent_outputs = {}
            
            # Analyze layout regions
            layout_analysis = self._analyze_layout(doc.layout_regions)
            
            # Analyze visual elements
            element_analysis = self._analyze_visual_elements(doc.visual_elements)
            
            # Calculate visual statistics
            visual_stats = self._calculate_visual_statistics(doc)
            
            # Store results
            doc.agent_outputs["vision"] = {
                "status": "completed",
                "layout_analysis": layout_analysis,
                "element_analysis": element_analysis,
                "visual_statistics": visual_stats,
                "timestamp": "now"
            }
            
            logger.info(f"✅ Vision analysis: {len(doc.visual_elements)} elements, {len(doc.layout_regions)} layout regions")
            
            return doc
            
        except Exception as e:
            logger.error(f"❌ Vision analysis failed: {e}")
            if not hasattr(doc, 'agent_outputs'):
                doc.agent_outputs = {}
            doc.agent_outputs["vision"] = {"error": str(e)}
            return doc
    
    def _analyze_layout(self, layout_regions: List) -> Dict[str, Any]:
        """Analyze document layout"""
        analysis = {
            "total_regions": len(layout_regions),
            "region_types": {},
            "page_distribution": {},
            "average_confidence": 0.0
        }
        
        if not layout_regions:
            return analysis
        
        # Count regions by type
        for region in layout_regions:
            region_type = region.label
            analysis["region_types"][region_type] = analysis["region_types"].get(region_type, 0) + 1
            
            # Count by page
            page_num = region.page_num
            analysis["page_distribution"][page_num] = analysis["page_distribution"].get(page_num, 0) + 1
        
        # Calculate average confidence
        confidences = [r.confidence for r in layout_regions]
        analysis["average_confidence"] = sum(confidences) / len(confidences)
        
        return analysis
    
    def _analyze_visual_elements(self, visual_elements: List) -> Dict[str, Any]:
        """Analyze visual elements"""
        analysis = {
            "total_elements": len(visual_elements),
            "element_types": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "page_coverage": {}
        }
        
        if not visual_elements:
            return analysis
        
        # Count elements by type
        for element in visual_elements:
            elem_type = element.element_type
            analysis["element_types"][elem_type] = analysis["element_types"].get(elem_type, 0) + 1
            
            # Categorize confidence
            conf = element.confidence
            if conf > 0.7:
                analysis["confidence_distribution"]["high"] += 1
            elif conf > 0.4:
                analysis["confidence_distribution"]["medium"] += 1
            else:
                analysis["confidence_distribution"]["low"] += 1
            
            # Track page coverage
            page_num = element.page_num
            if page_num not in analysis["page_coverage"]:
                analysis["page_coverage"][page_num] = {"elements": 0, "types": set()}
            analysis["page_coverage"][page_num]["elements"] += 1
            analysis["page_coverage"][page_num]["types"].add(elem_type)
        
        # Convert sets to lists for JSON serialization
        for page in analysis["page_coverage"]:
            analysis["page_coverage"][page]["types"] = list(analysis["page_coverage"][page]["types"])
        
        return analysis
    
    def _calculate_visual_statistics(self, doc: MultiModalDocument) -> Dict[str, Any]:
        """Calculate comprehensive visual statistics"""
        stats = {
            "document_complexity": 0.0,
            "visual_balance": 0.0,
            "element_density": 0.0,
            "layout_coherence": 0.0
        }
        
        # Calculate document complexity
        total_elements = len(doc.visual_elements) + len(doc.layout_regions)
        total_pages = len(doc.images) if hasattr(doc, 'images') and doc.images else 1
        
        if total_pages > 0:
            stats["element_density"] = total_elements / total_pages
            
            # Simple complexity metric
            unique_element_types = len(set([e.element_type for e in doc.visual_elements]))
            unique_layout_types = len(set([r.label for r in doc.layout_regions]))
            stats["document_complexity"] = (unique_element_types + unique_layout_types) / 10
        
        # Calculate visual balance (simple heuristic)
        if doc.visual_elements:
            # Check if elements are distributed across pages
            pages_with_elements = len(set([e.page_num for e in doc.visual_elements]))
            stats["visual_balance"] = pages_with_elements / total_pages if total_pages > 0 else 0
        
        # Calculate layout coherence
        if doc.layout_regions:
            # Check if similar layout regions appear on multiple pages
            layout_types_by_page = {}
            for region in doc.layout_regions:
                page = region.page_num
                if page not in layout_types_by_page:
                    layout_types_by_page[page] = set()
                layout_types_by_page[page].add(region.label)
            
            # Calculate similarity between pages
            if len(layout_types_by_page) > 1:
                common_types = set.intersection(*layout_types_by_page.values())
                stats["layout_coherence"] = len(common_types) / len(set([r.label for r in doc.layout_regions]))
        
        return stats