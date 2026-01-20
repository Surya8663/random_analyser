# app/agents/fusion_agent.py - UPDATED FOR MULTIMODALDOCUMENT
from typing import Dict, Any, List
from app.core.models import MultiModalDocument
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FusionAgent:
    """Fusion Agent for aligning text and visual information - UPDATED for MultiModalDocument"""
    
    def __init__(self):
        self._accepts_multi_modal = True
    
    async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
        """Align text and visual information"""
        try:
            logger.info("🔄 Running Fusion Agent (Multi-Modal)")
            
            # Align OCR text with layout regions
            text_layout_alignment = self._align_text_with_layout(doc)
            
            # Link visual elements to text content
            visual_text_alignment = self._align_visual_with_text(doc)
            
            # Cross-reference entities with visual elements
            entity_visual_alignment = self._align_entities_with_visual(doc)
            
            # Calculate overall alignment confidence
            alignment_confidence = self._calculate_alignment_confidence(
                text_layout_alignment, visual_text_alignment, entity_visual_alignment
            )
            
            # Store results
            doc.aligned_data = {
                "text_layout_alignment": text_layout_alignment,
                "visual_text_alignment": visual_text_alignment,
                "entity_visual_alignment": entity_visual_alignment,
                "alignment_confidence": alignment_confidence,
                "matches_found": len(text_layout_alignment) + len(visual_text_alignment) + len(entity_visual_alignment)
            }
            
            logger.info(f"✅ Fusion completed: {doc.aligned_data['matches_found']} matches found")
            
            return doc
            
        except Exception as e:
            logger.error(f"❌ Fusion failed: {e}")
            doc.aligned_data = {"error": str(e)}
            return doc
    
    def _align_text_with_layout(self, doc: MultiModalDocument) -> List[Dict[str, Any]]:
        """Align OCR text with layout regions"""
        alignments = []
        
        if not doc.ocr_results or not doc.layout_regions:
            return alignments
        
        try:
            # For each layout region, find OCR text within it
            for region in doc.layout_regions:
                page_num = region.page_num
                
                if page_num in doc.ocr_results:
                    ocr_result = doc.ocr_results[page_num]
                    
                    # Extract text from OCR that falls within this region
                    region_text = self._extract_text_from_region(ocr_result, region.bbox)
                    
                    if region_text and len(region_text.strip()) > 10:  # Minimum text length
                        alignment = {
                            "layout_region": region.label,
                            "page": page_num,
                            "text_snippet": region_text[:200],  # First 200 chars
                            "text_length": len(region_text),
                            "confidence": region.confidence * 0.8  # Weighted confidence
                        }
                        alignments.append(alignment)
            
        except Exception as e:
            logger.warning(f"Text-layout alignment failed: {e}")
        
        return alignments
    
    def _extract_text_from_region(self, ocr_result, bbox) -> str:
        """Extract text from OCR result within a bounding box"""
        try:
            # Simple implementation: check if OCR result has word-level data
            if hasattr(ocr_result, 'words') and ocr_result.words:
                # Filter words within the bbox
                region_words = []
                for word in ocr_result.words:
                    word_bbox = word.bbox
                    if (word_bbox.x1 >= bbox.x1 and word_bbox.x2 <= bbox.x2 and
                        word_bbox.y1 >= bbox.y1 and word_bbox.y2 <= bbox.y2):
                        region_words.append(word.text)
                
                return " ".join(region_words)
            else:
                # Fallback: return all text if no word-level data
                return ocr_result.text if hasattr(ocr_result, 'text') else ""
                
        except Exception as e:
            logger.warning(f"Text extraction from region failed: {e}")
            return ""
    
    def _align_visual_with_text(self, doc: MultiModalDocument) -> List[Dict[str, Any]]:
        """Link visual elements to text content"""
        alignments = []
        
        if not doc.visual_elements or not doc.raw_text:
            return alignments
        
        try:
            text_lower = doc.raw_text.lower()
            
            for element in doc.visual_elements:
                elem_type = element.element_type.lower()
                
                # Check if element type is mentioned in text
                if elem_type in text_lower:
                    # Find context around the mention
                    mention_context = self._find_mention_context(text_lower, elem_type)
                    
                    alignment = {
                        "element_type": element.element_type,
                        "page": element.page_num,
                        "mentioned_in_text": True,
                        "context": mention_context[:100],  # First 100 chars
                        "confidence": element.confidence * 0.9,
                        "bbox": element.bbox.to_list()
                    }
                    alignments.append(alignment)
                else:
                    # Check for related keywords
                    related_keywords = self._get_related_keywords(element.element_type)
                    for keyword in related_keywords:
                        if keyword in text_lower:
                            alignment = {
                                "element_type": element.element_type,
                                "page": element.page_num,
                                "mentioned_in_text": True,
                                "related_keyword": keyword,
                                "confidence": element.confidence * 0.7,
                                "bbox": element.bbox.to_list()
                            }
                            alignments.append(alignment)
                            break
        
        except Exception as e:
            logger.warning(f"Visual-text alignment failed: {e}")
        
        return alignments
    
    def _find_mention_context(self, text: str, keyword: str, context_chars: int = 50) -> str:
        """Find context around a keyword mention"""
        try:
            index = text.find(keyword)
            if index == -1:
                return ""
            
            start = max(0, index - context_chars)
            end = min(len(text), index + len(keyword) + context_chars)
            return text[start:end]
        except:
            return ""
    
    def _get_related_keywords(self, element_type: str) -> List[str]:
        """Get related keywords for an element type"""
        keyword_map = {
            "table": ["table", "tabular", "grid", "data", "rows", "columns"],
            "chart": ["chart", "graph", "plot", "diagram", "figure"],
            "signature": ["signature", "signed", "sign", "authorized", "approved"],
            "logo": ["logo", "brand", "company", "organization", "emblem"],
            "stamp": ["stamp", "seal", "official", "certified", "verified"]
        }
        return keyword_map.get(element_type, [])
    
    def _align_entities_with_visual(self, doc: MultiModalDocument) -> List[Dict[str, Any]]:
        """Align extracted entities with visual elements"""
        alignments = []
        
        if not hasattr(doc, 'extracted_entities') or not doc.extracted_entities:
            return alignments
        
        try:
            # For each entity type, check if mentioned near visual elements
            for entity_type, entities in doc.extracted_entities.items():
                for entity in entities[:10]:  # Limit to first 10 entities per type
                    entity_lower = entity.lower()
                    
                    # Check if entity is mentioned in text
                    if doc.raw_text and entity_lower in doc.raw_text.lower():
                        # Check if any visual elements are on the same page
                        page_elements = [e for e in doc.visual_elements if e.page_num == 0]  # Default to page 0
                        
                        if page_elements:
                            alignment = {
                                "entity_type": entity_type,
                                "entity_value": entity,
                                "visual_elements_on_page": len(page_elements),
                                "element_types": list(set([e.element_type for e in page_elements])),
                                "confidence": 0.6
                            }
                            alignments.append(alignment)
        
        except Exception as e:
            logger.warning(f"Entity-visual alignment failed: {e}")
        
        return alignments
    
    def _calculate_alignment_confidence(self, *alignment_lists) -> float:
        """Calculate overall alignment confidence"""
        total_matches = sum(len(lst) for lst in alignment_lists)
        
        if total_matches == 0:
            return 0.3  # Low confidence if no matches
        
        # Calculate confidence from alignments
        confidences = []
        for alignment_list in alignment_lists:
            for alignment in alignment_list:
                if "confidence" in alignment:
                    confidences.append(alignment["confidence"])
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            
            # Adjust based on number of matches
            match_factor = min(total_matches / 10, 1.0)  # Cap at 10 matches
            
            return (avg_confidence * 0.7 + match_factor * 0.3)
        
        return 0.5  # Default confidence