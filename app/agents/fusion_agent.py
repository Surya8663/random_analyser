# app/agents/fusion_agent.py - CORRECTED
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from app.agents.base_agent import BaseAgent
from app.core.models import MultiModalDocument, EnhancedVisualElement, LayoutRegion, OCRResult, OCRWord, BoundingBox
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FusionAgent(BaseAgent):
    """Real Fusion Agent for cross-modal alignment and integration - NO MOCK"""
    
    def __init__(self):
        super().__init__()
        self._accepts_multi_modal = True
    
    async def process(self, document: MultiModalDocument) -> MultiModalDocument:
        """Fuse visual and text information into coherent understanding - REAL IMPLEMENTATION"""
        try:
            logger.info("🔄 Running Fusion Agent (Cross-Modal Alignment)")
            
            # Record agent start
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_start(self.get_name())
            
            fields_extracted = []
            
            # Step 1: Align text with visual regions - REAL ALGORITHM
            text_visual_alignment = self._real_align_text_with_visual_regions(document)
            
            # Step 2: Create structured fields from OCR and visual data - REAL ALGORITHM
            extracted_fields = self._real_extract_structured_fields(document)
            
            # Step 3: Calculate cross-modal consistency - REAL METRICS
            consistency_metrics = self._real_calculate_consistency(document, text_visual_alignment)
            
            # Store all fusion results
            document.aligned_data = {
                "text_visual_alignment": text_visual_alignment,
                "consistency_metrics": consistency_metrics,
                "fusion_confidence": consistency_metrics.get("overall_consistency", 0.5)
            }
            
            # Record agent end
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_end(
                    agent_name=self.get_name(),
                    fields_extracted=fields_extracted
                )
            
            logger.info(f"✅ Fusion completed: {len(text_visual_alignment)} alignments, {len(extracted_fields)} fields")
            return document
            
        except Exception as e:
            logger.error(f"❌ Fusion failed: {e}")
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_end(
                    agent_name=self.get_name(),
                    fields_extracted=[],
                    errors=[str(e)]
                )
            
            document.errors.append(f"Fusion agent error: {str(e)}")
            return document
    
    def _real_align_text_with_visual_regions(self, document: MultiModalDocument) -> List[Dict[str, Any]]:
        """REAL alignment algorithm using spatial relationships"""
        alignments = []
        
        if not document.ocr_results or not document.visual_elements:
            return alignments
        
        try:
            # For each page with OCR results
            for page_num, ocr_result in document.ocr_results.items():
                # Get visual elements on this page
                page_elements = [
                    elem for elem in document.visual_elements 
                    if elem.page_num == page_num
                ]
                
                if not page_elements or not ocr_result.words:
                    continue
                
                # REAL ALGORITHM: Find which visual elements contain text
                for element in page_elements:
                    contained_words = []
                    total_confidence = 0
                    
                    # Check each OCR word against the element's bounding box
                    for word in ocr_result.words:
                        if self._is_word_in_element(word, element):
                            contained_words.append(word.text)
                            total_confidence += word.confidence
                    
                    if contained_words:
                        avg_confidence = total_confidence / len(contained_words) if contained_words else 0
                        alignment = {
                            "page": page_num,
                            "element_type": element.element_type,
                            "element_bbox": element.bbox.to_list(),
                            "contained_words_count": len(contained_words),
                            "sample_words": contained_words[:5],
                            "average_confidence": avg_confidence,
                            "alignment_confidence": min(element.confidence * avg_confidence, 1.0)
                        }
                        alignments.append(alignment)
        
        except Exception as e:
            logger.warning(f"Text-visual alignment error: {e}")
        
        return alignments
    
    def _is_word_in_element(self, word: OCRWord, element: EnhancedVisualElement) -> bool:
        """Check if a word is inside a visual element's bounding box"""
        try:
            word_center_x = (word.bbox.x1 + word.bbox.x2) / 2
            word_center_y = (word.bbox.y1 + word.bbox.y2) / 2
            
            return (
                element.bbox.x1 <= word_center_x <= element.bbox.x2 and
                element.bbox.y1 <= word_center_y <= element.bbox.y2
            )
        except:
            return False
    
    def _real_extract_structured_fields(self, document: MultiModalDocument) -> Dict[str, Dict[str, Any]]:
        """REAL field extraction from OCR text and visual elements"""
        fields = {}
        
        # Field 1: Document metadata
        metadata_provenance = self._record_provenance(
            field_name="document_id",
            extraction_method="system_generated",
            source_modality="metadata",
            confidence=1.0,
            reasoning_notes="System-generated document ID"
        )
        
        if metadata_provenance:
            document.add_field_with_provenance(
                field_name="document_id",
                field_type="text",
                value=document.document_id,
                confidence=1.0,
                provenance=metadata_provenance,
                modality_sources=["metadata"]
            )
            fields["document_id"] = document.extracted_fields["document_id"].dict()
        
        # Extract from OCR
        ocr_fields = self._extract_fields_from_ocr(document)
        for field_name, field_data in ocr_fields.items():
            provenance = self._record_provenance(
                field_name=field_name,
                extraction_method="ocr_pattern",
                source_modality="text",
                confidence=field_data.get("confidence", 0.5),
                reasoning_notes=f"Extracted from OCR text using pattern: {field_name}"
            )
            
            if provenance:
                document.add_field_with_provenance(
                    field_name=field_name,
                    field_type=self._infer_field_type(field_name),
                    value=field_data.get("value"),
                    confidence=field_data.get("confidence", 0.5),
                    provenance=provenance,
                    modality_sources=["text"]
                )
                fields[field_name] = document.extracted_fields[field_name].dict()
        
        # Extract from visual
        visual_fields = self._extract_fields_from_visual(document)
        for field_name, field_data in visual_fields.items():
            provenance = self._record_provenance(
                field_name=field_name,
                extraction_method="visual_detection",
                source_modality="visual",
                confidence=field_data.get("confidence", 0.5),
                reasoning_notes=f"Detected from visual analysis: {field_name}"
            )
            
            if provenance:
                if field_name in document.extracted_fields:
                    document.merge_field_provenance(
                        field_name=field_name,
                        new_provenance=provenance,
                        new_value=field_data.get("value"),
                        new_confidence=field_data.get("confidence", 0.5)
                    )
                else:
                    document.add_field_with_provenance(
                        field_name=field_name,
                        field_type=self._infer_field_type(field_name),
                        value=field_data.get("value"),
                        confidence=field_data.get("confidence", 0.5),
                        provenance=provenance,
                        modality_sources=["visual"]
                    )
                fields[field_name] = document.extracted_fields[field_name].dict()
        
        return fields
    
    def _infer_field_type(self, field_name: str) -> str:
        """Infer field type from field name"""
        if "date" in field_name.lower():
            return "date"
        elif any(word in field_name.lower() for word in ["amount", "total", "price", "cost"]):
            return "number"
        elif "signature" in field_name.lower():
            return "signature"
        elif any(word in field_name.lower() for word in ["count", "number_of"]):
            return "integer"
        else:
            return "text"
    
    def _extract_fields_from_ocr(self, document: MultiModalDocument) -> Dict[str, Dict[str, Any]]:
        """Extract fields from OCR text using pattern matching"""
        fields = {}
        
        if not document.ocr_results:
            return fields
        
        try:
            # Combine all OCR text
            all_text = ""
            for ocr_result in document.ocr_results.values():
                all_text += ocr_result.text + "\n"
            
            # REAL PATTERN MATCHING - no mock data
            patterns = [
                (r'Invoice\s*(?:No|Number|#)?\s*[:#]?\s*([A-Z0-9\-]+)', "invoice_number", 0.7),
                (r'Date\s*[:]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})', "date", 0.8),
                (r'Total\s*(?:Amount|Amt)?\s*[:$]?\s*(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', "total_amount", 0.6),
                (r'Amount\s*[:$]?\s*(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', "amount", 0.6),
                (r'Name\s*[:]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', "name", 0.5),
            ]
            
            import re
            for pattern, field_name, confidence in patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                if matches:
                    # Take the first match
                    value = matches[0] if isinstance(matches[0], str) else matches[0][0] if matches[0] else ""
                    if value:
                        fields[field_name] = {
                            "value": value.strip(),
                            "confidence": confidence,
                            "source": "ocr_pattern",
                            "modalities": ["text"]
                        }
        
        except Exception as e:
            logger.warning(f"OCR field extraction error: {e}")
        
        return fields
    
    def _extract_fields_from_visual(self, document: MultiModalDocument) -> Dict[str, Dict[str, Any]]:
        """Extract fields from visual element analysis"""
        fields = {}
        
        if not document.visual_elements:
            return fields
        
        try:
            # REAL VISUAL ANALYSIS - count elements by type
            element_counts = {}
            for element in document.visual_elements:
                elem_type = element.element_type
                element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
            
            # Create fields from visual element counts
            for elem_type, count in element_counts.items():
                fields[f"visual_{elem_type}_count"] = {
                    "value": count,
                    "confidence": 0.9,
                    "source": "visual_analysis",
                    "modalities": ["visual"]
                }
            
            # Check for specific important elements
            has_signature = any(elem.element_type == "signature" for elem in document.visual_elements)
            if has_signature:
                fields["signature_present"] = {
                    "value": True,
                    "confidence": 0.8,
                    "source": "visual_detection",
                    "modalities": ["visual"]
                }
            
            has_table = any(elem.element_type == "table" for elem in document.visual_elements)
            if has_table:
                fields["table_present"] = {
                    "value": True,
                    "confidence": 0.7,
                    "source": "visual_detection",
                    "modalities": ["visual"]
                }
        
        except Exception as e:
            logger.warning(f"Visual field extraction error: {e}")
        
        return fields
    
    def _extract_fields_from_entities(self, document: MultiModalDocument) -> Dict[str, Dict[str, Any]]:
        """Extract fields from entity extraction results"""
        fields = {}
        
        try:
            # Convert entity lists to structured fields
            entity_mapping = {
                "dates": "extracted_date",
                "amounts": "extracted_amount",
                "names": "extracted_name",
                "identifiers": "extracted_id"
            }
            
            for entity_type, field_name in entity_mapping.items():
                if entity_type in document.extracted_entities:
                    entities = document.extracted_entities[entity_type]
                    if entities:
                        # Take the first entity
                        if isinstance(entities[0], dict):
                            value = entities[0].get("value", str(entities[0]))
                        else:
                            value = str(entities[0])
                        
                        fields[field_name] = {
                            "value": value,
                            "confidence": 0.6,
                            "source": "entity_extraction",
                            "modalities": ["text"]
                        }
        
        except Exception as e:
            logger.warning(f"Entity field extraction error: {e}")
        
        return fields
    
    def _real_calculate_consistency(self, document: MultiModalDocument, alignments: List[Dict]) -> Dict[str, Any]:
        """Calculate REAL consistency metrics between modalities"""
        metrics = {
            "text_present": False,
            "visual_present": False,
            "alignment_ratio": 0.0,
            "overall_consistency": 0.0
        }
        
        try:
            # Check if text is present
            metrics["text_present"] = bool(document.ocr_results and any(
                len(ocr.words) > 0 for ocr in document.ocr_results.values()
            ))
            
            # Check if visual elements are present
            metrics["visual_present"] = bool(document.visual_elements)
            
            # Calculate alignment ratio
            if metrics["text_present"] and metrics["visual_present"]:
                total_visual_elements = len(document.visual_elements)
                aligned_elements = len(alignments)
                
                if total_visual_elements > 0:
                    metrics["alignment_ratio"] = aligned_elements / total_visual_elements
            
            # Calculate overall consistency
            consistency_factors = []
            
            if metrics["text_present"]:
                consistency_factors.append(0.3)
            
            if metrics["visual_present"]:
                consistency_factors.append(0.3)
            
            if metrics["alignment_ratio"] > 0:
                consistency_factors.append(metrics["alignment_ratio"] * 0.4)
            
            if consistency_factors:
                metrics["overall_consistency"] = sum(consistency_factors) / len(consistency_factors)
            else:
                metrics["overall_consistency"] = 0.5  # Default
            
        except Exception as e:
            logger.warning(f"Consistency calculation error: {e}")
        
        return metrics