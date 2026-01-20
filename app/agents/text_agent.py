# app/agents/text_agent.py - CORRECTED
from typing import Dict, Any, List, Tuple
import re
from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.core.models import MultiModalDocument, OCRResult, OCRWord
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextAgent(BaseAgent):
    """Real Text Agent for OCR analysis and text understanding"""
    
    def __init__(self):
        super().__init__()
        self._accepts_multi_modal = True
        
        # Precompiled regex patterns for entity extraction
        self.date_patterns = [
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),  # MM/DD/YYYY or DD-MM-YYYY
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),    # YYYY/MM/DD
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', re.IGNORECASE),
        ]
        
        self.amount_patterns = [
            re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'),  # $1,000.00
            re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR)\b', re.IGNORECASE),
            re.compile(r'\b\d+(?:\.\d{2})?\b(?=\s*(?:dollars|euros|pounds|rupees))', re.IGNORECASE),
        ]
        
        self.id_patterns = [
            re.compile(r'\b(?:ID|No|#)[:\s]*([A-Z0-9\-]{5,20})\b', re.IGNORECASE),
            re.compile(r'\b[A-Z]{2,3}\d{5,10}\b'),  # ABC123456
            re.compile(r'\b\d{9,12}\b'),  # 9-12 digit IDs
        ]
        
        self.name_patterns = [
            re.compile(r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'),
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b'),
        ]
    
    async def process(self, document: MultiModalDocument) -> MultiModalDocument:
        """Perform text analysis and entity extraction"""
        try:
            logger.info("📝 Running Text Agent (Entity Extraction)")
            
            # Record agent start
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_start(self.get_name())
            
            fields_extracted = []
            
            if not document.ocr_results:
                logger.warning("No OCR results found for text analysis")
                return document
            
            # Step 1: Clean and structure OCR text
            structured_text = self._structure_ocr_text(document.ocr_results)
            document.processing_metadata["text_structure"] = structured_text
            
            # Step 2: Extract entities from text
            entities = self._extract_entities_from_ocr(document.ocr_results)
            document.extracted_entities = entities
            
            # Record provenance for extracted entities
            for entity_type, entity_list in entities.items():
                if entity_list:
                    for i, entity in enumerate(entity_list[:5]):  # Limit to first 5
                        field_name = f"{entity_type}_{i}"
                        
                        # Find source bbox for this entity
                        source_bbox = None
                        source_page = 0
                        
                        # Try to find the word in OCR results
                        if document.ocr_results:
                            for page_num, ocr_result in document.ocr_results.items():
                                for word in ocr_result.words:
                                    if isinstance(entity, str) and entity.lower() in word.text.lower():
                                        source_bbox = word.bbox
                                        source_page = page_num
                                        break
                        
                        self._record_provenance(
                            field_name=field_name,
                            extraction_method="regex_pattern",
                            source_modality="text",
                            confidence=0.7,  # Base confidence for regex extraction
                            source_bbox=source_bbox,
                            source_page=source_page,
                            reasoning_notes=f"Extracted {entity_type} using pattern matching"
                        )
                        fields_extracted.append(field_name)
            
            # Step 3: Segment text by semantic blocks
            text_segments = self._segment_text_by_content(document.ocr_results)
            document.processing_metadata["text_segments"] = text_segments
            
            # Step 4: Calculate text quality metrics
            quality_metrics = self._calculate_text_quality(document.ocr_results)
            document.processing_metadata["text_quality"] = quality_metrics
            
            # Step 5: Identify document sections
            sections = self._identify_document_sections(document.ocr_results)
            document.processing_metadata["document_sections"] = sections
            
            # Record agent end
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_end(
                    agent_name=self.get_name(),
                    fields_extracted=fields_extracted
                )
            
            logger.info(f"✅ Text analysis completed: {sum(len(v) for v in entities.values())} entities extracted")
            return document
            
        except Exception as e:
            logger.error(f"❌ Text analysis failed: {e}")
            if self._provenance_tracker:
                self._provenance_tracker.record_agent_end(
                    agent_name=self.get_name(),
                    fields_extracted=[],
                    errors=[str(e)]
                )
            
            document.errors.append(f"Text agent error: {str(e)}")
            return document
    
    def _structure_ocr_text(self, ocr_results: Dict[int, OCRResult]) -> Dict[str, Any]:
        """Structure OCR text into hierarchical format"""
        structure = {
            "pages": {},
            "total_pages": len(ocr_results),
            "total_words": 0,
            "total_characters": 0
        }
        
        for page_num, ocr_result in ocr_results.items():
            page_text = ocr_result.text
            words = ocr_result.words
            
            # Group words into lines
            lines = self._group_words_into_lines(words)
            
            # Identify paragraphs based on spacing
            paragraphs = self._identify_paragraphs(lines)
            
            # Calculate page statistics
            word_count = len(words)
            char_count = len(page_text)
            
            structure["pages"][page_num] = {
                "text": page_text,
                "word_count": word_count,
                "char_count": char_count,
                "average_confidence": ocr_result.average_confidence,
                "lines_count": len(lines),
                "paragraphs_count": len(paragraphs),
                "lines": [{"text": line_text, "word_count": len(line_words)} 
                         for line_text, line_words in lines],
                "paragraphs": paragraphs
            }
            
            structure["total_words"] += word_count
            structure["total_characters"] += char_count
        
        return structure
    
    def _group_words_into_lines(self, words: List[OCRWord]) -> List[Tuple[str, List[OCRWord]]]:
        """Group words into lines based on y-coordinate"""
        if not words:
            return []
        
        # Sort words by y-coordinate, then x-coordinate
        sorted_words = sorted(words, key=lambda w: (w.bbox.y1, w.bbox.x1))
        
        lines = []
        current_line = []
        current_y = sorted_words[0].bbox.y1
        y_threshold = 0.01  # 1% of image height
        
        for word in sorted_words:
            if abs(word.bbox.y1 - current_y) < y_threshold:
                current_line.append(word)
            else:
                if current_line:
                    # Sort words in line by x-coordinate
                    current_line.sort(key=lambda w: w.bbox.x1)
                    line_text = " ".join(w.text for w in current_line)
                    lines.append((line_text, current_line))
                
                current_line = [word]
                current_y = word.bbox.y1
        
        # Add last line
        if current_line:
            current_line.sort(key=lambda w: w.bbox.x1)
            line_text = " ".join(w.text for w in current_line)
            lines.append((line_text, current_line))
        
        return lines
    
    def _identify_paragraphs(self, lines: List[Tuple[str, List[OCRWord]]]) -> List[Dict[str, Any]]:
        """Identify paragraphs based on vertical spacing"""
        if len(lines) < 2:
            return [{"text": lines[0][0] if lines else "", "line_count": len(lines)}]
        
        paragraphs = []
        current_paragraph = []
        previous_bottom = None
        
        for line_text, line_words in lines:
            if not line_words:
                continue
            
            current_top = min(w.bbox.y1 for w in line_words)
            current_bottom = max(w.bbox.y2 for w in line_words)
            
            if previous_bottom is None:
                current_paragraph.append(line_text)
            else:
                # Check vertical gap
                vertical_gap = current_top - previous_bottom
                if vertical_gap > 0.02:  # 2% gap indicates new paragraph
                    if current_paragraph:
                        paragraphs.append({
                            "text": "\n".join(current_paragraph),
                            "line_count": len(current_paragraph)
                        })
                    current_paragraph = [line_text]
                else:
                    current_paragraph.append(line_text)
            
            previous_bottom = current_bottom
        
        # Add last paragraph
        if current_paragraph:
            paragraphs.append({
                "text": "\n".join(current_paragraph),
                "line_count": len(current_paragraph)
            })
        
        return paragraphs
    
    def _extract_entities_from_ocr(self, ocr_results: Dict[int, OCRResult]) -> Dict[str, List[str]]:
        """Extract structured entities from OCR text - SIMPLIFIED REAL VERSION"""
        entities = {
            "dates": [],
            "amounts": [],
            "identifiers": [],
            "names": [],
            "addresses": [],
            "email_phones": []
        }
        
        for page_num, ocr_result in ocr_results.items():
            page_text = ocr_result.text
            
            # Simple regex patterns for real extraction
            import re
            
            # Extract dates
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}'
            ]
            
            for pattern in date_patterns:
                for match in re.finditer(pattern, page_text, re.IGNORECASE):
                    entities["dates"].append(match.group())
            
            # Extract amounts
            amount_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
            for match in re.finditer(amount_pattern, page_text):
                entities["amounts"].append(match.group())
            
            # Extract identifiers (like invoice numbers)
            id_pattern = r'[A-Z]{2,}\d{5,}|INV[-\s]?\d+'
            for match in re.finditer(id_pattern, page_text, re.IGNORECASE):
                entities["identifiers"].append(match.group())
            
            # Extract names (simple pattern)
            name_pattern = r'[A-Z][a-z]+ [A-Z][a-z]+'
            for match in re.finditer(name_pattern, page_text):
                entities["names"].append(match.group())
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _get_context(self, text: str, start: int, end: int, context_chars: int = 50) -> str:
        """Get context around a match"""
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        return text[context_start:context_end]
    
    def _segment_text_by_content(self, ocr_results: Dict[int, OCRResult]) -> List[Dict[str, Any]]:
        """Segment text by content type (header, body, footer, table)"""
        segments = []
        
        for page_num, ocr_result in ocr_results.items():
            lines = self._group_words_into_lines(ocr_result.words)
            
            for i, (line_text, line_words) in enumerate(lines):
                if not line_words:
                    continue
                
                # Determine content type based on position and content
                avg_y = sum(w.bbox.y1 for w in line_words) / len(line_words)
                
                if avg_y < 0.15:
                    content_type = "header"
                elif avg_y > 0.85:
                    content_type = "footer"
                elif any(word.text.isupper() and len(word.text) > 3 for word in line_words):
                    content_type = "heading"
                elif ":" in line_text and len(line_text.split(":")) == 2:
                    content_type = "key_value"
                else:
                    content_type = "body"
                
                segments.append({
                    "page": page_num,
                    "type": content_type,
                    "text": line_text,
                    "bbox": self._get_line_bbox(line_words),
                    "line_number": i
                })
        
        return segments
    
    def _get_line_bbox(self, words: List[OCRWord]) -> Dict[str, float]:
        """Calculate bounding box for a line of words"""
        if not words:
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        
        x1 = min(w.bbox.x1 for w in words)
        y1 = min(w.bbox.y1 for w in words)
        x2 = max(w.bbox.x2 for w in words)
        y2 = max(w.bbox.y2 for w in words)
        
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    
    def _calculate_text_quality(self, ocr_results: Dict[int, OCRResult]) -> Dict[str, Any]:
        """Calculate text quality metrics"""
        if not ocr_results:
            return {"error": "No OCR results"}
        
        confidences = [ocr.average_confidence for ocr in ocr_results.values()]
        word_confidences = []
        
        for ocr_result in ocr_results.values():
            word_confidences.extend([w.confidence for w in ocr_result.words])
        
        return {
            "page_level": {
                "mean_confidence": float(sum(confidences) / len(confidences)) if confidences else 0,
                "min_confidence": float(min(confidences)) if confidences else 0,
                "max_confidence": float(max(confidences)) if confidences else 0,
                "pages_with_low_confidence": sum(1 for c in confidences if c < 0.7)
            },
            "word_level": {
                "mean_confidence": float(sum(word_confidences) / len(word_confidences)) if word_confidences else 0,
                "words_with_low_confidence": sum(1 for c in word_confidences if c < 0.6),
                "total_words": len(word_confidences)
            }
        }
    
    def _identify_document_sections(self, ocr_results: Dict[int, OCRResult]) -> List[Dict[str, Any]]:
        """Identify logical document sections"""
        sections = []
        current_section = None
        
        for page_num, ocr_result in sorted(ocr_results.items()):
            page_text = ocr_result.text.lower()
            
            # Common section headers
            section_keywords = {
                "introduction": ["introduction", "abstract", "summary", "overview"],
                "methodology": ["methodology", "approach", "procedure", "method"],
                "results": ["results", "findings", "analysis", "data"],
                "discussion": ["discussion", "conclusion", "recommendation", "future work"],
                "references": ["references", "bibliography", "works cited"],
                "appendix": ["appendix", "attachment", "annex"],
            }
            
            # Check for section headers in first few lines
            lines = page_text.split('\n')[:10]
            for line in lines:
                for section_name, keywords in section_keywords.items():
                    if any(keyword in line for keyword in keywords):
                        if current_section:
                            sections.append(current_section)
                        
                        current_section = {
                            "name": section_name,
                            "start_page": page_num,
                            "header_line": line[:100],
                            "content_pages": [page_num]
                        }
                        break
                
                if current_section:
                    break
            
            # Add page to current section
            if current_section and page_num not in current_section["content_pages"]:
                current_section["content_pages"].append(page_num)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections