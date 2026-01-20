# app/core/models.py - COMPLETE FIXED VERSION
from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime
import uuid
import numpy as np

# ========== ENUMS ==========
class DocumentType(str, Enum):
    FINANCIAL_REPORT = "financial_report"
    INVOICE = "invoice"
    CONTRACT = "contract"
    FORM = "form"
    RESEARCH_PAPER = "research_paper"
    PRESENTATION = "presentation"
    LETTER = "letter"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class AgentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"

class ProcessingStep(str, Enum):
    UPLOAD = "upload"
    PREPROCESSING = "preprocessing"
    PROCESSING = "processing"
    VISION = "vision"
    TEXT = "text"
    FUSION = "fusion"
    VALIDATION = "validation"
    RESULTS = "results"
    QUERY = "query"
    ERROR = "error"
    COMPLETED = "completed"

class QualityScore(BaseModel):
    sharpness: float = Field(..., ge=0, le=1)
    brightness: float = Field(..., ge=0, le=1)
    contrast: float = Field(..., ge=0, le=1)
    noise_level: float = Field(..., ge=0, le=1)
    overall: float = Field(..., ge=0, le=1)

class ContradictionType(str, Enum):
    CHART_TEXT_CONFLICT = "chart_text_conflict"
    NUMERIC_INCONSISTENCY = "numeric_inconsistency"
    DATE_MISMATCH = "date_mismatch"
    SIGNATURE_ABSENCE = "signature_absence"
    CALCULATION_ERROR = "calculation_error"
    SUMMARY_TABLE_MISMATCH = "summary_table_mismatch"
    FORMAT_ISSUE = "format_issue"
    DATA_TYPE_MISMATCH = "data_type_mismatch"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class QueryType(str, Enum):
    TEXT = "text"
    VISUAL = "visual"
    MIXED = "mixed"
    SEMANTIC = "semantic"

# ========== MULTI-MODAL ENHANCEMENTS ==========

class BoundingBox(BaseModel):
    """Bounding box coordinates: [x1, y1, x2, y2]"""
    x1: float
    y1: float 
    x2: float
    y2: float
    
    def to_list(self) -> List[float]:
        """Convert to list format for compatibility"""
        return [self.x1, self.y1, self.x2, self.y2]

class LayoutRegion(BaseModel):
    """Layout analysis result from LayoutLM/Document AI"""
    bbox: BoundingBox
    label: str  # "title", "paragraph", "table", "figure", "header", "footer"
    confidence: float
    page_num: int
    text_content: Optional[str] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 4)

class OCRWord(BaseModel):
    """Single word from OCR with precise location"""
    text: str
    bbox: BoundingBox
    confidence: float
    page_num: int
    line_num: Optional[int] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 4)

class OCRResult(BaseModel):
    """Full OCR result for one page"""
    page_num: int
    text: str
    words: List[OCRWord]
    average_confidence: float
    image_shape: Optional[Tuple[int, int]] = None  # (height, width)
    
    @validator('average_confidence')
    def validate_confidence(cls, v):
        return round(v, 4)

# ========== CORE MODELS (MUST COME BEFORE ENHANCED MODELS) ==========

class VisualElement(BaseModel):
    element_type: str
    bbox: List[int]
    confidence: float
    page_num: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def confidence_percentage(cls, v):
        return round(v, 4)

class EnhancedVisualElement(VisualElement):
    """Enhanced version with better bounding box support"""
    bbox: BoundingBox  # Override to use BoundingBox instead of List[int]
    text_content: Optional[str] = None  # OCR text in this region
    ocr_confidence: Optional[float] = None

class ExtractedField(BaseModel):
    value: Any
    confidence: float = Field(..., ge=0, le=1)
    sources: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def confidence_percentage(cls, v):
        return round(v, 4)

class Contradiction(BaseModel):
    contradiction_type: ContradictionType
    severity: SeverityLevel
    field_a: Optional[str] = None
    field_b: Optional[str] = None
    value_a: Optional[Any] = None
    value_b: Optional[Any] = None
    explanation: str
    confidence: float
    recommendation: Optional[str] = None
    
    @validator('confidence')
    def confidence_percentage(cls, v):
        return round(v, 4)

class AgentResult(BaseModel):
    """Standardized agent output for UI display"""
    agent_name: str
    status: AgentStatus = AgentStatus.PENDING
    confidence: float = Field(default=0.0, ge=0, le=1)
    summary: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    @validator('confidence')
    def confidence_percentage(cls, v):
        return round(v, 4)
    
    class Config:
        use_enum_values = True

# ========== MAIN PROCESSING MODELS ==========

class ProcessingState(BaseModel):
    """State for LangGraph workflow"""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    images: List[Any] = Field(default_factory=list)
    
    # Preprocessing results
    quality_scores: Dict[int, QualityScore] = Field(default_factory=dict)
    document_type: Optional[DocumentType] = None
    layout_strategy: Optional[str] = None
    
    # Vision results
    visual_elements: Dict[int, List[VisualElement]] = Field(default_factory=dict)
    chart_analysis: Dict[str, Any] = Field(default_factory=dict)
    table_structures: Dict[str, Any] = Field(default_factory=dict)
    signature_verification: Dict[str, Any] = Field(default_factory=dict)
    
    # Text results
    ocr_results: Dict[int, Any] = Field(default_factory=dict)
    ocr_confidence: Dict[int, float] = Field(default_factory=dict)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # ✅ CRITICAL FIX: Added missing extracted_text field
    extracted_text: str = Field(default="")
    
    # Fusion results
    aligned_data: Dict[str, Any] = Field(default_factory=dict)
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    temporal_consistency: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation results
    contradictions: List[Contradiction] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0, le=1)
    compliance_issues: List[str] = Field(default_factory=list)
    
    # Explainability results
    explanations: Dict[str, str] = Field(default_factory=dict)
    review_recommendations: List[str] = Field(default_factory=list)
    
    # Final output
    extracted_fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    processing_start: datetime = Field(default_factory=datetime.now)
    processing_end: Optional[datetime] = None
    
    # Agent outputs
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # ========== UI-FRIENDLY METHODS ==========
    def to_ui_summary(self) -> 'UIResultSummary':
        """Convert to UI-friendly summary"""
        return UIResultSummary(
            document_id=self.document_id,
            document_type=self.document_type.value if self.document_type else "unknown",
            processing_time_seconds=self.get_processing_time(),
            confidence_scores=self.get_confidence_scores(),
            overall_confidence=self.calculate_overall_confidence(),
            visual_elements=self.count_visual_elements(),
            extracted_fields=len(self.extracted_fields),
            contradictions=len(self.contradictions),
            validation_issues=len(self.compliance_issues),
            agent_summaries=self.get_agent_summaries(),
            top_findings=self.extract_top_findings(),
            recommendations=self.review_recommendations,
            risk_level=self.calculate_risk_level()
        )
    
    def get_processing_time(self) -> float:
        """Calculate processing time in seconds"""
        if self.processing_end:
            return (self.processing_end - self.processing_start).total_seconds()
        return 0.0
    
    def get_confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for UI display"""
        scores = {
            "ocr": self.calculate_average_confidence(self.ocr_confidence.values()),
            "vision": self.calculate_vision_confidence(),
            "fusion": self.calculate_average_confidence(self.field_confidences.values()),
            "validation": 1.0 - self.risk_score,  # Higher risk = lower validation confidence
        }
        
        # Round for UI display
        return {k: round(v, 3) for k, v in scores.items()}
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence for UI"""
        scores = self.get_confidence_scores()
        if not scores:
            return 0.0
        return round(sum(scores.values()) / len(scores), 3)
    
    def count_visual_elements(self) -> int:
        """Count total visual elements"""
        total = 0
        for page_elements in self.visual_elements.values():
            total += len(page_elements)
        return total
    
    def get_agent_summaries(self) -> Dict[str, str]:
        """Get agent summaries for UI"""
        summaries = {}
        
        # Vision agent summary
        vision_count = self.count_visual_elements()
        summaries["vision"] = f"Detected {vision_count} visual elements"
        
        # Text agent summary
        total_words = sum(len(str(text).split()) for text in self.ocr_results.values())
        summaries["text"] = f"Extracted {total_words} words"
        
        # Fusion agent summary
        summaries["fusion"] = f"Aligned {len(self.extracted_fields)} fields"
        
        # Validation agent summary
        critical_issues = sum(1 for c in self.contradictions if c.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL])
        summaries["validation"] = f"Found {len(self.contradictions)} issues ({critical_issues} critical)"
        
        return summaries
    
    def extract_top_findings(self) -> List[str]:
        """Extract top findings for UI display"""
        findings = []
        
        # Document type
        if self.document_type:
            findings.append(f"Document type: {self.document_type.value.replace('_', ' ').title()}")
        
        # Visual elements
        vision_count = self.count_visual_elements()
        if vision_count > 0:
            findings.append(f"Found {vision_count} visual elements")
        
        # Key contradictions
        if self.contradictions:
            high_severity = sum(1 for c in self.contradictions if c.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL])
            findings.append(f"Validation: {len(self.contradictions)} issues ({high_severity} critical)")
        
        # Extracted fields
        if self.extracted_fields:
            findings.append(f"Extracted {len(self.extracted_fields)} structured fields")
        
        # Quality score
        if self.quality_scores:
            avg_quality = sum(qs.overall for qs in self.quality_scores.values()) / len(self.quality_scores)
            findings.append(f"Document quality: {avg_quality:.0%}")
        
        return findings[:5]  # Limit to top 5
    
    def calculate_risk_level(self) -> str:
        """Calculate risk level for UI display"""
        if not self.contradictions:
            return "low"
        
        severities = [c.severity for c in self.contradictions]
        
        if SeverityLevel.CRITICAL in severities:
            return "high"
        elif SeverityLevel.HIGH in severities:
            return "medium"
        else:
            return "low"
    
    def calculate_average_confidence(self, values) -> float:
        """Calculate average confidence from a list of values"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def calculate_vision_confidence(self) -> float:
        """Calculate vision agent confidence"""
        if not self.visual_elements:
            return 0.0
        
        confidences = []
        for page_elements in self.visual_elements.values():
            for element in page_elements:
                confidences.append(element.confidence)
        
        return self.calculate_average_confidence(confidences) if confidences else 0.0

# ========== MULTI-MODAL DOCUMENT MODEL ==========

class MultiModalDocument(BaseModel):
    """
    UNIFIED DOCUMENT with BOTH text and visual data
    This extends ProcessingState with better multi-modal support
    """
    
    # Basic metadata (same as ProcessingState)
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    
    # Document classification
    document_type: Optional[DocumentType] = None
    
    # TEXT MODALITY
    raw_text: str = ""
    ocr_results: Dict[int, OCRResult] = Field(default_factory=dict)
    text_chunks: List[str] = Field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # VISUAL MODALITY
    images: List[Any] = Field(default_factory=list)  # Keep for compatibility
    layout_regions: List[LayoutRegion] = Field(default_factory=list)
    visual_elements: List[EnhancedVisualElement] = Field(default_factory=list)  # ✅ FIXED: Changed from dict to list
    
    # Quality scores (from your existing model)
    quality_scores: Dict[int, QualityScore] = Field(default_factory=dict)
    
    # Agent results
    chart_analysis: Dict[str, Any] = Field(default_factory=dict)
    table_structures: Dict[str, Any] = Field(default_factory=dict)
    signature_verification: Dict[str, Any] = Field(default_factory=dict)
    
    # Fusion results
    aligned_data: Dict[str, Any] = Field(default_factory=dict)
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    temporal_consistency: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation results (from your existing model)
    contradictions: List[Contradiction] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0, le=1)
    compliance_issues: List[str] = Field(default_factory=list)
    
    # Explainability results
    explanations: Dict[str, str] = Field(default_factory=dict)
    review_recommendations: List[str] = Field(default_factory=list)  # ✅ ADDED
    
    # Agent outputs (for UI)
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)  # ✅ ADDED
    processing_start: datetime = Field(default_factory=datetime.now)
    processing_end: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True  # For numpy arrays
    
    # ========== CONVERSION METHODS ==========
    
    @classmethod
    def from_processing_state(cls, state: ProcessingState) -> 'MultiModalDocument':
        """Convert existing ProcessingState to MultiModalDocument"""
        # Extract raw text
        raw_text = ""
        if hasattr(state, 'extracted_text') and state.extracted_text:
            raw_text = state.extracted_text
        elif hasattr(state, 'ocr_results') and state.ocr_results:
            # Build text from OCR results
            text_parts = []
            for page_num, result in state.ocr_results.items():
                if isinstance(result, dict) and 'text' in result:
                    text_parts.append(result['text'])
                elif isinstance(result, str):
                    text_parts.append(result)
            raw_text = "\n".join(text_parts)
        
        doc = cls(
            document_id=state.document_id,
            file_path=state.file_path,
            file_type=state.file_type,
            document_type=state.document_type,
            raw_text=raw_text,
            extracted_entities=state.extracted_entities if hasattr(state, 'extracted_entities') else {},
            semantic_analysis=state.semantic_analysis if hasattr(state, 'semantic_analysis') else {},
            quality_scores=state.quality_scores if hasattr(state, 'quality_scores') else {},
            chart_analysis=state.chart_analysis if hasattr(state, 'chart_analysis') else {},
            table_structures=state.table_structures if hasattr(state, 'table_structures') else {},
            signature_verification=state.signature_verification if hasattr(state, 'signature_verification') else {},
            aligned_data=state.aligned_data if hasattr(state, 'aligned_data') else {},
            field_confidences=state.field_confidences if hasattr(state, 'field_confidences') else {},
            temporal_consistency=state.temporal_consistency if hasattr(state, 'temporal_consistency') else {},
            contradictions=state.contradictions if hasattr(state, 'contradictions') else [],
            risk_score=state.risk_score if hasattr(state, 'risk_score') else 0.0,
            compliance_issues=state.compliance_issues if hasattr(state, 'compliance_issues') else [],
            explanations=state.explanations if hasattr(state, 'explanations') else {},
            review_recommendations=state.review_recommendations if hasattr(state, 'review_recommendations') else [],
            agent_outputs=state.agent_outputs if hasattr(state, 'agent_outputs') else {},
            processing_metadata=state.processing_metadata if hasattr(state, 'processing_metadata') else {},
            processing_start=state.processing_start if hasattr(state, 'processing_start') else datetime.now(),
            processing_end=state.processing_end if hasattr(state, 'processing_end') else None,
            errors=state.errors if hasattr(state, 'errors') else [],
            images=state.images if hasattr(state, 'images') else []
        )
        
        # ✅ FIX: Convert visual elements to new format
        visual_elements_list = []
        if hasattr(state, 'visual_elements') and state.visual_elements:
            for page_num, elements in state.visual_elements.items():
                if isinstance(elements, list):
                    for elem in elements:
                        if hasattr(elem, 'bbox') and isinstance(elem.bbox, list) and len(elem.bbox) == 4:
                            enhanced_elem = EnhancedVisualElement(
                                element_type=elem.element_type,
                                bbox=BoundingBox(
                                    x1=elem.bbox[0],
                                    y1=elem.bbox[1],
                                    x2=elem.bbox[2],
                                    y2=elem.bbox[3]
                                ),
                                confidence=elem.confidence,
                                page_num=elem.page_num,
                                metadata=elem.metadata if hasattr(elem, 'metadata') else {}
                            )
                            visual_elements_list.append(enhanced_elem)
        
        doc.visual_elements = visual_elements_list
        
        # ✅ FIX: Convert OCR results
        if hasattr(state, 'ocr_results') and state.ocr_results:
            for page_num, result in state.ocr_results.items():
                if isinstance(result, dict):
                    # Create basic OCRResult from old format
                    words = []
                    if 'words' in result and isinstance(result['words'], list):
                        # Convert word dicts to OCRWord objects
                        for word_dict in result['words'][:50]:  # Limit to 50 words
                            if isinstance(word_dict, dict) and 'bbox' in word_dict:
                                words.append(OCRWord(
                                    text=word_dict.get('text', ''),
                                    bbox=BoundingBox(
                                        x1=word_dict['bbox'][0],
                                        y1=word_dict['bbox'][1],
                                        x2=word_dict['bbox'][2],
                                        y2=word_dict['bbox'][3]
                                    ),
                                    confidence=word_dict.get('confidence', 0.8),
                                    page_num=page_num
                                ))
                    
                    # Get text from result
                    text = result.get('text', '')
                    if not text and words:
                        # Build text from words
                        text = ' '.join([word.text for word in words])
                    
                    ocr_result = OCRResult(
                        page_num=page_num,
                        text=text,
                        words=words,
                        average_confidence=result.get('confidence', 0.8)
                    )
                    doc.ocr_results[page_num] = ocr_result
        
        # ✅ FIX: Restore layout regions from processing_metadata
        if hasattr(state, 'processing_metadata') and state.processing_metadata:
            if 'layout_regions' in state.processing_metadata:
                layout_regions_data = state.processing_metadata['layout_regions']
                if isinstance(layout_regions_data, list):
                    for region_data in layout_regions_data:
                        if isinstance(region_data, dict) and 'bbox' in region_data:
                            doc.add_layout_region(LayoutRegion(
                                bbox=BoundingBox(
                                    x1=region_data['bbox'][0],
                                    y1=region_data['bbox'][1],
                                    x2=region_data['bbox'][2],
                                    y2=region_data['bbox'][3]
                                ),
                                label=region_data.get('label', 'unknown'),
                                confidence=region_data.get('confidence', 0.7),
                                page_num=region_data.get('page', 0),
                                text_content=region_data.get('text_content')
                            ))
        
        return doc
    
    def to_processing_state(self) -> ProcessingState:
        """Convert back to ProcessingState for compatibility"""
        # ✅ FIX: Ensure we have all fields
        state = ProcessingState(
            document_id=self.document_id,
            file_path=self.file_path,
            file_type=self.file_type,
            images=self.images,
            quality_scores=self.quality_scores,
            document_type=self.document_type,
            extracted_text=self.raw_text,  # ✅ FIX: Use raw_text
            extracted_entities=self.extracted_entities,
            semantic_analysis=self.semantic_analysis,
            chart_analysis=self.chart_analysis,
            table_structures=self.table_structures,
            signature_verification=self.signature_verification,
            aligned_data=self.aligned_data,
            field_confidences=self.field_confidences,
            temporal_consistency=self.temporal_consistency,
            contradictions=self.contradictions,
            risk_score=self.risk_score,
            compliance_issues=self.compliance_issues,
            explanations=self.explanations,
            review_recommendations=self.review_recommendations,
            agent_outputs=self.agent_outputs,
            processing_metadata=self.processing_metadata.copy() if self.processing_metadata else {},  # ✅ FIX: Copy to avoid mutation
            processing_start=self.processing_start,
            processing_end=self.processing_end,
            errors=self.errors
        )
        
        # ✅ FIX: Convert visual elements back
        visual_elements_dict = {}
        if isinstance(self.visual_elements, list):
            for elem in self.visual_elements:
                if hasattr(elem, 'page_num'):
                    if elem.page_num not in visual_elements_dict:
                        visual_elements_dict[elem.page_num] = []
                    
                    visual_elements_dict[elem.page_num].append(
                        VisualElement(
                            element_type=elem.element_type,
                            bbox=elem.bbox.to_list(),
                            confidence=elem.confidence,
                            page_num=elem.page_num,
                            metadata=elem.metadata if hasattr(elem, 'metadata') else {}
                        )
                    )
        
        state.visual_elements = visual_elements_dict
        
        # ✅ FIX: Convert OCR results back
        for page_num, ocr_result in self.ocr_results.items():
            # Convert OCRResult to old format
            word_dicts = []
            for word in ocr_result.words[:50]:  # Limit for compatibility
                word_dicts.append({
                    "text": word.text,
                    "bbox": word.bbox.to_list(),
                    "confidence": word.confidence
                })
            
            state.ocr_results[page_num] = {
                "text": ocr_result.text,
                "confidence": ocr_result.average_confidence,
                "words": word_dicts
            }
            state.ocr_confidence[page_num] = ocr_result.average_confidence
        
        # ✅ FIX: Store layout regions in processing_metadata so they're not lost
        if self.layout_regions:
            if 'layout_regions' not in state.processing_metadata:
                state.processing_metadata['layout_regions'] = []
            
            for region in self.layout_regions:
                state.processing_metadata['layout_regions'].append({
                    "label": region.label,
                    "bbox": region.bbox.to_list(),
                    "page": region.page_num,
                    "confidence": region.confidence,
                    "text_content": region.text_content
                })
        
        return state
    
    # ========== HELPER METHODS ==========
    
    def get_visual_elements_by_type(self, element_type: str) -> List[EnhancedVisualElement]:
        """Get all visual elements of a specific type"""
        return [elem for elem in self.visual_elements if elem.element_type == element_type]
    
    def get_layout_regions_by_label(self, label: str) -> List[LayoutRegion]:
        """Get all layout regions with a specific label"""
        return [region for region in self.layout_regions if region.label == label]
    
    def add_visual_element(self, element: EnhancedVisualElement):
        """Add a visual element"""
        self.visual_elements.append(element)
    
    def add_layout_region(self, region: LayoutRegion):
        """Add a layout region"""
        self.layout_regions.append(region)
    
    def add_ocr_result(self, page_num: int, ocr_result: OCRResult):
        """Add OCR result for a page"""
        self.ocr_results[page_num] = ocr_result
        # ✅ FIX: Properly concatenate text
        if ocr_result.text:
            if not self.raw_text:
                self.raw_text = ocr_result.text
            else:
                self.raw_text += "\n" + ocr_result.text
    
    def get_processing_time(self) -> float:
        """Calculate processing time in seconds"""
        if self.processing_end:
            return (self.processing_end - self.processing_start).total_seconds()
        return 0.0

# ========== UI-SPECIFIC MODELS ==========

class UIProcessingState(BaseModel):
    """UI-optimized processing state"""
    document_id: str
    current_step: ProcessingStep = ProcessingStep.UPLOAD
    overall_progress: float = Field(default=0.0, ge=0, le=1)
    
    # Agent status for progress tracking
    agents: Dict[str, AgentResult] = Field(default_factory=dict)
    
    # Document metadata
    document_type: Optional[DocumentType] = None
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    
    # Processing results (simplified for UI)
    visual_elements_count: int = 0
    extracted_fields_count: int = 0
    contradictions_count: int = 0
    overall_confidence: float = Field(default=0.0, ge=0, le=1)
    
    # Timestamps
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    
    # UI-specific fields
    user_message: Optional[str] = None
    next_action: Optional[str] = None
    can_proceed: bool = False
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
    
    def update_progress(self, step: ProcessingStep, progress: float):
        """Update progress for a specific step"""
        self.current_step = step
        self.overall_progress = progress
        self.last_updated = datetime.now()
    
    def set_agent_status(self, agent_name: str, status: AgentStatus, 
                        confidence: float = None, summary: str = None):
        """Update agent status"""
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentResult(agent_name=agent_name)
        
        self.agents[agent_name].status = status
        if confidence is not None:
            self.agents[agent_name].confidence = confidence
        if summary is not None:
            self.agents[agent_name].summary = summary
        
        self.last_updated = datetime.now()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary for UI"""
        completed_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.COMPLETED)
        total_agents = len(self.agents)
        
        return {
            "current_step": self.current_step.value,
            "progress_percentage": int(self.overall_progress * 100),
            "agents_completed": f"{completed_agents}/{total_agents}",
            "can_proceed": self.can_proceed,
            "user_message": self.user_message,
            "next_action": self.next_action
        }

class UIResultSummary(BaseModel):
    """Summary of results for UI display"""
    document_id: str
    document_type: str
    processing_time_seconds: float
    
    # Confidence scores
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    overall_confidence: float = Field(default=0.0, ge=0, le=1)
    
    # Counts for quick overview
    visual_elements: int = 0
    extracted_fields: int = 0
    contradictions: int = 0
    validation_issues: int = 0
    
    # Agent summaries
    agent_summaries: Dict[str, str] = Field(default_factory=dict)
    
    # Key findings (for quick insight)
    top_findings: List[str] = Field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "low"  # low, medium, high
    
    # UI display properties
    display_color: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('display_color', always=True)
    def set_display_color(cls, v, values):
        """Set display color based on risk level"""
        risk_level = values.get('risk_level', 'low')
        color_map = {
            'low': '#10b981',    # green
            'medium': '#f59e0b', # amber
            'high': '#ef4444'    # red
        }
        return color_map.get(risk_level, '#6b7280')
    
    def to_ui_card(self) -> Dict[str, Any]:
        """Convert to UI card format"""
        return {
            "id": self.document_id,
            "title": f"Document Analysis",
            "subtitle": f"Type: {self.document_type.replace('_', ' ').title()}",
            "confidence": f"{self.overall_confidence:.0%}",
            "stats": {
                "elements": self.visual_elements,
                "fields": self.extracted_fields,
                "issues": self.contradictions,
                "time": f"{self.processing_time_seconds:.1f}s"
            },
            "color": self.display_color,
            "risk": self.risk_level
        }

class QueryRequest(BaseModel):
    """Query request model - UI optimized"""
    document_id: str
    question: str
    query_type: QueryType = QueryType.TEXT
    include_visual: bool = False
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)

class QueryResponse(BaseModel):
    """Query response model - UI optimized"""
    success: bool
    document_id: str
    question: str
    answer: str
    confidence: float = Field(..., ge=0, le=1)
    sources: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    
    # UI display properties
    display_type: str = "text"  # text, visual, mixed, structured
    has_visual_content: bool = False
    recommended_action: Optional[str] = None
    confidence_color: Optional[str] = None
    
    class Config:
        use_enum_values = True
    
    @validator('confidence_color', always=True)
    def set_confidence_color(cls, v, values):
        """Set color based on confidence score"""
        confidence = values.get('confidence', 0)
        if confidence >= 0.8:
            return '#10b981'  # green
        elif confidence >= 0.6:
            return '#f59e0b'  # amber
        else:
            return '#ef4444'  # red

class VisualizationData(BaseModel):
    """Visualization data model - UI optimized"""
    document_id: str
    has_original_image: bool = False
    detected_elements: List[Dict[str, Any]] = Field(default_factory=list)
    visualization_available: bool = False
    
    # UI-friendly properties
    element_count: int = 0
    elements_by_type: List[Dict[str, Any]] = Field(default_factory=list)
    ui_friendly: bool = True
    
    def get_element_summary(self) -> Dict[str, int]:
        """Get element count by type"""
        summary = {}
        for elem in self.detected_elements:
            elem_type = elem.get('type', 'unknown')
            summary[elem_type] = summary.get(elem_type, 0) + 1
        return summary

class UploadResponse(BaseModel):
    """Upload response model - UI optimized"""
    success: bool
    document_id: str
    message: str
    next_step: str
    
    # UI state
    ui_state: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProcessingResponse(BaseModel):
    """Processing response model - UI optimized"""
    success: bool
    document_id: str
    processing_complete: bool
    result_available: bool
    
    # UI state
    ui_state: Dict[str, Any] = Field(default_factory=dict)

class StatusResponse(BaseModel):
    """Status response model - UI optimized"""
    document_id: str
    status: str
    timestamp: Optional[str] = None
    error: Optional[str] = None
    
    # UI state
    ui_state: Optional[Dict[str, Any]] = None

# ========== AGENT CONFIGURATION MODELS ==========

class AgentConfiguration(BaseModel):
    """Agent configuration for UI settings"""
    enabled: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    max_processing_time: Optional[int] = None  # seconds
    
    # UI display
    display_name: str
    description: str
    icon: str

class ProcessingConfiguration(BaseModel):
    """Processing configuration from UI"""
    document_id: str
    
    # Agent configurations
    agents: Dict[str, AgentConfiguration] = Field(default_factory=dict)
    
    # Processing options
    enable_quality_check: bool = True
    enable_validation: bool = True
    enable_rag_indexing: bool = True
    
    # UI state
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }