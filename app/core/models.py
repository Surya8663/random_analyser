# app/core/models.py - CORRECTED VERSION
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

# ========== BOUNDING BOX BASE MODEL ==========

class BoundingBox(BaseModel):
    """Bounding box coordinates: [x1, y1, x2, y2] in normalized format (0-1)"""
    x1: float
    y1: float 
    x2: float
    y2: float
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure x2 > x1 and y2 > y1
        if self.x2 <= self.x1:
            self.x2 = self.x1 + 0.001
        if self.y2 <= self.y1:
            self.y2 = self.y1 + 0.001
    
    def to_list(self) -> List[float]:
        """Convert to list format for compatibility"""
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_pixel_list(self, width: int, height: int) -> List[int]:
        """Convert to pixel coordinates"""
        return [
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height)
        ]
    
    def is_normalized(self) -> bool:
        """Check if coordinates are normalized (0-1)"""
        return all(0.0 <= coord <= 1.0 for coord in [self.x1, self.y1, self.x2, self.y2])
    
    @classmethod
    def from_pixels(cls, x1: int, y1: int, x2: int, y2: int, 
                   width: int, height: int) -> 'BoundingBox':
        """Create from pixel coordinates"""
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        
        return cls(
            x1=max(0.0, min(x1 / width, 1.0)),
            y1=max(0.0, min(y1 / height, 1.0)),
            x2=max(0.0, min(x2 / width, 1.0)),
            y2=max(0.0, min(y2 / height, 1.0))
        )

# In your existing models.py, update the LayoutRegion class:
class LayoutRegion(BaseModel):
    """Layout analysis result"""
    bbox: BoundingBox
    label: str  # "title", "paragraph", "table", "figure", "header", "footer"
    confidence: float
    page_num: int
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)  # ADD THIS LINE
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 4)

# ========== VISUAL ELEMENT MODELS ==========

class VisualElement(BaseModel):
    """Base visual element model - accepts both normalized and pixel coordinates"""
    element_type: str
    bbox: Union[BoundingBox, List[float]]  # Can be BoundingBox or list
    confidence: float
    page_num: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('bbox', pre=True)
    def validate_bbox(cls, v):
        """Convert list to BoundingBox if needed"""
        if isinstance(v, BoundingBox):
            return v
        elif isinstance(v, list) and len(v) == 4:
            # Check if values are normalized
            max_val = max(v)
            if max_val > 1.0:
                # Assume pixel coordinates, but we'll handle conversion elsewhere
                return BoundingBox(x1=float(v[0]), y1=float(v[1]), 
                                 x2=float(v[2]), y2=float(v[3]))
            else:
                return BoundingBox(x1=float(v[0]), y1=float(v[1]), 
                                 x2=float(v[2]), y2=float(v[3]))
        return v
    
    @validator('confidence')
    def confidence_percentage(cls, v):
        return round(v, 4)
    
    def get_normalized_bbox(self) -> BoundingBox:
        """Get normalized bounding box"""
        if isinstance(self.bbox, BoundingBox):
            return self.bbox
        else:
            return BoundingBox(x1=float(self.bbox[0]), y1=float(self.bbox[1]),
                             x2=float(self.bbox[2]), y2=float(self.bbox[3]))

class EnhancedVisualElement(VisualElement):
    """Enhanced version with text content"""
    bbox: BoundingBox  # Force BoundingBox type
    text_content: Optional[str] = None  # OCR text in this region
    ocr_confidence: Optional[float] = None
    
    @validator('bbox', pre=True)
    def ensure_bbox_type(cls, v):
        if isinstance(v, BoundingBox):
            return v
        elif isinstance(v, list) and len(v) == 4:
            return BoundingBox(x1=float(v[0]), y1=float(v[1]), 
                             x2=float(v[2]), y2=float(v[3]))
        raise ValueError("bbox must be BoundingBox or list of 4 floats")

# ========== OCR MODELS ==========

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

# ========== OTHER MODELS ==========

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
    
    # Text content
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
    
    # Helper methods
    def get_processing_time(self) -> float:
        """Calculate processing time in seconds"""
        if self.processing_end:
            return (self.processing_end - self.processing_start).total_seconds()
        return 0.0
class ProvenanceRecord(BaseModel):
    """Record of how a field was extracted"""
    agent_name: str
    extraction_method: str
    source_modality: str  # "text", "visual", "fusion", "metadata"
    source_region_id: Optional[str] = None
    source_bbox: Optional[BoundingBox] = None
    source_page: Optional[int] = None
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    reasoning_notes: Optional[str] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 4)

class ExplainableField(BaseModel):
    """Field with full provenance tracking"""
    field_name: str
    field_type: str  # "text", "number", "date", "boolean", "signature"
    value: Any
    confidence: float = Field(..., ge=0, le=1)
    provenance: List[ProvenanceRecord] = Field(default_factory=list)
    modality_sources: List[str] = Field(default_factory=list)  # ["text", "visual", "fusion"]
    final_source: str  # Which provenance record was selected as final
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 4)
    
    def get_primary_provenance(self) -> Optional[ProvenanceRecord]:
        """Get the provenance record that was used as final source"""
        for record in self.provenance:
            if record.agent_name == self.final_source:
                return record
        return self.provenance[0] if self.provenance else None

# ========== PHASE 4: EVALUATION SCHEMAS ==========

class GroundTruthField(BaseModel):
    """Ground truth for evaluation (if available)"""
    field_name: str
    true_value: Any
    value_type: str
    importance_weight: float = Field(default=1.0, ge=0, le=2)

class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics"""
    # Field-level metrics
    field_precision: float = Field(..., ge=0, le=1)
    field_recall: float = Field(..., ge=0, le=1)
    field_f1: float = Field(..., ge=0, le=1)
    field_coverage: float = Field(..., ge=0, le=1)
    
    # Entity extraction metrics
    entity_precision: float = Field(..., ge=0, le=1)
    entity_recall: float = Field(..., ge=0, le=1)
    entity_f1: float = Field(..., ge=0, le=1)
    
    # Alignment metrics
    alignment_accuracy: float = Field(..., ge=0, le=1)
    cross_modal_consistency: float = Field(..., ge=0, le=1)
    
    # Risk detection metrics
    risk_detection_precision: float = Field(..., ge=0, le=1)
    risk_detection_recall: float = Field(..., ge=0, le=1)
    
    # Overall metrics
    overall_accuracy: float = Field(..., ge=0, le=1)
    processing_success_rate: float = Field(..., ge=0, le=1)
    average_confidence: float = Field(..., ge=0, le=1)
    
    # Performance metrics
    processing_time_seconds: float
    agents_success_rate: Dict[str, float] = Field(default_factory=dict)
    
    # Field-level breakdown
    field_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EvaluationReport(BaseModel):
    """Complete evaluation report for a document"""
    document_id: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    has_ground_truth: bool = False
    metrics: EvaluationMetrics
    field_comparisons: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    successful_agents: List[str] = Field(default_factory=list)
    failed_agents: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
# ========== MULTI-MODAL DOCUMENT MODEL ==========

# In your models.py, update the MultiModalDocument class:
class MultiModalDocument(BaseModel):
    """UNIFIED DOCUMENT with BOTH text and visual data"""
    
    # Basic metadata
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
    images: List[Any] = Field(default_factory=list)
    layout_regions: List[LayoutRegion] = Field(default_factory=list)
    visual_elements: List[EnhancedVisualElement] = Field(default_factory=list)
    
    # Quality scores
    quality_scores: Dict[int, QualityScore] = Field(default_factory=dict)
    
    # Agent results
    chart_analysis: Dict[str, Any] = Field(default_factory=dict)
    table_structures: Dict[str, Any] = Field(default_factory=dict)
    signature_verification: Dict[str, Any] = Field(default_factory=dict)
    
    # Fusion results
    aligned_data: Dict[str, Any] = Field(default_factory=dict)
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    temporal_consistency: Dict[str, Any] = Field(default_factory=dict)
    
    # ADD THESE MISSING FIELDS:
    extracted_fields: Dict[str, ExplainableField] = Field(default_factory=dict)  # ADD THIS LINE
    contradictions: List[Contradiction] = Field(default_factory=list)  # ADD THIS LINE if missing
    risk_score: float = Field(default=0.0, ge=0, le=1)  # ADD THIS LINE if missing
    compliance_issues: List[str] = Field(default_factory=list)  # ADD THIS LINE if missing
    
    # Explainability results
    explanations: Dict[str, str] = Field(default_factory=dict)
    review_recommendations: List[str] = Field(default_factory=list)
    
    # Agent outputs
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_start: datetime = Field(default_factory=datetime.now)
    processing_end: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)
    
    ground_truth: Optional[Dict[str, GroundTruthField]] = None
    evaluation_report: Optional[EvaluationReport] = None
    provenance_graph: Dict[str, List[ProvenanceRecord]] = Field(default_factory=dict)
    agent_trace: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
    
    
    
    # ========== CONVERSION METHODS ==========
    
    @classmethod
    def from_processing_state(cls, state: ProcessingState) -> 'MultiModalDocument':
        """Convert existing ProcessingState to MultiModalDocument"""
        # Extract raw text
        raw_text = state.extracted_text if hasattr(state, 'extracted_text') and state.extracted_text else ""
        
        doc = cls(
            document_id=state.document_id,
            file_path=state.file_path,
            file_type=state.file_type,
            document_type=state.document_type,
            raw_text=raw_text,
            extracted_entities=state.extracted_entities,
            semantic_analysis=state.semantic_analysis,
            quality_scores=state.quality_scores,
            chart_analysis=state.chart_analysis,
            table_structures=state.table_structures,
            signature_verification=state.signature_verification,
            aligned_data=state.aligned_data,
            field_confidences=state.field_confidences,
            temporal_consistency=state.temporal_consistency,
            contradictions=state.contradictions,
            risk_score=state.risk_score,
            compliance_issues=state.compliance_issues,
            explanations=state.explanations,
            review_recommendations=state.review_recommendations,
            agent_outputs=state.agent_outputs,
            processing_metadata=state.processing_metadata.copy() if state.processing_metadata else {},
            processing_start=state.processing_start,
            processing_end=state.processing_end,
            errors=state.errors,
            images=state.images if hasattr(state, 'images') else []
        )
        
        # Convert visual elements
        visual_elements_list = []
        if hasattr(state, 'visual_elements') and state.visual_elements:
            for page_num, elements in state.visual_elements.items():
                if isinstance(elements, list):
                    for elem in elements:
                        if hasattr(elem, 'bbox'):
                            bbox = elem.get_normalized_bbox()
                            enhanced_elem = EnhancedVisualElement(
                                element_type=elem.element_type,
                                bbox=bbox,
                                confidence=elem.confidence,
                                page_num=elem.page_num,
                                metadata=elem.metadata if hasattr(elem, 'metadata') else {},
                                text_content=None
                            )
                            visual_elements_list.append(enhanced_elem)
        
        doc.visual_elements = visual_elements_list
        
        # Store layout regions from processing_metadata
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
    
    def add_field_with_provenance(self, 
                                 field_name: str, 
                                 field_type: str,
                                 value: Any, 
                                 confidence: float,
                                 provenance: ProvenanceRecord,
                                 modality_sources: List[str]):
        """Add a field with full provenance tracking"""
        explainable_field = ExplainableField(
            field_name=field_name,
            field_type=field_type,
            value=value,
            confidence=confidence,
            provenance=[provenance],
            modality_sources=modality_sources,
            final_source=provenance.agent_name,
            metadata={}
        )
        self.extracted_fields[field_name] = explainable_field
        
        # Store in provenance graph
        if field_name not in self.provenance_graph:
            self.provenance_graph[field_name] = []
        self.provenance_graph[field_name].append(provenance)
    
    def merge_field_provenance(self, 
                              field_name: str, 
                              new_provenance: ProvenanceRecord,
                              new_value: Any = None,
                              new_confidence: float = None):
        """Merge new provenance into existing field"""
        if field_name in self.extracted_fields:
            field = self.extracted_fields[field_name]
            field.provenance.append(new_provenance)
            
            # Update if new provenance has higher confidence
            if new_confidence and new_confidence > field.confidence:
                field.value = new_value if new_value is not None else field.value
                field.confidence = new_confidence
                field.final_source = new_provenance.agent_name
            
            # Update modality sources
            if new_provenance.source_modality not in field.modality_sources:
                field.modality_sources.append(new_provenance.source_modality)
    


    def to_processing_state(self) -> ProcessingState:
        """Convert back to ProcessingState for compatibility"""
        state = ProcessingState(
            document_id=self.document_id,
            file_path=self.file_path,
            file_type=self.file_type,
            images=self.images,
            quality_scores=self.quality_scores,
            document_type=self.document_type,
            extracted_text=self.raw_text,
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
            processing_metadata=self.processing_metadata.copy() if self.processing_metadata else {},
            processing_start=self.processing_start,
            processing_end=self.processing_end,
            errors=self.errors
        )
        
        # Convert visual elements - convert to pixel coordinates for ProcessingState
        visual_elements_dict = {}
        if isinstance(self.visual_elements, list):
            for elem in self.visual_elements:
                if hasattr(elem, 'page_num'):
                    page = elem.page_num
                    if page not in visual_elements_dict:
                        visual_elements_dict[page] = []
                    
                    # Get pixel coordinates if image exists
                    pixel_bbox = [0, 0, 100, 100]  # Default
                    if page < len(self.images) and self.images[page] is not None:
                        height, width = self.images[page].shape[:2]
                        pixel_bbox = elem.bbox.to_pixel_list(width, height)
                    
                    # Create VisualElement (not EnhancedVisualElement) for ProcessingState
                    visual_element = VisualElement(
                        element_type=elem.element_type,
                        bbox=pixel_bbox,  # Pixel coordinates
                        confidence=elem.confidence,
                        page_num=page,
                        metadata=elem.metadata if hasattr(elem, 'metadata') else {}
                    )
                    
                    visual_elements_dict[page].append(visual_element)
        
        state.visual_elements = visual_elements_dict
        
        # Store layout regions in processing_metadata
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
    
    def add_visual_element(self, element: EnhancedVisualElement):
        """Add a visual element"""
        self.visual_elements.append(element)
    
    def add_layout_region(self, region: LayoutRegion):
        """Add a layout region"""
        self.layout_regions.append(region)
    
    def add_ocr_result(self, page_num: int, ocr_result: OCRResult):
        """Add OCR result for a page"""
        self.ocr_results[page_num] = ocr_result
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
    
    def get_pixel_bbox(self, bbox: BoundingBox, page_num: int = 0) -> List[int]:
        """Convert normalized bbox to pixel coordinates"""
        if page_num < len(self.images) and self.images[page_num] is not None:
            height, width = self.images[page_num].shape[:2]
            return bbox.to_pixel_list(width, height)
        return [0, 0, 0, 0]
    
    def get_normalized_bbox(self, pixel_bbox: List[int], page_num: int = 0) -> List[float]:
        """Convert pixel bbox to normalized coordinates"""
        if page_num < len(self.images) and self.images[page_num] is not None:
            height, width = self.images[page_num].shape[:2]
            if width > 0 and height > 0:
                return [
                    pixel_bbox[0] / width,
                    pixel_bbox[1] / height,
                    pixel_bbox[2] / width,
                    pixel_bbox[3] / height
                ]
        return [0.0, 0.0, 1.0, 1.0]