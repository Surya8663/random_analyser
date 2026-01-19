from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid

class DocumentType(str, Enum):
    FINANCIAL_REPORT = "financial_report"
    INVOICE = "invoice"
    CONTRACT = "contract"
    FORM = "form"
    RESEARCH_PAPER = "research_paper"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class QualityScore(BaseModel):
    sharpness: float = Field(..., ge=0, le=1, description="Image sharpness score")
    brightness: float = Field(..., ge=0, le=1, description="Image brightness score")
    contrast: float = Field(..., ge=0, le=1, description="Image contrast score")
    noise_level: float = Field(..., ge=0, le=1, description="Image noise level (lower is better)")
    overall: float = Field(..., ge=0, le=1, description="Overall quality score")
    
    @validator('overall')
    def validate_overall(cls, v, values):
        """Ensure overall score is consistent with other scores"""
        if 'sharpness' in values and 'brightness' in values and 'contrast' in values:
            avg = (values['sharpness'] + values['brightness'] + values['contrast'] + (1 - values.get('noise_level', 0))) / 4
            if abs(v - avg) > 0.2:
                return avg
        return v

class ContradictionType(str, Enum):
    CHART_TEXT_CONFLICT = "chart_text_conflict"
    NUMERIC_INCONSISTENCY = "numeric_inconsistency"
    DATE_MISMATCH = "date_mismatch"
    SIGNATURE_ABSENCE = "signature_absence"
    CALCULATION_ERROR = "calculation_error"
    SUMMARY_TABLE_MISMATCH = "summary_table_mismatch"
    VISUAL_TEXT_MISMATCH = "visual_text_mismatch"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedField(BaseModel):
    value: Any = Field(..., description="Extracted field value")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for extraction")
    sources: List[str] = Field(default_factory=list, description="Sources that contributed to this field")
    modalities: List[str] = Field(default_factory=list, description="Modalities used (text, visual, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Round confidence to 2 decimal places"""
        return round(v, 2)

class VisualElement(BaseModel):
    element_type: str = Field(..., description="Type of visual element")
    bbox: List[int] = Field(..., min_items=4, max_items=4, description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    page_num: int = Field(..., ge=0, description="Page number (0-indexed)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional element metadata")
    
    @validator('bbox')
    def validate_bbox(cls, v):
        """Ensure bounding box coordinates are valid"""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        if v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("Invalid bounding box coordinates")
        return v

class Contradiction(BaseModel):
    contradiction_type: ContradictionType = Field(..., description="Type of contradiction")
    severity: SeverityLevel = Field(..., description="Severity level")
    field_a: str = Field(..., description="First field involved")
    field_b: str = Field(..., description="Second field involved")
    value_a: Any = Field(..., description="Value of first field")
    value_b: Any = Field(..., description="Value of second field")
    explanation: str = Field(..., description="Explanation of the contradiction")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in contradiction detection")
    
    @validator('explanation')
    def validate_explanation(cls, v):
        """Ensure explanation is not empty"""
        if not v or not v.strip():
            raise ValueError("Explanation cannot be empty")
        return v.strip()

class ProcessingState(BaseModel):
    """State for LangGraph workflow"""
    # Document identification
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    file_path: Optional[str] = Field(None, description="Path to original file")
    file_type: Optional[str] = Field(None, description="File type/extension")
    images: List[Any] = Field(default_factory=list, description="Document images for processing")
    
    # Preprocessing results
    quality_scores: Dict[int, QualityScore] = Field(default_factory=dict, description="Quality scores per page")
    document_type: Optional[DocumentType] = Field(None, description="Document classification")
    layout_strategy: Optional[str] = Field(None, description="Layout processing strategy")
    
    # Vision results
    visual_elements: Dict[int, List[VisualElement]] = Field(default_factory=dict, description="Detected visual elements per page")
    chart_analysis: Dict[str, Any] = Field(default_factory=dict, description="Chart analysis results")
    table_structures: Dict[str, Any] = Field(default_factory=dict, description="Table structure analysis")
    signature_verification: Dict[str, Any] = Field(default_factory=dict, description="Signature verification results")
    
    # Text results
    extracted_text: Optional[str] = Field(None, description="Full extracted text content")
    ocr_results: Dict[int, Any] = Field(default_factory=dict, description="OCR results per page")
    ocr_confidence: Dict[int, float] = Field(default_factory=dict, description="OCR confidence scores per page")
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted named entities")
    semantic_analysis: Dict[str, Any] = Field(default_factory=dict, description="Semantic analysis results")
    
    # Fusion results
    aligned_data: Dict[str, Any] = Field(default_factory=dict, description="Cross-modal alignment results")
    field_confidences: Dict[str, float] = Field(default_factory=dict, description="Field-level confidence scores")
    temporal_consistency: Dict[str, Any] = Field(default_factory=dict, description="Temporal consistency checks")
    
    # Validation results
    contradictions: List[Contradiction] = Field(default_factory=list, description="Detected contradictions")
    risk_score: float = Field(default=0.0, ge=0, le=1, description="Overall risk score")
    compliance_issues: List[str] = Field(default_factory=list, description="Compliance issues found")
    
    # Explainability results
    explanations: Dict[str, str] = Field(default_factory=dict, description="Processing explanations")
    review_recommendations: List[str] = Field(default_factory=list, description="Human review recommendations")
    
    # Final output
    extracted_fields: Dict[str, ExtractedField] = Field(default_factory=dict, description="Final extracted fields")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    
    # Timing
    processing_start: datetime = Field(default_factory=datetime.now, description="Processing start time")
    processing_end: Optional[datetime] = Field(None, description="Processing end time")
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @validator('risk_score')
    def validate_risk_score(cls, v):
        """Round risk score to 2 decimal places"""
        return round(v, 2)
    
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate total processing time in seconds"""
        if self.processing_end:
            return (self.processing_end - self.processing_start).total_seconds()
        return None
    
    def add_error(self, error: str):
        """Add an error to the error list"""
        self.errors.append(error)
    
    def add_extracted_field(self, name: str, value: Any, confidence: float, 
                          sources: List[str], modalities: List[str], metadata: Dict[str, Any] = None):
        """Add an extracted field"""
        self.extracted_fields[name] = ExtractedField(
            value=value,
            confidence=confidence,
            sources=sources,
            modalities=modalities,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value if self.document_type else None,
            "processing_time": self.processing_time,
            "extracted_fields_count": len(self.extracted_fields),
            "errors_count": len(self.errors),
            "risk_score": self.risk_score
        }