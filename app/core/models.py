from pydantic import BaseModel, Field, ConfigDict
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
    sharpness: float = Field(default=0.5, ge=0, le=1)
    brightness: float = Field(default=0.5, ge=0, le=1)
    contrast: float = Field(default=0.5, ge=0, le=1)
    noise_level: float = Field(default=0.5, ge=0, le=1)
    overall: float = Field(default=0.5, ge=0, le=1)

class ContradictionType(str, Enum):
    CHART_TEXT_CONFLICT = "chart_text_conflict"
    NUMERIC_INCONSISTENCY = "numeric_inconsistency"
    DATE_MISMATCH = "date_mismatch"
    SIGNATURE_ABSENCE = "signature_absence"
    CALCULATION_ERROR = "calculation_error"
    SUMMARY_TABLE_MISMATCH = "summary_table_mismatch"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedField(BaseModel):
    value: Any = Field(default="")
    confidence: float = Field(default=0.5, ge=0, le=1)
    sources: List[str] = Field(default_factory=lambda: ["unknown"])
    modalities: List[str] = Field(default_factory=lambda: ["unknown"])
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VisualElement(BaseModel):
    element_type: str = Field(default="unknown")
    bbox: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    confidence: float = Field(default=0.5, ge=0, le=1)
    page_num: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Contradiction(BaseModel):
    contradiction_type: ContradictionType = Field(default=ContradictionType.CHART_TEXT_CONFLICT)
    severity: SeverityLevel = Field(default=SeverityLevel.LOW)
    field_a: str = Field(default="")
    field_b: str = Field(default="")
    value_a: Any = Field(default="")
    value_b: Any = Field(default="")
    explanation: str = Field(default="No explanation provided")
    confidence: float = Field(default=0.5, ge=0, le=1)

class ProcessingState(BaseModel):
    """State for LangGraph workflow with explicit defaults"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Required fields
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    images: List[Any] = Field(default_factory=list)
    
    # Preprocessing results
    quality_scores: Dict[int, QualityScore] = Field(default_factory=dict)
    document_type: DocumentType = Field(default=DocumentType.UNKNOWN)
    layout_strategy: Optional[str] = None
    
    # Vision results
    visual_elements: Dict[int, List[VisualElement]] = Field(default_factory=dict)
    chart_analysis: Dict[str, Any] = Field(default_factory=dict)
    table_structures: Dict[str, Any] = Field(default_factory=dict)
    signature_verification: Dict[str, Any] = Field(default_factory=dict)
    
    # Text results
    ocr_results: Dict[int, Any] = Field(default_factory=dict)
    ocr_confidence: Dict[int, float] = Field(default_factory=dict)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=lambda: {
        "dates": [], 
        "amounts": [], 
        "names": []
    })
    semantic_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Fusion results
    aligned_data: Dict[str, Any] = Field(default_factory=dict)
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    temporal_consistency: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation results
    contradictions: List[Contradiction] = Field(default_factory=list)
    risk_score: float = Field(default=0.5, ge=0, le=1)
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
    
    def ensure_required_fields(self) -> None:
        """Ensure all required fields exist with proper defaults"""
        # This method can be called to ensure state consistency
        if not self.extracted_entities:
            self.extracted_entities = {"dates": [], "amounts": [], "names": []}
        
        if not self.extracted_fields:
            # Create at least one field to prevent empty output
            from .models import ExtractedField
            self.extracted_fields = {
                "document_info": ExtractedField(
                    value={"document_id": self.document_id, "processed": True},
                    confidence=1.0,
                    sources=["system"],
                    modalities=["metadata"]
                )
            }
        
        if not self.processing_metadata:
            self.processing_metadata = {
                "document_id": self.document_id,
                "processing_start": self.processing_start.isoformat()
            }