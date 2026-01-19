from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid

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
    VISION = "vision"
    TEXT = "text"
    FUSION = "fusion"
    VALIDATION = "validation"
    RESULTS = "results"
    QUERY = "query"

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

# ========== CORE MODELS ==========
class ExtractedField(BaseModel):
    value: Any
    confidence: float = Field(..., ge=0, le=1)
    sources: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def confidence_percentage(cls, v):
        return round(v, 4)

class VisualElement(BaseModel):
    element_type: str
    bbox: List[int]
    confidence: float
    page_num: int
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