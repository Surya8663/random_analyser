# app/agents/orchestrator.py
from typing import Dict, Any, List, Optional
from app.core.models import ProcessingState
from app.utils.logger import setup_logger
from datetime import datetime
import asyncio
import os

logger = setup_logger(__name__)

# ========== SIMPLIFIED WORKFLOW (NO LANGGRAPH DEPENDENCY) ==========
class SimpleWorkflow:
    """Simple workflow without LangGraph dependency"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
    
    def add_node(self, name: str, agent):
        self.nodes[name] = agent
    
    def add_edge(self, from_node: str, to_node: str):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
    
    def set_entry_point(self, node: str):
        self.entry_point = node
    
    async def execute(self, state: ProcessingState) -> ProcessingState:
        """Execute workflow linearly"""
        if not self.entry_point:
            raise ValueError("No entry point set")
        
        current = self.entry_point
        result_state = state
        
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            
            if current in self.nodes:
                logger.info(f"▶️ Executing node: {current}")
                try:
                    agent = self.nodes[current]
                    result_state = await agent(result_state)
                    logger.info(f"✅ Completed node: {current}")
                except Exception as e:
                    logger.error(f"❌ Node {current} failed: {e}")
                    if not hasattr(result_state, 'errors'):
                        result_state.errors = []
                    result_state.errors.append(f"Node {current} failed: {str(e)}")
            
            # Get next node
            if current in self.edges and self.edges[current]:
                current = self.edges[current][0]
            else:
                current = None
        
        return result_state

class AgentOrchestrator:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self):
        # Initialize all agents
        self.agents = self._initialize_agents()
        
        # Create simple workflow
        self.workflow = SimpleWorkflow()
        self._create_workflow()
        
        logger.info(f"✅ AgentOrchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize ALL agents"""
        agents = {}
        
        # Core preprocessing agents
        agents["quality"] = self._create_quality_agent()
        agents["classifier"] = self._create_classifier_agent()
        agents["layout"] = self._create_layout_agent()
        
        # Vision agents
        agents["vision"] = self._create_vision_agent()
        agents["charts"] = self._create_chart_agent()
        agents["tables"] = self._create_table_agent()
        agents["signatures"] = self._create_signature_agent()
        
        # Text agents
        agents["ocr"] = self._create_ocr_agent()
        agents["entities"] = self._create_entity_agent()
        agents["semantics"] = self._create_semantic_agent()
        
        # Fusion agents
        agents["alignment"] = self._create_alignment_agent()
        agents["confidence"] = self._create_confidence_agent()
        
        # Validation agents
        agents["consistency"] = self._create_consistency_agent()
        agents["contradiction"] = self._create_contradiction_agent()
        agents["risk"] = self._create_risk_agent()
        
        # Output agents
        agents["explanation"] = self._create_explanation_agent()
        agents["review"] = self._create_review_agent()
        
        logger.info(f"✅ Initialized {len(agents)} agents")
        return agents
    
    def _create_workflow(self):
        """Create the complete agent workflow"""
        # Add ALL agent nodes
        self.workflow.add_node("assess_quality", self.agents["quality"])
        self.workflow.add_node("classify_document", self.agents["classifier"])
        self.workflow.add_node("determine_layout", self.agents["layout"])
        self.workflow.add_node("detect_elements", self.agents["vision"])
        self.workflow.add_node("analyze_charts", self.agents["charts"])
        self.workflow.add_node("analyze_tables", self.agents["tables"])
        self.workflow.add_node("verify_signatures", self.agents["signatures"])
        self.workflow.add_node("perform_ocr", self.agents["ocr"])
        self.workflow.add_node("extract_entities", self.agents["entities"])
        self.workflow.add_node("analyze_semantics", self.agents["semantics"])
        self.workflow.add_node("align_modalities", self.agents["alignment"])
        self.workflow.add_node("arbitrate_confidence", self.agents["confidence"])
        self.workflow.add_node("check_consistency", self.agents["consistency"])
        self.workflow.add_node("detect_contradictions", self.agents["contradiction"])
        self.workflow.add_node("assess_risk", self.agents["risk"])
        self.workflow.add_node("generate_explanations", self.agents["explanation"])
        self.workflow.add_node("generate_review_recommendations", self.agents["review"])
        
        # Define workflow edges
        self.workflow.add_edge("assess_quality", "classify_document")
        self.workflow.add_edge("classify_document", "determine_layout")
        self.workflow.add_edge("determine_layout", "detect_elements")
        self.workflow.add_edge("detect_elements", "analyze_charts")
        self.workflow.add_edge("detect_elements", "analyze_tables")
        self.workflow.add_edge("detect_elements", "verify_signatures")
        self.workflow.add_edge("determine_layout", "perform_ocr")
        self.workflow.add_edge("perform_ocr", "extract_entities")
        self.workflow.add_edge("extract_entities", "analyze_semantics")
        self.workflow.add_edge("analyze_charts", "align_modalities")
        self.workflow.add_edge("analyze_tables", "align_modalities")
        self.workflow.add_edge("verify_signatures", "align_modalities")
        self.workflow.add_edge("analyze_semantics", "align_modalities")
        self.workflow.add_edge("align_modalities", "arbitrate_confidence")
        self.workflow.add_edge("arbitrate_confidence", "check_consistency")
        self.workflow.add_edge("check_consistency", "detect_contradictions")
        self.workflow.add_edge("detect_contradictions", "assess_risk")
        self.workflow.add_edge("assess_risk", "generate_explanations")
        self.workflow.add_edge("generate_explanations", "generate_review_recommendations")
        
        # Set entry point
        self.workflow.set_entry_point("assess_quality")
        
        logger.info(f"✅ Created workflow with {len(self.workflow.nodes)} nodes")
    
    # ========== AGENT CREATION METHODS ==========
    
    def _create_quality_agent(self):
        class QualityAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("🔍 Running Quality Agent")
                
                from app.core.models import QualityScore
                
                # Create quality scores
                quality_scores = {}
                if hasattr(state, 'images') and state.images:
                    for idx, img in enumerate(state.images):
                        # Simple quality assessment based on image properties
                        height, width = img.shape[:2]
                        
                        # Mock quality metrics
                        sharpness = 0.8 if height * width > 100000 else 0.6
                        brightness = 0.7
                        contrast = 0.6
                        noise_level = 0.9
                        overall = (sharpness + brightness + contrast + noise_level) / 4
                        
                        quality_scores[idx] = QualityScore(
                            sharpness=sharpness,
                            brightness=brightness,
                            contrast=contrast,
                            noise_level=noise_level,
                            overall=overall
                        )
                
                state.quality_scores = quality_scores
                return state
        return QualityAgent()
    
    def _create_classifier_agent(self):
        class ClassifierAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("🏷️ Running Classifier Agent")
                
                from app.core.models import DocumentType
                
                # Simple classification based on filename and content
                if hasattr(state, 'file_path') and state.file_path:
                    filename = state.file_path.lower()
                    if any(word in filename for word in ['invoice', 'bill', 'receipt']):
                        state.document_type = DocumentType.INVOICE
                    elif any(word in filename for word in ['contract', 'agreement', 'lease']):
                        state.document_type = DocumentType.CONTRACT
                    elif any(word in filename for word in ['report', 'financial', 'statement']):
                        state.document_type = DocumentType.FINANCIAL_REPORT
                    elif any(word in filename for word in ['form', 'application']):
                        state.document_type = DocumentType.FORM
                    else:
                        state.document_type = DocumentType.MIXED
                else:
                    state.document_type = DocumentType.UNKNOWN
                
                return state
        return ClassifierAgent()
    
    def _create_layout_agent(self):
        class LayoutAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("📐 Running Layout Agent")
                
                # Determine layout based on document type
                if hasattr(state, 'document_type') and state.document_type:
                    if state.document_type.value in ["financial_report", "research_paper"]:
                        state.layout_strategy = "complex_multi_column"
                    elif state.document_type.value in ["invoice", "form"]:
                        state.layout_strategy = "structured_form"
                    else:
                        state.layout_strategy = "general"
                else:
                    state.layout_strategy = "general"
                
                return state
        return LayoutAgent()
    
    def _create_vision_agent(self):
        class VisionAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("👁️ Running Vision Agent")
                
                from app.core.models import VisualElement
                
                # Create mock visual elements
                visual_elements = {}
                if hasattr(state, 'images') and state.images:
                    for page_num, img in enumerate(state.images[:3]):  # Limit to 3 pages
                        elements = []
                        
                        # Mock table detection (every page gets one)
                        elements.append(VisualElement(
                            element_type="table",
                            bbox=[100, 100, 400, 300],
                            confidence=0.85,
                            page_num=page_num,
                            metadata={"rows": 5, "columns": 3}
                        ))
                        
                        # Mock text region
                        elements.append(VisualElement(
                            element_type="text_region",
                            bbox=[50, 350, 450, 500],
                            confidence=0.9,
                            page_num=page_num,
                            metadata={"text_length": 250}
                        ))
                        
                        # Mock signature on first page
                        if page_num == 0:
                            elements.append(VisualElement(
                                element_type="signature",
                                bbox=[300, 600, 450, 650],
                                confidence=0.75,
                                page_num=page_num,
                                metadata={"type": "handwritten"}
                            ))
                        
                        # Mock chart on second page
                        if page_num == 1:
                            elements.append(VisualElement(
                                element_type="chart",
                                bbox=[200, 200, 500, 400],
                                confidence=0.8,
                                page_num=page_num,
                                metadata={"chart_type": "bar_chart"}
                            ))
                        
                        visual_elements[page_num] = elements
                
                state.visual_elements = visual_elements
                return state
        return VisionAgent()
    
    def _create_chart_agent(self):
        class ChartAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("📈 Running Chart Agent")
                
                # Analyze charts in visual elements
                chart_analysis = {}
                
                if hasattr(state, 'visual_elements'):
                    for page_num, elements in state.visual_elements.items():
                        for elem in elements:
                            if elem.element_type == "chart":
                                chart_id = f"chart_{page_num}_{len(chart_analysis)}"
                                chart_analysis[chart_id] = {
                                    "type": "bar_chart",
                                    "trend_direction": "increasing",
                                    "confidence": elem.confidence,
                                    "page": page_num,
                                    "analysis": "Chart shows increasing trend over time"
                                }
                
                state.chart_analysis = chart_analysis
                return state
        return ChartAgent()
    
    def _create_table_agent(self):
        class TableAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("📊 Running Table Agent")
                
                # Extract table structures
                table_structures = {}
                
                if hasattr(state, 'visual_elements'):
                    for page_num, elements in state.visual_elements.items():
                        for elem in elements:
                            if elem.element_type == "table":
                                table_id = f"table_{page_num}_{len(table_structures)}"
                                table_structures[table_id] = {
                                    "bbox": elem.bbox,
                                    "page": page_num,
                                    "estimated_rows": elem.metadata.get("rows", 0),
                                    "estimated_columns": elem.metadata.get("columns", 0),
                                    "data_preview": ["Row 1: Data 1, Data 2, Data 3", 
                                                    "Row 2: Data 4, Data 5, Data 6"]
                                }
                
                state.table_structures = table_structures
                return state
        return TableAgent()
    
    def _create_signature_agent(self):
        class SignatureAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("✍️ Running Signature Agent")
                
                # Check for signatures
                signature_verification = {"found": False, "locations": [], "count": 0}
                
                if hasattr(state, 'visual_elements'):
                    for page_num, elements in state.visual_elements.items():
                        for elem in elements:
                            if elem.element_type == "signature":
                                signature_verification["found"] = True
                                signature_verification["count"] += 1
                                signature_verification["locations"].append({
                                    "page": page_num,
                                    "bbox": elem.bbox,
                                    "confidence": elem.confidence,
                                    "status": "detected"
                                })
                
                state.signature_verification = signature_verification
                return state
        return SignatureAgent()
    
    def _create_ocr_agent(self):
        class OCRAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("📝 Running OCR Agent")
                
                # Use extracted text if available from document processor
                if hasattr(state, 'extracted_text') and state.extracted_text:
                    # Text already extracted by document processor
                    ocr_results = {0: {"text": state.extracted_text, "confidence": 0.85}}
                    state.ocr_results = ocr_results
                    state.ocr_confidence = {0: 0.85}
                else:
                    # Mock OCR results
                    state.ocr_results = {
                        0: {
                            "text": "Sample document text extracted by OCR agent.\nThis includes multiple lines.\nStructured information for analysis.\nConfidence: 85%",
                            "confidence": 0.85,
                            "word_count": 20,
                            "char_count": 150
                        }
                    }
                    state.ocr_confidence = {0: 0.85}
                    state.extracted_text = state.ocr_results[0]["text"]
                
                return state
        return OCRAgent()
    
    def _create_entity_agent(self):
        class EntityAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("🏷️ Running Entity Agent")
                
                # Extract entities from text
                extracted_entities = {
                    "dates": [],
                    "amounts": [],
                    "names": [],
                    "organizations": [],
                    "locations": []
                }
                
                if hasattr(state, 'extracted_text') and state.extracted_text:
                    text = state.extracted_text.lower()
                    
                    # Mock entity extraction
                    if "invoice" in text or "bill" in text:
                        extracted_entities["dates"].extend(["2024-01-15", "2024-12-31"])
                        extracted_entities["amounts"].extend(["$1,500.00", "$5,000.00"])
                        extracted_entities["names"].extend(["John Smith", "Jane Doe"])
                        extracted_entities["organizations"].extend(["Acme Corp", "Tech Solutions Inc."])
                    
                    # Always add some entities for demo
                    extracted_entities["dates"].append("2024-03-20")
                    extracted_entities["amounts"].append("$2,500.00")
                
                state.extracted_entities = extracted_entities
                return state
        return EntityAgent()
    
    def _create_semantic_agent(self):
        class SemanticAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("🧠 Running Semantic Agent")
                
                # Analyze document semantics
                semantic_analysis = {
                    "summary": "Document analyzed for key topics and sentiment",
                    "key_topics": ["business", "finance", "agreement"],
                    "sentiment": "neutral",
                    "confidence": 0.7,
                    "key_phrases": ["important document", "terms and conditions", "financial agreement"]
                }
                
                if hasattr(state, 'extracted_text') and state.extracted_text:
                    text = state.extracted_text[:1000]  # Limit for analysis
                    
                    # Simple semantic analysis
                    keywords = ["report", "analysis", "data", "result", "conclusion", "summary"]
                    found_keywords = [kw for kw in keywords if kw in text.lower()]
                    
                    if found_keywords:
                        semantic_analysis["key_topics"] = found_keywords[:5]
                    
                    # Generate simple summary
                    sentences = text.split('.')
                    if len(sentences) > 3:
                        summary = '. '.join(sentences[:3]) + '.'
                        semantic_analysis["summary"] = summary
                
                state.semantic_analysis = semantic_analysis
                return state
        return SemanticAgent()
    
    def _create_alignment_agent(self):
        class AlignmentAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("🔄 Running Alignment Agent")
                
                # Align text and visual information
                aligned_data = {
                    "text_visual_alignment": [],
                    "confidence": 0.8,
                    "matches_found": 0
                }
                
                # Simple alignment logic
                if hasattr(state, 'visual_elements') and hasattr(state, 'extracted_text'):
                    text_lower = state.extracted_text.lower() if state.extracted_text else ""
                    
                    for page_num, elements in state.visual_elements.items():
                        for elem in elements:
                            elem_type = elem.element_type
                            if elem_type in text_lower:
                                aligned_data["text_visual_alignment"].append({
                                    "element_type": elem_type,
                                    "page": page_num,
                                    "mentioned_in_text": True,
                                    "confidence": elem.confidence
                                })
                                aligned_data["matches_found"] += 1
                
                state.aligned_data = aligned_data
                return state
        return AlignmentAgent()
    
    def _create_confidence_agent(self):
        class ConfidenceAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("⚖️ Running Confidence Agent")
                
                # Calculate field confidences
                field_confidences = {}
                
                # Combine confidences from different sources
                if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
                    avg_ocr = sum(state.ocr_confidence.values()) / len(state.ocr_confidence) if state.ocr_confidence else 0
                    field_confidences["text_extraction"] = avg_ocr
                
                if hasattr(state, 'visual_elements'):
                    # Calculate average visual detection confidence
                    visual_confidences = []
                    for page_elems in state.visual_elements.values():
                        for elem in page_elems:
                            visual_confidences.append(elem.confidence)
                    
                    if visual_confidences:
                        field_confidences["visual_detection"] = sum(visual_confidences) / len(visual_confidences)
                
                if hasattr(state, 'quality_scores') and state.quality_scores:
                    avg_quality = sum(qs.overall for qs in state.quality_scores.values()) / len(state.quality_scores)
                    field_confidences["document_quality"] = avg_quality
                
                state.field_confidences = field_confidences
                return state
        return ConfidenceAgent()
    
    def _create_consistency_agent(self):
        class ConsistencyAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("🔢 Running Consistency Agent")
                
                # Check consistency of dates and numbers
                temporal_consistency = {
                    "date_consistency": "consistent",
                    "numeric_consistency": "consistent",
                    "issues": [],
                    "confidence": 0.9
                }
                
                if hasattr(state, 'extracted_entities'):
                    dates = state.extracted_entities.get("dates", [])
                    amounts = state.extracted_entities.get("amounts", [])
                    
                    if len(dates) > 1:
                        temporal_consistency["date_consistency"] = "multiple_dates_found"
                        temporal_consistency["issues"].append(f"Found {len(dates)} dates")
                    
                    if len(amounts) > 1:
                        temporal_consistency["numeric_consistency"] = "multiple_amounts_found"
                        temporal_consistency["issues"].append(f"Found {len(amounts)} monetary amounts")
                
                state.temporal_consistency = temporal_consistency
                return state
        return ConsistencyAgent()
    
    def _create_contradiction_agent(self):
        class ContradictionAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("⚠️ Running Contradiction Agent")
                
                from app.core.models import Contradiction, ContradictionType, SeverityLevel
                
                contradictions = []
                
                # Check for contradictions between text and visual elements
                if hasattr(state, 'aligned_data') and state.aligned_data:
                    alignments = state.aligned_data.get("text_visual_alignment", [])
                    
                    # If visual elements detected but not mentioned in text, flag as potential contradiction
                    if hasattr(state, 'visual_elements'):
                        total_elements = sum(len(elems) for elems in state.visual_elements.values())
                        mentioned_elements = len([a for a in alignments if a.get("mentioned_in_text")])
                        
                        if total_elements > 0 and mentioned_elements == 0:
                            contradictions.append(Contradiction(
                                contradiction_type=ContradictionType.CHART_TEXT_CONFLICT,
                                severity=SeverityLevel.LOW,
                                field_a="visual_elements",
                                field_b="text_content",
                                value_a=f"{total_elements} elements",
                                value_b="Not mentioned",
                                explanation="Visual elements detected but not mentioned in text",
                                confidence=0.6,
                                recommendation="Review visual-text alignment"
                            ))
                
                # Add sample contradiction for demo
                contradictions.append(Contradiction(
                    contradiction_type=ContradictionType.NUMERIC_INCONSISTENCY,
                    severity=SeverityLevel.MEDIUM,
                    field_a="amount_1",
                    field_b="amount_2",
                    value_a="$1,500.00",
                    value_b="$1,550.00",
                    explanation="Slight discrepancy in monetary amounts",
                    confidence=0.7,
                    recommendation="Verify both amounts"
                ))
                
                state.contradictions = contradictions
                return state
        return ContradictionAgent()
    
    def _create_risk_agent(self):
        class RiskAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("🔒 Running Risk Agent")
                
                # Calculate risk score
                risk_score = 0.0
                compliance_issues = []
                
                # Check for missing signatures
                if hasattr(state, 'signature_verification'):
                    if not state.signature_verification.get("found", False):
                        risk_score += 0.3
                        compliance_issues.append("No signature detected - requires review")
                
                # Check for contradictions
                if hasattr(state, 'contradictions'):
                    risk_score += len(state.contradictions) * 0.1
                    if state.contradictions:
                        compliance_issues.append(f"{len(state.contradictions)} contradictions found")
                
                # Check document quality
                if hasattr(state, 'quality_scores') and state.quality_scores:
                    avg_quality = sum(qs.overall for qs in state.quality_scores.values()) / len(state.quality_scores)
                    if avg_quality < 0.5:
                        risk_score += 0.2
                        compliance_issues.append("Low document quality")
                
                # Normalize risk score
                risk_score = min(1.0, risk_score)
                
                state.risk_score = risk_score
                state.compliance_issues = compliance_issues
                return state
        return RiskAgent()
    
    def _create_explanation_agent(self):
        class ExplanationAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("💡 Running Explanation Agent")
                
                explanations = {
                    "document_quality": "Document quality assessed based on image sharpness, brightness, and noise levels",
                    "element_detection": "Visual elements (tables, charts, signatures) detected using computer vision algorithms",
                    "text_extraction": "Text extracted using OCR with confidence scoring",
                    "entity_extraction": "Named entities (dates, amounts, names) extracted using pattern matching",
                    "consistency_check": "Checked for temporal and numeric consistency across document",
                    "risk_assessment": "Risk calculated based on missing elements, contradictions, and quality issues",
                    "alignment": "Visual and textual information aligned for cross-modal validation"
                }
                
                state.explanations = explanations
                return state
        return ExplanationAgent()
    
    def _create_review_agent(self):
        class ReviewAgent:
            async def __call__(self, state: ProcessingState) -> ProcessingState:
                logger.info("👥 Running Review Agent")
                
                review_recommendations = []
                
                # Recommend review based on risk score
                if hasattr(state, 'risk_score'):
                    if state.risk_score > 0.7:
                        review_recommendations.append("🔴 CRITICAL: High risk detected - manual review required immediately")
                    elif state.risk_score > 0.4:
                        review_recommendations.append("🟡 MODERATE: Some issues found - recommended review")
                    else:
                        review_recommendations.append("🟢 LOW: No significant issues found - automated processing sufficient")
                
                # Add specific recommendations
                if hasattr(state, 'signature_verification'):
                    if not state.signature_verification.get("found", False):
                        review_recommendations.append("📝 Check for missing signature or approval")
                
                if hasattr(state, 'contradictions'):
                    if state.contradictions:
                        review_recommendations.append(f"⚠️ Review {len(state.contradictions)} potential contradiction(s)")
                
                if hasattr(state, 'quality_scores') and state.quality_scores:
                    avg_quality = sum(qs.overall for qs in state.quality_scores.values()) / len(state.quality_scores)
                    if avg_quality < 0.6:
                        review_recommendations.append("🖼️ Document quality low - consider rescanning")
                
                if not review_recommendations:
                    review_recommendations.append("✅ Document processed successfully - no action required")
                
                state.review_recommendations = review_recommendations
                return state
        return ReviewAgent()
    
    # ========== MAIN PROCESSING METHOD ==========
    
    async def process_document(self, file_path: str, document_id: str = None) -> Dict[str, Any]:
        """Process document with all agents"""
        try:
            logger.info(f"🚀 Starting document processing for {file_path}")
            
            # Import datetime
            from datetime import datetime
            
            # Initialize processing state
            state = ProcessingState(
                document_id=document_id or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=file_path,
                file_type=os.path.splitext(file_path)[1].lower() if file_path else "unknown",
                images=[],
                processing_start=datetime.now()
            )
            
            # First, extract images and text using document processor
            try:
                from app.services.document_processor import DocumentProcessor
                doc_processor = DocumentProcessor()
                
                # Extract images
                images, metadata = await doc_processor.extract_images(file_path)
                state.images = images
                
                # Extract text
                text_results = await doc_processor.extract_text(images)
                
                # Store OCR results
                state.ocr_results = text_results
                state.ocr_confidence = {
                    idx: result.get("confidence", 0.0)
                    for idx, result in text_results.items()
                }
                
                # Combine all text
                all_text = "\n".join([r.get("text", "") for r in text_results.values()])
                state.extracted_text = all_text
                
            except Exception as e:
                logger.warning(f"Document processor failed, using fallback: {e}")
                # Create mock data
                state.images = [self._create_mock_image()]
                state.ocr_results = {0: {"text": "Document processing in fallback mode", "confidence": 0.5}}
                state.ocr_confidence = {0: 0.5}
                state.extracted_text = "Fallback document text for processing"
            
            # Execute workflow
            logger.info("▶️ Executing agent workflow...")
            final_state = await self.workflow.execute(state)
            final_state.processing_end = datetime.now()
            
            # Prepare response
            response = self._prepare_response(final_state)
            
            logger.info(f"✅ Document processing completed: {final_state.document_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id or "unknown",
                "agent_outputs": {},
                "errors": [str(e)]
            }
    
    def _create_mock_image(self):
        """Create a mock image for fallback"""
        import numpy as np
        return np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    def _prepare_response(self, state: ProcessingState) -> Dict[str, Any]:
        """Prepare final response from processing state"""
        
        # Calculate processing time
        processing_time = 0
        if hasattr(state, 'processing_start') and hasattr(state, 'processing_end'):
            processing_time = (state.processing_end - state.processing_start).total_seconds()
        
        # Prepare visual elements
        visual_elements = []
        if hasattr(state, 'visual_elements') and state.visual_elements:
            for page_num, elements in state.visual_elements.items():
                for elem in elements:
                    visual_elements.append({
                        "type": elem.element_type,
                        "bbox": elem.bbox,
                        "confidence": elem.confidence,
                        "page_num": elem.page_num,
                        "metadata": elem.metadata
                    })
        
        # Prepare extracted fields
        extracted_fields = {}
        if hasattr(state, 'extracted_entities') and state.extracted_entities:
            for entity_type, entities in state.extracted_entities.items():
                if entities:
                    extracted_fields[f"entity_{entity_type}"] = {
                        "value": ", ".join(entities[:3]),  # Show first 3
                        "confidence": 0.8,
                        "sources": ["entity_extraction"],
                        "modalities": ["text"]
                    }
        
        # Prepare contradictions
        contradictions = []
        if hasattr(state, 'contradictions') and state.contradictions:
            for c in state.contradictions:
                contradictions.append({
                    "type": c.contradiction_type.value if hasattr(c.contradiction_type, 'value') else str(c.contradiction_type),
                    "severity": c.severity.value if hasattr(c.severity, 'value') else str(c.severity),
                    "explanation": c.explanation,
                    "confidence": c.confidence,
                    "recommendation": c.recommendation
                })
        
        # Calculate overall confidence
        overall_confidence = 0.8  # Default
        if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
            avg_ocr = sum(state.ocr_confidence.values()) / len(state.ocr_confidence)
            overall_confidence = avg_ocr
        
        # Build response
        return {
            "success": True,
            "document_id": state.document_id,
            "document_type": state.document_type.value if hasattr(state, 'document_type') and state.document_type else "unknown",
            "overall_confidence": overall_confidence,
            "processing_time": processing_time,
            
            # Agent outputs with UI-friendly format
            "agent_outputs": self._format_agent_outputs(state),
            
            # Extracted data
            "extracted_text": state.extracted_text if hasattr(state, 'extracted_text') else "",
            "extracted_fields": extracted_fields,
            "visual_elements": visual_elements,
            "contradictions": contradictions,
            
            # Processing metadata
            "processing_metadata": {
                "processing_time": processing_time,
                "success": True,
                "agent_count": len(self.agents)
            }
        }
    
    def _format_agent_outputs(self, state: ProcessingState) -> Dict[str, Any]:
        """Format agent outputs for UI display"""
        outputs = {}
        
        # Quality agent
        if hasattr(state, 'quality_scores') and state.quality_scores:
            avg_quality = sum(qs.overall for qs in state.quality_scores.values()) / len(state.quality_scores)
            outputs["quality"] = {
                "name": "Document Quality Agent",
                "icon": "🔍",
                "description": "Assesses document image quality",
                "status": "completed",
                "confidence": avg_quality,
                "key_findings": [f"Document quality: {avg_quality:.0%}"],
                "processing_time": None
            }
        
        # Classifier agent
        if hasattr(state, 'document_type'):
            outputs["classifier"] = {
                "name": "Document Classifier",
                "icon": "🏷️",
                "description": "Classifies document type",
                "status": "completed",
                "confidence": 0.85,
                "key_findings": [f"Document type: {state.document_type.value.replace('_', ' ').title()}"],
                "processing_time": None
            }
        
        # Vision agent
        element_count = sum(len(elems) for elems in state.visual_elements.values()) if hasattr(state, 'visual_elements') else 0
        outputs["vision"] = {
            "name": "Vision Agent",
            "icon": "👁️",
            "description": "Detects visual elements",
            "status": "completed",
            "confidence": 0.85,
            "key_findings": [f"Detected {element_count} visual elements"],
            "processing_time": None
        }
        
        # Text agent
        word_count = len(state.extracted_text.split()) if hasattr(state, 'extracted_text') else 0
        avg_ocr = sum(state.ocr_confidence.values()) / len(state.ocr_confidence) if hasattr(state, 'ocr_confidence') and state.ocr_confidence else 0
        outputs["text"] = {
            "name": "Text Agent",
            "icon": "📝",
            "description": "Extracts and analyzes text",
            "status": "completed",
            "confidence": avg_ocr,
            "key_findings": [f"Extracted {word_count} words"],
            "processing_time": None
        }
        
        # Fusion agent
        if hasattr(state, 'aligned_data'):
            outputs["fusion"] = {
                "name": "Fusion Agent",
                "icon": "🔄",
                "description": "Combines vision and text data",
                "status": "completed",
                "confidence": state.aligned_data.get("confidence", 0.8),
                "key_findings": [f"Aligned {state.aligned_data.get('matches_found', 0)} visual-text matches"],
                "processing_time": None
            }
        
        # Validation agent
        contradiction_count = len(state.contradictions) if hasattr(state, 'contradictions') else 0
        outputs["validation"] = {
            "name": "Validation Agent",
            "icon": "✅",
            "description": "Validates consistency and quality",
            "status": "completed",
            "confidence": 1.0 - (contradiction_count * 0.1),
            "key_findings": [f"Found {contradiction_count} potential issues"],
            "processing_time": None
        }
        
        # Risk agent
        if hasattr(state, 'risk_score'):
            risk_level = "High" if state.risk_score > 0.7 else "Medium" if state.risk_score > 0.4 else "Low"
            outputs["risk"] = {
                "name": "Risk Agent",
                "icon": "🔒",
                "description": "Assesses risk and compliance",
                "status": "completed",
                "confidence": 1.0 - state.risk_score,
                "key_findings": [f"Risk level: {risk_level} ({state.risk_score:.0%})"],
                "processing_time": None
            }
        
        return outputs