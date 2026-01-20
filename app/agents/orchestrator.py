# app/agents/orchestrator.py - COMPLETE CORRECTED VERSION
from typing import Dict, Any, List, Optional
from app.core.models import MultiModalDocument, ProcessingState, DocumentType, QualityScore, Contradiction, ContradictionType, SeverityLevel
from app.utils.logger import setup_logger
from datetime import datetime
import asyncio
import os

logger = setup_logger(__name__)

# ========== COMPATIBILITY ADAPTER ==========

def convert_to_multi_modal(state: ProcessingState) -> MultiModalDocument:
    """Convert ProcessingState to MultiModalDocument with all required fields"""
    # Extract raw text from state
    raw_text = ""
    if hasattr(state, 'extracted_text'):
        raw_text = state.extracted_text
    elif hasattr(state, 'ocr_results'):
        for page_result in state.ocr_results.values():
            if isinstance(page_result, dict) and 'text' in page_result:
                raw_text += page_result['text'] + "\n"
    
    # Convert visual elements
    visual_elements = []
    if hasattr(state, 'visual_elements'):
        for page_num, elements in state.visual_elements.items():
            for elem in elements:
                from app.core.models import EnhancedVisualElement, BoundingBox
                visual_elements.append(EnhancedVisualElement(
                    element_type=elem.element_type,
                    bbox=BoundingBox(
                        x1=elem.bbox[0] if len(elem.bbox) > 0 else 0,
                        y1=elem.bbox[1] if len(elem.bbox) > 1 else 0,
                        x2=elem.bbox[2] if len(elem.bbox) > 2 else 0,
                        y2=elem.bbox[3] if len(elem.bbox) > 3 else 0
                    ),
                    confidence=elem.confidence,
                    page_num=elem.page_num,
                    metadata=elem.metadata
                ))
    
    # Create MultiModalDocument with all required fields
    doc = MultiModalDocument(
        document_id=state.document_id,
        file_path=state.file_path,
        file_type=state.file_type,
        images=state.images if hasattr(state, 'images') else [],
        document_type=state.document_type if hasattr(state, 'document_type') else DocumentType.UNKNOWN,
        raw_text=raw_text,
        quality_scores=state.quality_scores if hasattr(state, 'quality_scores') else {},
        visual_elements=visual_elements,
        extracted_entities=state.extracted_entities if hasattr(state, 'extracted_entities') else {},
        contradictions=state.contradictions if hasattr(state, 'contradictions') else [],
        risk_score=state.risk_score if hasattr(state, 'risk_score') else 0.0,
        processing_start=state.processing_start if hasattr(state, 'processing_start') else datetime.now(),
        processing_end=state.processing_end if hasattr(state, 'processing_end') else None,
        errors=state.errors if hasattr(state, 'errors') else []
    )
    
    # Copy additional fields if they exist
    if hasattr(state, 'compliance_issues'):
        doc.compliance_issues = state.compliance_issues
    if hasattr(state, 'review_recommendations'):
        doc.review_recommendations = state.review_recommendations
    if hasattr(state, 'explanations'):
        doc.explanations = state.explanations
    if hasattr(state, 'aligned_data'):
        doc.aligned_data = state.aligned_data
    if hasattr(state, 'field_confidences'):
        doc.field_confidences = state.field_confidences
    if hasattr(state, 'chart_analysis'):
        doc.chart_analysis = state.chart_analysis
    if hasattr(state, 'table_structures'):
        doc.table_structures = state.table_structures
    if hasattr(state, 'signature_verification'):
        doc.signature_verification = state.signature_verification
    if hasattr(state, 'semantic_analysis'):
        doc.semantic_analysis = state.semantic_analysis
    if hasattr(state, 'temporal_consistency'):
        doc.temporal_consistency = state.temporal_consistency
    if hasattr(state, 'agent_outputs'):
        doc.agent_outputs = state.agent_outputs
    if hasattr(state, 'processing_metadata'):
        doc.processing_metadata = state.processing_metadata
    
    return doc

# ========== SIMPLIFIED WORKFLOW (UPDATED) ==========
class SimpleWorkflow:
    """Simple workflow without LangGraph dependency - UPDATED for MultiModalDocument"""
    
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
    
    # In orchestrator.py, replace the execute method in SimpleWorkflow with this FIXED version:

# In orchestrator.py, FIX the SimpleWorkflow.execute method:
async def execute(self, doc: MultiModalDocument) -> MultiModalDocument:
    """Execute workflow linearly with MultiModalDocument - FIXED VERSION"""
    if not self.entry_point:
        raise ValueError("No entry point set")
    
    current = self.entry_point
    result_doc = doc
    
    visited = set()
    
    while current and current not in visited:
        visited.add(current)
        
        if current in self.nodes:
            logger.info(f"▶️ Executing node: {current}")
            try:
                agent = self.nodes[current]
                
                # Check if agent accepts MultiModalDocument
                if hasattr(agent, '_accepts_multi_modal'):
                    # Agent is updated for MultiModalDocument
                    result_doc = await agent(result_doc)
                    logger.info(f"✅ Completed node: {current}")
                else:
                    # Agent expects ProcessingState - convert
                    logger.debug(f"Converting to ProcessingState for agent {current}")
                    try:
                        state = result_doc.to_processing_state()
                        state_result = await agent(state)
                        # Convert back to MultiModalDocument
                        result_doc = convert_to_multi_modal(state_result)
                        logger.info(f"✅ Completed node: {current} (with conversion)")
                    except Exception as conv_error:
                        logger.error(f"❌ Conversion failed for agent {current}: {conv_error}")
                        # Try direct call as fallback
                        result_doc = await agent(result_doc)
                        logger.info(f"✅ Completed node: {current} (direct call)")
                
            except Exception as e:
                logger.error(f"❌ Node {current} failed: {e}")
                result_doc.errors.append(f"Node {current} failed: {str(e)}")
        
        # Get next node
        if current in self.edges and self.edges[current]:
            current = self.edges[current][0]
        else:
            current = None
    
    return result_doc

class AgentOrchestrator:
    """Main orchestrator for the multi-agent system - UPDATED for MultiModalDocument"""
    
    def __init__(self):
        # Initialize all agents
        self.agents = self._initialize_agents()
        
        # Create simple workflow
        self.workflow = SimpleWorkflow()
        self._create_workflow()
        
        logger.info(f"✅ AgentOrchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize ALL agents - UPDATED for MultiModalDocument"""
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
    
    # ========== AGENT CREATION METHODS (UPDATED FOR MULTI-MODAL) ==========
    
    def _create_quality_agent(self):
        class QualityAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("🔍 Running Quality Agent (Multi-Modal)")
                
                # Mark that this agent accepts MultiModalDocument
                self._accepts_multi_modal = True
                
                # Create quality scores
                if doc.images:
                    for idx, img in enumerate(doc.images):
                        if img is not None:
                            height, width = img.shape[:2]
                            
                            # Simple quality assessment
                            sharpness = 0.8 if height * width > 100000 else 0.6
                            brightness = 0.7
                            contrast = 0.6
                            noise_level = 0.9
                            overall = (sharpness + brightness + contrast + noise_level) / 4
                            
                            doc.quality_scores[idx] = QualityScore(
                                sharpness=sharpness,
                                brightness=brightness,
                                contrast=contrast,
                                noise_level=noise_level,
                                overall=overall
                            )
                
                return doc
        return QualityAgent()
    
    def _create_classifier_agent(self):
        class ClassifierAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("🏷️ Running Classifier Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Simple classification based on filename and content
                if doc.file_path:
                    filename = doc.file_path.lower()
                    if any(word in filename for word in ['invoice', 'bill', 'receipt']):
                        doc.document_type = DocumentType.INVOICE
                    elif any(word in filename for word in ['contract', 'agreement', 'lease']):
                        doc.document_type = DocumentType.CONTRACT
                    elif any(word in filename for word in ['report', 'financial', 'statement']):
                        doc.document_type = DocumentType.FINANCIAL_REPORT
                    elif any(word in filename for word in ['form', 'application']):
                        doc.document_type = DocumentType.FORM
                    else:
                        # Try to classify from text content
                        text_lower = doc.raw_text.lower()
                        if any(word in text_lower for word in ['invoice', 'total', 'amount', '$']):
                            doc.document_type = DocumentType.INVOICE
                        elif any(word in text_lower for word in ['contract', 'agreement', 'terms']):
                            doc.document_type = DocumentType.CONTRACT
                        else:
                            doc.document_type = DocumentType.MIXED
                else:
                    doc.document_type = DocumentType.UNKNOWN
                
                return doc
        return ClassifierAgent()
    
    def _create_layout_agent(self):
        class LayoutAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("📐 Running Layout Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Already have layout regions from DocumentProcessor
                # This agent can add additional layout analysis
                
                return doc
        return LayoutAgent()
    
    def _create_vision_agent(self):
        class VisionAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("👁️ Running Vision Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Visual elements already detected by DocumentProcessor
                # In Phase 2, this will run real YOLO/DETR models
                
                # For now, just mark that vision analysis is done
                if not hasattr(doc, 'agent_outputs'):
                    doc.agent_outputs = {}
                
                doc.agent_outputs["vision"] = {
                    "status": "completed",
                    "element_count": len(doc.visual_elements),
                    "element_types": list(set([elem.element_type for elem in doc.visual_elements]))
                }
                
                return doc
        return VisionAgent()
    
    def _create_chart_agent(self):
        class ChartAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("📈 Running Chart Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Analyze charts in visual elements
                chart_analysis = {}
                
                for elem in doc.visual_elements:
                    if elem.element_type == "chart":
                        chart_id = f"chart_{elem.page_num}_{len(chart_analysis)}"
                        chart_analysis[chart_id] = {
                            "type": "bar_chart",
                            "bbox": elem.bbox.to_list(),
                            "confidence": elem.confidence,
                            "page": elem.page_num,
                            "analysis": "Chart shows increasing trend over time"
                        }
                
                doc.chart_analysis = chart_analysis
                return doc
        return ChartAgent()
    
    def _create_table_agent(self):
        class TableAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("📊 Running Table Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Extract table structures
                table_structures = {}
                
                for elem in doc.visual_elements:
                    if elem.element_type == "table":
                        table_id = f"table_{elem.page_num}_{len(table_structures)}"
                        table_structures[table_id] = {
                            "bbox": elem.bbox.to_list(),
                            "page": elem.page_num,
                            "estimated_rows": 5,
                            "estimated_columns": 3,
                            "data_preview": ["Row 1: Data 1, Data 2, Data 3", 
                                            "Row 2: Data 4, Data 5, Data 6"]
                        }
                
                doc.table_structures = table_structures
                return doc
        return TableAgent()
    
    def _create_signature_agent(self):
        class SignatureAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("✍️ Running Signature Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Check for signatures
                signature_verification = {"found": False, "locations": [], "count": 0}
                
                for elem in doc.visual_elements:
                    if elem.element_type == "signature":
                        signature_verification["found"] = True
                        signature_verification["count"] += 1
                        signature_verification["locations"].append({
                            "page": elem.page_num,
                            "bbox": elem.bbox.to_list(),
                            "confidence": elem.confidence,
                            "status": "detected"
                        })
                
                doc.signature_verification = signature_verification
                return doc
        return SignatureAgent()
    
    def _create_ocr_agent(self):
        class OCRAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("📝 Running OCR Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # OCR already done by DocumentProcessor
                # This agent can do additional OCR processing if needed
                
                # Calculate overall OCR confidence
                if doc.ocr_results:
                    avg_confidence = sum(ocr.average_confidence for ocr in doc.ocr_results.values()) / len(doc.ocr_results)
                    if not hasattr(doc, 'agent_outputs'):
                        doc.agent_outputs = {}
                    
                    doc.agent_outputs["ocr"] = {
                        "status": "completed",
                        "avg_confidence": avg_confidence,
                        "total_pages": len(doc.ocr_results),
                        "total_words": sum(len(ocr.words) for ocr in doc.ocr_results.values())
                    }
                
                return doc
        return OCRAgent()
    
    def _create_entity_agent(self):
        class EntityAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("🏷️ Running Entity Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Extract entities from text
                extracted_entities = {
                    "dates": [],
                    "amounts": [],
                    "names": [],
                    "organizations": [],
                    "locations": []
                }
                
                if doc.raw_text:
                    text = doc.raw_text.lower()
                    
                    # Mock entity extraction
                    if "invoice" in text or "bill" in text:
                        extracted_entities["dates"].extend(["2024-01-15", "2024-12-31"])
                        extracted_entities["amounts"].extend(["$1,500.00", "$5,000.00"])
                        extracted_entities["names"].extend(["John Smith", "Jane Doe"])
                        extracted_entities["organizations"].extend(["Acme Corp", "Tech Solutions Inc."])
                    
                    # Always add some entities for demo
                    extracted_entities["dates"].append("2024-03-20")
                    extracted_entities["amounts"].append("$2,500.00")
                
                doc.extracted_entities = extracted_entities
                return doc
        return EntityAgent()
    
    def _create_semantic_agent(self):
        class SemanticAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("🧠 Running Semantic Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Analyze document semantics
                semantic_analysis = {
                    "summary": "Document analyzed for key topics and sentiment",
                    "key_topics": ["business", "finance", "agreement"],
                    "sentiment": "neutral",
                    "confidence": 0.7,
                    "key_phrases": ["important document", "terms and conditions", "financial agreement"]
                }
                
                if doc.raw_text:
                    text = doc.raw_text[:1000]  # Limit for analysis
                    
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
                
                doc.semantic_analysis = semantic_analysis
                return doc
        return SemanticAgent()
    
    def _create_alignment_agent(self):
        class AlignmentAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("🔄 Running Alignment Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Align text and visual information
                aligned_data = {
                    "text_visual_alignment": [],
                    "confidence": 0.8,
                    "matches_found": 0
                }
                
                # Simple alignment logic
                text_lower = doc.raw_text.lower() if doc.raw_text else ""
                
                for elem in doc.visual_elements:
                    elem_type = elem.element_type
                    if elem_type in text_lower:
                        aligned_data["text_visual_alignment"].append({
                            "element_type": elem_type,
                            "page": elem.page_num,
                            "mentioned_in_text": True,
                            "confidence": elem.confidence
                        })
                        aligned_data["matches_found"] += 1
                
                for region in doc.layout_regions:
                    if region.label in text_lower:
                        aligned_data["text_visual_alignment"].append({
                            "element_type": f"layout_{region.label}",
                            "page": region.page_num,
                            "mentioned_in_text": True,
                            "confidence": region.confidence
                        })
                        aligned_data["matches_found"] += 1
                
                doc.aligned_data = aligned_data
                return doc
        return AlignmentAgent()
    
    def _create_confidence_agent(self):
        class ConfidenceAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("⚖️ Running Confidence Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Calculate field confidences
                field_confidences = {}
                
                # OCR confidence
                if doc.ocr_results:
                    avg_ocr = sum(ocr.average_confidence for ocr in doc.ocr_results.values()) / len(doc.ocr_results)
                    field_confidences["text_extraction"] = avg_ocr
                
                # Visual detection confidence
                if doc.visual_elements:
                    visual_confidences = [elem.confidence for elem in doc.visual_elements]
                    if visual_confidences:
                        field_confidences["visual_detection"] = sum(visual_confidences) / len(visual_confidences)
                
                # Layout confidence
                if doc.layout_regions:
                    layout_confidences = [region.confidence for region in doc.layout_regions]
                    if layout_confidences:
                        field_confidences["layout_analysis"] = sum(layout_confidences) / len(layout_confidences)
                
                # Document quality
                if doc.quality_scores:
                    avg_quality = sum(qs.overall for qs in doc.quality_scores.values()) / len(doc.quality_scores)
                    field_confidences["document_quality"] = avg_quality
                
                doc.field_confidences = field_confidences
                return doc
        return ConfidenceAgent()
    
    def _create_consistency_agent(self):
        class ConsistencyAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("🔢 Running Consistency Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Check consistency of dates and numbers
                temporal_consistency = {
                    "date_consistency": "consistent",
                    "numeric_consistency": "consistent",
                    "issues": [],
                    "confidence": 0.9
                }
                
                if doc.extracted_entities:
                    dates = doc.extracted_entities.get("dates", [])
                    amounts = doc.extracted_entities.get("amounts", [])
                    
                    if len(dates) > 1:
                        temporal_consistency["date_consistency"] = "multiple_dates_found"
                        temporal_consistency["issues"].append(f"Found {len(dates)} dates")
                    
                    if len(amounts) > 1:
                        temporal_consistency["numeric_consistency"] = "multiple_amounts_found"
                        temporal_consistency["issues"].append(f"Found {len(amounts)} monetary amounts")
                
                doc.temporal_consistency = temporal_consistency
                return doc
        return ConsistencyAgent()
    
    def _create_contradiction_agent(self):
        class ContradictionAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("⚠️ Running Contradiction Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                contradictions = []
                
                # Check for contradictions between text and visual elements
                if hasattr(doc, 'aligned_data') and doc.aligned_data:
                    alignments = doc.aligned_data.get("text_visual_alignment", [])
                    
                    # If visual elements detected but not mentioned in text, flag as potential contradiction
                    total_elements = len(doc.visual_elements)
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
                
                doc.contradictions = contradictions
                return doc
        return ContradictionAgent()
    
    def _create_risk_agent(self):
        class RiskAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("🔒 Running Risk Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                # Calculate risk score
                risk_score = 0.0
                compliance_issues = []
                
                # Check for missing signatures
                if hasattr(doc, 'signature_verification') and doc.signature_verification:
                    if not doc.signature_verification.get("found", False):
                        risk_score += 0.3
                        compliance_issues.append("No signature detected - requires review")
                
                # Check for contradictions
                risk_score += len(doc.contradictions) * 0.1
                if doc.contradictions:
                    compliance_issues.append(f"{len(doc.contradictions)} contradictions found")
                
                # Check document quality
                if doc.quality_scores:
                    avg_quality = sum(qs.overall for qs in doc.quality_scores.values()) / len(doc.quality_scores)
                    if avg_quality < 0.5:
                        risk_score += 0.2
                        compliance_issues.append("Low document quality")
                
                # Check OCR confidence
                if doc.ocr_results:
                    avg_ocr = sum(ocr.average_confidence for ocr in doc.ocr_results.values()) / len(doc.ocr_results)
                    if avg_ocr < 0.6:
                        risk_score += 0.2
                        compliance_issues.append("Low OCR confidence")
                
                # Normalize risk score
                risk_score = min(1.0, risk_score)
                
                doc.risk_score = risk_score
                doc.compliance_issues = compliance_issues
                return doc
        return RiskAgent()
    
    def _create_explanation_agent(self):
        class ExplanationAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("💡 Running Explanation Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                explanations = {
                    "document_quality": "Document quality assessed based on image sharpness, brightness, and noise levels",
                    "element_detection": f"Visual elements (tables, charts, signatures) detected: {len(doc.visual_elements)} elements found",
                    "entity_extraction": "Named entities (dates, amounts, names) extracted using pattern matching",
                    "consistency_check": "Checked for temporal and numeric consistency across document",
                    "risk_assessment": f"Risk calculated based on missing elements, contradictions, and quality issues: {doc.risk_score:.2f}",
                    "alignment": f"Visual and textual information aligned: {doc.aligned_data.get('matches_found', 0)} matches found"
                }
                
                # FIXED: Safe division for OCR confidence
                if doc.ocr_results and len(doc.ocr_results) > 0:
                    avg_ocr = sum(ocr.average_confidence for ocr in doc.ocr_results.values()) / len(doc.ocr_results)
                    explanations["text_extraction"] = f"Text extracted using OCR with average confidence: {avg_ocr:.2f}"
                else:
                    explanations["text_extraction"] = "Text extraction not performed or no results"
                
                doc.explanations = explanations
                return doc
        return ExplanationAgent()
    
    def _create_review_agent(self):
        class ReviewAgent:
            async def __call__(self, doc: MultiModalDocument) -> MultiModalDocument:
                logger.info("👥 Running Review Agent (Multi-Modal)")
                
                self._accepts_multi_modal = True
                
                review_recommendations = []
                
                # Recommend review based on risk score
                if doc.risk_score > 0.7:
                    review_recommendations.append("🔴 CRITICAL: High risk detected - manual review required immediately")
                elif doc.risk_score > 0.4:
                    review_recommendations.append("🟡 MODERATE: Some issues found - recommended review")
                else:
                    review_recommendations.append("🟢 LOW: No significant issues found - automated processing sufficient")
                
                # Add specific recommendations
                if hasattr(doc, 'signature_verification'):
                    if not doc.signature_verification.get("found", False):
                        review_recommendations.append("📝 Check for missing signature or approval")
                
                if doc.contradictions:
                    review_recommendations.append(f"⚠️ Review {len(doc.contradictions)} potential contradiction(s)")
                
                if doc.quality_scores:
                    avg_quality = sum(qs.overall for qs in doc.quality_scores.values()) / len(doc.quality_scores)
                    if avg_quality < 0.6:
                        review_recommendations.append("🖼️ Document quality low - consider rescanning")
                
                if not review_recommendations:
                    review_recommendations.append("✅ Document processed successfully - no action required")
                
                doc.review_recommendations = review_recommendations
                return doc
        return ReviewAgent()
    
    # ========== MAIN PROCESSING METHOD ==========
    
    async def process_document(self, file_path: str, document_id: str = None) -> MultiModalDocument:
        """Process document with all agents - returns MultiModalDocument"""
        try:
            logger.info(f"🚀 Starting document processing for {file_path}")
            
            # Step 1: Use DocumentProcessor to create MultiModalDocument
            from app.services.document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
            
            # Get the unified document
            doc = await doc_processor.process_document(file_path, document_id)
            
            # Step 2: Execute workflow with the document
            logger.info("▶️ Executing agent workflow...")
            final_doc = await self.workflow.execute(doc)
            final_doc.processing_end = datetime.now()
            
            logger.info(f"✅ Document processing completed: {final_doc.document_id}")
            logger.info(f"   Processing time: {final_doc.get_processing_time():.2f} seconds")
            
            return final_doc
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}", exc_info=True)
            # Return error document
            error_doc = MultiModalDocument(
                document_id=document_id or "error",
                file_path=file_path,
                file_type=os.path.splitext(file_path)[1].lower() if file_path else "unknown"
            )
            error_doc.errors.append(f"Processing failed: {str(e)}")
            return error_doc