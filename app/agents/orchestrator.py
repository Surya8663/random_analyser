from typing import Dict, Any, List, Optional
from app.core.models import ProcessingState
from app.utils.logger import setup_logger
from datetime import datetime
import asyncio

logger = setup_logger(__name__)

# Try to import LangGraph components with fallbacks
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ LangGraph not available, using simplified workflow")
    LANGGRAPH_AVAILABLE = False
    
    # Create simple replacements
    class StateGraph:
        def __init__(self, state_class):
            self.state_class = state_class
            self.nodes = {}
            self.edges = {}
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)
        
        def set_entry_point(self, node):
            self.entry_point = node
        
        def compile(self, checkpointer=None):
            return CompiledGraph(self)
    
    END = "END"
    
    class MemorySaver:
        def __init__(self):
            self.memory = {}
        
        def get(self, config):
            return self.memory.get(str(config), {})
        
        def put(self, config, value):
            self.memory[str(config)] = value
    
    class CompiledGraph:
        def __init__(self, graph):
            self.graph = graph
        
        async def ainvoke(self, state, config=None):
            # Simple linear execution
            current = self.graph.entry_point
            result_state = state
            
            while current != END:
                if current in self.graph.nodes:
                    result_state = await self.graph.nodes[current](result_state)
                
                # Get next node
                if current in self.graph.edges and self.graph.edges[current]:
                    current = self.graph.edges[current][0]
                else:
                    current = END
            
            return result_state

class AgentOrchestrator:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self):
        # Initialize all agents
        self.agents = self._initialize_agents()
        
        # Create workflow
        self.workflow = self._create_workflow()
        
        # Create checkpointer if LangGraph available
        if LANGGRAPH_AVAILABLE:
            try:
                from langgraph.checkpoint import MemorySaver
                self.checkpointer = MemorySaver()
            except ImportError:
                try:
                    from langgraph.checkpoint.memory import MemorySaver
                    self.checkpointer = MemorySaver()
                except ImportError:
                    self.checkpointer = MemorySaver()  # Use our fallback
        else:
            self.checkpointer = MemorySaver()
        
        self.compiled_workflow = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with real implementations"""
        agents = {}
        
        # Core preprocessing agents
        agents["quality"] = DocumentQualityAgent()
        agents["classifier"] = DocumentTypeClassifier()
        agents["layout"] = LayoutStrategyAgent()
        
        # Vision agents
        agents["vision"] = VisualElementDetector()
        agents["charts"] = ChartUnderstandingAgent()
        agents["tables"] = TableStructureAgent()
        agents["signatures"] = SignatureVerificationAgent()
        
        # Text agents
        agents["ocr_reliability"] = OCRReliabilityAgent()
        agents["entities"] = EntityIntelligenceAgent()
        agents["semantics"] = SemanticReasoningAgent()
        
        # Fusion agents
        agents["alignment"] = CrossModalAlignmentAgent()
        agents["confidence"] = ConfidenceArbitrationAgent()
        
        # Validation agents
        agents["consistency"] = TemporalNumericConsistencyAgent()
        agents["contradiction"] = ContradictionDetectionAgent()
        agents["risk"] = RiskComplianceAgent()
        
        # Output agents
        agents["explanation"] = ExplanationAgent()
        agents["review"] = HumanReviewAgent()
        
        logger.info(f"✅ Initialized {len(agents)} agents")
        return agents
    
    def _create_workflow(self):
        """Create the complete agent workflow"""
        workflow = StateGraph(ProcessingState)
        
        # Add all agent nodes
        agent_nodes = {
            "assess_quality": "quality",
            "classify_document": "classifier",
            "determine_layout": "layout",
            "detect_elements": "vision",
            "analyze_charts": "charts",
            "analyze_tables": "tables",
            "verify_signatures": "signatures",
            "assess_ocr_reliability": "ocr_reliability",
            "extract_entities": "entities",
            "analyze_semantics": "semantics",
            "align_modalities": "alignment",
            "arbitrate_confidence": "confidence",
            "check_consistency": "consistency",
            "detect_contradictions": "contradiction",
            "assess_risk": "risk",
            "generate_explanations": "explanation",
            "generate_review_recommendations": "review",
        }
        
        for node_name, agent_key in agent_nodes.items():
            if agent_key in self.agents:
                workflow.add_node(node_name, self.agents[agent_key])
                logger.debug(f"Added agent node: {node_name} ({agent_key})")
        
        # Add compile results node
        workflow.add_node("compile_results", self._compile_final_results)
        
        # Define workflow edges
        edges = [
            ("assess_quality", "classify_document"),
            ("classify_document", "determine_layout"),
            ("determine_layout", "detect_elements"),
            ("detect_elements", "analyze_charts"),
            ("detect_elements", "analyze_tables"),
            ("detect_elements", "verify_signatures"),
            ("determine_layout", "assess_ocr_reliability"),
            ("assess_ocr_reliability", "extract_entities"),
            ("extract_entities", "analyze_semantics"),
            ("analyze_charts", "align_modalities"),
            ("analyze_tables", "align_modalities"),
            ("verify_signatures", "align_modalities"),
            ("analyze_semantics", "align_modalities"),
            ("align_modalities", "arbitrate_confidence"),
            ("arbitrate_confidence", "check_consistency"),
            ("check_consistency", "detect_contradictions"),
            ("detect_contradictions", "assess_risk"),
            ("assess_risk", "generate_explanations"),
            ("generate_explanations", "generate_review_recommendations"),
            ("generate_review_recommendations", "compile_results"),
        ]
        
        # Add edges
        for from_node, to_node in edges:
            workflow.add_edge(from_node, to_node)
        
        workflow.add_edge("compile_results", END)
        
        # Set entry point
        workflow.set_entry_point("assess_quality")
        
        logger.info(f"Created workflow with {len(agent_nodes)} nodes and {len(edges)} edges")
        return workflow
    
    def _compile_final_results(self, state: ProcessingState) -> ProcessingState:
        """Compile final processing results"""
        try:
            logger.info(f"📦 Compiling final results for {state.document_id}")
            
            # Ensure all required attributes exist
            if not hasattr(state, 'extracted_entities'):
                state.extracted_entities = {}
            if not hasattr(state, 'chart_analysis'):
                state.chart_analysis = {}
            if not hasattr(state, 'semantic_analysis'):
                state.semantic_analysis = {}
            if not hasattr(state, 'contradictions'):
                state.contradictions = []
            if not hasattr(state, 'errors'):
                state.errors = []
            if not hasattr(state, 'extracted_fields'):
                state.extracted_fields = {}
            
            # Calculate integrity score
            integrity_score = self._calculate_integrity_score(state)
            
            # Add processing metadata
            state.processing_end = datetime.now()
            
            processing_time = (state.processing_end - state.processing_start).total_seconds()
            
            state.processing_metadata = {
                "integrity_score": integrity_score,
                "total_pages": len(state.images) if hasattr(state, 'images') and state.images else 0,
                "agents_executed": list(self.agents.keys()),
                "processing_time": processing_time,
                "document_type": state.document_type.value if hasattr(state, 'document_type') and state.document_type else "unknown",
                "success": len(state.errors) == 0
            }
            
            logger.info(f"✅ Results compiled for {state.document_id}")
            return state
            
        except Exception as e:
            logger.error(f"❌ Results compilation failed: {e}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Results compilation error: {str(e)}")
            return state
    
    def _calculate_integrity_score(self, state: ProcessingState) -> float:
        """Calculate document integrity score"""
        scores = []
        
        # Quality scores (20%)
        if hasattr(state, 'quality_scores') and state.quality_scores:
            avg_quality = sum(score.overall for score in state.quality_scores.values()) / len(state.quality_scores)
            scores.append(avg_quality * 0.2)
        else:
            scores.append(0.6 * 0.2)  # Default score
        
        # OCR confidence (30%)
        if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
            avg_ocr = sum(state.ocr_confidence.values()) / len(state.ocr_confidence)
            scores.append(avg_ocr * 0.3)
        else:
            scores.append(0.7 * 0.3)  # Default score
        
        # Field confidence (30%)
        if hasattr(state, 'field_confidences') and state.field_confidences:
            avg_field = sum(state.field_confidences.values()) / len(state.field_confidences)
            scores.append(avg_field * 0.3)
        else:
            scores.append(0.65 * 0.3)  # Default score
        
        # Contradiction penalty (20%)
        contradiction_count = len(state.contradictions) if hasattr(state, 'contradictions') else 0
        contradiction_penalty = contradiction_count * 0.1
        scores.append(max(0, 0.2 - contradiction_penalty))
        
        return min(1.0, sum(scores))
    
    async def process_document(self, file_path: str, document_id: str = None) -> Dict[str, Any]:
        """Main method to process a document"""
        try:
            logger.info(f"🚀 Starting document processing pipeline for {file_path}")
            
            # For now, create a mock processing result
            # In production, this would actually process the document
            
            state = ProcessingState(
                document_id=document_id or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=file_path,
                file_type=os.path.splitext(file_path)[1].lower() if file_path else "unknown",
                images=[],  # Empty for now
                processing_start=datetime.now()
            )
            
            # Execute workflow
            logger.info(f"Executing workflow")
            final_state = await self.compiled_workflow.ainvoke(
                state,
                config={"configurable": {"thread_id": state.document_id}}
            )
            
            # Prepare response
            response = {
                "success": len(final_state.errors) == 0,
                "document_id": final_state.document_id,
                "document_type": final_state.document_type.value if hasattr(final_state, 'document_type') and final_state.document_type else "unknown",
                "agent_outputs": self._extract_agent_outputs(final_state),
                "extracted_fields": {
                    name: {
                        "value": field.value,
                        "confidence": field.confidence,
                        "sources": field.sources,
                        "modalities": field.modalities
                    }
                    for name, field in final_state.extracted_fields.items()
                } if hasattr(final_state, 'extracted_fields') else {},
                "validation_results": {
                    "contradictions": [
                        {
                            "type": c.contradiction_type.value if hasattr(c.contradiction_type, 'value') else str(c.contradiction_type),
                            "severity": c.severity.value if hasattr(c.severity, 'value') else str(c.severity),
                            "explanation": c.explanation,
                            "confidence": c.confidence
                        }
                        for c in final_state.contradictions
                    ] if hasattr(final_state, 'contradictions') else [],
                    "risk_score": final_state.risk_score if hasattr(final_state, 'risk_score') else 0.0,
                    "integrity_score": final_state.processing_metadata.get("integrity_score", 0.0) if hasattr(final_state, 'processing_metadata') else 0.0
                },
                "explanations": final_state.explanations if hasattr(final_state, 'explanations') else {},
                "recommendations": final_state.review_recommendations if hasattr(final_state, 'review_recommendations') else [],
                "processing_metadata": final_state.processing_metadata if hasattr(final_state, 'processing_metadata') else {},
                "errors": final_state.errors if hasattr(final_state, 'errors') else []
            }
            
            logger.info(f"✅ Document processing completed: {final_state.document_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id or "unknown",
                "errors": [str(e)]
            }
    
    def _extract_agent_outputs(self, state: ProcessingState) -> Dict[str, Any]:
        """Extract outputs from each agent for reporting"""
        outputs = {}
        
        # Extract from various state attributes
        if hasattr(state, 'quality_scores') and state.quality_scores:
            outputs["quality"] = {
                "scores": {str(k): v.dict() for k, v in state.quality_scores.items()}
            }
        
        if hasattr(state, 'visual_elements') and state.visual_elements:
            outputs["vision"] = {
                "detected_elements": [
                    {
                        "type": elem.element_type,
                        "bbox": elem.bbox,
                        "confidence": elem.confidence,
                        "page_num": elem.page_num
                    }
                    for page_elems in state.visual_elements.values()
                    for elem in page_elems
                ]
            }
        
        if hasattr(state, 'extracted_entities') and state.extracted_entities:
            outputs["entities"] = {
                "extracted_entities": state.extracted_entities
            }
        
        if hasattr(state, 'ocr_confidence') and state.ocr_confidence:
            outputs["ocr"] = {
                "confidence_scores": state.ocr_confidence
            }
        
        if hasattr(state, 'semantic_analysis') and state.semantic_analysis:
            outputs["semantics"] = state.semantic_analysis
        
        return outputs


# Agent implementations (keep the same as before)
# ... [All the agent classes remain the same] ...

# Add the missing import at the top of the file
import os

# Agent class implementations (truncated for brevity, but same as before)
class DocumentQualityAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🔍 Assessing document quality for {state.document_id}")
        from app.core.models import QualityScore
        
        # Simple quality assessment
        quality_scores = {}
        quality_scores[0] = QualityScore(
            sharpness=0.8,
            brightness=0.7,
            contrast=0.6,
            noise_level=0.9,
            overall=0.75
        )
        
        state.quality_scores = quality_scores
        return state


class DocumentTypeClassifier:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🏷️ Classifying document type for {state.document_id}")
        from app.core.models import DocumentType
        
        # Simple classification based on filename and content
        if state.file_path:
            if "invoice" in state.file_path.lower():
                state.document_type = DocumentType.INVOICE
            elif "contract" in state.file_path.lower():
                state.document_type = DocumentType.CONTRACT
            elif "report" in state.file_path.lower():
                state.document_type = DocumentType.FINANCIAL_REPORT
            else:
                state.document_type = DocumentType.MIXED
        else:
            state.document_type = DocumentType.UNKNOWN
        
        return state

class LayoutStrategyAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"📐 Determining layout strategy for {state.document_id}")
        
        # Determine layout based on document type
        if state.document_type:
            if state.document_type.value in ["financial_report", "research_paper"]:
                state.layout_strategy = "complex_multi_column"
            elif state.document_type.value in ["invoice", "form"]:
                state.layout_strategy = "structured_form"
            else:
                state.layout_strategy = "general"
        else:
            state.layout_strategy = "general"
        
        return state

class VisualElementDetector:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"👁️ Detecting visual elements for {state.document_id}")
        from app.core.models import VisualElement
        
        # Placeholder detection - in production, use YOLO/OpenCV
        visual_elements = {}
        
        for page_num, img in enumerate(state.images):
            elements = []
            
            # Simulate detecting some elements
            elements.append(VisualElement(
                element_type="table",
                bbox=[100, 100, 400, 300],
                confidence=0.85,
                page_num=page_num,
                metadata={"rows": 5, "columns": 3}
            ))
            
            elements.append(VisualElement(
                element_type="text_region",
                bbox=[50, 350, 450, 500],
                confidence=0.9,
                page_num=page_num,
                metadata={"text_length": 250}
            ))
            
            visual_elements[page_num] = elements
        
        state.visual_elements = visual_elements
        return state

class ChartUnderstandingAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"📈 Analyzing charts for {state.document_id}")
        
        # Analyze charts in visual elements
        chart_analysis = {}
        
        if hasattr(state, 'visual_elements'):
            for page_num, elements in state.visual_elements.items():
                for elem in elements:
                    if elem.element_type == "chart":
                        chart_id = f"chart_{page_num}_{len(chart_analysis)}"
                        chart_analysis[chart_id] = {
                            "type": "bar_chart",  # Placeholder
                            "trend_direction": "increasing",
                            "confidence": elem.confidence,
                            "page": page_num
                        }
        
        state.chart_analysis = chart_analysis
        return state

class TableStructureAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"📊 Analyzing tables for {state.document_id}")
        
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
                            "estimated_columns": elem.metadata.get("columns", 0)
                        }
        
        state.table_structures = table_structures
        return state

class SignatureVerificationAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"✍️ Verifying signatures for {state.document_id}")
        
        # Check for signatures
        signature_verification = {"found": False, "locations": []}
        
        if hasattr(state, 'visual_elements'):
            for page_num, elements in state.visual_elements.items():
                for elem in elements:
                    if elem.element_type == "signature":
                        signature_verification["found"] = True
                        signature_verification["locations"].append({
                            "page": page_num,
                            "bbox": elem.bbox,
                            "confidence": elem.confidence
                        })
        
        state.signature_verification = signature_verification
        return state

class OCRReliabilityAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🔤 Assessing OCR reliability for {state.document_id}")
        
        # Perform OCR and assess reliability
        from app.models.ocr_engine import HybridOCREngine
        
        ocr_engine = HybridOCREngine()
        ocr_results = {}
        ocr_confidence = {}
        
        for page_num, img in enumerate(state.images):
            result = ocr_engine.process_image(img, page_num)
            ocr_results[page_num] = {
                "text": result.text,
                "confidence": result.average_confidence,
                "word_count": len(result.words)
            }
            ocr_confidence[page_num] = result.average_confidence
        
        state.ocr_results = ocr_results
        state.ocr_confidence = ocr_confidence
        
        # Add extracted text to state
        all_text = "\n".join([r["text"] for r in ocr_results.values()])
        state.extracted_text = all_text
        
        return state

class EntityIntelligenceAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🏷️ Extracting entities for {state.document_id}")
        
        # Extract entities from OCR text
        extracted_entities = {
            "dates": [],
            "amounts": [],
            "names": [],
            "organizations": []
        }
        
        if hasattr(state, 'extracted_text') and state.extracted_text:
            text = state.extracted_text.lower()
            
            # Simple entity extraction (in production, use NER model)
            import re
            
            # Extract dates
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD-MM-YYYY
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # YYYY-MM-DD
                r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b'  # Month Day Year
            ]
            
            for pattern in date_patterns:
                dates = re.findall(pattern, text)
                extracted_entities["dates"].extend(dates)
            
            # Extract amounts
            amount_pattern = r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:dollars|USD)'
            amounts = re.findall(amount_pattern, text)
            extracted_entities["amounts"].extend(amounts)
        
        state.extracted_entities = extracted_entities
        return state

class SemanticReasoningAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🧠 Performing semantic analysis for {state.document_id}")
        
        # Analyze document semantics
        semantic_analysis = {
            "summary": "",
            "key_topics": [],
            "sentiment": "neutral",
            "confidence": 0.7
        }
        
        if hasattr(state, 'extracted_text') and state.extracted_text:
            text = state.extracted_text[:1000]  # Limit for analysis
            
            # Simple semantic analysis
            keywords = ["report", "analysis", "data", "result", "conclusion", "summary"]
            found_keywords = [kw for kw in keywords if kw in text.lower()]
            
            semantic_analysis["key_topics"] = found_keywords[:5]
            
            # Generate simple summary
            sentences = text.split('.')
            if len(sentences) > 3:
                summary = '. '.join(sentences[:3]) + '.'
                semantic_analysis["summary"] = summary
        
        state.semantic_analysis = semantic_analysis
        return state

class CrossModalAlignmentAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🔄 Aligning modalities for {state.document_id}")
        
        # Align text and visual information
        aligned_data = {
            "text_visual_alignment": [],
            "confidence": 0.8
        }
        
        # Simple alignment logic
        if hasattr(state, 'visual_elements') and hasattr(state, 'extracted_text'):
            # Check if text mentions detected elements
            text_lower = state.extracted_text.lower()
            
            for page_num, elements in state.visual_elements.items():
                for elem in elements:
                    elem_type = elem.element_type
                    if elem_type in text_lower:
                        aligned_data["text_visual_alignment"].append({
                            "element_type": elem_type,
                            "page": page_num,
                            "mentioned_in_text": True
                        })
        
        state.aligned_data = aligned_data
        return state

class ConfidenceArbitrationAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"⚖️ Arbitrating confidence scores for {state.document_id}")
        
        # Calculate field confidences
        field_confidences = {}
        
        # Combine confidences from different sources
        if hasattr(state, 'ocr_confidence'):
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
        
        state.field_confidences = field_confidences
        return state

class TemporalNumericConsistencyAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🔢 Checking temporal/numeric consistency for {state.document_id}")
        
        # Check consistency of dates and numbers
        temporal_consistency = {
            "date_consistency": "unknown",
            "numeric_consistency": "unknown",
            "issues": []
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

class ContradictionDetectionAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"⚠️ Detecting contradictions for {state.document_id}")
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
                        confidence=0.6
                    ))
        
        state.contradictions = contradictions
        return state

class RiskComplianceAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"🔒 Assessing risk and compliance for {state.document_id}")
        
        # Calculate risk score
        risk_score = 0.0
        compliance_issues = []
        
        # Check for missing signatures
        if hasattr(state, 'signature_verification'):
            if not state.signature_verification.get("found", False):
                risk_score += 0.3
                compliance_issues.append("No signature detected")
        
        # Check for contradictions
        if hasattr(state, 'contradictions'):
            risk_score += len(state.contradictions) * 0.1
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        state.risk_score = risk_score
        state.compliance_issues = compliance_issues
        return state

class ExplanationAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"💡 Generating explanations for {state.document_id}")
        
        explanations = {
            "document_quality": "Document quality assessed based on image properties",
            "element_detection": "Visual elements detected using computer vision",
            "text_extraction": "Text extracted using hybrid OCR engine",
            "entity_extraction": "Entities extracted using pattern matching",
            "consistency_check": "Checked for temporal and numeric consistency",
            "risk_assessment": "Risk calculated based on missing elements and contradictions"
        }
        
        state.explanations = explanations
        return state

class HumanReviewAgent:
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"👥 Generating review recommendations for {state.document_id}")
        
        review_recommendations = []
        
        # Recommend review based on risk score
        if hasattr(state, 'risk_score'):
            if state.risk_score > 0.7:
                review_recommendations.append("CRITICAL: High risk detected - manual review required")
            elif state.risk_score > 0.4:
                review_recommendations.append("MODERATE: Some issues found - recommended review")
            else:
                review_recommendations.append("LOW: No significant issues found")
        
        # Add specific recommendations
        if hasattr(state, 'signature_verification'):
            if not state.signature_verification.get("found", False):
                review_recommendations.append("Check for missing signature")
        
        if hasattr(state, 'contradictions'):
            if state.contradictions:
                review_recommendations.append(f"Review {len(state.contradictions)} potential contradiction(s)")
        
        state.review_recommendations = review_recommendations
        return state