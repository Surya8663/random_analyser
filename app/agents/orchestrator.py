# app/agents/orchestrator.py - CORRECTED
from typing import Dict, Any
from datetime import datetime
import asyncio
from app.core.models import MultiModalDocument
from app.eval.evaluator import DocumentEvaluator
from app.explain.explainability import ExplainabilityGenerator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class Phase3Orchestrator:
    """Phase 3 Orchestrator with explainability and evaluation"""
    
    def __init__(self):
        self.vision_agent = None
        self.text_agent = None
        self.fusion_agent = None
        self.reasoning_agent = None
        self.explainability_generator = ExplainabilityGenerator()
        self.evaluator = DocumentEvaluator()
        logger.info("✅ Phase 3 Orchestrator with explainability initialized")
    
    def _get_vision_agent(self):
        if self.vision_agent is None:
            from app.agents.vision_agent import VisionAgent
            self.vision_agent = VisionAgent()
        return self.vision_agent
    
    def _get_text_agent(self):
        if self.text_agent is None:
            from app.agents.text_agent import TextAgent
            self.text_agent = TextAgent()
        return self.text_agent
    
    def _get_fusion_agent(self):
        if self.fusion_agent is None:
            from app.agents.fusion_agent import FusionAgent
            self.fusion_agent = FusionAgent()
        return self.fusion_agent
    
    def _get_reasoning_agent(self):
        if self.reasoning_agent is None:
            from app.agents.reasoning_agent import ReasoningAgent
            self.reasoning_agent = ReasoningAgent()
        return self.reasoning_agent
    
    async def process(self, document: MultiModalDocument) -> MultiModalDocument:
        """Process document through all agents with explainability"""
        try:
            logger.info(f"🚀 Starting Phase 3 workflow for document {document.document_id}")
            
            # Attach explainability tracking
            self.explainability_generator.attach_to_document(document)
            
            # Get provenance tracker
            provenance_tracker = self.explainability_generator.provenance_tracker
            
            current = document
            
            # Execute agents in sequence
            agents = [
                ("Vision", self._get_vision_agent()),
                ("Text", self._get_text_agent()),
                ("Fusion", self._get_fusion_agent()),
                ("Reasoning", self._get_reasoning_agent())
            ]
            
            for name, agent in agents:
                try:
                    # Log agent selection and routing decision
                    logger.info(f"📋 Agent selected: {name} for document {document.document_id}")
                    logger.info(f"🔄 Routing decision: Sequential execution (pre-defined workflow)")
                    
                    # Set provenance tracker for agent
                    if hasattr(agent, 'set_provenance_tracker'):
                        agent.set_provenance_tracker(provenance_tracker)
                    
                    logger.info(f"▶️ Executing {name} agent...")
                    current = await agent.process(current)
                    logger.info(f"✅ {name} agent completed successfully")
                except Exception as e:
                    logger.error(f"❌ {name} agent failed: {e}")
                    if not hasattr(current, 'errors'):
                        current.errors = []
                    current.errors.append(f"{name} agent error: {str(e)}")
                    # Record agent failure in provenance
                    if provenance_tracker:
                        provenance_tracker.record_agent_end(
                            agent_name=name,
                            fields_extracted=[],
                            errors=[str(e)]
                        )
                
                # Small delay to prevent overwhelming system
                await asyncio.sleep(0.1)
            
            # Generate explainability report
            explainability_report = self.explainability_generator.generate_explainability_report(current)
            current.processing_metadata["explainability_report"] = explainability_report
            
            # Generate evaluation report
            evaluation_report = await self.evaluator.evaluate_document(current)
            current.evaluation_report = evaluation_report
            
            # Final processing timestamp
            current.processing_end = datetime.now()
            
            logger.info(f"✅ Phase 3 workflow completed for {document.document_id}")
            logger.info(f"   Processing time: {current.get_processing_time():.2f} seconds")
            logger.info(f"   Errors: {len(current.errors)}")
            if explainability_report and 'field_explanations' in explainability_report:
                logger.info(f"   Explainability report generated: {len(explainability_report.get('field_explanations', {}))} fields explained")
            
            return current
            
        except Exception as e:
            logger.error(f"❌ Phase 3 workflow failed: {e}", exc_info=True)
            document.errors.append(f"Phase 3 workflow error: {str(e)}")
            document.processing_end = datetime.now()
            
            # Set default values
            if not hasattr(document, 'risk_score'):
                document.risk_score = 0.5
            if not hasattr(document, 'contradictions'):
                document.contradictions = []
            if not hasattr(document, 'review_recommendations'):
                document.review_recommendations = ["⚠️ System error - manual review required"]
            
            return document
    
    async def process_document(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Process document and return structured results with explainability"""
        try:
            # Log the start of document processing
            logger.info(f"📄 Processing document {document.document_id} through orchestrator")
            
            # Execute workflow
            result_doc = await self.process(document)
            
            # Prepare response
            response = {
                "success": len(result_doc.errors) == 0,
                "document_id": result_doc.document_id,
                "processing_time": result_doc.get_processing_time(),
                "document_type": result_doc.document_type.value if result_doc.document_type else "unknown",
                "risk_score": result_doc.risk_score,
                "contradictions_count": len(result_doc.contradictions),
                "extracted_fields_count": len(result_doc.extracted_fields) if hasattr(result_doc, 'extracted_fields') else 0,
                "errors": result_doc.errors,
                "recommendations": result_doc.review_recommendations
            }
            
            # Add detailed results if successful
            if response["success"]:
                response["detailed_results"] = {
                    "document_understanding": {
                        "type": result_doc.document_type.value if result_doc.document_type else "unknown",
                        "validation_score": result_doc.processing_metadata.get("reasoning", {}).get("validation_results", {}).get("score", 0) 
                        if hasattr(result_doc, 'processing_metadata') and "reasoning" in result_doc.processing_metadata else 0
                    },
                    "content_summary": {
                        "text": {
                            "length": len(result_doc.raw_text),
                            "pages": len(result_doc.ocr_results),
                            "entities": {k: len(v) for k, v in result_doc.extracted_entities.items()}
                        } if hasattr(result_doc, 'extracted_entities') else {},
                        "visual": {
                            "elements": len(result_doc.visual_elements),
                            "tables": sum(1 for e in result_doc.visual_elements if e.element_type == "table"),
                            "signatures": sum(1 for e in result_doc.visual_elements if e.element_type == "signature")
                        } if hasattr(result_doc, 'visual_elements') else {}
                    },
                    "extracted_information": {
                        field_name: {
                            "value": field.value,
                            "type": field.field_type,
                            "confidence": field.confidence,
                            "sources": field.modality_sources
                        }
                        for field_name, field in result_doc.extracted_fields.items()
                    } if hasattr(result_doc, 'extracted_fields') else {},
                    "quality_assessment": {
                        "risk_score": result_doc.risk_score,
                        "risk_level": "HIGH" if result_doc.risk_score > 0.7 else 
                                    "MEDIUM" if result_doc.risk_score > 0.4 else "LOW",
                        "contradictions": [
                            {
                                "type": c.contradiction_type.value,
                                "severity": c.severity.value,
                                "explanation": c.explanation
                            }
                            for c in result_doc.contradictions
                        ]
                    } if hasattr(result_doc, 'contradictions') else {},
                    "explainability": result_doc.processing_metadata.get("explainability_report", {}),
                    "evaluation": result_doc.evaluation_report.dict() if hasattr(result_doc, 'evaluation_report') and result_doc.evaluation_report else None
                }
            
            logger.info(f"✅ Document {document.document_id} processing completed: success={response['success']}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return {
                "success": False,
                "document_id": document.document_id if hasattr(document, 'document_id') else "unknown",
                "error": str(e),
                "processing_time": document.get_processing_time() if hasattr(document, 'get_processing_time') else 0
            }