from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.utils.logger import setup_logger
import json

logger = setup_logger(__name__)

class ValidationState(BaseModel):
    """State for Validation Agent"""
    document_id: str
    fused_results: Dict[str, Any] = Field(default_factory=dict)
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    flags: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

class ValidationAgent:
    """Validation Agent for cross-checking and quality assurance"""
    
    def create_graph(self) -> StateGraph:
        """Create LangGraph for validation processing"""
        workflow = StateGraph(ValidationState)
        
        # Add nodes
        workflow.add_node("cross_check", self.cross_check)
        workflow.add_node("detect_inconsistencies", self.detect_inconsistencies)
        workflow.add_node("generate_flags", self.generate_flags)
        workflow.add_node("provide_recommendations", self.provide_recommendations)
        
        # Add edges
        workflow.add_edge("cross_check", "detect_inconsistencies")
        workflow.add_edge("detect_inconsistencies", "generate_flags")
        workflow.add_edge("generate_flags", "provide_recommendations")
        workflow.add_edge("provide_recommendations", END)
        
        # Set entry point
        workflow.set_entry_point("cross_check")
        
        return workflow
    
    def cross_check(self, state: ValidationState) -> ValidationState:
        """Cross-check extracted values"""
        try:
            logger.info(f"🔍 Cross-checking document {state.document_id}")
            
            validation_results = {
                "consistency_checks": [],
                "plausibility_checks": [],
                "completeness_checks": [],
                "timestamp": None
            }
            
            # Extract fields from fused results
            fields = {}
            if "structured_output" in state.fused_results:
                fields = state.fused_results["structured_output"].get("fields", {})
            elif "extracted_fields" in state.fused_results:
                fields = state.fused_results["extracted_fields"]
            
            # Check field consistency
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict):
                    consistency = self._check_field_consistency(field_data)
                    validation_results["consistency_checks"].append({
                        "field": field_name,
                        "status": consistency["status"],
                        "details": consistency["details"],
                        "confidence": field_data.get("confidence", 0)
                    })
            
            # Check plausibility
            plausibility = self._check_plausibility(fields)
            validation_results["plausibility_checks"] = plausibility
            
            # Check completeness
            completeness = self._check_completeness(state.fused_results)
            validation_results["completeness_checks"] = completeness
            
            validation_results["timestamp"] = "now"
            state.validation_results = validation_results
            logger.info(f"✅ Completed {len(validation_results['consistency_checks'])} checks")
            
        except Exception as e:
            error_msg = f"Cross-checking failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def _check_field_consistency(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency of a field"""
        confidence = field_data.get("confidence", 0.0)
        sources = field_data.get("sources", [])
        
        if confidence < 0.5:
            return {
                "status": "LOW_CONFIDENCE",
                "details": f"Confidence score {confidence:.2f} is below threshold"
            }
        elif len(sources) <= 1:
            return {
                "status": "SINGLE_SOURCE",
                "details": "Field extracted from single source only"
            }
        else:
            return {
                "status": "CONSISTENT",
                "details": f"Field validated across {len(sources)} sources"
            }
    
    def _check_plausibility(self, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check plausibility of field values"""
        plausibility_checks = []
        
        for field_name, field_data in fields.items():
            if isinstance(field_data, dict):
                value = field_data.get("value", "")
                
                # Check numeric fields
                if isinstance(value, (int, float)):
                    # Check for unusually large/small values
                    if "amount" in field_name.lower() or "total" in field_name.lower():
                        if value > 1000000:
                            plausibility_checks.append({
                                "field": field_name,
                                "issue": "UNUSUALLY_LARGE_VALUE",
                                "value": value,
                                "threshold": 1000000,
                                "severity": "MEDIUM"
                            })
                        elif value < 0:
                            plausibility_checks.append({
                                "field": field_name,
                                "issue": "NEGATIVE_VALUE",
                                "value": value,
                                "severity": "HIGH"
                            })
                
                # Check date fields
                elif "date" in field_name.lower():
                    # Validate date format (simplified)
                    if isinstance(value, str):
                        import re
                        date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}'
                        if not re.match(date_pattern, value):
                            plausibility_checks.append({
                                "field": field_name,
                                "issue": "INVALID_DATE_FORMAT",
                                "value": value,
                                "severity": "MEDIUM"
                            })
        
        return plausibility_checks
    
    def _check_completeness(self, fused_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check completeness of extraction"""
        # Check what types of information were extracted
        extracted_types = []
        
        if "agent_outputs" in fused_results:
            agents = fused_results["agent_outputs"]
            extracted_types = list(agents.keys())
        
        # Define expected extraction types based on document type
        expected_types = ["text", "vision", "entities", "semantics"]
        
        missing_types = [t for t in expected_types if t not in extracted_types]
        
        return {
            "extracted_types": extracted_types,
            "expected_types": expected_types,
            "missing_types": missing_types,
            "completeness_score": len(extracted_types) / len(expected_types) if expected_types else 1.0
        }
    
    def detect_inconsistencies(self, state: ValidationState) -> ValidationState:
        """Detect inconsistencies in the data"""
        try:
            logger.info(f"⚠️ Detecting inconsistencies for document {state.document_id}")
            
            inconsistencies = []
            
            # Extract data from fused results
            agent_outputs = state.fused_results.get("agent_outputs", {})
            extracted_fields = state.fused_results.get("extracted_fields", {})
            
            # Check for contradictions between vision and text
            if "vision" in agent_outputs and "text" in agent_outputs:
                vision_data = agent_outputs["vision"]
                text_data = agent_outputs["text"]
                
                # Check if visual elements are mentioned in text
                if "detected_elements" in vision_data:
                    elements = vision_data["detected_elements"]
                    text_content = text_data.get("extracted_text", "").lower()
                    
                    for element in elements[:5]:  # Check first 5 elements
                        elem_type = element.get("type", "").lower()
                        if elem_type and elem_type not in text_content:
                            inconsistencies.append({
                                "type": "visual_text_mismatch",
                                "element_type": elem_type,
                                "description": f"{elem_type} detected visually but not mentioned in text",
                                "severity": "low"
                            })
            
            # Check extracted fields for contradictions
            field_values = {}
            for field_name, field_data in extracted_fields.items():
                if isinstance(field_data, dict):
                    field_values[field_name] = field_data.get("value")
            
            # Simple contradiction detection: check for duplicate fields with different values
            for field1, value1 in field_values.items():
                for field2, value2 in field_values.items():
                    if field1 != field2 and field1 in field2:
                        if value1 != value2:
                            inconsistencies.append({
                                "type": "field_contradiction",
                                "fields": [field1, field2],
                                "values": [value1, value2],
                                "description": f"Contradiction between {field1} and {field2}",
                                "severity": "medium"
                            })
            
            state.validation_results["inconsistencies"] = inconsistencies
            logger.info(f"✅ Found {len(inconsistencies)} inconsistencies")
            
        except Exception as e:
            error_msg = f"Inconsistency detection failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def generate_flags(self, state: ValidationState) -> ValidationState:
        """Generate validation flags"""
        try:
            logger.info(f"🚩 Generating flags for document {state.document_id}")
            
            flags = []
            
            # Add flags from consistency checks
            for check in state.validation_results.get("consistency_checks", []):
                if check["status"] != "CONSISTENT":
                    flags.append({
                        "type": "CONSISTENCY_FLAG",
                        "field": check.get("field", "unknown"),
                        "status": check["status"],
                        "reason": check["details"],
                        "confidence": check.get("confidence", 0),
                        "priority": "HIGH" if check["status"] == "LOW_CONFIDENCE" else "MEDIUM"
                    })
            
            # Add flags from plausibility checks
            for check in state.validation_results.get("plausibility_checks", []):
                flags.append({
                    "type": "PLAUSIBILITY_FLAG",
                    "field": check.get("field", "unknown"),
                    "status": "IMPLAUSIBLE_VALUE",
                    "reason": f"{check['issue']}: {check['value']}",
                    "priority": check.get("severity", "MEDIUM").upper()
                })
            
            # Add flags from inconsistencies
            for inconsistency in state.validation_results.get("inconsistencies", []):
                flags.append({
                    "type": "INCONSISTENCY_FLAG",
                    "fields": inconsistency.get("fields", ["multiple"]),
                    "status": "CONTRADICTION_DETECTED",
                    "reason": inconsistency["description"],
                    "priority": inconsistency.get("severity", "MEDIUM").upper()
                })
            
            # Add completeness flag if needed
            completeness = state.validation_results.get("completeness_checks", {})
            if completeness.get("completeness_score", 1.0) < 0.5:
                flags.append({
                    "type": "COMPLETENESS_FLAG",
                    "field": "document",
                    "status": "INCOMPLETE_EXTRACTION",
                    "reason": f"Only extracted {len(completeness.get('extracted_types', []))} of {len(completeness.get('expected_types', []))} expected information types",
                    "priority": "LOW"
                })
            
            state.flags = flags
            logger.info(f"✅ Generated {len(flags)} validation flags")
            
        except Exception as e:
            error_msg = f"Flag generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def provide_recommendations(self, state: ValidationState) -> ValidationState:
        """Provide recommendations based on validation results"""
        try:
            logger.info(f"💡 Providing recommendations for document {state.document_id}")
            
            recommendations = []
            
            # Recommendations based on flags
            high_priority_flags = [f for f in state.flags if f.get("priority") == "HIGH"]
            medium_priority_flags = [f for f in state.flags if f.get("priority") == "MEDIUM"]
            
            if high_priority_flags:
                recommendations.append(f"⚠️ {len(high_priority_flags)} high-priority issues require immediate attention")
                for flag in high_priority_flags[:3]:  # Show top 3
                    recommendations.append(f"  • {flag.get('field')}: {flag.get('reason')}")
            
            if medium_priority_flags:
                recommendations.append(f"ℹ️ {len(medium_priority_flags)} medium-priority issues should be reviewed")
            
            if not state.flags:
                recommendations.append("✅ Document validation passed all checks. No manual review needed.")
            
            # Processing recommendations
            completeness = state.validation_results.get("completeness_checks", {})
            missing_types = completeness.get("missing_types", [])
            
            if missing_types:
                recommendations.append(f"🔧 Consider reprocessing to extract missing information types: {', '.join(missing_types)}")
            
            # Quality recommendations
            consistency_checks = state.validation_results.get("consistency_checks", [])
            low_confidence_count = len([c for c in consistency_checks if c.get("status") == "LOW_CONFIDENCE"])
            
            if low_confidence_count > 0:
                recommendations.append(f"📊 {low_confidence_count} fields have low confidence scores (< 0.5). Consider manual verification.")
            
            state.recommendations = recommendations
            logger.info(f"✅ Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            error_msg = f"Recommendation generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def validate_document(self, document_id: str,
                               fused_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document through validation pipeline"""
        try:
            logger.info(f"🔬 Starting validation for document {document_id}")
            
            # Initialize state
            state = ValidationState(
                document_id=document_id,
                fused_results=fused_results
            )
            
            # Create and run graph
            graph = self.create_graph()
            compiled_graph = graph.compile()
            
            # Execute graph
            result_state = compiled_graph.invoke(state)
            
            # Prepare response
            response = {
                "success": len(result_state.errors) == 0,
                "document_id": result_state.document_id,
                "validation_summary": {
                    "total_checks": len(result_state.validation_results.get("consistency_checks", [])),
                    "flags_generated": len(result_state.flags),
                    "inconsistencies_found": len(result_state.validation_results.get("inconsistencies", [])),
                    "overall_status": "PASS" if not result_state.flags else 
                                     "REVIEW_NEEDED" if any(f.get("priority") == "HIGH" for f in result_state.flags) 
                                     else "WARNING"
                },
                "validation_results": result_state.validation_results,
                "flags": result_state.flags,
                "recommendations": result_state.recommendations,
                "errors": result_state.errors
            }
            
            logger.info(f"✅ Validation completed for {document_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}", exc_info=True)
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e)
            }