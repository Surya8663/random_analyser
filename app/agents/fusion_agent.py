from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.utils.logger import setup_logger
import json
import traceback

logger = setup_logger(__name__)

class ValidationState(BaseModel):
    """State for Validation Agent with explicit defaults"""
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
        """Cross-check extracted values with input validation"""
        try:
            logger.info(f"Cross-checking document {state.document_id}")
            
            # VALIDATE INPUTS (CRITICAL FIX)
            if not state.fused_results:
                error_msg = "No fused results provided for validation"
                logger.error(error_msg)
                state.errors.append(error_msg)
                return state
            
            # Ensure fused_results has required structure
            if 'structured_output' not in state.fused_results:
                state.fused_results['structured_output'] = {'fields': {}}
            
            if 'fields' not in state.fused_results['structured_output']:
                state.fused_results['structured_output']['fields'] = {}
            
            validation_results = {
                "consistency_checks": [],
                "plausibility_checks": [],
                "completeness_checks": []
            }
            
            # Check field consistency
            fields = state.fused_results['structured_output']['fields']
            for field_name, field_data in fields.items():
                if not isinstance(field_data, dict):
                    logger.warning(f"Invalid field data for {field_name}: {type(field_data)}")
                    continue
                    
                consistency = self._check_field_consistency(field_data)
                validation_results["consistency_checks"].append({
                    "field": field_name,
                    "status": consistency["status"],
                    "details": consistency["details"]
                })
            
            # Check plausibility
            plausibility = self._check_plausibility(fields)
            validation_results["plausibility_checks"] = plausibility
            
            # Check completeness
            completeness = self._check_completeness(fields)
            validation_results["completeness_checks"] = completeness
            
            state.validation_results = validation_results
            logger.info(f"Completed {len(validation_results['consistency_checks'])} checks")
            
        except Exception as e:
            error_msg = f"Cross-checking failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            state.errors.append(error_msg)
        
        return state
    
    def _check_field_consistency(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency of a field with validation"""
        try:
            confidence = field_data.get("confidence", 0.0)
            sources = field_data.get("source", "")
            
            if isinstance(sources, str):
                source_list = sources.split(" + ") if " + " in sources else [sources]
            elif isinstance(sources, list):
                source_list = sources
            else:
                source_list = []
            
            if confidence < 0.5:
                return {
                    "status": "LOW_CONFIDENCE",
                    "details": f"Confidence score {confidence:.2f} is below threshold"
                }
            elif len(source_list) == 1:
                return {
                    "status": "SINGLE_SOURCE",
                    "details": "Field extracted from single modality only"
                }
            else:
                return {
                    "status": "CONSISTENT",
                    "details": "Field validated across multiple modalities"
                }
        except Exception as e:
            return {
                "status": "VALIDATION_ERROR",
                "details": f"Consistency check failed: {str(e)}"
            }
    
    def _check_plausibility(self, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check plausibility of field values"""
        plausibility_checks = []
        
        try:
            # Check for numeric fields with reasonable values
            for field_name, field_data in fields.items():
                if not isinstance(field_data, dict):
                    continue
                    
                value = field_data.get("value", "")
                
                # Check if value contains numbers
                if any(char.isdigit() for char in str(value)):
                    # Extract numbers
                    import re
                    numbers = re.findall(r'\d+\.?\d*', str(value))
                    
                    if numbers:
                        for num in numbers:
                            try:
                                num_float = float(num)
                                if "amount" in field_name.lower() or "total" in field_name.lower():
                                    if num_float > 1000000:  # Unusually large amount
                                        plausibility_checks.append({
                                            "field": field_name,
                                            "issue": "UNUSUALLY_LARGE_VALUE",
                                            "value": num_float,
                                            "threshold": 1000000
                                        })
                            except ValueError:
                                continue
        except Exception as e:
            logger.warning(f"Plausibility check error: {e}")
        
        return plausibility_checks
    
    def _check_completeness(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Check completeness of extraction"""
        try:
            required_fields = ["date", "amount", "signature", "table"]
            found_fields = []
            
            for field_name in fields.keys():
                if not isinstance(field_name, str):
                    continue
                    
                for required in required_fields:
                    if required in field_name.lower():
                        found_fields.append(required)
            
            missing_fields = [f for f in required_fields if f not in found_fields]
            
            return {
                "total_required": len(required_fields),
                "found": len(found_fields),
                "missing": missing_fields,
                "completeness_score": len(found_fields) / len(required_fields) if required_fields else 1.0
            }
        except Exception as e:
            return {
                "total_required": 4,
                "found": 0,
                "missing": ["date", "amount", "signature", "table"],
                "completeness_score": 0.0,
                "error": str(e)
            }
    
    def detect_inconsistencies(self, state: ValidationState) -> ValidationState:
        """Detect inconsistencies in the data"""
        try:
            logger.info(f"Detecting inconsistencies for document {state.document_id}")
            
            inconsistencies = []
            
            # Validate input
            if 'structured_output' not in state.fused_results:
                logger.warning("No structured_output in fused_results")
                state.validation_results["inconsistencies"] = inconsistencies
                return state
            
            fields = state.fused_results['structured_output'].get('fields', {})
            
            # Check for contradictory information
            for field_name, field_data in fields.items():
                if not isinstance(field_data, dict):
                    continue
                    
                if "chart" in field_name.lower():
                    # Check if chart data contradicts text
                    chart_inconsistency = self._check_chart_text_consistency(
                        field_data, fields
                    )
                    if chart_inconsistency:
                        inconsistencies.append(chart_inconsistency)
            
            # Check temporal consistency
            temporal_issues = self._check_temporal_consistency(fields)
            inconsistencies.extend(temporal_issues)
            
            # Check numeric consistency
            numeric_issues = self._check_numeric_consistency(fields)
            inconsistencies.extend(numeric_issues)
            
            state.validation_results["inconsistencies"] = inconsistencies
            logger.info(f"Found {len(inconsistencies)} inconsistencies")
            
        except Exception as e:
            error_msg = f"Inconsistency detection failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            state.errors.append(error_msg)
        
        return state
    
    def _check_chart_text_consistency(self, chart_field: Dict[str, Any], 
                                     all_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if chart data contradicts text"""
        try:
            # Look for text fields about trends or comparisons
            text_fields = [
                (name, data) for name, data in all_fields.items()
                if isinstance(data, dict) and "text" in str(data.get("source", "")).lower() 
                and ("trend" in name.lower() or "comparison" in name.lower())
            ]
            
            if text_fields:
                # This would involve comparing chart analysis with text descriptions
                # For now, return a placeholder check
                return {
                    "type": "chart_text_contradiction",
                    "chart_field": list(chart_field.keys())[0] if chart_field else "unknown",
                    "description": "Potential contradiction between chart data and text description",
                    "severity": "medium"
                }
        except Exception as e:
            logger.warning(f"Chart-text consistency check error: {e}")
        
        return None
    
    def _check_temporal_consistency(self, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check temporal consistency"""
        temporal_issues = []
        
        try:
            dates = []
            
            # Extract dates from fields
            for field_name, field_data in fields.items():
                if not isinstance(field_data, dict):
                    continue
                    
                if "date" in field_name.lower():
                    dates.append({
                        "field": field_name,
                        "value": field_data.get("value", "")
                    })
            
            # Check if dates are in chronological order
            if len(dates) > 1:
                temporal_issues.append({
                    "type": "multiple_dates_found",
                    "dates": dates,
                    "check_needed": "Chronological order verification",
                    "severity": "low"
                })
        except Exception as e:
            logger.warning(f"Temporal consistency check error: {e}")
        
        return temporal_issues
    
    def _check_numeric_consistency(self, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check numeric consistency"""
        numeric_issues = []
        
        try:
            amounts = []
            
            # Extract amounts from fields
            for field_name, field_data in fields.items():
                if not isinstance(field_data, dict):
                    continue
                    
                if "amount" in field_name.lower() or "total" in field_name.lower():
                    value = field_data.get("value", "")
                    # Extract numbers
                    import re
                    numbers = re.findall(r'\d+\.?\d*', str(value))
                    if numbers:
                        try:
                            amounts.append({
                                "field": field_name,
                                "value": float(numbers[0])
                            })
                        except ValueError:
                            continue
            
            # Check if amounts are consistent
            if len(amounts) > 1:
                values = [a["value"] for a in amounts]
                avg_value = sum(values) / len(values)
                
                for amount in amounts:
                    if avg_value != 0:
                        deviation = abs(amount["value"] - avg_value) / avg_value
                        if deviation > 0.5:  # More than 50% deviation
                            numeric_issues.append({
                                "type": "amount_inconsistency",
                                "field": amount["field"],
                                "value": amount["value"],
                                "average": avg_value,
                                "deviation": f"{deviation:.1%}",
                                "severity": "medium"
                            })
        except Exception as e:
            logger.warning(f"Numeric consistency check error: {e}")
        
        return numeric_issues
    
    def generate_flags(self, state: ValidationState) -> ValidationState:
        """Generate validation flags"""
        try:
            logger.info(f"Generating flags for document {state.document_id}")
            
            flags = []
            
            # Add flags from consistency checks
            for check in state.validation_results.get("consistency_checks", []):
                if check.get("status") != "CONSISTENT":
                    flags.append({
                        "type": "CONSISTENCY_FLAG",
                        "field": check.get("field", "unknown"),
                        "status": check["status"],
                        "reason": check.get("details", "Unknown issue"),
                        "priority": "HIGH" if check["status"] == "LOW_CONFIDENCE" else "MEDIUM"
                    })
            
            # Add flags from plausibility checks
            for check in state.validation_results.get("plausibility_checks", []):
                flags.append({
                    "type": "PLAUSIBILITY_FLAG",
                    "field": check.get("field", "unknown"),
                    "status": "IMPLAUSIBLE_VALUE",
                    "reason": f"{check.get('issue', 'Unknown')}: {check.get('value', 'N/A')} exceeds threshold {check.get('threshold', 'N/A')}",
                    "priority": "MEDIUM"
                })
            
            # Add flags from inconsistencies
            for inconsistency in state.validation_results.get("inconsistencies", []):
                flags.append({
                    "type": "INCONSISTENCY_FLAG",
                    "field": inconsistency.get("chart_field", "multiple"),
                    "status": "CONTRADICTION_DETECTED",
                    "reason": inconsistency.get("description", "Unknown inconsistency"),
                    "priority": inconsistency.get("severity", "MEDIUM").upper()
                })
            
            # Add completeness flag if needed
            completeness = state.validation_results.get("completeness_checks", {})
            if completeness.get("completeness_score", 1.0) < 0.7:
                flags.append({
                    "type": "COMPLETENESS_FLAG",
                    "field": "document",
                    "status": "INCOMPLETE_EXTRACTION",
                    "reason": f"Only {completeness.get('found', 0)} of {completeness.get('total_required', 0)} required fields found",
                    "priority": "LOW"
                })
            
            # Ensure at least one flag if there were errors
            if not flags and state.errors:
                flags.append({
                    "type": "SYSTEM_FLAG",
                    "field": "system",
                    "status": "PROCESSING_ERRORS",
                    "reason": f"Found {len(state.errors)} processing errors",
                    "priority": "HIGH"
                })
            
            state.flags = flags
            logger.info(f"Generated {len(flags)} validation flags")
            
        except Exception as e:
            error_msg = f"Flag generation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            state.errors.append(error_msg)
        
        return state
    
    def provide_recommendations(self, state: ValidationState) -> ValidationState:
        """Provide recommendations based on validation results"""
        try:
            logger.info(f"Providing recommendations for document {state.document_id}")
            
            recommendations = []
            
            # Recommendations based on flags
            for flag in state.flags:
                if flag.get("priority") == "HIGH":
                    recommendations.append(
                        f"Manual review required for {flag.get('field', 'unknown field')}: {flag.get('reason', 'Unknown reason')}"
                    )
                elif flag.get("priority") == "MEDIUM":
                    recommendations.append(
                        f"Consider verifying {flag.get('field', 'unknown field')}: {flag.get('reason', 'Unknown reason')}"
                    )
            
            # General recommendations
            if not state.flags:
                recommendations.append(
                    "Document validation passed all checks. No manual review needed."
                )
            elif len([f for f in state.flags if f.get("priority") == "HIGH"]) == 0:
                recommendations.append(
                    "Document has minor issues. Consider batch review if multiple similar documents."
                )
            else:
                recommendations.append(
                    "Document has critical issues requiring immediate manual review."
                )
            
            # Add processing recommendations
            completeness = state.validation_results.get("completeness_checks", {})
            if completeness.get("missing"):
                recommendations.append(
                    f"Consider reprocessing to extract missing fields: {', '.join(completeness.get('missing', []))}"
                )
            
            # Add error-related recommendations
            if state.errors:
                recommendations.append(
                    f"Review processing errors: {len(state.errors)} error(s) occurred during validation"
                )
            
            state.recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            error_msg = f"Recommendation generation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            state.errors.append(error_msg)
        
        return state
    
    async def validate_document(self, document_id: str,
                               fused_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document through validation pipeline with guaranteed output"""
        try:
            # Validate inputs
            if not document_id:
                raise ValueError("document_id is required")
            
            # Initialize state
            state = ValidationState(
                document_id=document_id,
                fused_results=fused_results if fused_results else {}
            )
            
            # Create and run graph
            graph = self.create_graph()
            compiled_graph = graph.compile()
            
            # Execute graph
            result_state = compiled_graph.invoke(state)
            
            # Ensure we have some output even if validation failed
            if not result_state.flags and not result_state.errors:
                result_state.flags = [{
                    "type": "NO_FLAGS",
                    "field": "system",
                    "status": "NO_ISSUES_DETECTED",
                    "reason": "Validation completed but no specific issues were flagged",
                    "priority": "LOW"
                }]
            
            if not result_state.recommendations:
                result_state.recommendations = ["No specific recommendations generated"]
            
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
                "flags": result_state.flags,
                "recommendations": result_state.recommendations,
                "errors": result_state.errors,
                "validation_details": result_state.validation_results
            }
            
            logger.info(f"Validation completed for {document_id}: {response['validation_summary']['overall_status']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            logger.error(traceback.format_exc())
            
            # GUARANTEED RETURN EVEN ON FAILURE
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e),
                "validation_summary": {
                    "total_checks": 0,
                    "flags_generated": 0,
                    "inconsistencies_found": 0,
                    "overall_status": "ERROR"
                },
                "flags": [{
                    "type": "VALIDATION_ERROR",
                    "field": "system",
                    "status": "VALIDATION_FAILED",
                    "reason": f"Validation process failed: {str(e)}",
                    "priority": "HIGH"
                }],
                "recommendations": ["Investigate validation failure"],
                "errors": [str(e)],
                "validation_details": {}
            }