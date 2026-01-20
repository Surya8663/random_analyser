# app/agents/reasoning_agent.py - CORRECTED
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.core.models import MultiModalDocument, DocumentType, Contradiction, ContradictionType, SeverityLevel
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ReasoningAgent(BaseAgent):
    """Real Reasoning Agent for final document understanding and validation - CORRECTED"""
    
    def __init__(self):
        self._accepts_multi_modal = True
        
        # Field importance weights
        self.field_weights = {
            "date": 1.0,
            "amount": 1.2,
            "identifier": 0.9,
            "name": 0.8,
            "signature": 1.5,
            "table": 1.1
        }
    
    async def process(self, document: MultiModalDocument) -> MultiModalDocument:
        """Perform final reasoning and validation - CORRECTED"""
        try:
            logger.info("🧠 Running Reasoning Agent (Final Validation)")
            
            # Step 1: Comprehensive document validation
            validation_results = self._validate_document_comprehensively(document)
            
            # Step 2: Calculate overall confidence scores
            confidence_scores = self._calculate_confidence_scores(document)
            
            # Step 3: Detect contradictions and inconsistencies
            contradictions = self._detect_contradictions(document)
            
            # Step 4: Calculate risk assessment
            risk_assessment = self._assess_risk(document, contradictions)
            
            # Step 5: Generate review recommendations
            recommendations = self._generate_recommendations(document, contradictions, risk_assessment)
            
            # Store all reasoning results
            if not hasattr(document, 'processing_metadata'):
                document.processing_metadata = {}
            
            document.processing_metadata["reasoning"] = {
                "validation_results": validation_results,
                "confidence_scores": confidence_scores,
                "risk_assessment": risk_assessment
            }
            
            document.contradictions = contradictions
            document.review_recommendations = recommendations
            document.risk_score = risk_assessment["overall_risk"]
            
            logger.info(f"✅ Reasoning completed: {len(contradictions)} contradictions, risk: {document.risk_score:.2f}")
            return document
            
        except Exception as e:
            logger.error(f"❌ Reasoning failed: {e}")
            document.errors.append(f"Reasoning agent error: {str(e)}")
            
            # Set default values if reasoning failed
            if not hasattr(document, 'risk_score'):
                document.risk_score = 0.5
            if not hasattr(document, 'contradictions'):
                document.contradictions = []
            if not hasattr(document, 'review_recommendations'):
                document.review_recommendations = ["⚠️ Reasoning incomplete - manual review recommended"]
            
            return document
    
    def _validate_document_comprehensively(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Perform comprehensive document validation - CORRECTED"""
        validation = {
            "passed": True,
            "checks": [],
            "issues": [],
            "score": 0.0
        }
        
        check_results = []
        
        # Check 1: Document has content
        has_text = bool(document.raw_text)
        has_visual = bool(document.visual_elements)
        
        if not has_text and not has_visual:
            validation["passed"] = False
            validation["issues"].append("Document has no text or visual content")
            check_results.append({"check": "has_content", "passed": False, "weight": 1.0})
        else:
            check_results.append({"check": "has_content", "passed": True, "weight": 1.0})
        
        # Check 2: OCR quality (if available)
        if hasattr(document, 'ocr_results') and document.ocr_results:
            try:
                avg_confidence = sum(ocr.average_confidence for ocr in document.ocr_results.values()) / len(document.ocr_results)
                ocr_passed = avg_confidence > 0.6
                check_results.append({
                    "check": "ocr_quality", 
                    "passed": ocr_passed, 
                    "weight": 0.8,
                    "details": {"average_confidence": avg_confidence}
                })
                if not ocr_passed:
                    validation["issues"].append(f"Low OCR confidence: {avg_confidence:.2f}")
            except:
                check_results.append({
                    "check": "ocr_quality", 
                    "passed": False, 
                    "weight": 0.8,
                    "details": {"error": "Could not calculate OCR confidence"}
                })
        
        # Check 3: Visual element consistency
        if has_visual:
            visual_types = set(elem.element_type for elem in document.visual_elements)
            visual_passed = len(visual_types) > 0
            check_results.append({
                "check": "visual_consistency",
                "passed": visual_passed,
                "weight": 0.7,
                "details": {"element_types": list(visual_types)}
            })
        
        # Check 4: Field extraction (if available)
        if hasattr(document, 'extracted_fields') and document.extracted_fields:
            field_count = len(document.extracted_fields)
            fields_passed = field_count > 0
            check_results.append({
                "check": "field_extraction",
                "passed": fields_passed,
                "weight": 1.0,
                "details": {"field_count": field_count}
            })
        
        # Calculate validation score
        total_weight = sum(check["weight"] for check in check_results)
        passed_weight = sum(check["weight"] for check in check_results if check["passed"])
        
        if total_weight > 0:
            validation["score"] = passed_weight / total_weight
            validation["passed"] = validation["score"] > 0.6
        
        validation["checks"] = check_results
        return validation
    
    def _calculate_confidence_scores(self, document: MultiModalDocument) -> Dict[str, float]:
        """Calculate confidence scores for different aspects - CORRECTED"""
        scores = {
            "overall": 0.0,
            "text_extraction": 0.0,
            "visual_analysis": 0.0,
            "entity_extraction": 0.0,
            "fusion": 0.0,
            "validation": 0.0
        }
        
        # Text extraction confidence
        if hasattr(document, 'ocr_results') and document.ocr_results:
            try:
                text_confidences = [ocr.average_confidence for ocr in document.ocr_results.values()]
                scores["text_extraction"] = sum(text_confidences) / len(text_confidences)
            except:
                scores["text_extraction"] = 0.5
        
        # Visual analysis confidence
        if hasattr(document, 'visual_elements') and document.visual_elements:
            try:
                visual_confidences = [elem.confidence for elem in document.visual_elements]
                scores["visual_analysis"] = sum(visual_confidences) / len(visual_confidences)
            except:
                scores["visual_analysis"] = 0.5
        
        # Entity extraction confidence
        if hasattr(document, 'extracted_entities') and document.extracted_entities:
            try:
                # Check if we have any entities
                has_entities = any(document.extracted_entities.values())
                scores["entity_extraction"] = 0.7 if has_entities else 0.3
            except:
                scores["entity_extraction"] = 0.3
        
        # Fusion confidence
        if hasattr(document, 'aligned_data') and document.aligned_data:
            try:
                scores["fusion"] = document.aligned_data.get("fusion_confidence", 0.5)
            except:
                scores["fusion"] = 0.5
        
        # Validation confidence
        if "reasoning" in document.processing_metadata:
            try:
                validation_score = document.processing_metadata["reasoning"]["validation_results"]["score"]
                scores["validation"] = validation_score
            except:
                scores["validation"] = 0.5
        
        # Overall confidence (weighted average)
        weights = {
            "text_extraction": 0.25,
            "visual_analysis": 0.20,
            "entity_extraction": 0.15,
            "fusion": 0.25,
            "validation": 0.15
        }
        
        overall = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if scores[key] > 0:
                overall += scores[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            scores["overall"] = overall / total_weight
        else:
            scores["overall"] = 0.5
        
        return scores
    
    def _detect_contradictions(self, document: MultiModalDocument) -> List[Contradiction]:
        """Detect contradictions and inconsistencies - CORRECTED"""
        contradictions = []
        
        # Check 1: Text vs Visual contradictions
        if hasattr(document, 'aligned_data') and document.aligned_data:
            try:
                alignment_count = len(document.aligned_data.get("text_visual_alignment", []))
                visual_count = len(document.visual_elements) if hasattr(document, 'visual_elements') else 0
                
                if visual_count > 0 and alignment_count == 0:
                    contradictions.append(Contradiction(
                        contradiction_type=ContradictionType.CHART_TEXT_CONFLICT,
                        severity=SeverityLevel.MEDIUM,
                        field_a="visual_elements",
                        field_b="text_alignment",
                        value_a=f"{visual_count} elements",
                        value_b="No alignments",
                        explanation="Visual elements detected but no text alignment found",
                        confidence=0.7,
                        recommendation="Check OCR quality or adjust alignment thresholds"
                    ))
            except:
                pass  # Skip if there's an error
        
        # Check 2: Multiple values for same field type
        if hasattr(document, 'extracted_entities') and document.extracted_entities:
            try:
                for entity_type, entities in document.extracted_entities.items():
                    if isinstance(entities, list) and len(entities) > 3:  # Too many values of same type
                        # Convert to strings and get unique count
                        entity_strings = []
                        for entity in entities:
                            if isinstance(entity, dict):
                                entity_strings.append(str(entity.get("value", entity)))
                            else:
                                entity_strings.append(str(entity))
                        
                        unique_values = len(set(entity_strings))
                        
                        if unique_values > 1:
                            contradictions.append(Contradiction(
                                contradiction_type=ContradictionType.NUMERIC_INCONSISTENCY 
                                if entity_type in ["amounts", "dates"] 
                                else ContradictionType.DATA_TYPE_MISMATCH,
                                severity=SeverityLevel.LOW,
                                field_a=f"{entity_type}_count",
                                field_b=f"{entity_type}_unique",
                                value_a=len(entities),
                                value_b=unique_values,
                                explanation=f"Multiple {entity_type} values found: {len(entities)} total, {unique_values} unique",
                                confidence=0.6,
                                recommendation=f"Verify which {entity_type} value is correct"
                            ))
            except:
                pass  # Skip if there's an error
        
        # Check 3: Missing signatures for document types that typically have them
        if hasattr(document, 'document_type') and document.document_type:
            try:
                if document.document_type in [DocumentType.CONTRACT, DocumentType.INVOICE]:
                    has_signature = any(
                        elem.element_type == "signature" for elem in document.visual_elements
                    ) if hasattr(document, 'visual_elements') else False
                    
                    if not has_signature:
                        contradictions.append(Contradiction(
                            contradiction_type=ContradictionType.SIGNATURE_ABSENCE,
                            severity=SeverityLevel.HIGH if document.document_type == DocumentType.CONTRACT else SeverityLevel.MEDIUM,
                            field_a="document_type",
                            field_b="signature_present",
                            value_a=document.document_type.value,
                            value_b=False,
                            explanation=f"{document.document_type.value} typically requires a signature but none detected",
                            confidence=0.8,
                            recommendation="Check for signatures or confirm if document is unsigned"
                        ))
            except:
                pass  # Skip if there's an error
        
        # Check 4: Date inconsistencies
        if hasattr(document, 'extracted_entities') and "dates" in document.extracted_entities:
            try:
                dates = document.extracted_entities["dates"]
                if isinstance(dates, list) and len(dates) > 1:
                    # Get first few dates as strings
                    date_strings = []
                    for date in dates[:3]:
                        if isinstance(date, dict):
                            date_strings.append(str(date.get("value", date)))
                        else:
                            date_strings.append(str(date))
                    
                    contradictions.append(Contradiction(
                        contradiction_type=ContradictionType.DATE_MISMATCH,
                        severity=SeverityLevel.LOW,
                        field_a="date_count",
                        field_b="date_values",
                        value_a=len(dates),
                        value_b=", ".join(date_strings),
                        explanation=f"Multiple dates found in document: {len(dates)} total",
                        confidence=0.5,
                        recommendation="Verify which date is correct for this document"
                    ))
            except:
                pass  # Skip if there's an error
        
        return contradictions
    
    def _assess_risk(self, document: MultiModalDocument, contradictions: List[Contradiction]) -> Dict[str, Any]:
        """Assess document risk based on various factors - CORRECTED"""
        risk_score = 0.0
        risk_factors = []
        
        # Factor 1: Document type risk
        if hasattr(document, 'document_type') and document.document_type:
            doc_type_risk = {
                DocumentType.CONTRACT: 0.7,
                DocumentType.FINANCIAL_REPORT: 0.6,
                DocumentType.INVOICE: 0.5,
                DocumentType.FORM: 0.3,
                DocumentType.RESEARCH_PAPER: 0.2,
                DocumentType.UNKNOWN: 0.5
            }
            
            risk = doc_type_risk.get(document.document_type, 0.4)
            risk_score += risk * 0.2
            risk_factors.append({
                "factor": "document_type",
                "risk": risk,
                "weight": 0.2
            })
        
        # Factor 2: Missing signatures for important docs
        if hasattr(document, 'document_type') and document.document_type:
            if document.document_type in [DocumentType.CONTRACT, DocumentType.INVOICE]:
                has_signature = any(
                    elem.element_type == "signature" for elem in document.visual_elements
                ) if hasattr(document, 'visual_elements') else False
                
                if not has_signature:
                    risk_score += 0.3
                    risk_factors.append({
                        "factor": "missing_signature",
                        "risk": 0.3,
                        "weight": 0.15
                    })
        
        # Factor 3: OCR quality
        if hasattr(document, 'ocr_results') and document.ocr_results:
            try:
                avg_confidence = sum(ocr.average_confidence for ocr in document.ocr_results.values()) / len(document.ocr_results)
                if avg_confidence < 0.7:
                    risk_adjustment = (0.7 - avg_confidence) * 0.5
                    risk_score += risk_adjustment
                    risk_factors.append({
                        "factor": "low_ocr_confidence",
                        "risk": risk_adjustment,
                        "weight": 0.2,
                        "details": {"average_confidence": avg_confidence}
                    })
            except:
                pass  # Skip if can't calculate
        
        # Factor 4: Contradictions
        contradiction_risk = len(contradictions) * 0.05
        risk_score += contradiction_risk
        risk_factors.append({
            "factor": "contradictions",
            "risk": contradiction_risk,
            "weight": 0.25,
            "details": {"count": len(contradictions)}
        })
        
        # Factor 5: Missing key fields
        expected_fields = {
            DocumentType.INVOICE: ["date", "amount", "identifier"],
            DocumentType.CONTRACT: ["date", "name", "signature"],
            DocumentType.FINANCIAL_REPORT: ["date", "amount", "table"]
        }
        
        if hasattr(document, 'document_type') and document.document_type in expected_fields:
            if hasattr(document, 'extracted_fields') and document.extracted_fields:
                missing_fields = []
                for field in expected_fields[document.document_type]:
                    # Check if field exists in extracted fields
                    field_exists = any(
                        field in key.lower() for key in document.extracted_fields.keys()
                    )
                    
                    if not field_exists:
                        missing_fields.append(field)
                
                if missing_fields:
                    risk_adjustment = len(missing_fields) * 0.1
                    risk_score += risk_adjustment
                    risk_factors.append({
                        "factor": "missing_expected_fields",
                        "risk": risk_adjustment,
                        "weight": 0.2,
                        "details": {"missing_fields": missing_fields}
                    })
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "CRITICAL"
        elif risk_score > 0.5:
            risk_level = "HIGH"
        elif risk_score > 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "overall_risk": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "normalized_score": risk_score
        }
    
    def _generate_recommendations(self, document: MultiModalDocument, 
                                contradictions: List[Contradiction], 
                                risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate review recommendations based on analysis - CORRECTED"""
        recommendations = []
        
        # Based on risk level
        risk_level = risk_assessment.get("risk_level", "LOW")
        
        if risk_level == "CRITICAL":
            recommendations.append("🔴 CRITICAL: Immediate manual review required. Document has high-risk issues.")
        elif risk_level == "HIGH":
            recommendations.append("🟠 HIGH: Manual review recommended. Multiple issues detected.")
        elif risk_level == "MEDIUM":
            recommendations.append("🟡 MEDIUM: Recommended review. Some issues found.")
        else:
            recommendations.append("🟢 LOW: Automated processing sufficient. No significant issues.")
        
        # Specific recommendations based on contradictions
        for contradiction in contradictions:
            if contradiction.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                recommendations.append(f"⚠️ Review: {contradiction.explanation}")
        
        # OCR quality recommendations
        if hasattr(document, 'ocr_results') and document.ocr_results:
            try:
                avg_confidence = sum(ocr.average_confidence for ocr in document.ocr_results.values()) / len(document.ocr_results)
                if avg_confidence < 0.6:
                    recommendations.append("🔍 Low OCR confidence detected. Consider rescanning document.")
            except:
                pass  # Skip if can't calculate
        
        # Missing signatures for important docs
        if hasattr(document, 'document_type') and document.document_type:
            if document.document_type in [DocumentType.CONTRACT, DocumentType.INVOICE]:
                has_signature = any(
                    elem.element_type == "signature" for elem in document.visual_elements
                ) if hasattr(document, 'visual_elements') else False
                
                if not has_signature:
                    recommendations.append("📝 No signature detected. Verify if document should be signed.")
        
        # Missing expected fields
        if hasattr(document, 'extracted_fields') and document.extracted_fields:
            field_count = len(document.extracted_fields)
            if field_count < 3:
                recommendations.append(f"ℹ️ Only {field_count} fields extracted. Document may be incomplete.")
        
        # Add positive feedback if everything looks good
        if not recommendations or (len(recommendations) == 1 and "🟢" in recommendations[0]):
            recommendations.append("✅ Document processed successfully with high confidence.")
        
        return recommendations