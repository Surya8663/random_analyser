# app/eval/evaluator.py - CORRECTED
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from app.core.models import MultiModalDocument, EvaluationReport, GroundTruthField, EvaluationMetrics
from app.eval.metrics import MetricsCalculator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentEvaluator:
    """Main evaluator for document processing results"""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    async def evaluate_document(self, 
                               document: MultiModalDocument,
                               ground_truth_file: Optional[str] = None) -> EvaluationReport:
        """Evaluate a processed document"""
        try:
            logger.info(f"🔍 Evaluating document: {document.document_id}")
            
            # Load ground truth if provided
            if ground_truth_file and Path(ground_truth_file).exists():
                document.ground_truth = self._load_ground_truth(ground_truth_file)
                logger.info(f"Loaded ground truth from {ground_truth_file}")
            
            # Calculate processing time
            processing_time = document.get_processing_time()
            
            # Generate evaluation report
            report = MetricsCalculator.generate_complete_report(document, processing_time)
            
            # If ground truth exists, calculate field comparisons
            if document.ground_truth:
                self._calculate_field_comparisons(document, report)
            
            # Store report in document
            document.evaluation_report = report
            
            # Save report to file
            self._save_report(report, document.document_id)
            
            logger.info(f"✅ Evaluation completed for {document.document_id}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            # Create minimal error report
            return EvaluationReport(
                document_id=document.document_id,
                has_ground_truth=False,
                metrics=EvaluationMetrics(
                    field_precision=0.0,
                    field_recall=0.0,
                    field_f1=0.0,
                    field_coverage=0.0,
                    entity_precision=0.0,
                    entity_recall=0.0,
                    entity_f1=0.0,
                    alignment_accuracy=0.0,
                    cross_modal_consistency=0.0,
                    risk_detection_precision=0.0,
                    risk_detection_recall=0.0,
                    overall_accuracy=0.0,
                    processing_success_rate=0.0,
                    average_confidence=0.0,
                    processing_time_seconds=0.0
                ),
                recommendations=["Evaluation failed - manual review required"]
            )
    
    def _load_ground_truth(self, ground_truth_file: str) -> Dict[str, GroundTruthField]:
        """Load ground truth from JSON file"""
        try:
            with open(ground_truth_file, 'r') as f:
                data = json.load(f)
            
            ground_truth = {}
            for field_name, field_data in data.items():
                ground_truth[field_name] = GroundTruthField(
                    field_name=field_name,
                    true_value=field_data.get("value"),
                    value_type=field_data.get("type", "text"),
                    importance_weight=field_data.get("weight", 1.0)
                )
            
            return ground_truth
            
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return {}
    
    def _calculate_field_comparisons(self, document: MultiModalDocument, 
                                   report: EvaluationReport):
        """Calculate field-level comparisons with ground truth"""
        if not document.ground_truth:
            return
        
        field_comparisons = {}
        
        for field_name, ground_truth_field in document.ground_truth.items():
            extracted_field = document.extracted_fields.get(field_name)
            
            comparison = {
                "ground_truth": ground_truth_field.true_value,
                "has_extracted": extracted_field is not None,
                "importance": ground_truth_field.importance_weight
            }
            
            if extracted_field:
                comparison.update({
                    "extracted_value": extracted_field.value,
                    "confidence": extracted_field.confidence,
                    "is_correct": self._compare_values(
                        extracted_field.value, 
                        ground_truth_field.true_value,
                        ground_truth_field.value_type
                    ),
                    "sources": extracted_field.modality_sources
                })
            
            field_comparisons[field_name] = comparison
        
        report.field_comparisons = field_comparisons
        
        # Update metrics based on ground truth comparison
        self._update_metrics_with_ground_truth(document, report)
    
    def _compare_values(self, extracted: Any, truth: Any, value_type: str) -> bool:
        """Compare extracted value with ground truth"""
        if extracted is None or truth is None:
            return False
        
        try:
            if value_type == "number":
                # Compare numbers with tolerance
                extracted_num = float(extracted) if not isinstance(extracted, (int, float)) else extracted
                truth_num = float(truth) if not isinstance(truth, (int, float)) else truth
                return abs(extracted_num - truth_num) < 0.01
            
            elif value_type == "date":
                # Simple string comparison for dates
                return str(extracted).strip() == str(truth).strip()
            
            else:  # text or other
                # Case-insensitive comparison
                return str(extracted).strip().lower() == str(truth).strip().lower()
                
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(extracted).strip().lower() == str(truth).strip().lower()
    
    def _update_metrics_with_ground_truth(self, document: MultiModalDocument, 
                                        report: EvaluationReport):
        """Update metrics based on ground truth comparison"""
        if not report.field_comparisons:
            return
        
        # Calculate precision and recall
        total_fields = len(report.field_comparisons)
        correct_fields = sum(1 for comp in report.field_comparisons.values() 
                           if comp.get("is_correct", False))
        extracted_fields = sum(1 for comp in report.field_comparisons.values() 
                             if comp.get("has_extracted", False))
        
        # Update metrics
        if total_fields > 0:
            report.metrics.field_precision = correct_fields / extracted_fields if extracted_fields > 0 else 0.0
            report.metrics.field_recall = correct_fields / total_fields
            report.metrics.field_f1 = 2 * report.metrics.field_precision * report.metrics.field_recall / \
                                     (report.metrics.field_precision + report.metrics.field_recall) \
                                     if report.metrics.field_precision + report.metrics.field_recall > 0 else 0.0
            report.metrics.overall_accuracy = report.metrics.field_f1
    
    def _save_report(self, report: EvaluationReport, document_id: str):
        """Save evaluation report to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{document_id}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert to dict
            report_dict = report.dict()
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"📄 Evaluation report saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
    
    def generate_summary_report(self, reports_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate summary report from multiple evaluation reports"""
        if reports_dir:
            report_files = list(Path(reports_dir).glob("eval_*.json"))
        else:
            report_files = list(self.output_dir.glob("eval_*.json"))
        
        if not report_files:
            return {"error": "No evaluation reports found"}
        
        reports = []
        metrics_summary = {
            "total_documents": len(report_files),
            "average_accuracy": 0.0,
            "average_processing_time": 0.0,
            "success_rate": 0.0
        }
        
        accuracy_values = []
        processing_times = []
        success_flags = []
        
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    reports.append(report_data)
                    
                    # Extract metrics
                    metrics = report_data.get("metrics", {})
                    accuracy = metrics.get("overall_accuracy", 0.0)
                    processing_time = metrics.get("processing_time_seconds", 0.0)
                    success_rate = metrics.get("processing_success_rate", 0.0)
                    
                    accuracy_values.append(accuracy)
                    processing_times.append(processing_time)
                    success_flags.append(success_rate > 0.7)
                    
            except Exception as e:
                logger.error(f"Failed to read report {report_file}: {e}")
        
        if accuracy_values:
            metrics_summary["average_accuracy"] = sum(accuracy_values) / len(accuracy_values)
            metrics_summary["average_processing_time"] = sum(processing_times) / len(processing_times)
            metrics_summary["success_rate"] = sum(success_flags) / len(success_flags) if success_flags else 0.0
        
        return {
            "summary": metrics_summary,
            "total_reports": len(reports),
            "generated_at": datetime.now().isoformat()
        }