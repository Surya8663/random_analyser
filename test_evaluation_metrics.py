# test_evaluation_metrics.py - CORRECTED
import asyncio
import json
import tempfile
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.models import MultiModalDocument, ExplainableField, ProvenanceRecord, BoundingBox
from app.eval.metrics import MetricsCalculator
from app.eval.evaluator import DocumentEvaluator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TestEvaluationMetrics:
    """Test evaluation metrics module"""
    
    @staticmethod
    def create_test_document_with_fields():
        """Create a test document with extracted fields"""
        document = MultiModalDocument(
            document_id="test_eval_001",
            file_path="test_invoice.jpg",
            file_type="image/jpeg",
            raw_text="Invoice No: INV-001\nDate: 2024-01-15\nTotal: $1000.00"
        )
        
        # Add OCR results simulation
        from app.core.models import OCRResult, OCRWord
        ocr_result = OCRResult(
            page_num=0,
            text="Invoice No: INV-001\nDate: 2024-01-15\nTotal: $1000.00",
            words=[
                OCRWord(text="Invoice", bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.15), confidence=0.9, page_num=0),
                OCRWord(text="No:", bbox=BoundingBox(x1=0.21, y1=0.1, x2=0.25, y2=0.15), confidence=0.9, page_num=0),
                OCRWord(text="INV-001", bbox=BoundingBox(x1=0.26, y1=0.1, x2=0.35, y2=0.15), confidence=0.85, page_num=0),
            ],
            average_confidence=0.88,
            image_shape=(800, 600)
        )
        document.ocr_results = {0: ocr_result}
        
        # Add extracted entities
        document.extracted_entities = {
            "dates": ["2024-01-15"],
            "amounts": ["$1000.00"],
            "identifiers": ["INV-001"]
        }
        
        # Add aligned data
        document.aligned_data = {
            "text_visual_alignment": [
                {"alignment_confidence": 0.8},
                {"alignment_confidence": 0.7}
            ],
            "consistency_metrics": {
                "overall_consistency": 0.75
            },
            "fusion_confidence": 0.8
        }
        
        # Add visual elements
        from app.core.models import EnhancedVisualElement
        document.visual_elements = [
            EnhancedVisualElement(
                element_type="signature",
                bbox=BoundingBox(x1=0.7, y1=0.8, x2=0.9, y2=0.9),
                confidence=0.8,
                page_num=0
            )
        ]
        
        # Add contradictions
        from app.core.models import Contradiction, ContradictionType, SeverityLevel
        document.contradictions = [
            Contradiction(
                contradiction_type=ContradictionType.NUMERIC_INCONSISTENCY,
                severity=SeverityLevel.MEDIUM,
                explanation="Possible calculation error",
                confidence=0.7
            )
        ]
        
        # Add extracted fields with provenance
        provenance = ProvenanceRecord(
            agent_name="TextAgent",
            extraction_method="regex_pattern",
            source_modality="text",
            confidence=0.85,
            reasoning_notes="Extracted from OCR text"
        )
        
        document.add_field_with_provenance(
            field_name="invoice_number",
            field_type="text",
            value="INV-001",
            confidence=0.85,
            provenance=provenance,
            modality_sources=["text"]
        )
        
        return document
    
    def test_field_metrics_calculation(self):
        """Test field metrics calculation"""
        print("🧪 Testing Field Metrics Calculation...")
        
        document = self.create_test_document_with_fields()
        metrics = MetricsCalculator.calculate_field_metrics(document)
        
        required_metrics = ["field_precision", "field_recall", "field_f1", "field_coverage"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0 <= metrics[metric] <= 1, f"Invalid {metric} value: {metrics[metric]}"
        
        print(f"✅ Field metrics: {metrics}")
        return True
    
    def test_entity_metrics_calculation(self):
        """Test entity metrics calculation"""
        print("🧪 Testing Entity Metrics Calculation...")
        
        document = self.create_test_document_with_fields()
        metrics = MetricsCalculator.calculate_entity_metrics(document)
        
        required_metrics = ["entity_precision", "entity_recall", "entity_f1"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0 <= metrics[metric] <= 1, f"Invalid {metric} value: {metrics[metric]}"
        
        print(f"✅ Entity metrics: {metrics}")
        return True
    
    def test_alignment_metrics_calculation(self):
        """Test alignment metrics calculation"""
        print("🧪 Testing Alignment Metrics Calculation...")
        
        document = self.create_test_document_with_fields()
        metrics = MetricsCalculator.calculate_alignment_metrics(document)
        
        required_metrics = ["alignment_accuracy", "cross_modal_consistency"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0 <= metrics[metric] <= 1, f"Invalid {metric} value: {metrics[metric]}"
        
        print(f"✅ Alignment metrics: {metrics}")
        return True
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation"""
        print("🧪 Testing Risk Metrics Calculation...")
        
        document = self.create_test_document_with_fields()
        metrics = MetricsCalculator.calculate_risk_metrics(document)
        
        required_metrics = ["risk_detection_precision", "risk_detection_recall"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 0 <= metrics[metric] <= 1, f"Invalid {metric} value: {metrics[metric]}"
        
        print(f"✅ Risk metrics: {metrics}")
        return True
    
    def test_complete_metrics_report(self):
        """Test complete metrics report generation"""
        print("🧪 Testing Complete Metrics Report...")
        
        document = self.create_test_document_with_fields()
        processing_time = 2.5  # Simulated processing time
        
        report = MetricsCalculator.generate_complete_report(document, processing_time)
        
        # Check report structure
        assert hasattr(report, 'document_id'), "Report should have document_id"
        assert hasattr(report, 'metrics'), "Report should have metrics"
        
        # Check metrics
        metrics = report.metrics
        assert 0 <= metrics.overall_accuracy <= 1, f"Invalid overall_accuracy: {metrics.overall_accuracy}"
        assert 0 <= metrics.processing_success_rate <= 1, f"Invalid success_rate: {metrics.processing_success_rate}"
        assert metrics.processing_time_seconds == processing_time, f"Processing time mismatch"
        
        print(f"✅ Complete report generated with accuracy: {metrics.overall_accuracy:.2f}")
        return True
    
    async def test_evaluator_with_ground_truth(self):
        """Test evaluator with ground truth"""
        print("🧪 Testing Evaluator with Ground Truth...")
        
        # Create temporary ground truth file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ground_truth = {
                "invoice_number": {
                    "value": "INV-001",
                    "type": "text",
                    "weight": 1.0
                },
                "invoice_date": {
                    "value": "2024-01-15",
                    "type": "date",
                    "weight": 0.8
                },
                "total_amount": {
                    "value": 1000.00,
                    "type": "number",
                    "weight": 1.2
                }
            }
            json.dump(ground_truth, f)
            ground_truth_file = f.name
        
        try:
            document = self.create_test_document_with_fields()
            evaluator = DocumentEvaluator()
            
            # Evaluate document
            report = await evaluator.evaluate_document(document, ground_truth_file)
            
            # Check report
            assert report is not None, "Evaluation report should be generated"
            assert hasattr(report, 'metrics'), "Report should have metrics"
            assert report.has_ground_truth, "Should have ground truth flag"
            
            # Check field comparisons
            assert report.field_comparisons, "Should have field comparisons with ground truth"
            
            print(f"✅ Evaluator with ground truth test passed")
            return True
            
        finally:
            # Clean up
            Path(ground_truth_file).unlink(missing_ok=True)
    
    async def test_evaluation_report_saving(self):
        """Test evaluation report saving"""
        print("🧪 Testing Evaluation Report Saving...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = DocumentEvaluator(output_dir=temp_dir)
            document = self.create_test_document_with_fields()
            
            # Evaluate and save report
            report = await evaluator.evaluate_document(document)
            
            # Check if file was created
            report_files = list(Path(temp_dir).glob("eval_*.json"))
            assert len(report_files) == 1, f"Expected 1 report file, found {len(report_files)}"
            
            # Load and verify the report
            with open(report_files[0], 'r') as f:
                saved_report = json.load(f)
                assert saved_report["document_id"] == document.document_id, "Document ID mismatch"
            
            print(f"✅ Evaluation report saved successfully")
            return True

async def run_evaluation_tests():
    """Run all evaluation tests"""
    print("=" * 60)
    print("RUNNING PHASE 4 EVALUATION TESTS")
    print("=" * 60)
    
    test = TestEvaluationMetrics()
    
    try:
        # Test 1: Field metrics
        print("\n1. Testing field metrics calculation...")
        if not test.test_field_metrics_calculation():
            return False
        
        # Test 2: Entity metrics
        print("\n2. Testing entity metrics calculation...")
        if not test.test_entity_metrics_calculation():
            return False
        
        # Test 3: Alignment metrics
        print("\n3. Testing alignment metrics calculation...")
        if not test.test_alignment_metrics_calculation():
            return False
        
        # Test 4: Risk metrics
        print("\n4. Testing risk metrics calculation...")
        if not test.test_risk_metrics_calculation():
            return False
        
        # Test 5: Complete report
        print("\n5. Testing complete metrics report...")
        if not test.test_complete_metrics_report():
            return False
        
        # Test 6: Evaluator with ground truth
        print("\n6. Testing evaluator with ground truth...")
        if not await test.test_evaluator_with_ground_truth():
            return False
        
        # Test 7: Report saving
        print("\n7. Testing evaluation report saving...")
        if not await test.test_evaluation_report_saving():
            return False
        
        print("\n" + "=" * 60)
        print("🎉 ALL EVALUATION TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_evaluation_tests())
    exit(0 if success else 1)