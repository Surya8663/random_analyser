# test_explainability.py - CORRECTED
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.models import MultiModalDocument, ProvenanceRecord, ExplainableField, BoundingBox
from app.explain.provenance import ProvenanceTracker
from app.explain.explainability import ExplainabilityGenerator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TestExplainability:
    """Test explainability module"""
    
    @staticmethod
    def create_test_document():
        """Create a test document with provenance data"""
        document = MultiModalDocument(
            document_id="test_explain_001",
            file_path="test.jpg",
            file_type="image/jpeg"
        )
        
        # Add some test fields with provenance
        provenance1 = ProvenanceRecord(
            agent_name="TextAgent",
            extraction_method="regex_pattern",
            source_modality="text",
            source_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.2),
            source_page=0,
            confidence=0.85,
            reasoning_notes="Extracted using date pattern"
        )
        
        provenance2 = ProvenanceRecord(
            agent_name="VisionAgent",
            extraction_method="visual_detection",
            source_modality="visual",
            source_bbox=BoundingBox(x1=0.4, y1=0.4, x2=0.6, y2=0.5),
            source_page=0,
            confidence=0.75,
            reasoning_notes="Detected signature region"
        )
        
        # Add fields
        document.add_field_with_provenance(
            field_name="invoice_date",
            field_type="date",
            value="2024-01-15",
            confidence=0.85,
            provenance=provenance1,
            modality_sources=["text"]
        )
        
        document.add_field_with_provenance(
            field_name="signature_present",
            field_type="boolean",
            value=True,
            confidence=0.75,
            provenance=provenance2,
            modality_sources=["visual"]
        )
        
        return document
    
    def test_provenance_tracking(self):
        """Test provenance tracking"""
        print("🧪 Testing Provenance Tracking...")
        
        tracker = ProvenanceTracker()
        document = self.create_test_document()
        tracker.start_document(document)
        
        # Record agent activities
        tracker.record_agent_start("TextAgent")
        tracker.record_agent_end("TextAgent", ["invoice_date", "total_amount"])
        
        tracker.record_agent_start("VisionAgent")
        tracker.record_agent_end("VisionAgent", ["signature_present"], ["Low confidence on logo detection"])
        
        # Get timeline
        timeline = tracker.get_processing_timeline()
        
        assert len(timeline) == 2, f"Expected 2 agents in timeline, got {len(timeline)}"
        assert timeline[0]["agent"] == "TextAgent", f"First agent should be TextAgent"
        assert "invoice_date" in timeline[0]["fields_extracted"], "TextAgent should have extracted invoice_date"
        
        print("✅ Provenance tracking test passed")
        return True
    
    def test_explainability_generation(self):
        """Test explainability report generation"""
        print("🧪 Testing Explainability Generation...")
        
        document = self.create_test_document()
        generator = ExplainabilityGenerator()
        
        # Attach generator to document
        generator.attach_to_document(document)
        
        # Generate report
        report = generator.generate_explainability_report(document)
        
        # Verify report structure
        required_keys = ["document_id", "field_explanations", "agent_contributions", 
                        "modality_analysis", "confidence_analysis"]
        
        for key in required_keys:
            assert key in report, f"Missing key in explainability report: {key}"
        
        # Check field explanations
        field_explanations = report.get("field_explanations", {})
        assert len(field_explanations) == 2, f"Expected 2 field explanations, got {len(field_explanations)}"
        
        # Check confidence analysis
        confidence_analysis = report.get("confidence_analysis", {})
        assert "average" in confidence_analysis, "Missing average confidence"
        assert 0 <= confidence_analysis["average"] <= 1, f"Invalid average confidence: {confidence_analysis['average']}"
        
        print("✅ Explainability generation test passed")
        return True
    
    async def test_full_explainability_pipeline(self):
        """Test full explainability pipeline"""
        print("🧪 Testing Full Explainability Pipeline...")
        
        from app.agents.orchestrator import Phase3Orchestrator
        from test_full_pipeline import TestFullPipeline
        
        # Create sample document
        test_pipeline = TestFullPipeline()
        sample_document = test_pipeline.create_sample_document()
        
        # Create orchestrator
        orchestrator = Phase3Orchestrator()
        
        # Process document
        result = await orchestrator.process_document(sample_document)
        
        # Check for explainability report
        detailed_results = result.get("detailed_results", {})
        explainability_report = detailed_results.get("explainability", {})
        
        assert explainability_report is not None, "Explainability report should be generated"
        assert "field_explanations" in explainability_report, "Missing field explanations"
        
        print("✅ Full explainability pipeline test passed")
        return True

async def run_explainability_tests():
    """Run all explainability tests"""
    print("=" * 60)
    print("RUNNING PHASE 4 EXPLAINABILITY TESTS")
    print("=" * 60)
    
    test = TestExplainability()
    
    try:
        # Test 1: Provenance tracking
        print("\n1. Testing provenance tracking...")
        if not test.test_provenance_tracking():
            return False
        
        # Test 2: Explainability generation
        print("\n2. Testing explainability generation...")
        if not test.test_explainability_generation():
            return False
        
        # Test 3: Full pipeline
        print("\n3. Testing full explainability pipeline...")
        if not await test.test_full_explainability_pipeline():
            return False
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXPLAINABILITY TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Explainability tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_explainability_tests())
    exit(0 if success else 1)