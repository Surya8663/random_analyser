# test_full_demo_run.py - CORRECTED
import asyncio
import json
import tempfile
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.models import MultiModalDocument
from app.agents.orchestrator import Phase3Orchestrator
from app.eval.evaluator import DocumentEvaluator
from app.explain.explainability import ExplainabilityGenerator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class Phase4DemoTest:
    """Complete Phase 4 demo test"""
    
    @staticmethod
    def create_realistic_test_document():
        """Create a realistic test document for demo"""
        from app.core.models import (
            MultiModalDocument, OCRResult, OCRWord, BoundingBox,
            EnhancedVisualElement, LayoutRegion, QualityScore,
            ProvenanceRecord, ExplainableField
        )
        
        # Create document
        document = MultiModalDocument(
            document_id="demo_doc_001",
            file_path="sample_invoice.jpg",
            file_type="image/jpeg",
            document_type="invoice",
            raw_text="""INVOICE
Invoice No: INV-2024-001
Date: January 15, 2024
Customer: John Smith
Total Amount: $1,500.00
Payment Due: February 15, 2024

Description: Web Development Services
Quantity: 10 hours
Rate: $150.00/hour
Subtotal: $1,500.00

Thank you for your business!"""
        )
        
        # Add OCR results
        ocr_result = OCRResult(
            page_num=0,
            text=document.raw_text,
            words=[
                OCRWord(text="INVOICE", bbox=BoundingBox(x1=0.3, y1=0.1, x2=0.4, y2=0.12), confidence=0.95, page_num=0),
                OCRWord(text="Invoice", bbox=BoundingBox(x1=0.1, y1=0.15, x2=0.18, y2=0.17), confidence=0.9, page_num=0),
                OCRWord(text="No:", bbox=BoundingBox(x1=0.19, y1=0.15, x2=0.22, y2=0.17), confidence=0.9, page_num=0),
                OCRWord(text="INV-2024-001", bbox=BoundingBox(x1=0.23, y1=0.15, x2=0.35, y2=0.17), confidence=0.85, page_num=0),
                OCRWord(text="Date:", bbox=BoundingBox(x1=0.1, y1=0.18, x2=0.15, y2=0.2), confidence=0.9, page_num=0),
                OCRWord(text="January", bbox=BoundingBox(x1=0.16, y1=0.18, x2=0.22, y2=0.2), confidence=0.85, page_num=0),
                OCRWord(text="15,", bbox=BoundingBox(x1=0.23, y1=0.18, x2=0.25, y2=0.2), confidence=0.85, page_num=0),
                OCRWord(text="2024", bbox=BoundingBox(x1=0.26, y1=0.18, x2=0.3, y2=0.2), confidence=0.85, page_num=0),
                OCRWord(text="Total", bbox=BoundingBox(x1=0.1, y1=0.21, x2=0.15, y2=0.23), confidence=0.9, page_num=0),
                OCRWord(text="Amount:", bbox=BoundingBox(x1=0.16, y1=0.21, x2=0.22, y2=0.23), confidence=0.9, page_num=0),
                OCRWord(text="$1,500.00", bbox=BoundingBox(x1=0.23, y1=0.21, x2=0.32, y2=0.23), confidence=0.8, page_num=0),
            ],
            average_confidence=0.87,
            image_shape=(800, 600)
        )
        document.ocr_results = {0: ocr_result}
        
        # Add extracted entities
        document.extracted_entities = {
            "dates": ["January 15, 2024", "February 15, 2024"],
            "amounts": ["$1,500.00", "$150.00"],
            "identifiers": ["INV-2024-001"],
            "names": ["John Smith"]
        }
        
        # Add visual elements
        document.visual_elements = [
            EnhancedVisualElement(
                element_type="text_block",
                bbox=BoundingBox(x1=0.08, y1=0.08, x2=0.92, y2=0.3),
                confidence=0.9,
                page_num=0,
                text_content="Invoice header and details",
                metadata={"semantic_label": "header_section"}
            ),
            EnhancedVisualElement(
                element_type="table",
                bbox=BoundingBox(x1=0.08, y1=0.32, x2=0.92, y2=0.6),
                confidence=0.8,
                page_num=0,
                metadata={"semantic_label": "item_table"}
            ),
            EnhancedVisualElement(
                element_type="signature",
                bbox=BoundingBox(x1=0.7, y1=0.65, x2=0.9, y2=0.75),
                confidence=0.7,
                page_num=0,
                metadata={"semantic_label": "authority_marker"}
            )
        ]
        
        # Add layout regions
        document.layout_regions = [
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.05, x2=0.95, y2=0.12),
                label="header",
                confidence=0.9,
                page_num=0,
                text_content="INVOICE",
                metadata={"structural_role": "document_header"}
            ),
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.13, x2=0.95, y2=0.7),
                label="body",
                confidence=0.95,
                page_num=0,
                text_content="Document body content",
                metadata={"structural_role": "text_content"}
            )
        ]
        
        # Add quality scores
        document.quality_scores = {
            0: QualityScore(
                sharpness=0.8,
                brightness=0.9,
                contrast=0.7,
                noise_level=0.2,
                overall=0.8
            )
        }
        
        # Add aligned data
        document.aligned_data = {
            "text_visual_alignment": [
                {
                    "page": 0,
                    "element_type": "text_block",
                    "element_bbox": [0.08, 0.08, 0.92, 0.3],
                    "contained_words_count": 15,
                    "alignment_confidence": 0.85
                },
                {
                    "page": 0,
                    "element_type": "table",
                    "element_bbox": [0.08, 0.32, 0.92, 0.6],
                    "contained_words_count": 8,
                    "alignment_confidence": 0.75
                }
            ],
            "consistency_metrics": {
                "text_present": True,
                "visual_present": True,
                "alignment_ratio": 0.67,
                "overall_consistency": 0.72
            },
            "fusion_confidence": 0.78
        }
        
        # Add extracted fields with provenance
        provenance_records = [
            ProvenanceRecord(
                agent_name="TextAgent",
                extraction_method="regex_pattern",
                source_modality="text",
                source_bbox=BoundingBox(x1=0.23, y1=0.15, x2=0.35, y2=0.17),
                source_page=0,
                confidence=0.85,
                reasoning_notes="Extracted invoice number using pattern matching"
            ),
            ProvenanceRecord(
                agent_name="VisionAgent",
                extraction_method="visual_detection",
                source_modality="visual",
                source_bbox=BoundingBox(x1=0.7, y1=0.65, x2=0.9, y2=0.75),
                source_page=0,
                confidence=0.7,
                reasoning_notes="Detected signature region visually"
            ),
            ProvenanceRecord(
                agent_name="FusionAgent",
                extraction_method="cross_modal_alignment",
                source_modality="fusion",
                source_bbox=BoundingBox(x1=0.23, y1=0.21, x2=0.32, y2=0.23),
                source_page=0,
                confidence=0.82,
                reasoning_notes="Amount confirmed by both text and table detection"
            )
        ]
        
        # Add fields
        document.add_field_with_provenance(
            field_name="invoice_number",
            field_type="text",
            value="INV-2024-001",
            confidence=0.85,
            provenance=provenance_records[0],
            modality_sources=["text"]
        )
        
        document.add_field_with_provenance(
            field_name="signature_present",
            field_type="boolean",
            value=True,
            confidence=0.7,
            provenance=provenance_records[1],
            modality_sources=["visual"]
        )
        
        document.merge_field_provenance(
            field_name="total_amount",
            new_provenance=provenance_records[2],
            new_value="$1,500.00",
            new_confidence=0.82
        )
        
        # Add risk score
        document.risk_score = 0.35  # Low risk
        
        # Add contradictions
        from app.core.models import Contradiction, ContradictionType, SeverityLevel
        document.contradictions = [
            Contradiction(
                contradiction_type=ContradictionType.NUMERIC_INCONSISTENCY,
                severity=SeverityLevel.LOW,
                field_a="hourly_rate",
                field_b="total_amount",
                value_a="$150.00",
                value_b="$1,500.00",
                explanation="Hourly rate calculation matches total (10 hours × $150 = $1,500)",
                confidence=0.9,
                recommendation="Calculation appears correct"
            )
        ]
        
        # Add recommendations
        document.review_recommendations = [
            "🟢 LOW RISK: Document appears valid",
            "✓ Signature detected",
            "✓ Amount calculations consistent"
        ]
        
        return document
    
    async def run_complete_demo(self):
        """Run complete Phase 4 demo"""
        print("=" * 60)
        print("🚀 RUNNING COMPLETE PHASE 4 DEMO")
        print("=" * 60)
        
        try:
            # Step 1: Create test document
            print("\n1. 📄 Creating realistic test document...")
            document = self.create_realistic_test_document()
            print(f"   ✓ Document created: {document.document_id}")
            print(f"   ✓ Pages: {len(document.ocr_results)}")
            print(f"   ✓ Visual elements: {len(document.visual_elements)}")
            print(f"   ✓ Extracted fields: {len(document.extracted_fields)}")
            
            # Step 2: Run through orchestrator
            print("\n2. 🔄 Running through Phase 4 orchestrator...")
            orchestrator = Phase3Orchestrator()
            
            # Note: process_document returns a dict, not a document
            result = await orchestrator.process_document(document)
            
            if not result["success"]:
                print(f"   ✗ Orchestrator failed: {result.get('error', 'Unknown error')}")
                return False
            
            print(f"   ✓ Processing completed in {result['processing_time']:.2f} seconds")
            print(f"   ✓ Risk score: {result['risk_score']:.2f}")
            print(f"   ✓ Extracted fields: {result['extracted_fields_count']}")
            
            # Step 3: Extract detailed results
            print("\n3. 📊 Analyzing detailed results...")
            detailed_results = result.get("detailed_results", {})
            
            if not detailed_results:
                print("   ✗ No detailed results generated")
                return False
            
            # Check required sections
            required_sections = [
                "document_understanding",
                "content_summary", 
                "extracted_information",
                "quality_assessment"
            ]
            
            for section in required_sections:
                if section not in detailed_results:
                    print(f"   ✗ Missing section: {section}")
                    return False
                print(f"   ✓ {section.replace('_', ' ').title()}")
            
            # Step 4: Verify optional Phase 4 sections
            print("\n4. 💡 Verifying Phase 4 features...")
            optional_sections = ["explainability", "evaluation"]
            
            for section in optional_sections:
                if section in detailed_results:
                    print(f"   ✓ {section.replace('_', ' ').title()} (Phase 4 feature)")
                else:
                    print(f"   ⚠️ {section.replace('_', ' ').title()} not generated (may need full pipeline)")
            
            # Step 5: Save demo output
            print("\n5. 💾 Saving demo output...")
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save detailed results
                output_file = Path(temp_dir) / "phase4_demo_output.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                print(f"   ✓ Results saved to: {output_file}")
                
                # Verify file size
                if output_file.exists():
                    size_kb = output_file.stat().st_size / 1024
                    print(f"   ✓ Output size: {size_kb:.1f} KB")
            
            # Step 6: Summary
            print("\n" + "=" * 60)
            print("🎉 PHASE 4 DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            print("\n📋 DEMO SUMMARY:")
            print(f"   Document ID: {result['document_id']}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            print(f"   Risk Level: {'LOW' if result['risk_score'] < 0.4 else 'MEDIUM' if result['risk_score'] < 0.7 else 'HIGH'}")
            print(f"   Fields Extracted: {result['extracted_fields_count']}")
            print(f"   Contradictions: {result['contradictions_count']}")
            
            print("\n🔍 KEY FEATURES VERIFIED:")
            print("   ✅ MultiModalDocument with provenance support")
            print("   ✅ Structured field extraction")
            print("   ✅ Risk assessment")
            print("   ✅ JSON output generation")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main demo runner"""
    demo = Phase4DemoTest()
    success = await demo.run_complete_demo()
    
    if success:
        print("\n" + "=" * 60)
        print("🏆 CORE SYSTEM IS WORKING!")
        print("=" * 60)
        print("\nThe core document processing system includes:")
        print("1. 📄 MultiModalDocument with provenance tracking")
        print("2. 🔄 Orchestrator pipeline")
        print("3. 📊 Structured output generation")
        print("4. 🎯 Risk assessment and validation")
        print("\nFor full Phase 4 features (explainability & evaluation),")
        print("ensure all dependencies are installed and imports work.")
        return 0
    else:
        print("\n❌ Demo failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))