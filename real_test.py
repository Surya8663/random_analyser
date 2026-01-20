# test_real_upload.py
import asyncio
import tempfile
import json
from pathlib import Path

async def test_real_phase4_upload():
    """Test actual document upload and processing"""
    print("🚀 TESTING REAL DOCUMENT UPLOAD TO PHASE 4")
    print("=" * 60)
    
    # Create a realistic test document
    test_content = """INVOICE

Invoice Number: INV-2024-001
Date: January 21, 2024
Customer: Acme Corporation
Total Amount: $1,250.75
Due Date: February 21, 2024

Description: Consulting Services
Quantity: 25 hours
Rate: $50.03/hour
Subtotal: $1,250.75

Terms: Net 30
"""
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        f.write(test_content)
        file_path = f.name
    
    try:
        # Import the actual document processor
        from app.services.document_processor import DocumentProcessor
        from app.agents.orchestrator import Phase3Orchestrator
        
        print("\n1. Processing document with DocumentProcessor...")
        processor = DocumentProcessor()
        document = await processor.process_document(file_path, "real_upload_test_001")
        
        print(f"   ✅ Document created: {document.document_id}")
        print(f"   ✅ Text extracted: {len(document.raw_text)} chars")
        
        print("\n2. Running Phase 3 orchestrator...")
        orchestrator = Phase3Orchestrator()
        result = await orchestrator.process_document(document)
        
        print(f"   ✅ Processing completed: {result.get('success', False)}")
        print(f"   ✅ Risk score: {result.get('risk_score', 0):.2f}")
        print(f"   ✅ Fields extracted: {result.get('extracted_fields_count', 0)}")
        
        print("\n3. Checking for real extracted data...")
        detailed = result.get("detailed_results", {})
        if "extracted_information" in detailed:
            fields = detailed["extracted_information"].get("fields", {})
            print(f"   Found {len(fields)} extracted fields:")
            for field_name, field_info in list(fields.items())[:5]:  # Show first 5
                print(f"   • {field_name}: {field_info.get('value', 'N/A')}")
        
        print("\n4. Verifying explainability and evaluation...")
        if "explainability" in detailed:
            print("   ✅ Explainability report generated")
            exp_fields = detailed["explainability"].get("field_explanations", {})
            print(f"   Fields explained: {len(exp_fields)}")
        
        if "evaluation" in detailed:
            print("   ✅ Evaluation report generated")
            eval_metrics = detailed["evaluation"].get("metrics", {})
            print(f"   Overall accuracy: {eval_metrics.get('overall_accuracy', 0):.2f}")
        
        print("\n" + "=" * 60)
        print("📊 REAL PROCESSING TEST RESULTS:")
        print(f"   • Document processed: {'Yes' if result.get('success') else 'No'}")
        print(f"   • Real text extraction: {len(document.raw_text) > 0}")
        print(f"   • Real field extraction: {result.get('extracted_fields_count', 0) > 0}")
        print(f"   • Real risk assessment: {'Yes' if 'risk_score' in result else 'No'}")
        print(f"   • Real reports generated: {'Yes' if 'explainability' in detailed else 'No'}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        Path(file_path).unlink(missing_ok=True)

if __name__ == "__main__":
    print("🔍 This test will show if Phase 4 processes REAL documents or uses mock data")
    success = asyncio.run(test_real_phase4_upload())
    exit(0 if success else 1)