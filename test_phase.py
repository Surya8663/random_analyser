# test_phase4_fixed.py
import asyncio
import tempfile
from pathlib import Path

async def test_fixed_phase4():
    print("🧪 Testing FIXED Phase 4 implementation...")
    
    # Create a test file
    test_content = "Test invoice\nInvoice No: INV-001\nAmount: $100.00"
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        f.write(test_content)
        file_path = f.name
    
    try:
        # Test DocumentProcessor
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        document = await processor.process_document(file_path, "test_fixed_001")
        
        print(f"✅ Document created: {document.document_id}")
        print(f"✅ Has text: {len(document.raw_text) > 0}")
        print(f"✅ Has images: {len(document.images) > 0}")
        
        # Test API integration
        from app.api.routes import phase4_background_processing
        print("✅ phase4_background_processing function is available")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(file_path).unlink(missing_ok=True)

if __name__ == "__main__":
    success = asyncio.run(test_fixed_phase4())
    print("\n" + "="*60)
    print("🎉 Phase 4 FIXED!" if success else "❌ Still has issues")