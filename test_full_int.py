# test_full_int.py - SIMPLIFIED
import sys
import os
import asyncio
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.models import MultiModalDocument
from app.cv.visual_pipeline import VisualPipeline

async def test_full_integration():
    """Test full integration without agents"""
    print("🧪 Testing Full CV Integration...")
    
    # Create test image
    height, width = 1000, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add realistic elements
    cv2.rectangle(image, (50, 30), (750, 120), (245, 245, 245), -1)
    cv2.putText(image, "TEST INVOICE", (100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Add table
    cv2.rectangle(image, (50, 150), (750, 450), (250, 250, 250), -1)
    cv2.line(image, (50, 150), (750, 150), (200, 200, 200), 2)
    cv2.line(image, (50, 450), (750, 450), (200, 200, 200), 2)
    
    # Add signature area
    cv2.rectangle(image, (500, 500), (700, 550), (248, 248, 248), -1)
    
    print(f"Created test document: {width}x{height}")
    
    # Test VisualPipeline
    print("\n🔄 Testing VisualPipeline...")
    pipeline = VisualPipeline()
    layout_regions, visual_elements = pipeline.process_page(image, 0)
    
    print(f"✅ Results:")
    print(f"  - Layout regions: {len(layout_regions)}")
    print(f"  - Visual elements: {len(visual_elements)}")
    
    # Create MultiModalDocument
    print("\n📄 Creating MultiModalDocument...")
    doc = MultiModalDocument(
        document_id="integration_test",
        file_path="test_document.jpg",
        file_type=".jpg"
    )
    doc.images = [image]
    doc.layout_regions = layout_regions
    doc.visual_elements = visual_elements
    
    # Test conversion to ProcessingState
    print("\n🔄 Testing ProcessingState conversion...")
    try:
        state = doc.to_processing_state()
        print(f"✅ ProcessingState created successfully!")
        print(f"  - Document ID: {state.document_id}")
        print(f"  - Visual elements: {sum(len(v) for v in state.visual_elements.values())}")
        
        # Verify bbox types
        for page_num, elements in state.visual_elements.items():
            for i, elem in enumerate(elements[:2]):
                if isinstance(elem.bbox, list):
                    print(f"    ✓ Page {page_num}, Element {i+1}: Bbox is list of {len(elem.bbox)} values")
                    print(f"      Values: {elem.bbox[:4]}")
                    if all(isinstance(coord, (int, np.integer)) for coord in elem.bbox):
                        print(f"      ✓ All values are integers")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save visualization
    print("\n🎨 Saving visualization...")
    vis_image = pipeline.visualize_results(image, layout_regions, visual_elements)
    cv2.imwrite("full_integration_test.jpg", vis_image)
    print("✅ Visualization saved: full_integration_test.jpg")
    
    print("\n🎉 INTEGRATION TEST COMPLETE!")
    return True

async def main():
    """Main test function"""
    print("="*60)
    print("📊 COMPREHENSIVE DOCUMENT ANALYSIS TEST")
    print("="*60)
    
    success = await test_full_integration()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED! Ready for Phase 3.")
    else:
        print("❌ TESTS FAILED. Check logs for details.")
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)