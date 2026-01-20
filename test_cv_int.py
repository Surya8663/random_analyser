# test_simple_cv.py - CORRECTED
import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cv.visual_pipeline import VisualPipeline
from app.core.models import MultiModalDocument, BoundingBox

def test_simple():
    """Simple test to verify CV pipeline works"""
    print("🧪 Simple CV Pipeline Test")
    
    # Create a test image
    height, width = 600, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add test elements
    cv2.rectangle(image, (50, 50), (750, 100), (200, 200, 200), -1)
    cv2.putText(image, "TEST DOCUMENT", (100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.rectangle(image, (50, 150), (750, 400), (220, 220, 220), -1)
    cv2.rectangle(image, (50, 450), (200, 500), (240, 240, 240), -1)
    
    print(f"Created test image: {width}x{height}")
    
    # Test VisualPipeline
    print("\n🔄 Testing VisualPipeline...")
    pipeline = VisualPipeline()
    
    layout_regions, visual_elements = pipeline.process_page(image, page_num=0)
    
    print(f"✅ Results:")
    print(f"  - Layout regions: {len(layout_regions)}")
    print(f"  - Visual elements: {len(visual_elements)}")
    
    if layout_regions:
        print(f"  - Layout region types: {[r.label for r in layout_regions]}")
    
    if visual_elements:
        print(f"  - Visual element types: {[e.element_type for e in visual_elements]}")
        print(f"  - Element confidences: {[e.confidence for e in visual_elements]}")
        
        # Verify normalized coordinates
        for i, elem in enumerate(visual_elements):
            bbox = elem.bbox
            print(f"    Element {i+1} bbox: [{bbox.x1:.3f}, {bbox.y1:.3f}, {bbox.x2:.3f}, {bbox.y2:.3f}]")
            print(f"      Normalized: {all(0.0 <= coord <= 1.0 for coord in [bbox.x1, bbox.y1, bbox.x2, bbox.y2])}")
    
    # Test visualization
    print("\n🎨 Testing visualization...")
    vis_image = pipeline.visualize_results(image, layout_regions, visual_elements)
    cv2.imwrite("simple_test_output.jpg", vis_image)
    print(f"✅ Visualization saved to: simple_test_output.jpg")
    
    # Test with MultiModalDocument
    print("\n📄 Testing MultiModalDocument integration...")
    doc = MultiModalDocument(
        document_id="simple_test",
        file_path="simple_test.jpg",
        file_type=".jpg"
    )
    doc.images = [image]
    doc.layout_regions = layout_regions
    doc.visual_elements = visual_elements
    
    print(f"✅ Created MultiModalDocument:")
    print(f"  - ID: {doc.document_id}")
    print(f"  - Images: {len(doc.images)}")
    print(f"  - Layout regions: {len(doc.layout_regions)}")
    print(f"  - Visual elements: {len(doc.visual_elements)}")
    
    # Test conversion to ProcessingState
    try:
        state = doc.to_processing_state()
        print(f"✅ Successfully converted to ProcessingState")
        print(f"  - Visual elements in state: {sum(len(v) for v in state.visual_elements.values())}")
        
        # Verify bbox types in ProcessingState
        for page_num, elements in state.visual_elements.items():
            for i, elem in enumerate(elements[:2]):
                print(f"    Page {page_num}, Element {i+1}: Bbox type: {type(elem.bbox)}")
                if isinstance(elem.bbox, list):
                    print(f"      Bbox values: {elem.bbox[:4]}")
                    print(f"      All ints: {all(isinstance(coord, (int, np.integer)) for coord in elem.bbox)}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
    
    return len(layout_regions) > 0

if __name__ == "__main__":
    success = test_simple()
    if success:
        print("\n🎉 SIMPLE TEST PASSED! CV pipeline is working.")
    else:
        print("\n⚠️ SIMPLE TEST: No layout regions detected.")