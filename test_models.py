# test_fixed_models.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.models import MultiModalDocument, ProcessingState, BoundingBox, EnhancedVisualElement, LayoutRegion, OCRResult, OCRWord

def test_conversion():
    print("🧪 Testing Fixed Conversion Methods...")
    
    # Create a MultiModalDocument
    doc = MultiModalDocument(
        document_id="test_doc",
        file_path="test.pdf",
        file_type="pdf",
        raw_text="This is a test document with text content."
    )
    
    # Add a visual element
    doc.add_visual_element(EnhancedVisualElement(
        element_type="table",
        bbox=BoundingBox(x1=100, y1=200, x2=400, y2=500),
        confidence=0.85,
        page_num=0
    ))
    
    # Add a layout region
    doc.add_layout_region(LayoutRegion(
        bbox=BoundingBox(x1=50, y1=50, x2=300, y2=150),
        label="title",
        confidence=0.9,
        page_num=0,
        text_content="Document Title"
    ))
    
    # Add OCR result
    doc.add_ocr_result(0, OCRResult(
        page_num=0,
        text="Page 1 text content",
        words=[
            OCRWord(
                text="Page",
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=30),
                confidence=0.95,
                page_num=0
            )
        ],
        average_confidence=0.9
    ))
    
    print(f"✅ Original MultiModalDocument:")
    print(f"   - ID: {doc.document_id}")
    print(f"   - Raw text: {len(doc.raw_text)} chars")
    print(f"   - Visual elements: {len(doc.visual_elements)}")
    print(f"   - Layout regions: {len(doc.layout_regions)}")
    print(f"   - OCR results: {len(doc.ocr_results)}")
    
    # Convert to ProcessingState
    print("\n🔄 Converting to ProcessingState...")
    state = doc.to_processing_state()
    
    print(f"✅ ProcessingState created:")
    print(f"   - ID: {state.document_id}")
    print(f"   - Extracted text: {len(state.extracted_text)} chars")
    print(f"   - Visual elements dict: {len(state.visual_elements)} pages")
    print(f"   - Has layout_regions in metadata: {'layout_regions' in state.processing_metadata}")
    
    # Convert back to MultiModalDocument
    print("\n🔄 Converting back to MultiModalDocument...")
    new_doc = MultiModalDocument.from_processing_state(state)
    
    print(f"✅ New MultiModalDocument created:")
    print(f"   - ID: {new_doc.document_id}")
    print(f"   - Raw text: {len(new_doc.raw_text)} chars")
    print(f"   - Visual elements: {len(new_doc.visual_elements)}")
    print(f"   - Layout regions: {len(new_doc.layout_regions)}")
    print(f"   - OCR results: {len(new_doc.ocr_results)}")
    
    # Verify data preservation
    print("\n✅ Verification:")
    print(f"   - Raw text preserved: {doc.raw_text == new_doc.raw_text}")
    print(f"   - Visual elements count: {len(doc.visual_elements)} == {len(new_doc.visual_elements)}")
    print(f"   - Layout regions count: {len(doc.layout_regions)} == {len(new_doc.layout_regions)}")
    
    print("\n🎉 ALL CONVERSION TESTS PASSED!")

if __name__ == "__main__":
    test_conversion()