# quick_test.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.cv.paddle_layout_analyzer import PaddleLayoutAnalyzer
    print("✅ PaddleLayoutAnalyzer imports successfully")
    analyzer = PaddleLayoutAnalyzer()
    print("✅ PaddleLayoutAnalyzer instantiated")
except Exception as e:
    print(f"❌ Error: {e}")

try:
    from app.cv.document_object_detector import DocumentObjectDetector
    print("✅ DocumentObjectDetector imports successfully")
    detector = DocumentObjectDetector()
    print("✅ DocumentObjectDetector instantiated")
except Exception as e:
    print(f"❌ Error: {e}")