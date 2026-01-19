import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class OCRWord:
    """Data class for OCR word results"""
    text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    page_num: int

@dataclass
class OCRResult:
    """Data class for complete OCR results"""
    page_num: int
    text: str
    words: List[OCRWord]
    average_confidence: float
    engine_used: str
    metadata: Dict[str, Any]

class HybridOCREngine:
    """Hybrid OCR Engine with Tesseract"""
    
    def __init__(self):
        try:
            from app.core.config import settings
            self.confidence_threshold = getattr(settings, 'OCR_CONFIDENCE_THRESHOLD', 0.7)
            tesseract_path = getattr(settings, 'TESSERACT_PATH', None)
            
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"✅ Tesseract path set: {tesseract_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load settings: {e}")
            self.confidence_threshold = 0.7
        
        self.tesseract_config = '--oem 3 --psm 6'
        
        # Try to find Tesseract
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Tesseract not found: {e}")
            logger.info("💡 Install instructions in README")
            raise
    
    # Rest of the class remains the same...
    # [Keep all other methods as they were]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Denoise
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            return processed
            
        except Exception as e:
            logger.warning(f"⚠️ Image preprocessing failed: {e}")
            return image
    
    def process_image(self, image: np.ndarray, page_num: int = 0) -> OCRResult:
        """
        Process image with OCR
        
        Args:
            image: Input image (OpenCV format)
            page_num: Page number
            
        Returns:
            OCRResult object
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Convert to PIL Image for Tesseract
            from PIL import Image
            pil_image = Image.fromarray(processed_image)
            
            # Get data with bounding boxes
            data = pytesseract.image_to_data(
                pil_image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract words and confidence
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:  # Non-empty text
                    confidence = float(data['conf'][i])
                    if confidence < 0:  # Tesseract uses -1 for empty
                        continue
                    
                    confidence_normalized = confidence / 100.0
                    
                    word = OCRWord(
                        text=text,
                        confidence=confidence_normalized,
                        bbox=[
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ],
                        page_num=page_num
                    )
                    words.append(word)
                    confidences.append(word.confidence)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Full text
            full_text = pytesseract.image_to_string(pil_image)
            
            result = OCRResult(
                page_num=page_num,
                text=full_text,
                words=words,
                average_confidence=avg_confidence,
                engine_used="tesseract",
                metadata={
                    "total_words": len(words),
                    "min_confidence": min(confidences) if confidences else 0,
                    "max_confidence": max(confidences) if confidences else 0,
                    "config_used": self.tesseract_config
                }
            )
            
            logger.debug(f"Page {page_num}: {len(words)} words, confidence: {avg_confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tesseract OCR failed: {e}")
            return self._create_empty_result(page_num, "tesseract_failed")
    
    def _create_empty_result(self, page_num: int, engine: str) -> OCRResult:
        """Create empty result for failed OCR"""
        return OCRResult(
            page_num=page_num,
            text="",
            words=[],
            average_confidence=0.0,
            engine_used=engine,
            metadata={"error": "OCR processing failed"}
        )
    
    def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Process multiple images"""
        results = []
        for idx, image in enumerate(images):
            try:
                result = self.process_image(image, page_num=idx)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ OCR failed for image {idx}: {e}")
                results.append(self._create_empty_result(idx, "processing_error"))
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the OCR engine"""
        try:
            version = pytesseract.get_tesseract_version()
            return {
                "tesseract_available": True,
                "tesseract_version": version,
                "confidence_threshold": self.confidence_threshold,
                "default_config": self.tesseract_config
            }
        except:
            return {
                "tesseract_available": False,
                "error": "Tesseract not found"
            }