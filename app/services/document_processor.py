import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    """Main document processing service"""
    
    def __init__(self):
        try:
            from app.core.config import settings
            self.settings = settings
            logger.info(f"✅ Settings loaded for DocumentProcessor")
        except Exception as e:
            logger.error(f"❌ Failed to load settings: {e}")
            # Create minimal settings
            class MinimalSettings:
                UPLOAD_DIR = "uploads"
                ALLOWED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg"]
                OCR_CONFIDENCE_THRESHOLD = 0.7
                TESSERACT_PATH = None
            
            self.settings = MinimalSettings()
        
        # Initialize OCR engine with proper error handling
        try:
            from app.models.ocr_engine import HybridOCREngine
            logger.info(f"📝 Initializing OCR engine")
            self.ocr_engine = HybridOCREngine()
            logger.info("✅ OCR engine initialized successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import OCR engine: {e}")
            self.ocr_engine = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR engine: {e}")
            self.ocr_engine = None
    
    async def extract_images(self, file_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Extract images from document file
        """
        try:
            logger.info(f"📄 Extracting from {file_path}")
            
            images = []
            metadata = {
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1].lower(),
                "success": False
            }
            
            file_ext = metadata["file_type"]
            
            if file_ext == '.pdf':
                # Try to extract from PDF
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(dpi=150)
                        img_data = pix.tobytes("png")
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            images.append(img)
                    doc.close()
                    metadata["success"] = True
                    logger.info(f"✅ Extracted {len(images)} pages from PDF")
                except ImportError:
                    logger.warning("⚠️ PyMuPDF not available, using fallback")
                    # Fallback: create dummy image
                    dummy_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
                    images = [dummy_image]
                    metadata["success"] = True
                    metadata["warning"] = "PDF extraction not available"
            
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                # Load image file
                img = cv2.imread(file_path)
                if img is not None:
                    images.append(img)
                    metadata["success"] = True
                    logger.info(f"✅ Loaded image: {img.shape}")
                else:
                    raise ValueError(f"Failed to load image: {file_path}")
            
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            metadata["page_count"] = len(images)
            
            if images:
                # Add image dimensions to metadata
                metadata["image_dimensions"] = [
                    {"width": img.shape[1], "height": img.shape[0]}
                    for img in images
                ]
            
            return images, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to extract images: {e}")
            return [], {"error": str(e), "success": False}
    
    async def extract_text(self, images: List[np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """
        Extract text from images using OCR
        """
        try:
            logger.info(f"🔤 Extracting text from {len(images)} images")
            
            results = {}
            
            if self.ocr_engine:
                for idx, image in enumerate(images):
                    try:
                        # Preprocess image for better OCR
                        if len(image.shape) == 3:
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = image.copy()
                        
                        # Apply thresholding
                        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Perform OCR
                        ocr_result = self.ocr_engine.process_image(processed, idx)
                        
                        results[idx] = {
                            "text": ocr_result.text,
                            "confidence": ocr_result.average_confidence,
                            "word_count": len(ocr_result.words),
                            "engine_used": ocr_result.engine_used
                        }
                        
                        logger.debug(f"Page {idx}: {len(ocr_result.words)} words, confidence: {ocr_result.average_confidence:.2f}")
                    
                    except Exception as page_error:
                        logger.error(f"❌ OCR failed for page {idx}: {page_error}")
                        results[idx] = {
                            "text": f"OCR failed: {str(page_error)}",
                            "confidence": 0,
                            "word_count": 0,
                            "engine_used": "error"
                        }
            else:
                # Mock OCR results
                logger.warning("⚠️ OCR engine not available, using mock results")
                for idx in range(len(images)):
                    results[idx] = {
                        "text": f"This is mock OCR text for page {idx}.\nDocument is being processed in limited mode.",
                        "confidence": 0.5,
                        "word_count": 15,
                        "engine_used": "mock"
                    }
            
            logger.info(f"✅ Text extraction completed for {len(images)} pages")
            return results
            
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
            return {0: {"text": f"OCR failed: {str(e)}", "confidence": 0, "word_count": 0, "engine_used": "error"}}
    
    async def process_document(self, file_path: str, document_id: str = None) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        """
        try:
            logger.info(f"🚀 Processing document: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Extract images
            images, metadata = await self.extract_images(file_path)
            
            if not images:
                raise ValueError(f"No images extracted from {file_path}")
            
            # Extract text
            text_results = await self.extract_text(images)
            
            # Prepare response
            response = {
                "success": True,
                "document_id": document_id or os.path.basename(file_path),
                "file_metadata": metadata,
                "text_results": text_results,
                "total_pages": len(images),
                "total_text_length": sum(len(r.get("text", "")) for r in text_results.values()),
                "avg_confidence": np.mean([r.get("confidence", 0) for r in text_results.values()]) if text_results else 0,
                "processing_mode": "full" if self.ocr_engine else "limited"
            }
            
            logger.info(f"✅ Document processing completed: {response['document_id']}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Document processing failed for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id or "unknown"
            }