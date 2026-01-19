import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class PDFProcessor:
    """PDF-specific processing utilities"""
    
    def __init__(self):
        logger.info("✅ PDFProcessor initialized")
    
    def extract_text_direct(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text directly from PDF using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary of text per page
        """
        try:
            logger.info(f"📖 Extracting text directly from PDF: {pdf_path}")
            
            text_by_page = {}
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_by_page[page_num] = text
            
            doc.close()
            
            logger.info(f"✅ Direct text extraction completed: {len(text_by_page)} pages")
            return text_by_page
            
        except Exception as e:
            logger.error(f"❌ Direct PDF text extraction failed: {e}", exc_info=True)
            return {}
    
    def extract_images_from_pdf(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """
        Extract images from PDF using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image extraction
            
        Returns:
            List of extracted images as numpy arrays
        """
        try:
            logger.info(f"🖼️ Extracting images from PDF: {pdf_path}")
            
            images = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get page as image
                pix = page.get_pixmap(dpi=dpi)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    images.append(img)
            
            doc.close()
            
            logger.info(f"✅ Extracted {len(images)} images from PDF")
            return images
            
        except Exception as e:
            logger.error(f"❌ PDF image extraction failed: {e}", exc_info=True)
            return []
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDF metadata dictionary
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            # Add additional information
            metadata["page_count"] = len(doc)
            metadata["file_size"] = os.path.getsize(pdf_path)
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to get PDF metadata: {e}")
            return {}
    
    def search_in_pdf(self, pdf_path: str, search_term: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        Search for text in PDF
        
        Args:
            pdf_path: Path to PDF file
            search_term: Text to search for
            
        Returns:
            Search results per page
        """
        try:
            results = {}
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_instances = page.search_for(search_term)
                
                if text_instances:
                    results[page_num] = [
                        {
                            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                            "text": search_term
                        }
                        for rect in text_instances
                    ]
            
            doc.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ PDF search failed: {e}")
            return {}
    
    def merge_pdf_text_and_images(self, pdf_path: str, 
                                 extracted_text: Dict[int, str],
                                 extracted_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Merge text and image extraction results
        
        Args:
            pdf_path: Path to PDF file
            extracted_text: Text extracted from PDF
            extracted_images: Images extracted from PDF
            
        Returns:
            Merged results
        """
        try:
            metadata = self.get_pdf_metadata(pdf_path)
            
            merged_results = {
                "metadata": metadata,
                "text_by_page": extracted_text,
                "image_count": len(extracted_images),
                "total_text_length": sum(len(text) for text in extracted_text.values()),
                "pages_with_text": len(extracted_text)
            }
            
            return merged_results
            
        except Exception as e:
            logger.error(f"❌ PDF merge failed: {e}")
            return {}