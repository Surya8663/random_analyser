# app/cv/paddle_layout_analyzer.py - CORRECTED
import cv2
import numpy as np
from typing import List
from app.core.models import LayoutRegion, BoundingBox
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class PaddleLayoutAnalyzer:
    """Windows-friendly layout analysis using PaddleOCR"""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en'):
        self.ocr = None
        
        try:
            logger.info("📦 Attempting to load PaddleOCR...")
            
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                show_log=False,
                use_gpu=False,
                enable_mkldnn=True,
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=1.5
            )
            logger.info("✅ PaddleOCR loaded successfully")
        except ImportError:
            logger.warning("⚠️ PaddleOCR not available. Using OpenCV-based fallback.")
        except Exception as e:
            logger.error(f"❌ Failed to load PaddleOCR: {e}")
    
    def analyze(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Analyze document layout using PaddleOCR (or fallback)
        
        Args:
            image: Document image (BGR or RGB)
            
        Returns:
            List of layout regions with normalized coordinates (0-1)
        """
        height, width = image.shape[:2]
        
        # Fallback if PaddleOCR not available
        if self.ocr is None:
            return self._opencv_fallback_analysis(image, height, width)
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 1:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run PaddleOCR
            result = self.ocr.ocr(image_rgb, cls=True)
            
            layout_regions = []
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        bbox_data, text_data = line
                        
                        # Extract bounding box
                        points = np.array(bbox_data, dtype=np.float32)
                        x_coords = points[:, 0]
                        y_coords = points[:, 1]
                        
                        x_min = float(x_coords.min())
                        y_min = float(y_coords.min())
                        x_max = float(x_coords.max())
                        y_max = float(y_coords.max())
                        
                        # Normalize coordinates
                        x1 = max(0.0, min(x_min / width, 1.0))
                        y1 = max(0.0, min(y_min / height, 1.0))
                        x2 = max(0.0, min(x_max / width, 1.0))
                        y2 = max(0.0, min(y_max / height, 1.0))
                        
                        # Skip too small regions
                        if (x2 - x1) * (y2 - y1) < 0.001:
                            continue
                        
                        # Extract text and confidence
                        text_content = text_data[0] if text_data else ""
                        confidence = float(text_data[1]) if len(text_data) > 1 else 0.7
                        
                        # Classify region type
                        region_type = self._classify_region(text_content, x1, y1, x2, y2)
                        
                        # Create layout region
                        region = LayoutRegion(
                            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                            label=region_type,
                            confidence=confidence,
                            page_num=0,
                            text_content=text_content[:200]
                        )
                        
                        layout_regions.append(region)
            
            logger.info(f"✅ PaddleOCR detected {len(layout_regions)} text regions")
            return layout_regions
            
        except Exception as e:
            logger.error(f"❌ PaddleOCR analysis failed: {e}")
            return self._opencv_fallback_analysis(image, height, width)
    
    def _classify_region(self, text: str, x1: float, y1: float, 
                        x2: float, y2: float) -> str:
        """Classify region based on text content and position"""
        text_lower = text.lower()
        
        # Position-based classification
        if y1 < 0.1:
            return "header"
        elif y2 > 0.9:
            return "footer"
        elif y1 < 0.2:
            return "title" if len(text) < 50 else "header"
        
        # Content-based classification
        table_keywords = ["table", "total", "sum", "amount", "price", "qty"]
        if any(keyword in text_lower for keyword in table_keywords):
            return "table"
        
        signature_keywords = ["signature", "signed", "authorized", "approved", "sign"]
        if any(keyword in text_lower for keyword in signature_keywords):
            return "signature"
        
        # Shape-based classification
        aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
        if aspect_ratio > 2.0:
            return "table_row"
        elif aspect_ratio < 0.5:
            return "sidebar"
        elif len(text) > 100:
            return "paragraph"
        elif len(text.split()) > 3:
            return "text"
        else:
            return "label"
    
    def _opencv_fallback_analysis(self, image: np.ndarray, height: int, width: int) -> List[LayoutRegion]:
        """OpenCV-based fallback layout analysis"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            layout_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Normalize coordinates
                x1 = x / width
                y1 = y / height
                x2 = (x + w) / width
                y2 = (y + h) / height
                
                if (x2 - x1) * (y2 - y1) < 0.005:
                    continue
                
                # Classify region
                aspect_ratio = w / h if h > 0 else 1.0
                
                if y1 < 0.15:
                    region_type = "header"
                elif y2 > 0.85:
                    region_type = "footer"
                elif aspect_ratio > 2.0:
                    region_type = "table"
                elif area > 10000:
                    region_type = "text_block"
                else:
                    region_type = "element"
                
                region = LayoutRegion(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    label=region_type,
                    confidence=0.6,
                    page_num=0,
                    text_content=None
                )
                
                layout_regions.append(region)
            
            # If no regions found, create default ones
            if not layout_regions:
                layout_regions = self._create_default_regions(height, width)
            
            logger.info(f"⚠️ OpenCV fallback detected {len(layout_regions)} regions")
            return layout_regions
            
        except Exception as e:
            logger.error(f"❌ OpenCV fallback failed: {e}")
            return self._create_default_regions(height, width)
    
    def _create_default_regions(self, height: int, width: int) -> List[LayoutRegion]:
        """Create default layout regions"""
        return [
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.05, x2=0.95, y2=0.15),
                label="header",
                confidence=0.7,
                page_num=0,
                text_content="Document Header Area"
            ),
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.2, x2=0.95, y2=0.8),
                label="text",
                confidence=0.8,
                page_num=0,
                text_content="Main Content Area"
            ),
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.85, x2=0.95, y2=0.95),
                label="footer",
                confidence=0.7,
                page_num=0,
                text_content="Footer Area"
            )
        ]
    
    def visualize(self, image: np.ndarray, layout_regions: List[LayoutRegion]) -> np.ndarray:
        """Visualize layout regions on image"""
        try:
            # Create a copy for visualization
            vis_image = image.copy()
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
            
            height, width = vis_image.shape[:2]
            
            # Color map for different region types
            color_map = {
                "header": (255, 0, 0),      # Blue
                "footer": (0, 255, 0),      # Green
                "title": (0, 0, 255),       # Red
                "text": (255, 255, 0),      # Cyan
                "paragraph": (0, 255, 255), # Yellow
                "table": (255, 0, 255),     # Magenta
                "signature": (255, 128, 0), # Orange
                "label": (128, 128, 128)    # Gray
            }
            
            for region in layout_regions:
                # Convert normalized to pixel coordinates
                x1 = int(region.bbox.x1 * width)
                y1 = int(region.bbox.y1 * height)
                x2 = int(region.bbox.x2 * width)
                y2 = int(region.bbox.y2 * height)
                
                # Get color
                color = color_map.get(region.label, (200, 200, 200))
                
                # Draw rectangle
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{region.label} ({region.confidence:.2f})"
                cv2.putText(vis_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add text snippet if available
                if region.text_content and len(region.text_content) > 0:
                    text_snippet = region.text_content[:30] + ("..." if len(region.text_content) > 30 else "")
                    cv2.putText(vis_image, text_snippet, (x1, y2 + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return image