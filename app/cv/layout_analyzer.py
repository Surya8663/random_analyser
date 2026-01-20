# app/cv/layout_analyzer.py - CORRECTED LAYOUTPARSER API
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import layoutparser as lp
from app.core.models import LayoutRegion, BoundingBox
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class LayoutAnalyzer:
    """Real layout analysis using LayoutParser - CORRECTED API"""
    
    def __init__(self):
        self.model = None
        try:
            logger.info("📦 Loading LayoutParser model...")
            
            # CORRECT API for LayoutParser 0.3.4
            self.model = lp.models.Detectron2LayoutModel(
                config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            logger.info("✅ LayoutParser model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load LayoutParser model: {e}")
            logger.info("⚠️ Will use enhanced fallback analysis")
    
    def analyze(self, image: np.ndarray, page_num: int = 0) -> List[LayoutRegion]:
        """
        Analyze document layout and extract regions
        """
        if self.model is None:
            logger.warning("⚠️ LayoutParser model not loaded, using enhanced fallback")
            return self._enhanced_fallback_analysis(image, page_num)
        
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Convert grayscale to RGB
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = image
            
            # Get image dimensions
            height, width = image_rgb.shape[:2]
            
            # Run layout detection
            layout = self.model.detect(image_rgb)
            
            # Convert to LayoutRegion objects
            layout_regions = []
            
            for block in layout:
                # Get bounding box coordinates - CORRECT API
                x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
                
                # Normalize coordinates (0-1 range)
                x1_norm = max(0, min(x1 / width, 1))
                y1_norm = max(0, min(y1 / height, 1))
                x2_norm = max(0, min(x2 / width, 1))
                y2_norm = max(0, min(y2 / height, 1))
                
                # Create LayoutRegion
                region = LayoutRegion(
                    bbox=BoundingBox(
                        x1=float(x1_norm),
                        y1=float(y1_norm),
                        x2=float(x2_norm),
                        y2=float(y2_norm)
                    ),
                    label=str(block.type).lower(),
                    confidence=float(block.score),
                    page_num=page_num,
                    text_content=None
                )
                
                layout_regions.append(region)
            
            logger.info(f"✅ LayoutParser detected {len(layout_regions)} regions on page {page_num}")
            return layout_regions
            
        except Exception as e:
            logger.error(f"❌ Layout analysis failed: {e}")
            return self._enhanced_fallback_analysis(image, page_num)
    
    def _enhanced_fallback_analysis(self, image: np.ndarray, page_num: int) -> List[LayoutRegion]:
        """Enhanced fallback layout analysis with edge detection"""
        layout_regions = []
        
        try:
            height, width = image.shape[:2]
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Edge detection for text blocks
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours[:10]:  # Limit to 10 largest contours
                area = cv2.contourArea(contour)
                if area > 5000:  # Only significant areas
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Normalize coordinates
                    x1_norm = x / width
                    y1_norm = y / height
                    x2_norm = (x + w) / width
                    y2_norm = (y + h) / height
                    
                    # Determine region type based on position and size
                    aspect_ratio = w / h if h > 0 else 1
                    
                    if y1_norm < 0.2:  # Top of page
                        label = "header"
                        confidence = 0.7
                    elif y2_norm > 0.8:  # Bottom of page
                        label = "footer"
                        confidence = 0.7
                    elif aspect_ratio > 2.0:  # Wide region
                        label = "table"
                        confidence = 0.6
                    else:
                        label = "text"
                        confidence = 0.8
                    
                    region = LayoutRegion(
                        bbox=BoundingBox(
                            x1=float(x1_norm),
                            y1=float(y1_norm),
                            x2=float(x2_norm),
                            y2=float(y2_norm)
                        ),
                        label=label,
                        confidence=confidence,
                        page_num=page_num,
                        text_content=None
                    )
                    
                    layout_regions.append(region)
            
            # If no significant contours found, add default regions
            if not layout_regions:
                layout_regions = self._create_default_regions(height, width, page_num)
            
            logger.info(f"✅ Enhanced fallback detected {len(layout_regions)} regions")
            return layout_regions
            
        except Exception as e:
            logger.warning(f"Enhanced fallback analysis failed: {e}")
            return self._create_default_regions(height, width, page_num)
    
    def _create_default_regions(self, height: int, width: int, page_num: int) -> List[LayoutRegion]:
        """Create default layout regions"""
        return [
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.05, x2=0.95, y2=0.15),
                label="header",
                confidence=0.7,
                page_num=page_num,
                text_content="Header region"
            ),
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.2, x2=0.95, y2=0.8),
                label="text",
                confidence=0.8,
                page_num=page_num,
                text_content="Main text content"
            ),
            LayoutRegion(
                bbox=BoundingBox(x1=0.05, y1=0.85, x2=0.95, y2=0.95),
                label="footer",
                confidence=0.7,
                page_num=page_num,
                text_content="Footer region"
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
                "text": (0, 255, 0),      # Green
                "title": (255, 0, 0),     # Blue
                "list": (0, 0, 255),      # Red
                "table": (255, 255, 0),   # Cyan
                "figure": (255, 0, 255),  # Magenta
                "header": (0, 255, 255),  # Yellow
                "footer": (128, 0, 128)   # Purple
            }
            
            for region in layout_regions:
                # Convert normalized coordinates to pixel coordinates
                x1 = int(region.bbox.x1 * width)
                y1 = int(region.bbox.y1 * height)
                x2 = int(region.bbox.x2 * width)
                y2 = int(region.bbox.y2 * height)
                
                # Get color for region type
                color = color_map.get(region.label, (128, 128, 128))
                
                # Draw rectangle
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{region.label} ({region.confidence:.2f})"
                cv2.putText(vis_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return image