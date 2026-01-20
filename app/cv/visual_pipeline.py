# app/cv/visual_pipeline.py - CORRECTED
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from app.core.models import LayoutRegion, EnhancedVisualElement, BoundingBox
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VisualPipeline:
    """Main pipeline for visual document analysis - FIXED INTERFACE"""
    
    def __init__(self):
        # Initialize layout analyzer
        self.layout_analyzer = None
        try:
            from app.cv.paddle_layout_analyzer import PaddleLayoutAnalyzer
            self.layout_analyzer = PaddleLayoutAnalyzer()
            logger.info("✅ Using PaddleLayoutAnalyzer")
        except ImportError as e:
            logger.warning(f"⚠️ PaddleLayoutAnalyzer not available: {e}")
            try:
                from app.cv.layout_analyzer import LayoutAnalyzer
                self.layout_analyzer = LayoutAnalyzer()
                logger.info("✅ Using LayoutAnalyzer")
            except ImportError as e2:
                logger.error(f"❌ No layout analyzer available: {e2}")
                self.layout_analyzer = None
        
        # Initialize object detector
        self.object_detector = None
        try:
            from app.cv.document_object_detector import DocumentObjectDetector
            self.object_detector = DocumentObjectDetector()
            logger.info("✅ DocumentObjectDetector loaded")
        except ImportError as e:
            logger.warning(f"⚠️ DocumentObjectDetector not available: {e}")
            try:
                from app.cv.object_detector import ObjectDetector
                self.object_detector = ObjectDetector()
                logger.info("✅ ObjectDetector loaded")
            except ImportError as e2:
                logger.error(f"❌ No object detector available: {e2}")
                self.object_detector = None
        
        # Configuration
        self.min_confidence = {
            "layout": 0.3,
            "object": 0.25
        }
        
        logger.info("✅ VisualPipeline initialized")
    
    def process_page(self, image: np.ndarray, page_num: int = 0) -> Tuple[List[LayoutRegion], List[EnhancedVisualElement]]:
        """
        Process a single document page
        
        Returns:
            Tuple of (layout_regions, visual_elements) with NORMALIZED coordinates (0-1)
        """
        try:
            logger.info(f"👁️ Processing page {page_num}")
            
            # Step 1: Layout Analysis
            layout_regions = self._safe_layout_analysis(image)
            
            # Step 2: Object Detection  
            visual_elements = self._safe_object_detection(image)
            
            # Step 3: Filter and post-process
            layout_regions = self._filter_by_confidence(layout_regions, self.min_confidence["layout"])
            visual_elements = self._filter_by_confidence(visual_elements, self.min_confidence["object"])
            
            # Step 4: Clean up elements
            visual_elements = self._filter_small_elements(visual_elements)
            visual_elements = self._filter_overlapping_elements(visual_elements)
            
            # Step 5: Add page number to all elements
            for region in layout_regions:
                region.page_num = page_num
            for element in visual_elements:
                element.page_num = page_num
            
            logger.info(f"✅ Page {page_num}: {len(layout_regions)} layout regions, {len(visual_elements)} visual elements")
            
            return layout_regions, visual_elements
            
        except Exception as e:
            logger.error(f"❌ Visual pipeline failed for page {page_num}: {e}")
            return [], []
    
    def _safe_layout_analysis(self, image: np.ndarray) -> List[LayoutRegion]:
        """Safe layout analysis with error handling"""
        if self.layout_analyzer is None:
            logger.warning("⚠️ No layout analyzer available")
            return []
        
        try:
            # Call the analyze method (no page_num parameter)
            regions = self.layout_analyzer.analyze(image)
            
            # Validate regions have normalized coordinates
            validated = []
            for region in regions:
                if isinstance(region.bbox, BoundingBox) and region.bbox.is_normalized():
                    validated.append(region)
                else:
                    logger.warning(f"Region with invalid bbox: {region.bbox}")
            
            return validated
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return []
    
    def _safe_object_detection(self, image: np.ndarray) -> List[EnhancedVisualElement]:
        """Safe object detection with error handling"""
        if self.object_detector is None:
            logger.warning("⚠️ Object detection not available")
            return []
        
        try:
            # Call the detect method (no page_num parameter)
            elements = self.object_detector.detect(image)
            
            # Validate elements have normalized coordinates
            validated = []
            for element in elements:
                if isinstance(element.bbox, BoundingBox) and element.bbox.is_normalized():
                    validated.append(element)
                else:
                    logger.warning(f"Element with invalid bbox: {element.bbox}")
            
            return validated
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def _filter_by_confidence(self, items: List, min_confidence: float) -> List:
        """Filter items by minimum confidence"""
        return [item for item in items if item.confidence >= min_confidence]
    
    def _filter_small_elements(self, elements: List[EnhancedVisualElement]) -> List[EnhancedVisualElement]:
        """Filter out very small elements"""
        filtered = []
        for element in elements:
            area = (element.bbox.x2 - element.bbox.x1) * (element.bbox.y2 - element.bbox.y1)
            if area > 0.001:  # At least 0.1% of image
                filtered.append(element)
        return filtered
    
    def _filter_overlapping_elements(self, elements: List[EnhancedVisualElement]) -> List[EnhancedVisualElement]:
        """Filter out overlapping elements, keeping highest confidence ones"""
        if len(elements) <= 1:
            return elements
        
        # Sort by confidence (highest first)
        sorted_elements = sorted(elements, key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for i, elem in enumerate(sorted_elements):
            is_overlap = False
            
            # Check against all higher confidence elements
            for j in range(i):
                if self._calculate_iou(elem, sorted_elements[j]) > 0.4:
                    is_overlap = True
                    break
            
            if not is_overlap:
                filtered.append(elem)
        
        return filtered
    
    def _calculate_iou(self, elem1: EnhancedVisualElement, elem2: EnhancedVisualElement) -> float:
        """Calculate Intersection over Union between two elements"""
        b1 = elem1.bbox
        b2 = elem2.bbox
        
        # Calculate intersection
        x_left = max(b1.x1, b2.x1)
        y_top = max(b1.y1, b2.y1)
        x_right = min(b1.x2, b2.x2)
        y_bottom = min(b1.y2, b2.y2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
        area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def visualize_results(self, image: np.ndarray, 
                         layout_regions: List[LayoutRegion],
                         visual_elements: List[EnhancedVisualElement]) -> np.ndarray:
        """Create visualization of all analysis results"""
        try:
            # Start with layout visualization
            if self.layout_analyzer and hasattr(self.layout_analyzer, 'visualize'):
                vis_image = self.layout_analyzer.visualize(image, layout_regions)
            else:
                vis_image = image.copy()
                if len(vis_image.shape) == 2:
                    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
            
            # Add object detection visualization if available
            if self.object_detector and hasattr(self.object_detector, 'visualize'):
                vis_image = self.object_detector.visualize(vis_image, visual_elements)
            
            # Add statistics overlay
            stats = self.get_statistics(layout_regions, visual_elements)
            
            # Add stats text
            height, width = vis_image.shape[:2]
            
            # Background for text
            cv2.rectangle(vis_image, (5, 5), (250, 100), (0, 0, 0), -1)
            cv2.rectangle(vis_image, (5, 5), (250, 100), (255, 255, 255), 1)
            
            # Layout stats
            cv2.putText(vis_image, f"Layout Regions: {stats['layout_regions_count']}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Element stats
            cv2.putText(vis_image, f"Visual Elements: {stats['visual_elements_count']}", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Confidence stats
            cv2.putText(vis_image, f"Layout Confidence: {stats['average_confidence']['layout']:.2f}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(vis_image, f"Element Confidence: {stats['average_confidence']['elements']:.2f}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return image
    
    def get_statistics(self, layout_regions: List[LayoutRegion], 
                      visual_elements: List[EnhancedVisualElement]) -> Dict[str, Any]:
        """Get statistics about the analysis"""
        stats = {
            "layout_regions_count": len(layout_regions),
            "visual_elements_count": len(visual_elements),
            "layout_distribution": {},
            "element_distribution": {},
            "average_confidence": {
                "layout": 0.0,
                "elements": 0.0
            }
        }
        
        # Count layout regions by type
        for region in layout_regions:
            stats["layout_distribution"][region.label] = stats["layout_distribution"].get(region.label, 0) + 1
        
        # Count visual elements by type
        for element in visual_elements:
            stats["element_distribution"][element.element_type] = stats["element_distribution"].get(element.element_type, 0) + 1
        
        # Calculate average confidences
        if layout_regions:
            stats["average_confidence"]["layout"] = sum(r.confidence for r in layout_regions) / len(layout_regions)
        
        if visual_elements:
            stats["average_confidence"]["elements"] = sum(e.confidence for e in visual_elements) / len(visual_elements)
        
        return stats
    
    def save_visualization(self, image: np.ndarray, 
                          layout_regions: List[LayoutRegion],
                          visual_elements: List[EnhancedVisualElement],
                          output_path: str = "visual_analysis_output.jpg") -> bool:
        """Save visualization to file"""
        try:
            vis_image = self.visualize_results(image, layout_regions, visual_elements)
            cv2.imwrite(output_path, vis_image)
            logger.info(f"✅ Visualization saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save visualization: {e}")
            return False