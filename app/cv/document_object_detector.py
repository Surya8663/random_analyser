# app/cv/document_object_detector.py - REAL DETECTION VERSION
import cv2
import numpy as np
from typing import List
from app.core.models import EnhancedVisualElement, BoundingBox
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentObjectDetector:
    """Document element detector using REAL image processing"""
    
    def __init__(self):
        logger.info("✅ DocumentObjectDetector initialized (REAL detection)")
    
    def detect(self, image: np.ndarray) -> List[EnhancedVisualElement]:
        """REAL detection of document elements - NO MOCK DATA"""
        try:
            logger.info(f"🔍 Starting REAL document element detection")
            height, width = image.shape[:2]
            
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            elements = []
            
            # 1. Detect tables using line detection
            table_elements = self._detect_tables_real(gray, height, width)
            elements.extend(table_elements)
            
            # 2. Detect signatures using texture analysis
            signature_elements = self._detect_signatures_real(gray, height, width)
            elements.extend(signature_elements)
            
            # 3. Detect logos using shape detection
            logo_elements = self._detect_logos_real(gray, height, width)
            elements.extend(logo_elements)
            
            # 4. Detect text blocks
            text_elements = self._detect_text_blocks_real(gray, height, width)
            elements.extend(text_elements)
            
            # 5. Post-process: remove duplicates and small elements
            elements = self._post_process_elements(elements)
            
            logger.info(f"✅ REAL detection found {len(elements)} elements")
            
            # Return REAL results, even if empty
            return elements
            
        except Exception as e:
            logger.error(f"❌ REAL document detection failed: {e}")
            # Return empty list, NOT mock data
            return []
    
    def _detect_tables_real(self, gray: np.ndarray, height: int, width: int) -> List[EnhancedVisualElement]:
        """REAL table detection using line and grid detection"""
        elements = []
        
        try:
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, 
                                   minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    # Classify as horizontal or vertical
                    if abs(angle) < 30:  # Horizontal
                        horizontal_lines.append((min(y1, y2), max(y1, y2), 
                                               min(x1, x2), max(x1, x2)))
                    elif abs(angle) > 60:  # Vertical
                        vertical_lines.append((min(x1, x2), max(x1, x2),
                                             min(y1, y2), max(y1, y2)))
                
                # Look for grid patterns (tables)
                if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                    # Group nearby lines
                    horizontal_groups = self._group_lines(horizontal_lines, axis='y', threshold=10)
                    vertical_groups = self._group_lines(vertical_lines, axis='x', threshold=10)
                    
                    # Create table elements for detected grids
                    for h_group in horizontal_groups:
                        for v_group in vertical_groups:
                            if len(h_group) >= 2 and len(v_group) >= 2:
                                # Calculate table bounds
                                min_y = min([h[0] for h in h_group])
                                max_y = max([h[1] for h in h_group])
                                min_x = min([v[0] for v in v_group])
                                max_x = max([v[1] for v in v_group])
                                
                                # Check if this looks like a table (grid structure)
                                if (max_y - min_y) > 50 and (max_x - min_x) > 100:
                                    # Normalize coordinates
                                    x1_norm = min_x / width
                                    y1_norm = min_y / height
                                    x2_norm = max_x / width
                                    y2_norm = max_y / height
                                    
                                    # Skip if too small
                                    area = (x2_norm - x1_norm) * (y2_norm - y1_norm)
                                    if area < 0.01:  # Less than 1% of image
                                        continue
                                    
                                    element = EnhancedVisualElement(
                                        element_type="table",
                                        bbox=BoundingBox(x1=x1_norm, y1=y1_norm, 
                                                        x2=x2_norm, y2=y2_norm),
                                        confidence=0.7 + (min(len(h_group), 5) * 0.05),  # More lines = higher confidence
                                        page_num=0,
                                        metadata={
                                            "detection_method": "grid_detection",
                                            "horizontal_lines": len(h_group),
                                            "vertical_lines": len(v_group),
                                            "is_grid": True
                                        }
                                    )
                                    elements.append(element)
            
        except Exception as e:
            logger.warning(f"Table detection error: {e}")
        
        return elements
    
    def _detect_signatures_real(self, gray: np.ndarray, height: int, width: int) -> List[EnhancedVisualElement]:
        """REAL signature detection using texture and pattern analysis"""
        elements = []
        
        try:
            # Look in common signature areas (bottom-right quadrant)
            signature_roi = gray[int(height * 0.7):height, int(width * 0.6):width]
            
            if signature_roi.size > 0:
                # Analyze texture for signature-like patterns
                edges = cv2.Canny(signature_roi, 50, 150)
                
                # Calculate texture features
                laplacian_var = cv2.Laplacian(signature_roi, cv2.CV_64F).var()
                edge_density = np.sum(edges > 0) / edges.size
                
                # Signature indicators: high edge density, moderate variation
                if edge_density > 0.05 and 100 < laplacian_var < 1000:
                    # Find contours in the signature area
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if 500 < area < 5000:  # Signature-sized contours
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Adjust coordinates to full image
                            x_full = x + int(width * 0.6)
                            y_full = y + int(height * 0.7)
                            
                            # Normalize
                            x1_norm = x_full / width
                            y1_norm = y_full / height
                            x2_norm = (x_full + w) / width
                            y2_norm = (y_full + h) / height
                            
                            element = EnhancedVisualElement(
                                element_type="signature",
                                bbox=BoundingBox(x1=x1_norm, y1=y1_norm, 
                                                x2=x2_norm, y2=y2_norm),
                                confidence=0.6 + min(edge_density * 2, 0.3),
                                page_num=0,
                                metadata={
                                    "detection_method": "texture_analysis",
                                    "edge_density": float(edge_density),
                                    "texture_variance": float(laplacian_var),
                                    "contour_area": float(area)
                                }
                            )
                            elements.append(element)
            
        except Exception as e:
            logger.warning(f"Signature detection error: {e}")
        
        return elements
    
    def _detect_logos_real(self, gray: np.ndarray, height: int, width: int) -> List[EnhancedVisualElement]:
        """REAL logo detection using shape and corner detection"""
        elements = []
        
        try:
            # Look in common logo areas (top corners)
            logo_areas = [
                (0, 0, int(width * 0.3), int(height * 0.2)),  # Top-left
                (int(width * 0.7), 0, width, int(height * 0.2))  # Top-right
            ]
            
            for x1, y1, x2, y2 in logo_areas:
                roi = gray[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Detect corners (logos often have distinct corners)
                    corners = cv2.goodFeaturesToTrack(roi, maxCorners=20, 
                                                     qualityLevel=0.01, minDistance=10)
                    
                    if corners is not None and len(corners) > 4:
                        # Look for geometric shapes (circles, rectangles)
                        edges = cv2.Canny(roi, 50, 150)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                                      cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if 1000 < area < 10000:  # Logo-sized
                                x, y, w, h = cv2.boundingRect(contour)
                                
                                # Check shape properties
                                perimeter = cv2.arcLength(contour, True)
                                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                                
                                # Adjust coordinates
                                x_full = x + x1
                                y_full = y + y1
                                
                                # Normalize
                                x1_norm = x_full / width
                                y1_norm = y_full / height
                                x2_norm = (x_full + w) / width
                                y2_norm = (y_full + h) / height
                                
                                # Determine logo type
                                logo_type = "stamp" if circularity > 0.7 else "logo"
                                
                                element = EnhancedVisualElement(
                                    element_type=logo_type,
                                    bbox=BoundingBox(x1=x1_norm, y1=y1_norm, 
                                                    x2=x2_norm, y2=y2_norm),
                                    confidence=0.5 + min(circularity, 0.4),
                                    page_num=0,
                                    metadata={
                                        "detection_method": "shape_detection",
                                        "corners_detected": len(corners),
                                        "circularity": float(circularity),
                                        "area": float(area)
                                    }
                                )
                                elements.append(element)
                                break  # Found one logo in this area
            
        except Exception as e:
            logger.warning(f"Logo detection error: {e}")
        
        return elements
    
    def _detect_text_blocks_real(self, gray: np.ndarray, height: int, width: int) -> List[EnhancedVisualElement]:
        """REAL text block detection using connected components"""
        elements = []
        
        try:
            # Apply Otsu's threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to connect text
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=2)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, 
                                                                                   connectivity=8)
            
            for i in range(1, num_labels):  # Skip background (0)
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Filter by size and aspect ratio (text-like)
                if 100 < area < 10000 and 0.1 < w/h < 10:
                    # Normalize
                    x1_norm = x / width
                    y1_norm = y / height
                    x2_norm = (x + w) / width
                    y2_norm = (y + h) / height
                    
                    element = EnhancedVisualElement(
                        element_type="text_block",
                        bbox=BoundingBox(x1=x1_norm, y1=y1_norm, 
                                        x2=x2_norm, y2=y2_norm),
                        confidence=0.8,
                        page_num=0,
                        metadata={
                            "detection_method": "connected_components",
                            "pixel_area": int(area),
                            "aspect_ratio": float(w/h)
                        }
                    )
                    elements.append(element)
            
        except Exception as e:
            logger.warning(f"Text block detection error: {e}")
        
        return elements
    
    def _group_lines(self, lines, axis='y', threshold=10):
        """Group lines that are close together"""
        if not lines:
            return []
        
        # Sort lines by their position
        if axis == 'y':
            lines.sort(key=lambda x: x[0])  # Sort by y position
            groups = []
            current_group = [lines[0]]
            
            for line in lines[1:]:
                # Check if close to current group
                last_line = current_group[-1]
                if abs(line[0] - last_line[0]) < threshold:
                    current_group.append(line)
                else:
                    groups.append(current_group)
                    current_group = [line]
            
            if current_group:
                groups.append(current_group)
        else:  # axis == 'x'
            lines.sort(key=lambda x: x[0])  # Sort by x position
            groups = []
            current_group = [lines[0]]
            
            for line in lines[1:]:
                last_line = current_group[-1]
                if abs(line[0] - last_line[0]) < threshold:
                    current_group.append(line)
                else:
                    groups.append(current_group)
                    current_group = [line]
            
            if current_group:
                groups.append(current_group)
        
        return groups
    
    def _post_process_elements(self, elements: List[EnhancedVisualElement]) -> List[EnhancedVisualElement]:
        """Remove duplicates and small elements"""
        if not elements:
            return []
        
        # Sort by confidence (highest first)
        elements.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for elem in elements:
            # Check if overlaps with any higher confidence element
            overlap = False
            for kept in filtered:
                iou = self._calculate_iou(elem, kept)
                if iou > 0.3:  # 30% overlap threshold
                    overlap = True
                    break
            
            if not overlap:
                # Also check size
                area = (elem.bbox.x2 - elem.bbox.x1) * (elem.bbox.y2 - elem.bbox.y1)
                if area > 0.001:  # At least 0.1% of image
                    filtered.append(elem)
        
        return filtered
    
    def _calculate_iou(self, elem1: EnhancedVisualElement, elem2: EnhancedVisualElement) -> float:
        """Calculate Intersection over Union"""
        b1 = elem1.bbox
        b2 = elem2.bbox
        
        # Intersection
        x_left = max(b1.x1, b2.x1)
        y_top = max(b1.y1, b2.y1)
        x_right = min(b1.x2, b2.x2)
        y_bottom = min(b1.y2, b2.y2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Union
        area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
        area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize(self, image: np.ndarray, visual_elements: List[EnhancedVisualElement]) -> np.ndarray:
        """Visualize detected elements"""
        try:
            vis_image = image.copy()
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
            
            height, width = vis_image.shape[:2]
            
            color_map = {
                "table": (0, 255, 255),    # Yellow
                "signature": (0, 255, 0),  # Green
                "logo": (255, 0, 0),       # Blue
                "stamp": (255, 128, 0),    # Orange
                "text_block": (128, 0, 128), # Purple
            }
            
            for element in visual_elements:
                # Convert normalized to pixel coordinates
                x1 = int(element.bbox.x1 * width)
                y1 = int(element.bbox.y1 * height)
                x2 = int(element.bbox.x2 * width)
                y2 = int(element.bbox.y2 * height)
                
                # Get color
                color = color_map.get(element.element_type, (200, 200, 200))
                
                # Draw rectangle
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{element.element_type} ({element.confidence:.2f})"
                cv2.putText(vis_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add detection method in smaller text
                method = element.metadata.get('detection_method', 'unknown')
                cv2.putText(vis_image, method, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return image