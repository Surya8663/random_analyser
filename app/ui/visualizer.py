"""
Visualization system for multi-modal document analysis.
Generates interactive HTML reports with agent overlays, provenance trails, and risk visualizations.
"""
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import cv2

from app.core.models import MultiModalDocument, EnhancedVisualElement, LayoutRegion, Contradiction
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentVisualizer:
    """Generate interactive visualizations for document analysis"""
    
    # Color scheme for agents
    AGENT_COLORS = {
        "vision_agent": "rgba(59, 130, 246, 0.7)",  # Blue
        "text_agent": "rgba(34, 197, 94, 0.7)",     # Green
        "fusion_agent": "rgba(168, 85, 247, 0.7)",  # Purple
        "reasoning_agent": "rgba(249, 115, 22, 0.7)", # Orange
        "explainability_agent": "rgba(20, 184, 166, 0.7)" # Teal
    }
    
    # Element type icons
    ELEMENT_ICONS = {
        "table": "📊",
        "signature": "✍️",
        "logo": "🏢",
        "text_block": "📝",
        "header": "📑",
        "footer": "📄",
        "figure": "🖼️"
    }
    
    def __init__(self, output_dir: str = "visualization_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_html_report(self, document: MultiModalDocument) -> Path:
        """Generate complete HTML report for a document"""
        try:
            logger.info(f"🎨 Generating HTML report for {document.document_id}")
            
            # Create report data structure
            report_data = self._prepare_report_data(document)
            
            # Generate HTML content
            html_content = self._generate_html_content(report_data)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{document.document_id}_{timestamp}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"📄 HTML report saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise
    
    def _prepare_report_data(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Prepare all data needed for the report"""
        report_data = {
            "document_info": self._extract_document_info(document),
            "agent_results": self._extract_agent_results(document),
            "visual_elements": self._prepare_visual_elements(document),
            "extracted_fields": self._prepare_extracted_fields(document),
            "contradictions": self._prepare_contradictions(document),
            "processing_timeline": self._prepare_processing_timeline(document),
            "images": self._prepare_document_images(document)
        }
        
        return report_data
    
    def _extract_document_info(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Extract basic document information"""
        return {
            "document_id": document.document_id,
            "document_type": document.document_type.value if document.document_type else "unknown",
            "file_path": document.file_path,
            "file_type": document.file_type,
            "processing_time": document.get_processing_time(),
            "risk_score": document.risk_score if hasattr(document, 'risk_score') else 0.0,
            "risk_level": self._get_risk_level(document.risk_score if hasattr(document, 'risk_score') else 0.0),
            "total_pages": len(document.images) if hasattr(document, 'images') else 0,
            "total_errors": len(document.errors) if hasattr(document, 'errors') else 0,
            "total_contradictions": len(document.contradictions) if hasattr(document, 'contradictions') else 0,
            "total_fields": len(document.extracted_fields) if hasattr(document, 'extracted_fields') else 0
        }
    
    def _extract_agent_results(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Extract results from each agent"""
        agent_results = {}
        
        # Vision Agent
        if hasattr(document, 'visual_elements') and document.visual_elements:
            agent_results["vision_agent"] = {
                "elements_analyzed": len(document.visual_elements),
                "element_types": self._count_element_types(document.visual_elements),
                "average_confidence": np.mean([e.confidence for e in document.visual_elements]) 
                if document.visual_elements else 0.0,
                "status": "completed"
            }
        
        # Text Agent
        if hasattr(document, 'ocr_results') and document.ocr_results:
            total_words = sum(len(ocr.words) for ocr in document.ocr_results.values())
            avg_confidence = np.mean([ocr.average_confidence for ocr in document.ocr_results.values()])
            
            agent_results["text_agent"] = {
                "pages_processed": len(document.ocr_results),
                "total_words": total_words,
                "average_confidence": avg_confidence,
                "entities_extracted": sum(len(entities) for entities in document.extracted_entities.values()) 
                if hasattr(document, 'extracted_entities') else 0,
                "status": "completed"
            }
        
        # Fusion Agent
        if hasattr(document, 'aligned_data') and document.aligned_data:
            alignments = document.aligned_data.get("text_visual_alignment", [])
            agent_results["fusion_agent"] = {
                "alignments_created": len(alignments),
                "consistency_score": document.aligned_data.get("consistency_metrics", {}).get("overall_consistency", 0),
                "status": "completed"
            }
        
        # Reasoning Agent
        if hasattr(document, 'contradictions'):
            agent_results["reasoning_agent"] = {
                "contradictions_found": len(document.contradictions),
                "risk_assessment": document.risk_score if hasattr(document, 'risk_score') else 0.0,
                "recommendations": len(document.review_recommendations) 
                if hasattr(document, 'review_recommendations') else 0,
                "status": "completed"
            }
        
        # Explainability Agent
        if hasattr(document, 'processing_metadata') and document.processing_metadata:
            explainability = document.processing_metadata.get("explainability_report", {})
            agent_results["explainability_agent"] = {
                "fields_explained": len(explainability.get("field_explanations", {})),
                "decision_points": len(explainability.get("decision_points", [])),
                "status": "completed"
            }
        
        return agent_results
    
    def _prepare_visual_elements(self, document: MultiModalDocument) -> List[Dict[str, Any]]:
        """Prepare visual elements for visualization"""
        elements = []
        
        if not hasattr(document, 'visual_elements') or not document.visual_elements:
            return elements
        
        for idx, element in enumerate(document.visual_elements):
            elem_data = {
                "id": f"element_{idx}",
                "type": element.element_type,
                "icon": self.ELEMENT_ICONS.get(element.element_type, "📄"),
                "bbox": element.bbox.to_list(),
                "page": element.page_num,
                "confidence": element.confidence,
                "color": self.AGENT_COLORS["vision_agent"],
                "agent": "vision_agent",
                "metadata": element.metadata or {}
            }
            
            # Add text content if available
            if hasattr(element, 'text_content') and element.text_content:
                elem_data["text_content"] = element.text_content[:100] + "..." if len(element.text_content) > 100 else element.text_content
            
            elements.append(elem_data)
        
        return elements
    
    def _prepare_extracted_fields(self, document: MultiModalDocument) -> List[Dict[str, Any]]:
        """Prepare extracted fields for display"""
        fields = []
        
        if not hasattr(document, 'extracted_fields') or not document.extracted_fields:
            return fields
        
        for field_name, field in document.extracted_fields.items():
            # Determine which agent extracted this field
            agent = "fusion_agent"  # Default
            if hasattr(field, 'provenance') and field.provenance:
                agent = field.provenance[0].agent_name if field.provenance else "fusion_agent"
            
            field_data = {
                "name": field_name,
                "value": str(field.value) if field.value is not None else "",
                "type": field.field_type,
                "confidence": field.confidence,
                "agent": agent,
                "color": self.AGENT_COLORS.get(agent, self.AGENT_COLORS["fusion_agent"]),
                "sources": field.modality_sources if hasattr(field, 'modality_sources') else [],
                "provenance_count": len(field.provenance) if hasattr(field, 'provenance') else 0
            }
            
            fields.append(field_data)
        
        return fields
    
    def _prepare_contradictions(self, document: MultiModalDocument) -> List[Dict[str, Any]]:
        """Prepare contradictions for display"""
        contradictions = []
        
        if not hasattr(document, 'contradictions') or not document.contradictions:
            return contradictions
        
        for idx, contradiction in enumerate(document.contradictions):
            contr_data = {
                "id": f"contradiction_{idx}",
                "type": contradiction.contradiction_type.value,
                "severity": contradiction.severity.value,
                "explanation": contradiction.explanation,
                "confidence": contradiction.confidence,
                "fields_involved": [contradiction.field_a, contradiction.field_b] 
                if contradiction.field_a and contradiction.field_b else [],
                "recommendation": contradiction.recommendation,
                "severity_color": self._get_severity_color(contradiction.severity.value)
            }
            
            contradictions.append(contr_data)
        
        return contradictions
    
    def _prepare_processing_timeline(self, document: MultiModalDocument) -> List[Dict[str, Any]]:
        """Prepare processing timeline data"""
        timeline = []
        
        # Base timeline from document processing
        if hasattr(document, 'processing_start'):
            timeline.append({
                "event": "Processing started",
                "timestamp": document.processing_start.isoformat(),
                "agent": "orchestrator",
                "duration": None
            })
        
        if hasattr(document, 'processing_end'):
            timeline.append({
                "event": "Processing completed",
                "timestamp": document.processing_end.isoformat(),
                "agent": "orchestrator",
                "duration": document.get_processing_time()
            })
        
        # Add agent events from metadata if available
        if hasattr(document, 'processing_metadata'):
            metadata = document.processing_metadata
            
            for agent_name in ["vision_analysis", "text_quality", "reasoning"]:
                if agent_name in metadata:
                    timeline.append({
                        "event": f"{agent_name.replace('_', ' ').title()} completed",
                        "timestamp": datetime.now().isoformat(),  # Would be real timestamps in production
                        "agent": agent_name,
                        "duration": None
                    })
        
        return timeline
    
    def _prepare_document_images(self, document: MultiModalDocument) -> List[Dict[str, Any]]:
        """Prepare document images for display (base64 encoded)"""
        images = []
        
        if not hasattr(document, 'images') or not document.images:
            return images
        
        for idx, image in enumerate(document.images[:5]):  # Limit to first 5 pages
            if image is not None:
                try:
                    # Convert image to base64
                    if len(image.shape) == 2:  # Grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    
                    # Resize for display
                    height, width = image.shape[:2]
                    max_size = 800
                    if width > max_size or height > max_size:
                        scale = max_size / max(width, height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        image = cv2.resize(image, (new_width, new_height))
                    
                    # Encode as JPEG
                    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    images.append({
                        "page": idx + 1,
                        "data_url": f"data:image/jpeg;base64,{img_base64}",
                        "width": image.shape[1],
                        "height": image.shape[0]
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to encode image {idx}: {e}")
                    # Add placeholder
                    images.append({
                        "page": idx + 1,
                        "data_url": "",
                        "width": 600,
                        "height": 800,
                        "error": True
                    })
        
        return images
    
    def _count_element_types(self, visual_elements: List[EnhancedVisualElement]) -> Dict[str, int]:
        """Count element types"""
        counts = {}
        for element in visual_elements:
            elem_type = element.element_type
            counts[elem_type] = counts.get(elem_type, 0) + 1
        return counts
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to level"""
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level"""
        colors = {
            "low": "#10b981",      # Green
            "medium": "#f59e0b",   # Yellow
            "high": "#ef4444",     # Red
            "critical": "#7c3aed"  # Purple
        }
        return colors.get(severity.lower(), "#6b7280")
    
    def _generate_html_content(self, report_data: Dict[str, Any]) -> str:
        """Generate complete HTML content"""
        # Read HTML template
        template_path = Path(__file__).parent / "templates" / "report_template.html"
        
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            # Fallback: generate minimal template
            template = self._generate_minimal_template()
        
        # Convert data to JSON for JavaScript
        json_data = json.dumps(report_data, default=str)
        
        # Replace placeholders in template
        html = template.replace("{{REPORT_DATA}}", json_data)
        
        return html
    
    def _generate_minimal_template(self) -> str:
        """Generate minimal HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Analysis Report</title>
    <style>
        :root {
            --vision-color: #3b82f6;
            --text-color: #22c55e;
            --fusion-color: #a855f7;
            --reasoning-color: #f97316;
            --explainability-color: #14b8a6;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8fafc;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .document-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .risk-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }
        
        .risk-high { background: #ef4444; }
        .risk-medium { background: #f59e0b; }
        .risk-low { background: #10b981; }
        
        .agent-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .agent-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid;
        }
        
        .agent-vision { border-left-color: var(--vision-color); }
        .agent-text { border-left-color: var(--text-color); }
        .agent-fusion { border-left-color: var(--fusion-color); }
        .agent-reasoning { border-left-color: var(--reasoning-color); }
        .agent-explainability { border-left-color: var(--explainability-color); }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
        }
        
        .image-viewer {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .image-container {
            position: relative;
            margin-bottom: 20px;
        }
        
        .document-image {
            max-width: 100%;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
        }
        
        .element-overlay {
            position: absolute;
            border: 2px solid;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .element-overlay:hover {
            transform: scale(1.02);
            box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .field-list, .contradiction-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .field-item, .contradiction-item {
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 6px;
            border-left: 4px solid;
            background: #f9fafb;
        }
        
        .timeline {
            position: relative;
            padding-left: 20px;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #e5e7eb;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 20px;
            padding-left: 20px;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -6px;
            top: 0;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #3b82f6;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: white;
            border-radius: 10px;
            padding: 30px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .close-modal {
            float: right;
            font-size: 24px;
            cursor: pointer;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .agent-cards {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="document-header">
                <div>
                    <h1>📄 Document Analysis Report</h1>
                    <p class="document-id">ID: <span id="document-id"></span></p>
                </div>
                <div id="risk-badge" class="risk-badge"></div>
            </div>
            
            <div class="document-stats">
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div><strong>Type:</strong> <span id="document-type"></span></div>
                    <div><strong>Processing Time:</strong> <span id="processing-time"></span>s</div>
                    <div><strong>Pages:</strong> <span id="total-pages"></span></div>
                    <div><strong>Fields Extracted:</strong> <span id="total-fields"></span></div>
                    <div><strong>Contradictions:</strong> <span id="total-contradictions"></span></div>
                </div>
            </div>
        </header>
        
        <div class="agent-cards" id="agent-cards">
            <!-- Agent cards will be populated by JavaScript -->
        </div>
        
        <div class="main-content">
            <div class="image-viewer">
                <h2>📷 Document Viewer</h2>
                <div class="image-controls">
                    <button onclick="prevPage()">← Previous</button>
                    <span id="page-indicator">Page 1 of 1</span>
                    <button onclick="nextPage()">Next →</button>
                </div>
                <div class="image-container" id="image-container">
                    <!-- Image and overlays will be added here -->
                </div>
                <div class="overlay-controls">
                    <label><input type="checkbox" id="show-vision" checked> Vision Agent</label>
                    <label><input type="checkbox" id="show-text" checked> Text Agent</label>
                    <label><input type="checkbox" id="show-fusion" checked> Fusion Agent</label>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="panel">
                    <h3>📋 Extracted Fields</h3>
                    <div class="field-list" id="field-list">
                        <!-- Fields will be populated by JavaScript -->
                    </div>
                </div>
                
                <div class="panel">
                    <h3>⚠️ Contradictions</h3>
                    <div class="contradiction-list" id="contradiction-list">
                        <!-- Contradictions will be populated by JavaScript -->
                    </div>
                </div>
                
                <div class="panel">
                    <h3>⏱️ Processing Timeline</h3>
                    <div class="timeline" id="timeline">
                        <!-- Timeline will be populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="modal" id="element-modal">
        <div class="modal-content">
            <span class="close-modal" onclick="closeModal()">×</span>
            <h2 id="modal-title">Element Details</h2>
            <div id="modal-content">
                <!-- Modal content will be populated by JavaScript -->
            </div>
        </div>
    </div>
    
    <script>
        // Report data will be injected here
        const reportData = {{REPORT_DATA}};
        
        // Your JavaScript code here...
    </script>
</body>
</html>
"""