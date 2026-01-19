from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
from datetime import datetime
import json

from app.core.config import settings
from app.agents.orchestrator import AgentOrchestrator
from app.rag.retriever import MultiModalRetriever
from app.utils.logger import setup_logger
from app.utils.error_handler import handle_api_error
from app.core.models import ProcessingState, UIResultSummary, AgentResult, UIProcessingState

logger = setup_logger(__name__)

router = APIRouter()

# Initialize components
orchestrator = AgentOrchestrator()
retriever = MultiModalRetriever()

# Store processing status (in production, use Redis or database)
processing_status = {}
ui_states = {}  # Store UI-specific states

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing - UI OPTIMIZED
    
    Supports PDF and image files
    """
    try:
        logger.info(f"Upload request received for file: {file.filename}")
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create upload directory
        upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, f"original{file_ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {file_path}")
        
        # Initialize UI state
        ui_states[document_id] = UIProcessingState(
            document_id=document_id,
            current_step="upload",
            overall_progress=0.1,
            agents={
                "vision": AgentResult(agent_name="vision", status="pending"),
                "text": AgentResult(agent_name="text", status="pending"),
                "fusion": AgentResult(agent_name="fusion", status="pending"),
                "validation": AgentResult(agent_name="validation", status="pending")
            },
            user_message="Document uploaded successfully",
            next_action="Configure processing options",
            can_proceed=True
        )
        
        # Update processing status
        processing_status[document_id] = {
            "status": "uploaded",
            "timestamp": datetime.now().isoformat(),
            "ui_state": "ready_for_processing"
        }
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            file_path
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document uploaded successfully",
            "next_step": "process",
            "ui_state": {
                "current_step": "upload",
                "progress": 0.1,
                "message": "Ready to configure processing",
                "can_proceed": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def process_document(
    document_id: str,
    reprocess: bool = False
):
    """
    Trigger document processing - UI OPTIMIZED
    
    Can be used to reprocess an already uploaded document
    """
    try:
        logger.info(f"Process request for document: {document_id}")
        
        # Check if document exists
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        if not os.path.exists(doc_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Update UI state
        if document_id in ui_states:
            ui_states[document_id].current_step = "processing"
            ui_states[document_id].overall_progress = 0.3
            ui_states[document_id].user_message = "Processing started"
            ui_states[document_id].next_action = "Wait for processing to complete"
            ui_states[document_id].can_proceed = False
        
        # Find original file
        original_files = [
            f for f in os.listdir(doc_dir) 
            if f.startswith("original")
        ]
        
        if not original_files:
            raise HTTPException(
                status_code=404,
                detail=f"Original file not found for document {document_id}"
            )
        
        file_path = os.path.join(doc_dir, original_files[0])
        
        # Start processing
        result = await orchestrator.process_document(file_path)
        
        # Save result
        result_file = os.path.join(doc_dir, "processing_result.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Update UI state
        if document_id in ui_states:
            ui_states[document_id].current_step = "results"
            ui_states[document_id].overall_progress = 1.0
            ui_states[document_id].user_message = "Processing completed successfully"
            ui_states[document_id].next_action = "View results"
            ui_states[document_id].can_proceed = True
            
            # Update agent status
            for agent_name in ui_states[document_id].agents:
                ui_states[document_id].agents[agent_name].status = "completed"
                ui_states[document_id].agents[agent_name].confidence = result.get(
                    "confidence_scores", {}
                ).get(agent_name, 0.8)
        
        # Update processing status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False),
            "ui_state": "ready_for_viewing"
        }
        
        return {
            "success": True,
            "document_id": document_id,
            "processing_complete": True,
            "result_available": True,
            "ui_state": {
                "current_step": "results",
                "progress": 1.0,
                "message": "Document processed successfully",
                "can_proceed": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        
        # Update UI state with error
        if document_id in ui_states:
            ui_states[document_id].current_step = "error"
            ui_states[document_id].user_message = f"Processing failed: {str(e)}"
            ui_states[document_id].next_action = "Try again or upload a new document"
            ui_states[document_id].can_proceed = True
        
        # Update status with error
        processing_status[document_id] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "ui_state": "error"
        }
        
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_document(
    query_request: dict
):
    """
    Query processed documents - UI OPTIMIZED
    
    Example request:
    {
        "document_id": "doc_123",
        "question": "What does the chart imply about revenue?"
    }
    """
    try:
        document_id = query_request.get("document_id")
        question = query_request.get("question")
        
        if not document_id or not question:
            raise HTTPException(
                status_code=400,
                detail="document_id and question are required"
            )
        
        logger.info(f"Query request: document={document_id}, question={question[:50]}...")
        
        # Load processing results
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Processing results not found for document {document_id}"
            )
        
        with open(result_file, "r") as f:
            processing_results = json.load(f)
        
        # Extract relevant information for answering
        extracted_data = processing_results.get("extracted_data", {})
        
        # Generate answer with UI-friendly format
        answer_data = generate_ui_answer(question, extracted_data, processing_results)
        
        return {
            "success": True,
            "document_id": document_id,
            "question": question,
            "answer": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "sources": answer_data["sources"],
            "supporting_evidence": answer_data["evidence"],
            "ui_display": {
                "type": answer_data["type"],
                "has_visual": answer_data["has_visual"],
                "recommended_action": answer_data["recommended_action"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{document_id}")
async def get_results(document_id: str):
    """
    Get processing results for a document - UI OPTIMIZED VERSION
    """
    try:
        logger.info(f"Results request for document: {document_id}")
        
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Results not found for document {document_id}"
            )
        
        with open(result_file, "r") as f:
            results = json.load(f)
        
        # Add status information
        status = processing_status.get(document_id, {"status": "unknown"})
        
        # UI-optimized response structure
        ui_optimized_results = {
            "success": True,
            "document_id": document_id,
            "status": status["status"],
            "document_type": results.get("document_type", "unknown"),
            "processing_time": results.get("processing_metadata", {}).get("duration_seconds"),
            "overall_confidence": results.get("confidence_scores", {}).get("overall", 0.0),
            
            # Agent outputs in UI-friendly format
            "agent_outputs": format_agent_outputs_for_ui(results),
            
            # Presentation-ready data
            "visual_elements": format_visual_elements_for_ui(results.get("visual_elements", [])),
            "extracted_fields": format_fields_for_ui(results.get("extracted_data", {}).get("fields", {})),
            "contradictions": format_contradictions_for_ui(results.get("contradictions", [])),
            "validation_issues": results.get("compliance_issues", []),
            
            # Summary for quick overview
            "summary": {
                "total_pages": results.get("processing_metadata", {}).get("page_count", 1),
                "total_elements": len(results.get("visual_elements", [])),
                "total_fields": len(results.get("extracted_data", {}).get("fields", {})),
                "risk_level": calculate_risk_level(results.get("contradictions", [])),
                "recommendations": results.get("review_recommendations", [])
            },
            
            # Raw data for advanced users (collapsed by default)
            "raw_data": {
                "available": True,
                "size": os.path.getsize(result_file)
            }
        }
        
        return ui_optimized_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{document_id}")
async def get_status(document_id: str):
    """
    Get processing status for a document - UI OPTIMIZED
    """
    try:
        status = processing_status.get(document_id, {"status": "not_found"})
        ui_state = ui_states.get(document_id)
        
        response = {
            "document_id": document_id,
            "status": status["status"],
            "timestamp": status.get("timestamp"),
            "error": status.get("error")
        }
        
        # Add UI state if available
        if ui_state:
            response["ui_state"] = {
                "current_step": ui_state.current_step,
                "overall_progress": ui_state.overall_progress,
                "user_message": ui_state.user_message,
                "next_action": ui_state.next_action,
                "can_proceed": ui_state.can_proceed,
                "agent_status": {
                    agent_name: {
                        "status": agent.status,
                        "progress": agent.processing_time or 0.0,
                        "confidence": agent.confidence
                    }
                    for agent_name, agent in ui_state.agents.items()
                }
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualize/{document_id}")
async def visualize_document(document_id: str):
    """
    Generate visualization of document with detected elements
    Returns annotated image or visualization data - UI OPTIMIZED
    """
    try:
        logger.info(f"Visualization request for document: {document_id}")
        
        # Check if document exists
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        if not os.path.exists(doc_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Check for processing results
        result_file = os.path.join(doc_dir, "processing_result.json")
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Processing results not found for document {document_id}"
            )
        
        # Load results
        with open(result_file, "r") as f:
            results = json.load(f)
        
        # Check for original image
        original_files = [
            f for f in os.listdir(doc_dir) 
            if f.startswith("original") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        visualization_data = {
            "document_id": document_id,
            "has_original_image": len(original_files) > 0,
            "detected_elements": [],
            "visualization_available": False,
            "ui_friendly": True
        }
        
        # Extract detected elements in UI-friendly format
        visual_elements = results.get("visual_elements", [])
        if visual_elements:
            visualization_data["detected_elements"] = [
                {
                    "id": f"elem_{i}",
                    "type": elem.get("type", "unknown"),
                    "confidence": elem.get("confidence", 0),
                    "page": elem.get("page_num", 1),
                    "bounds": elem.get("bbox", []),
                    "label": f"{elem.get('type', 'Element').title()} (Page {elem.get('page_num', 1)})",
                    "description": generate_element_description(elem)
                }
                for i, elem in enumerate(visual_elements)
            ]
            visualization_data["element_count"] = len(visual_elements)
            visualization_data["visualization_available"] = True
            
            # Group elements by type for UI display
            elements_by_type = {}
            for elem in visual_elements:
                elem_type = elem.get("type", "unknown")
                if elem_type not in elements_by_type:
                    elements_by_type[elem_type] = 0
                elements_by_type[elem_type] += 1
            
            visualization_data["elements_by_type"] = [
                {"type": elem_type, "count": count, "icon": get_element_icon(elem_type)}
                for elem_type, count in elements_by_type.items()
            ]
        
        # If we have an image, add its info
        if original_files:
            original_path = os.path.join(doc_dir, original_files[0])
            file_size = os.path.getsize(original_path)
            visualization_data.update({
                "original_image": {
                    "filename": original_files[0],
                    "size_kb": file_size / 1024,
                    "available": True
                }
            })
        
        return {
            "success": True,
            "visualization": visualization_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/index")
async def index_document(
    document_id: str,
    background_tasks: BackgroundTasks
):
    """
    Index a processed document in the RAG system - UI OPTIMIZED
    """
    try:
        logger.info(f"RAG index request for document: {document_id}")
        
        # Load processing results
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Processing results not found for document {document_id}"
            )
        
        with open(result_file, "r") as f:
            processing_results = json.load(f)
        
        # Extract text content for indexing
        text_content = extract_text_for_indexing(processing_results)
        
        # Extract images if available
        images = []
        # In production, load actual images
        
        # Start background indexing
        background_tasks.add_task(
            index_document_background,
            document_id,
            text_content,
            images,
            processing_results
        )
        
        # Update UI state
        if document_id in ui_states:
            ui_states[document_id].user_message = "Document indexing started"
            ui_states[document_id].next_action = "Wait for indexing to complete"
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document indexing started",
            "text_length": len(text_content),
            "estimated_time": "30 seconds",  # Placeholder
            "ui_state": {
                "message": "Indexing document for search...",
                "progress": 0.5
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG indexing request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/search")
async def search_documents(
    search_request: dict
):
    """
    Search indexed documents - UI OPTIMIZED
    
    Example request:
    {
        "query": "revenue chart contradiction",
        "query_type": "text",
        "limit": 5
    }
    """
    try:
        query = search_request.get("query")
        query_type = search_request.get("query_type", "text")
        limit = search_request.get("limit", 5)
        
        if not query:
            raise HTTPException(
                status_code=400,
                detail="query is required"
            )
        
        logger.info(f"RAG search: {query[:50]}...")
        
        # Perform search
        results = await retriever.search_documents(
            query=query,
            query_type=query_type,
            limit=limit
        )
        
        # Format results for UI
        ui_formatted_results = []
        for result in results:
            ui_result = {
                "document_id": result.get("document_id", "unknown"),
                "score": result.get("score", 0),
                "confidence": result.get("confidence", 0),
                "type": result.get("type", "text"),
                "preview": result.get("text_snippet", "")[:200] + "..." if result.get("text_snippet") else "",
                "metadata": {
                    "page": result.get("page", 1),
                    "source": result.get("source", "unknown"),
                    "has_visual": "image_data" in result
                }
            }
            ui_formatted_results.append(ui_result)
        
        return {
            "success": True,
            "query": query,
            "results": ui_formatted_results,
            "count": len(results),
            "ui_summary": {
                "total_matches": len(results),
                "best_match_score": max([r.get("score", 0) for r in results]) if results else 0,
                "has_visual_results": any("image_data" in r for r in results)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ui/state/{document_id}")
async def get_ui_state(document_id: str):
    """
    Get UI-specific state for a document
    """
    try:
        if document_id not in ui_states:
            raise HTTPException(
                status_code=404,
                detail=f"UI state not found for document {document_id}"
            )
        
        return {
            "success": True,
            "document_id": document_id,
            "ui_state": ui_states[document_id].dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"UI state retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== HELPER FUNCTIONS ==========

def format_agent_outputs_for_ui(results: dict) -> dict:
    """Format agent outputs for clean UI display"""
    agent_outputs = {}
    
    agent_configs = {
        "vision": {
            "name": "Vision Agent",
            "icon": "👁️",
            "description": "Analyzes visual elements"
        },
        "text": {
            "name": "Text Agent", 
            "icon": "📝",
            "description": "Extracts and analyzes text"
        },
        "fusion": {
            "name": "Fusion Agent",
            "icon": "🔄",
            "description": "Combines vision and text data"
        },
        "validation": {
            "name": "Validation Agent",
            "icon": "✅",
            "description": "Validates consistency and quality"
        }
    }
    
    for agent_key, config in agent_configs.items():
        confidence = results.get("confidence_scores", {}).get(agent_key, 0.0)
        
        # Generate key findings based on agent type
        key_findings = []
        if agent_key == "vision":
            elements = results.get("visual_elements", [])
            key_findings.append(f"Detected {len(elements)} visual elements")
        elif agent_key == "text":
            text = results.get("extracted_text", "")
            word_count = len(text.split())
            key_findings.append(f"Extracted {word_count} words")
        elif agent_key == "validation":
            contradictions = results.get("contradictions", [])
            key_findings.append(f"Found {len(contradictions)} potential issues")
        
        agent_outputs[agent_key] = {
            "name": config["name"],
            "icon": config["icon"],
            "description": config["description"],
            "status": "completed",
            "confidence": confidence,
            "key_findings": key_findings,
            "processing_time": results.get("processing_metadata", {}).get(f"{agent_key}_time", 0),
            "success": True
        }
    
    return agent_outputs

def format_visual_elements_for_ui(visual_elements):
    """Format visual elements for clean UI display"""
    formatted = []
    for i, elem in enumerate(visual_elements):
        formatted.append({
            "id": f"elem_{i}",
            "type": elem.get("type", "unknown"),
            "confidence": elem.get("confidence", 0.0),
            "page": elem.get("page_num", 1),
            "bounds": elem.get("bbox", []),
            "label": f"{elem.get('type', 'Element').title()} (Page {elem.get('page_num', 1)})",
            "description": generate_element_description(elem),
            "ui_color": get_element_color(elem.get("type"))
        })
    return formatted

def format_fields_for_ui(fields):
    """Format extracted fields for clean UI display"""
    formatted = {}
    for field_name, field_data in fields.items():
        formatted[field_name] = {
            "value": field_data.get("value", ""),
            "confidence": field_data.get("confidence", 0.0),
            "sources": field_data.get("sources", []),
            "modalities": field_data.get("modalities", []),
            "display_value": format_field_value(field_data.get("value")),
            "importance": "high" if field_data.get("confidence", 0) > 0.8 else "medium" if field_data.get("confidence", 0) > 0.5 else "low"
        }
    return formatted

def format_contradictions_for_ui(contradictions):
    """Format contradictions for clean UI display"""
    formatted = []
    for contra in contradictions:
        severity = contra.get("severity", "medium")
        formatted.append({
            "id": f"issue_{len(formatted)}",
            "type": contra.get("contradiction_type", "unknown").replace("_", " ").title(),
            "severity": severity,
            "confidence": contra.get("confidence", 0.0),
            "description": contra.get("explanation", "No description"),
            "recommendation": f"Review {contra.get('field_a', 'field A')} and {contra.get('field_b', 'field B')}",
            "fields_involved": [contra.get('field_a'), contra.get('field_b')],
            "ui_icon": get_severity_icon(severity),
            "ui_color": get_severity_color(severity)
        })
    return formatted

def generate_ui_answer(question: str, extracted_data: dict, processing_results: dict) -> dict:
    """Generate answer with UI-friendly format"""
    question_lower = question.lower()
    
    # Initialize response
    response = {
        "answer": "I've analyzed the document and found relevant information.",
        "confidence": 0.8,
        "sources": ["document_analysis"],
        "evidence": [],
        "type": "general",
        "has_visual": False,
        "recommended_action": "Review the detailed results for more information"
    }
    
    # Check for specific question types
    if any(word in question_lower for word in ["chart", "graph", "figure"]):
        elements = processing_results.get("visual_elements", [])
        chart_elements = [e for e in elements if e.get("type") in ["chart", "graph"]]
        
        if chart_elements:
            response.update({
                "answer": f"Found {len(chart_elements)} chart(s) in the document. The charts provide visual representation of data that complements the textual information.",
                "confidence": 0.9,
                "type": "visual",
                "has_visual": True,
                "evidence": [f"Chart on page {e.get('page_num', 1)}" for e in chart_elements[:3]]
            })
    
    elif any(word in question_lower for word in ["table", "data", "numbers"]):
        elements = processing_results.get("visual_elements", [])
        table_elements = [e for e in elements if e.get("type") == "table"]
        
        if table_elements:
            response.update({
                "answer": f"Found {len(table_elements)} table(s) containing structured data. Tables organize information for easy comparison and analysis.",
                "confidence": 0.85,
                "type": "structured",
                "evidence": [f"Table on page {e.get('page_num', 1)}" for e in table_elements[:3]]
            })
    
    elif any(word in question_lower for word in ["signature", "sign", "approval"]):
        elements = processing_results.get("visual_elements", [])
        signature_elements = [e for e in elements if e.get("type") == "signature"]
        
        if signature_elements:
            response.update({
                "answer": f"Found {len(signature_elements)} signature(s) in the document. Signatures indicate approval or authorization.",
                "confidence": 0.95 if signature_elements else 0.7,
                "type": "authentication",
                "evidence": [f"Signature on page {e.get('page_num', 1)}" for e in signature_elements]
            })
    
    elif any(word in question_lower for word in ["contradiction", "error", "issue", "problem"]):
        contradictions = processing_results.get("contradictions", [])
        
        if contradictions:
            critical_count = sum(1 for c in contradictions if c.get("severity") in ["high", "critical"])
            response.update({
                "answer": f"Found {len(contradictions)} potential issues in the document ({critical_count} critical). These may indicate inconsistencies or errors that need review.",
                "confidence": 0.9,
                "type": "validation",
                "evidence": [c.get("explanation", "Issue found") for c in contradictions[:2]]
            })
    
    return response

def extract_text_for_indexing(processing_results: dict) -> str:
    """Extract text content for RAG indexing"""
    text_parts = []
    
    # Extract from text results
    text_summary = processing_results.get("agent_outputs", {}).get("text", {}).get("summary", {})
    if text_summary:
        text_parts.append(str(text_summary))
    
    # Extract from OCR results
    ocr_results = processing_results.get("agent_outputs", {}).get("text", {}).get("ocr_results", [])
    for ocr_result in ocr_results:
        if "text_preview" in ocr_result:
            text_parts.append(ocr_result["text_preview"])
    
    # Extract from fields
    fields = processing_results.get("extracted_data", {}).get("fields", {})
    for field_name, field_data in fields.items():
        text_parts.append(f"{field_name}: {field_data.get('value', '')}")
    
    return "\n".join(text_parts)

def calculate_risk_level(contradictions: list) -> str:
    """Calculate overall risk level for UI display"""
    if not contradictions:
        return "low"
    
    severities = [c.get("severity", "medium") for c in contradictions]
    
    if any(s == "critical" for s in severities):
        return "high"
    elif any(s == "high" for s in severities):
        return "medium"
    else:
        return "low"

def generate_element_description(element: dict) -> str:
    """Generate human-readable description for visual element"""
    elem_type = element.get("type", "element")
    page = element.get("page_num", 1)
    confidence = element.get("confidence", 0) * 100
    
    descriptions = {
        "table": f"Data table on page {page}",
        "chart": f"Chart/graph on page {page}",
        "figure": f"Figure/image on page {page}",
        "signature": f"Signature area on page {page}",
        "text": f"Text block on page {page}"
    }
    
    base_desc = descriptions.get(elem_type, f"{elem_type.title()} on page {page}")
    return f"{base_desc} ({confidence:.0f}% confidence)"

def get_element_icon(elem_type: str) -> str:
    """Get icon for element type"""
    icons = {
        "table": "📊",
        "chart": "📈",
        "figure": "🖼️",
        "signature": "✍️",
        "text": "📝"
    }
    return icons.get(elem_type, "🔍")

def get_element_color(elem_type: str) -> str:
    """Get color for element type"""
    colors = {
        "table": "#3b82f6",  # blue
        "chart": "#10b981",  # green
        "figure": "#8b5cf6", # purple
        "signature": "#f59e0b", # amber
        "text": "#6b7280"    # gray
    }
    return colors.get(elem_type, "#9ca3af")

def get_severity_icon(severity: str) -> str:
    """Get icon for severity level"""
    icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢"
    }
    return icons.get(severity, "⚪")

def get_severity_color(severity: str) -> str:
    """Get color for severity level"""
    colors = {
        "critical": "#ef4444",  # red
        "high": "#f97316",     # orange
        "medium": "#eab308",   # yellow
        "low": "#22c55e"       # green
    }
    return colors.get(severity, "#6b7280")

def format_field_value(value) -> str:
    """Format field value for display"""
    if value is None:
        return "Not found"
    if isinstance(value, (int, float)):
        return f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
    if isinstance(value, str):
        return value[:100] + "..." if len(value) > 100 else value
    return str(value)

async def process_document_background(document_id: str, file_path: str):
    """Background task for document processing with UI state updates"""
    try:
        logger.info(f"Starting background processing for {document_id}")
        
        # Update UI state
        if document_id in ui_states:
            ui_state = ui_states[document_id]
            ui_state.current_step = "processing"
            ui_state.overall_progress = 0.3
            ui_state.user_message = "Processing document with AI agents"
            ui_state.next_action = "Processing in progress..."
            ui_state.can_proceed = False
            
            # Update agent statuses
            for agent_name in ui_state.agents:
                ui_state.agents[agent_name].status = "processing"
        
        # Update processing status
        processing_status[document_id] = {
            "status": "processing",
            "timestamp": datetime.now().isoformat(),
            "ui_state": "processing"
        }
        
        # Simulate agent processing with progress updates
        agents = ["vision", "text", "fusion", "validation"]
        for i, agent_name in enumerate(agents):
            await simulate_agent_processing(document_id, agent_name, i/len(agents))
        
        # Process document
        result = await orchestrator.process_document(file_path)
        
        # Save result
        doc_dir = os.path.dirname(file_path)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Update UI state
        if document_id in ui_states:
            ui_state = ui_states[document_id]
            ui_state.current_step = "results"
            ui_state.overall_progress = 1.0
            ui_state.user_message = "Processing completed successfully"
            ui_state.next_action = "View analysis results"
            ui_state.can_proceed = True
            
            # Update with actual results
            ui_state.document_type = result.get("document_type", "unknown")
            ui_state.visual_elements_count = len(result.get("visual_elements", []))
            ui_state.extracted_fields_count = len(result.get("extracted_data", {}).get("fields", {}))
            ui_state.contradictions_count = len(result.get("contradictions", []))
            ui_state.overall_confidence = result.get("confidence_scores", {}).get("overall", 0.0)
            
            # Update agent statuses
            for agent_name, agent in ui_state.agents.items():
                agent.status = "completed"
                agent.confidence = result.get("confidence_scores", {}).get(agent_name, 0.8)
                agent.summary = f"{agent_name.title()} agent completed successfully"
        
        # Update processing status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False),
            "ui_state": "ready_for_viewing"
        }
        
        logger.info(f"Background processing completed for {document_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")
        
        # Update UI state with error
        if document_id in ui_states:
            ui_states[document_id].current_step = "error"
            ui_states[document_id].user_message = f"Processing failed: {str(e)}"
            ui_states[document_id].next_action = "Try again or upload a new document"
            ui_states[document_id].can_proceed = True
        
        processing_status[document_id] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "ui_state": "error"
        }

async def simulate_agent_processing(document_id: str, agent_name: str, base_progress: float):
    """Simulate agent processing for UI progress updates"""
    import asyncio
    
    if document_id not in ui_states:
        return
    
    ui_state = ui_states[document_id]
    
    # Update agent status
    if agent_name in ui_state.agents:
        ui_state.agents[agent_name].status = "processing"
    
    # Simulate processing time
    steps = 5
    for step in range(steps):
        await asyncio.sleep(0.5)  # Simulate work
        
        # Update overall progress
        progress = base_progress + (step + 1) / (steps * len(ui_state.agents))
        ui_state.overall_progress = min(progress, 0.9)
        
        # Update agent-specific progress
        if agent_name in ui_state.agents:
            ui_state.agents[agent_name].processing_time = (step + 1) / steps
    
    # Mark agent as completed
    if agent_name in ui_state.agents:
        ui_state.agents[agent_name].status = "completed"
        ui_state.agents[agent_name].processing_time = 1.0

async def index_document_background(document_id: str, text_content: str, 
                                   images: list, metadata: dict):
    """Background task for document indexing"""
    try:
        logger.info(f"Starting background indexing for {document_id}")
        
        success = await retriever.index_document(
            document_id=document_id,
            text_content=text_content,
            images=images,
            metadata=metadata
        )
        
        if success:
            logger.info(f"Background indexing completed for {document_id}")
            
            # Update UI state
            if document_id in ui_states:
                ui_states[document_id].user_message = "Document indexed for search"
                ui_states[document_id].next_action = "You can now search this document"
        else:
            logger.error(f"Background indexing failed for {document_id}")
            
    except Exception as e:
        logger.error(f"Background indexing failed for {document_id}: {e}")