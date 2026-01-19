# routes.py - FIXED VERSION
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
from datetime import datetime
import json
from pathlib import Path

from app.core.config import settings
from app.services.document_processor import DocumentProcessor
from app.agents.orchestrator import AgentOrchestrator
from app.rag.retriever import MultiModalRetriever
from app.utils.logger import setup_logger
from app.utils.error_handler import handle_api_error
from app.core.models import ProcessingState, UIResultSummary, AgentResult, UIProcessingState, ProcessingStep

logger = setup_logger(__name__)

router = APIRouter()

# Initialize components
orchestrator = AgentOrchestrator()
retriever = MultiModalRetriever()

# Store processing status
processing_status = {}
ui_states = {}

# Supported extensions
SUPPORTED_EXTENSIONS = {
    '.pdf', '.png', '.jpg', '.jpeg', '.jpe', '.bmp', '.tiff', '.tif',
    '.doc', '.docx', '.txt', '.csv', '.rtf'
}

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,  # FIXED: Parameter defined
    file: UploadFile = File(...)
):
    """
    Upload a document for processing
    """
    try:
        logger.info(f"Upload request received for file: {file.filename}")
        
        # Get file extension
        file_path = Path(file.filename)
        file_ext = file_path.suffix.lower()
        
        # Validate file type
        if file_ext not in SUPPORTED_EXTENSIONS:
            # Try to detect by content type
            content_type = file.content_type or ''
            if not any(ext in content_type for ext in ['pdf', 'image', 'text', 'msword', 'csv']):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not supported. Allowed: {list(SUPPORTED_EXTENSIONS)}"
                )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create upload directory
        upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file with original extension
        file_path = os.path.join(upload_dir, f"original{file_ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {file_path}")
        
        # Initialize UI state
        ui_states[document_id] = UIProcessingState(
            document_id=document_id,
            current_step=ProcessingStep.UPLOAD,
            overall_progress=0.1,
            agents={
                "quality": AgentResult(agent_name="quality", status="pending"),
                "classifier": AgentResult(agent_name="classifier", status="pending"),
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
            "filename": file.filename,
            "file_size": os.path.getsize(file_path),
            "file_type": file_ext,
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
            "filename": file.filename,
            "file_size": os.path.getsize(file_path),
            "file_type": file_ext,
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
    Trigger document processing with ALL AGENTS
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
            ui_state = ui_states[document_id]
            # ✅ FIXED: Use string instead of ProcessingStep.PROCESSING
            ui_state.current_step = "processing"
            ui_state.overall_progress = 0.3
            ui_state.user_message = "Processing started with all agents"
            ui_state.next_action = "Wait for processing to complete"
            ui_state.can_proceed = False
            
            # Update all agent statuses to processing
            for agent_name in ui_state.agents:
                ui_state.agents[agent_name].status = "processing"
        
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
        
        # Start processing with orchestrator
        result = await orchestrator.process_document(file_path, document_id)
        
        # Save result
        result_file = os.path.join(doc_dir, "processing_result.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Update UI state with agent outputs
        if document_id in ui_states:
            ui_state = ui_states[document_id]
            # ✅ FIXED: Use string instead of ProcessingStep.RESULTS
            ui_state.current_step = "results"
            ui_state.overall_progress = 1.0
            ui_state.user_message = "Processing completed successfully"
            ui_state.next_action = "View results"
            ui_state.can_proceed = True
            
            # Update document type
            if result.get("document_type"):
                ui_state.document_type = result.get("document_type")
            
            # Update agent statuses with actual results
            if "agent_outputs" in result:
                for agent_name, agent_data in result["agent_outputs"].items():
                    if agent_name in ui_state.agents:
                        ui_state.agents[agent_name].status = "completed"
                        ui_state.agents[agent_name].confidence = agent_data.get("confidence", 0.8)
                        ui_state.agents[agent_name].summary = agent_data.get("summary", "")
                        ui_state.agents[agent_name].key_findings = agent_data.get("key_findings", [])
            
            # Update counts
            ui_state.visual_elements_count = len(result.get("visual_elements", []))
            ui_state.extracted_fields_count = len(result.get("extracted_fields", {}))
            ui_state.contradictions_count = len(result.get("contradictions", []))
            ui_state.overall_confidence = result.get("overall_confidence", 0.0)
        
        # Update processing status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False),
            "agent_count": len(result.get("agent_outputs", {})),
            "element_count": len(result.get("visual_elements", [])),
            "field_count": len(result.get("extracted_fields", {})),
            "ui_state": "ready_for_viewing"
        }
        
        return {
            "success": True,
            "document_id": document_id,
            "processing_complete": True,
            "result_available": True,
            "agents_executed": list(result.get("agent_outputs", {}).keys()),
            "ui_state": {
                "current_step": "results",
                "progress": 1.0,
                "message": "Document processed successfully",
                "can_proceed": True,
                "agent_count": len(result.get("agent_outputs", {}))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        
        # Update UI state with error
        if document_id in ui_states:
            # ✅ FIXED: Use string instead of ProcessingStep.ERROR
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

# ========== BACKGROUND TASK FUNCTIONS ==========

async def process_document_background(document_id: str, file_path: str):
    """Background task for document processing"""
    try:
        logger.info(f"Starting background processing for {document_id}")
        
        # Update UI state
        if document_id in ui_states:
            ui_state = ui_states[document_id]
            # ✅ FIXED: Use string instead of ProcessingStep.PROCESSING
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
        
        # Process document
        result = await orchestrator.process_document(file_path, document_id)
        
        # Save result
        doc_dir = os.path.dirname(file_path)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Update UI state
        if document_id in ui_states:
            ui_state = ui_states[document_id]
            # ✅ FIXED: Use string instead of ProcessingStep.RESULTS
            ui_state.current_step = "results"
            ui_state.overall_progress = 1.0
            ui_state.user_message = "Processing completed successfully"
            ui_state.next_action = "View analysis results"
            ui_state.can_proceed = True
            
            # Update with actual results
            ui_state.document_type = result.get("document_type", "unknown")
            ui_state.visual_elements_count = len(result.get("visual_elements", []))
            ui_state.extracted_fields_count = len(result.get("extracted_fields", {}))
            ui_state.contradictions_count = len(result.get("contradictions", []))
            ui_state.overall_confidence = result.get("overall_confidence", 0.0)
            
            # Update agent statuses
            if "agent_outputs" in result:
                for agent_name, agent_data in result["agent_outputs"].items():
                    if agent_name in ui_state.agents:
                        ui_state.agents[agent_name].status = "completed"
                        ui_state.agents[agent_name].confidence = agent_data.get("confidence", 0.8)
                        ui_state.agents[agent_name].summary = f"{agent_name.title()} agent completed successfully"
        
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
            # ✅ FIXED: Use string instead of ProcessingStep.ERROR
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

async def index_document_background(document_id: str, metadata: dict):
    """Background task for document indexing"""
    try:
        logger.info(f"Starting background indexing for {document_id}")
        
        # Extract text from results
        text_content = metadata.get("extracted_text", "")
        
        if text_content:
            # Index in RAG system
            success = await retriever.index_document(
                document_id=document_id,
                text_content=text_content,
                images=None,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Background indexing completed for {document_id}")
                
                # Update UI state
                if document_id in ui_states:
                    ui_states[document_id].user_message = "Document indexed for search"
                    ui_states[document_id].next_action = "You can now search this document"
        else:
            logger.warning(f"No text content for indexing document {document_id}")
            
    except Exception as e:
        logger.error(f"Background indexing failed for {document_id}: {e}")

# ========== OTHER ROUTES (KEEP AS BEFORE) ==========

@router.post("/query")
async def query_document(query_request: dict):
    """Query processed documents"""
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
        
        # Extract relevant information
        extracted_text = processing_results.get("extracted_text", "")
        
        # Simple keyword-based answering for demonstration
        answer_data = generate_simple_answer(question, extracted_text, processing_results)
        
        return {
            "success": True,
            "document_id": document_id,
            "question": question,
            "answer": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "sources": answer_data["sources"],
            "supporting_evidence": answer_data["evidence"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{document_id}")
async def get_results(document_id: str):
    """Get processing results for a document"""
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
        
        # UI-optimized response
        ui_optimized_results = {
            "success": True,
            "document_id": document_id,
            "status": status["status"],
            "document_type": results.get("document_type", "unknown"),
            "processing_time": results.get("processing_time"),
            "overall_confidence": results.get("overall_confidence", 0.0),
            
            # Agent outputs
            "agent_outputs": results.get("agent_outputs", {}),
            
            # Data
            "visual_elements": results.get("visual_elements", []),
            "extracted_fields": results.get("extracted_fields", {}),
            "contradictions": results.get("contradictions", []),
            "extracted_text": results.get("extracted_text", ""),
            
            # Summary
            "summary": {
                "total_pages": results.get("processing_metadata", {}).get("page_count", 1),
                "total_elements": len(results.get("visual_elements", [])),
                "total_fields": len(results.get("extracted_fields", {})),
                "risk_level": "low" if len(results.get("contradictions", [])) == 0 else "medium"
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
    """Get processing status for a document"""
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
                "current_step": ui_state.current_step.value if hasattr(ui_state.current_step, 'value') else ui_state.current_step,
                "overall_progress": ui_state.overall_progress,
                "user_message": ui_state.user_message,
                "next_action": ui_state.next_action,
                "can_proceed": ui_state.can_proceed,
                "agent_status": {
                    agent_name: {
                        "status": agent.status,
                        "confidence": agent.confidence
                    }
                    for agent_name, agent in ui_state.agents.items()
                }
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_simple_answer(question: str, extracted_text: str, processing_results: dict) -> dict:
    """Generate simple answer based on keywords"""
    question_lower = question.lower()
    
    response = {
        "answer": "Based on the document analysis, I found relevant information.",
        "confidence": 0.7,
        "sources": ["document_analysis"],
        "evidence": []
    }
    
    # Check for keywords
    if any(word in question_lower for word in ["table", "data", "numbers"]):
        elements = processing_results.get("visual_elements", [])
        tables = [e for e in elements if e.get("type") == "table"]
        
        if tables:
            response.update({
                "answer": f"The document contains {len(tables)} table(s) with structured data.",
                "confidence": 0.85,
                "evidence": [f"Table on page {e.get('page_num', 1)}" for e in tables[:2]]
            })
    
    elif any(word in question_lower for word in ["chart", "graph", "figure"]):
        elements = processing_results.get("visual_elements", [])
        charts = [e for e in elements if e.get("type") in ["chart", "graph"]]
        
        if charts:
            response.update({
                "answer": f"The document contains {len(charts)} chart(s) for data visualization.",
                "confidence": 0.8,
                "evidence": [f"Chart on page {e.get('page_num', 1)}" for e in charts[:2]]
            })
    
    elif any(word in question_lower for word in ["signature", "sign", "approval"]):
        elements = processing_results.get("visual_elements", [])
        signatures = [e for e in elements if e.get("type") == "signature"]
        
        if signatures:
            response.update({
                "answer": f"The document contains {len(signatures)} signature(s) indicating approval.",
                "confidence": 0.9,
                "evidence": [f"Signature on page {e.get('page_num', 1)}" for e in signatures]
            })
    
    elif any(word in question_lower for word in ["summary", "overview", "about"]):
        text_preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        response.update({
            "answer": f"Document summary: {text_preview}",
            "confidence": 0.75
        })
    
    return response