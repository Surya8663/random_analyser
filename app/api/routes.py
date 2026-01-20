# app/api/routes.py - UPDATED FOR PHASE 3
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from typing import Dict, Any, List
import uuid
import os
import shutil
from datetime import datetime

from app.core.models import MultiModalDocument, ProcessingStep
from app.services.document_processor import DocumentProcessor
from app.agents.orchestrator import Phase3Orchestrator
from app.core.config import settings

router = APIRouter()
phase3_orchestrator = Phase3Orchestrator()

# Global storage
processing_status = {}
processing_results = {}

@router.post("/phase3/process")
async def phase3_process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Phase 3 document processing endpoint"""
    try:
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = os.path.join(settings.UPLOAD_DIR, f"phase3_{document_id}")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(upload_dir, f"original{file_ext}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update status
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 10,
            "message": "File uploaded, starting Phase 3 processing...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        # Start background processing
        background_tasks.add_task(
            phase3_background_processing,
            document_id,
            file_path
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document uploaded for Phase 3 processing",
            "phase": 3,
            "status_url": f"/api/v1/phase3/status/{document_id}",
            "results_url": f"/api/v1/phase3/results/{document_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def phase3_background_processing(document_id: str, file_path: str):
    """Background processing for Phase 3"""
    try:
        # Step 1: Update status
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 30,
            "message": "Running computer vision pipeline...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        # Step 2: Run DocumentProcessor (Phase 2)
        from app.services.document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        
        base_document = await doc_processor.process_document(file_path, document_id)
        
        # Step 3: Update status
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 60,
            "message": "Running Phase 3 intelligence pipeline...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        # Step 4: Run Phase 3 orchestration
        result = await phase3_orchestrator.process_document(base_document)
        
        # Step 5: Store results
        processing_results[document_id] = result
        
        # Step 6: Update final status
        processing_status[document_id] = {
            "step": ProcessingStep.RESULTS.value,
            "progress": 100,
            "message": f"Phase 3 processing completed. Risk score: {result.get('risk_score', 0):.2f}",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
    except Exception as e:
        processing_status[document_id] = {
            "step": ProcessingStep.ERROR.value,
            "progress": 0,
            "message": f"Processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        processing_results[document_id] = {
            "success": False,
            "error": str(e),
            "document_id": document_id
        }

@router.get("/phase3/status/{document_id}")
async def phase3_status(document_id: str):
    """Get Phase 3 processing status"""
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return processing_status[document_id]

@router.get("/phase3/results/{document_id}")
async def phase3_results(document_id: str):
    """Get Phase 3 processing results"""
    if document_id not in processing_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    result = processing_results[document_id]
    
    # Format response
    response = {
        "phase": 3,
        "document_id": document_id,
        "success": result.get("success", False),
        "processing_time": result.get("processing_time", 0),
        "timestamp": datetime.now().isoformat()
    }
    
    if result.get("success"):
        response.update({
            "document_type": result.get("document_type", "unknown"),
            "risk_score": result.get("risk_score", 0),
            "risk_level": "HIGH" if result.get("risk_score", 0) > 0.7 else 
                         "MEDIUM" if result.get("risk_score", 0) > 0.4 else "LOW",
            "contradictions_count": result.get("contradictions_count", 0),
            "extracted_fields_count": result.get("extracted_fields_count", 0),
            "recommendations": result.get("recommendations", []),
            "detailed_results": result.get("detailed_results", {})
        })
    
    if "error" in result:
        response["error"] = result["error"]
    
    return response

@router.get("/phase3/test")
async def phase3_test():
    """Test Phase 3 endpoints"""
    return {
        "phase": 3,
        "status": "active",
        "endpoints": {
            "process": "/api/v1/phase3/process",
            "status": "/api/v1/phase3/status/{document_id}",
            "results": "/api/v1/phase3/results/{document_id}",
            "test": "/api/v1/phase3/test"
        },
        "agents": ["vision", "text", "fusion", "reasoning"],
        "timestamp": datetime.now().isoformat()
    }