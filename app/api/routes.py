# app/api/routes.py - COMPLETE FIXED VERSION
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import uuid
import os
import shutil
from datetime import datetime
import json

from app.core.models import MultiModalDocument, ProcessingStep
from app.services.document_processor import DocumentProcessor
from app.agents.orchestrator import Phase3Orchestrator
from app.core.config import settings
from app.eval.evaluator import DocumentEvaluator
from app.explain.explainability import ExplainabilityGenerator

router = APIRouter()
phase3_orchestrator = Phase3Orchestrator()
evaluator = DocumentEvaluator()
explainability_gen = ExplainabilityGenerator()

# Global storage for processing state
processing_status = {}
processing_results = {}
explainability_reports = {}
evaluation_reports = {}


@router.post("/phase4/process")
async def phase4_process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ground_truth_file: Optional[str] = None
):
    """
    Phase 4 document processing with explainability and evaluation
    NOW FIXED to use the actual process_document method
    """
    try:
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = os.path.join(settings.UPLOAD_DIR, f"phase4_{document_id}")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(upload_dir, f"original{file_ext}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update status
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 10,
            "message": "File uploaded, starting Phase 4 processing...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        # Start background processing - NOW IT WILL WORK!
        background_tasks.add_task(
            phase4_background_processing,
            document_id,
            file_path,
            ground_truth_file
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document uploaded for Phase 4 processing with explainability",
            "phase": 4,
            "status_url": f"/api/v1/phase4/status/{document_id}",
            "results_url": f"/api/v1/phase4/results/{document_id}",
            "explain_url": f"/api/v1/phase4/explain/{document_id}",
            "eval_url": f"/api/v1/phase4/evaluate/{document_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def phase4_background_processing(document_id: str, file_path: str, ground_truth_file: Optional[str] = None):
    """
    Background processing for Phase 4 - NOW FIXED!
    """
    try:
        # Step 1: Update status
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 30,
            "message": "Running document processor...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        # Step 2: Run DocumentProcessor - NOW WITH WORKING METHOD!
        logger = setup_logger("phase4_processing")
        logger.info(f"Starting Phase 4 processing for document: {document_id}")
        
        doc_processor = DocumentProcessor()
        
        # THIS LINE NOW WORKS BECAUSE process_document METHOD EXISTS!
        base_document = await doc_processor.process_document(file_path, document_id)
        
        # Step 3: Update status
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 60,
            "message": "Running Phase 4 intelligence pipeline with explainability...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        # Step 4: Run Phase 4 orchestration
        logger.info(f"Running Phase 3 orchestrator for document: {document_id}")
        result = await phase3_orchestrator.process_document(base_document)
        
        # Step 5: Extract explainability and evaluation reports
        explainability_report = result.get("detailed_results", {}).get("explainability", {})
        evaluation_report = result.get("detailed_results", {}).get("evaluation", {})
        
        # Step 6: Store all results
        processing_results[document_id] = result
        explainability_reports[document_id] = explainability_report
        evaluation_reports[document_id] = evaluation_report
        
        # Step 7: Update final status
        processing_status[document_id] = {
            "step": ProcessingStep.RESULTS.value,
            "progress": 100,
            "message": f"Phase 4 processing completed. Risk score: {result.get('risk_score', 0):.2f}",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        logger.info(f"Phase 4 processing completed for document: {document_id}")
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        logger.error(f"Phase 4 processing failed for {document_id}: {error_msg}")
        
        processing_status[document_id] = {
            "step": ProcessingStep.ERROR.value,
            "progress": 0,
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        processing_results[document_id] = {
            "success": False,
            "error": error_msg,
            "document_id": document_id
        }


@router.get("/phase4/status/{document_id}")
async def phase4_status(document_id: str):
    """Get Phase 4 processing status"""
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return processing_status[document_id]


@router.get("/phase4/results/{document_id}")
async def phase4_results(document_id: str):
    """Get Phase 4 processing results"""
    if document_id not in processing_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    result = processing_results[document_id]
    
    # Format response
    response = {
        "phase": 4,
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


@router.get("/phase4/explain/{document_id}")
async def phase4_explain(document_id: str):
    """Get Phase 4 explainability report"""
    if document_id not in explainability_reports:
        raise HTTPException(status_code=404, detail="Explainability report not found")
    
    return {
        "document_id": document_id,
        "phase": 4,
        "explainability_report": explainability_reports[document_id],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/phase4/evaluate/{document_id}")
async def phase4_evaluate(document_id: str):
    """Get Phase 4 evaluation report"""
    if document_id not in evaluation_reports:
        raise HTTPException(status_code=404, detail="Evaluation report not found")
    
    return {
        "document_id": document_id,
        "phase": 4,
        "evaluation_report": evaluation_reports[document_id],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/phase4/summary")
async def phase4_summary():
    """Get summary of all Phase 4 evaluations"""
    summary = evaluator.generate_summary_report()
    return {
        "phase": 4,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/phase4/test")
async def phase4_test():
    """Test Phase 4 endpoints"""
    return {
        "phase": 4,
        "status": "active",
        "endpoints": {
            "process": "/api/v1/phase4/process",
            "status": "/api/v1/phase4/status/{document_id}",
            "results": "/api/v1/phase4/results/{document_id}",
            "explain": "/api/v1/phase4/explain/{document_id}",
            "evaluate": "/api/v1/phase4/evaluate/{document_id}",
            "summary": "/api/v1/phase4/summary",
            "test": "/api/v1/phase4/test"
        },
        "features": ["explainability", "evaluation", "provenance_tracking", "real_metrics"],
        "timestamp": datetime.now().isoformat()
    }


def setup_logger(name: str):
    """Helper function to set up logger"""
    import logging
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger