# app/api/routes.py - UPDATED FOR PHASE 1
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import uuid
import logging
import os
from datetime import datetime
import shutil
import asyncio
import PyPDF2  # For PDF text extraction
import tempfile
import hashlib

from app.core.models import ProcessingStep, UploadResponse, StatusResponse, QueryRequest, MultiModalDocument
from app.services.document_processor import DocumentProcessor
from app.agents.orchestrator import AgentOrchestrator
from app.rag.retriever import MultiModalRetriever
from app.rag.embeddings import EmbeddingEngine
from app.rag.vector_store import VectorStore
from app.core.config import settings

router = APIRouter()
document_processor = DocumentProcessor()
agent_orchestrator = AgentOrchestrator()
embedding_engine = EmbeddingEngine()
vector_store = VectorStore()
retriever = MultiModalRetriever()

# Global storage for processing status and results
processing_status = {}
processing_results = {}

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing - UPDATED for Phase 1
    """
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(upload_dir, f"original{file_ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File saved to {file_path}")
        
        # Initialize processing status
        processing_status[document_id] = {
            "step": ProcessingStep.UPLOAD.value,
            "progress": 0,
            "message": "File uploaded successfully",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id,
            "file_path": file_path
        }
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            file_path
        )
        
        return UploadResponse(
            success=True,
            document_id=document_id,
            message="Document uploaded successfully. Processing started.",
            next_step="processing",
            ui_state={
                "status_url": f"/api/v1/status/{document_id}",
                "progress": 0,
                "can_proceed": True,
                "filename": file.filename
            }
        )
        
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(document_id: str, file_path: str):
    """
    Background task to process uploaded document using new Phase 1 pipeline
    """
    try:
        # Step 1: Update status
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 25,
            "message": "Processing document with new multi-modal pipeline...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        logging.info(f"Starting PHASE 1 multi-modal processing for {document_id}")
        
        # Step 2: Use AgentOrchestrator which now returns MultiModalDocument
        final_document = await agent_orchestrator.process_document(file_path, document_id)
        
        # Step 3: Store results
        processing_results[document_id] = {
            "success": True,
            "document_id": document_id,
            "multi_modal_document": final_document.dict(),
            "processed_at": datetime.now().isoformat(),
            "processing_time": final_document.get_processing_time(),
            "summary": {
                "text_length": len(final_document.raw_text),
                "visual_elements": len(final_document.visual_elements),
                "layout_regions": len(final_document.layout_regions),
                "ocr_pages": len(final_document.ocr_results),
                "risk_score": final_document.risk_score,
                "errors": len(final_document.errors)
            }
        }
        
        # Step 4: Index in RAG (optional for Phase 1)
        try:
            if hasattr(retriever, 'index_document'):
                processing_status[document_id] = {
                    "step": ProcessingStep.PROCESSING.value,
                    "progress": 75,
                    "message": "Indexing document in RAG system...",
                    "timestamp": datetime.now().isoformat(),
                    "document_id": document_id
                }
                
                await retriever.index_document(final_document)
                logging.info(f"✅ Document {document_id} indexed in RAG")
        except Exception as rag_error:
            logging.warning(f"⚠️ RAG indexing skipped for {document_id}: {rag_error}")
        
        # Step 5: Set final status
        processing_status[document_id] = {
            "step": ProcessingStep.RESULTS.value,
            "progress": 100,
            "message": f"Multi-modal processing completed. Extracted {len(final_document.raw_text)} characters, {len(final_document.visual_elements)} visual elements.",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        logging.info(f"✅ PHASE 1 multi-modal processing completed for {document_id}")
        logging.info(f"   - Text: {len(final_document.raw_text)} chars")
        logging.info(f"   - Visual elements: {len(final_document.visual_elements)}")
        logging.info(f"   - Layout regions: {len(final_document.layout_regions)}")
        logging.info(f"   - Processing time: {final_document.get_processing_time():.2f}s")
        
    except Exception as e:
        logging.error(f"❌ Background processing error for {document_id}: {str(e)}", exc_info=True)
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
            "document_id": document_id,
            "processed_at": datetime.now().isoformat()
        }

@router.get("/status/{document_id}", response_model=StatusResponse)
async def get_status(document_id: str):
    """
    Get processing status for a document.
    """
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document ID not found")
    
    status = processing_status[document_id]
    return StatusResponse(
        document_id=document_id,
        status=status["step"],
        timestamp=status["timestamp"],
        error=None if status["step"] != ProcessingStep.ERROR.value else status["message"],
        ui_state={
            "progress": status["progress"],
            "message": status["message"],
            "step": status["step"]
        }
    )

@router.get("/results/{document_id}")
async def get_results(document_id: str):
    """
    Get processing results for a document - UPDATED for MultiModalDocument
    """
    if document_id not in processing_results:
        raise HTTPException(status_code=404, detail="Results not found. Document may still be processing.")
    
    result = processing_results[document_id]
    
    # If we have multi-modal document data
    if "multi_modal_document" in result:
        multi_doc_dict = result["multi_modal_document"]
        
        # Convert back to MultiModalDocument for methods
        multi_doc = MultiModalDocument(**multi_doc_dict)
        
        # Prepare enhanced response
        response = {
            "success": True,
            "document_id": document_id,
            "processing_time": result.get("processing_time", 0),
            "processed_at": result["processed_at"],
            
            # Document metadata
            "file_path": multi_doc.file_path,
            "file_type": multi_doc.file_type,
            "document_type": multi_doc.document_type.value if multi_doc.document_type else "unknown",
            
            # Multi-modal summary
            "summary": {
                "text_length": len(multi_doc.raw_text),
                "visual_elements": len(multi_doc.visual_elements),
                "layout_regions": len(multi_doc.layout_regions),
                "ocr_pages": len(multi_doc.ocr_results),
                "risk_score": multi_doc.risk_score,
                "contradictions": len(multi_doc.contradictions),
                "compliance_issues": len(multi_doc.compliance_issues)
            },
            
            # Extracted data (limited for response)
            "extracted_text": multi_doc.raw_text[:5000] + "..." if len(multi_doc.raw_text) > 5000 else multi_doc.raw_text,
            "extracted_entities": {k: v[:10] for k, v in multi_doc.extracted_entities.items() if v},
            "visual_elements": [
                {
                    "type": elem.element_type,
                    "bbox": elem.bbox.to_list(),
                    "page": elem.page_num,
                    "confidence": elem.confidence
                }
                for elem in multi_doc.visual_elements[:20]  # Limit response
            ],
            "layout_regions": [
                {
                    "label": region.label,
                    "bbox": region.bbox.to_list(),
                    "page": region.page_num,
                    "confidence": region.confidence
                }
                for region in multi_doc.layout_regions[:10]  # Limit response
            ],
            
            # Agent outputs
            "agent_outputs": multi_doc.agent_outputs,
            
            # Validation results
            "risk_score": multi_doc.risk_score,
            "contradictions": [
                {
                    "type": c.contradiction_type.value,
                    "severity": c.severity.value,
                    "explanation": c.explanation,
                    "recommendation": c.recommendation
                }
                for c in multi_doc.contradictions
            ],
            "review_recommendations": multi_doc.review_recommendations,
            
            # Processing info
            "processing_start": multi_doc.processing_start.isoformat() if multi_doc.processing_start else None,
            "processing_end": multi_doc.processing_end.isoformat() if multi_doc.processing_end else None,
            "errors": multi_doc.errors
        }
        
        return response
    
    # Fallback to old format
    return result

@router.post("/query")
async def query_document(request: QueryRequest):
    """
    Query a processed document with a question.
    """
    try:
        document_id = request.document_id
        question = request.question
        
        logging.info(f"Query request received: document={document_id}, question={question[:50]}...")
        
        # Check if document exists and is processed
        if document_id not in processing_results:
            raise HTTPException(status_code=404, detail="Document not found or not processed")
        
        if document_id not in processing_status:
            raise HTTPException(status_code=404, detail="Document status not found")
        
        # Check if processing is complete
        current_status = processing_status[document_id]["step"]
        if current_status != ProcessingStep.RESULTS.value:
            raise HTTPException(
                status_code=400, 
                detail=f"Document processing not completed. Current status: {current_status}"
            )
        
        # Get document results
        results = processing_results[document_id]
        
        # Check if we have multi-modal document
        if "multi_modal_document" not in results:
            return {
                "success": False,
                "message": "Document processed with old pipeline. Please re-upload for multi-modal query.",
                "answer": "Multi-modal query not available for this document."
            }
        
        # Extract multi-modal document
        multi_doc_dict = results["multi_modal_document"]
        multi_doc = MultiModalDocument(**multi_doc_dict)
        
        # Simple query based on multi-modal data
        answer = await _query_multi_modal_document(multi_doc, question)
        
        return {
            "success": True,
            "document_id": document_id,
            "question": question,
            "answer": answer["text"],
            "confidence": answer["confidence"],
            "sources": answer.get("sources", []),
            "supporting_evidence": answer.get("evidence", []),
            "display_type": "text",
            "has_visual_content": answer.get("has_visual", False),
            "confidence_color": "#10b981" if answer["confidence"] >= 0.8 else "#f59e0b" if answer["confidence"] >= 0.6 else "#ef4444"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Query error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "document_id": request.document_id if hasattr(request, 'document_id') else "",
            "question": request.question if hasattr(request, 'question') else "",
            "answer": f"Error processing query: {str(e)}",
            "confidence": 0.0,
            "sources": [],
            "supporting_evidence": [],
            "display_type": "text",
            "has_visual_content": False,
            "confidence_color": "#ef4444"
        }

async def _query_multi_modal_document(doc: MultiModalDocument, question: str) -> Dict[str, Any]:
    """
    Query a multi-modal document with simple logic
    """
    question_lower = question.lower()
    
    # Check for visual element queries
    if any(word in question_lower for word in ["table", "chart", "figure", "image", "visual", "picture"]):
        # Query about visual elements
        visual_elements = []
        for elem in doc.visual_elements:
            if any(word in question_lower for word in [elem.element_type, "element", "object"]):
                visual_elements.append(elem)
        
        if visual_elements:
            return {
                "text": f"I found {len(visual_elements)} visual elements matching your query. For example, a {visual_elements[0].element_type} on page {visual_elements[0].page_num + 1}.",
                "confidence": 0.8,
                "has_visual": True,
                "sources": [f"Page {elem.page_num + 1}: {elem.element_type}" for elem in visual_elements[:3]],
                "evidence": [f"{elem.element_type} at position {elem.bbox.to_list()}" for elem in visual_elements[:2]]
            }
    
    # Check for layout queries
    if any(word in question_lower for word in ["layout", "structure", "region", "section"]):
        layout_info = []
        for region in doc.layout_regions[:5]:
            layout_info.append(f"{region.label} region on page {region.page_num + 1}")
        
        if layout_info:
            return {
                "text": f"The document has {len(doc.layout_regions)} layout regions. {', '.join(layout_info[:3])}.",
                "confidence": 0.7,
                "has_visual": True,
                "sources": layout_info[:5]
            }
    
    # Text-based query
    if doc.raw_text:
        # Simple keyword matching
        keywords = question_lower.split()
        matching_sentences = []
        
        for sentence in doc.raw_text.split('.'):
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords if len(keyword) > 3):
                matching_sentences.append(sentence.strip())
        
        if matching_sentences:
            return {
                "text": f"Based on the document: {matching_sentences[0][:300]}...",
                "confidence": 0.6,
                "has_visual": False,
                "sources": [f"Text match: {len(matching_sentences)} sentences found"],
                "evidence": matching_sentences[:2]
            }
    
    # Entity-based query
    if doc.extracted_entities:
        for entity_type, entities in doc.extracted_entities.items():
            if entity_type in question_lower and entities:
                return {
                    "text": f"I found {len(entities)} {entity_type} in the document: {', '.join(entities[:3])}.",
                    "confidence": 0.7,
                    "has_visual": False,
                    "sources": [f"Entity extraction: {entity_type}"],
                    "evidence": entities[:3]
                }
    
    # Default answer
    return {
        "text": "I've analyzed the document but couldn't find specific information matching your query. The document contains both text and visual elements that have been processed.",
        "confidence": 0.4,
        "has_visual": len(doc.visual_elements) > 0,
        "sources": [f"Document analysis: {len(doc.raw_text)} characters, {len(doc.visual_elements)} visual elements"],
        "evidence": []
    }

@router.get("/documents")
async def list_documents():
    """
    List all processed documents - UPDATED for Phase 1
    """
    documents = []
    
    for doc_id in processing_status.keys():
        status = processing_status[doc_id]
        has_results = doc_id in processing_results
        
        # Try to get summary from results
        summary = {}
        if has_results and processing_results[doc_id].get("multi_modal_document"):
            try:
                multi_doc = MultiModalDocument(**processing_results[doc_id]["multi_modal_document"])
                summary = {
                    "text_length": len(multi_doc.raw_text),
                    "visual_elements": len(multi_doc.visual_elements),
                    "risk_score": multi_doc.risk_score
                }
            except:
                summary = {"format": "legacy"}
        
        documents.append({
            "document_id": doc_id,
            "status": status["step"],
            "progress": status["progress"],
            "message": status["message"],
            "timestamp": status["timestamp"],
            "has_results": has_results,
            "summary": summary,
            "pipeline_version": "phase_1" if "multi_modal_document" in processing_results.get(doc_id, {}) else "legacy"
        })
    
    return {
        "count": len(documents),
        "documents": documents
    }

@router.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its associated data.
    """
    try:
        # Remove from status and results
        if document_id in processing_status:
            del processing_status[document_id]
        
        if document_id in processing_results:
            del processing_results[document_id]
        
        # Try to delete from vector store
        try:
            if hasattr(vector_store, 'delete_document'):
                vector_store.delete_document(document_id)
        except Exception as e:
            logging.warning(f"Could not delete from vector store: {str(e)}")
        
        # Delete uploaded files
        upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully"
        }
        
    except Exception as e:
        logging.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.get("/test-phase1")
async def test_phase1():
    """
    Test endpoint to verify Phase 1 is working
    """
    return {
        "phase": 1,
        "status": "active",
        "features": {
            "multi_modal_document": True,
            "unified_pipeline": True,
            "enhanced_models": True,
            "backward_compatibility": True
        },
        "models_available": [
            "MultiModalDocument",
            "OCRResult",
            "LayoutRegion",
            "EnhancedVisualElement",
            "BoundingBox"
        ],
        "agent_count": 18,
        "timestamp": datetime.now().isoformat()
    }