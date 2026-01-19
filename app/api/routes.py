from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
import json
from datetime import datetime
import asyncio

from app.core.config import settings
from app.agents.orchestrator import AgentOrchestrator
from app.rag.retriever import MultiModalRetriever
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Store processing status and results (in production, use Redis)
processing_status = {}
processing_results = {}

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing
    """
    try:
        logger.info(f"📤 Upload request received for file: {file.filename}")
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
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
        
        logger.info(f"✅ File saved to {file_path}")
        
        # Initialize status
        processing_status[document_id] = {
            "status": "uploaded",
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "file_path": file_path
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
            "message": "Document uploaded and processing started",
            "filename": file.filename,
            "status_endpoint": f"/api/v1/status/{document_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process")
async def process_document(
    document_id: str,
    reprocess: bool = False
):
    """
    Trigger document processing or reprocessing
    """
    try:
        logger.info(f"⚙️ Process request for document: {document_id}")
        
        # Check if document exists
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        if not os.path.exists(doc_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
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
        
        # Update status
        processing_status[document_id] = {
            "status": "processing",
            "timestamp": datetime.now().isoformat(),
            "reprocess": reprocess
        }
        
        # Process document
        from app.main import app
        processor = app.state.document_processor
        
        result = await processor.process_document(file_path, document_id)
        
        # Store results
        processing_results[document_id] = result
        
        # Update status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False)
        }
        
        # Index in RAG if successful
        if result.get("success"):
            try:
                # Extract text content for indexing
                text_content = ""
                if "text_results" in result:
                    for page_result in result["text_results"].values():
                        text_content += page_result.get("text", "") + "\n"
                
                if text_content and hasattr(app.state, 'retriever') and app.state.retriever:
                    await app.state.retriever.index_document(
                        document_id=document_id,
                        text_content=text_content,
                        metadata={
                            "filename": original_files[0],
                            "processing_time": datetime.now().isoformat()
                        }
                    )
                    logger.info(f"✅ Document {document_id} indexed in RAG")
            except Exception as e:
                logger.warning(f"RAG indexing failed for {document_id}: {e}")
        
        return {
            "success": True,
            "document_id": document_id,
            "processing_complete": True,
            "result_available": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        
        # Update status with error
        processing_status[document_id] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/query")
async def query_document(
    query_request: dict
):
    """
    Query processed documents using RAG
    """
    try:
        document_id = query_request.get("document_id")
        question = query_request.get("question")
        
        if not document_id or not question:
            raise HTTPException(
                status_code=400,
                detail="document_id and question are required"
            )
        
        logger.info(f"🔍 Query request: document={document_id}, question={question[:50]}...")
        
        # Check if document exists and is processed
        if document_id not in processing_results:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found or not processed"
            )
        
        results = processing_results[document_id]
        
        # Get retriever from app state
        from app.main import app
        if not hasattr(app.state, 'retriever'):
            raise HTTPException(
                status_code=503,
                detail="RAG system not available"
            )
        
        # Perform RAG search
        rag_results = await app.state.retriever.search_documents(
            query=question,
            filters={"document_id": document_id},
            limit=3
        )
        
        # Generate answer using retrieved context
        context_text = ""
        sources = []
        
        if rag_results:
            for i, result in enumerate(rag_results[:3]):
                context_text += f"[Source {i+1}]: {result.get('text', '')}\n\n"
                sources.append({
                    "score": result.get("score", 0),
                    "text_preview": result.get("text", "")[:200] + "..."
                })
        
        # Combine with extracted fields for better answer
        extracted_info = ""
        if "text_results" in results:
            for page_num, page_data in results["text_results"].items():
                extracted_info += f"Page {page_num}: {page_data.get('text', '')[:200]}...\n"
        
        # Generate answer (simplified - in production use LLM)
        if context_text or extracted_info:
            answer = f"Based on the document analysis:\n\n"
            
            if extracted_info:
                answer += f"Extracted information:\n{extracted_info}\n"
            
            if context_text:
                answer += f"Relevant context from document:\n{context_text[:1000]}..."
            
            answer += f"\n\nAnswer to your question '{question}': The document contains relevant information as shown above."
        else:
            answer = f"Unable to find specific information about '{question}' in the document."
        
        return {
            "success": True,
            "document_id": document_id,
            "question": question,
            "answer": answer,
            "confidence": 0.7 if context_text or extracted_info else 0.3,
            "sources": sources,
            "has_context": bool(context_text or extracted_info)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/results/{document_id}")
async def get_results(document_id: str):
    """
    Get processing results for a document
    """
    try:
        logger.info(f"📄 Results request for document: {document_id}")
        
        # Check if results exist
        if document_id not in processing_results:
            # Check if file exists but not processed
            doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
            if os.path.exists(doc_dir):
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {document_id} found but not processed yet"
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {document_id} not found"
                )
        
        results = processing_results[document_id]
        status = processing_status.get(document_id, {"status": "unknown"})
        
        # Add document content preview
        enriched_results = results.copy()
        enriched_results["status"] = status["status"]
        enriched_results["filename"] = status.get("filename", "")
        
        return {
            "success": True,
            "document_id": document_id,
            "status": status["status"],
            "results": enriched_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Results retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.get("/status/{document_id}")
async def get_status(document_id: str):
    """
    Get processing status for a document
    """
    try:
        status = processing_status.get(document_id)
        
        if not status:
            # Check if document directory exists
            doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
            if os.path.exists(doc_dir):
                status = {
                    "status": "uploaded",
                    "timestamp": datetime.fromtimestamp(os.path.getctime(doc_dir)).isoformat()
                }
            else:
                status = {
                    "status": "not_found",
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "document_id": document_id,
            "status": status["status"],
            "timestamp": status.get("timestamp"),
            "filename": status.get("filename", ""),
            "error": status.get("error")
        }
        
    except Exception as e:
        logger.error(f"❌ Status retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/search")
async def rag_search(
    search_request: dict
):
    """
    Search across indexed documents using RAG
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
        
        logger.info(f"🔎 RAG search: {query[:50]}...")
        
        # Get retriever from app state
        from app.main import app
        if not hasattr(app.state, 'retriever'):
            raise HTTPException(
                status_code=503,
                detail="RAG system not available"
            )
        
        # Perform search
        results = await app.state.retriever.search_documents(
            query=query,
            query_type=query_type,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document_id": result.get("document_id", "unknown"),
                "score": result.get("score", 0),
                "text": result.get("text", "")[:500],
                "metadata": result.get("metadata", {})
            })
        
        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RAG search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")

async def process_document_background(document_id: str, file_path: str):
    """Background task for document processing"""
    try:
        logger.info(f"🔄 Starting background processing for {document_id}")
        
        # Update status
        processing_status[document_id] = {
            "status": "processing",
            "timestamp": datetime.now().isoformat(),
            "filename": os.path.basename(file_path)
        }
        
        # Get processor from app state
        from app.main import app
        processor = app.state.document_processor
        
        # Process document
        result = await processor.process_document(file_path, document_id)
        
        # Store results
        processing_results[document_id] = result
        
        # Update status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False)
        }
        
        logger.info(f"✅ Background processing completed for {document_id}")
        
    except Exception as e:
        logger.error(f"❌ Background processing failed for {document_id}: {e}", exc_info=True)
        
        processing_status[document_id] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }