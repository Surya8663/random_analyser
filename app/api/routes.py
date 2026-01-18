from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import uuid
import shutil
from datetime import datetime
import traceback
import json 

from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Initialize components with robust error handling
# Initialize components with robust error handling
try:
    from app.agents.orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator()
    logger.info("✅ AgentOrchestrator initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize AgentOrchestrator: {e}")
    logger.error(traceback.format_exc())
    
    # Create a minimal orchestrator as fallback
    class MinimalOrchestrator:
        async def process_document(self, images, file_path=None):
            import uuid
            from datetime import datetime
            return {
                "success": True,
                "document_id": str(uuid.uuid4()),
                "document_type": "UNKNOWN",
                "extracted_fields": {
                    "info": {
                        "value": f"Document processed in fallback mode: {file_path or 'unknown'}",
                        "confidence": 0.5,
                        "sources": ["fallback"],
                        "modalities": ["textual"]
                    },
                    "sample_data": {
                        "value": {"test": "This is sample data from fallback mode"},
                        "confidence": 0.3,
                        "sources": ["test"],
                        "modalities": ["metadata"]
                    }
                },
                "validation_results": {
                    "contradictions": [],
                    "risk_score": 0.3,
                    "integrity_score": 0.7
                },
                "explanations": {"processing": "Running in fallback mode"},
                "recommendations": ["System is running in basic mode"],
                "processing_metadata": {
                    "mode": "fallback",
                    "timestamp": datetime.now().isoformat()
                },
                "errors": []
            }
    
    orchestrator = MinimalOrchestrator()
    logger.warning("⚠️ Using fallback orchestrator")

# Initialize retriever if available
try:
    from app.rag.retriever import MultiModalRetriever
    retriever = MultiModalRetriever()
    logger.info("✅ MultiModalRetriever initialized successfully")
except Exception as e:
    logger.warning(f"⚠️ MultiModalRetriever not available: {e}")
    retriever = None

# ... rest of the routes.py file continues as before ...

# Store processing status (in production, use Redis or database)
processing_status = {}

def validate_orchestrator():
    """Validate that orchestrator is available"""
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Document processing service is unavailable"
        )

@router.get("/quick-test")
async def quick_test():
    """Quick test endpoint that returns immediately"""
    return {
        "status": "ok",
        "message": "API is responsive",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "upload": "/api/v1/upload",
            "status": "/api/v1/status/{id}",
            "results": "/api/v1/results/{id}",
            "health": "/health"
        }
    }

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing
    
    Supports PDF and image files
    """
    try:
        logger.info(f"Upload request received for file: {file.filename}")
        
        # Validate file type quickly
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
        
        # Save uploaded file (quick operation)
        file_path = os.path.join(upload_dir, f"original{file_ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {file_path}")
        
        # IMMEDIATELY return response, start processing in background
        # Don't wait for background task to complete
        background_tasks.add_task(
            process_document_background,
            document_id,
            file_path
        )
        
        # Set initial status
        processing_status[document_id] = {
            "status": "uploaded",
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document uploaded and processing started",
            "status_endpoint": f"/api/v1/status/{document_id}",
            "results_endpoint": f"/api/v1/results/{document_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process")
async def process_document(
    document_id: str,
    reprocess: bool = False
):
    """
    Trigger document processing
    
    Can be used to reprocess an already uploaded document
    """
    try:
        validate_orchestrator()
        
        logger.info(f"Process request for document: {document_id}")
        
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
        
        # Process document
        result = await orchestrator.process_document([], file_path)
        
        # VALIDATE RESULT (CRITICAL FIX)
        if not result or 'success' not in result:
            logger.error(f"Invalid result from orchestrator: {result}")
            raise HTTPException(
                status_code=500,
                detail="Document processing failed: Invalid response from orchestrator"
            )
        
        # Save result
        result_file = os.path.join(doc_dir, "processing_result.json")
        import json
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Update status
        processing_status[document_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False)
        }
        
        logger.info(f"Processing completed for {document_id}: success={result.get('success')}")
        
        return {
            "success": True,
            "document_id": document_id,
            "processing_complete": True,
            "result_available": True,
            "result_summary": {
                "fields_extracted": len(result.get("extracted_fields", {})),
                "has_errors": len(result.get("errors", [])) > 0,
                "integrity_score": result.get("validation_results", {}).get("integrity_score", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        
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
    Query processed documents
    
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
        
        import json
        with open(result_file, "r") as f:
            processing_results = json.load(f)
        
        # Validate processing results
        if not processing_results or 'success' not in processing_results:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid processing results for document {document_id}"
            )
        
        # Extract relevant information for answering
        extracted_data = processing_results
        
        # Generate answer
        answer = generate_answer(question, extracted_data)
        
        return {
            "success": True,
            "document_id": document_id,
            "question": question,
            "answer": answer,
            "confidence": 0.8,
            "sources": ["extracted_fields", "validation_results"],
            "response_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/results/{document_id}")
async def get_results(document_id: str):
    """
    Get processing results for a document
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
        
        import json
        with open(result_file, "r") as f:
            results = json.load(f)
        
        # VALIDATE RESULTS (CRITICAL FIX)
        if not results:
            raise HTTPException(
                status_code=500,
                detail=f"Empty results for document {document_id}"
            )
        
        # Add status information
        status = processing_status.get(document_id, {"status": "unknown"})
        
        response = {
            "success": True,
            "document_id": document_id,
            "status": status["status"],
            "results": results,
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Results retrieved for {document_id}: success={results.get('success')}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results retrieval failed: {e}")
        logger.error(traceback.format_exc())
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
                status = {"status": "uploaded", "timestamp": datetime.now().isoformat()}
            else:
                status = {"status": "not_found", "timestamp": datetime.now().isoformat()}
        
        response = {
            "document_id": document_id,
            "status": status["status"],
            "timestamp": status.get("timestamp", datetime.now().isoformat()),
            "error": status.get("error"),
            "success": status.get("success", status["status"] == "completed")
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/visualize/{document_id}")
async def visualize_document(document_id: str):
    """
    Generate visualization of document with detected elements
    Returns annotated image or visualization data
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
        import json
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
            "visualization_available": False
        }
        
        # Extract detected elements from results
        extracted_fields = results.get("extracted_fields", {})
        element_count = 0
        
        # Count different types of elements
        element_types = {}
        for field_name in extracted_fields.keys():
            if "chart" in field_name.lower():
                element_types["chart"] = element_types.get("chart", 0) + 1
                element_count += 1
            elif "table" in field_name.lower():
                element_types["table"] = element_types.get("table", 0) + 1
                element_count += 1
        
        visualization_data.update({
            "element_count": element_count,
            "visualization_available": element_count > 0,
            "elements_by_type": element_types,
            "extracted_fields_count": len(extracted_fields)
        })
        
        return {
            "success": True,
            "visualization": visualization_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/rag/index")
async def index_document(
    document_id: str,
    background_tasks: BackgroundTasks
):
    """
    Index a processed document in the RAG system
    """
    try:
        if retriever is None:
            raise HTTPException(
                status_code=503,
                detail="RAG indexing service is unavailable"
            )
        
        logger.info(f"RAG index request for document: {document_id}")
        
        # Load processing results
        doc_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        result_file = os.path.join(doc_dir, "processing_result.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(
                status_code=404,
                detail=f"Processing results not found for document {document_id}"
            )
        
        import json
        with open(result_file, "r") as f:
            processing_results = json.load(f)
        
        # Extract text content for indexing
        text_content = extract_text_for_indexing(processing_results)
        
        if not text_content or len(text_content.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Not enough text content for indexing"
            )
        
        # Extract images if available
        images = []
        
        # Start background indexing
        background_tasks.add_task(
            index_document_background,
            document_id,
            text_content,
            images,
            processing_results
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document indexing started",
            "text_length": len(text_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG indexing request failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"RAG indexing request failed: {str(e)}")

@router.post("/rag/search")
async def search_documents(
    search_request: dict
):
    """
    Search indexed documents
    
    Example request:
    {
        "query": "revenue chart contradiction",
        "query_type": "text",
        "limit": 5
    }
    """
    try:
        if retriever is None:
            raise HTTPException(
                status_code=503,
                detail="RAG search service is unavailable"
            )
        
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
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "search_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")

def generate_answer(question: str, extracted_data: dict) -> str:
    """Generate answer from extracted data with improved logic"""
    
    # Extract relevant parts from the data
    extracted_fields = extracted_data.get("extracted_fields", {})
    validation_results = extracted_data.get("validation_results", {})
    contradictions = validation_results.get("contradictions", [])
    
    # Simple rule-based answering
    if "chart" in question.lower():
        chart_fields = [f for f in extracted_fields.keys() if "chart" in f.lower()]
        if chart_fields:
            return f"Found {len(chart_fields)} chart(s) in the document. Chart fields: {', '.join(chart_fields)}"
        else:
            return "No charts were detected in this document."
    
    elif "table" in question.lower():
        table_fields = [f for f in extracted_fields.keys() if "table" in f.lower()]
        if table_fields:
            return f"Found {len(table_fields)} table(s) in the document. Table fields: {', '.join(table_fields)}"
        else:
            return "No tables were detected in this document."
    
    elif "contradiction" in question.lower() or "inconsistency" in question.lower():
        if contradictions:
            return f"Found {len(contradictions)} contradiction(s) in the document. Types: {', '.join([c.get('type', 'unknown') for c in contradictions])}"
        else:
            return "No contradictions were found in the document."
    
    elif "risk" in question.lower():
        risk_score = validation_results.get("risk_score", 0)
        return f"The document has a risk score of {risk_score:.2f} (on a scale of 0 to 1)."
    
    elif "integrity" in question.lower():
        integrity_score = validation_results.get("integrity_score", 0)
        return f"The document integrity score is {integrity_score:.2f} (on a scale of 0 to 1)."
    
    elif "type" in question.lower() and "document" in question.lower():
        doc_type = extracted_data.get("document_type", "unknown")
        return f"The document type is: {doc_type}"
    
    else:
        # General answer
        field_count = len(extracted_fields)
        if field_count > 0:
            sample_fields = list(extracted_fields.keys())[:3]
            return f"The document analysis extracted {field_count} fields including: {', '.join(sample_fields)}. Ask about specific fields for more details."
        else:
            return "The document analysis did not extract any specific fields. This could be due to document complexity or processing limitations."

def extract_text_for_indexing(processing_results: dict) -> str:
    """Extract text content for RAG indexing"""
    text_parts = []
    
    # Extract from extracted fields
    extracted_fields = processing_results.get("extracted_fields", {})
    for field_name, field_data in extracted_fields.items():
        value = field_data.get("value", "")
        if value:
            if isinstance(value, (list, dict)):
                text_parts.append(f"{field_name}: {json.dumps(value)}")
            else:
                text_parts.append(f"{field_name}: {value}")
    
    # Extract from document type
    doc_type = processing_results.get("document_type", "unknown")
    text_parts.append(f"Document type: {doc_type}")
    
    # Extract from validation results
    validation_results = processing_results.get("validation_results", {})
    if validation_results.get("contradictions"):
        text_parts.append(f"Contradictions found: {len(validation_results['contradictions'])}")
    
    integrity_score = validation_results.get("integrity_score", 0)
    text_parts.append(f"Document integrity score: {integrity_score:.2f}")
    
    return "\n".join(text_parts)

async def process_document_background(document_id: str, file_path: str):
    """Background task for document processing - SIMPLIFIED"""
    try:
        logger.info(f"Starting background processing for {document_id}")
        
        # Update status to processing
        processing_status[document_id] = {
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        }
        
        # SIMPLE PROCESSING - avoid complex operations
        try:
            # Read file to get bytes
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Process with orchestrator
            result = await orchestrator.process_document([file_bytes], file_path)
            
            # Validate result
            if not result or 'success' not in result:
                logger.error(f"Invalid result from orchestrator for {document_id}")
                raise ValueError("Invalid orchestrator response")
            
            # Save result
            doc_dir = os.path.dirname(file_path)
            result_file = os.path.join(doc_dir, "processing_result.json")
            
            import json
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            
            # Update status
            processing_status[document_id] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "success": result.get("success", False)
            }
            
            logger.info(f"Background processing completed for {document_id}: success={result.get('success')}")
            
        except Exception as processing_error:
            logger.error(f"Processing failed for {document_id}: {processing_error}")
            
            # Create a simple error result
            error_result = {
                "success": False,
                "error": str(processing_error),
                "document_id": document_id,
                "extracted_fields": {},
                "errors": [str(processing_error)]
            }
            
            # Save error result
            doc_dir = os.path.dirname(file_path)
            result_file = os.path.join(doc_dir, "processing_result.json")
            
            import json
            with open(result_file, "w") as f:
                json.dump(error_result, f, indent=2)
            
            # Update status with error
            processing_status[document_id] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(processing_error)
            }
        
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")
        logger.error(traceback.format_exc())
        
        processing_status[document_id] = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Background processing failed: {str(e)}"
        }

async def index_document_background(document_id: str, text_content: str, 
                                   images: list, metadata: dict):
    """Background task for document indexing with error handling"""
    try:
        if retriever is None:
            logger.error(f"Cannot index document {document_id}: retriever is None")
            return
        
        logger.info(f"Starting background indexing for {document_id}")
        
        success = await retriever.index_document(
            document_id=document_id,
            text_content=text_content,
            images=images,
            metadata=metadata
        )
        
        if success:
            logger.info(f"Background indexing completed for {document_id}")
        else:
            logger.error(f"Background indexing failed for {document_id}")
        
    except Exception as e:
        logger.error(f"Background indexing failed for {document_id}: {e}")
        logger.error(traceback.format_exc())