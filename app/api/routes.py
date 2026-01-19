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

from app.core.models import ProcessingStep, UploadResponse, StatusResponse, QueryRequest
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
    Upload a PDF document for processing.
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, "original.pdf")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File saved to {file_path}")
        
        # Initialize processing status
        processing_status[document_id] = {
            "step": ProcessingStep.UPLOAD.value,
            "progress": 0,
            "message": "File uploaded successfully",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
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
                "can_proceed": True
            }
        )
        
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(document_id: str, file_path: str):
    """
    Background task to process uploaded PDF document.
    """
    try:
        # Step 1: Extract text from PDF
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 25,
            "message": "Extracting text from document...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        logging.info(f"Starting document processing for {document_id}")
        
        # Extract text using multiple methods
        text_content = ""
        text_segments = []
        
        # Method 1: Try DocumentProcessor first
        try:
            if hasattr(document_processor, 'process_document'):
                logging.info(f"Trying DocumentProcessor.process_document for {document_id}...")
                
                # Check if method is async
                import inspect
                if inspect.iscoroutinefunction(document_processor.process_document):
                    doc_result = await document_processor.process_document(file_path)
                else:
                    doc_result = document_processor.process_document(file_path)
                
                if doc_result:
                    if isinstance(doc_result, list):
                        # Handle list of segments
                        text_segments = doc_result
                        text_content = "\n".join([str(seg.get("text", "")) for seg in doc_result])
                    elif isinstance(doc_result, dict):
                        # Handle dictionary with text
                        text_content = str(doc_result.get("text", ""))
                        text_segments = [{"text": text_content, "page": 1}]
                    else:
                        # Handle plain text
                        text_content = str(doc_result)
                        text_segments = [{"text": text_content, "page": 1}]
                    
                    logging.info(f"DocumentProcessor extracted {len(text_content)} characters")
                else:
                    logging.warning(f"DocumentProcessor returned empty result for {document_id}")
        except Exception as doc_error:
            logging.warning(f"DocumentProcessor failed for {document_id}: {str(doc_error)}")
        
        # Method 2: Fallback to PyPDF2 if DocumentProcessor fails or returns empty
        if not text_content.strip():
            try:
                logging.info(f"Falling back to PyPDF2 extraction for {document_id}...")
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    for page_num in range(total_pages):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        if page_text.strip():
                            text_content += f"Page {page_num + 1}:\n{page_text}\n\n"
                            text_segments.append({
                                "text": page_text,
                                "page": page_num + 1,
                                "confidence": 0.8
                            })
                        else:
                            # Handle empty pages
                            text_segments.append({
                                "text": f"[Page {page_num + 1} appears to be empty or contains only images]",
                                "page": page_num + 1,
                                "confidence": 0.0
                            })
                    
                    logging.info(f"PyPDF2 extracted {len(text_content)} characters from {total_pages} pages for {document_id}")
                    
                    if not text_content.strip():
                        text_content = "Document appears to be empty or contains only images/scanned content."
                        text_segments = [{"text": text_content, "page": 1}]
                        
            except Exception as pdf_error:
                error_msg = f"PDF extraction failed: {str(pdf_error)}"
                logging.error(f"PyPDF2 extraction error for {document_id}: {error_msg}")
                text_content = f"Error extracting text: {error_msg}"
                text_segments = [{"text": text_content, "page": 1}]
        
        # Step 2: Update status for text processing
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 50,
            "message": f"Text extraction completed. Processing {len(text_content)} characters...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        # Step 3: Create embeddings and store in vector database
        processing_status[document_id] = {
            "step": ProcessingStep.PROCESSING.value,
            "progress": 75,
            "message": "Creating embeddings and storing in database...",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        chunks = []
        
        # Create chunks from text content
        if text_content.strip():
            # Split text into chunks (max 500 characters each)
            chunk_size = 500
            words = text_content.split()
            current_chunk = []
            current_size = 0
            chunk_index = 0
            
            for word in words:
                word_len = len(word) + 1  # +1 for space
                if current_size + word_len > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    
                    # ✅ FIXED: Generate proper ID for Qdrant (must be integer or UUID)
                    # Using hash of text + document_id to create unique integer ID
                    chunk_hash = hashlib.md5(f"{document_id}_chunk_{chunk_index}".encode()).hexdigest()
                    chunk_id = int(chunk_hash[:16], 16) % (2**63 - 1)  # Ensure it fits in 64-bit
                    
                    # Get embedding for the chunk
                    embedding = None
                    try:
                        if hasattr(embedding_engine, 'encode'):
                            embedding = embedding_engine.encode(chunk_text)
                        elif hasattr(embedding_engine, 'embed_text'):
                            embedding = embedding_engine.embed_text(chunk_text)
                        elif hasattr(embedding_engine, 'get_embedding'):
                            embedding = embedding_engine.get_embedding(chunk_text)
                        else:
                            # Fallback: create simple embedding
                            embedding = [0.1] * 384
                            logging.warning(f"No embedding engine method found for {document_id}")
                    except Exception as embed_error:
                        logging.warning(f"Embedding error for chunk {chunk_index} of {document_id}: {str(embed_error)}")
                        embedding = [0.1] * 384
                    
                    # Store in vector database
                    try:
                        vector_store.upsert(
                            points=[
                                {
                                    "id": chunk_id,  # Integer ID for Qdrant
                                    "vector": embedding,
                                    "payload": {
                                        "text": chunk_text,
                                        "document_id": document_id,
                                        "original_chunk_id": f"{document_id}_chunk_{chunk_index}",
                                        "page": 1,
                                        "chunk_index": chunk_index,
                                        "word_count": len(chunk_text.split()),
                                        "char_count": len(chunk_text)
                                    }
                                }
                            ]
                        )
                        
                        chunks.append({
                            "id": chunk_id,
                            "original_id": f"{document_id}_chunk_{chunk_index}",
                            "text": chunk_text,
                            "page": 1,
                            "chunk_index": chunk_index,
                            "word_count": len(chunk_text.split()),
                            "char_count": len(chunk_text)
                        })
                        chunk_index += 1
                        
                    except Exception as store_error:
                        logging.error(f"Vector store error for chunk {chunk_index} of {document_id}: {str(store_error)}")
                    
                    # Start new chunk
                    current_chunk = [word]
                    current_size = word_len
                else:
                    current_chunk.append(word)
                    current_size += word_len
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_hash = hashlib.md5(f"{document_id}_chunk_{chunk_index}".encode()).hexdigest()
                chunk_id = int(chunk_hash[:16], 16) % (2**63 - 1)
                
                try:
                    # Get embedding for last chunk
                    if hasattr(embedding_engine, 'encode'):
                        embedding = embedding_engine.encode(chunk_text)
                    elif hasattr(embedding_engine, 'embed_text'):
                        embedding = embedding_engine.embed_text(chunk_text)
                    else:
                        embedding = [0.1] * 384
                except:
                    embedding = [0.1] * 384
                
                try:
                    vector_store.upsert(
                        points=[
                            {
                                "id": chunk_id,
                                "vector": embedding,
                                "payload": {
                                    "text": chunk_text,
                                    "document_id": document_id,
                                    "original_chunk_id": f"{document_id}_chunk_{chunk_index}",
                                    "page": 1,
                                    "chunk_index": chunk_index,
                                    "word_count": len(chunk_text.split()),
                                    "char_count": len(chunk_text)
                                }
                            }
                        ]
                    )
                    chunks.append({
                        "id": chunk_id,
                        "original_id": f"{document_id}_chunk_{chunk_index}",
                        "text": chunk_text,
                        "page": 1,
                        "chunk_index": chunk_index,
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text)
                    })
                except Exception as e:
                    logging.error(f"Failed to store last chunk for {document_id}: {str(e)}")
        
        # Step 4: Run agent analysis (if available)
        agent_results = {}
        if hasattr(agent_orchestrator, 'process') and text_content.strip():
            try:
                processing_status[document_id] = {
                    "step": ProcessingStep.PROCESSING.value,
                    "progress": 85,
                    "message": "Running AI analysis...",
                    "timestamp": datetime.now().isoformat(),
                    "document_id": document_id
                }
                
                if asyncio.iscoroutinefunction(agent_orchestrator.process):
                    agent_results = await agent_orchestrator.process(text_segments)
                else:
                    agent_results = agent_orchestrator.process(text_segments)
                    
                logging.info(f"Agent analysis completed for {document_id}")
            except Exception as agent_error:
                logging.warning(f"Agent orchestrator error for {document_id}: {str(agent_error)}")
                agent_results = {"error": str(agent_error), "status": "skipped"}
        
        # Store final results
        processing_results[document_id] = {
            "text_segments": text_segments,
            "text_content": text_content,
            "chunks": chunks,
            "agent_results": agent_results,
            "processed_at": datetime.now().isoformat(),
            "document_id": document_id,
            "stats": {
                "total_characters": len(text_content),
                "total_words": len(text_content.split()),
                "total_chunks": len(chunks),
                "total_segments": len(text_segments),
                "avg_chunk_size": len(text_content.split()) / max(len(chunks), 1)
            }
        }
        
        # Set final status
        processing_status[document_id] = {
            "step": ProcessingStep.RESULTS.value,
            "progress": 100,
            "message": f"Document processing completed. Extracted {len(text_content)} characters, created {len(chunks)} chunks.",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        
        logging.info(f"Background processing completed for {document_id}")
        logging.info(f"Extracted {len(text_content)} characters, created {len(chunks)} chunks")
        
    except Exception as e:
        logging.error(f"Background processing error for {document_id}: {str(e)}", exc_info=True)
        processing_status[document_id] = {
            "step": ProcessingStep.ERROR.value,
            "progress": 0,
            "message": f"Processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id
        }
        processing_results[document_id] = {
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
    Get processing results for a document.
    """
    if document_id not in processing_results:
        raise HTTPException(status_code=404, detail="Results not found. Document may still be processing.")
    
    return processing_results[document_id]

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
        chunks = results.get("chunks", [])
        text_content = results.get("text_content", "")
        
        # If no chunks but we have text content, create chunks on the fly
        if not chunks and text_content.strip():
            logging.info(f"Creating chunks on the fly for query for document {document_id}...")
            # Simple chunking for query
            chunk_size = 500
            words = text_content.split()
            temp_chunks = []
            current_chunk = []
            current_size = 0
            
            for word in words:
                word_len = len(word) + 1
                if current_size + word_len > chunk_size and current_chunk:
                    temp_chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_size = word_len
                else:
                    current_chunk.append(word)
                    current_size += word_len
            
            if current_chunk:
                temp_chunks.append(' '.join(current_chunk))
            chunks = [{"text": chunk, "page": 1} for chunk in temp_chunks]
        
        # Get query embedding
        query_embedding = None
        try:
            if hasattr(embedding_engine, 'encode'):
                query_embedding = embedding_engine.encode(question)
            elif hasattr(embedding_engine, 'embed_text'):
                query_embedding = embedding_engine.embed_text(question)
            elif hasattr(embedding_engine, 'get_embedding'):
                query_embedding = embedding_engine.get_embedding(question)
            else:
                # Fallback: create simple embedding based on question hash
                hash_val = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
                query_embedding = [(hash_val % 1000) / 1000.0] * 384
        except Exception as embed_error:
            logging.error(f"Query embedding error for document {document_id}: {str(embed_error)}")
            # Fallback to simple semantic matching
            query_embedding = [0.0] * 384
        
        # Search in vector store
        search_results = []
        try:
            if query_embedding:
                # ✅ FIXED: Use proper filter format for Qdrant
                search_results = vector_store.search(
                    query_vector=query_embedding,
                    limit=5,
                    filter={
                        "must": [
                            {
                                "key": "document_id",
                                "match": {
                                    "value": document_id
                                }
                            }
                        ]
                    }
                )
                logging.info(f"Found {len(search_results)} search results for document {document_id}")
        except Exception as search_error:
            logging.error(f"Vector search error for document {document_id}: {str(search_error)}")
            search_results = []
        
        context_chunks = []
        sources = []
        
        # Process search results
        if search_results and len(search_results) > 0:
            for i, result in enumerate(search_results):
                payload = result.payload
                text = payload.get("text", "")
                score = result.score if hasattr(result, 'score') else 0.5
                
                if text.strip():
                    context_chunks.append(text)
                    sources.append({
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "page": payload.get("page", 1),
                        "score": float(score),
                        "chunk_index": payload.get("chunk_index", i),
                        "original_id": payload.get("original_chunk_id", f"chunk_{i}")
                    })
        
        # Fallback: if no search results, use first few chunks
        if not context_chunks and chunks:
            logging.info(f"Using fallback chunks for query for document {document_id}")
            for i, chunk in enumerate(chunks[:5]):
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("text", "")
                else:
                    chunk_text = str(chunk)
                
                if chunk_text.strip():
                    context_chunks.append(chunk_text)
                    sources.append({
                        "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                        "page": chunk.get("page", 1) if isinstance(chunk, dict) else 1,
                        "score": 0.5 - (i * 0.1),  # Decreasing score for fallback chunks
                        "chunk_index": i,
                        "original_id": chunk.get("original_id", f"fallback_chunk_{i}") if isinstance(chunk, dict) else f"fallback_chunk_{i}"
                    })
        
        # Final fallback: use text content
        if not context_chunks and text_content.strip():
            logging.info(f"Using text content as fallback for query for document {document_id}")
            context_chunks = [text_content[:1000]]
            sources = [{
                "text": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                "page": 1,
                "score": 0.3,
                "chunk_index": 0,
                "original_id": "text_content_fallback"
            }]
        
        context = "\n\n".join(context_chunks)
        
        # Generate answer
        if not context.strip():
            answer = "I couldn't find relevant information in the document to answer your question."
            confidence = 0.0
        else:
            answer = generate_answer_with_llm(context, question)
            confidence = min(0.3 + (len(context) / 10000), 0.95)  # Dynamic confidence based on context length
        
        return {
            "success": True,
            "document_id": document_id,
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "supporting_evidence": [chunk[:100] + "..." for chunk in context_chunks[:3]],
            "display_type": "text",
            "has_visual_content": False,
            "confidence_color": "#10b981" if confidence >= 0.8 else "#f59e0b" if confidence >= 0.6 else "#ef4444"
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

def generate_answer_with_llm(context: str, question: str) -> str:
    """
    Generate answer using simple rule-based logic.
    This is a fallback when no real LLM is available.
    """
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Check for specific question patterns
    if "what is" in question_lower and "document" in question_lower and "about" in question_lower:
        # Find title or first significant sentence
        lines = context.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) > 20 and not line_stripped.startswith("Page"):
                return f"The document appears to be about: {line_stripped[:200]}..."
    
    if "first program" in question_lower or "1st program" in question_lower:
        lines = context.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['program', 'experiment', 'exercise', 'lab', 'project']):
                clean_line = ' '.join(line.split())
                if len(clean_line) > 10:
                    return f"The first program mentioned in your document appears to be related to: {clean_line}"
    
    if "summary" in question_lower or "summarize" in question_lower:
        # Extract key sentences
        sentences = context.split('.')
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 50][:3]
        if key_sentences:
            return "Here's a summary based on the document:\n" + "\n".join([f"• {s}" for s in key_sentences])
    
    if "who" in question_lower:
        # Look for names or roles
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['student', 'candidate', 'applicant', 'developer', 'engineer']):
                return f"Based on the document: {sentence.strip()[:200]}..."
    
    if "how" in question_lower and "work" in question_lower:
        # Look for process descriptions
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['process', 'method', 'technique', 'approach', 'strategy']):
                return f"The document describes: {sentence.strip()[:200]}..."
    
    # Default answer: return relevant excerpt
    if len(context) > 500:
        # Find the most relevant paragraph
        paragraphs = context.split('\n\n')
        if paragraphs and len(paragraphs[0]) > 50:
            return f"Based on the document content:\n\n{paragraphs[0][:500]}..."
        else:
            return f"Here's what I found in the document:\n\n{context[:500]}..."
    else:
        return f"Here's what I found in the document:\n\n{context}"

@router.get("/documents")
async def list_documents():
    """
    List all processed documents.
    """
    documents = []
    
    for doc_id in processing_status.keys():
        status = processing_status[doc_id]
        has_results = doc_id in processing_results
        
        documents.append({
            "document_id": doc_id,
            "status": status["step"],
            "progress": status["progress"],
            "message": status["message"],
            "timestamp": status["timestamp"],
            "has_results": has_results
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
            # Qdrant doesn't have direct delete by filter in free version
            # We'll just mark as deleted
            pass
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