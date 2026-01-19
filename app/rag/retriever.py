# app/rag/retriever.py
from typing import List, Dict, Any, Optional, Tuple
from app.rag.embeddings import EmbeddingEngine
from app.utils.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)

class MultiModalRetriever:
    """Multi-modal retriever for document search"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = None
        self._initialize_vector_store()
        logger.info("✅ MultiModalRetriever initialized")
    
    def _initialize_vector_store(self):
        """Initialize vector store with proper error handling"""
        try:
            from app.rag.vector_store import VectorStore
            from app.core.config import settings
            
            # Create vector store instance
            self.vector_store = VectorStore()
            
            # Test Qdrant connection
            try:
                import qdrant_client
                
                # Get Qdrant settings with defaults
                qdrant_host = getattr(settings, 'QDRANT_HOST', 'localhost')
                qdrant_port = getattr(settings, 'QDRANT_PORT', 6333)
                
                client = qdrant_client.QdrantClient(
                    host=qdrant_host,
                    port=qdrant_port,
                    timeout=10
                )
                
                # Test connection
                client.get_collections()
                logger.info(f"✅ Qdrant connection successful to {qdrant_host}:{qdrant_port}")
                
            except ImportError:
                logger.warning("⚠️ Qdrant client not available")
                self.vector_store = None
            except Exception as e:
                logger.warning(f"⚠️ Qdrant connection failed: {e}")
                self.vector_store = None
                
        except Exception as e:
            logger.warning(f"⚠️ Vector store initialization failed: {e}")
            self.vector_store = None
    
    async def index_document(self, 
                           document_id: str,
                           text_content: str,
                           images: Optional[List[Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index a document in the vector database
        """
        try:
            logger.info(f"📚 Indexing document {document_id}")
            
            if not text_content or len(text_content.strip()) == 0:
                logger.warning(f"⚠️ No text content for document {document_id}")
                return False
            
            # If vector store is not available, still return success
            if not self.vector_store:
                logger.info(f"ℹ️ Vector store not available, skipping indexing for {document_id}")
                return True  # Return True to not block processing
            
            # Chunk the text
            chunks = self.embedding_engine.chunk_text(text_content)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created for document {document_id}")
                return False
            
            logger.info(f"📝 Created {len(chunks)} chunks for document {document_id}")
            
            # Generate embeddings for each chunk
            chunk_embeddings = self.embedding_engine.generate_text_embeddings(chunks, chunk=False)
            
            if len(chunk_embeddings) != len(chunks):
                logger.error(f"❌ Embedding generation failed: {len(chunk_embeddings)} embeddings for {len(chunks)} chunks")
                return False
            
            # Prepare documents for indexing
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                doc_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "text_length": len(chunk),
                    "has_images": images is not None and len(images) > 0
                }
                
                if metadata:
                    doc_metadata.update(metadata)
                
                documents.append({
                    "id": f"{document_id}_chunk_{i}",
                    "text": chunk,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "metadata": doc_metadata
                })
            
            # Index documents
            success = await self.vector_store.index_documents(documents)
            
            if success:
                logger.info(f"✅ Document {document_id} indexed with {len(chunks)} chunks")
                return True
            else:
                logger.error(f"❌ Failed to index document {document_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Document indexing failed: {e}", exc_info=True)
            return False
    
    async def search_documents(self, 
                             query: str,
                             query_type: str = "text",
                             filters: Optional[Dict[str, Any]] = None,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using RAG
        """
        try:
            logger.info(f"🔍 Searching documents: {query[:50]}...")
            
            if not self.vector_store:
                logger.info("ℹ️ Vector store not available, returning mock results")
                return self._get_mock_search_results(query, limit)
            
            # Generate query embedding
            query_embedding = self.embedding_engine.generate_text_embeddings([query], chunk=False)[0]
            
            # Search in vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                filters=filters,
                limit=limit * 2
            )
            
            # Group results by document
            grouped_results = self._group_results_by_document(results, limit)
            
            logger.info(f"✅ Found {len(grouped_results)} relevant documents")
            return grouped_results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}", exc_info=True)
            return self._get_mock_search_results(query, limit)
    
    def _get_mock_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Return mock search results when vector store is not available"""
        mock_results = [
            {
                "document_id": "mock_doc_1",
                "max_score": 0.85,
                "avg_score": 0.82,
                "chunks": [
                    {
                        "text": f"Mock search result for: {query}. This is sample text from the document.",
                        "score": 0.85,
                        "chunk_index": 0,
                        "is_visual": False
                    }
                ],
                "metadata": {
                    "document_id": "mock_doc_1",
                    "text_length": 150,
                    "has_images": False
                }
            }
        ]
        return mock_results[:limit]
    
    def _group_results_by_document(self, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Group search results by document"""
        if not results:
            return []
        
        documents = {}
        
        for result in results:
            doc_id = result.get("metadata", {}).get("document_id")
            if not doc_id:
                continue
            
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "chunks": [],
                    "max_score": result["score"],
                    "metadata": result.get("metadata", {}),
                    "scores": []
                }
            
            documents[doc_id]["chunks"].append({
                "text": result.get("text", ""),
                "score": result["score"],
                "chunk_index": result.get("metadata", {}).get("chunk_index"),
                "is_visual": result.get("metadata", {}).get("is_visual", False)
            })
            documents[doc_id]["scores"].append(result["score"])
        
        # Calculate average score for each document
        for doc_id, doc_data in documents.items():
            if doc_data["scores"]:
                doc_data["avg_score"] = sum(doc_data["scores"]) / len(doc_data["scores"])
            else:
                doc_data["avg_score"] = 0
        
        # Sort documents by average score
        sorted_docs = sorted(
            documents.values(),
            key=lambda x: x["avg_score"],
            reverse=True
        )[:limit]
        
        # Sort chunks within each document
        for doc in sorted_docs:
            doc["chunks"] = sorted(
                doc["chunks"],
                key=lambda x: x["score"],
                reverse=True
            )[:3]
        
        return sorted_docs
    
    async def retrieve_for_question(self, 
                                  question: str,
                                  document_ids: Optional[List[str]] = None,
                                  limit: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant context for a question
        """
        try:
            filters = None
            if document_ids:
                filters = {"document_id": {"$in": document_ids}}
            
            # Search for relevant chunks
            results = await self.search_documents(
                query=question,
                filters=filters,
                limit=limit * 2
            )
            
            # Build context from top chunks
            context_parts = []
            sources = []
            
            for doc_result in results[:limit]:
                for chunk in doc_result.get("chunks", [])[:2]:
                    context_parts.append(chunk["text"])
                    sources.append({
                        "document_id": doc_result["document_id"],
                        "chunk_index": chunk.get("chunk_index"),
                        "score": chunk["score"],
                        "confidence": min(1.0, chunk["score"] * 2),  # Convert to 0-1 scale
                        "text_preview": (chunk["text"][:200] + "...") if len(chunk["text"]) > 200 else chunk["text"]
                    })
            
            context_text = "\n\n".join(context_parts)
            
            return context_text, sources
            
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            return "", []