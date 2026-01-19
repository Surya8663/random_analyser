from typing import List, Dict, Any, Optional, Tuple
from app.rag.embeddings import EmbeddingEngine
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class MultiModalRetriever:
    """Multi-modal retriever for document search"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        logger.info("✅ MultiModalRetriever initialized")
    
    async def initialize(self) -> bool:
        """Initialize the retriever"""
        try:
            logger.info("🔄 Initializing retriever")
            
            # Get settings safely
            try:
                from app.core.config import settings
                qdrant_host = getattr(settings, 'QDRANT_HOST', 'localhost')
                qdrant_port = getattr(settings, 'QDRANT_PORT', 6333)
                qdrant_collection = getattr(settings, 'QDRANT_COLLECTION', 'document_embeddings')
            except:
                qdrant_host = 'localhost'
                qdrant_port = 6333
                qdrant_collection = 'document_embeddings'
            
            # Check if Qdrant is available
            try:
                from qdrant_client import QdrantClient
                
                client = QdrantClient(
                    host=qdrant_host,
                    port=qdrant_port,
                    timeout=10
                )
                # Test connection
                client.get_collections()
                logger.info(f"✅ Qdrant connection successful to {qdrant_host}:{qdrant_port}")
                return True
            except ImportError:
                logger.warning("⚠️ Qdrant client not available")
                return False
            except Exception as e:
                logger.warning(f"⚠️ Qdrant connection failed: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Retriever initialization failed: {e}")
            return False
    
    # Rest of the class remains the same...
    # [Keep all other methods as they were]
    
    async def index_document(self, 
                           document_id: str,
                           text_content: str,
                           images: Optional[List[Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index a document in the vector database with proper chunking
        
        Args:
            document_id: Unique document identifier
            text_content: Text content of the document
            images: Optional list of document images
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            logger.info(f"📚 Indexing document {document_id}")
            
            if not text_content:
                logger.warning(f"⚠️ No text content for document {document_id}")
                return False
            
            # Chunk the text
            chunks = self.embedding_engine.chunk_text(text_content)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created for document {document_id}")
                return False
            
            logger.info(f"📝 Created {len(chunks)} chunks for document {document_id}")
            
            # Generate embeddings for each chunk
            chunk_embeddings = self.embedding_engine.generate_text_embeddings(chunks, chunk=False)
            
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
                    "embedding": embedding,
                    "metadata": doc_metadata
                })
            
            # Index documents in vector store
            success = await self.vector_store.index_documents(documents)
            
            if success:
                logger.info(f"✅ Document {document_id} indexed with {len(chunks)} chunks")
                
                # Index images if available
                if images and len(images) > 0:
                    await self._index_images(document_id, images, metadata)
            else:
                logger.error(f"❌ Failed to index document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Document indexing failed: {e}", exc_info=True)
            return False
    
    async def _index_images(self, document_id: str, images: List[Any], metadata: Optional[Dict[str, Any]] = None):
        """Index document images"""
        try:
            logger.info(f"🖼️ Indexing images for document {document_id}")
            
            # Generate visual embeddings
            visual_embeddings = self.embedding_engine.generate_visual_embeddings(images[:3])  # Limit to 3 images
            
            # Prepare image documents
            image_docs = []
            for i, (img, embedding) in enumerate(zip(images[:3], visual_embeddings)):
                img_metadata = {
                    "document_id": document_id,
                    "image_index": i,
                    "is_visual": True,
                    "image_count": len(images)
                }
                
                if metadata:
                    img_metadata.update(metadata)
                
                image_docs.append({
                    "id": f"{document_id}_image_{i}",
                    "text": f"Image {i+1} from document {document_id}",
                    "embedding": embedding,
                    "metadata": img_metadata
                })
            
            # Index images
            await self.vector_store.index_documents(image_docs)
            logger.info(f"✅ Indexed {len(image_docs)} images for document {document_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Image indexing failed: {e}")
    
    async def search_documents(self, 
                             query: str,
                             query_type: str = "text",
                             filters: Optional[Dict[str, Any]] = None,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using RAG
        
        Args:
            query: Search query
            query_type: Type of query ("text", "visual", "multi_modal")
            filters: Optional filters
            limit: Number of results
            
        Returns:
            List of matching documents with scores
        """
        try:
            logger.info(f"🔍 Searching documents: {query[:50]}...")
            
            # Generate query embedding
            if query_type == "text":
                query_embedding = self.embedding_engine.generate_text_embeddings([query], chunk=False)[0]
            else:
                # For other query types, use text embedding as fallback
                query_embedding = self.embedding_engine.generate_text_embeddings([query], chunk=False)[0]
            
            # Search in vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                filters=filters,
                limit=limit * 2  # Get more results for filtering
            )
            
            # Group results by document and select best chunks
            grouped_results = self._group_results_by_document(results, limit)
            
            logger.info(f"✅ Found {len(grouped_results)} relevant documents")
            return grouped_results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}", exc_info=True)
            return []
    
    def _group_results_by_document(self, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Group search results by document and select best chunks"""
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
                    "metadata": result.get("metadata", {})
                }
            
            documents[doc_id]["chunks"].append({
                "text": result.get("text", ""),
                "score": result["score"],
                "chunk_index": result.get("metadata", {}).get("chunk_index"),
                "is_visual": result.get("metadata", {}).get("is_visual", False)
            })
        
        # Sort documents by max score and limit
        sorted_docs = sorted(
            documents.values(),
            key=lambda x: x["max_score"],
            reverse=True
        )[:limit]
        
        # For each document, select top chunks
        for doc in sorted_docs:
            doc["chunks"] = sorted(
                doc["chunks"],
                key=lambda x: x["score"],
                reverse=True
            )[:3]  # Top 3 chunks per document
        
        return sorted_docs
    
    async def retrieve_for_question(self, 
                                  question: str,
                                  document_ids: Optional[List[str]] = None,
                                  limit: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant context for a question
        
        Args:
            question: Question to answer
            document_ids: Optional list of document IDs to search within
            limit: Number of chunks to retrieve
            
        Returns:
            Tuple of (context_text, sources)
        """
        try:
            filters = None
            if document_ids:
                filters = {"document_id": {"$in": document_ids}}
            
            # Search for relevant chunks
            results = await self.search_documents(
                query=question,
                filters=filters,
                limit=limit * 2  # Get more for better selection
            )
            
            # Build context from top chunks
            context_parts = []
            sources = []
            
            for doc_result in results[:limit]:  # Top documents
                for chunk in doc_result.get("chunks", [])[:2]:  # Top 2 chunks per doc
                    context_parts.append(chunk["text"])
                    sources.append({
                        "document_id": doc_result["document_id"],
                        "chunk_index": chunk.get("chunk_index"),
                        "score": chunk["score"],
                        "text_preview": chunk["text"][:200] + "..."
                    })
            
            context_text = "\n\n".join(context_parts)
            
            return context_text, sources
            
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            return "", []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from index"""
        try:
            success = await self.vector_store.delete_document(document_id)
            
            if success:
                logger.info(f"🗑️ Document {document_id} deleted from index")
            else:
                logger.warning(f"⚠️ Failed to delete document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Document deletion failed: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = await self.vector_store.get_stats()
            
            return {
                "vector_store": "Qdrant",
                "stats": stats,
                "embedding_dimension": self.embedding_engine.get_embedding_dimensions()["text"]
            }
        except Exception as e:
            logger.error(f"❌ Failed to get index stats: {e}")
            return {}