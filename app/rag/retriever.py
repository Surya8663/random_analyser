import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.rag.embeddings import EmbeddingEngine
from app.core.config import settings

class MultiModalRetriever:
    """Retriever for document chunks from Qdrant."""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.embedding_engine = EmbeddingEngine()
        self.collection_name = settings.QDRANT_COLLECTION
        
        # Ensure collection exists
        self._ensure_collection()
        
        logging.info("✅ MultiModalRetriever initialized")
    
    def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_engine.get_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                logging.info(f"Created collection: {self.collection_name}")
            else:
                logging.info(f"Collection exists: {self.collection_name}")
                
        except Exception as e:
            logging.error(f"Error ensuring collection: {str(e)}")
            raise
    
    def retrieve(self, 
                query: str, 
                document_id: Optional[str] = None,
                top_k: int = 5,
                score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            document_id: Optional filter for specific document
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_engine.encode_text(query)
            
            # Build filter
            query_filter = None
            if document_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                payload = result.payload or {}
                results.append({
                    "id": str(result.id),
                    "text": payload.get("text", ""),
                    "score": float(result.score),
                    "metadata": {
                        "document_id": payload.get("document_id"),
                        "page": payload.get("page", 1),
                        "chunk_index": payload.get("chunk_index", 0)
                    }
                })
            
            logging.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logging.error(f"Retrieval error: {str(e)}")
            return []
    
    def retrieve_by_document(self, document_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document."""
        try:
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            for point in scroll_results[0]:
                payload = point.payload or {}
                chunks.append({
                    "id": str(point.id),
                    "text": payload.get("text", ""),
                    "metadata": {
                        "document_id": payload.get("document_id"),
                        "page": payload.get("page", 1),
                        "chunk_index": payload.get("chunk_index", 0)
                    }
                })
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error retrieving document chunks: {str(e)}")
            return []