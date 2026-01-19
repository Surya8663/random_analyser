# app/rag/vector_store.py
from typing import List, Dict, Any, Optional
import numpy as np
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """Wrapper for Qdrant vector database operations - FIXED"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client with proper error handling"""
        try:
            from app.core.config import settings
            
            self.collection_name = getattr(settings, 'QDRANT_COLLECTION', 'document_embeddings')
            self.embedding_dimension = getattr(settings, 'EMBEDDING_DIMENSION', 384)
            
            # Try to import and initialize Qdrant
            try:
                from qdrant_client import QdrantClient
                
                qdrant_host = getattr(settings, 'QDRANT_HOST', 'localhost')
                qdrant_port = getattr(settings, 'QDRANT_PORT', 6333)
                
                self.client = QdrantClient(
                    host=qdrant_host,
                    port=qdrant_port,
                    timeout=30
                )
                
                logger.info(f"✅ VectorStore initialized for collection: {self.collection_name}")
                
            except ImportError:
                logger.warning("⚠️ Qdrant client not available")
                self.client = None
                
        except Exception as e:
            logger.warning(f"⚠️ VectorStore initialization failed: {e}")
            self.client = None
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents in the vector store
        """
        try:
            if not documents:
                logger.warning("⚠️ No documents to index")
                return False
            
            if not self.client:
                logger.warning("⚠️ Qdrant client not available, skipping indexing")
                return True  # Return True to not block processing
            
            # Import Qdrant models
            from qdrant_client.http import models
            from qdrant_client.http.models import PointStruct
            
            points = []
            for doc in documents:
                point_id = doc.get("id")
                embedding = doc.get("embedding")
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                if embedding is None:
                    logger.warning(f"⚠️ Skipping document {point_id}: No embedding")
                    continue
                
                # Ensure embedding is list
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Prepare payload
                payload = {
                    "text": text[:1000],  # Limit text in payload
                    "full_text": text,
                    **metadata
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            if not points:
                logger.warning("⚠️ No valid points to index")
                return False
            
            # Check if collection exists, create if not
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"📁 Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE
                    )
                )
            
            # Upsert points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"✅ Indexed {len(points)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document indexing failed: {e}", exc_info=True)
            return False
    
    async def search(self, 
                    query_embedding: np.ndarray,
                    filters: Optional[Dict[str, Any]] = None,
                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        """
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant client not available, returning empty results")
                return []
            
            from qdrant_client.http import models
            
            # Prepare filter
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, dict):
                        # Handle special operators
                        if "$in" in value:
                            conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchAny(any=value["$in"])
                                )
                            )
                    else:
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    qdrant_filter = models.Filter(must=conditions)
            
            # Ensure embedding is list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=0.3
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("full_text") or hit.payload.get("text", ""),
                    "metadata": hit.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}", exc_info=True)
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all vectors for a document
        """
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant client not available")
                return True
            
            from qdrant_client.http import models
            
            # Find points for this document
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000
            )
            
            point_ids = [point.id for point in scroll_result[0]]
            
            if point_ids:
                # Delete points
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                logger.info(f"🗑️ Deleted {len(point_ids)} points for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.client:
                return {
                    "status": "qdrant_not_available",
                    "collection_name": self.collection_name,
                    "ready": False
                }
            
            info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "total_points": info.points_count,
                "segments_count": info.segments_count,
                "ready": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "ready": False
            }