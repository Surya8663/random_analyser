from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """Wrapper for Qdrant vector database operations"""
    
    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        logger.info(f"✅ VectorStore initialized for collection: {self.collection_name}")
    
    async def initialize(self) -> bool:
        """Initialize connection and create collection if needed"""
        try:
            # Connect to Qdrant
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=30
            )
            
            logger.info(f"🔗 Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"📁 Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"✅ Collection {self.collection_name} created")
            else:
                logger.info(f"📁 Collection {self.collection_name} already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            return False
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents in the vector store
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            Success status
        """
        try:
            if not documents:
                logger.warning("⚠️ No documents to index")
                return False
            
            points = []
            for doc in documents:
                point_id = doc.get("id") or str(uuid.uuid4())
                embedding = doc.get("embedding")
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                if embedding is None:
                    logger.warning(f"⚠️ Skipping document {point_id}: No embedding")
                    continue
                
                # Ensure embedding is the right dimension
                if len(embedding) != self.embedding_dimension:
                    logger.warning(f"⚠️ Embedding dimension mismatch for {point_id}: "
                                 f"expected {self.embedding_dimension}, got {len(embedding)}")
                    # Try to truncate or pad if close
                    if len(embedding) > self.embedding_dimension:
                        embedding = embedding[:self.embedding_dimension]
                    else:
                        padding = np.zeros(self.embedding_dimension - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                
                # Prepare payload
                payload = {
                    "text": text[:1000],  # Limit text in payload
                    "full_text": text,
                    **metadata
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    payload=payload
                )
                points.append(point)
            
            if not points:
                logger.warning("⚠️ No valid points to index")
                return False
            
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
        
        Args:
            query_embedding: Query embedding vector
            filters: Optional filters
            limit: Number of results
            
        Returns:
            List of search results
        """
        try:
            # Prepare filter
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, dict):
                        # Handle special operators
                        if "$in" in value:
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=models.MatchAny(any=value["$in"])
                                )
                            )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Ensure embedding is list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=0.3  # Minimum relevance score
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
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Success status
        """
        try:
            # Find points for this document
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
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
            info = self.client.get_collection(self.collection_name)
            
            # Count documents (unique document_ids)
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=False
            )
            
            # Extract unique document IDs
            doc_ids = set()
            for point in scroll_result[0]:
                # Need to fetch payload for document_id
                point_info = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point.id],
                    with_payload=True
                )
                if point_info and point_info[0].payload:
                    doc_id = point_info[0].payload.get("document_id")
                    if doc_id:
                        doc_ids.add(doc_id)
            
            return {
                "collection_name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "total_points": info.points_count,
                "unique_documents": len(doc_ids),
                "segments_count": info.segments_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            # Try to get collection info
            info = self.client.get_collection(self.collection_name)
            
            return {
                "status": "healthy",
                "collection": self.collection_name,
                "points": info.points_count,
                "ready": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "ready": False
            }