import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams, 
    Distance, 
    PointStruct,
    Filter, 
    FieldCondition, 
    MatchValue
)

from app.core.config import settings

class VectorStore:
    """Vector store for document embeddings using Qdrant."""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=10  # Added timeout
        )
        self.collection_name = settings.QDRANT_COLLECTION
        
        # Don't try to verify collection on init - let it fail gracefully
        # We'll handle collection creation during upsert
        self._collection_initialized = False
        
        logging.info(f"✅ VectorStore initialized for collection: {self.collection_name}")
    
    def _ensure_collection(self, vector_size: int = 384):
        """Ensure the collection exists with correct configuration."""
        try:
            # Try to get collections first
            try:
                collections = self.client.get_collections()
                collection_names = [c.name for c in collections.collections]
            except Exception as e:
                logging.warning(f"Could not get collections: {str(e)}")
                collection_names = []
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logging.info(f"Created collection: {self.collection_name}")
                self._collection_initialized = True
            else:
                logging.info(f"Collection exists: {self.collection_name}")
                self._collection_initialized = True
                
        except Exception as e:
            logging.error(f"Error ensuring collection: {str(e)}")
            # Don't raise, we'll try to create on upsert
    
    def upsert(self, points: List[Dict[str, Any]]):
        """Upsert points into the vector store."""
        try:
            # Ensure collection exists before upserting
            if not self._collection_initialized:
                self._ensure_collection()
            
            point_structs = []
            for point in points:
                point_id = point.get("id")
                vector = point.get("vector", [])
                payload = point.get("payload", {})
                
                if point_id and vector:
                    point_structs.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload
                        )
                    )
            
            if point_structs:
                # Try to upsert, if collection doesn't exist, create it first
                try:
                    operation_info = self.client.upsert(
                        collection_name=self.collection_name,
                        points=point_structs,
                        wait=True
                    )
                    logging.info(f"Upserted {len(point_structs)} points. Status: {operation_info.status}")
                except Exception as upsert_error:
                    # Collection might not exist, create it
                    if "not found" in str(upsert_error).lower():
                        logging.info(f"Collection not found, creating it...")
                        self._ensure_collection(len(vector) if vector else 384)
                        # Retry upsert
                        operation_info = self.client.upsert(
                            collection_name=self.collection_name,
                            points=point_structs,
                            wait=True
                        )
                        logging.info(f"Upserted {len(point_structs)} points after creating collection.")
                    else:
                        raise upsert_error
                
        except Exception as e:
            logging.error(f"Error upserting points: {str(e)}")
            # Don't crash, just log the error
    
    def search(self, 
               query_vector: List[float],
               limit: int = 5,
               score_threshold: float = 0.3,
               query_filter: Optional[Dict] = None) -> List[Any]:
        """Search for similar vectors."""
        try:
            # Ensure collection exists
            if not self._collection_initialized:
                self._ensure_collection()
            
            # Convert filter dict to Qdrant Filter
            qdrant_filter = None
            if query_filter:
                must_conditions = []
                for condition in query_filter.get("must", []):
                    if condition.get("key") == "document_id":
                        must_conditions.append(
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=condition.get("match", {}).get("value"))
                            )
                        )
                
                if must_conditions:
                    qdrant_filter = Filter(must=must_conditions)
            
            try:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=limit,
                    score_threshold=score_threshold
                )
                
                return search_results
            except Exception as search_error:
                if "not found" in str(search_error).lower():
                    logging.warning(f"Collection {self.collection_name} not found for search")
                    return []
                else:
                    raise search_error
                
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []
    
    def delete_by_document(self, document_id: str):
        """Delete all points for a specific document."""
        try:
            # Ensure collection exists
            if not self._collection_initialized:
                self._ensure_collection()
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logging.info(f"Deleted vectors for document: {document_id}")
            
        except Exception as e:
            logging.error(f"Error deleting document vectors: {str(e)}")
    
    def count_documents(self) -> int:
        """Count total points in the collection."""
        try:
            # Ensure collection exists
            if not self._collection_initialized:
                self._ensure_collection()
            
            try:
                count_result = self.client.count(
                    collection_name=self.collection_name
                )
                return count_result.count
            except Exception as count_error:
                if "not found" in str(count_error).lower():
                    return 0
                else:
                    raise count_error
        except Exception as e:
            logging.error(f"Error counting documents: {str(e)}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            # Ensure collection exists
            if not self._collection_initialized:
                self._ensure_collection()
            
            try:
                info = self.client.get_collection(self.collection_name)
                return {
                    "name": self.collection_name,
                    "vector_size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                    "points_count": info.points_count
                }
            except Exception as info_error:
                if "not found" in str(info_error).lower():
                    return {"name": self.collection_name, "status": "not_found"}
                else:
                    # Handle version compatibility issues
                    logging.warning(f"Could not get detailed collection info: {str(info_error)}")
                    return {
                        "name": self.collection_name,
                        "status": "unknown"
                    }
        except Exception as e:
            logging.error(f"Error getting collection info: {str(e)}")
            return {"name": self.collection_name, "error": str(e)}