from typing import List, Dict, Any, Optional, Tuple  # Added Tuple import
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
    
    async def index_document(self, 
                           document_id: str,
                           text_content: str,
                           images: Optional[List[Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index a document
        """
        try:
            logger.info(f"📚 Indexing document {document_id}")
            
            if not text_content:
                logger.warning(f"⚠️ No text content for document {document_id}")
                return False
            
            # Generate embeddings
            embeddings = self.embedding_engine.generate_text_embeddings(text_content)
            
            logger.info(f"✅ Generated embeddings for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document indexing failed: {e}")
            return False
    
    async def search_documents(self, 
                             query: str,
                             query_type: str = "text",
                             filters: Optional[Dict[str, Any]] = None,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents
        """
        try:
            logger.info(f"🔍 Searching documents: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_engine.generate_text_embeddings([query], chunk=False)[0]
            
            # For now, return mock results
            # In production, this would search in Qdrant
            
            mock_results = [
                {
                    "document_id": "doc_123",
                    "score": 0.85,
                    "text": f"Relevant content about {query} found in document.",
                    "metadata": {"source": "mock_search"}
                }
            ]
            
            return mock_results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []
    
    async def retrieve_for_question(self, 
                                  question: str,
                                  document_ids: Optional[List[str]] = None,
                                  limit: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant context for a question
        """
        try:
            # Search for relevant content
            results = await self.search_documents(
                query=question,
                limit=limit
            )
            
            # Build context text
            context_parts = []
            sources = []
            
            for result in results:
                context_parts.append(result.get("text", ""))
                sources.append({
                    "document_id": result.get("document_id", "unknown"),
                    "score": result.get("score", 0),
                    "text_preview": result.get("text", "")[:200]
                })
            
            context_text = "\n\n".join(context_parts)
            
            return context_text, sources
            
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            return "", []