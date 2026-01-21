"""
Multi-modal vector index for efficient similarity search across text, visual, and fused embeddings.
Uses FAISS for production-ready vector search.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import pickle
from pathlib import Path
import uuid
from datetime import datetime

# Import logger at the top
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import FAISS
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("✅ FAISS imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ FAISS not available: {e}. RAG will work with reduced performance.")

from app.core.models import MultiModalDocument

class MultiModalIndexer:
    """Index and search across multi-modal embeddings"""
    
    def __init__(self, index_dir: str = "rag_indices"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        self.indices = {}  # modality -> index data
        self.metadata = {}  # modality -> list of metadata
        self.documents = {}  # document_id -> document metadata
        self.embeddings = {}  # modality -> list of embeddings (for fallback)
        
        # Configuration
        self.text_dim = 384  # all-MiniLM-L6-v2 dimension
        self.visual_dim = 512  # CLIP dimension
        self.fused_dim = 896  # text + visual concatenated
        
        logger.info(f"📁 MultiModalIndexer initialized with index_dir: {self.index_dir}")
        logger.info(f"📊 FAISS available: {FAISS_AVAILABLE}")
    
    def create_index(self, modality: str, dimension: int):
        """Create index for a modality"""
        try:
            if FAISS_AVAILABLE:
                # Create FAISS index
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                index = faiss.IndexIDMap(index)
                self.indices[modality] = index
            else:
                # Create simple in-memory storage
                self.indices[modality] = {
                    "dimension": dimension,
                    "vectors": [],
                    "ids": []
                }
            
            self.metadata[modality] = []
            self.embeddings[modality] = []
            logger.info(f"✅ Created {modality} index with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to create {modality} index: {e}")
            self.indices[modality] = None
            self.metadata[modality] = []
            self.embeddings[modality] = []
    
    def index_document(self, document: MultiModalDocument, embeddings: Dict[str, Any]):
        """Index all embeddings from a document"""
        doc_id = document.document_id
        
        # Store document metadata
        self.documents[doc_id] = {
            "document_id": doc_id,
            "document_type": document.document_type.value if document.document_type else "unknown",
            "indexed_at": datetime.now().isoformat(),
            "risk_score": document.risk_score if hasattr(document, 'risk_score') else 0.0,
            "contradictions": len(document.contradictions) if hasattr(document, 'contradictions') else 0
        }
        
        # Index text chunks
        if embeddings.get("text_chunks"):
            if "text" not in self.indices:
                self.create_index("text", self.text_dim)
            
            for chunk in embeddings["text_chunks"]:
                self._add_to_index(
                    modality="text",
                    embedding=np.array(chunk["embedding"], dtype=np.float32),
                    metadata={
                        **chunk["metadata"],
                        "document_id": doc_id,
                        "chunk_id": chunk["id"],
                        "content": chunk["content"],
                        "modality": "text"
                    }
                )
        
        # Index visual regions
        if embeddings.get("visual_regions"):
            if "visual" not in self.indices:
                self.create_index("visual", self.visual_dim)
            
            for region in embeddings["visual_regions"]:
                self._add_to_index(
                    modality="visual",
                    embedding=np.array(region["embedding"], dtype=np.float32),
                    metadata={
                        **region.get("metadata", {}),
                        "document_id": doc_id,
                        "region_id": region["id"],
                        "element_type": region["element_type"],
                        "bbox": region["bbox"],
                        "page": region["page"],
                        "modality": "visual"
                    }
                )
        
        # Index fused elements
        if embeddings.get("fused_elements"):
            if "fused" not in self.indices:
                self.create_index("fused", self.fused_dim)
            
            for element in embeddings["fused_elements"]:
                self._add_to_index(
                    modality="fused",
                    embedding=np.array(element["embedding"], dtype=np.float32),
                    metadata={
                        **element.get("metadata", {}),
                        "document_id": doc_id,
                        "element_id": element["id"],
                        "text_content": element.get("text_content", ""),
                        "alignment_confidence": element.get("alignment_confidence", 0),
                        "modality": "fused"
                    }
                )
        
        logger.info(f"✅ Indexed document {doc_id}: "
                   f"{len(embeddings.get('text_chunks', []))} text, "
                   f"{len(embeddings.get('visual_regions', []))} visual, "
                   f"{len(embeddings.get('fused_elements', []))} fused")
    
    def _add_to_index(self, modality: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add single embedding to index"""
        if modality not in self.indices or self.indices[modality] is None:
            return
        
        # Generate unique ID for this embedding
        idx = len(self.metadata[modality])
        
        if FAISS_AVAILABLE:
            # Add to FAISS index
            embedding = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(embedding)  # Normalize for cosine similarity
            self.indices[modality].add_with_ids(embedding, np.array([idx]))
        else:
            # Add to simple index
            embedding_norm = embedding / np.linalg.norm(embedding)
            self.indices[modality]["vectors"].append(embedding_norm)
            self.indices[modality]["ids"].append(idx)
        
        # Store metadata and embedding for fallback search
        self.metadata[modality].append(metadata)
        self.embeddings[modality].append(embedding.flatten())
    
    def _faiss_search(self, index, query_embedding: np.ndarray, k: int):
        """Search using FAISS"""
        distances, indices = index.search(query_embedding, k)
        return distances[0], indices[0]
    
    def _simple_search(self, index_data: Dict, query_embedding: np.ndarray, k: int):
        """Simple cosine similarity search for when FAISS is not available"""
        vectors = index_data["vectors"]
        ids = index_data["ids"]
        
        if not vectors:
            return np.array([]), np.array([])
        
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate cosine similarities
        similarities = []
        for vec in vectors:
            sim = np.dot(query_norm, vec)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get top k
        if len(similarities) <= k:
            top_indices = np.arange(len(similarities))
            top_scores = similarities
        else:
            # Get indices of top k similarities
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_scores = similarities[top_indices]
            
            # Sort descending
            sorted_order = np.argsort(top_scores)[::-1]
            top_indices = top_indices[sorted_order]
            top_scores = top_scores[sorted_order]
        
        return top_scores, ids[top_indices] if len(ids) > 0 else top_indices
    
    def search(self, 
               query_embedding: np.ndarray, 
               modality: str = "text",
               k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        if modality not in self.indices or self.indices[modality] is None:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize query
        if FAISS_AVAILABLE:
            faiss.normalize_L2(query_embedding)
            distances, indices = self._faiss_search(self.indices[modality], query_embedding, k * 2)
        else:
            distances, indices = self._simple_search(self.indices[modality], query_embedding.flatten(), k * 2)
        
        results = []
        seen_docs = set()
        
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx < 0 or idx >= len(self.metadata[modality]):
                continue
            
            metadata = self.metadata[modality][idx]
            doc_id = metadata.get("document_id")
            
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if key in metadata and metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Deduplicate by document (optional)
            if doc_id in seen_docs:
                continue
            
            # Convert distance to similarity score (0-1)
            similarity = float((distance + 1) / 2)  # Cosine similarity ranges [-1, 1]
            
            results.append({
                "score": similarity,
                "metadata": metadata,
                "distance": float(distance),
                "rank": len(results) + 1
            })
            
            seen_docs.add(doc_id)
            
            if len(results) >= k:
                break
        
        return results
    
    def hybrid_search(self,
                     text_query: Optional[str] = None,
                     text_embedding: Optional[np.ndarray] = None,
                     visual_embedding: Optional[np.ndarray] = None,
                     k: int = 5,
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Perform hybrid search across all modalities"""
        from .multimodal_embedder import MultiModalEmbedder
        
        embedder = MultiModalEmbedder()
        embedder.initialize()
        
        results = {}
        
        # Text search
        if text_query:
            query_embedding = embedder.embed_text(text_query)
            results["text"] = self.search(query_embedding, "text", k, filters)
        
        # Visual search
        if visual_embedding is not None:
            results["visual"] = self.search(visual_embedding, "visual", k, filters)
        
        # Fused search (if both text and visual)
        if text_query and visual_embedding is not None:
            fused_embedding = embedder.embed_fused(
                embedder.embed_text(text_query) if text_query else None,
                visual_embedding
            )
            results["fused"] = self.search(fused_embedding, "fused", k, filters)
        
        return results
    
    def save_index(self, name: str = "default"):
        """Save index to disk"""
        save_path = self.index_dir / name
        save_path.mkdir(exist_ok=True, parents=True)
        
        try:
            # Save FAISS indices if available
            if FAISS_AVAILABLE:
                for modality, index in self.indices.items():
                    if index is not None and modality in ["text", "visual", "fused"]:
                        faiss.write_index(index, str(save_path / f"{modality}.faiss"))
            
            # Save all data
            save_data = {
                "metadata": self.metadata,
                "documents": self.documents,
                "embeddings": self.embeddings,
                "indices_info": {
                    modality: {
                        "dimension": self.indices[modality]["dimension"] if not FAISS_AVAILABLE else 
                                    self.indices[modality].d if hasattr(self.indices[modality], 'd') else 0,
                        "size": len(self.metadata.get(modality, []))
                    }
                    for modality in ["text", "visual", "fused"]
                    if modality in self.indices
                }
            }
            
            with open(save_path / "index_data.pkl", "wb") as f:
                pickle.dump(save_data, f)
            
            # Save config
            config = {
                "text_dim": self.text_dim,
                "visual_dim": self.visual_dim,
                "fused_dim": self.fused_dim,
                "faiss_available": FAISS_AVAILABLE,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(save_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 Saved index '{name}' to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index '{name}': {e}")
    
    def load_index(self, name: str = "default"):
        """Load index from disk"""
        load_path = self.index_dir / name
        
        if not load_path.exists():
            logger.warning(f"Index '{name}' not found at {load_path}")
            return False
        
        try:
            # Load config first
            config_path = load_path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.text_dim = config.get("text_dim", self.text_dim)
                    self.visual_dim = config.get("visual_dim", self.visual_dim)
                    self.fused_dim = config.get("fused_dim", self.fused_dim)
            
            # Load FAISS indices if available
            if FAISS_AVAILABLE:
                for modality in ["text", "visual", "fused"]:
                    index_path = load_path / f"{modality}.faiss"
                    if index_path.exists():
                        self.indices[modality] = faiss.read_index(str(index_path))
            
            # Load all data
            data_path = load_path / "index_data.pkl"
            if data_path.exists():
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
                    self.metadata = data.get("metadata", {})
                    self.documents = data.get("documents", {})
                    self.embeddings = data.get("embeddings", {})
                    
                    # Recreate simple indices if FAISS not available
                    if not FAISS_AVAILABLE:
                        for modality in ["text", "visual", "fused"]:
                            if modality in self.metadata:
                                dim = data.get("indices_info", {}).get(modality, {}).get("dimension", 
                                    self.text_dim if modality == "text" else 
                                    self.visual_dim if modality == "visual" else 
                                    self.fused_dim)
                                self.indices[modality] = {
                                    "dimension": dim,
                                    "vectors": [],
                                    "ids": []
                                }
                                # Recreate vectors from embeddings
                                for idx, embedding in enumerate(self.embeddings.get(modality, [])):
                                    if idx < len(self.metadata[modality]):
                                        vec = embedding / np.linalg.norm(embedding)
                                        self.indices[modality]["vectors"].append(vec)
                                        self.indices[modality]["ids"].append(idx)
            
            logger.info(f"📂 Loaded index '{name}' from {load_path}")
            logger.info(f"   Documents: {len(self.documents)}")
            logger.info(f"   Text chunks: {len(self.metadata.get('text', []))}")
            logger.info(f"   Visual regions: {len(self.metadata.get('visual', []))}")
            logger.info(f"   Fused elements: {len(self.metadata.get('fused', []))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index '{name}': {e}")
            return False
    
    def get_document_stats(self, document_id: str) -> Dict[str, Any]:
        """Get statistics for a document in the index"""
        if document_id not in self.documents:
            return {}
        
        stats = {
            "document_info": self.documents[document_id],
            "indexed_items": {
                "text": 0,
                "visual": 0,
                "fused": 0
            }
        }
        
        # Count items per modality
        for modality in ["text", "visual", "fused"]:
            if modality in self.metadata:
                stats["indexed_items"][modality] = sum(
                    1 for meta in self.metadata[modality]
                    if meta.get("document_id") == document_id
                )
        
        return stats
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the index"""
        return list(self.documents.values())