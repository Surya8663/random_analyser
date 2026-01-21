"""
Multi-modal retriever that integrates with the existing agent pipeline.
Retrieves relevant text, visual, and fused content based on queries.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime

from app.core.models import MultiModalDocument
from app.utils.logger import setup_logger
from .multimodal_embedder import MultiModalEmbedder
from .indexer import MultiModalIndexer

logger = setup_logger(__name__)

class MultiModalRetriever:
    """Retriever for multi-modal document queries"""
    
    def __init__(self, index_name: str = "default"):
        self.embedder = MultiModalEmbedder()
        self.indexer = MultiModalIndexer()
        self.index_name = index_name
        
        # Initialize components
        self.embedder.initialize()
        self.indexer.load_index(index_name)
        
        # Query cache for performance
        self.query_cache = {}
        
    def index_document(self, document: MultiModalDocument) -> bool:
        """Index a processed document for retrieval"""
        try:
            # Extract embeddings from document
            embeddings = self.embedder.extract_document_embeddings(document)
            
            # Add to index
            self.indexer.index_document(document, embeddings)
            
            # Save index
            self.indexer.save_index(self.index_name)
            
            logger.info(f"✅ Indexed document {document.document_id} for retrieval")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document {document.document_id}: {e}")
            return False
    
    def retrieve(self,
                query: str,
                modality: str = "hybrid",
                k: int = 5,
                filters: Optional[Dict[str, Any]] = None,
                include_content: bool = True) -> Dict[str, Any]:
        """
        Retrieve relevant content based on query
        
        Args:
            query: Text query string
            modality: "text", "visual", "fused", or "hybrid"
            k: Number of results per modality
            filters: Optional filters (document_type, risk_level, etc.)
            include_content: Whether to include full content in results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            
            results = {
                "query": query,
                "modality": modality,
                "timestamp": datetime.now().isoformat(),
                "text_results": [],
                "visual_results": [],
                "fused_results": [],
                "document_hits": set()
            }
            
            # Perform search based on modality
            if modality in ["text", "hybrid"]:
                text_results = self.indexer.search(query_embedding, "text", k, filters)
                results["text_results"] = self._format_text_results(text_results, include_content)
                results["document_hits"].update(r["document_id"] for r in text_results)
            
            if modality in ["visual", "hybrid"]:
                # For visual search with text query, we need to handle differently
                # Could use text-to-visual cross-modal search or just use fused
                visual_results = self.indexer.search(query_embedding, "visual", k, filters)
                results["visual_results"] = self._format_visual_results(visual_results, include_content)
                results["document_hits"].update(r["document_id"] for r in visual_results)
            
            if modality in ["fused", "hybrid"]:
                fused_results = self.indexer.search(query_embedding, "fused", k, filters)
                results["fused_results"] = self._format_fused_results(fused_results, include_content)
                results["document_hits"].update(r["document_id"] for r in fused_results)
            
            # Convert document_hits to list
            results["document_hits"] = list(results["document_hits"])
            results["total_hits"] = len(results["document_hits"])
            
            # Add aggregated scores
            all_results = results["text_results"] + results["visual_results"] + results["fused_results"]
            if all_results:
                results["average_score"] = sum(r["score"] for r in all_results) / len(all_results)
                results["max_score"] = max(r["score"] for r in all_results)
            else:
                results["average_score"] = 0
                results["max_score"] = 0
            
            logger.info(f"🔍 Retrieved {results['total_hits']} documents for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "text_results": [],
                "visual_results": [],
                "fused_results": [],
                "document_hits": [],
                "total_hits": 0
            }
    
    def _format_text_results(self, results: List[Dict], include_content: bool) -> List[Dict]:
        """Format text search results"""
        formatted = []
        for result in results:
            metadata = result.get("metadata", {})
            formatted.append({
                "document_id": metadata.get("document_id"),
                "score": result.get("score", 0),
                "content": metadata.get("content", "") if include_content else "",
                "page": metadata.get("page", 0),
                "type": metadata.get("type", "text"),
                "agent": metadata.get("agent", "text_agent"),
                "modality": "text",
                "metadata": {
                    k: v for k, v in metadata.items()
                    if k not in ["document_id", "content", "page", "type", "agent"]
                }
            })
        return formatted
    
    def _format_visual_results(self, results: List[Dict], include_content: bool) -> List[Dict]:
        """Format visual search results"""
        formatted = []
        for result in results:
            metadata = result.get("metadata", {})
            formatted.append({
                "document_id": metadata.get("document_id"),
                "score": result.get("score", 0),
                "element_type": metadata.get("element_type", ""),
                "bbox": metadata.get("bbox", []),
                "page": metadata.get("page", 0),
                "confidence": metadata.get("confidence", 0),
                "agent": metadata.get("agent", "vision_agent"),
                "modality": "visual",
                "metadata": {
                    k: v for k, v in metadata.items()
                    if k not in ["document_id", "element_type", "bbox", "page", "confidence", "agent"]
                }
            })
        return formatted
    
    def _format_fused_results(self, results: List[Dict], include_content: bool) -> List[Dict]:
        """Format fused search results"""
        formatted = []
        for result in results:
            metadata = result.get("metadata", {})
            formatted.append({
                "document_id": metadata.get("document_id"),
                "score": result.get("score", 0),
                "text_content": metadata.get("text_content", "") if include_content else "",
                "alignment_confidence": metadata.get("alignment_confidence", 0),
                "page": metadata.get("page", 0),
                "agent": metadata.get("agent", "fusion_agent"),
                "modality": "fused",
                "metadata": {
                    k: v for k, v in metadata.items()
                    if k not in ["document_id", "text_content", "alignment_confidence", "page", "agent"]
                }
            })
        return formatted
    
    def semantic_search(self,
                       query: str,
                       document_id: Optional[str] = None,
                       k: int = 10) -> Dict[str, Any]:
        """
        Semantic search within a specific document or across all documents
        
        This is particularly useful for questions like:
        - "What does the table on page 2 say about revenue?"
        - "Show all high-risk signatures in this document"
        """
        filters = {}
        if document_id:
            filters["document_id"] = document_id
        
        # First, try to understand the query intent
        query_lower = query.lower()
        
        # Check for page-specific queries
        page_num = None
        if "page" in query_lower:
            import re
            page_match = re.search(r'page\s*(\d+)', query_lower)
            if page_match:
                page_num = int(page_match.group(1)) - 1  # Convert to 0-indexed
                filters["page"] = page_num
        
        # Check for element-specific queries
        element_type = None
        element_keywords = {
            "table": ["table", "chart", "graph", "figure"],
            "signature": ["signature", "sign", "signed"],
            "logo": ["logo", "brand", "emblem"],
            "header": ["header", "title", "heading"],
            "footer": ["footer", "bottom"]
        }
        
        for elem_type, keywords in element_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                element_type = elem_type
                filters["element_type"] = elem_type
                break
        
        # Check for risk-related queries
        if any(word in query_lower for word in ["risk", "contradiction", "error", "issue", "problem"]):
            filters["risk_level"] = "high"
        
        # Perform retrieval
        results = self.retrieve(
            query=query,
            modality="hybrid",
            k=k,
            filters=filters if filters else None,
            include_content=True
        )
        
        # Add query analysis
        results["query_analysis"] = {
            "original_query": query,
            "page_filter": page_num,
            "element_filter": element_type,
            "risk_filter": "risk" in query_lower or "contradiction" in query_lower
        }
        
        return results
    
    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get retrieval summary for a document"""
        stats = self.indexer.get_document_stats(document_id)
        
        if not stats:
            return {"error": "Document not found in index"}
        
        # Get top queries that would match this document
        # (simplified - in production you'd track actual queries)
        summary = {
            "document_id": document_id,
            "indexed_at": stats["document_info"].get("indexed_at"),
            "indexed_items": stats["indexed_items"],
            "retrieval_ready": sum(stats["indexed_items"].values()) > 0,
            "suggested_queries": self._generate_suggested_queries(stats)
        }
        
        return summary
    
    def _generate_suggested_queries(self, stats: Dict[str, Any]) -> List[str]:
        """Generate suggested queries based on document content"""
        suggestions = []
        doc_info = stats["document_info"]
        
        # Basic suggestions based on document type
        doc_type = doc_info.get("document_type", "").lower()
        
        if doc_type == "invoice":
            suggestions.extend([
                "What is the total amount?",
                "What is the invoice number?",
                "When was this invoice issued?",
                "Who is the vendor?",
                "Are there any signatures?"
            ])
        elif doc_type == "contract":
            suggestions.extend([
                "What are the key terms?",
                "Who are the parties involved?",
                "What is the effective date?",
                "Are there any signatures?",
                "What are the termination clauses?"
            ])
        
        # Add risk-related suggestions if applicable
        risk_score = doc_info.get("risk_score", 0)
        if risk_score > 0.5:
            suggestions.append("What are the risk factors in this document?")
        
        contradictions = doc_info.get("contradictions", 0)
        if contradictions > 0:
            suggestions.append("What contradictions were found?")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def save_state(self):
        """Save retriever state"""
        self.indexer.save_index(self.index_name)
        logger.info(f"💾 Saved retriever state for index '{self.index_name}'")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        docs = self.indexer.list_documents()
        
        total_text = sum(
            1 for meta in self.indexer.metadata.get("text", [])
        ) if "text" in self.indexer.metadata else 0
        
        total_visual = sum(
            1 for meta in self.indexer.metadata.get("visual", [])
        ) if "visual" in self.indexer.metadata else 0
        
        total_fused = sum(
            1 for meta in self.indexer.metadata.get("fused", [])
        ) if "fused" in self.indexer.metadata else 0
        
        return {
            "total_documents": len(docs),
            "total_items": total_text + total_visual + total_fused,
            "text_items": total_text,
            "visual_items": total_visual,
            "fused_items": total_fused,
            "document_types": {
                doc["document_type"]: sum(1 for d in docs if d["document_type"] == doc["document_type"])
                for doc in docs
            },
            "index_name": self.index_name,
            "last_updated": datetime.now().isoformat()
        }