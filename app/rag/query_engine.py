"""
Query engine that provides natural language answers based on retrieved content.
Integrates with the retriever to generate comprehensive answers.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from app.core.models import MultiModalDocument
from app.utils.logger import setup_logger
from .retriever import MultiModalRetriever

logger = setup_logger(__name__)

class QueryEngine:
    """Natural language query engine for multi-modal documents"""
    
    def __init__(self, retriever: Optional[MultiModalRetriever] = None):
        self.retriever = retriever or MultiModalRetriever()
        self.answer_cache = {}
        
    def query(self,
             query: str,
             document_id: Optional[str] = None,
             modality: str = "hybrid",
             include_sources: bool = True,
             max_results: int = 5) -> Dict[str, Any]:
        """
        Answer a natural language query about documents
        
        Args:
            query: Natural language question
            document_id: Optional specific document
            modality: Search modality
            include_sources: Whether to include source information
            max_results: Maximum number of results to consider
        """
        try:
            logger.info(f"🤖 Processing query: '{query}'")
            
            # First, retrieve relevant content
            retrieval_results = self.retriever.semantic_search(
                query=query,
                document_id=document_id,
                k=max_results * 2  # Get more results for better answer generation
            )
            
            if retrieval_results.get("total_hits", 0) == 0:
                return self._generate_no_results_response(query)
            
            # Generate answer based on retrieved content
            answer = self._generate_answer(query, retrieval_results)
            
            # Prepare response
            response = {
                "query": query,
                "answer": answer,
                "confidence": self._calculate_confidence(retrieval_results),
                "timestamp": datetime.now().isoformat(),
                "total_documents_found": retrieval_results.get("total_hits", 0),
                "query_analysis": retrieval_results.get("query_analysis", {})
            }
            
            if include_sources:
                response["sources"] = self._extract_sources(retrieval_results)
                response["source_count"] = len(response["sources"])
            
            # Add retrieval metadata
            response["retrieval_metadata"] = {
                "text_results": len(retrieval_results.get("text_results", [])),
                "visual_results": len(retrieval_results.get("visual_results", [])),
                "fused_results": len(retrieval_results.get("fused_results", [])),
                "modality_used": modality
            }
            
            logger.info(f"✅ Generated answer for query: '{query}'")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "query": query,
                "answer": f"I encountered an error processing your query: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_answer(self, query: str, retrieval_results: Dict[str, Any]) -> str:
        """Generate natural language answer from retrieved content"""
        # Extract all relevant content
        all_content = []
        
        # Text content
        for result in retrieval_results.get("text_results", []):
            content = result.get("content", "")
            if content and len(content) > 10:
                all_content.append({
                    "text": content,
                    "score": result.get("score", 0),
                    "type": "text",
                    "page": result.get("page", 0)
                })
        
        # Fused content (text + visual)
        for result in retrieval_results.get("fused_results", []):
            content = result.get("text_content", "")
            if content and len(content) > 10:
                all_content.append({
                    "text": content,
                    "score": result.get("score", 0),
                    "type": "fused",
                    "page": result.get("page", 0),
                    "alignment_confidence": result.get("alignment_confidence", 0)
                })
        
        # Sort by relevance score
        all_content.sort(key=lambda x: x["score"], reverse=True)
        
        if not all_content:
            return "I couldn't find any relevant information to answer your question."
        
        # Analyze query type
        query_lower = query.lower()
        
        # For simple factual queries
        if any(word in query_lower for word in ["what", "where", "when", "who", "how much", "how many"]):
            return self._generate_factual_answer(query, all_content)
        
        # For comparison/analysis queries
        elif any(word in query_lower for word in ["compare", "difference", "similar", "analyze", "summary"]):
            return self._generate_analytical_answer(query, all_content)
        
        # For risk/contradiction queries
        elif any(word in query_lower for word in ["risk", "contradiction", "error", "issue", "problem"]):
            return self._generate_risk_answer(query, all_content, retrieval_results)
        
        # Default: generate comprehensive answer
        else:
            return self._generate_comprehensive_answer(query, all_content)
    
    def _generate_factual_answer(self, query: str, content: List[Dict]) -> str:
        """Generate answer for factual queries"""
        # Extract key information from top content
        top_content = content[:3]  # Use top 3 most relevant
        
        answers = []
        for item in top_content:
            text = item["text"]
            # Try to extract specific information based on query
            if "date" in query.lower():
                # Look for dates in text
                import re
                dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                if dates:
                    answers.append(f"Found date: {dates[0]}")
            
            elif "amount" in query.lower() or "total" in query.lower() or "price" in query.lower():
                # Look for amounts
                import re
                amounts = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
                if amounts:
                    answers.append(f"Amount mentioned: {amounts[0]}")
            
            elif "signature" in query.lower():
                answers.append(f"Signature-related content found on page {item.get('page', 'unknown')}")
            
            else:
                # Generic extraction - take first sentence
                sentences = text.split('.')
                if sentences and len(sentences[0]) > 20:
                    answers.append(sentences[0] + ".")
        
        if answers:
            return "Based on the document content:\n\n" + "\n".join(f"- {ans}" for ans in answers[:3])
        else:
            # Fallback to showing most relevant content
            return f"The most relevant information found:\n\n{content[0]['text'][:200]}..."
    
    def _generate_analytical_answer(self, query: str, content: List[Dict]) -> str:
        """Generate answer for analytical queries"""
        if len(content) < 2:
            return "I need more information to provide a comparison or analysis."
        
        # Group content by document/page
        content_by_page = {}
        for item in content[:5]:  # Limit to top 5
            page = item.get("page", 0)
            if page not in content_by_page:
                content_by_page[page] = []
            content_by_page[page].append(item)
        
        analysis_parts = []
        
        if len(content_by_page) > 1:
            analysis_parts.append(f"Found relevant information across {len(content_by_page)} different pages:")
            
            for page, items in list(content_by_page.items())[:3]:  # Limit to 3 pages
                page_text = " ".join([item["text"][:100] for item in items[:2]])
                analysis_parts.append(f"• Page {page + 1}: {page_text}...")
        else:
            analysis_parts.append("Found relevant information:")
            for i, item in enumerate(content[:3]):
                analysis_parts.append(f"• {item['text'][:150]}...")
        
        return "\n\n".join(analysis_parts)
    
    def _generate_risk_answer(self, query: str, content: List[Dict], retrieval_results: Dict) -> str:
        """Generate answer for risk/contradiction queries"""
        # Check if we have risk information in metadata
        documents_with_risk = []
        
        # Extract document IDs from results
        doc_ids = set()
        for result_type in ["text_results", "visual_results", "fused_results"]:
            for result in retrieval_results.get(result_type, []):
                doc_id = result.get("document_id")
                if doc_id:
                    doc_ids.add(doc_id)
        
        # Get risk information for these documents
        for doc_id in list(doc_ids)[:3]:  # Limit to 3 documents
            stats = self.retriever.indexer.get_document_stats(doc_id)
            if stats and stats.get("document_info", {}).get("risk_score", 0) > 0.3:
                risk_score = stats["document_info"]["risk_score"]
                contradictions = stats["document_info"].get("contradictions", 0)
                
                risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"
                documents_with_risk.append({
                    "document_id": doc_id,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "contradictions": contradictions
                })
        
        if documents_with_risk:
            answer_parts = ["Risk analysis found:"]
            for doc in documents_with_risk:
                answer_parts.append(
                    f"• Document {doc['document_id'][:8]}...: {doc['risk_level']} risk "
                    f"(score: {doc['risk_score']:.2f}), {doc['contradictions']} contradictions"
                )
            return "\n".join(answer_parts)
        else:
            return "No significant risks or contradictions were found in the retrieved documents."
    
    def _generate_comprehensive_answer(self, query: str, content: List[Dict]) -> str:
        """Generate comprehensive answer for general queries"""
        # Use top 3 content items
        top_content = content[:3]
        
        answer_parts = [f"Regarding your question '{query}':"]
        
        for i, item in enumerate(top_content, 1):
            # Truncate text for readability
            text = item["text"]
            if len(text) > 200:
                text = text[:200] + "..."
            
            source_info = f" (page {item.get('page', 'unknown') + 1})" if item.get('page') is not None else ""
            answer_parts.append(f"{i}. {text}{source_info}")
        
        answer_parts.append("\nThis information is based on the most relevant sections found in the document(s).")
        
        return "\n\n".join(answer_parts)
    
    def _calculate_confidence(self, retrieval_results: Dict[str, Any]) -> float:
        """Calculate confidence score for the answer"""
        scores = []
        
        # Collect all scores
        for result_type in ["text_results", "visual_results", "fused_results"]:
            for result in retrieval_results.get(result_type, []):
                scores.append(result.get("score", 0))
        
        if not scores:
            return 0.0
        
        # Weighted average with emphasis on top scores
        top_scores = sorted(scores, reverse=True)[:3]
        if top_scores:
            return sum(top_scores) / len(top_scores)
        else:
            return sum(scores) / len(scores)
    
    def _extract_sources(self, retrieval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source information from retrieval results"""
        sources = []
        
        # Process each modality
        for result_type, result_name in [
            ("text_results", "Text"),
            ("visual_results", "Visual"),
            ("fused_results", "Fused")
        ]:
            for result in retrieval_results.get(result_type, []):
                source = {
                    "type": result_name,
                    "document_id": result.get("document_id"),
                    "score": result.get("score", 0),
                    "page": result.get("page"),
                    "agent": result.get("agent")
                }
                
                # Add type-specific information
                if result_type == "text_results":
                    source["content_preview"] = result.get("content", "")[:100] + "..."
                elif result_type == "visual_results":
                    source["element_type"] = result.get("element_type")
                    source["bbox"] = result.get("bbox")
                elif result_type == "fused_results":
                    source["alignment_confidence"] = result.get("alignment_confidence")
                    source["content_preview"] = result.get("text_content", "")[:100] + "..."
                
                sources.append(source)
        
        # Sort by score
        sources.sort(key=lambda x: x["score"], reverse=True)
        return sources[:10]  # Limit to top 10 sources
    
    def _generate_no_results_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no results are found"""
        return {
            "query": query,
            "answer": "I couldn't find any information relevant to your query in the indexed documents.",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "suggestions": [
                "Try rephrasing your query",
                "Be more specific about what you're looking for",
                "Check if the relevant documents have been indexed"
            ]
        }
    
    def batch_query(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        results = []
        for query in queries:
            results.append(self.query(query, **kwargs))
        return results
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics about queries processed"""
        return {
            "total_queries_cached": len(self.answer_cache),
            "cache_hit_rate": 0,  # Would track in production
            "average_confidence": 0,  # Would track in production
            "most_common_queries": []  # Would track in production
        }