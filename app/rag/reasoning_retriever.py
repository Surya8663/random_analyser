"""
Reasoning-aware retriever that combines RAG similarity with agent confidence
and reasoning outputs for superior document understanding.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict

from app.utils.logger import setup_logger
from app.rag.agent_router import AgentRouter
from app.rag.indexer import MultiModalIndexer
from app.rag.multimodal_embedder import MultiModalEmbedder
from app.core.models import MultiModalDocument

logger = setup_logger(__name__)


class ReasoningRetriever:
    """
    Advanced retriever that integrates:
    1. RAG similarity scores
    2. Agent confidence scores
    3. Risk assessment
    4. Provenance reliability
    5. Cross-modal consistency
    """
    
    def __init__(self, agent_router: Optional[AgentRouter] = None):
        self.agent_router = agent_router or AgentRouter()
        self.indexer = MultiModalIndexer()
        self.embedder = MultiModalEmbedder()
        
        # Scoring weights (can be tuned)
        self.scoring_weights = {
            'similarity': 0.35,
            'agent_confidence': 0.25,
            'risk_factor': 0.20,
            'provenance_reliability': 0.15,
            'cross_modal_consistency': 0.05
        }
        
        # Cache for performance
        self.query_cache = {}
        self.document_metadata_cache = {}
    
    async def retrieve_with_reasoning(self, 
                                     query: str,
                                     document_id: Optional[str] = None,
                                     k: int = 10,
                                     include_explanations: bool = True) -> Dict[str, Any]:
        """
        Perform reasoning-aware retrieval
        
        Args:
            query: Natural language query
            document_id: Optional specific document
            k: Number of results to return
            include_explanations: Whether to include reasoning explanations
        
        Returns:
            Comprehensive retrieval results with reasoning
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Route query through agents
            routing_result = await self.agent_router.route_query(query, context={"document_id": document_id})
            
            # Step 2: Perform traditional RAG retrieval
            rag_results = await self._perform_rag_retrieval(query, document_id, k * 2)
            
            # Step 3: Combine RAG results with agent outputs
            combined_results = await self._combine_rag_with_agents(
                rag_results, 
                routing_result, 
                k
            )
            
            # Step 4: Apply reasoning-based re-ranking
            reranked_results = await self._apply_reasoning_reranking(
                combined_results, 
                routing_result
            )
            
            # Step 5: Generate explanations if requested
            explanations = {}
            if include_explanations:
                explanations = await self._generate_explanations(
                    reranked_results, 
                    routing_result
                )
            
            # Step 6: Compile final result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_result = {
                "query": query,
                "document_id": document_id,
                "total_results": len(reranked_results),
                "results": reranked_results[:k],
                "explanations": explanations,
                "routing_summary": {
                    "agents_used": list(routing_result.get('agent_contributions', {}).keys()),
                    "overall_confidence": routing_result.get('overall_confidence', 0),
                    "primary_modality": routing_result.get('primary_modality', 'text')
                },
                "retrieval_metadata": {
                    "rag_results_count": rag_results.get('total_hits', 0),
                    "reasoning_applied": True,
                    "scoring_weights": self.scoring_weights,
                    "processing_time": processing_time
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"🔍 Reasoning-aware retrieval: {len(reranked_results)} results, {processing_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Reasoning retrieval failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "explanations": {},
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _perform_rag_retrieval(self, 
                                    query: str, 
                                    document_id: Optional[str],
                                    k: int) -> Dict[str, Any]:
        """Perform traditional RAG retrieval"""
        try:
            # Build filters
            filters = {}
            if document_id:
                filters['document_id'] = document_id
            
            # Perform hybrid search
            rag_results = self.agent_router.retriever.hybrid_search(
                text_query=query,
                k=k,
                filters=filters if filters else None
            )
            
            # Combine all results
            all_results = []
            
            # Text results
            for result in rag_results.get('text', []):
                result['modality'] = 'text'
                all_results.append(result)
            
            # Visual results
            for result in rag_results.get('visual', []):
                result['modality'] = 'visual'
                all_results.append(result)
            
            # Fused results
            for result in rag_results.get('fused', []):
                result['modality'] = 'fused'
                all_results.append(result)
            
            # Sort by score
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                'total_hits': len(all_results),
                'results': all_results,
                'modality_breakdown': {
                    'text': len(rag_results.get('text', [])),
                    'visual': len(rag_results.get('visual', [])),
                    'fused': len(rag_results.get('fused', []))
                }
            }
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return {
                'total_hits': 0,
                'results': [],
                'modality_breakdown': {}
            }
    
    async def _combine_rag_with_agents(self, 
                                      rag_results: Dict[str, Any], 
                                      routing_result: Dict[str, Any],
                                      k: int) -> List[Dict[str, Any]]:
        """Combine RAG results with agent outputs"""
        combined = []
        
        # Extract agent contributions
        agent_contributions = routing_result.get('agent_contributions', {})
        agent_results = routing_result.get('agent_results', {})
        
        # Process each RAG result
        for rag_result in rag_results.get('results', [])[:k * 2]:  # Consider more results
            result_id = rag_result.get('metadata', {}).get('document_id', 'unknown')
            
            # Start with RAG score
            rag_score = rag_result.get('score', 0)
            
            # Collect agent confidence for this document
            agent_scores = []
            agent_explanations = []
            
            for agent_name, agent_result in agent_results.items():
                if 'error' in agent_result:
                    continue
                
                agent_confidence = agent_result.get('confidence', 0)
                agent_weight = agent_contributions.get(agent_name, {}).get('weight', 0.3)
                
                # Check if agent has results for this document
                agent_doc_results = agent_result.get('results', {}).get('results', [])
                if isinstance(agent_doc_results, list):
                    for agent_doc_result in agent_doc_results:
                        agent_doc_id = agent_doc_result.get('metadata', {}).get('document_id')
                        if agent_doc_id == result_id:
                            agent_scores.append(agent_confidence * agent_weight)
                            
                            # Extract agent explanation
                            agent_explanation = self._extract_agent_explanation(
                                agent_name, 
                                agent_doc_result
                            )
                            if agent_explanation:
                                agent_explanations.append({
                                    'agent': agent_name,
                                    'explanation': agent_explanation,
                                    'confidence': agent_confidence
                                })
            
            # Calculate combined score
            combined_score = self._calculate_combined_score(
                rag_score, 
                agent_scores, 
                rag_result
            )
            
            # Create combined result
            combined_result = {
                **rag_result,
                'combined_score': combined_score,
                'rag_score': rag_score,
                'agent_scores': agent_scores,
                'average_agent_confidence': np.mean(agent_scores) if agent_scores else 0,
                'agent_explanations': agent_explanations,
                'reasoning_applied': len(agent_scores) > 0
            }
            
            combined.append(combined_result)
        
        return combined
    
    async def _apply_reasoning_reranking(self, 
                                        combined_results: List[Dict[str, Any]], 
                                        routing_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply reasoning-based re-ranking to results"""
        if not combined_results:
            return []
        
        reranked = []
        
        for result in combined_results:
            # Extract document metadata
            doc_id = result.get('metadata', {}).get('document_id')
            
            if not doc_id:
                reranked.append(result)
                continue
            
            # Get document risk and other reasoning factors
            document_metadata = await self._get_document_metadata(doc_id)
            
            # Calculate reasoning score
            reasoning_score = self._calculate_reasoning_score(
                result, 
                document_metadata, 
                routing_result
            )
            
            # Update result with reasoning information
            result['reasoning_score'] = reasoning_score
            result['document_metadata'] = document_metadata
            
            # Calculate final score (weighted combination)
            final_score = (
                result['combined_score'] * 0.7 + 
                reasoning_score * 0.3
            )
            result['final_score'] = final_score
            
            reranked.append(result)
        
        # Sort by final score
        reranked.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return reranked
    
    async def _generate_explanations(self, 
                                   results: List[Dict[str, Any]], 
                                   routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations for retrieval results"""
        explanations = {
            "scoring_explanation": {},
            "agent_contributions": {},
            "reasoning_factors": {},
            "top_result_explanation": ""
        }
        
        if not results:
            return explanations
        
        # Explain scoring
        explanations["scoring_explanation"] = {
            "weights_used": self.scoring_weights,
            "score_components": [
                "similarity: RAG similarity to query",
                "agent_confidence: Confidence from relevant agents",
                "risk_factor: Document risk assessment",
                "provenance_reliability: Source reliability",
                "cross_modal_consistency: Consistency across modalities"
            ]
        }
        
        # Explain agent contributions
        agent_contributions = routing_result.get('agent_contributions', {})
        for agent_name, contribution in agent_contributions.items():
            explanations["agent_contributions"][agent_name] = {
                "confidence": contribution.get('confidence', 0),
                "weight": contribution.get('weight', 0),
                "role": self._explain_agent_role(agent_name)
            }
        
        # Explain top result
        top_result = results[0] if results else {}
        if top_result:
            explanations["top_result_explanation"] = self._explain_top_result(top_result)
        
        # Explain reasoning factors
        for i, result in enumerate(results[:3]):
            doc_id = result.get('metadata', {}).get('document_id')
            if doc_id:
                explanations["reasoning_factors"][doc_id] = {
                    "rank": i + 1,
                    "final_score": result.get('final_score', 0),
                    "rag_score": result.get('rag_score', 0),
                    "reasoning_score": result.get('reasoning_score', 0),
                    "key_factors": self._extract_key_factors(result)
                }
        
        return explanations
    
    async def _get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get document metadata including risk and reasoning information"""
        # Check cache first
        if document_id in self.document_metadata_cache:
            return self.document_metadata_cache[document_id]
        
        try:
            # Get document stats from indexer
            stats = self.indexer.get_document_stats(document_id)
            
            if not stats:
                # Return default metadata
                metadata = {
                    "risk_score": 0.5,
                    "contradictions": 0,
                    "extracted_fields": 0,
                    "agent_success": True,
                    "provenance_quality": 0.7
                }
            else:
                doc_info = stats.get("document_info", {})
                metadata = {
                    "risk_score": doc_info.get("risk_score", 0.5),
                    "contradictions": doc_info.get("contradictions", 0),
                    "extracted_fields": stats.get("indexed_items", {}).get("text", 0),
                    "agent_success": True,  # Would check actual agent success
                    "provenance_quality": 0.8  # Would calculate from provenance
                }
            
            # Cache the result
            self.document_metadata_cache[document_id] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get document metadata for {document_id}: {e}")
            return {
                "risk_score": 0.5,
                "contradictions": 0,
                "extracted_fields": 0,
                "agent_success": True,
                "provenance_quality": 0.7
            }
    
    def _calculate_combined_score(self, 
                                 rag_score: float, 
                                 agent_scores: List[float], 
                                 rag_result: Dict[str, Any]) -> float:
        """Calculate combined score from RAG and agent outputs"""
        if not agent_scores:
            return rag_score
        
        # Average agent confidence
        avg_agent_confidence = np.mean(agent_scores)
        
        # Modality factor (fused results get boost)
        modality = rag_result.get('modality', 'text')
        modality_factor = 1.0
        if modality == 'fused':
            modality_factor = 1.2
        elif modality == 'visual':
            modality_factor = 1.1
        
        # Calculate combined score
        combined = (
            rag_score * self.scoring_weights['similarity'] +
            avg_agent_confidence * self.scoring_weights['agent_confidence']
        ) * modality_factor
        
        return min(combined, 1.0)
    
    def _calculate_reasoning_score(self, 
                                  result: Dict[str, Any], 
                                  document_metadata: Dict[str, Any],
                                  routing_result: Dict[str, Any]) -> float:
        """Calculate reasoning-based score"""
        reasoning_score = 0.0
        
        # Risk factor (inverse - lower risk is better)
        risk_score = document_metadata.get('risk_score', 0.5)
        risk_factor = 1.0 - risk_score  # Lower risk gives higher score
        
        # Provenance reliability
        provenance_quality = document_metadata.get('provenance_quality', 0.7)
        
        # Agent success factor
        agent_success = document_metadata.get('agent_success', True)
        agent_factor = 1.0 if agent_success else 0.5
        
        # Cross-modal consistency (if fused result)
        modality = result.get('modality', 'text')
        cross_modal_factor = 1.0
        if modality == 'fused':
            cross_modal_factor = 1.1
        
        # Query intent alignment
        primary_modality = routing_result.get('primary_modality', 'text')
        modality_alignment = 1.0
        if modality == primary_modality or (modality == 'fused' and primary_modality in ['text', 'visual']):
            modality_alignment = 1.2
        
        # Calculate reasoning score
        reasoning_score = (
            risk_factor * self.scoring_weights['risk_factor'] +
            provenance_quality * self.scoring_weights['provenance_reliability'] +
            cross_modal_factor * self.scoring_weights['cross_modal_consistency']
        ) * agent_factor * modality_alignment
        
        return min(reasoning_score, 1.0)
    
    def _extract_agent_explanation(self, agent_name: str, agent_result: Dict[str, Any]) -> str:
        """Extract explanation from agent result"""
        if agent_name == 'text_agent':
            return "Text agent found relevant content matching query"
        elif agent_name == 'vision_agent':
            element_type = agent_result.get('metadata', {}).get('element_type', 'element')
            return f"Vision agent detected {element_type}"
        elif agent_name == 'fusion_agent':
            return "Fusion agent aligned text and visual information"
        elif agent_name == 'reasoning_agent':
            analysis = agent_result.get('analysis', {})
            return analysis.get('summary', 'Reasoning analysis performed')
        else:
            return f"{agent_name} contributed to result"
    
    def _explain_agent_role(self, agent_name: str) -> str:
        """Explain what an agent does"""
        roles = {
            'vision_agent': 'Analyzes visual elements (tables, signatures, logos)',
            'text_agent': 'Extracts and understands text content',
            'fusion_agent': 'Aligns text and visual information',
            'reasoning_agent': 'Assesses risk, detects contradictions, validates',
            'explainability_agent': 'Tracks provenance and explains decisions'
        }
        return roles.get(agent_name, 'Processes document information')
    
    def _explain_top_result(self, result: Dict[str, Any]) -> str:
        """Generate explanation for top result"""
        doc_id = result.get('metadata', {}).get('document_id', 'document')
        final_score = result.get('final_score', 0)
        rag_score = result.get('rag_score', 0)
        reasoning_score = result.get('reasoning_score', 0)
        modality = result.get('modality', 'text')
        
        explanation = f"Document {doc_id[:8]}... ranked highest with score {final_score:.2f}. "
        
        if rag_score > 0.7:
            explanation += "It has strong semantic relevance to your query. "
        
        if reasoning_score > 0.6:
            explanation += "Reasoning analysis shows high confidence. "
        
        if modality == 'fused':
            explanation += "This result combines both text and visual information. "
        elif modality == 'visual':
            explanation += "This result is based on visual content analysis. "
        
        if result.get('agent_explanations'):
            explanation += f"{len(result['agent_explanations'])} agents contributed to this result."
        
        return explanation
    
    def _extract_key_factors(self, result: Dict[str, Any]) -> List[str]:
        """Extract key factors that contributed to the ranking"""
        factors = []
        
        rag_score = result.get('rag_score', 0)
        if rag_score > 0.8:
            factors.append("high_semantic_similarity")
        elif rag_score < 0.4:
            factors.append("low_semantic_similarity")
        
        reasoning_score = result.get('reasoning_score', 0)
        if reasoning_score > 0.7:
            factors.append("strong_reasoning_confidence")
        
        modality = result.get('modality', 'text')
        if modality == 'fused':
            factors.append("cross_modal_alignment")
        
        agent_explanations = result.get('agent_explanations', [])
        if len(agent_explanations) >= 2:
            factors.append("multi_agent_support")
        
        document_metadata = result.get('document_metadata', {})
        risk_score = document_metadata.get('risk_score', 0.5)
        if risk_score < 0.3:
            factors.append("low_risk")
        elif risk_score > 0.7:
            factors.append("high_risk")
        
        return factors
    
    async def batch_retrieve(self, 
                            queries: List[str], 
                            document_ids: Optional[List[str]] = None,
                            k: int = 5) -> List[Dict[str, Any]]:
        """Perform batch retrieval"""
        tasks = []
        
        for i, query in enumerate(queries):
            doc_id = document_ids[i] if document_ids and i < len(document_ids) else None
            tasks.append(self.retrieve_with_reasoning(query, doc_id, k))
        
        return await asyncio.gather(*tasks)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        index_stats = self.indexer.get_index_stats()
        
        return {
            "index_statistics": index_stats,
            "scoring_weights": self.scoring_weights,
            "cache_sizes": {
                "query_cache": len(self.query_cache),
                "document_metadata_cache": len(self.document_metadata_cache)
            },
            "agent_router_initialized": self.agent_router is not None
        }


# Example usage
async def test_reasoning_retriever():
    """Test the reasoning retriever"""
    retriever = ReasoningRetriever()
    
    test_queries = [
        "Find high-risk contracts with signatures",
        "Show me tables with financial data",
        "What documents have date inconsistencies?"
    ]
    
    for query in test_queries:
        result = await retriever.retrieve_with_reasoning(query, k=3)
        print(f"\nQuery: {query}")
        print(f"Results: {result.get('total_results', 0)}")
        
        if result.get('results'):
            top_result = result['results'][0]
            print(f"Top score: {top_result.get('final_score', 0):.2f}")
            print(f"Explanation: {result.get('explanations', {}).get('top_result_explanation', '')[:100]}...")


if __name__ == "__main__":
    asyncio.run(test_reasoning_retriever())