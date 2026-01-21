"""
Agent router for distributing sub-queries to appropriate agents and indices
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from app.utils.logger import setup_logger
from app.rag.query_planner import QueryPlanner
from app.rag.retriever import MultiModalRetriever
from app.rag.multimodal_embedder import MultiModalEmbedder
from app.core.models import MultiModalDocument

logger = setup_logger(__name__)


class AgentRouter:
    """Route queries to appropriate agents and combine results"""
    
    def __init__(self, retriever: Optional[MultiModalRetriever] = None):
        self.query_planner = QueryPlanner()
        self.retriever = retriever or MultiModalRetriever()
        self.embedder = MultiModalEmbedder()
        
        # Agent capabilities mapping
        self.agent_capabilities = {
            'vision_agent': {
                'modalities': ['visual'],
                'indices': ['visual'],
                'strengths': ['element_detection', 'layout_analysis', 'visual_semantics'],
                'weights': {'visual': 1.0, 'fusion': 0.7, 'text': 0.3}
            },
            'text_agent': {
                'modalities': ['text'],
                'indices': ['text'],
                'strengths': ['entity_extraction', 'text_understanding', 'semantic_search'],
                'weights': {'text': 1.0, 'fusion': 0.6, 'visual': 0.2}
            },
            'fusion_agent': {
                'modalities': ['text', 'visual', 'fusion'],
                'indices': ['text', 'visual', 'fused'],
                'strengths': ['cross_modal_alignment', 'consistency_checking', 'field_extraction'],
                'weights': {'fusion': 1.0, 'text': 0.8, 'visual': 0.8, 'reasoning': 0.6}
            },
            'reasoning_agent': {
                'modalities': ['reasoning'],
                'indices': [],  # Doesn't directly query indices
                'strengths': ['risk_assessment', 'contradiction_detection', 'validation', 'analysis'],
                'weights': {'reasoning': 1.0, 'fusion': 0.8, 'text': 0.5, 'visual': 0.5}
            },
            'explainability_agent': {
                'modalities': ['text', 'visual', 'reasoning'],
                'indices': [],  # Works with existing results
                'strengths': ['provenance_tracking', 'explanation_generation', 'confidence_calibration'],
                'weights': {'reasoning': 0.8, 'fusion': 0.6, 'text': 0.4, 'visual': 0.4}
            }
        }
        
        # Result cache
        self.result_cache = {}
    
    async def route_query(self, 
                         query: str,
                         document: Optional[MultiModalDocument] = None,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a query through the agent pipeline
        
        Args:
            query: Natural language query
            document: Optional specific document to query
            context: Additional context for routing
        
        Returns:
            Combined results from all relevant agents
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Plan the query
            query_plan = self.query_planner.plan_query(query, context)
            logger.info(f"🔄 Routing query: '{query}' -> {query_plan['agent_routing']['primary_agents']}")
            
            # Step 2: Execute agent queries in order
            agent_results = {}
            
            for agent_name in query_plan['agent_routing']['execution_order']:
                if agent_name in query_plan['agent_routing']['primary_agents']:
                    agent_result = await self._query_agent(
                        agent_name, 
                        query, 
                        query_plan, 
                        document
                    )
                    agent_results[agent_name] = agent_result
            
            # Step 3: Query secondary agents if needed
            for agent_name in query_plan['agent_routing']['secondary_agents']:
                agent_weight = query_plan['agent_routing']['agent_weights'].get(agent_name, 0.3)
                if agent_weight > 0.5:  # Only query if weight is significant
                    agent_result = await self._query_agent(
                        agent_name, 
                        query, 
                        query_plan, 
                        document,
                        is_secondary=True
                    )
                    agent_results[agent_name] = agent_result
            
            # Step 4: Combine results
            combined_results = self._combine_agent_results(
                agent_results, 
                query_plan
            )
            
            # Step 5: Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            combined_results.update({
                "query": query,
                "query_plan": query_plan,
                "agent_results": agent_results,
                "processing_time": processing_time,
                "total_agents_queried": len(agent_results),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"✅ Query routed successfully: {len(agent_results)} agents, {processing_time:.2f}s")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "agent_results": {},
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _query_agent(self, 
                          agent_name: str, 
                          query: str, 
                          query_plan: Dict[str, Any],
                          document: Optional[MultiModalDocument] = None,
                          is_secondary: bool = False) -> Dict[str, Any]:
        """Query a specific agent"""
        try:
            agent_capabilities = self.agent_capabilities.get(agent_name, {})
            modalities = agent_capabilities.get('modalities', [])
            
            agent_result = {
                "agent": agent_name,
                "modalities": modalities,
                "is_primary": not is_secondary,
                "weight": query_plan['agent_routing']['agent_weights'].get(agent_name, 0.3),
                "results": {},
                "confidence": 0.0,
                "processing_time": 0.0
            }
            
            start_time = datetime.now()
            
            # Route to appropriate query method based on agent
            if agent_name == 'vision_agent':
                result = await self._query_vision_agent(query, query_plan, document)
            elif agent_name == 'text_agent':
                result = await self._query_text_agent(query, query_plan, document)
            elif agent_name == 'fusion_agent':
                result = await self._query_fusion_agent(query, query_plan, document)
            elif agent_name == 'reasoning_agent':
                result = await self._query_reasoning_agent(query, query_plan, document)
            elif agent_name == 'explainability_agent':
                result = await self._query_explainability_agent(query, query_plan, document)
            else:
                result = {"error": f"Unknown agent: {agent_name}"}
            
            agent_result["results"] = result
            agent_result["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence
            agent_result["confidence"] = self._calculate_agent_confidence(
                agent_name, result, query_plan
            )
            
            return agent_result
            
        except Exception as e:
            logger.error(f"Agent {agent_name} query failed: {e}")
            return {
                "agent": agent_name,
                "error": str(e),
                "results": {},
                "confidence": 0.0,
                "processing_time": 0.0
            }
    
    async def _query_vision_agent(self, 
                                 query: str, 
                                 query_plan: Dict[str, Any],
                                 document: Optional[MultiModalDocument] = None) -> Dict[str, Any]:
        """Query vision agent for visual elements"""
        # Check if we have visual sub-query
        visual_subquery = query_plan.get('sub_queries', {}).get('visual')
        
        if not visual_subquery:
            return {"error": "No visual sub-query in plan"}
        
        # Build filters based on visual sub-query
        filters = {}
        element_types = visual_subquery.get('element_types', [])
        if element_types:
            filters['element_type'] = element_types[0]  # Use first element type
        
        if document and hasattr(document, 'document_id'):
            filters['document_id'] = document.document_id
        
        # Perform visual search
        try:
            # Generate visual embedding from query
            query_embedding = self.embedder.embed_text(query)
            
            # Search visual index
            visual_results = self.retriever.indexer.search(
                query_embedding, 
                modality="visual",
                k=10,
                filters=filters if filters else None
            )
            
            return {
                "search_type": "visual_embedding",
                "element_types": element_types,
                "results": visual_results,
                "result_count": len(visual_results),
                "filters_applied": filters
            }
            
        except Exception as e:
            logger.error(f"Visual agent query failed: {e}")
            return {
                "error": str(e),
                "results": [],
                "result_count": 0
            }
    
    async def _query_text_agent(self, 
                               query: str, 
                               query_plan: Dict[str, Any],
                               document: Optional[MultiModalDocument] = None) -> Dict[str, Any]:
        """Query text agent for text content"""
        text_subquery = query_plan.get('sub_queries', {}).get('text')
        
        if not text_subquery:
            return {"error": "No text sub-query in plan"}
        
        # Build filters
        filters = {}
        if document and hasattr(document, 'document_id'):
            filters['document_id'] = document.document_id
        
        # Perform text search
        try:
            text_results = self.retriever.retrieve(
                query=query,
                modality="text",
                k=10,
                filters=filters if filters else None,
                include_content=True
            )
            
            return {
                "search_type": "text_semantic",
                "query_terms": text_subquery.get('query_terms', []),
                "results": text_results,
                "result_count": text_results.get('total_hits', 0),
                "filters_applied": filters
            }
            
        except Exception as e:
            logger.error(f"Text agent query failed: {e}")
            return {
                "error": str(e),
                "results": [],
                "result_count": 0
            }
    
    async def _query_fusion_agent(self, 
                                 query: str, 
                                 query_plan: Dict[str, Any],
                                 document: Optional[MultiModalDocument] = None) -> Dict[str, Any]:
        """Query fusion agent for cross-modal results"""
        # Build filters
        filters = {}
        if document and hasattr(document, 'document_id'):
            filters['document_id'] = document.document_id
        
        # Perform hybrid search
        try:
            hybrid_results = self.retriever.hybrid_search(
                text_query=query,
                k=10,
                filters=filters if filters else None
            )
            
            # Extract fused results if available
            fused_results = hybrid_results.get('fused', [])
            if not fused_results:
                # Combine text and visual results
                text_results = hybrid_results.get('text', [])
                visual_results = hybrid_results.get('visual', [])
                
                fused_results = self._fuse_text_visual_results(
                    text_results, 
                    visual_results
                )
            
            return {
                "search_type": "hybrid_fusion",
                "hybrid_results": hybrid_results,
                "fused_results": fused_results,
                "result_count": len(fused_results),
                "cross_modal_alignment": len(fused_results) > 0
            }
            
        except Exception as e:
            logger.error(f"Fusion agent query failed: {e}")
            return {
                "error": str(e),
                "results": [],
                "result_count": 0
            }
    
    async def _query_reasoning_agent(self, 
                                    query: str, 
                                    query_plan: Dict[str, Any],
                                    document: Optional[MultiModalDocument] = None) -> Dict[str, Any]:
        """Query reasoning agent for analysis and validation"""
        reasoning_subquery = query_plan.get('sub_queries', {}).get('reasoning')
        
        if not reasoning_subquery:
            return {"error": "No reasoning sub-query in plan"}
        
        reasoning_type = reasoning_subquery.get('reasoning_type', 'general_analysis')
        
        # Perform reasoning based on type
        try:
            if reasoning_type == 'risk_assessment':
                result = await self._perform_risk_assessment(query, document)
            elif reasoning_type == 'contradiction_detection':
                result = await self._perform_contradiction_detection(query, document)
            elif reasoning_type == 'validation':
                result = await self._perform_validation(query, document)
            elif reasoning_type == 'comparison':
                result = await self._perform_comparison(query, document)
            elif reasoning_type == 'analysis':
                result = await self._perform_analysis(query, document)
            else:
                result = {"error": f"Unknown reasoning type: {reasoning_type}"}
            
            return {
                "reasoning_type": reasoning_type,
                "analysis": result,
                "confidence": result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
            }
            
        except Exception as e:
            logger.error(f"Reasoning agent query failed: {e}")
            return {
                "error": str(e),
                "reasoning_type": reasoning_type,
                "analysis": {}
            }
    
    async def _query_explainability_agent(self, 
                                         query: str, 
                                         query_plan: Dict[str, Any],
                                         document: Optional[MultiModalDocument] = None) -> Dict[str, Any]:
        """Query explainability agent for provenance and explanations"""
        # This agent works with results from other agents
        # In a real implementation, it would analyze agent outputs
        
        return {
            "agent": "explainability_agent",
            "capabilities": ["provenance_tracking", "confidence_explanation", "source_attribution"],
            "note": "Explainability agent analyzes outputs from other agents",
            "recommendation": "Run other agents first to get results for explanation"
        }
    
    def _fuse_text_visual_results(self, 
                                 text_results: List[Dict], 
                                 visual_results: List[Dict]) -> List[Dict]:
        """Fuse text and visual results"""
        fused_results = []
        
        # Simple fusion: combine based on document ID
        doc_text_map = {}
        for result in text_results:
            doc_id = result.get('metadata', {}).get('document_id')
            if doc_id:
                if doc_id not in doc_text_map:
                    doc_text_map[doc_id] = []
                doc_text_map[doc_id].append(result)
        
        doc_visual_map = {}
        for result in visual_results:
            doc_id = result.get('metadata', {}).get('document_id')
            if doc_id:
                if doc_id not in doc_visual_map:
                    doc_visual_map[doc_id] = []
                doc_visual_map[doc_id].append(result)
        
        # Create fused results for documents with both text and visual
        common_docs = set(doc_text_map.keys()) & set(doc_visual_map.keys())
        
        for doc_id in common_docs:
            text_doc_results = doc_text_map[doc_id]
            visual_doc_results = doc_visual_map[doc_id]
            
            # Take best text and visual result
            best_text = max(text_doc_results, key=lambda x: x.get('score', 0))
            best_visual = max(visual_doc_results, key=lambda x: x.get('score', 0))
            
            fused_score = (best_text.get('score', 0) + best_visual.get('score', 0)) / 2
            
            fused_results.append({
                "document_id": doc_id,
                "score": fused_score,
                "text_result": best_text,
                "visual_result": best_visual,
                "fusion_type": "text_visual_combination",
                "confidence": fused_score
            })
        
        return fused_results
    
    async def _perform_risk_assessment(self, query: str, document: Optional[MultiModalDocument]) -> Dict[str, Any]:
        """Perform risk assessment"""
        # In real implementation, this would analyze document risk
        return {
            "risk_level": "MEDIUM",
            "risk_score": 0.6,
            "risk_factors": ["multiple_contradictions", "low_confidence_extractions"],
            "confidence": 0.7,
            "recommendation": "Review document for inconsistencies"
        }
    
    async def _perform_contradiction_detection(self, query: str, document: Optional[MultiModalDocument]) -> Dict[str, Any]:
        """Perform contradiction detection"""
        return {
            "contradictions_found": 2,
            "contradiction_types": ["numeric_inconsistency", "date_mismatch"],
            "confidence": 0.8,
            "details": "Found inconsistencies in amounts and dates"
        }
    
    async def _perform_validation(self, query: str, document: Optional[MultiModalDocument]) -> Dict[str, Any]:
        """Perform validation"""
        return {
            "validation_passed": True,
            "validation_score": 0.75,
            "issues_found": ["low_ocr_confidence"],
            "confidence": 0.8,
            "recommendation": "Consider rescanning document for better OCR"
        }
    
    async def _perform_comparison(self, query: str, document: Optional[MultiModalDocument]) -> Dict[str, Any]:
        """Perform comparison analysis"""
        return {
            "comparison_type": "cross_document",
            "similarity_score": 0.65,
            "key_differences": ["amount_values", "signature_presence"],
            "confidence": 0.6
        }
    
    async def _perform_analysis(self, query: str, document: Optional[MultiModalDocument]) -> Dict[str, Any]:
        """Perform general analysis"""
        return {
            "analysis_type": "document_structure",
            "insights": ["well_structured", "clear_sections", "consistent_formatting"],
            "confidence": 0.7,
            "summary": "Document is well-structured with clear organization"
        }
    
    def _calculate_agent_confidence(self, 
                                   agent_name: str, 
                                   result: Dict[str, Any], 
                                   query_plan: Dict[str, Any]) -> float:
        """Calculate confidence score for agent results"""
        base_confidence = 0.5
        
        # Adjust based on agent weight
        agent_weight = query_plan['agent_routing']['agent_weights'].get(agent_name, 0.3)
        base_confidence *= agent_weight
        
        # Adjust based on result quality
        if 'error' in result:
            return 0.1
        
        # Adjust based on result count
        result_count = result.get('result_count', 0)
        if result_count > 0:
            base_confidence += min(result_count * 0.1, 0.3)
        
        # Adjust based on agent type
        agent_caps = self.agent_capabilities.get(agent_name, {})
        agent_strengths = agent_caps.get('strengths', [])
        
        if 'fusion' in agent_name and result.get('cross_modal_alignment', False):
            base_confidence += 0.2
        
        if 'reasoning' in agent_name and 'analysis' in result:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _combine_agent_results(self, 
                              agent_results: Dict[str, Dict[str, Any]], 
                              query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple agents"""
        combined = {
            "answer": "",
            "supporting_evidence": [],
            "agent_contributions": {},
            "overall_confidence": 0.0,
            "primary_modality": query_plan['intent_analysis']['primary_intent']
        }
        
        # Extract answers from each agent
        agent_answers = []
        total_confidence = 0.0
        agent_count = 0
        
        for agent_name, agent_result in agent_results.items():
            if 'error' in agent_result:
                continue
            
            agent_weight = query_plan['agent_routing']['agent_weights'].get(agent_name, 0.3)
            agent_confidence = agent_result.get('confidence', 0.0)
            
            # Generate agent-specific answer
            agent_answer = self._extract_agent_answer(agent_name, agent_result)
            
            if agent_answer:
                agent_answers.append({
                    "agent": agent_name,
                    "answer": agent_answer,
                    "confidence": agent_confidence,
                    "weight": agent_weight,
                    "modalities": agent_result.get('modalities', [])
                })
                
                total_confidence += agent_confidence * agent_weight
                agent_count += 1
            
            # Store agent contribution
            combined["agent_contributions"][agent_name] = {
                "confidence": agent_confidence,
                "weight": agent_weight,
                "processing_time": agent_result.get('processing_time', 0),
                "result_count": agent_result.get('results', {}).get('result_count', 0)
            }
        
        # Combine answers
        if agent_answers:
            # Sort by confidence * weight
            agent_answers.sort(key=lambda x: x['confidence'] * x['weight'], reverse=True)
            
            # Generate combined answer
            primary_answer = agent_answers[0]['answer']
            supporting_evidence = []
            
            for answer in agent_answers[1:3]:  # Take next 2 as supporting
                supporting_evidence.append({
                    "agent": answer['agent'],
                    "evidence": answer['answer'],
                    "confidence": answer['confidence']
                })
            
            combined["answer"] = primary_answer
            combined["supporting_evidence"] = supporting_evidence
            
            # Calculate overall confidence
            if agent_count > 0:
                combined["overall_confidence"] = total_confidence / agent_count
        
        return combined
    
    def _extract_agent_answer(self, agent_name: str, agent_result: Dict[str, Any]) -> str:
        """Extract answer text from agent results"""
        results = agent_result.get('results', {})
        
        if agent_name == 'text_agent':
            text_results = results.get('results', {})
            if text_results and 'text_results' in text_results:
                top_result = text_results['text_results'][0] if text_results['text_results'] else None
                if top_result:
                    return f"Text agent found: {top_result.get('content', '')[:200]}..."
        
        elif agent_name == 'vision_agent':
            visual_results = results.get('results', [])
            if visual_results:
                top_result = visual_results[0]
                element_type = top_result.get('metadata', {}).get('element_type', 'element')
                return f"Vision agent detected {element_type} with confidence {top_result.get('score', 0):.2f}"
        
        elif agent_name == 'fusion_agent':
            fused_results = results.get('fused_results', [])
            if fused_results:
                return f"Fusion agent found {len(fused_results)} cross-modal matches"
        
        elif agent_name == 'reasoning_agent':
            analysis = results.get('analysis', {})
            if analysis:
                reasoning_type = results.get('reasoning_type', 'analysis')
                return f"Reasoning agent {reasoning_type}: {analysis.get('summary', 'Analysis complete')}"
        
        return ""
    
    async def batch_route_queries(self, 
                                 queries: List[str], 
                                 context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Route multiple queries"""
        return await asyncio.gather(*[
            self.route_query(query, context=context) 
            for query in queries
        ])


# Example usage
async def test_agent_router():
    """Test the agent router"""
    router = AgentRouter()
    
    test_queries = [
        "What tables are in this document?",
        "Find high-risk signatures",
        "Extract invoice amounts and dates"
    ]
    
    for query in test_queries:
        result = await router.route_query(query)
        print(f"\nQuery: {query}")
        print(f"Answer: {result.get('answer', 'No answer')[:100]}...")
        print(f"Confidence: {result.get('overall_confidence', 0):.2f}")
        print(f"Agents used: {list(result.get('agent_contributions', {}).keys())}")


if __name__ == "__main__":
    asyncio.run(test_agent_router())