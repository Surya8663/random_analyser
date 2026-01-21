"""
Query planner for decomposing natural language queries into sub-queries
for different modalities and reasoning aspects.
"""
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from app.utils.logger import setup_logger
from app.core.models import DocumentType

logger = setup_logger(__name__)


class QueryPlanner:
    """Plan and decompose queries for multi-modal reasoning"""
    
    def __init__(self):
        # Query patterns for different intents
        self.text_patterns = [
            (r'what (is|are) (the|this)', 'factual_extraction'),
            (r'extract (.*)', 'field_extraction'),
            (r'find (.*)', 'search_query'),
            (r'show me (.*)', 'display_query'),
            (r'list (.*)', 'listing_query'),
            (r'how much|how many', 'quantitative_query'),
            (r'when|where|who', 'temporal_spatial_personal_query'),
        ]
        
        self.visual_patterns = [
            (r'signature', 'signature_detection'),
            (r'table|chart|graph|figure', 'tabular_visual'),
            (r'logo|stamp|seal', 'branding_element'),
            (r'image|picture|photo', 'general_visual'),
            (r'layout|format|design', 'layout_analysis'),
        ]
        
        self.reasoning_patterns = [
            (r'risk|danger|warning', 'risk_assessment'),
            (r'contradiction|inconsistency|conflict', 'contradiction_detection'),
            (r'validate|verify|check', 'validation_query'),
            (r'compare|difference|similar', 'comparison_query'),
            (r'analyze|analysis|study', 'analytical_query'),
            (r'summary|overview', 'summarization_query'),
            (r'recommend|suggest|advise', 'recommendation_query'),
        ]
        
        # Modality weights
        self.modality_weights = {
            'text': 1.0,
            'visual': 1.2,
            'reasoning': 1.5
        }
        
        # Agent routing preferences
        self.agent_routing = {
            'factual_extraction': ['text_agent', 'fusion_agent'],
            'field_extraction': ['text_agent', 'vision_agent'],
            'signature_detection': ['vision_agent', 'reasoning_agent'],
            'risk_assessment': ['reasoning_agent', 'fusion_agent'],
            'contradiction_detection': ['reasoning_agent', 'fusion_agent'],
            'tabular_visual': ['vision_agent', 'fusion_agent'],
        }
    
    def plan_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plan a query by decomposing it into sub-queries for different modalities
        
        Args:
            query: Natural language query
            context: Optional context (document type, previous queries, etc.)
        
        Returns:
            Query plan with sub-queries and routing instructions
        """
        query_lower = query.lower().strip()
        
        # Analyze query intent
        intent_analysis = self._analyze_query_intent(query_lower)
        
        # Generate sub-queries for each modality
        sub_queries = self._generate_sub_queries(query, intent_analysis)
        
        # Determine agent routing
        agent_routing = self._determine_agent_routing(intent_analysis, context)
        
        # Determine modality priorities
        modality_priorities = self._determine_modality_priorities(intent_analysis)
        
        # Estimate query complexity
        complexity_score = self._estimate_complexity(query, intent_analysis)
        
        plan = {
            "original_query": query,
            "intent_analysis": intent_analysis,
            "sub_queries": sub_queries,
            "agent_routing": agent_routing,
            "modality_priorities": modality_priorities,
            "complexity_score": complexity_score,
            "requires_fusion": len(intent_analysis.get('modalities', [])) > 1,
            "requires_reasoning": 'reasoning' in intent_analysis.get('modalities', []),
            "timestamp": np.datetime64('now').astype(str)
        }
        
        logger.info(f"📋 Query planned: '{query}' -> {intent_analysis.get('primary_intent', 'unknown')}")
        
        return plan
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent behind a query"""
        intent_scores = {
            'text': 0.0,
            'visual': 0.0,
            'reasoning': 0.0
        }
        
        detected_intents = []
        modalities = set()
        
        # Check text patterns
        for pattern, intent_type in self.text_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                intent_scores['text'] += 0.3
                detected_intents.append(intent_type)
                modalities.add('text')
        
        # Check visual patterns
        for pattern, intent_type in self.visual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                intent_scores['visual'] += 0.4
                detected_intents.append(intent_type)
                modalities.add('visual')
        
        # Check reasoning patterns
        for pattern, intent_type in self.reasoning_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                intent_scores['reasoning'] += 0.5
                detected_intents.append(intent_type)
                modalities.add('reasoning')
        
        # Boost scores for certain keywords
        boosting_keywords = {
            'text': ['extract', 'find', 'search', 'text', 'content'],
            'visual': ['show', 'display', 'see', 'look', 'image'],
            'reasoning': ['why', 'how', 'analyze', 'explain', 'understand']
        }
        
        for modality, keywords in boosting_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    intent_scores[modality] += 0.1
        
        # Normalize scores
        total_score = sum(intent_scores.values())
        if total_score > 0:
            intent_scores = {k: v/total_score for k, v in intent_scores.items()}
        
        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else 'text'
        
        # Determine primary specific intent
        primary_specific_intent = detected_intents[0] if detected_intents else 'general_query'
        
        return {
            'primary_intent': primary_intent,
            'primary_specific_intent': primary_specific_intent,
            'intent_scores': intent_scores,
            'detected_intents': detected_intents,
            'modalities': list(modalities),
            'confidence': max(intent_scores.values()) if intent_scores else 0.0
        }
    
    def _generate_sub_queries(self, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sub-queries for different modalities"""
        sub_queries = {}
        
        # Text sub-query (focus on factual extraction)
        if 'text' in intent_analysis['modalities']:
            sub_queries['text'] = self._generate_text_sub_query(query, intent_analysis)
        
        # Visual sub-query (focus on visual elements)
        if 'visual' in intent_analysis['modalities']:
            sub_queries['visual'] = self._generate_visual_sub_query(query, intent_analysis)
        
        # Reasoning sub-query (focus on analysis and validation)
        if 'reasoning' in intent_analysis['modalities']:
            sub_queries['reasoning'] = self._generate_reasoning_sub_query(query, intent_analysis)
        
        # Fusion sub-query (combines text and visual)
        if 'text' in intent_analysis['modalities'] and 'visual' in intent_analysis['modalities']:
            sub_queries['fusion'] = self._generate_fusion_sub_query(query, intent_analysis)
        
        return sub_queries
    
    def _generate_text_sub_query(self, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text-focused sub-query"""
        # Extract key terms for text search
        text_terms = self._extract_key_terms(query, ['text', 'content', 'word', 'sentence', 'paragraph'])
        
        # Remove visual/reasoning specific terms
        visual_terms = ['signature', 'table', 'chart', 'image', 'logo', 'stamp']
        reasoning_terms = ['risk', 'contradiction', 'analyze', 'compare', 'validate']
        
        filtered_terms = [
            term for term in text_terms 
            if term not in visual_terms and term not in reasoning_terms
        ]
        
        # If no specific terms, use the whole query
        if not filtered_terms:
            filtered_terms = text_terms
        
        return {
            'type': 'text_search',
            'query_terms': filtered_terms,
            'focus': 'text_content',
            'search_type': 'semantic',
            'fields': ['text_chunks', 'extracted_entities'],
            'confidence': intent_analysis['intent_scores'].get('text', 0.0)
        }
    
    def _generate_visual_sub_query(self, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual-focused sub-query"""
        # Extract visual element types
        element_types = []
        
        if 'signature' in query.lower():
            element_types.append('signature')
        if any(term in query.lower() for term in ['table', 'chart', 'graph']):
            element_types.append('table')
        if any(term in query.lower() for term in ['logo', 'stamp', 'seal']):
            element_types.append('logo')
        if any(term in query.lower() for term in ['image', 'picture', 'photo']):
            element_types.append('image_region')
        
        # If no specific element type, use general visual search
        if not element_types:
            element_types = ['visual_element']
        
        return {
            'type': 'visual_search',
            'element_types': element_types,
            'focus': 'visual_elements',
            'search_type': 'embedding_similarity',
            'fields': ['visual_regions', 'layout_regions'],
            'confidence': intent_analysis['intent_scores'].get('visual', 0.0)
        }
    
    def _generate_reasoning_sub_query(self, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning-focused sub-query"""
        reasoning_type = 'general_analysis'
        
        if any(term in query.lower() for term in ['risk', 'danger', 'warning']):
            reasoning_type = 'risk_assessment'
        elif any(term in query.lower() for term in ['contradiction', 'inconsistency']):
            reasoning_type = 'contradiction_detection'
        elif any(term in query.lower() for term in ['validate', 'verify']):
            reasoning_type = 'validation'
        elif any(term in query.lower() for term in ['compare', 'difference']):
            reasoning_type = 'comparison'
        elif any(term in query.lower() for term in ['analyze', 'analysis']):
            reasoning_type = 'analysis'
        elif any(term in query.lower() for term in ['summary', 'overview']):
            reasoning_type = 'summarization'
        
        return {
            'type': 'reasoning_analysis',
            'reasoning_type': reasoning_type,
            'focus': 'reasoning_outputs',
            'search_type': 'structured_query',
            'fields': ['contradictions', 'risk_score', 'validation_results'],
            'confidence': intent_analysis['intent_scores'].get('reasoning', 0.0)
        }
    
    def _generate_fusion_sub_query(self, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fusion sub-query (combines text and visual)"""
        return {
            'type': 'fusion_search',
            'focus': 'aligned_data',
            'search_type': 'cross_modal',
            'fields': ['fused_elements', 'text_visual_alignment'],
            'confidence': (intent_analysis['intent_scores'].get('text', 0.0) + 
                          intent_analysis['intent_scores'].get('visual', 0.0)) / 2
        }
    
    def _determine_agent_routing(self, intent_analysis: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Determine which agents to route the query to"""
        routing = {
            'primary_agents': [],
            'secondary_agents': [],
            'agent_weights': {},
            'execution_order': []
        }
        
        # Get specific intent
        specific_intent = intent_analysis.get('primary_specific_intent', 'general_query')
        
        # Determine primary agents based on intent
        if specific_intent in self.agent_routing:
            routing['primary_agents'] = self.agent_routing[specific_intent]
        else:
            # Default routing based on modalities
            modalities = intent_analysis.get('modalities', [])
            if 'text' in modalities:
                routing['primary_agents'].append('text_agent')
            if 'visual' in modalities:
                routing['primary_agents'].append('vision_agent')
            if 'reasoning' in modalities or 'fusion' in intent_analysis.get('detected_intents', []):
                routing['primary_agents'].append('fusion_agent')
            if 'reasoning' in modalities:
                routing['primary_agents'].append('reasoning_agent')
        
        # Add secondary agents
        all_agents = ['vision_agent', 'text_agent', 'fusion_agent', 'reasoning_agent', 'explainability_agent']
        routing['secondary_agents'] = [agent for agent in all_agents if agent not in routing['primary_agents']]
        
        # Assign weights
        for agent in routing['primary_agents']:
            routing['agent_weights'][agent] = 1.0
        
        for agent in routing['secondary_agents']:
            routing['agent_weights'][agent] = 0.3
        
        # Determine execution order
        if 'reasoning' in intent_analysis.get('modalities', []):
            # For reasoning queries, run fusion first, then reasoning
            routing['execution_order'] = ['fusion_agent', 'reasoning_agent']
        elif 'visual' in intent_analysis.get('modalities', []):
            # For visual queries, run vision first, then fusion
            routing['execution_order'] = ['vision_agent', 'fusion_agent']
        else:
            # For text queries, run text agent first
            routing['execution_order'] = ['text_agent', 'fusion_agent']
        
        # Add context-based adjustments
        if context and 'document_type' in context:
            doc_type = context['document_type']
            if doc_type in [DocumentType.INVOICE, DocumentType.CONTRACT]:
                # For financial docs, prioritize fusion and reasoning
                if 'fusion_agent' in routing['agent_weights']:
                    routing['agent_weights']['fusion_agent'] = 1.2
                if 'reasoning_agent' in routing['agent_weights']:
                    routing['agent_weights']['reasoning_agent'] = 1.2
        
        return routing
    
    def _determine_modality_priorities(self, intent_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Determine priority weights for different modalities"""
        priorities = {}
        
        for modality in ['text', 'visual', 'reasoning']:
            base_score = intent_analysis['intent_scores'].get(modality, 0.0)
            weight = self.modality_weights.get(modality, 1.0)
            priorities[modality] = base_score * weight
        
        # Normalize priorities
        total = sum(priorities.values())
        if total > 0:
            priorities = {k: v/total for k, v in priorities.items()}
        
        return priorities
    
    def _estimate_complexity(self, query: str, intent_analysis: Dict[str, Any]) -> float:
        """Estimate query complexity (0-1 scale)"""
        complexity = 0.0
        
        # Base complexity from query length
        query_words = len(query.split())
        complexity += min(query_words / 20, 0.3)  # Max 0.3 for length
        
        # Complexity from number of modalities
        modality_count = len(intent_analysis.get('modalities', []))
        complexity += min(modality_count * 0.2, 0.4)  # Max 0.4 for modalities
        
        # Complexity from reasoning requirements
        if 'reasoning' in intent_analysis.get('modalities', []):
            complexity += 0.3
        
        # Cap at 1.0
        return min(complexity, 1.0)
    
    def _extract_key_terms(self, query: str, stop_words: List[str] = None) -> List[str]:
        """Extract key terms from query"""
        if stop_words is None:
            stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        
        # Simple term extraction (in production, use NLP library)
        words = query.lower().split()
        
        # Remove stop words and short words
        key_terms = [
            word for word in words 
            if word not in stop_words and len(word) > 2
        ]
        
        # Remove duplicates but preserve order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    def batch_plan_queries(self, queries: List[str], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Plan multiple queries"""
        return [self.plan_query(query, context) for query in queries]


if __name__ == "__main__":
    # Test the query planner
    planner = QueryPlanner()
    
    test_queries = [
        "What invoices with signatures have high risk?",
        "Show me all tables in the document",
        "Extract the total amount from the invoice",
        "Analyze the risk factors in this contract",
        "Find contradictions between text and charts"
    ]
    
    for query in test_queries:
        plan = planner.plan_query(query)
        print(f"\nQuery: {query}")
        print(f"Primary Intent: {plan['intent_analysis']['primary_intent']}")
        print(f"Modalities: {plan['intent_analysis']['modalities']}")
        print(f"Agents: {plan['agent_routing']['primary_agents']}")