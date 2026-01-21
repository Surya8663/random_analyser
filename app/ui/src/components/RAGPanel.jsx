import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  FiSearch, 
  FiFilter, 
  FiChevronDown, 
  FiChevronUp, 
  FiMessageSquare,
  FiEye,
  FiFileText,
  FiShuffle
} from 'react-icons/fi'
import { Tooltip } from 'react-tooltip'

const RAGPanel = ({ documentId, onQuerySubmit }) => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [filters, setFilters] = useState({
    modality: 'all',
    agent: 'all',
    minConfidence: 0.5,
    riskLevel: 'all'
  })
  const [expandedResult, setExpandedResult] = useState(null)
  const [queryHistory, setQueryHistory] = useState([])
  
  const MODALITIES = [
    { id: 'text', label: 'Text', icon: FiFileText, color: '#3b82f6' },
    { id: 'visual', label: 'Visual', icon: FiEye, color: '#8b5cf6' },
    { id: 'fused', label: 'Fused', icon: FiShuffle, color: '#10b981' }
  ]
  
  const AGENTS = [
    { id: 'vision_agent', label: 'Vision', color: '#8b5cf6' },
    { id: 'text_agent', label: 'Text', color: '#3b82f6' },
    { id: 'fusion_agent', label: 'Fusion', color: '#10b981' },
    { id: 'reasoning_agent', label: 'Reasoning', color: '#f59e0b' }
  ]
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim() || isLoading) return
    
    setIsLoading(true)
    
    try {
      // In production, this would call the API
      const mockResults = generateMockResults(query)
      setResults(mockResults)
      
      // Add to query history
      setQueryHistory(prev => [
        { query, timestamp: new Date().toISOString(), resultCount: mockResults.length },
        ...prev.slice(0, 9)
      ])
      
      // Call parent handler
      if (onQuerySubmit) {
        onQuerySubmit(query)
      }
    } catch (error) {
      console.error('Query failed:', error)
    } finally {
      setIsLoading(false)
    }
  }
  
  const generateMockResults = (queryText) => {
    const mockResults = []
    const modalities = ['text', 'visual', 'fused']
    const agents = AGENTS.map(a => a.id)
    const contentTypes = ['paragraph', 'table', 'signature', 'header', 'figure']
    
    for (let i = 0; i < 8; i++) {
      const modality = modalities[Math.floor(Math.random() * modalities.length)]
      const agent = agents[Math.floor(Math.random() * agents.length)]
      const confidence = Math.random() * 0.4 + 0.6 // 0.6 to 1.0
      const contentType = contentTypes[Math.floor(Math.random() * contentTypes.length)]
      
      mockResults.push({
        id: `result_${i}`,
        query: queryText,
        modality,
        agent,
        confidence,
        score: Math.random() * 0.3 + 0.7, // 0.7 to 1.0
        contentType,
        content: `This is sample ${contentType} content that matches your query "${queryText}". The AI found this relevant based on semantic analysis and cross-modal understanding.`,
        page: Math.floor(Math.random() * 5) + 1,
        boundingBox: modality === 'visual' || modality === 'fused' ? {
          x: Math.random() * 80 + 10,
          y: Math.random() * 80 + 10,
          width: Math.random() * 30 + 10,
          height: Math.random() * 20 + 5
        } : null,
        timestamp: new Date().toISOString(),
        reasoning: {
          factors: ['semantic_similarity', 'agent_confidence', 'cross_modal_alignment'],
          explanation: `Result ranked highly due to strong ${modality} match and ${agent} confidence.`
        }
      })
    }
    
    // Sort by score
    return mockResults.sort((a, b) => b.score - a.score)
  }
  
  const filteredResults = results.filter(result => {
    if (filters.modality !== 'all' && result.modality !== filters.modality) {
      return false
    }
    if (filters.agent !== 'all' && result.agent !== filters.agent) {
      return false
    }
    if (result.confidence < filters.minConfidence) {
      return false
    }
    return true
  })
  
  const getModalityIcon = (modality) => {
    const mod = MODALITIES.find(m => m.id === modality)
    return mod ? mod.icon : FiFileText
  }
  
  const getAgentColor = (agent) => {
    const agentConfig = AGENTS.find(a => a.id === agent)
    return agentConfig ? agentConfig.color : '#6b7280'
  }
  
  const getScoreColor = (score) => {
    if (score >= 0.9) return '#10b981'
    if (score >= 0.7) return '#f59e0b'
    return '#ef4444'
  }
  
  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold gradient-text">Reasoning-Aware RAG</h2>
          <p className="text-sm text-dark-muted">
            Query documents with AI understanding and reasoning
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="px-3 py-1 rounded-lg bg-dark-card text-sm">
            {results.length} results
          </div>
        </div>
      </div>
      
      {/* Query input */}
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about your document (e.g., 'Show me high-risk signatures' or 'What tables contain financial data?')"
            className="w-full px-6 py-4 rounded-xl bg-dark-card border border-dark-border focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/30"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="absolute right-3 top-3 px-4 py-2 rounded-lg bg-primary-500 hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                Searching...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <FiSearch className="w-5 h-5" />
                Search
              </div>
            )}
          </button>
        </div>
      </form>
      
      {/* Filters */}
      <div className="mb-6 p-4 rounded-lg bg-dark-card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <FiFilter className="w-5 h-5" />
            <span className="font-medium">Filters</span>
          </div>
          
          <button
            onClick={() => setFilters({
              modality: 'all',
              agent: 'all',
              minConfidence: 0.5,
              riskLevel: 'all'
            })}
            className="text-sm text-primary-500 hover:text-primary-400"
          >
            Clear all
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Modality filter */}
          <div>
            <label className="block text-sm text-dark-muted mb-2">Modality</label>
            <select
              value={filters.modality}
              onChange={(e) => setFilters(prev => ({ ...prev, modality: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-dark-bg border border-dark-border focus:outline-none focus:border-primary-500"
            >
              <option value="all">All modalities</option>
              {MODALITIES.map(mod => (
                <option key={mod.id} value={mod.id}>{mod.label}</option>
              ))}
            </select>
          </div>
          
          {/* Agent filter */}
          <div>
            <label className="block text-sm text-dark-muted mb-2">Agent</label>
            <select
              value={filters.agent}
              onChange={(e) => setFilters(prev => ({ ...prev, agent: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-dark-bg border border-dark-border focus:outline-none focus:border-primary-500"
            >
              <option value="all">All agents</option>
              {AGENTS.map(agent => (
                <option key={agent.id} value={agent.id}>{agent.label}</option>
              ))}
            </select>
          </div>
          
          {/* Confidence filter */}
          <div>
            <label className="block text-sm text-dark-muted mb-2">
              Min Confidence: {(filters.minConfidence * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.minConfidence}
              onChange={(e) => setFilters(prev => ({ ...prev, minConfidence: parseFloat(e.target.value) }))}
              className="w-full accent-primary-500"
            />
          </div>
          
          {/* Risk filter */}
          <div>
            <label className="block text-sm text-dark-muted mb-2">Risk Level</label>
            <select
              value={filters.riskLevel}
              onChange={(e) => setFilters(prev => ({ ...prev, riskLevel: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-dark-bg border border-dark-border focus:outline-none focus:border-primary-500"
            >
              <option value="all">All risk levels</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Results */}
      <div className="space-y-4">
        <AnimatePresence>
          {filteredResults.map((result, index) => {
            const ModalityIcon = getModalityIcon(result.modality)
            const agentColor = getAgentColor(result.agent)
            const scoreColor = getScoreColor(result.score)
            const isExpanded = expandedResult === result.id
            
            return (
              <motion.div
                key={result.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.05 }}
                className="rounded-lg bg-dark-card border border-dark-border overflow-hidden"
              >
                <div 
                  className="p-4 cursor-pointer hover:bg-dark-border/50 transition-colors"
                  onClick={() => setExpandedResult(isExpanded ? null : result.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div 
                          className="w-8 h-8 rounded-lg flex items-center justify-center"
                          style={{ backgroundColor: `${agentColor}20` }}
                        >
                          <ModalityIcon style={{ color: agentColor }} />
                        </div>
                        
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium capitalize">
                              {result.contentType}
                            </span>
                            <span className="text-xs px-2 py-1 rounded bg-dark-border">
                              Page {result.page}
                            </span>
                          </div>
                          <div className="text-sm text-dark-muted">
                            via {result.agent.replace('_agent', '')} agent • {result.modality} modality
                          </div>
                        </div>
                      </div>
                      
                      <div className="mt-2 text-sm">
                        {result.content.slice(0, 150)}
                        {result.content.length > 150 && '...'}
                      </div>
                    </div>
                    
                    <div className="flex flex-col items-end gap-2">
                      <div className="flex items-center gap-2">
                        <div className="text-sm text-dark-muted">Score:</div>
                        <div 
                          className="px-2 py-1 rounded text-sm font-bold"
                          style={{ 
                            backgroundColor: `${scoreColor}20`,
                            color: scoreColor
                          }}
                        >
                          {(result.score * 100).toFixed(0)}%
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <div className="text-sm text-dark-muted">Confidence:</div>
                        <div className="text-sm font-medium">
                          {(result.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      
                      <button className="p-1">
                        {isExpanded ? (
                          <FiChevronUp className="w-5 h-5" />
                        ) : (
                          <FiChevronDown className="w-5 h-5" />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
                
                {/* Expanded details */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="border-t border-dark-border"
                    >
                      <div className="p-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                          <div>
                            <h4 className="text-sm font-medium mb-2">Reasoning Factors</h4>
                            <div className="flex flex-wrap gap-2">
                              {result.reasoning.factors.map((factor, idx) => (
                                <span
                                  key={idx}
                                  className="px-2 py-1 text-xs rounded bg-dark-border"
                                >
                                  {factor}
                                </span>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="text-sm font-medium mb-2">Explanation</h4>
                            <p className="text-sm text-dark-muted">
                              {result.reasoning.explanation}
                            </p>
                          </div>
                        </div>
                        
                        {result.boundingBox && (
                          <div>
                            <h4 className="text-sm font-medium mb-2">Visual Location</h4>
                            <div className="text-sm text-dark-muted">
                              Bounding box at ({result.boundingBox.x.toFixed(1)}%, {result.boundingBox.y.toFixed(1)}%)
                              • Size: {result.boundingBox.width.toFixed(1)}% × {result.boundingBox.height.toFixed(1)}%
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </AnimatePresence>
        
        {filteredResults.length === 0 && query && (
          <div className="text-center py-12">
            <FiMessageSquare className="w-12 h-12 mx-auto text-dark-muted mb-4" />
            <h3 className="text-lg font-medium mb-2">No results found</h3>
            <p className="text-dark-muted">
              Try adjusting your filters or using different search terms
            </p>
          </div>
        )}
      </div>
      
      {/* Query history */}
      {queryHistory.length > 0 && (
        <div className="mt-8 pt-6 border-t border-dark-border">
          <h3 className="font-medium mb-3">Recent Queries</h3>
          <div className="flex flex-wrap gap-2">
            {queryHistory.slice(0, 5).map((item, index) => (
              <button
                key={index}
                onClick={() => setQuery(item.query)}
                className="px-3 py-2 rounded-lg bg-dark-card hover:bg-dark-border transition-colors text-sm"
              >
                {item.query.slice(0, 30)}
                {item.query.length > 30 && '...'}
                <span className="ml-2 text-xs text-dark-muted">
                  ({item.resultCount})
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
      
      <Tooltip id="rag-tooltip" />
    </div>
  )
}

export default RAGPanel