// src/components/RAGPanel.jsx
import React, { useState } from 'react';
import { FiSearch, FiFilter, FiMessageSquare, FiFileText, FiEye, FiShuffle, FiChevronDown, FiChevronUp } from 'react-icons/fi';
import DocumentAPI from '../api/client';

const MODALITIES = [
  { id: 'text', label: 'Text', icon: FiFileText, color: '#3b82f6' },
  { id: 'visual', label: 'Visual', icon: FiEye, color: '#8b5cf6' },
  { id: 'fused', label: 'Fused', icon: FiShuffle, color: '#10b981' }
];

const AGENTS = [
  { id: 'vision_agent', label: 'Vision', color: '#8b5cf6' },
  { id: 'text_agent', label: 'Text', color: '#3b82f6' },
  { id: 'fusion_agent', label: 'Fusion', color: '#10b981' },
  { id: 'reasoning_agent', label: 'Reasoning', color: '#f59e0b' }
];

const RAGPanel = ({ documentId }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [filters, setFilters] = useState({
    modality: 'all',
    agent: 'all',
    minConfidence: 0.5
  });
  const [expandedResult, setExpandedResult] = useState(null);
  const [queryHistory, setQueryHistory] = useState([]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim() || !documentId) return;

    setIsLoading(true);
    setSearchError(null);

    try {
      const response = await DocumentAPI.queryDocument(documentId, query);
      
      // Format response based on your backend structure
      const formattedResults = formatResults(response, query);
      setResults(formattedResults);

      // Add to query history
      setQueryHistory(prev => [
        { 
          query, 
          timestamp: new Date().toISOString(), 
          resultCount: formattedResults.length 
        },
        ...prev.slice(0, 4)
      ]);

    } catch (error) {
      console.error('Search failed:', error);
      setSearchError(error.message || 'Search failed. Please try again.');
      
      // Show mock results for demo purposes
      const mockResults = generateMockResults(query);
      setResults(mockResults);
    } finally {
      setIsLoading(false);
    }
  };

  const formatResults = (response, queryText) => {
    // Adapt this based on your actual backend response structure
    if (response.results && Array.isArray(response.results)) {
      return response.results.map((result, index) => ({
        id: `result_${index}`,
        query: queryText,
        modality: result.modality || 'text',
        agent: result.agent || 'text_agent',
        confidence: result.confidence || 0.8,
        score: result.score || 0.9,
        contentType: result.content_type || 'text',
        content: result.content || result.text || 'No content available',
        page: result.page || 1,
        boundingBox: result.bbox,
        timestamp: new Date().toISOString(),
        reasoning: {
          factors: result.reasoning_factors || ['semantic_similarity', 'agent_confidence'],
          explanation: result.explanation || `Found in ${result.modality || 'text'} analysis`
        }
      }));
    }

    // If response structure is different, return empty array
    return [];
  };

  const generateMockResults = (queryText) => {
    // For demo purposes only
    const mockResults = [];
    const contentTypes = ['paragraph', 'table', 'signature', 'header', 'figure'];
    
    for (let i = 0; i < 5; i++) {
      const modality = MODALITIES[Math.floor(Math.random() * MODALITIES.length)].id;
      const agent = AGENTS[Math.floor(Math.random() * AGENTS.length)].id;
      const confidence = Math.random() * 0.4 + 0.6;
      const contentType = contentTypes[Math.floor(Math.random() * contentTypes.length)];
      
      mockResults.push({
        id: `mock_${i}`,
        query: queryText,
        modality,
        agent,
        confidence,
        score: Math.random() * 0.3 + 0.7,
        contentType,
        content: `This is sample content that matches your query "${queryText}". Found during ${agent.replace('_agent', '')} analysis with ${(confidence * 100).toFixed(0)}% confidence.`,
        page: Math.floor(Math.random() * 5) + 1,
        boundingBox: modality !== 'text' ? {
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
      });
    }
    
    return mockResults.sort((a, b) => b.score - a.score);
  };

  const filteredResults = results.filter(result => {
    if (filters.modality !== 'all' && result.modality !== filters.modality) {
      return false;
    }
    if (filters.agent !== 'all' && result.agent !== filters.agent) {
      return false;
    }
    if (result.confidence < filters.minConfidence) {
      return false;
    }
    return true;
  });

  const getModalityIcon = (modality) => {
    const mod = MODALITIES.find(m => m.id === modality);
    return mod ? mod.icon : FiFileText;
  };

  const getAgentColor = (agent) => {
    const agentConfig = AGENTS.find(a => a.id === agent);
    return agentConfig ? agentConfig.color : '#6b7280';
  };

  const getScoreColor = (score) => {
    if (score >= 0.9) return '#10b981';
    if (score >= 0.7) return '#f59e0b';
    return '#ef4444';
  };

  const clearFilters = () => {
    setFilters({
      modality: 'all',
      agent: 'all',
      minConfidence: 0.5
    });
  };

  return (
    <div className="panel rag-panel">
      <div className="panel-header">
        <div>
          <h3>🔍 RAG Query Interface</h3>
          <p className="panel-subtitle">
            {documentId ? `Document ID: ${documentId}` : 'Upload a document to start searching'}
          </p>
        </div>
        <div className="results-count">
          {results.length} results
        </div>
      </div>

      {/* Search form */}
      <form onSubmit={handleSearch} className="search-form">
        <div className="search-input-wrapper">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about your document (e.g., 'Show me signatures' or 'Find financial tables')"
            className="search-input"
            disabled={isLoading || !documentId}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim() || !documentId}
            className="search-button"
          >
            {isLoading ? (
              <div className="loading-indicator">
                <div className="spinner" />
                Searching...
              </div>
            ) : (
              <div className="search-button-content">
                <FiSearch />
                <span>Search</span>
              </div>
            )}
          </button>
        </div>
        
        {!documentId && (
          <p className="warning-message">
            Please upload a document first to enable search
          </p>
        )}
      </form>

      {searchError && (
        <div className="error-banner">
          <div className="error-content">
            <span>{searchError}</span>
            <small>(Showing demo data)</small>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="filters-section">
        <div className="filters-header">
          <div className="filters-title">
            <FiFilter />
            <span>Filters</span>
          </div>
          <button onClick={clearFilters} className="clear-filters">
            Clear all
          </button>
        </div>

        <div className="filters-grid">
          <div className="filter-group">
            <label>Modality</label>
            <select
              value={filters.modality}
              onChange={(e) => setFilters(prev => ({ ...prev, modality: e.target.value }))}
              className="filter-select"
            >
              <option value="all">All modalities</option>
              {MODALITIES.map(mod => (
                <option key={mod.id} value={mod.id}>{mod.label}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Agent</label>
            <select
              value={filters.agent}
              onChange={(e) => setFilters(prev => ({ ...prev, agent: e.target.value }))}
              className="filter-select"
            >
              <option value="all">All agents</option>
              {AGENTS.map(agent => (
                <option key={agent.id} value={agent.id}>{agent.label}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Min Confidence: {(filters.minConfidence * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.minConfidence}
              onChange={(e) => setFilters(prev => ({ ...prev, minConfidence: parseFloat(e.target.value) }))}
              className="confidence-slider"
            />
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="results-section">
        {filteredResults.length > 0 ? (
          filteredResults.map((result, index) => {
            const ModalityIcon = getModalityIcon(result.modality);
            const agentColor = getAgentColor(result.agent);
            const scoreColor = getScoreColor(result.score);
            const isExpanded = expandedResult === result.id;

            return (
              <div key={result.id} className="result-card">
                <div 
                  className="result-header"
                  onClick={() => setExpandedResult(isExpanded ? null : result.id)}
                >
                  <div className="result-icon" style={{ backgroundColor: `${agentColor}20` }}>
                    <ModalityIcon style={{ color: agentColor }} />
                  </div>
                  
                  <div className="result-info">
                    <div className="result-title">
                      <span className="content-type">{result.contentType}</span>
                      <span className="page-tag">Page {result.page}</span>
                    </div>
                    <div className="result-subtitle">
                      via {result.agent.replace('_agent', '')} agent • {result.modality} modality
                    </div>
                  </div>

                  <div className="result-stats">
                    <div className="score-badge" style={{ 
                      backgroundColor: `${scoreColor}20`,
                      color: scoreColor
                    }}>
                      Score: {(result.score * 100).toFixed(0)}%
                    </div>
                    <div className="confidence">
                      Conf: {(result.confidence * 100).toFixed(0)}%
                    </div>
                    <button className="expand-button">
                      {isExpanded ? <FiChevronUp /> : <FiChevronDown />}
                    </button>
                  </div>
                </div>

                <div className="result-preview">
                  {result.content.length > 200 ? `${result.content.slice(0, 200)}...` : result.content}
                </div>

                {isExpanded && (
                  <div className="result-details">
                    <div className="details-grid">
                      <div className="detail-section">
                        <h4>Reasoning Factors</h4>
                        <div className="factors-list">
                          {result.reasoning.factors.map((factor, idx) => (
                            <span key={idx} className="factor-tag">
                              {factor}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="detail-section">
                        <h4>Explanation</h4>
                        <p>{result.reasoning.explanation}</p>
                      </div>
                    </div>
                    
                    {result.boundingBox && (
                      <div className="detail-section">
                        <h4>Visual Location</h4>
                        <p>
                          Bounding box at ({result.boundingBox.x.toFixed(1)}%, {result.boundingBox.y.toFixed(1)}%)
                          • Size: {result.boundingBox.width.toFixed(1)}% × {result.boundingBox.height.toFixed(1)}%
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        ) : query && (
          <div className="no-results">
            <FiMessageSquare className="no-results-icon" />
            <h4>No results found</h4>
            <p>Try adjusting your filters or using different search terms</p>
          </div>
        )}
      </div>

      {/* Query history */}
      {queryHistory.length > 0 && (
        <div className="query-history">
          <h4>Recent Queries</h4>
          <div className="history-items">
            {queryHistory.map((item, index) => (
              <button
                key={index}
                onClick={() => setQuery(item.query)}
                className="history-item"
              >
                <span className="query-text">
                  {item.query.length > 30 ? `${item.query.slice(0, 30)}...` : item.query}
                </span>
                <span className="result-count">({item.resultCount})</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        .panel {
          background: #1e293b;
          border-radius: 12px;
          padding: 1.5rem;
          border: 1px solid #334155;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 1.5rem;
        }

        .panel-header h3 {
          margin: 0 0 0.25rem 0;
          color: #e2e8f0;
          font-size: 1.2rem;
        }

        .panel-subtitle {
          color: #94a3b8;
          font-size: 0.875rem;
          margin: 0;
        }

        .results-count {
          background: #334155;
          color: #cbd5e1;
          padding: 0.25rem 0.75rem;
          border-radius: 1rem;
          font-size: 0.875rem;
        }

        .search-form {
          margin-bottom: 1.5rem;
        }

        .search-input-wrapper {
          position: relative;
          display: flex;
          gap: 0.5rem;
        }

        .search-input {
          flex: 1;
          background: #0f172a;
          border: 1px solid #334155;
          border-radius: 8px;
          padding: 0.875rem 1rem;
          color: #e2e8f0;
          font-size: 0.95rem;
          transition: border-color 0.2s;
        }

        .search-input:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .search-input:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .search-button {
          background: #3b82f6;
          color: white;
          border: none;
          border-radius: 8px;
          padding: 0 1.5rem;
          cursor: pointer;
          font-weight: 500;
          display: flex;
          align-items: center;
          justify-content: center;
          min-width: 100px;
          transition: background-color 0.2s;
        }

        .search-button:hover:not(:disabled) {
          background: #2563eb;
        }

        .search-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .search-button-content {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .loading-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .warning-message {
          color: #f59e0b;
          font-size: 0.875rem;
          margin-top: 0.5rem;
        }

        .error-banner {
          background: #fee2e2;
          border: 1px solid #fecaca;
          border-radius: 8px;
          padding: 0.75rem;
          margin-bottom: 1.5rem;
        }

        .error-content {
          color: #dc2626;
          font-size: 0.875rem;
        }

        .error-content small {
          color: #f87171;
          display: block;
          margin-top: 0.25rem;
        }

        .filters-section {
          background: #0f172a;
          border-radius: 8px;
          padding: 1rem;
          margin-bottom: 1.5rem;
        }

        .filters-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .filters-title {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #cbd5e1;
          font-weight: 500;
        }

        .clear-filters {
          background: none;
          border: none;
          color: #3b82f6;
          cursor: pointer;
          font-size: 0.875rem;
        }

        .clear-filters:hover {
          text-decoration: underline;
        }

        .filters-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
        }

        .filter-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .filter-group label {
          color: #94a3b8;
          font-size: 0.875rem;
        }

        .filter-select {
          background: #1e293b;
          border: 1px solid #334155;
          border-radius: 6px;
          padding: 0.5rem;
          color: #e2e8f0;
          font-size: 0.875rem;
        }

        .filter-select:focus {
          outline: none;
          border-color: #3b82f6;
        }

        .confidence-slider {
          width: 100%;
          accent-color: #3b82f6;
        }

        .results-section {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .result-card {
          background: #0f172a;
          border: 1px solid #334155;
          border-radius: 8px;
          overflow: hidden;
        }

        .result-header {
          display: flex;
          align-items: flex-start;
          gap: 1rem;
          padding: 1rem;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .result-header:hover {
          background: rgba(255, 255, 255, 0.02);
        }

        .result-icon {
          width: 40px;
          height: 40px;
          border-radius: 6px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .result-info {
          flex: 1;
        }

        .result-title {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.25rem;
        }

        .content-type {
          color: #e2e8f0;
          font-weight: 500;
          text-transform: capitalize;
        }

        .page-tag {
          background: #334155;
          color: #cbd5e1;
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
        }

        .result-subtitle {
          color: #94a3b8;
          font-size: 0.875rem;
        }

        .result-stats {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 0.25rem;
        }

        .score-badge {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .confidence {
          color: #94a3b8;
          font-size: 0.875rem;
        }

        .expand-button {
          background: none;
          border: none;
          color: #94a3b8;
          cursor: pointer;
          padding: 0.25rem;
        }

        .result-preview {
          padding: 0 1rem 1rem 1rem;
          color: #cbd5e1;
          font-size: 0.875rem;
          line-height: 1.5;
        }

        .result-details {
          border-top: 1px solid #334155;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.01);
        }

        .details-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
          margin-bottom: 1rem;
        }

        .detail-section h4 {
          color: #e2e8f0;
          font-size: 0.875rem;
          margin: 0 0 0.5rem 0;
        }

        .factors-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.25rem;
        }

        .factor-tag {
          background: #334155;
          color: #cbd5e1;
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
        }

        .detail-section p {
          color: #94a3b8;
          font-size: 0.875rem;
          line-height: 1.5;
          margin: 0;
        }

        .no-results {
          text-align: center;
          padding: 3rem 1rem;
          color: #64748b;
        }

        .no-results-icon {
          width: 48px;
          height: 48px;
          margin: 0 auto 1rem;
          display: block;
        }

        .no-results h4 {
          color: #94a3b8;
          margin: 0 0 0.5rem 0;
        }

        .no-results p {
          color: #64748b;
          margin: 0;
          font-size: 0.875rem;
        }

        .query-history {
          margin-top: 1.5rem;
          padding-top: 1.5rem;
          border-top: 1px solid #334155;
        }

        .query-history h4 {
          color: #e2e8f0;
          font-size: 0.875rem;
          margin: 0 0 0.75rem 0;
        }

        .history-items {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }

        .history-item {
          background: #0f172a;
          border: 1px solid #334155;
          border-radius: 6px;
          padding: 0.5rem 0.75rem;
          color: #cbd5e1;
          font-size: 0.875rem;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          transition: all 0.2s;
        }

        .history-item:hover {
          background: #334155;
          border-color: #475569;
        }

        .query-text {
          max-width: 200px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .result-count {
          color: #94a3b8;
          font-size: 0.75rem;
        }
      `}</style>
    </div>
  );
};

export default RAGPanel;