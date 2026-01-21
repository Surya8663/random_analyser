// src/components/AgentTimeline.jsx
import React, { useState, useEffect } from 'react';
import { FiEye, FiFileText, FiCpu, FiCheck, FiAlertCircle, FiClock, FiPlay, FiRefreshCw, FiGitMerge, FiZap } from 'react-icons/fi';
import axios from 'axios';

const AGENT_CONFIG = {
  vision_agent: {
    name: 'Vision Agent',
    icon: FiEye,
    color: '#8b5cf6',
    description: 'Analyzes visual elements, detects tables, signatures, logos'
  },
  text_agent: {
    name: 'Text Agent',
    icon: FiFileText,
    color: '#3b82f6',
    description: 'Extracts text, performs OCR, identifies entities'
  },
  fusion_agent: {
    name: 'Fusion Agent',
    icon: FiGitMerge,
    color: '#10b981',
    description: 'Aligns text and visual information, cross-modal analysis'
  },
  reasoning_agent: {
    name: 'Reasoning Agent',
    icon: FiCpu,
    color: '#f59e0b',
    description: 'Validates content, detects contradictions, assesses risk'
  }
};

const AgentTimeline = ({ documentId, onStartProcessing }) => {
  const [agents, setAgents] = useState([]);
  const [isPolling, setIsPolling] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isInstantMode, setIsInstantMode] = useState(false);
  const [pollingError, setPollingError] = useState(null);
  const [processingTime, setProcessingTime] = useState(0);
  const [parallelMode, setParallelMode] = useState(false);

  // Initialize agents from config
  useEffect(() => {
    const initialAgents = Object.keys(AGENT_CONFIG).map(agentId => ({
      ...AGENT_CONFIG[agentId],
      id: agentId,
      status: 'pending',
      progress: 0,
      startTime: null,
      endTime: null,
      duration: null,
      errors: [],
      fieldsExtracted: []
    }));
    setAgents(initialAgents);
  }, []);

  // Fetch status every 2 seconds when documentId changes
  useEffect(() => {
    if (!documentId) return;

    let pollInterval;
    let startTime = Date.now();

    const fetchStatus = async () => {
      try {
        const response = await axios.get(`http://127.0.0.1:8000/api/v1/status/${documentId}`);
        const status = response.data;
        
        setProcessingTime(Math.round((Date.now() - startTime) / 1000));
        
        // Check if parallel mode
        if (status.parallel_mode) {
          setParallelMode(true);
        }
        
        // Update agents based on status
        if (status.agents) {
          const updatedAgents = agents.map(agent => {
            const agentStatus = status.agents[agent.id];
            if (agentStatus) {
              return {
                ...agent,
                status: agentStatus.status || 'pending',
                progress: agentStatus.progress || 0,
                startTime: agentStatus.start_time || null,
                endTime: agentStatus.end_time || null,
                duration: agentStatus.duration || null,
                errors: agentStatus.errors || [],
                fieldsExtracted: agentStatus.fields_extracted || [],
                cached: agentStatus.cached || false
              };
            }
            return agent;
          });
          setAgents(updatedAgents);
        } else {
          // Update from processing_steps if available
          const updatedAgents = agents.map(agent => {
            const agentStatus = status.processing_steps?.find(step => step.agent === agent.id);
            return {
              ...agent,
              status: agentStatus?.status || 'pending',
              progress: agentStatus?.progress || 0,
              startTime: agentStatus?.start_time,
              endTime: agentStatus?.end_time,
              duration: agentStatus?.duration,
              errors: agentStatus?.errors || [],
              fieldsExtracted: agentStatus?.fields_extracted || []
            };
          });
          setAgents(updatedAgents);
        }
        
        // Check if processing is complete
        const allCompleted = agents.every(agent => 
          agent.status === 'completed' || agent.status === 'error'
        );
        
        if (status.status === 'completed' && isPolling) {
          setIsPolling(false);
          clearInterval(pollInterval);
          setIsInstantMode(status.cached || false);
          
          // If instant mode, show success message
          if (status.cached) {
            console.log('✅ Processing completed instantly from cache!');
          }
        }
        
        setPollingError(null);
      } catch (error) {
        console.error('Failed to fetch status:', error);
        setPollingError('Failed to fetch agent status');
      }
    };

    // Initial fetch
    fetchStatus();

    // Start polling if document is processing
    if (isPolling) {
      pollInterval = setInterval(fetchStatus, 1000); // Faster polling for better UX
    }

    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [documentId, isPolling, agents]);

  const handleStartProcessing = async () => {
    if (!documentId) return;
    
    setIsProcessing(true);
    setIsInstantMode(false);
    setParallelMode(false);
    setProcessingTime(0);
    
    try {
      const response = await axios.post(`http://127.0.0.1:8000/api/v1/process/${documentId}`);
      console.log('Processing started:', response.data);
      
      if (response.data.parallel_mode) {
        setParallelMode(true);
      }
      
      setIsPolling(true);
      if (onStartProcessing) {
        onStartProcessing();
      }
    } catch (error) {
      console.error('Failed to start processing:', error);
      alert(`Failed to start processing: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleInstantProcess = async () => {
    if (!documentId) return;
    
    setIsProcessing(true);
    setIsInstantMode(true);
    setParallelMode(true);
    setProcessingTime(0);
    
    try {
      // Try instant processing endpoint first
      const response = await axios.post(`http://127.0.0.1:8000/api/v1/process/instant/${documentId}`);
      console.log('Instant processing response:', response.data);
      
      if (response.data.instant || response.data.cached) {
        // Immediate results - update UI instantly
        setAgents(prev => prev.map(agent => ({
          ...agent,
          status: 'completed',
          progress: 100,
          startTime: new Date().toISOString(),
          endTime: new Date().toISOString(),
          duration: 0.1,
          cached: true
        })));
        
        setIsPolling(true); // Trigger one status check to get results
        
        // Show success message
        setTimeout(() => {
          alert('⚡ Processing completed instantly from cache!');
        }, 100);
      } else {
        // Fallback to normal processing
        await handleStartProcessing();
      }
    } catch (error) {
      console.error('Instant processing failed:', error);
      // Fallback to normal processing
      await handleStartProcessing();
    } finally {
      setIsProcessing(false);
    }
  };

  const getStatusIcon = (status, cached = false) => {
    if (cached) {
      return <FiZap className="text-yellow-500" />;
    }
    
    switch (status) {
      case 'completed':
        return <FiCheck className="text-green-500" />;
      case 'error':
        return <FiAlertCircle className="text-red-500" />;
      case 'processing':
      case 'running':
        return <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />;
      default:
        return <FiClock className="text-gray-500" />;
    }
  };

  const getStatusText = (status, cached = false) => {
    if (cached) return 'Cached';
    
    switch (status) {
      case 'completed': return 'Completed';
      case 'error': return 'Error';
      case 'processing': 
      case 'running': return 'Processing';
      default: return 'Pending';
    }
  };

  const getOverallProgress = () => {
    if (agents.length === 0) return 0;
    const totalProgress = agents.reduce((sum, agent) => sum + agent.progress, 0);
    return Math.round(totalProgress / agents.length);
  };

  const getCompletedAgents = () => {
    return agents.filter(agent => agent.status === 'completed').length;
  };

  return (
    <div className="panel agent-timeline">
      <div className="panel-header">
        <div>
          <h3>🤖 Agent Execution Timeline</h3>
          <p className="panel-subtitle">
            {parallelMode ? 'Parallel Processing' : 'Sequential Processing'} • {getCompletedAgents()}/{agents.length} agents complete
          </p>
        </div>
        
        <div className="processing-controls">
          {isInstantMode && (
            <div className="instant-badge">
              <FiZap /> Instant Mode
            </div>
          )}
          
          {documentId && (
            <div className="control-buttons">
              <button
                onClick={handleInstantProcess}
                disabled={isProcessing || isPolling}
                className="instant-process-btn"
                title="Try instant processing from cache"
              >
                <FiZap />
                <span>Instant Process</span>
              </button>
              
              <button
                onClick={handleStartProcessing}
                disabled={isProcessing || isPolling}
                className="start-process-btn"
                title="Start normal processing"
              >
                {isProcessing ? 'Starting...' : '▶️ Normal Process'}
              </button>
            </div>
          )}
        </div>
      </div>

      {pollingError && (
        <div className="error-banner">
          <FiAlertCircle />
          <span>{pollingError}</span>
        </div>
      )}

      {/* Overall progress bar */}
      <div className="overall-progress">
        <div className="progress-header">
          <span>Overall Progress</span>
          <span className="progress-text">{getOverallProgress()}%</span>
        </div>
        <div className="progress-container large">
          <div 
            className="progress-bar"
            style={{ 
              width: `${getOverallProgress()}%`,
              background: isInstantMode 
                ? 'linear-gradient(90deg, #f59e0b, #fbbf24)' 
                : 'linear-gradient(90deg, #3b82f6, #8b5cf6)'
            }}
          />
        </div>
        <div className="progress-stats">
          <div className="stat">
            <span className="stat-label">Time:</span>
            <span className="stat-value">{processingTime}s</span>
          </div>
          <div className="stat">
            <span className="stat-label">Mode:</span>
            <span className={`stat-value ${isInstantMode ? 'instant' : 'normal'}`}>
              {isInstantMode ? 'Instant' : parallelMode ? 'Parallel' : 'Normal'}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Status:</span>
            <span className={`stat-value ${isPolling ? 'processing' : 'idle'}`}>
              {isPolling ? 'Processing' : 'Ready'}
            </span>
          </div>
        </div>
      </div>

      <div className="agents-list">
        {agents.map((agent, index) => {
          const isInstant = agent.cached || isInstantMode;
          
          return (
            <div key={agent.id} className={`agent-item ${agent.status} ${isInstant ? 'instant' : ''}`}>
              <div className="agent-icon-container" style={{ 
                backgroundColor: `${agent.color}${isInstant ? '40' : '20'}`,
                border: isInstant ? `2px solid ${agent.color}` : 'none'
              }}>
                <agent.icon style={{ color: agent.color }} />
                {isInstant && (
                  <div className="instant-indicator">
                    <FiZap size={12} />
                  </div>
                )}
              </div>
              
              <div className="agent-content">
                <div className="agent-header">
                  <div className="agent-name-section">
                    <h4>{agent.name}</h4>
                    <div className="agent-status">
                      {getStatusIcon(agent.status, isInstant)}
                      <span className={isInstant ? 'instant-text' : ''}>
                        {getStatusText(agent.status, isInstant)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="agent-stats">
                    {agent.duration && (
                      <div className="agent-duration">
                        {agent.duration.toFixed(2)}s
                      </div>
                    )}
                    <div className="agent-progress-text">
                      {agent.progress}%
                    </div>
                  </div>
                </div>

                <p className="agent-description">{agent.description}</p>

                {/* Progress bar */}
                <div className="progress-container">
                  <div 
                    className="progress-bar"
                    style={{ 
                      width: `${agent.progress}%`,
                      backgroundColor: agent.color,
                      animation: isInstant ? 'pulse 2s infinite' : 'none'
                    }}
                  />
                </div>

                {/* Extracted fields */}
                {agent.fieldsExtracted && agent.fieldsExtracted.length > 0 && (
                  <div className="extracted-fields">
                    <small>Extracted: </small>
                    {agent.fieldsExtracted.slice(0, 3).map((field, idx) => (
                      <span key={idx} className="field-tag">
                        {field}
                      </span>
                    ))}
                    {agent.fieldsExtracted.length > 3 && (
                      <span className="field-tag">+{agent.fieldsExtracted.length - 3}</span>
                    )}
                  </div>
                )}

                {/* Errors */}
                {agent.errors && agent.errors.length > 0 && (
                  <div className="agent-errors">
                    <FiAlertCircle className="error-icon" />
                    <span>{agent.errors[0]}</span>
                  </div>
                )}

                {/* Timing info */}
                {(agent.startTime || agent.endTime) && (
                  <div className="timing-info">
                    {agent.startTime && (
                      <small>Started: {new Date(agent.startTime).toLocaleTimeString()}</small>
                    )}
                    {agent.endTime && (
                      <small> • Ended: {new Date(agent.endTime).toLocaleTimeString()}</small>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {!documentId && (
        <div className="no-document">
          <p>Upload a document to start processing</p>
        </div>
      )}

      {/* Cache info */}
      {isInstantMode && (
        <div className="cache-info">
          <div className="cache-icon">
            <FiZap />
          </div>
          <div className="cache-details">
            <h4>⚡ Instant Processing Active</h4>
            <p>Results loaded from cache. Processing completed in &lt;100ms.</p>
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
          flex-wrap: wrap;
          gap: 1rem;
        }

        .panel-header h3 {
          margin: 0;
          color: #e2e8f0;
          font-size: 1.2rem;
        }

        .panel-subtitle {
          color: #94a3b8;
          font-size: 0.875rem;
          margin: 0.25rem 0 0 0;
        }

        .processing-controls {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .instant-badge {
          background: rgba(245, 158, 11, 0.2);
          color: #f59e0b;
          padding: 0.25rem 0.75rem;
          border-radius: 1rem;
          font-size: 0.75rem;
          font-weight: 600;
          display: flex;
          align-items: center;
          gap: 0.25rem;
          align-self: flex-end;
        }

        .control-buttons {
          display: flex;
          gap: 0.5rem;
        }

        .instant-process-btn {
          background: linear-gradient(90deg, #f59e0b, #fbbf24);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
          transition: all 0.2s;
        }

        .instant-process-btn:hover:not(:disabled) {
          background: linear-gradient(90deg, #d97706, #f59e0b);
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
        }

        .instant-process-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .start-process-btn {
          background: #10b981;
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
        }

        .start-process-btn:hover:not(:disabled) {
          background: #059669;
        }

        .start-process-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .error-banner {
          background: #fee2e2;
          border: 1px solid #fecaca;
          color: #dc2626;
          padding: 0.75rem;
          border-radius: 6px;
          margin-bottom: 1rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
        }

        .overall-progress {
          background: #0f172a;
          border-radius: 8px;
          padding: 1rem;
          margin-bottom: 1.5rem;
          border: 1px solid #334155;
        }

        .progress-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.75rem;
        }

        .progress-header span {
          color: #e2e8f0;
          font-weight: 500;
        }

        .progress-text {
          color: #3b82f6;
          font-weight: 700;
        }

        .progress-container {
          height: 6px;
          background: #334155;
          border-radius: 3px;
          margin: 0.75rem 0;
          overflow: hidden;
        }

        .progress-container.large {
          height: 10px;
        }

        .progress-bar {
          height: 100%;
          border-radius: 3px;
          transition: width 0.3s ease;
        }

        .progress-stats {
          display: flex;
          gap: 1.5rem;
          margin-top: 1rem;
        }

        .stat {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .stat-label {
          color: #94a3b8;
          font-size: 0.75rem;
        }

        .stat-value {
          color: #e2e8f0;
          font-weight: 600;
          font-size: 0.875rem;
        }

        .stat-value.instant {
          color: #f59e0b;
        }

        .stat-value.normal {
          color: #10b981;
        }

        .stat-value.processing {
          color: #3b82f6;
        }

        .stat-value.idle {
          color: #94a3b8;
        }

        .agents-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .agent-item {
          display: flex;
          gap: 1rem;
          padding: 1rem;
          background: #0f172a;
          border-radius: 8px;
          border-left: 4px solid #334155;
          transition: all 0.2s;
        }

        .agent-item.instant {
          border-left-color: #f59e0b;
          background: rgba(245, 158, 11, 0.05);
        }

        .agent-item.completed {
          border-left-color: #10b981;
        }

        .agent-item.processing,
        .agent-item.running {
          border-left-color: #3b82f6;
        }

        .agent-item.error {
          border-left-color: #ef4444;
        }

        .agent-icon-container {
          width: 48px;
          height: 48px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          position: relative;
        }

        .agent-icon-container svg {
          width: 24px;
          height: 24px;
        }

        .instant-indicator {
          position: absolute;
          top: -6px;
          right: -6px;
          background: #f59e0b;
          color: white;
          border-radius: 50%;
          width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          border: 2px solid #1e293b;
        }

        .agent-content {
          flex: 1;
        }

        .agent-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .agent-name-section {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .agent-name-section h4 {
          margin: 0;
          color: #e2e8f0;
          font-size: 1rem;
        }

        .agent-status {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          font-size: 0.875rem;
        }

        .instant-text {
          color: #f59e0b;
          font-weight: 600;
        }

        .agent-stats {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .agent-duration {
          font-size: 0.875rem;
          color: #94a3b8;
          background: rgba(148, 163, 184, 0.1);
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
        }

        .agent-progress-text {
          font-size: 0.875rem;
          font-weight: 600;
          color: #3b82f6;
          min-width: 40px;
          text-align: right;
        }

        .agent-description {
          color: #94a3b8;
          font-size: 0.875rem;
          margin: 0.5rem 0;
        }

        .extracted-fields {
          display: flex;
          flex-wrap: wrap;
          gap: 0.25rem;
          margin-top: 0.5rem;
          align-items: center;
        }

        .extracted-fields small {
          color: #64748b;
          font-size: 0.75rem;
        }

        .field-tag {
          background: #334155;
          color: #cbd5e1;
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
        }

        .agent-errors {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #ef4444;
          font-size: 0.875rem;
          margin-top: 0.5rem;
        }

        .error-icon {
          flex-shrink: 0;
        }

        .timing-info {
          margin-top: 0.5rem;
          color: #64748b;
          font-size: 0.75rem;
        }

        .no-document {
          text-align: center;
          padding: 2rem;
          color: #64748b;
          font-style: italic;
        }

        .cache-info {
          background: rgba(245, 158, 11, 0.1);
          border: 1px solid rgba(245, 158, 11, 0.3);
          border-radius: 8px;
          padding: 1rem;
          margin-top: 1.5rem;
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .cache-icon {
          background: #f59e0b;
          color: white;
          width: 40px;
          height: 40px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .cache-details h4 {
          margin: 0 0 0.25rem 0;
          color: #f59e0b;
          font-size: 0.875rem;
        }

        .cache-details p {
          margin: 0;
          color: #94a3b8;
          font-size: 0.875rem;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }

        @media (max-width: 768px) {
          .panel-header {
            flex-direction: column;
          }
          
          .processing-controls {
            width: 100%;
          }
          
          .control-buttons {
            flex-direction: column;
            width: 100%;
          }
          
          .instant-process-btn,
          .start-process-btn {
            width: 100%;
            justify-content: center;
          }
          
          .agent-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
          }
          
          .agent-stats {
            width: 100%;
            justify-content: space-between;
          }
        }
      `}</style>
    </div>
  );
};

export default AgentTimeline;