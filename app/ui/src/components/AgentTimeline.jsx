import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  FiEye, 
  FiFileText, 
  FiMerge, 
  FiCpu, 
  FiCheck, 
  FiAlertCircle,
  FiClock,
  FiPlay
} from 'react-icons/fi'
import { Tooltip } from 'react-tooltip'

const AGENT_CONFIG = {
  vision_agent: {
    name: 'Vision Agent',
    icon: FiEye,
    color: '#8b5cf6', // Purple
    description: 'Analyzes visual elements, detects tables, signatures, logos'
  },
  text_agent: {
    name: 'Text Agent',
    icon: FiFileText,
    color: '#3b82f6', // Blue
    description: 'Extracts text, performs OCR, identifies entities'
  },
  fusion_agent: {
    name: 'Fusion Agent',
    icon: FiMerge,
    color: '#10b981', // Green
    description: 'Aligns text and visual information, cross-modal analysis'
  },
  reasoning_agent: {
    name: 'Reasoning Agent',
    icon: FiCpu,
    color: '#f59e0b', // Amber
    description: 'Validates content, detects contradictions, assesses risk'
  }
}

const AgentTimeline = ({ processingState, documentId }) => {
  const [agents, setAgents] = useState([])
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  
  // Parse processing state to extract agent timeline
  useEffect(() => {
    if (!processingState) return
    
    const agentData = []
    
    // Extract agent execution data from processing state
    if (processingState.agent_timeline) {
      Object.entries(processingState.agent_timeline).forEach(([agentName, timeline]) => {
        const config = AGENT_CONFIG[agentName] || {
          name: agentName.replace('_', ' '),
          icon: FiPlay,
          color: '#6b7280',
          description: 'Processes document'
        }
        
        agentData.push({
          ...config,
          id: agentName,
          startTime: timeline.start_time,
          endTime: timeline.end_time,
          duration: timeline.duration,
          status: timeline.status || 'completed',
          fieldsExtracted: timeline.fields_extracted || [],
          errors: timeline.errors || []
        })
      })
    }
    
    // Sort by start time
    agentData.sort((a, b) => new Date(a.startTime) - new Date(b.startTime))
    setAgents(agentData)
    
    // Calculate total timeline duration
    if (agentData.length > 0) {
      const lastAgent = agentData[agentData.length - 1]
      const totalDuration = agentData.reduce((max, agent) => 
        Math.max(max, agent.duration || 0), 0
      )
      setCurrentTime(Math.min(totalDuration, 100))
    }
  }, [processingState])
  
  // Auto-play animation
  useEffect(() => {
    if (!isPlaying || agents.length === 0) return
    
    const interval = setInterval(() => {
      setCurrentTime(prev => {
        const maxTime = agents.reduce((max, agent) => 
          Math.max(max, agent.duration || 100), 100
        )
        return prev >= maxTime ? 0 : prev + 1
      })
    }, 100)
    
    return () => clearInterval(interval)
  }, [isPlaying, agents])
  
  const getAgentStatus = (agent) => {
    if (agent.errors && agent.errors.length > 0) {
      return 'error'
    }
    if (agent.endTime) {
      return 'completed'
    }
    if (agent.startTime) {
      return 'processing'
    }
    return 'pending'
  }
  
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <FiCheck className="text-green-500" />
      case 'error':
        return <FiAlertCircle className="text-red-500" />
      case 'processing':
        return <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />
      default:
        return <FiClock className="text-gray-500" />
    }
  }
  
  const getProgressPosition = (agent) => {
    if (!agent.startTime || !agent.duration) return 0
    
    const startPercent = (new Date(agent.startTime) - new Date(agents[0].startTime)) / 1000
    const currentPercent = (currentTime - startPercent) / agent.duration
    
    return Math.max(0, Math.min(100, currentPercent * 100))
  }
  
  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold gradient-text">Agent Execution Timeline</h2>
          <p className="text-sm text-dark-muted">Real-time visualization of AI agent workflow</p>
        </div>
        
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-dark-card hover:bg-dark-border transition-colors"
        >
          {isPlaying ? 'Pause' : 'Play'} Animation
        </button>
      </div>
      
      <div className="space-y-4">
        <AnimatePresence>
          {agents.map((agent, index) => {
            const status = getAgentStatus(agent)
            const progress = getProgressPosition(agent)
            const AgentIcon = agent.icon
            
            return (
              <motion.div
                key={agent.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="relative"
              >
                {/* Timeline connector */}
                {index < agents.length - 1 && (
                  <div 
                    className="absolute left-5 top-10 w-0.5 h-full bg-dark-border z-0"
                    style={{ height: 'calc(100% + 1rem)' }}
                  />
                )}
                
                <div className="relative z-10 flex items-start gap-4 p-4 rounded-lg bg-dark-card hover:bg-dark-border transition-colors">
                  {/* Agent icon */}
                  <div 
                    className="flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${agent.color}20` }}
                  >
                    <AgentIcon className="w-6 h-6" style={{ color: agent.color }} />
                  </div>
                  
                  {/* Agent info */}
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold">{agent.name}</h3>
                        {getStatusIcon(status)}
                      </div>
                      
                      <div className="text-sm text-dark-muted">
                        {agent.duration ? `${agent.duration.toFixed(2)}s` : 'Pending'}
                      </div>
                    </div>
                    
                    {/* Progress bar */}
                    <div className="h-2 bg-dark-border rounded-full overflow-hidden mb-3">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: agent.color }}
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    
                    {/* Agent details */}
                    <div className="text-sm text-dark-muted mb-2">
                      {agent.description}
                    </div>
                    
                    {/* Extracted fields */}
                    {agent.fieldsExtracted && agent.fieldsExtracted.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-2">
                        {agent.fieldsExtracted.slice(0, 3).map((field, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 text-xs rounded bg-dark-border"
                            data-tooltip-id="field-tooltip"
                            data-tooltip-content={field}
                          >
                            {field.length > 15 ? `${field.slice(0, 15)}...` : field}
                          </span>
                        ))}
                        {agent.fieldsExtracted.length > 3 && (
                          <span className="px-2 py-1 text-xs rounded bg-dark-border">
                            +{agent.fieldsExtracted.length - 3} more
                          </span>
                        )}
                      </div>
                    )}
                    
                    {/* Errors */}
                    {agent.errors && agent.errors.length > 0 && (
                      <div className="mt-2 p-2 rounded bg-red-500/10 border border-red-500/20">
                        <p className="text-xs text-red-400">
                          {agent.errors[0].slice(0, 100)}...
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
      
      {/* Timeline controls */}
      <div className="mt-6 pt-6 border-t border-dark-border">
        <div className="flex items-center justify-between">
          <div className="text-sm text-dark-muted">
            {agents.length} agents executed • {agents.filter(a => getAgentStatus(a) === 'completed').length} completed
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-sm">
              <span className="text-dark-muted">Current time: </span>
              <span className="font-mono">{currentTime.toFixed(1)}s</span>
            </div>
            
            <input
              type="range"
              min="0"
              max="100"
              value={currentTime}
              onChange={(e) => setCurrentTime(parseFloat(e.target.value))}
              className="w-32 accent-primary-500"
            />
          </div>
        </div>
      </div>
      
      <Tooltip id="field-tooltip" />
    </div>
  )
}

export default AgentTimeline