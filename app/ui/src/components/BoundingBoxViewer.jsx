import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiZoomIn, FiZoomOut, FiMaximize, FiInfo, FiBox } from 'react-icons/fi'
import html2canvas from 'html2canvas'
import { Tooltip } from 'react-tooltip'

const BoundingBoxViewer = ({ documentId, processingState }) => {
  const [scale, setScale] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [selectedBox, setSelectedBox] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [showLabels, setShowLabels] = useState(true)
  const [showConfidence, setShowConfidence] = useState(true)
  
  const containerRef = useRef(null)
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  
  const [documentImage, setDocumentImage] = useState(null)
  const [boundingBoxes, setBoundingBoxes] = useState([])
  const [agentColors, setAgentColors] = useState({})
  
  // Load document image and bounding boxes
  useEffect(() => {
    if (!documentId) return
    
    // In production, this would fetch from API
    // For now, use mock data
    const mockImage = `https://picsum.photos/800/1000?random=${documentId}`
    setDocumentImage(mockImage)
    
    // Generate mock bounding boxes
    const mockBoxes = generateMockBoundingBoxes()
    setBoundingBoxes(mockBoxes)
    
    // Set agent colors
    const colors = {
      vision_agent: '#8b5cf6',
      text_agent: '#3b82f6',
      fusion_agent: '#10b981',
      reasoning_agent: '#f59e0b'
    }
    setAgentColors(colors)
  }, [documentId])
  
  // Update boxes from processing state
  useEffect(() => {
    if (!processingState || !processingState.visual_elements) return
    
    const boxes = processingState.visual_elements.map(element => ({
      id: `box_${element.id || Math.random()}`,
      x: element.bbox[0] * 100, // Convert to percentage
      y: element.bbox[1] * 100,
      width: (element.bbox[2] - element.bbox[0]) * 100,
      height: (element.bbox[3] - element.bbox[1]) * 100,
      agent: element.agent || 'vision_agent',
      confidence: element.confidence || 0.8,
      type: element.element_type || 'unknown',
      text: element.text_content || '',
      metadata: element.metadata || {}
    }))
    
    setBoundingBoxes(boxes)
  }, [processingState])
  
  const generateMockBoundingBoxes = () => {
    const boxes = []
    const agents = ['vision_agent', 'text_agent', 'fusion_agent', 'reasoning_agent']
    const types = ['table', 'signature', 'text_block', 'logo', 'header', 'paragraph']
    
    for (let i = 0; i < 15; i++) {
      boxes.push({
        id: `box_${i}`,
        x: Math.random() * 70 + 10,
        y: Math.random() * 70 + 10,
        width: Math.random() * 30 + 10,
        height: Math.random() * 20 + 5,
        agent: agents[Math.floor(Math.random() * agents.length)],
        confidence: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
        type: types[Math.floor(Math.random() * types.length)],
        text: `Sample text content for box ${i}`,
        metadata: {
          page: Math.floor(Math.random() * 3) + 1,
          semantic_label: 'important_element'
        }
      })
    }
    
    return boxes
  }
  
  const handleZoomIn = () => {
    setScale(prev => Math.min(prev * 1.2, 5))
  }
  
  const handleZoomOut = () => {
    setScale(prev => Math.max(prev / 1.2, 0.2))
  }
  
  const handleReset = () => {
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }
  
  const handleMouseDown = (e) => {
    if (e.target === containerRef.current) {
      setIsDragging(true)
    }
  }
  
  const handleMouseMove = (e) => {
    if (!isDragging) return
    
    setPosition(prev => ({
      x: prev.x + e.movementX,
      y: prev.y + e.movementY
    }))
  }
  
  const handleMouseUp = () => {
    setIsDragging(false)
  }
  
  const handleBoxClick = (box) => {
    setSelectedBox(box)
  }
  
  const exportImage = async () => {
    if (!containerRef.current) return
    
    try {
      const canvas = await html2canvas(containerRef.current, {
        backgroundColor: null,
        scale: 2
      })
      
      const link = document.createElement('a')
      link.download = `document_${documentId}_analysis.png`
      link.href = canvas.toDataURL('image/png')
      link.click()
    } catch (error) {
      console.error('Export failed:', error)
    }
  }
  
  const getBoxStyle = (box) => {
    const agentColor = agentColors[box.agent] || '#6b7280'
    const opacity = showConfidence ? box.confidence : 0.7
    
    return {
      left: `${box.x}%`,
      top: `${box.y}%`,
      width: `${box.width}%`,
      height: `${box.height}%`,
      borderColor: agentColor,
      backgroundColor: `${agentColor}${Math.floor(opacity * 50).toString(16).padStart(2, '0')}`,
      opacity: opacity
    }
  }
  
  const getAgentIcon = (agent) => {
    const icons = {
      vision_agent: '👁️',
      text_agent: '📝',
      fusion_agent: '🔄',
      reasoning_agent: '🤖'
    }
    return icons[agent] || '📦'
  }
  
  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold gradient-text">Multi-Modal Document Analysis</h2>
          <p className="text-sm text-dark-muted">
            Visual bounding boxes with agent attribution
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowLabels(!showLabels)}
            className={`px-3 py-2 rounded-lg transition-colors ${showLabels ? 'bg-primary-500' : 'bg-dark-card'}`}
          >
            <FiInfo className="w-5 h-5" />
          </button>
          
          <button
            onClick={() => setShowConfidence(!showConfidence)}
            className={`px-3 py-2 rounded-lg transition-colors ${showConfidence ? 'bg-primary-500' : 'bg-dark-card'}`}
          >
            <FiBox className="w-5 h-5" />
          </button>
          
          <button
            onClick={handleZoomIn}
            className="px-3 py-2 rounded-lg bg-dark-card hover:bg-dark-border transition-colors"
          >
            <FiZoomIn className="w-5 h-5" />
          </button>
          
          <button
            onClick={handleZoomOut}
            className="px-3 py-2 rounded-lg bg-dark-card hover:bg-dark-border transition-colors"
          >
            <FiZoomOut className="w-5 h-5" />
          </button>
          
          <button
            onClick={handleReset}
            className="px-3 py-2 rounded-lg bg-dark-card hover:bg-dark-border transition-colors"
          >
            <FiMaximize className="w-5 h-5" />
          </button>
          
          <button
            onClick={exportImage}
            className="px-4 py-2 rounded-lg bg-primary-500 hover:bg-primary-600 transition-colors font-medium"
          >
            Export
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main viewer */}
        <div className="lg:col-span-3">
          <div
            ref={containerRef}
            className="relative w-full h-[600px] bg-dark-card rounded-lg overflow-hidden border border-dark-border"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{
              cursor: isDragging ? 'grabbing' : 'grab'
            }}
          >
            {/* Document image */}
            {documentImage && (
              <motion.img
                ref={imageRef}
                src={documentImage}
                alt="Document"
                className="absolute top-0 left-0 w-full h-full object-contain"
                style={{
                  transform: `scale(${scale}) translate(${position.x}px, ${position.y}px)`,
                  transformOrigin: 'center'
                }}
              />
            )}
            
            {/* Bounding boxes */}
            <div className="absolute top-0 left-0 w-full h-full">
              {boundingBoxes.map((box) => (
                <motion.div
                  key={box.id}
                  className="absolute border-2 cursor-pointer transition-all hover:border-4 hover:z-10"
                  style={getBoxStyle(box)}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: box.confidence }}
                  transition={{ duration: 0.3 }}
                  onClick={() => handleBoxClick(box)}
                  data-tooltip-id="box-tooltip"
                  data-tooltip-content={`${box.type} (${box.agent}) - ${Math.round(box.confidence * 100)}%`}
                >
                  {showLabels && (
                    <div className="absolute -top-6 left-0 flex items-center gap-1">
                      <span className="text-xs font-medium px-2 py-1 rounded bg-dark-card">
                        {getAgentIcon(box.agent)} {box.type}
                      </span>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
            
            {/* Zoom and position indicator */}
            <div className="absolute bottom-4 right-4 px-3 py-2 rounded-lg bg-dark-card/80 backdrop-blur-sm">
              <div className="text-sm">
                Zoom: {(scale * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
        
        {/* Side panel */}
        <div className="lg:col-span-1">
          <div className="space-y-6">
            {/* Agent legend */}
            <div className="glass rounded-lg p-4">
              <h3 className="font-semibold mb-3">Agent Colors</h3>
              <div className="space-y-2">
                {Object.entries(agentColors).map(([agent, color]) => (
                  <div key={agent} className="flex items-center gap-2">
                    <div 
                      className="w-4 h-4 rounded"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-sm capitalize">
                      {agent.replace('_agent', '')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Selected box details */}
            <AnimatePresence>
              {selectedBox && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className="glass rounded-lg p-4"
                >
                  <h3 className="font-semibold mb-3">Selected Element</h3>
                  
                  <div className="space-y-3">
                    <div>
                      <div className="text-sm text-dark-muted">Type</div>
                      <div className="font-medium capitalize">{selectedBox.type}</div>
                    </div>
                    
                    <div>
                      <div className="text-sm text-dark-muted">Agent</div>
                      <div className="font-medium capitalize">
                        {selectedBox.agent.replace('_agent', '')}
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-sm text-dark-muted">Confidence</div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-2 bg-dark-border rounded-full overflow-hidden">
                          <div 
                            className="h-full rounded-full"
                            style={{ 
                              width: `${selectedBox.confidence * 100}%`,
                              backgroundColor: agentColors[selectedBox.agent]
                            }}
                          />
                        </div>
                        <span className="text-sm font-mono">
                          {(selectedBox.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-sm text-dark-muted">Position</div>
                      <div className="text-sm font-mono">
                        ({selectedBox.x.toFixed(1)}%, {selectedBox.y.toFixed(1)}%)
                      </div>
                    </div>
                    
                    {selectedBox.text && (
                      <div>
                        <div className="text-sm text-dark-muted">Content</div>
                        <div className="text-sm mt-1 p-2 rounded bg-dark-border">
                          {selectedBox.text.slice(0, 100)}
                          {selectedBox.text.length > 100 && '...'}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Statistics */}
            <div className="glass rounded-lg p-4">
              <h3 className="font-semibold mb-3">Statistics</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-dark-muted">Total Elements</span>
                  <span className="font-medium">{boundingBoxes.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-dark-muted">Avg Confidence</span>
                  <span className="font-medium">
                    {boundingBoxes.length > 0 
                      ? `${(boundingBoxes.reduce((sum, b) => sum + b.confidence, 0) / boundingBoxes.length * 100).toFixed(1)}%`
                      : '0%'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-dark-muted">Agents Involved</span>
                  <span className="font-medium">
                    {new Set(boundingBoxes.map(b => b.agent)).size}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <Tooltip id="box-tooltip" />
    </div>
  )
}

export default BoundingBoxViewer