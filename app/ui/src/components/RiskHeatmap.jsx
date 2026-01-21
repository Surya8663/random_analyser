import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  FiAlertTriangle, 
  FiCheckCircle, 
  FiAlertCircle,
  FiActivity,
  FiBarChart2,
  FiTrendingUp
} from 'react-icons/fi'
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts'

const RiskHeatmap = ({ documentId, processingState }) => {
  const [riskData, setRiskData] = useState([])
  const [contradictions, setContradictions] = useState([])
  const [selectedPage, setSelectedPage] = useState(1)
  const [heatmapMode, setHeatmapMode] = useState('risk') // 'risk', 'confidence', 'contradictions'
  
  // Generate mock data
  useEffect(() => {
    if (!documentId) return
    
    // Generate mock risk data
    const mockRiskData = []
    for (let i = 1; i <= 10; i++) {
      mockRiskData.push({
        page: i,
        riskScore: Math.random() * 0.8 + 0.2, // 0.2 to 1.0
        confidence: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
        contradictions: Math.floor(Math.random() * 5),
        extractedFields: Math.floor(Math.random() * 15) + 5,
        processingTime: Math.random() * 5 + 2 // 2 to 7 seconds
      })
    }
    setRiskData(mockRiskData)
    
    // Generate mock contradictions
    const mockContradictions = []
    const contradictionTypes = [
      'Numeric Inconsistency',
      'Date Mismatch',
      'Signature Absence',
      'Format Issue',
      'Data Type Mismatch'
    ]
    
    for (let i = 0; i < 8; i++) {
      mockContradictions.push({
        id: `contradiction_${i}`,
        type: contradictionTypes[Math.floor(Math.random() * contradictionTypes.length)],
        severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        page: Math.floor(Math.random() * 10) + 1,
        fieldA: `Field ${Math.floor(Math.random() * 10)}`,
        fieldB: `Field ${Math.floor(Math.random() * 10)}`,
        explanation: `Found inconsistency between ${['amounts', 'dates', 'signatures', 'formats'][Math.floor(Math.random() * 4)]}`,
        confidence: Math.random() * 0.4 + 0.6, // 0.6 to 1.0
        recommendation: 'Verify with source document'
      })
    }
    setContradictions(mockContradictions)
  }, [documentId])
  
  // Update from processing state
  useEffect(() => {
    if (!processingState) return
    
    if (processingState.risk_score !== undefined) {
      // Update risk data from processing state
      const newRiskData = riskData.map(item => {
        if (item.page === selectedPage) {
          return {
            ...item,
            riskScore: processingState.risk_score || item.riskScore,
            confidence: processingState.confidence || item.confidence
          }
        }
        return item
      })
      setRiskData(newRiskData)
    }
    
    if (processingState.contradictions) {
      setContradictions(processingState.contradictions.slice(0, 8))
    }
  }, [processingState, selectedPage])
  
  const getRiskLevel = (score) => {
    if (score >= 0.7) return 'critical'
    if (score >= 0.5) return 'high'
    if (score >= 0.3) return 'medium'
    return 'low'
  }
  
  const getRiskColor = (score) => {
    const level = getRiskLevel(score)
    switch (level) {
      case 'critical': return '#ef4444'
      case 'high': return '#f59e0b'
      case 'medium': return '#3b82f6'
      case 'low': return '#10b981'
      default: return '#6b7280'
    }
  }
  
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return '#ef4444'
      case 'medium': return '#f59e0b'
      case 'low': return '#3b82f6'
      default: return '#6b7280'
    }
  }
  
  const getHeatmapData = () => {
    return riskData.map(item => ({
      page: `Page ${item.page}`,
      risk: item.riskScore * 100,
      confidence: item.confidence * 100,
      contradictions: item.contradictions * 20,
      fields: item.extractedFields
    }))
  }
  
  const currentPageData = riskData.find(item => item.page === selectedPage)
  const pageContradictions = contradictions.filter(c => c.page === selectedPage)
  
  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold gradient-text">Risk Heatmap & Analysis</h2>
          <p className="text-sm text-dark-muted">
            Visual risk assessment across document pages
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex rounded-lg bg-dark-card p-1">
            {['risk', 'confidence', 'contradictions'].map((mode) => (
              <button
                key={mode}
                onClick={() => setHeatmapMode(mode)}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  heatmapMode === mode 
                    ? 'bg-primary-500' 
                    : 'hover:bg-dark-border'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="lg:col-span-2">
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={getHeatmapData()}
                margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="page" 
                  stroke="#94a3b8"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#94a3b8"
                  fontSize={12}
                  label={{
                    value: heatmapMode === 'risk' ? 'Risk Score (%)' : 
                           heatmapMode === 'confidence' ? 'Confidence (%)' : 'Contradictions',
                    angle: -90,
                    position: 'insideLeft',
                    style: { fill: '#94a3b8' }
                  }}
                />
                <RechartsTooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    borderColor: '#334155',
                    color: '#f1f5f9'
                  }}
                  formatter={(value) => [`${value.toFixed(1)}%`, heatmapMode]}
                />
                <Legend />
                
                {heatmapMode === 'risk' && (
                  <Bar 
                    dataKey="risk" 
                    name="Risk Score" 
                    fill="#ef4444"
                    barSize={20}
                  >
                    {getHeatmapData().map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={getRiskColor(entry.risk / 100)}
                        stroke={getRiskColor(entry.risk / 100)}
                        strokeWidth={2}
                      />
                    ))}
                  </Bar>
                )}
                
                {heatmapMode === 'confidence' && (
                  <Bar 
                    dataKey="confidence" 
                    name="Confidence" 
                    fill="#3b82f6"
                    barSize={20}
                  />
                )}
                
                {heatmapMode === 'contradictions' && (
                  <Bar 
                    dataKey="contradictions" 
                    name="Contradictions" 
                    fill="#f59e0b"
                    barSize={20}
                  />
                )}
                
                <Line
                  type="monotone"
                  dataKey="fields"
                  name="Extracted Fields"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Page selector */}
          <div className="mt-4">
            <div className="text-sm text-dark-muted mb-2">Select Page for Detail View</div>
            <div className="flex flex-wrap gap-2">
              {riskData.map((item) => {
                const riskLevel = getRiskLevel(item.riskScore)
                const isSelected = selectedPage === item.page
                
                return (
                  <button
                    key={item.page}
                    onClick={() => setSelectedPage(item.page)}
                    className={`relative px-3 py-2 rounded-lg transition-all ${
                      isSelected 
                        ? 'ring-2 ring-primary-500' 
                        : 'hover:bg-dark-border'
                    }`}
                    style={{
                      backgroundColor: isSelected 
                        ? `${getRiskColor(item.riskScore)}20` 
                        : '#1e293b'
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <div className="text-sm font-medium">Page {item.page}</div>
                      <div 
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: getRiskColor(item.riskScore) }}
                      />
                    </div>
                    <div className="text-xs text-dark-muted mt-1">
                      Risk: {(item.riskScore * 100).toFixed(0)}%
                    </div>
                  </button>
                )
              })}
            </div>
          </div>
        </div>
        
        {/* Current page analysis */}
        <div className="space-y-6">
          {currentPageData && (
            <div className="glass rounded-lg p-4">
              <h3 className="font-semibold mb-3">Page {selectedPage} Analysis</h3>
              
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-sm text-dark-muted">Risk Level</div>
                    <div className="text-sm font-bold" style={{ color: getRiskColor(currentPageData.riskScore) }}>
                      {getRiskLevel(currentPageData.riskScore).toUpperCase()}
                    </div>
                  </div>
                  <div className="h-2 bg-dark-border rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full transition-all duration-300"
                      style={{ 
                        width: `${currentPageData.riskScore * 100}%`,
                        backgroundColor: getRiskColor(currentPageData.riskScore)
                      }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-sm text-dark-muted">Confidence</div>
                    <div className="text-sm font-bold text-primary-500">
                      {(currentPageData.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="h-2 bg-dark-border rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full bg-primary-500 transition-all duration-300"
                      style={{ width: `${currentPageData.confidence * 100}%` }}
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3 pt-3">
                  <div className="text-center p-3 rounded-lg bg-dark-border">
                    <div className="text-2xl font-bold">{currentPageData.contradictions}</div>
                    <div className="text-xs text-dark-muted">Contradictions</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-dark-border">
                    <div className="text-2xl font-bold">{currentPageData.extractedFields}</div>
                    <div className="text-xs text-dark-muted">Fields Extracted</div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Risk indicators */}
          <div className="glass rounded-lg p-4">
            <h3 className="font-semibold mb-3">Risk Indicators</h3>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <span className="text-sm">Critical Risk</span>
                </div>
                <span className="text-sm text-dark-muted">≥ 70%</span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-amber-500" />
                  <span className="text-sm">High Risk</span>
                </div>
                <span className="text-sm text-dark-muted">50-69%</span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <span className="text-sm">Medium Risk</span>
                </div>
                <span className="text-sm text-dark-muted">30-49%</span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  <span className="text-sm">Low Risk</span>
                </div>
                <span className="text-sm text-dark-muted">≤ 29%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Contradictions list */}
      {pageContradictions.length > 0 && (
        <div className="mt-6 pt-6 border-t border-dark-border">
          <h3 className="font-semibold mb-3">Contradictions on Page {selectedPage}</h3>
          
          <div className="space-y-3">
            {pageContradictions.map((contradiction, index) => (
              <motion.div
                key={contradiction.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="p-3 rounded-lg bg-dark-border"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div 
                      className="px-2 py-1 rounded text-xs font-bold"
                      style={{ 
                        backgroundColor: `${getSeverityColor(contradiction.severity)}20`,
                        color: getSeverityColor(contradiction.severity)
                      }}
                    >
                      {contradiction.severity.toUpperCase()}
                    </div>
                    <span className="text-sm font-medium">{contradiction.type}</span>
                  </div>
                  
                  <div className="text-sm text-dark-muted">
                    Confidence: {(contradiction.confidence * 100).toFixed(0)}%
                  </div>
                </div>
                
                <div className="text-sm mb-2">{contradiction.explanation}</div>
                
                <div className="text-xs text-dark-muted">
                  Fields: {contradiction.fieldA} vs {contradiction.fieldB}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default RiskHeatmap