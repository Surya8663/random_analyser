import React, { useState, useEffect } from 'react'
import { Toaster } from 'react-hot-toast'
import Navigation from './components/Navigation'
import DocumentUpload from './components/DocumentUpload'
import AgentTimeline from './components/AgentTimeline'
import BoundingBoxViewer from './components/BoundingBoxViewer'
import RAGPanel from './components/RAGPanel'
import RiskHeatmap from './components/RiskHeatmap'
import ResultsDashboard from './components/ResultsDashboard'
import { useWebSocket } from './hooks/useWebSocket'
import { useDocumentProcessing } from './hooks/useDocumentProcessing'
import { DocumentProvider } from './context/DocumentContext'

function App() {
  const [activeTab, setActiveTab] = useState('upload')
  const [documentId, setDocumentId] = useState(null)
  const [processingState, setProcessingState] = useState(null)
  
  // Initialize WebSocket connection
  const { connect, disconnect, sendMessage } = useWebSocket({
    onMessage: (data) => {
      if (data.type === 'processing_update') {
        setProcessingState(data.state)
      }
    }
  })
  
  // Initialize document processing hook
  const { 
    uploadDocument, 
    getResults, 
    queryDocument, 
    status, 
    results 
  } = useDocumentProcessing()
  
  useEffect(() => {
    // Connect to WebSocket when component mounts
    connect()
    
    // Cleanup on unmount
    return () => {
      disconnect()
    }
  }, [])
  
  const handleDocumentUpload = async (file) => {
    try {
      const response = await uploadDocument(file)
      setDocumentId(response.document_id)
      setActiveTab('processing')
      
      // Send WebSocket message to start processing
      sendMessage({
        type: 'start_processing',
        document_id: response.document_id
      })
    } catch (error) {
      console.error('Upload failed:', error)
    }
  }
  
  const handleQuerySubmit = async (query) => {
    if (!documentId) return
    
    try {
      const response = await queryDocument(documentId, query)
      // Handle query results
      console.log('Query results:', response)
    } catch (error) {
      console.error('Query failed:', error)
    }
  }
  
  return (
    <DocumentProvider>
      <div className="min-h-screen bg-dark-bg text-dark-text">
        <Navigation 
          activeTab={activeTab}
          onTabChange={setActiveTab}
          documentId={documentId}
          status={status}
        />
        
        <main className="container mx-auto px-4 py-6">
          {activeTab === 'upload' && (
            <DocumentUpload onUpload={handleDocumentUpload} />
          )}
          
          {activeTab === 'processing' && documentId && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <BoundingBoxViewer 
                    documentId={documentId}
                    processingState={processingState}
                  />
                </div>
                <div>
                  <AgentTimeline 
                    processingState={processingState}
                    documentId={documentId}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <RiskHeatmap 
                  documentId={documentId}
                  processingState={processingState}
                />
                <RAGPanel 
                  documentId={documentId}
                  onQuerySubmit={handleQuerySubmit}
                />
              </div>
            </div>
          )}
          
          {activeTab === 'results' && documentId && (
            <ResultsDashboard 
              documentId={documentId}
              results={results}
            />
          )}
          
          {activeTab === 'benchmark' && (
            <div className="text-center py-12">
              <h2 className="text-2xl font-bold mb-4">Benchmark Dashboard</h2>
              <p className="text-dark-muted">
                Dataset-level evaluation coming soon
              </p>
            </div>
          )}
        </main>
        
        <Toaster 
          position="bottom-right"
          toastOptions={{
            style: {
              background: 'var(--card-bg)',
              color: 'var(--text)',
              border: '1px solid var(--border)',
            },
          }}
        />
      </div>
    </DocumentProvider>
  )
}

export default App