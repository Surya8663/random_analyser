// src/App.jsx - UPDATED
import React, { useState, useEffect } from 'react';
import UploadPanel from './components/UploadPanel';
import AgentTimeline from './components/AgentTimeline';
import RAGPanel from './components/RAGPanel';
import './App.css';
import axios from 'axios';

function App() {
  const [activeDocumentId, setActiveDocumentId] = useState(null);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);
  const [processingState, setProcessingState] = useState(null);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');

  // Check backend connection on startup
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/health');
        console.log('✅ Backend connected:', response.data);
        setIsBackendConnected(true);
        setBackendError(null);
        
        // Load existing documents
        try {
          const docsResponse = await axios.get('http://127.0.0.1:8000/api/v1/documents');
          setUploadedDocuments(docsResponse.data.documents || []);
        } catch (docsError) {
          console.log('No existing documents found');
        }
      } catch (error) {
        console.error('❌ Backend connection failed:', error.message);
        setIsBackendConnected(false);
        setBackendError('Backend connection failed. Make sure FastAPI is running on http://127.0.0.1:8000');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleUploadSuccess = (response) => {
    console.log('✅ Upload successful:', response);
    const documentId = response.document_id;
    
    if (documentId) {
      setActiveDocumentId(documentId);
      setProcessingState({ status: 'uploaded', message: 'Document uploaded successfully' });
      
      // Add to uploaded documents list
      setUploadedDocuments(prev => [
        ...prev,
        {
          document_id: documentId,
          filename: response.filename,
          uploaded_at: new Date().toISOString(),
          file_size: response.file_size
        }
      ]);
      
      // Switch to processing tab
      setActiveTab('processing');
    } else {
      console.error('No document_id in response:', response);
      alert('Upload succeeded but no document ID received');
    }
  };

  const handleStartProcessing = () => {
    if (!activeDocumentId) return;
    
    console.log('Starting processing for document:', activeDocumentId);
    setProcessingState({ status: 'processing', message: 'Processing started...' });
    
    // Call the processing endpoint
    axios.post(`http://127.0.0.1:8000/api/v1/process/${activeDocumentId}`)
      .then(response => {
        console.log('Processing started:', response.data);
        // AgentTimeline will handle the polling
      })
      .catch(error => {
        console.error('Failed to start processing:', error);
        setProcessingState({ status: 'error', message: 'Failed to start processing' });
      });
  };

  const handleDocumentSelect = (documentId) => {
    setActiveDocumentId(documentId);
    setActiveTab('processing');
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>Vision-Fusion Document Intelligence</h1>
          <p className="subtitle">Multi-Modal AI Document Auditing & Analysis System</p>
        </div>
        <div className="header-actions">
          <div className="backend-status">
            <span className={`status-indicator ${isBackendConnected ? 'connected' : 'disconnected'}`}>
              {isBackendConnected ? '✅ Backend Connected' : '❌ Backend Disconnected'}
            </span>
          </div>
          <div className="tab-buttons">
            <button 
              className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              📤 Upload
            </button>
            <button 
              className={`tab-btn ${activeTab === 'processing' ? 'active' : ''}`}
              onClick={() => activeDocumentId && setActiveTab('processing')}
              disabled={!activeDocumentId}
            >
              🤖 Processing
            </button>
            <button 
              className={`tab-btn ${activeTab === 'rag' ? 'active' : ''}`}
              onClick={() => activeDocumentId && setActiveTab('rag')}
              disabled={!activeDocumentId}
            >
              🔍 RAG Search
            </button>
          </div>
        </div>
      </header>

      {!isBackendConnected && (
        <div className="backend-error">
          <p>{backendError}</p>
          <button 
            onClick={() => window.location.reload()}
            className="retry-btn"
          >
            Retry Connection
          </button>
        </div>
      )}

      <main className="dashboard">
        {activeTab === 'upload' && (
          <div className="upload-section">
            <div className="upload-container">
              <UploadPanel onUploadSuccess={handleUploadSuccess} />
              
              {uploadedDocuments.length > 0 && (
                <div className="uploaded-documents">
                  <h3>📁 Uploaded Documents</h3>
                  <div className="documents-list">
                    {uploadedDocuments.map((doc, index) => (
                      <div 
                        key={doc.document_id} 
                        className={`document-item ${activeDocumentId === doc.document_id ? 'active' : ''}`}
                        onClick={() => handleDocumentSelect(doc.document_id)}
                      >
                        <div className="document-icon">📄</div>
                        <div className="document-info">
                          <div className="document-name">{doc.filename}</div>
                          <div className="document-meta">
                            <span className="document-id">ID: {doc.document_id.substring(0, 8)}...</span>
                            <span className="document-size">
                              {doc.file_size ? `(${Math.round(doc.file_size / 1024)} KB)` : ''}
                            </span>
                          </div>
                        </div>
                        <div className="document-actions">
                          <button 
                            className="process-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveDocumentId(doc.document_id);
                              setActiveTab('processing');
                            }}
                          >
                            Process
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'processing' && activeDocumentId && (
          <>
            <div className="document-info">
              <h2>
                Processing: <code>{activeDocumentId}</code>
                {uploadedDocuments.find(d => d.document_id === activeDocumentId)?.filename && (
                  <span className="filename">
                    {uploadedDocuments.find(d => d.document_id === activeDocumentId).filename}
                  </span>
                )}
              </h2>
              {processingState && (
                <div className={`processing-status ${processingState.status}`}>
                  Status: {processingState.message}
                </div>
              )}
              <div className="document-actions-bar">
                <button onClick={handleStartProcessing} className="action-btn primary">
                  ▶️ Start Processing
                </button>
                <button onClick={() => setActiveTab('rag')} className="action-btn secondary">
                  🔍 Query Document
                </button>
              </div>
            </div>

            <div className="dashboard-grid">
              <div className="grid-column">
                <AgentTimeline 
                  documentId={activeDocumentId}
                  onStartProcessing={handleStartProcessing}
                />
              </div>
              <div className="grid-column">
                <RAGPanel 
                  documentId={activeDocumentId}
                />
              </div>
            </div>
          </>
        )}

        {activeTab === 'rag' && activeDocumentId && (
          <div className="rag-section">
            <div className="rag-header">
              <h2>RAG Search: Document {activeDocumentId.substring(0, 8)}...</h2>
              <button onClick={() => setActiveTab('processing')} className="back-btn">
                ← Back to Processing
              </button>
            </div>
            <RAGPanel documentId={activeDocumentId} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Document Intelligence Control Center v1.0 • Backend: http://127.0.0.1:8000</p>
        <p className="footer-note">Real-time AI document processing with multi-agent orchestration</p>
      </footer>

      <style jsx>{`
        .app {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background: #0f172a;
          color: #e2e8f0;
        }

        .app-header {
          background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
          padding: 1rem 2rem;
          border-bottom: 1px solid #334155;
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .header-content h1 {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 0.25rem;
          background: linear-gradient(90deg, #60a5fa, #a78bfa);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .subtitle {
          color: #94a3b8;
          font-size: 0.9rem;
        }

        .header-actions {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          align-items: flex-end;
        }

        .backend-status {
          font-size: 0.875rem;
        }

        .status-indicator.connected {
          color: #22c55e;
        }

        .status-indicator.disconnected {
          color: #ef4444;
        }

        .tab-buttons {
          display: flex;
          gap: 0.5rem;
        }

        .tab-btn {
          background: #334155;
          color: #cbd5e1;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.9rem;
          transition: all 0.2s;
        }

        .tab-btn:hover:not(:disabled) {
          background: #475569;
        }

        .tab-btn.active {
          background: #3b82f6;
          color: white;
        }

        .tab-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .backend-error {
          background: #fee2e2;
          border: 1px solid #fecaca;
          color: #dc2626;
          padding: 1rem 2rem;
          text-align: center;
        }

        .backend-error p {
          margin: 0 0 1rem 0;
        }

        .retry-btn {
          background: #3b82f6;
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 6px;
          cursor: pointer;
        }

        .dashboard {
          flex: 1;
          padding: 2rem;
          max-width: 1400px;
          margin: 0 auto;
          width: 100%;
        }

        .upload-section {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .upload-container {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .uploaded-documents {
          background: #1e293b;
          border-radius: 12px;
          padding: 1.5rem;
          border: 1px solid #334155;
        }

        .uploaded-documents h3 {
          margin: 0 0 1rem 0;
          color: #e2e8f0;
        }

        .documents-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .document-item {
          background: #0f172a;
          border: 1px solid #334155;
          border-radius: 8px;
          padding: 1rem;
          display: flex;
          align-items: center;
          gap: 1rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .document-item:hover {
          background: #1e293b;
          border-color: #475569;
        }

        .document-item.active {
          border-color: #3b82f6;
          background: rgba(59, 130, 246, 0.1);
        }

        .document-icon {
          font-size: 1.5rem;
        }

        .document-info {
          flex: 1;
        }

        .document-name {
          color: #e2e8f0;
          font-weight: 500;
          margin-bottom: 0.25rem;
          word-break: break-all;
        }

        .document-meta {
          display: flex;
          gap: 1rem;
          color: #94a3b8;
          font-size: 0.875rem;
        }

        .document-id {
          font-family: monospace;
          background: rgba(255, 255, 255, 0.05);
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
        }

        .process-btn {
          background: #10b981;
          color: white;
          border: none;
          padding: 0.375rem 0.75rem;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.875rem;
        }

        .process-btn:hover {
          background: #059669;
        }

        .document-info {
          background: #1e293b;
          border-radius: 12px;
          padding: 1rem 1.5rem;
          margin-bottom: 2rem;
          border: 1px solid #334155;
        }

        .document-info h2 {
          color: #e2e8f0;
          font-size: 1.2rem;
          margin: 0 0 0.5rem 0;
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .document-info code {
          background: #0f172a;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          color: #60a5fa;
          font-family: monospace;
        }

        .filename {
          color: #94a3b8;
          font-size: 1rem;
          font-weight: normal;
        }

        .processing-status {
          color: #94a3b8;
          font-size: 0.875rem;
          margin-bottom: 1rem;
        }

        .processing-status.uploaded {
          color: #10b981;
        }

        .processing-status.processing {
          color: #f59e0b;
        }

        .processing-status.error {
          color: #ef4444;
        }

        .document-actions-bar {
          display: flex;
          gap: 1rem;
          margin-top: 1rem;
        }

        .action-btn {
          padding: 0.5rem 1rem;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .action-btn.primary {
          background: #3b82f6;
          color: white;
        }

        .action-btn.primary:hover {
          background: #2563eb;
        }

        .action-btn.secondary {
          background: #334155;
          color: #cbd5e1;
        }

        .action-btn.secondary:hover {
          background: #475569;
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
        }

        .grid-column {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .rag-section {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .rag-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: #1e293b;
          border-radius: 12px;
          padding: 1rem 1.5rem;
          border: 1px solid #334155;
        }

        .rag-header h2 {
          margin: 0;
          color: #e2e8f0;
        }

        .back-btn {
          background: #334155;
          color: #cbd5e1;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 6px;
          cursor: pointer;
        }

        .back-btn:hover {
          background: #475569;
        }

        .app-footer {
          background: #0f172a;
          padding: 1rem 2rem;
          text-align: center;
          color: #64748b;
          font-size: 0.85rem;
          border-top: 1px solid #334155;
        }

        .footer-note {
          font-size: 0.8rem;
          color: #475569;
          margin-top: 0.25rem;
        }

        @media (max-width: 1024px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
          }
          
          .app-header {
            flex-direction: column;
            text-align: center;
          }
          
          .header-actions {
            align-items: center;
            width: 100%;
          }
        }

        @media (max-width: 768px) {
          .dashboard {
            padding: 1rem;
          }
          
          .document-actions-bar {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
}

export default App;