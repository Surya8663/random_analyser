// src/components/UploadPanel.jsx - UPDATED
import React, { useState, useRef } from 'react';
import { FiUpload, FiCheck, FiAlertCircle } from 'react-icons/fi';
import axios from 'axios';

const UploadPanel = ({ onUploadSuccess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleFileUpload = async (file) => {
    // Reset state
    setError(null);
    setSuccessMessage(null);
    setIsUploading(true);
    setUploadProgress(0);

    try {
      console.log('📤 Starting upload for:', file.name);
      
      // Validate file type
      const allowedTypes = ['.pdf', '.docx', '.png', '.jpg', '.jpeg', '.txt'];
      const fileExt = '.' + file.name.split('.').pop().toLowerCase();
      
      if (!allowedTypes.includes(fileExt)) {
        throw new Error(`File type ${fileExt} not supported. Allowed: ${allowedTypes.join(', ')}`);
      }

      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Upload to backend with progress tracking
      const response = await axios.post('http://127.0.0.1:8000/api/v1/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(percentCompleted);
          }
        },
      });

      console.log('✅ Upload response:', response.data);
      
      // Extract document_id from response
      const documentId = response.data.document_id;
      if (!documentId) {
        throw new Error('No document ID received from server');
      }
      
      setUploadProgress(100);
      setSuccessMessage(`Document uploaded successfully! ID: ${documentId}`);

      // Notify parent component
      if (onUploadSuccess) {
        onUploadSuccess(response.data);
      }

      // Clear success message after 5 seconds
      setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);

    } catch (err) {
      console.error('❌ Upload error details:', err.response?.data || err.message);
      
      let errorMessage = 'Upload failed. Please try again.';
      if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="panel upload-panel">
      <h3>📤 Document Upload</h3>
      
      <div 
        className={`drop-zone ${isDragging ? 'dragging' : ''} ${isUploading ? 'uploading' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerFileInput}
      >
        {isUploading ? (
          <div className="uploading-state">
            <div className="upload-progress">
              <div className="progress-bar" style={{ width: `${uploadProgress}%` }}></div>
            </div>
            <p>Uploading... {uploadProgress}%</p>
            <p className="file-types">Please wait</p>
          </div>
        ) : (
          <>
            <FiUpload className="upload-icon" size={48} />
            <p className="drop-text">Drag & drop documents or</p>
            <button 
              className="upload-btn"
              onClick={(e) => {
                e.stopPropagation();
                triggerFileInput();
              }}
              disabled={isUploading}
            >
              Browse Files
            </button>
            <p className="file-types">Supports: PDF, DOCX, PNG, JPG, TXT</p>
          </>
        )}
        
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept=".pdf,.docx,.png,.jpg,.jpeg,.txt"
          style={{ display: 'none' }}
          disabled={isUploading}
        />
      </div>

      {error && (
        <div className="error-message">
          <FiAlertCircle />
          <span>{error}</span>
        </div>
      )}

      {successMessage && (
        <div className="success-message">
          <FiCheck />
          <span>{successMessage}</span>
        </div>
      )}

      <div className="queue-status">
        <p>📊 Status: <span className={`status ${isUploading ? 'uploading' : 'idle'}`}>
          {isUploading ? 'Uploading...' : 'Ready for upload'}
        </span></p>
      </div>

      <style jsx>{`
        .upload-panel {
          min-height: 300px;
          display: flex;
          flex-direction: column;
        }
        
        .drop-zone {
          flex: 1;
          border: 2px dashed #475569;
          border-radius: 8px;
          padding: 3rem 2rem;
          text-align: center;
          margin-bottom: 1rem;
          transition: all 0.2s;
          background: #1e293b;
          cursor: pointer;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }
        
        .drop-zone.dragging {
          border-color: #60a5fa;
          background: rgba(96, 165, 250, 0.1);
          transform: scale(1.02);
        }
        
        .drop-zone.uploading {
          border-color: #f59e0b;
          background: rgba(245, 158, 11, 0.05);
        }
        
        .upload-icon {
          color: #64748b;
          margin-bottom: 1rem;
        }
        
        .drop-text {
          color: #cbd5e1;
          margin-bottom: 1rem;
          font-size: 1rem;
        }
        
        .upload-btn {
          background: #3b82f6;
          color: white;
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
          margin: 1rem 0;
          transition: background 0.2s;
          font-size: 1rem;
        }
        
        .upload-btn:hover:not(:disabled) {
          background: #2563eb;
        }
        
        .upload-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .file-types {
          color: #94a3b8;
          font-size: 0.85rem;
          margin-top: 0.5rem;
        }
        
        .uploading-state {
          width: 100%;
        }
        
        .upload-progress {
          width: 100%;
          height: 8px;
          background: #334155;
          border-radius: 4px;
          margin: 1rem 0;
          overflow: hidden;
        }
        
        .progress-bar {
          height: 100%;
          background: linear-gradient(90deg, #3b82f6, #8b5cf6);
          border-radius: 4px;
          transition: width 0.3s ease;
        }
        
        .error-message {
          background: #fee2e2;
          border: 1px solid #fecaca;
          color: #dc2626;
          padding: 0.75rem 1rem;
          border-radius: 6px;
          margin: 1rem 0;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
        }
        
        .success-message {
          background: #d1fae5;
          border: 1px solid #a7f3d0;
          color: #059669;
          padding: 0.75rem 1rem;
          border-radius: 6px;
          margin: 1rem 0;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
        }
        
        .queue-status {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #334155;
        }
        
        .status {
          font-weight: 600;
        }
        
        .status.idle {
          color: #10b981;
        }
        
        .status.uploading {
          color: #f59e0b;
        }
      `}</style>
    </div>
  );
};

export default UploadPanel;