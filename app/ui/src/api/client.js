// src/api/client.js - Simplified for fast backend
import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

const DocumentAPI = {
  // Upload document
  uploadDocument: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return apiClient.post('/api/v1/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Start processing document
  startProcessing: async (documentId) => {
    return apiClient.post(`/api/v1/process/${documentId}`);
  },

  // Get processing status
  getStatus: async (documentId) => {
    return apiClient.get(`/api/v1/status/${documentId}`);
  },

  // Get processing results
  getResults: async (documentId) => {
    return apiClient.get(`/api/v1/results/${documentId}`);
  },

  // Query document (RAG)
  queryDocument: async (documentId, query) => {
    return apiClient.post('/api/v1/query', {
      document_id: documentId,
      query: query,
    });
  },

  // Health check
  checkHealth: async () => {
    return apiClient.get('/health');
  },

  // List documents
  listDocuments: async () => {
    return apiClient.get('/api/v1/documents');
  }
};

export default DocumentAPI;