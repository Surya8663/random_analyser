import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './App.css';

// Create root element
const rootElement = document.getElementById('root');

if (!rootElement) {
  console.error('Root element not found!');
} else {
  const root = ReactDOM.createRoot(rootElement);
  
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

// Error boundary for the root
window.addEventListener('error', (event) => {
  console.error('Global error caught:', event.error);
});

// Log app initialization
console.log('🚀 Document Intelligence Control Center initialized');