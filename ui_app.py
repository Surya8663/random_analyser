# ui_app.py - Modern Document Intelligence UI
# Clean, Professional Layout with Main Focus on Upload
# AI Agents Builder System Competition

import streamlit as st
import requests
import time
import json
import base64
from PIL import Image
import io
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import uuid
import hashlib

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="DocIntel AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_BASE_URL = "http://localhost:8000"
ENDPOINTS = {
    'upload': f"{API_BASE_URL}/api/v1/upload",
    'status': f"{API_BASE_URL}/api/v1/status",
    'results': f"{API_BASE_URL}/api/v1/results",
    'query': f"{API_BASE_URL}/api/v1/query",
    'search': f"{API_BASE_URL}/api/v1/rag/search",
    'process': f"{API_BASE_URL}/api/v1/process",
    'health': f"{API_BASE_URL}/health",
    'system': f"{API_BASE_URL}/system"
}

# ============================================================================
# CUSTOM CSS - MODERN & CLEAN
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background: #0f172a;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Upload area */
    .upload-container {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 5px 0 0 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(102, 126, 234, 0.2);
        color: white;
        border-bottom: 2px solid #667eea;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1e293b;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .status-success {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border-color: rgba(34, 197, 94, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    /* File info card */
    .file-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Success message */
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 10px;
        padding: 15px;
        color: #22c55e;
    }
    
    /* Warning message */
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 15px;
        color: #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CORE APPLICATION CLASS
# ============================================================================
class DocumentIntelligenceApp:
    """Main application controller"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'processing_started': False,
            'show_results': False,
            'current_document': None,
            'current_document_id': None,
            'uploaded_file': None,
            'recent_documents': [],
            'processing_history': [],
            'api_health': False,
            'processing_options': {
                'layout_analysis': True,
                'ocr_enhancement': True,
                'signature_detection': True,
                'data_extraction': True,
                'quality_validation': True,
                'rag_indexing': True
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def check_api_health(self) -> bool:
        """Check if backend API is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state['api_health'] = data.get('status') == 'healthy'
                return st.session_state['api_health']
            else:
                st.session_state['api_health'] = False
                return False
        except:
            st.session_state['api_health'] = False
            return False
    
    def upload_document(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """Upload document to backend"""
        try:
            files = {'file': (filename, file_bytes)}
            response = requests.post(ENDPOINTS['upload'], files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                document_id = data.get('document_id')
                
                if document_id:
                    # Store document info
                    doc_info = {
                        'id': document_id,
                        'name': filename,
                        'size': len(file_bytes),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Add to recent documents
                    if 'recent_documents' not in st.session_state:
                        st.session_state['recent_documents'] = []
                    
                    st.session_state['recent_documents'].insert(0, doc_info)
                    st.session_state['recent_documents'] = st.session_state['recent_documents'][:10]
                    
                    return document_id
            else:
                st.error(f"Upload failed: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
            return None
    
    def get_processing_status(self, document_id: str) -> Dict:
        """Get processing status"""
        try:
            response = requests.get(f"{ENDPOINTS['status']}/{document_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {'status': 'error', 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_results(self, document_id: str) -> Optional[Dict]:
        """Get processing results"""
        try:
            response = requests.get(f"{ENDPOINTS['results']}/{document_id}", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get results: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Results error: {str(e)}")
            return None
    
    def query_document(self, document_id: str, question: str) -> Optional[Dict]:
        """Query document using AI"""
        try:
            payload = {'document_id': document_id, 'question': question}
            response = requests.post(ENDPOINTS['query'], json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Query failed: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return None

# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================
def render_sidebar(app: DocumentIntelligenceApp):
    """Render sidebar with controls and info"""
    with st.sidebar:
        # App Logo
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 24px; margin-bottom: 5px;">📄</h1>
            <h3 style="margin: 0; color: #ffffff;">DocIntel AI</h3>
            <p style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
            Multi-Modal Document Intelligence
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status
        st.markdown("### System Status")
        api_healthy = app.check_api_health()
        
        if api_healthy:
            st.markdown('<div class="status-badge status-success">✅ API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-error">❌ API Disconnected</div>', unsafe_allow_html=True)
            with st.expander("Troubleshooting"):
                st.markdown("""
                1. Ensure backend is running:
                ```bash
                uvicorn app.main:app --reload --port 8000
                ```
                2. Check if port 8000 is available
                3. Verify all dependencies are installed
                """)
        
        # Processing Options
        st.markdown("---")
        st.markdown("### ⚙️ Processing Options")
        
        with st.expander("Configure Analysis", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state['processing_options']['layout_analysis'] = st.checkbox(
                    "Layout Analysis", 
                    value=st.session_state['processing_options']['layout_analysis'],
                    help="Detect tables, charts, figures"
                )
                st.session_state['processing_options']['ocr_enhancement'] = st.checkbox(
                    "OCR Enhancement", 
                    value=st.session_state['processing_options']['ocr_enhancement'],
                    help="Optimize text recognition"
                )
                st.session_state['processing_options']['signature_detection'] = st.checkbox(
                    "Signature Detection", 
                    value=st.session_state['processing_options']['signature_detection'],
                    help="Identify signatures"
                )
            
            with col2:
                st.session_state['processing_options']['data_extraction'] = st.checkbox(
                    "Data Extraction", 
                    value=st.session_state['processing_options']['data_extraction'],
                    help="Extract structured data"
                )
                st.session_state['processing_options']['quality_validation'] = st.checkbox(
                    "Quality Validation", 
                    value=st.session_state['processing_options']['quality_validation'],
                    help="Validate extraction quality"
                )
                st.session_state['processing_options']['rag_indexing'] = st.checkbox(
                    "RAG Indexing", 
                    value=st.session_state['processing_options']['rag_indexing'],
                    help="Index for semantic search"
                )
        
        # Recent Documents
        if st.session_state.get('recent_documents'):
            st.markdown("---")
            st.markdown("### 📁 Recent Documents")
            
            for doc in st.session_state['recent_documents'][:5]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📄 {doc.get('name', 'Document')[:20]}")
                with col2:
                    if st.button("Open", key=f"open_{doc['id']}", help="Open this document"):
                        st.session_state['current_document_id'] = doc['id']
                        # Try to get results
                        results = app.get_results(doc['id'])
                        if results:
                            st.session_state['results'] = results
                            st.session_state['show_results'] = True
                            st.rerun()
        
        # System Info
        st.markdown("---")
        with st.expander("ℹ️ About"):
            st.markdown("""
            **DocIntel AI Platform**
            
            **Version:** 1.0.0
            **Backend:** FastAPI + Python 3.11
            **AI Models:** Multi-Modal Transformers
            **Storage:** Vector Database
            
            **Competition:** AI Agents Builder System
            **Team:** AI Research Group
            
            For support, contact: support@docintel.ai
            """)

# ============================================================================
# MAIN UI COMPONENTS
# ============================================================================
def render_main_header():
    """Render main header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style="margin-bottom: 10px;">
            <h1 style="margin-bottom: 5px;">Multi-Modal Document Intelligence</h1>
            <p style="color: #94a3b8; font-size: 14px; margin-top: 0;">
            AI-powered document analysis with computer vision, OCR, and multi-agent reasoning
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: right; margin-top: 20px;">
            <p style="color: #64748b; font-size: 12px; margin: 0;">{datetime.now().strftime('%H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

def render_upload_section(app: DocumentIntelligenceApp):
    """Render main upload section"""
    st.markdown("## 📤 Upload Document")
    
    # File upload area
    uploaded_file = st.file_uploader(
        "Drag and drop your document here, or click to browse",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Store uploaded file
        st.session_state['uploaded_file'] = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'content': uploaded_file.getvalue()
        }
        
        # File info card
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"""
            <div class="file-card">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">📄</span>
                    <div>
                        <strong>{uploaded_file.name}</strong>
                        <p style="color: #94a3b8; margin: 0; font-size: 12px;">
                        {uploaded_file.size / 1024:.1f} KB • {uploaded_file.type}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("")
            if st.button("Remove", key="remove_file"):
                st.session_state['uploaded_file'] = None
                st.rerun()
        
        with col3:
            st.markdown("")
            # Process button
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                if not st.session_state['api_health']:
                    st.error("Cannot process: Backend API is not connected")
                else:
                    with st.spinner("Uploading document..."):
                        document_id = app.upload_document(
                            uploaded_file.getvalue(),
                            uploaded_file.name
                        )
                        
                        if document_id:
                            st.session_state['current_document_id'] = document_id
                            st.session_state['processing_started'] = True
                            st.success(f"Document uploaded successfully! ID: {document_id[:8]}")
                            st.rerun()
    
    else:
        # Upload prompt
        st.markdown("""
        <div class="upload-container">
            <div style="font-size: 48px; margin-bottom: 20px; color: #64748b;">📤</div>
            <h3 style="color: #ffffff; margin-bottom: 10px;">Upload Your Document</h3>
            <p style="color: #94a3b8; max-width: 400px; margin: 0 auto 20px;">
            Upload PDF or image files for AI-powered analysis. 
            Our system will extract text, detect layouts, and analyze content.
            </p>
            <p style="color: #64748b; font-size: 12px;">
            Maximum file size: 50MB • Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_features_section():
    """Render features overview"""
    st.markdown("---")
    st.markdown("## ✨ Key Features")
    
    features = [
        {
            "icon": "👁️",
            "title": "Computer Vision",
            "description": "Advanced layout analysis with table and chart detection"
        },
        {
            "icon": "📝",
            "title": "OCR Intelligence",
            "description": "High-accuracy text extraction with confidence scoring"
        },
        {
            "icon": "🤖",
            "title": "Multi-Agent System",
            "description": "Vision, text, fusion, and validation agents working together"
        },
        {
            "icon": "🔍",
            "title": "Semantic Search",
            "description": "Multi-modal RAG for intelligent document querying"
        },
        {
            "icon": "📊",
            "title": "Analytics Dashboard",
            "description": "Comprehensive insights and visualization tools"
        },
        {
            "icon": "⚡",
            "title": "Real-Time Processing",
            "description": "Live status updates and progress tracking"
        }
    ]
    
    cols = st.columns(3)
    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 32px; margin-bottom: 15px;">{feature['icon']}</div>
                <h4 style="margin: 0 0 10px 0; color: #ffffff;">{feature['title']}</h4>
                <p style="color: #94a3b8; font-size: 14px; line-height: 1.5; margin: 0;">
                {feature['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

def render_processing_view(app: DocumentIntelligenceApp, document_id: str):
    """Render processing view"""
    st.markdown("## 🔄 Processing Document")
    
    # Status tracking
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    max_polls = 180
    poll_count = 0
    
    while poll_count < max_polls:
        status_data = app.get_processing_status(document_id)
        current_status = status_data.get('status', 'unknown')
        
        # Update progress
        progress = 0.2
        status_msg = "Document uploaded"
        
        if current_status == 'processing':
            progress = 0.3 + (poll_count % 30) * 0.02
            status_msg = "AI agents processing document"
        elif current_status == 'completed':
            progress = 1.0
            status_msg = "Processing completed"
        elif current_status == 'error':
            progress = 1.0
            status_msg = f"Error: {status_data.get('error', 'Unknown error')}"
        
        # Update UI
        with status_placeholder.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                icon = "⚙️" if current_status == 'processing' else "✅" if current_status == 'completed' else "❌" if current_status == 'error' else "📤"
                st.markdown(f"<h1 style='font-size: 48px; text-align: center;'>{icon}</h1>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"### {status_msg}")
                st.markdown(f"**Document ID:** `{document_id[:12]}...`")
                elapsed = time.time() - start_time
                st.caption(f"Elapsed time: {elapsed:.1f}s")
        
        progress_bar.progress(progress)
        
        # Check completion
        if current_status in ['completed', 'error']:
            if current_status == 'completed':
                st.success("✅ Processing completed successfully!")
                
                # Fetch results
                with st.spinner("Loading results..."):
                    results = app.get_results(document_id)
                    if results:
                        st.session_state['results'] = results
                        st.session_state['show_results'] = True
                        st.rerun()
                    else:
                        st.error("Failed to load results")
            else:
                st.error(f"❌ Processing failed: {status_data.get('error', 'Unknown error')}")
            
            # Add action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Process New Document", use_container_width=True):
                    st.session_state['processing_started'] = False
                    st.session_state['uploaded_file'] = None
                    st.rerun()
            with col2:
                if st.button("📊 View Results Anyway", use_container_width=True):
                    st.session_state['show_results'] = True
                    st.rerun()
            break
        
        # Wait and continue
        time.sleep(2)
        poll_count += 1
        st.rerun()
    
    if poll_count >= max_polls:
        st.error("⏰ Processing timeout - taking too long")
        if st.button("Return to Upload", use_container_width=True):
            st.session_state['processing_started'] = False
            st.rerun()

def render_results_view(app: DocumentIntelligenceApp, results: Dict):
    """Render results view"""
    if not results:
        st.error("No results available")
        return
    
    document_id = results.get('document_id', 'Unknown')
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <h1 style="margin: 0;">📊 Analysis Results</h1>
            <span class="status-badge">DOC-{document_id[:8]}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("📄 New Document", use_container_width=True):
            st.session_state.clear()
            app.initialize_session_state()
            st.rerun()
    
    # Metrics
    validation_results = results.get('validation_results', {})
    extracted_fields = results.get('extracted_fields', {})
    
    integrity_score = validation_results.get('integrity_score', 0) * 100
    risk_score = validation_results.get('risk_score', 0) * 100
    field_count = len(extracted_fields)
    confidence_avg = np.mean([f.get('confidence', 0) * 100 for f in extracted_fields.values()]) if extracted_fields else 0
    
    metric_cols = st.columns(4)
    metrics = [
        (f"{integrity_score:.1f}%", "Integrity Score"),
        (f"{risk_score:.1f}%", "Risk Score"),
        (str(field_count), "Fields Extracted"),
        (f"{confidence_avg:.1f}%", "Avg Confidence")
    ]
    
    for idx, (value, label) in enumerate(metrics):
        with metric_cols[idx]:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{value}</p>
                <p class="metric-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Extracted Data", 
        "📊 Validation", 
        "🔍 Query", 
        "⚙️ Details"
    ])
    
    # Tab 1: Extracted Data
    with tab1:
        if extracted_fields:
            for field_name, field_data in extracted_fields.items():
                with st.expander(f"**{field_name.replace('_', ' ').title()}**", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        value = field_data.get('value', 'N/A')
                        if isinstance(value, (dict, list)):
                            st.json(value)
                        else:
                            st.code(str(value), language='text')
                    with col2:
                        confidence = field_data.get('confidence', 0) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
        else:
            st.info("No fields were extracted")
    
    # Tab 2: Validation
    with tab2:
        contradictions = validation_results.get('contradictions', [])
        
        if contradictions:
            st.markdown(f"#### ⚠️ {len(contradictions)} Contradictions Found")
            for idx, contra in enumerate(contradictions, 1):
                with st.expander(f"Contradiction #{idx}", expanded=False):
                    st.markdown(f"**Type:** {contra.get('type', 'Unknown')}")
                    st.markdown(f"**Severity:** {contra.get('severity', 'Unknown')}")
                    st.markdown(f"**Description:** {contra.get('description', 'No description')}")
        else:
            st.success("✅ No contradictions found")
        
        # Risk assessment
        st.markdown("#### 🎯 Risk Assessment")
        if risk_score > 70:
            st.error(f"High risk detected: {risk_score:.1f}%")
        elif risk_score > 30:
            st.warning(f"Moderate risk: {risk_score:.1f}%")
        else:
            st.success(f"Low risk: {risk_score:.1f}%")
        
        st.progress(risk_score / 100)
    
    # Tab 3: Query
    with tab3:
        st.markdown("### 🔍 Query Document")
        
        # Query input
        query = st.text_input(
            "Ask a question about your document:",
            placeholder="Example: 'What charts were detected?' or 'Show me all amounts'",
            key="query_input"
        )
        
        if query:
            with st.spinner("Processing query..."):
                response = app.query_document(document_id, query)
                
                if response and response.get('success'):
                    answer = response.get('answer', 'No answer provided')
                    confidence = response.get('confidence', 0) * 100
                    
                    # Display answer
                    st.markdown("#### Answer")
                    st.info(answer)
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col2:
                        sources = response.get('sources', [])
                        st.metric("Sources", len(sources))
        
        # Quick queries
        st.markdown("#### 💡 Quick Questions")
        quick_cols = st.columns(3)
        queries = [
            "What charts were detected?",
            "Show me validation results",
            "Extract all amounts",
            "What's the document type?",
            "Any contradictions found?",
            "Summary of findings"
        ]
        
        for idx, q in enumerate(queries):
            with quick_cols[idx % 3]:
                if st.button(q, key=f"quick_{idx}", use_container_width=True):
                    # Update query input using callback
                    st.session_state['query_input'] = q
                    st.rerun()
    
    # Tab 4: Details
    with tab4:
        metadata = results.get('processing_metadata', {})
        
        if metadata:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 Performance")
                st.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
                st.metric("Document Pages", metadata.get('total_pages', 1))
                st.metric("Agents Executed", len(metadata.get('agents_executed', [])))
            
            with col2:
                st.markdown("#### 📊 Statistics")
                st.metric("Fields Extracted", len(extracted_fields))
                st.metric("Errors Count", metadata.get('errors_count', 0))
                st.metric("Warnings Count", metadata.get('warnings_count', 0))
            
            # Raw data
            with st.expander("View Raw Data"):
                st.json(results)
        else:
            st.info("No processing metadata available")
    
    # Action buttons
    st.markdown("---")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("🔄 Reprocess", use_container_width=True):
            st.info("Reprocessing feature coming soon")
    
    with action_cols[1]:
        if st.button("📥 Export JSON", use_container_width=True):
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="Download",
                data=results_json,
                file_name=f"analysis_{document_id}.json",
                mime="application/json"
            )
    
    with action_cols[2]:
        if st.button("🏠 Return Home", use_container_width=True):
            st.session_state['show_results'] = False
            st.session_state.pop('results', None)
            st.rerun()

def render_getting_started():
    """Render getting started guide"""
    st.markdown("---")
    st.markdown("## 🚀 Getting Started")
    
    steps = [
        {
            "number": "1",
            "title": "Upload Document",
            "description": "Drag and drop or click to upload your PDF or image file"
        },
        {
            "number": "2",
            "title": "Configure Options",
            "description": "Select the analysis features you want to enable in the sidebar"
        },
        {
            "number": "3",
            "title": "Process & Analyze",
            "description": "Click 'Process Document' to start AI-powered analysis"
        },
        {
            "number": "4",
            "title": "View Results",
            "description": "Explore extracted data, validation results, and query your document"
        }
    ]
    
    step_cols = st.columns(4)
    for idx, step in enumerate(steps):
        with step_cols[idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    margin: 0 auto 15px;
                ">
                    {step['number']}
                </div>
                <h4 style="color: #ffffff; margin-bottom: 10px;">{step['title']}</h4>
                <p style="color: #94a3b8; font-size: 14px; line-height: 1.5;">
                {step['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    
    # Initialize app
    app = DocumentIntelligenceApp()
    
    # Render sidebar
    render_sidebar(app)
    
    # Render main header
    render_main_header()
    
    # Main content routing
    if st.session_state.get('processing_started') and not st.session_state.get('show_results'):
        if 'current_document_id' in st.session_state:
            render_processing_view(app, st.session_state['current_document_id'])
        else:
            st.error("No document ID found")
            st.session_state['processing_started'] = False
    
    elif st.session_state.get('show_results') and 'results' in st.session_state:
        render_results_view(app, st.session_state['results'])
    
    else:
        # Main dashboard
        render_upload_section(app)
        render_features_section()
        render_getting_started()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; color: #64748b; font-size: 12px;">
        <p>DocIntel AI Platform • AI Agents Builder System Competition</p>
        <p>© 2024 AI Research Group. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()