# ui_app.py - FIXED VERSION
import streamlit as st
import requests
import time
import json
import os
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import uuid
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Document Intelligence AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/upload"
STATUS_ENDPOINT = f"{API_BASE_URL}/api/v1/status"
RESULTS_ENDPOINT = f"{API_BASE_URL}/api/v1/results"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
RAG_SEARCH_ENDPOINT = f"{API_BASE_URL}/api/v1/rag/search"
PROCESS_ENDPOINT = f"{API_BASE_URL}/api/v1/process"

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 40px 0;
        position: relative;
    }
    
    .step-indicator::before {
        content: '';
        position: absolute;
        top: 20px;
        left: 10%;
        right: 10%;
        height: 2px;
        background-color: #2d3746;
        z-index: 1;
    }
    
    .step {
        text-align: center;
        position: relative;
        z-index: 2;
        flex: 1;
    }
    
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #2d3746;
        color: #718096;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px;
        font-weight: bold;
        font-size: 18px;
        border: 3px solid #2d3746;
        transition: all 0.3s ease;
    }
    
    .step.active .step-circle {
        background-color: #4f46e5;
        color: white;
        border-color: #6366f1;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.2);
    }
    
    .step.completed .step-circle {
        background-color: #10b981;
        color: white;
        border-color: #34d399;
    }
    
    .step-label {
        color: #718096;
        font-size: 14px;
        margin-top: 5px;
        font-weight: 500;
    }
    
    .step.active .step-label {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .status-processing {
        background-color: rgba(249, 115, 22, 0.2);
        color: #f97316;
    }
    
    .status-error {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 300px !important;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #4f46e5;
        border-radius: 12px;
        padding: 60px 40px;
        text-align: center;
        background: rgba(79, 70, 229, 0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 20px 0;
    }
    
    .upload-area:hover {
        border-color: #6366f1;
        background: rgba(79, 70, 229, 0.1);
    }
    
    .upload-area.dragover {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
    }
    
    /* File cards */
    .file-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .file-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(79, 70, 229, 0.5);
    }
    
    .file-icon {
        font-size: 24px;
        margin-right: 12px;
    }
    
    /* Section dividers */
    .section-divider {
        border-top: 1px solid #2d3746;
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

class DocumentIntelligenceUI:
    def __init__(self):
        self.current_document_id = None
        self.processing_status = {}
        self.results_data = {}
    
    def check_api_health(self) -> bool:
        """Check if backend API is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_documents(self, files: List[bytes], filenames: List[str]) -> Optional[List[str]]:
        """Upload multiple documents to backend"""
        try:
            uploaded_ids = []
            for file_bytes, filename in zip(files, filenames):
                files_dict = {'file': (filename, file_bytes)}
                response = requests.post(UPLOAD_ENDPOINT, files=files_dict)
                
                if response.status_code == 200:
                    data = response.json()
                    document_id = data.get('document_id')
                    uploaded_ids.append(document_id)
                else:
                    st.error(f"Upload failed for {filename}: {response.text}")
                    return None
            
            # Store first document ID for processing
            if uploaded_ids:
                st.session_state['document_ids'] = uploaded_ids
                st.session_state['current_document_id'] = uploaded_ids[0]
                return uploaded_ids
            
            return None
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
            return None
    
    def get_processing_status(self, document_id: str) -> Dict:
        """Get processing status from backend"""
        try:
            response = requests.get(f"{STATUS_ENDPOINT}/{document_id}")
            if response.status_code == 200:
                return response.json()
            return {"status": "unknown", "error": "Status check failed"}
        except:
            return {"status": "api_error", "error": "Cannot connect to API"}
    
    def get_results(self, document_id: str) -> Optional[Dict]:
        """Get processing results from backend"""
        try:
            response = requests.get(f"{RESULTS_ENDPOINT}/{document_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def query_document(self, document_id: str, question: str) -> Optional[Dict]:
        """Query document using RAG system"""
        try:
            payload = {
                "document_id": document_id,
                "question": question
            }
            response = requests.post(QUERY_ENDPOINT, json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def rag_search(self, query: str, query_type: str = "text", limit: int = 5) -> Optional[Dict]:
        """Search across indexed documents"""
        try:
            payload = {
                "query": query,
                "query_type": query_type,
                "limit": limit
            }
            response = requests.post(RAG_SEARCH_ENDPOINT, json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def reprocess_document(self, document_id: str) -> bool:
        """Trigger reprocessing of document"""
        try:
            payload = {
                "document_id": document_id,
                "reprocess": True
            }
            response = requests.post(PROCESS_ENDPOINT, json=payload)
            return response.status_code == 200
        except:
            return False

# ========== STEP INDICATOR COMPONENT ==========
def render_step_indicator(current_step: str):
    """Render step-by-step progress indicator"""
    steps = [
        {"id": "upload", "label": "Upload", "icon": "📤"},
        {"id": "process", "label": "Process", "icon": "⚙️"},
        {"id": "results", "label": "Results", "icon": "📊"},
        {"id": "query", "label": "Query", "icon": "🔍"}
    ]
    
    step_html = '<div class="step-indicator">'
    
    for i, step in enumerate(steps):
        step_class = ""
        if step["id"] == current_step:
            step_class = "active"
        elif steps.index(next(s for s in steps if s["id"] == current_step)) > i:
            step_class = "completed"
        
        step_html += f'''
        <div class="step {step_class}">
            <div class="step-circle">{step["icon"]}</div>
            <div class="step-label">STEP {i+1}<br>{step["label"]}</div>
        </div>
        '''
    
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

# ========== SIDEBAR COMPONENT ==========
def render_sidebar(ui: DocumentIntelligenceUI):
    """Render sidebar with navigation only"""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px; padding: 20px 0; border-bottom: 1px solid #2d3746;">
            <h1 style="font-size: 28px; margin-bottom: 10px; color: #4f46e5;">📊</h1>
            <h2 style="font-size: 20px; margin-bottom: 5px; color: white;">Document Intelligence</h2>
            <p style="font-size: 12px; color: #94a3b8;">AI-Powered Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status
        st.markdown("### 📈 System Status")
        api_status = ui.check_api_health()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            status_icon = "✅" if api_status else "❌"
            st.markdown(f"<h3>{status_icon}</h3>", unsafe_allow_html=True)
        with col2:
            status_text = "Connected" if api_status else "Disconnected"
            status_color = "status-success" if api_status else "status-error"
            st.markdown(f'<span class="status-badge {status_color}">{status_text}</span>', 
                       unsafe_allow_html=True)
        
        if not api_status:
            st.warning("⚠️ Start backend server:")
            st.code("uvicorn app.main:app --reload", language="bash")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        nav_options = {
            "🏠 Home": "home",
            "📤 Upload": "upload",
            "⚙️ Process": "process",
            "📊 Results": "results",
            "🔍 Query": "query"
        }
        
        for label, step_id in nav_options.items():
            if st.button(label, use_container_width=True, key=f"nav_{step_id}"):
                st.session_state['current_step'] = step_id
                st.rerun()
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Recent Documents
        if 'recent_docs' in st.session_state and st.session_state['recent_docs']:
            st.markdown("### 📁 Recent Documents")
            for doc in st.session_state['recent_docs'][:3]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc.get('name', 'Document')[:15]}...")
                with col2:
                    if st.button("↗️", key=f"open_{doc['id']}"):
                        st.session_state['document_id'] = doc['id']
                        st.session_state['current_step'] = "results"
                        st.rerun()

# ========== FILE UPLOAD COMPONENT ==========
def render_file_upload():
    """Render drag & drop file upload area"""
    st.markdown("""
    <div class="upload-area" id="uploadArea">
        <div style="font-size: 48px; margin-bottom: 20px;">📤</div>
        <h3 style="margin-bottom: 10px;">Drag & Drop Files Here</h3>
        <p style="color: #94a3b8; margin-bottom: 20px;">or click to browse files</p>
        <p style="color: #64748b; font-size: 12px;">
            Supported: PDF, PNG, JPG, JPEG, DOC, DOCX, TXT, CSV
        </p>
        <p style="color: #94a3b8; font-size: 11px; margin-top: 10px;">
            Multiple files supported (up to 10)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with drag & drop support
    uploaded_files = st.file_uploader(
        "",
        type=['pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx', 'txt', 'csv'],
        accept_multiple_files=True,
        help="Upload multiple documents for analysis",
        key="file_uploader_main",
        label_visibility="collapsed"
    )
    
    return uploaded_files

def display_uploaded_files(uploaded_files):
    """Display uploaded files with icons"""
    if not uploaded_files:
        return
    
    st.markdown("### 📁 Uploaded Files")
    
    for file in uploaded_files:
        file_ext = Path(file.name).suffix.lower()
        icon_map = {
            '.pdf': '📕',
            '.png': '🖼️',
            '.jpg': '🖼️',
            '.jpeg': '🖼️',
            '.doc': '📄',
            '.docx': '📄',
            '.txt': '📝',
            '.csv': '📊'
        }
        icon = icon_map.get(file_ext, '📎')
        
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.markdown(f'<div class="file-icon">{icon}</div>', unsafe_allow_html=True)
        with col2:
            st.text(file.name)
        with col3:
            st.text(f"{file.size / 1024:.1f} KB")

# ========== HOME PAGE ==========
def render_home_page(ui: DocumentIntelligenceUI):
    """Render landing/home page"""
    # Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 60px 0 40px 0;">
            <h1 style="font-size: 48px; margin-bottom: 20px; color: white;">
                Document Intelligence AI
            </h1>
            <p style="font-size: 18px; color: #94a3b8; line-height: 1.6;">
                Transform documents into actionable insights using AI-powered computer vision, 
                OCR, and multi-agent analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # How It Works - Simple 3-step flow
    st.markdown("## 🚀 How It Works")
    
    steps_cols = st.columns(3)
    with steps_cols[0]:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 12px; background: rgba(79, 70, 229, 0.05);">
            <div style="font-size: 40px; margin-bottom: 15px;">📤</div>
            <h3 style="margin: 10px 0;">Upload</h3>
            <p style="color: #94a3b8;">Upload PDFs, images, or documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_cols[1]:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 12px; background: rgba(79, 70, 229, 0.05);">
            <div style="font-size: 40px; margin-bottom: 15px;">⚙️</div>
            <h3 style="margin: 10px 0;">Process</h3>
            <p style="color: #94a3b8;">AI agents analyze content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_cols[2]:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 12px; background: rgba(79, 70, 229, 0.05);">
            <div style="font-size: 40px; margin-bottom: 15px;">🔍</div>
            <h3 style="margin: 10px 0;">Query</h3>
            <p style="color: #94a3b8;">Ask questions about your documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get Started Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started →", type="primary", use_container_width=True, key="get_started"):
            st.session_state['current_step'] = "upload"
            st.rerun()

# ========== UPLOAD STEP ==========
def render_upload_step(ui: DocumentIntelligenceUI):
    """Render upload step with drag & drop"""
    render_step_indicator("upload")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>📤 Upload Your Documents</h2>
        <p style="color: #94a3b8;">Upload multiple files for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main upload area
    uploaded_files = render_file_upload()
    
    if uploaded_files:
        display_uploaded_files(uploaded_files)
        
        # Store uploaded files in session state
        st.session_state['uploaded_files'] = uploaded_files
        
        # Proceed button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Proceed to Processing", type="primary", use_container_width=True):
                # Store files for processing
                file_bytes_list = [file.getvalue() for file in uploaded_files]
                filenames = [file.name for file in uploaded_files]
                
                # Upload to backend
                with st.spinner("Uploading documents..."):
                    document_ids = ui.upload_documents(file_bytes_list, filenames)
                    
                    if document_ids:
                        st.success(f"✅ Uploaded {len(document_ids)} documents successfully!")
                        st.session_state['document_ids'] = document_ids
                        st.session_state['current_step'] = "process"
                        st.rerun()
                    else:
                        st.error("❌ Upload failed")

# ========== PROCESS STEP ==========
def render_process_step(ui: DocumentIntelligenceUI):
    """Render processing step"""
    render_step_indicator("process")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>⚙️ Process Documents</h2>
        <p style="color: #94a3b8;">Configure processing options and start analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'document_ids' not in st.session_state or not st.session_state['document_ids']:
        st.warning("No documents uploaded. Please upload documents first.")
        if st.button("Go to Upload", key="go_to_upload"):
            st.session_state['current_step'] = "upload"
            st.rerun()
        return
    
    document_ids = st.session_state['document_ids']
    
    # Processing Options
    with st.expander("⚙️ Processing Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👁️ Vision Analysis")
            layout_detection = st.checkbox("Layout Detection", value=True)
            signature_detection = st.checkbox("Signature Detection", value=True)
            table_detection = st.checkbox("Table Detection", value=True)
        
        with col2:
            st.markdown("#### 📝 Text Analysis")
            ocr_enabled = st.checkbox("OCR Extraction", value=True)
            entity_extraction = st.checkbox("Entity Extraction", value=True)
            semantic_analysis = st.checkbox("Semantic Analysis", value=True)
    
    # Agent selection
    st.markdown("#### 🤖 AI Agents")
    agent_cols = st.columns(4)
    with agent_cols[0]:
        vision_agent = st.checkbox("Vision Agent", value=True)
    with agent_cols[1]:
        text_agent = st.checkbox("Text Agent", value=True)
    with agent_cols[2]:
        fusion_agent = st.checkbox("Fusion Agent", value=True)
    with agent_cols[3]:
        validation_agent = st.checkbox("Validation Agent", value=True)
    
    # Start Processing
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Processing All Documents", type="primary", use_container_width=True):
            st.session_state['current_step'] = "processing"
            st.session_state['current_document_index'] = 0
            st.rerun()

# ========== PROCESSING VIEW ==========
def render_processing_view(ui: DocumentIntelligenceUI):
    """Render real-time processing view"""
    render_step_indicator("process")
    
    if 'document_ids' not in st.session_state:
        st.error("No documents to process. Please upload documents first.")
        if st.button("Go to Upload"):
            st.session_state['current_step'] = "upload"
            st.rerun()
        return
    
    document_ids = st.session_state['document_ids']
    current_index = st.session_state.get('current_document_index', 0)
    
    if current_index >= len(document_ids):
        # All documents processed
        st.success("✅ All documents processed successfully!")
        if st.button("View Results", type="primary"):
            st.session_state['current_step'] = "results"
            st.session_state['current_document_id'] = document_ids[0]
            st.rerun()
        return
    
    current_document_id = document_ids[current_index]
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>🔄 Processing Document {current_index + 1} of {len(document_ids)}</h2>
        <p style="color: #94a3b8;">AI agents are analyzing your document</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status polling
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Agent progress tracking
    agents = [
        {"name": "Preprocessing", "icon": "🔍"},
        {"name": "Vision Agent", "icon": "👁️"},
        {"name": "Text Agent", "icon": "📝"},
        {"name": "Fusion Agent", "icon": "🔄"},
        {"name": "Validation Agent", "icon": "✅"}
    ]
    
    start_time = time.time()
    max_polls = 120  # 2 minutes timeout
    
    for poll_count in range(max_polls):
        status_data = ui.get_processing_status(current_document_id)
        current_status = status_data.get('status', 'unknown')
        
        # Update progress
        if current_status == 'uploaded':
            progress = 0.1
        elif current_status == 'processing':
            progress = 0.3 + (poll_count % 20) * 0.02
        elif current_status == 'completed':
            progress = 1.0
        elif current_status == 'error':
            progress = 1.0
        else:
            progress = min(0.2 + (poll_count * 0.01), 0.9)
        
        # Update UI
        elapsed = time.time() - start_time
        
        with status_placeholder.container():
            st.markdown(f"### Status: **{current_status.upper()}**")
            st.markdown(f"**Document ID:** `{current_document_id[:12]}...`")
            st.markdown(f"**Elapsed Time:** {elapsed:.1f}s")
            
            # Agent progress
            st.markdown("#### Agents Progress")
            cols = st.columns(len(agents))
            for i, agent in enumerate(agents):
                with cols[i]:
                    agent_progress = min(progress * len(agents), i + 1) / len(agents)
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 24px;">{agent['icon']}</div>
                        <div style="font-size: 12px; margin: 5px 0;">{agent['name']}</div>
                        <div style="height: 4px; background: #2d3746; border-radius: 2px; margin: 5px 0;">
                            <div style="height: 100%; width: {agent_progress*100}%; background: #4f46e5; border-radius: 2px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        progress_bar.progress(progress)
        
        if current_status in ['completed', 'error']:
            break
        
        time.sleep(1)
        st.rerun()
    
    # Final status
    if current_status == 'completed':
        st.success(f"✅ Document {current_index + 1} processed successfully!")
        
        # Fetch results
        with st.spinner("Loading results..."):
            results = ui.get_results(current_document_id)
            if results:
                if 'results' not in st.session_state:
                    st.session_state['results'] = {}
                st.session_state['results'][current_document_id] = results
                
                # Add to recent documents
                if 'recent_docs' not in st.session_state:
                    st.session_state['recent_docs'] = []
                
                st.session_state['recent_docs'].insert(0, {
                    'id': current_document_id,
                    'name': f"Document_{current_document_id[:8]}",
                    'timestamp': datetime.now().isoformat()
                })
                st.session_state['recent_docs'] = st.session_state['recent_docs'][:10]
        
        # Move to next document or finish
        if current_index + 1 < len(document_ids):
            st.session_state['current_document_index'] = current_index + 1
            st.rerun()
        else:
            # All documents processed
            st.success("✅ All documents processed successfully!")
            if st.button("📊 View Results", type="primary", use_container_width=True):
                st.session_state['current_step'] = "results"
                st.session_state['current_document_id'] = document_ids[0]
                st.rerun()
    
    elif current_status == 'error':
        error_msg = status_data.get('error', 'Unknown error')
        st.error(f"❌ Processing failed: {error_msg}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("📤 Upload New", use_container_width=True):
                st.session_state['current_step'] = "upload"
                st.rerun()
    
    else:  # Timeout
        st.warning("⏳ Processing is taking longer than expected")
        if st.button("Check Status Again", key="check_again"):
            st.rerun()

# ========== RESULTS VIEW ==========
def render_results_view(ui: DocumentIntelligenceUI):
    """Render structured results view"""
    render_step_indicator("results")
    
    if 'current_document_id' not in st.session_state:
        st.error("No document selected. Please process a document first.")
        if st.button("Go to Upload"):
            st.session_state['current_step'] = "upload"
            st.rerun()
        return
    
    document_id = st.session_state['current_document_id']
    
    # Load results
    if 'results' not in st.session_state or document_id not in st.session_state['results']:
        with st.spinner("Loading results..."):
            results = ui.get_results(document_id)
            if results:
                if 'results' not in st.session_state:
                    st.session_state['results'] = {}
                st.session_state['results'][document_id] = results
            else:
                st.error("Results not found. Please process the document first.")
                if st.button("Go to Process"):
                    st.session_state['current_step'] = "process"
                    st.rerun()
                return
    
    results = st.session_state['results'][document_id]
    
    # Header
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>📊 Analysis Results</h2>
        <p style="color: #94a3b8;">Document ID: <code>{document_id[:12]}...</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_conf = results.get('overall_confidence', 0.85) * 100
        st.metric("Confidence", f"{overall_conf:.0f}%")
    
    with col2:
        elements = len(results.get('visual_elements', []))
        st.metric("Visual Elements", elements)
    
    with col3:
        fields = len(results.get('extracted_fields', {}))
        st.metric("Fields Extracted", fields)
    
    with col4:
        issues = len(results.get('contradictions', []))
        st.metric("Issues Found", issues, delta_color="inverse")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary",
        "🤖 Agent Outputs",
        "👁️ Visual Analysis",
        "📝 Extracted Content",
        "✅ Validation"
    ])
    
    # Tab 1: Summary
    with tab1:
        st.markdown("### 📋 Document Summary")
        
        if 'document_type' in results:
            st.markdown(f"**Document Type:** `{results['document_type']}`")
        
        if 'processing_time' in results:
            st.markdown(f"**Processing Time:** {results['processing_time']}s")
        
        # Agent Status
        if 'agent_outputs' in results:
            st.markdown("#### 🤖 Agent Results")
            for agent_name, agent_data in results['agent_outputs'].items():
                with st.expander(f"{agent_data.get('name', agent_name.title())} {agent_data.get('icon', '')}"):
                    status = agent_data.get('status', 'completed')
                    confidence = agent_data.get('confidence', 0) * 100
                    
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        status_badge = "🟢" if status == 'completed' else "🟡" if status == 'processing' else "🔴"
                        st.markdown(f"**Status:** {status_badge} {status}")
                    with col_b:
                        st.progress(confidence/100, text=f"{confidence:.1f}% confidence")
                    
                    # Key findings
                    findings = agent_data.get('key_findings', [])
                    if findings:
                        st.markdown("**Key Findings:**")
                        for finding in findings[:5]:
                            st.markdown(f"- {finding}")
    
    # Tab 2: Agent Outputs
    with tab2:
        st.markdown("### 🤖 Agent Outputs")
        
        if 'agent_outputs' in results:
            for agent_name, agent_data in results['agent_outputs'].items():
                with st.expander(f"{agent_data.get('name', agent_name.title())} - {agent_data.get('description', '')}"):
                    st.json(agent_data)
        else:
            st.info("No agent outputs available")
    
    # Tab 3: Visual Analysis
    with tab3:
        st.markdown("### 👁️ Visual Element Detection")
        
        visual_elements = results.get('visual_elements', [])
        
        if visual_elements:
            # Group by type
            element_types = {}
            for element in visual_elements:
                elem_type = element.get('type', 'unknown')
                element_types[elem_type] = element_types.get(elem_type, 0) + 1
            
            # Pie chart
            if element_types:
                df = pd.DataFrame({
                    'Type': list(element_types.keys()),
                    'Count': list(element_types.values())
                })
                fig = px.pie(df, values='Count', names='Type', 
                            title="Detected Element Types",
                            color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            
            # Element list
            st.markdown("#### Detected Elements")
            for element in visual_elements[:10]:
                with st.expander(f"{element.get('type', 'Unknown').title()} - Confidence: {element.get('confidence', 0)*100:.1f}%"):
                    st.json(element)
        else:
            st.info("No visual elements detected")
    
    # Tab 4: Extracted Content
    with tab4:
        st.markdown("### 📝 Extracted Text & Data")
        
        # Text content
        text_content = results.get('extracted_text', '')
        if text_content:
            with st.expander("📄 Full Text", expanded=False):
                st.text_area("", text_content[:5000], height=200, 
                           label_visibility="collapsed")
        
        # Structured data
        extracted_fields = results.get('extracted_fields', {})
        if extracted_fields:
            st.markdown("#### 📊 Structured Data")
            
            # Convert to dataframe for better display
            rows = []
            for field_name, field_data in extracted_fields.items():
                rows.append({
                    "Field": field_name,
                    "Value": str(field_data.get('value', '')),
                    "Confidence": f"{field_data.get('confidence', 0)*100:.1f}%",
                    "Sources": ", ".join(field_data.get('sources', []))[:30]
                })
            
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No structured data extracted")
    
    # Tab 5: Validation
    with tab5:
        st.markdown("### ✅ Validation & Quality Check")
        
        contradictions = results.get('contradictions', [])
        
        if contradictions:
            st.warning(f"Found {len(contradictions)} potential issue(s)")
            
            for i, contradiction in enumerate(contradictions[:5]):
                severity = contradiction.get('severity', 'medium')
                severity_color = {
                    'low': '🟢',
                    'medium': '🟡', 
                    'high': '🟠',
                    'critical': '🔴'
                }.get(severity, '⚪')
                
                with st.expander(f"{severity_color} Issue {i+1}: {contradiction.get('type', 'Unknown')}"):
                    st.markdown(f"**Description:** {contradiction.get('description', 'No description')}")
                    st.markdown(f"**Severity:** {severity}")
                    st.markdown(f"**Confidence:** {contradiction.get('confidence', 0)*100:.1f}%")
                    
                    if contradiction.get('recommendation'):
                        st.info(f"**Recommendation:** {contradiction['recommendation']}")
        else:
            st.success("✅ No validation issues found")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Return Home", use_container_width=True):
            st.session_state.clear()
            st.session_state['current_step'] = "home"
            st.rerun()
    
    with col2:
        if st.button("🔍 Query Document", type="primary", use_container_width=True):
            st.session_state['current_step'] = "query"
            st.rerun()
    
    with col3:
        if st.button("📥 Export Results", use_container_width=True):
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name=f"analysis_{document_id}.json",
                mime="application/json",
                use_container_width=True
            )

# ========== QUERY STEP ==========
def render_query_step(ui: DocumentIntelligenceUI):
    """Render query interface"""
    render_step_indicator("query")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>🔍 Query Your Document</h2>
        <p style="color: #94a3b8;">Ask questions about the analyzed content</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'current_document_id' not in st.session_state:
        st.warning("No document selected. Please process a document first.")
        if st.button("Go to Upload", key="goto_upload_from_query"):
            st.session_state['current_step'] = "upload"
            st.rerun()
        return
    
    document_id = st.session_state['current_document_id']
    
    # Document selector if multiple documents
    if 'document_ids' in st.session_state and len(st.session_state['document_ids']) > 1:
        doc_options = {f"Document {i+1} ({doc_id[:8]}...)": doc_id 
                      for i, doc_id in enumerate(st.session_state['document_ids'])}
        selected_doc = st.selectbox("Select Document:", list(doc_options.keys()))
        document_id = doc_options[selected_doc]
    
    # Query interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Ask a question about your document:",
            placeholder="e.g., 'Show me all revenue figures' or 'Find signature locations'",
            key="query_input"
        )
    with col2:
        query_type = st.selectbox(
            "Query Type",
            ["Semantic", "Visual", "Mixed"],
            help="Semantic: text-based, Visual: image-based, Mixed: both"
        )
    
    if query and st.button("🔍 Search", type="primary", use_container_width=True):
        with st.spinner("Analyzing document..."):
            response = ui.query_document(document_id, query)
            
            if response and response.get('success'):
                st.success("✅ Answer found!")
                
                # Display answer
                st.markdown("### 🤖 AI Answer")
                st.info(response.get('answer', 'No answer provided'))
                
                # Confidence score
                confidence = response.get('confidence', 0) * 100
                st.metric("Answer Confidence", f"{confidence:.1f}%")
                
                # Sources
                sources = response.get('sources', [])
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")
                
                # Supporting evidence
                evidence = response.get('supporting_evidence', [])
                if evidence:
                    with st.expander("🔍 Supporting Evidence"):
                        for ev in evidence:
                            st.markdown(f"- {ev}")
            else:
                st.warning("No answer found. Try rephrasing your question.")
    
    # Example queries
    st.markdown("---")
    st.markdown("#### 💡 Example Questions")
    
    examples = [
        "What are the main findings in this document?",
        "Show me all tables related to financial data",
        "Are there any signatures in the document?",
        "Extract all dates mentioned",
        "What contradictions were found?",
        "Summarize the document in 3 bullet points"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state['query_input'] = example
                st.rerun()

# ========== MAIN APP ==========
def main():
    """Main Streamlit application with state machine"""
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = "home"
    if 'current_document_id' not in st.session_state:
        st.session_state['current_document_id'] = None
    if 'document_ids' not in st.session_state:
        st.session_state['document_ids'] = []
    if 'results' not in st.session_state:
        st.session_state['results'] = {}
    if 'recent_docs' not in st.session_state:
        st.session_state['recent_docs'] = []
    
    # Initialize UI handler
    ui = DocumentIntelligenceUI()
    
    # Always render sidebar
    render_sidebar(ui)
    
    # State machine for main content
    current_step = st.session_state['current_step']
    
    if current_step == "home":
        render_home_page(ui)
    
    elif current_step == "upload":
        render_upload_step(ui)
    
    elif current_step == "process":
        render_process_step(ui)
    
    elif current_step == "processing":
        render_processing_view(ui)
    
    elif current_step == "results":
        render_results_view(ui)
    
    elif current_step == "query":
        render_query_step(ui)
    
    else:
        # Fallback to home
        st.session_state['current_step'] = "home"
        st.rerun()

if __name__ == "__main__":
    main()