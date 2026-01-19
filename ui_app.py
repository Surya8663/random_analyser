# ui_app.py
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
    
    /* Cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(79, 70, 229, 0.4);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    .feature-card .icon {
        font-size: 40px;
        margin-bottom: 15px;
        display: block;
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
        min-width: 300px !important;
        max-width: 350px !important;
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
    
    def upload_document(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """Upload document to backend"""
        try:
            files = {'file': (filename, file_bytes)}
            response = requests.post(UPLOAD_ENDPOINT, files=files)
            
            if response.status_code == 200:
                data = response.json()
                document_id = data.get('document_id')
                st.session_state['document_id'] = document_id
                return document_id
            else:
                st.error(f"Upload failed: {response.text}")
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
    """Render sidebar with grouped controls"""
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
        
        # Step 1: Document Upload
        st.markdown("### 📤 1. Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF or image",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Supported formats: PDF, PNG, JPG, JPEG",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # File preview
            with st.expander("📄 File Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Filename", uploaded_file.name[:20] + ("..." if len(uploaded_file.name) > 20 else ""))
                with col2:
                    st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['current_step'] = "upload"
            
            if st.button("✅ Confirm Upload", type="primary", use_container_width=True):
                st.session_state['current_step'] = "process"
                st.rerun()
        
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

# ========== HOME PAGE ==========
def render_home_page(ui: DocumentIntelligenceUI):
    """Render landing/home page"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="font-size: 48px; margin-bottom: 20px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Document Intelligence AI
        </h1>
        <p style="font-size: 18px; color: #94a3b8; max-width: 800px; margin: 0 auto 40px;">
            Transform documents into actionable insights using AI-powered computer vision, 
            OCR, and multi-agent analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("### 🚀 How It Works")
    
    steps = [
        {"icon": "📤", "title": "Upload", "desc": "Upload PDF or image documents"},
        {"icon": "⚙️", "title": "Process", "desc": "AI agents analyze content"},
        {"icon": "📊", "title": "Review", "desc": "View structured insights"},
        {"icon": "🔍", "title": "Query", "desc": "Ask questions about content"}
    ]
    
    cols = st.columns(4)
    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: rgba(79, 70, 229, 0.05); border-radius: 12px;">
                <div style="font-size: 36px; margin-bottom: 15px;">{step['icon']}</div>
                <h4 style="margin: 0 0 10px 0;">{step['title']}</h4>
                <p style="color: #94a3b8; font-size: 14px; margin: 0;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Cards (Now clickable)
    st.markdown("### 🎯 Key Features")
    
    features = [
        {
            "icon": "👁️",
            "title": "Computer Vision",
            "desc": "Detect tables, charts, signatures",
            "action": "vision",
            "color": "#4f46e5"
        },
        {
            "icon": "📝",
            "title": "OCR Intelligence",
            "desc": "Extract text with confidence scoring",
            "action": "ocr",
            "color": "#f59e0b"
        },
        {
            "icon": "🤖",
            "title": "Multi-Agent System",
            "desc": "Validate & cross-check information",
            "action": "agents",
            "color": "#10b981"
        }
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i]:
            if st.button(
                f"**{feature['icon']} {feature['title']}**\n\n{feature['desc']}",
                key=f"feature_{feature['action']}",
                use_container_width=True,
                help=f"Click to learn more about {feature['title']}"
            ):
                st.session_state['feature_view'] = feature['action']
                st.rerun()
    
    # Get Started Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started →", type="primary", use_container_width=True, key="get_started"):
            st.session_state['current_step'] = "upload"
            st.rerun()

# ========== UPLOAD STEP ==========
def render_upload_step(ui: DocumentIntelligenceUI):
    """Render upload step"""
    render_step_indicator("upload")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>📤 Upload Your Document</h2>
        <p style="color: #94a3b8;">Start by uploading a PDF or image file for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="border: 2px dashed #4f46e5; border-radius: 12px; padding: 40px; text-align: center; background: rgba(79, 70, 229, 0.05);">
            <div style="font-size: 48px; margin-bottom: 20px;">📤</div>
            <h3 style="margin-bottom: 10px;">Drag & Drop</h3>
            <p style="color: #94a3b8; margin-bottom: 20px;">or click to browse files</p>
            <p style="color: #64748b; font-size: 12px;">Supported: PDF, PNG, JPG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File will be handled by sidebar
    st.info("👈 Use the sidebar to upload your document")

# ========== PROCESS STEP ==========
def render_process_step(ui: DocumentIntelligenceUI):
    """Render processing step"""
    render_step_indicator("process")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>⚙️ Process Document</h2>
        <p style="color: #94a3b8;">Configure processing options and start analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'uploaded_file' not in st.session_state:
        st.warning("Please upload a document first")
        if st.button("Go to Upload", key="go_to_upload"):
            st.session_state['current_step'] = "upload"
            st.rerun()
        return
    
    # Processing Options
    with st.expander("⚙️ Processing Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👁️ Vision Analysis")
            layout_detection = st.checkbox("Layout Detection", value=True, 
                                          help="Detect tables, figures, charts")
            signature_detection = st.checkbox("Signature Detection", value=True)
        
        with col2:
            st.markdown("#### 📝 Text Analysis")
            ocr_enabled = st.checkbox("OCR Extraction", value=True)
            semantic_analysis = st.checkbox("Semantic Analysis", value=True)
    
    # Start Processing
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            with st.spinner("Uploading document..."):
                uploaded_file = st.session_state['uploaded_file']
                document_id = ui.upload_document(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
            
            if document_id:
                st.session_state['document_id'] = document_id
                st.session_state['current_step'] = "processing"
                st.rerun()

# ========== PROCESSING VIEW ==========
def render_processing_view(ui: DocumentIntelligenceUI, document_id: str):
    """Render real-time processing view"""
    render_step_indicator("process")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>🔄 Processing Document</h2>
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
        status_data = ui.get_processing_status(document_id)
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
            st.markdown(f"**Document ID:** `{document_id[:12]}...`")
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
        st.success("✅ Processing completed successfully!")
        
        # Fetch results
        with st.spinner("Loading results..."):
            results = ui.get_results(document_id)
            if results:
                st.session_state['results'] = results
                # Add to recent documents
                if 'recent_docs' not in st.session_state:
                    st.session_state['recent_docs'] = []
                
                st.session_state['recent_docs'].insert(0, {
                    'id': document_id,
                    'name': f"Document_{document_id[:8]}",
                    'timestamp': datetime.now().isoformat()
                })
                st.session_state['recent_docs'] = st.session_state['recent_docs'][:10]
        
        if st.button("📊 View Results", type="primary", use_container_width=True):
            st.session_state['current_step'] = "results"
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
def render_results_view(ui: DocumentIntelligenceUI, results: Dict):
    """Render structured results view"""
    render_step_indicator("results")
    
    document_id = results.get('document_id', 'Unknown')
    
    # Header
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2>📊 Analysis Results</h2>
        <p style="color: #94a3b8;">Document ID: <code>{document_id[:12]}...</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall Score Card
    if 'results' in results:
        result_data = results['results']
        
        # Overall confidence
        overall_conf = result_data.get('overall_confidence', 0.85) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div style="text-align: center; background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%); border-radius: 12px; padding: 20px;">
                <div style="font-size: 32px; font-weight: bold; color: #4f46e5;">{overall_conf:.0f}%</div>
                <div style="color: #94a3b8; font-size: 14px;">Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            elements = len(result_data.get('visual_elements', []))
            st.metric("Visual Elements", elements, help="Tables, charts, figures detected")
        
        with col3:
            fields = len(result_data.get('extracted_fields', {}))
            st.metric("Fields Extracted", fields, help="Structured data fields")
        
        with col4:
            issues = len(result_data.get('contradictions', []))
            st.metric("Issues Found", issues, delta_color="inverse")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary",
        "👁️ Visual Analysis",
        "📝 Extracted Content",
        "✅ Validation"
    ])
    
    # Tab 1: Summary
    with tab1:
        st.markdown("### 📋 Document Summary")
        
        if 'results' in results:
            summary_data = results['results']
            
            # Document Type
            doc_type = summary_data.get('document_type', 'Unknown')
            st.markdown(f"**Document Type:** `{doc_type}`")
            
            # Processing Time
            if 'processing_metadata' in summary_data:
                metadata = summary_data['processing_metadata']
                st.markdown(f"**Processing Time:** {metadata.get('duration_seconds', 'N/A')}s")
            
            # Agent Status
            st.markdown("#### 🤖 Agent Results")
            
            agent_outputs = summary_data.get('agent_outputs', {})
            for agent_name, agent_data in agent_outputs.items():
                with st.expander(f"{agent_name.title()} Agent"):
                    status = agent_data.get('status', 'completed')
                    status_badge = "🟢" if status == 'completed' else "🟡" if status == 'processing' else "🔴"
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**Status:** {status_badge} {status}")
                    with col2:
                        confidence = agent_data.get('confidence', 0) * 100
                        st.progress(confidence/100, text=f"{confidence:.1f}% confidence")
                    
                    # Key findings
                    findings = agent_data.get('key_findings', [])
                    if findings:
                        st.markdown("**Key Findings:**")
                        for finding in findings[:5]:  # Show only top 5
                            st.markdown(f"- {finding}")
    
    # Tab 2: Visual Analysis
    with tab2:
        st.markdown("### 👁️ Visual Element Detection")
        
        visual_elements = results.get('results', {}).get('visual_elements', [])
        
        if visual_elements:
            # Group by type
            element_types = {}
            for element in visual_elements:
                elem_type = element.get('type', 'unknown')
                if elem_type not in element_types:
                    element_types[elem_type] = 0
                element_types[elem_type] += 1
            
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
            for element in visual_elements[:10]:  # Limit display
                with st.expander(f"{element.get('type', 'Unknown').title()} - Confidence: {element.get('confidence', 0)*100:.1f}%"):
                    st.json(element)
        else:
            st.info("No visual elements detected")
    
    # Tab 3: Extracted Content
    with tab3:
        st.markdown("### 📝 Extracted Text & Data")
        
        # Text content
        text_content = results.get('results', {}).get('extracted_text', '')
        if text_content:
            with st.expander("📄 Full Text", expanded=False):
                st.text_area("", text_content[:5000], height=200, 
                           label_visibility="collapsed")
        
        # Structured data
        extracted_fields = results.get('results', {}).get('extracted_fields', {})
        if extracted_fields:
            st.markdown("#### 📊 Structured Data")
            
            # Convert to dataframe for better display
            rows = []
            for field_name, field_data in extracted_fields.items():
                rows.append({
                    "Field": field_name,
                    "Value": str(field_data.get('value', '')),
                    "Confidence": f"{field_data.get('confidence', 0)*100:.1f}%",
                    "Source": ", ".join(field_data.get('sources', []))[:30]
                })
            
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No structured data extracted")
    
    # Tab 4: Validation
    with tab4:
        st.markdown("### ✅ Validation & Quality Check")
        
        contradictions = results.get('results', {}).get('contradictions', [])
        
        if contradictions:
            st.warning(f"Found {len(contradictions)} potential issue(s)")
            
            for i, contradiction in enumerate(contradictions[:5]):  # Limit display
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
        if st.button("🔄 Process New", use_container_width=True):
            st.session_state.clear()
            st.session_state['current_step'] = "upload"
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
    
    if 'document_id' not in st.session_state:
        st.warning("No document selected. Please process a document first.")
        if st.button("Go to Upload", key="goto_upload_from_query"):
            st.session_state['current_step'] = "upload"
            st.rerun()
        return
    
    document_id = st.session_state['document_id']
    
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
    if 'document_id' not in st.session_state:
        st.session_state['document_id'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = None
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
    
    elif current_step == "processing" and 'document_id' in st.session_state:
        render_processing_view(ui, st.session_state['document_id'])
    
    elif current_step == "results" and 'results' in st.session_state:
        render_results_view(ui, st.session_state['results'])
    
    elif current_step == "query":
        render_query_step(ui)
    
    else:
        # Fallback to home
        st.session_state['current_step'] = "home"
        st.rerun()

if __name__ == "__main__":
    main()