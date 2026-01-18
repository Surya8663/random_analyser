# ui_app.py
import streamlit as st
import requests
import time
import json
import base64
from PIL import Image
import io
import os
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import uuid

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Document Intelligence",
    page_icon="📄",
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
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Cards */
    .card {
        background-color: #1e2229;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #4f8bf9;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4f8bf9;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e2229;
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
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2229;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4f8bf9;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #1e2229;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
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

def render_sidebar(ui: DocumentIntelligenceUI):
    """Render sidebar with controls"""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 24px; margin-bottom: 5px;">📄</h1>
            <h2 style="font-size: 18px; margin-bottom: 0;">Multi-Modal Document Intelligence</h2>
            <p style="font-size: 12px; color: #888; margin-top: 5px;">AI Agents Builder System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # API Status
        st.markdown("### System Status")
        api_status = ui.check_api_health()
        status_color = "🟢" if api_status else "🔴"
        st.markdown(f"{status_color} Backend API: {'Connected' if api_status else 'Disconnected'}")
        
        if not api_status:
            st.warning("Backend server not detected. Please run: `uvicorn app.main:app --reload`")
        
        # Upload Section
        st.markdown("---")
        st.markdown("### 📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload PDF or image files for analysis"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "Type": uploaded_file.type
            }
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Selected File:**")
                st.write(file_details)
            with col2:
                st.markdown("")
                if st.button("Clear", key="clear_file"):
                    st.session_state.pop('uploaded_file', None)
                    st.rerun()
            
            # Processing Options
            st.markdown("### ⚙️ Processing Options")
            
            col1, col2 = st.columns(2)
            with col1:
                layout_detection = st.checkbox("Layout Detection", value=True, 
                                               help="Detect tables, figures, charts, signatures")
                ocr_validation = st.checkbox("OCR Validation", value=True,
                                             help="Validate OCR confidence scores")
            with col2:
                multi_agent = st.checkbox("Multi-Agent", value=True,
                                          help="Enable vision, text, fusion, and validation agents")
                rag_indexing = st.checkbox("RAG Indexing", value=True,
                                           help="Index document for semantic search")
            
            # Process Button
            if st.button("🚀 Process Document", use_container_width=True, type="primary"):
                if api_status:
                    with st.spinner("Uploading document..."):
                        document_id = ui.upload_document(
                            uploaded_file.getvalue(),
                            uploaded_file.name
                        )
                    
                    if document_id:
                        st.session_state['processing_started'] = True
                        st.session_state['document_id'] = document_id
                        st.rerun()
                else:
                    st.error("Cannot process: Backend API is not connected")
        
        # Recent Documents
        st.markdown("---")
        st.markdown("### 📁 Recent Documents")
        
        if 'recent_docs' in st.session_state and st.session_state['recent_docs']:
            for doc in st.session_state['recent_docs'][:5]:
                if st.button(f"📄 {doc.get('name', 'Document')}", key=f"recent_{doc.get('id')}"):
                    st.session_state['document_id'] = doc['id']
                    st.rerun()
        else:
            st.caption("No recent documents")
        
        # System Info
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        
        with st.expander("About"):
            st.markdown("""
            **Multi-Modal Document Intelligence System**
            
            Powered by:
            - Computer Vision (YOLO/OpenCV)
            - OCR Engine (Tesseract)
            - Multi-Agent System
            - Multi-Modal RAG (Qdrant)
            - Confidence Scoring Engine
            
            Competition: AI Agents Builder System
            """)

def render_processing_view(ui: DocumentIntelligenceUI, document_id: str):
    """Render real-time processing view"""
    st.markdown("## 🔄 Document Processing")
    
    # Status polling
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_elapsed = st.empty()
    
    start_time = time.time()
    status_messages = {
        "uploaded": "Document uploaded successfully",
        "processing": "Processing document with AI agents",
        "completed": "Processing completed",
        "error": "Processing failed"
    }
    
    status_icons = {
        "uploaded": "📤",
        "processing": "⚙️",
        "completed": "✅",
        "error": "❌"
    }
    
    # Poll for status updates
    max_polls = 60  # 60 seconds timeout
    poll_count = 0
    
    while poll_count < max_polls:
        status_data = ui.get_processing_status(document_id)
        current_status = status_data.get('status', 'unknown')
        
        # Update progress based on status
        if current_status == 'uploaded':
            progress = 0.2
        elif current_status == 'processing':
            progress = 0.5 + (poll_count % 10) * 0.05  # Animated progress
        elif current_status == 'completed':
            progress = 1.0
            break
        elif current_status == 'error':
            progress = 1.0
            break
        else:
            progress = min(0.1 + (poll_count * 0.01), 0.9)
        
        # Update UI
        elapsed = time.time() - start_time
        
        with status_placeholder.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                icon = status_icons.get(current_status, "⏳")
                st.markdown(f"<h1 style='font-size: 48px; text-align: center;'>{icon}</h1>", 
                           unsafe_allow_html=True)
            with col2:
                st.markdown(f"### {status_messages.get(current_status, 'Processing...')}")
                st.markdown(f"**Document ID:** `{document_id}`")
        
        progress_bar.progress(progress)
        status_text.markdown(f"**Status:** {current_status}")
        time_elapsed.markdown(f"**Time elapsed:** {elapsed:.1f}s")
        
        if current_status in ['completed', 'error']:
            break
        
        time.sleep(1)  # Poll every second
        poll_count += 1
        st.rerun()
    
    # Final status
    if poll_count >= max_polls:
        st.error("Processing timeout - document taking too long to process")
    elif current_status == 'error':
        error_msg = status_data.get('error', 'Unknown error')
        st.error(f"Processing failed: {error_msg}")
    elif current_status == 'completed':
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
                # Keep only last 10
                st.session_state['recent_docs'] = st.session_state['recent_docs'][:10]
        
        # Show continue button
        if st.button("View Results", type="primary", use_container_width=True):
            st.session_state['show_results'] = True
            st.rerun()

def render_results_view(ui: DocumentIntelligenceUI, results: Dict):
    """Render comprehensive results view"""
    document_id = results.get('document_id', 'Unknown')
    
    # Header with metrics
    st.markdown(f"## 📊 Analysis Results - `{document_id[:12]}...`")
    
    # Overall confidence score
    if 'results' in results and 'confidence_scores' in results['results']:
        confidence_scores = results['results']['confidence_scores']
        overall_score = confidence_scores.get('overall', 0) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: white; margin: 0;'>{:.1f}%</h3>
                <p style='color: white; margin: 0; font-size: 12px;'>Overall Confidence</p>
            </div>
            """.format(overall_score), unsafe_allow_html=True)
        
        with col2:
            ocr_score = confidence_scores.get('ocr', 0) * 100
            st.metric("OCR Confidence", f"{ocr_score:.1f}%")
        
        with col3:
            vision_score = confidence_scores.get('vision', 0) * 100
            st.metric("Vision Confidence", f"{vision_score:.1f}%")
        
        with col4:
            fusion_score = confidence_scores.get('fusion', 0) * 100
            st.metric("Fusion Confidence", f"{fusion_score:.1f}%")
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Extracted Text",
        "👁️ Visual Analysis",
        "📊 Structured Data",
        "🎯 Confidence Breakdown",
        "🔍 Multi-Modal Query"
    ])
    
    # Tab 1: Extracted Text
    with tab1:
        st.markdown("### Text Extraction Results")
        
        if 'results' in results and 'extracted_text' in results['results']:
            extracted_text = results['results']['extracted_text']
            
            # Text preview with confidence highlighting
            if isinstance(extracted_text, dict):
                text_content = extracted_text.get('content', '')
                confidence_map = extracted_text.get('confidence_map', {})
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("#### Text Content")
                    st.text_area("Extracted Text", text_content, height=300)
                
                with col2:
                    st.markdown("#### Confidence Distribution")
                    if confidence_map:
                        conf_df = pd.DataFrame(list(confidence_map.items()), 
                                              columns=['Confidence Range', 'Percentage'])
                        fig = px.bar(conf_df, x='Confidence Range', y='Percentage',
                                    title="Text Confidence Distribution")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.text_area("Extracted Text", str(extracted_text), height=300)
        else:
            st.info("No extracted text available")
    
    # Tab 2: Visual Analysis
    with tab2:
        st.markdown("### Visual Element Detection")
        
        # Check for visual elements data
        if 'results' in results and 'visual_elements' in results['results']:
            visual_elements = results['results']['visual_elements']
            
            # Display detected elements in a grid
            if visual_elements:
                cols = st.columns(3)
                element_types = {}
                
                for i, element in enumerate(visual_elements):
                    with cols[i % 3]:
                        element_type = element.get('type', 'unknown')
                        confidence = element.get('confidence', 0) * 100
                        
                        if element_type not in element_types:
                            element_types[element_type] = 0
                        element_types[element_type] += 1
                        
                        st.markdown(f"""
                        <div style='background-color: #1e2229; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                            <div style='font-size: 24px; margin-bottom: 10px;'>
                                {"📊" if element_type == 'table' else 
                                 "📈" if element_type == 'chart' else 
                                 "🖼️" if element_type == 'figure' else 
                                 "✍️" if element_type == 'signature' else "🔍"}
                            </div>
                            <h4 style='margin: 0;'>{element_type.title()}</h4>
                            <p style='margin: 5px 0; font-size: 12px; color: #888;'>
                                Confidence: {confidence:.1f}%
                            </p>
                            <p style='margin: 0; font-size: 12px; color: #888;'>
                                Bounds: {element.get('bounds', 'N/A')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Summary statistics
                st.markdown("### 📈 Detection Summary")
                if element_types:
                    summary_df = pd.DataFrame({
                        'Element Type': list(element_types.keys()),
                        'Count': list(element_types.values())
                    })
                    
                    fig = px.pie(summary_df, values='Count', names='Element Type',
                                title="Detected Element Distribution",
                                hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No visual elements detected")
        else:
            st.info("Visual analysis data not available")
            
            # Placeholder for annotated image
            st.markdown("#### Annotated Document Preview")
            st.caption("Note: Enable visualization endpoint in backend to see annotated images")
    
    # Tab 3: Structured Data
    with tab3:
        st.markdown("### Structured Data Extraction")
        
        if 'results' in results and 'structured_data' in results['results']:
            structured_data = results['results']['structured_data']
            
            # Convert to DataFrame for display
            if isinstance(structured_data, list):
                df = pd.DataFrame(structured_data)
                st.dataframe(df, use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"structured_data_{document_id}.csv",
                    mime="text/csv"
                )
            elif isinstance(structured_data, dict):
                # Display as key-value pairs
                for key, value in structured_data.items():
                    with st.expander(f"{key}"):
                        if isinstance(value, dict):
                            st.json(value)
                        elif isinstance(value, list):
                            st.write(pd.DataFrame(value))
                        else:
                            st.write(value)
            else:
                st.write(structured_data)
        else:
            st.info("No structured data available")
    
    # Tab 4: Confidence Breakdown
    with tab4:
        st.markdown("### Confidence Analysis")
        
        if 'results' in results and 'confidence_scores' in results['results']:
            confidence_scores = results['results']['confidence_scores']
            
            # Create radar chart for confidence scores
            categories = list(confidence_scores.keys())
            values = [v * 100 for v in confidence_scores.values()]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Confidence Scores'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Confidence Score Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            st.markdown("#### Detailed Confidence Metrics")
            for category, score in confidence_scores.items():
                score_percent = score * 100
                col1, col2, col3 = st.columns([2, 6, 2])
                with col1:
                    st.text(category.replace('_', ' ').title())
                with col2:
                    st.progress(score)
                with col3:
                    st.text(f"{score_percent:.1f}%")
        else:
            st.info("Confidence scores not available")
    
    # Tab 5: Multi-Modal Query
    with tab5:
        st.markdown("### 🔍 Multi-Modal Document Query")
        st.markdown("Ask questions about your document using AI-powered semantic search")
        
        # Query interface
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., 'Find all tables related to revenue' or 'Show me charts that mention profit'",
                key="rag_query"
            )
        with col2:
            query_type = st.selectbox(
                "Query Type",
                ["text", "visual", "mixed"],
                help="Text: semantic search, Visual: image-based search, Mixed: combined search"
            )
        
        if query:
            with st.spinner("Searching document..."):
                # Try document-specific query first
                response = ui.query_document(document_id, query)
                
                if response and response.get('success'):
                    answer = response.get('answer', '')
                    confidence = response.get('confidence', 0) * 100
                    
                    st.markdown("#### 🤖 AI Answer")
                    st.info(answer)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Answer Confidence", f"{confidence:.1f}%")
                    with col2:
                        st.metric("Sources", len(response.get('sources', [])))
                    
                    # Display sources if available
                    if response.get('sources'):
                        with st.expander("View Sources"):
                            for i, source in enumerate(response.get('sources', [])):
                                st.markdown(f"**Source {i+1}:** {source}")
                else:
                    # Fallback to RAG search
                    rag_response = ui.rag_search(query, query_type)
                    
                    if rag_response and rag_response.get('success'):
                        st.markdown("#### 🔍 Search Results")
                        
                        results = rag_response.get('results', [])
                        for i, result in enumerate(results[:5]):
                            with st.expander(f"Result {i+1} - Score: {result.get('score', 0):.3f}"):
                                if 'text' in result:
                                    st.markdown(f"**Text:** {result['text'][:500]}...")
                                if 'metadata' in result:
                                    st.markdown("**Metadata:**")
                                    st.json(result['metadata'])
                                if 'image_data' in result:
                                    st.markdown("**Image Data:** Available")
                    else:
                        st.warning("No results found for your query")
        
        # Example queries
        st.markdown("---")
        st.markdown("#### 💡 Example Queries")
        
        example_cols = st.columns(3)
        examples = [
            "Find all tables related to revenue",
            "Show me charts that mention profit",
            "Are there inconsistencies between text and tables?",
            "Extract all numerical data from charts",
            "Identify signature locations",
            "Find contradictory information"
        ]
        
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state['rag_query'] = example
                    st.rerun()
        
        # Advanced RAG Search
        with st.expander("Advanced RAG Search"):
            st.markdown("Search across all indexed documents")
            
            adv_query = st.text_input("RAG Query:", key="adv_rag_query")
            adv_limit = st.slider("Results Limit", 1, 20, 5)
            
            if st.button("Search All Documents", key="adv_search"):
                with st.spinner("Searching indexed documents..."):
                    response = ui.rag_search(adv_query, query_type, adv_limit)
                    
                    if response and response.get('success'):
                        st.markdown(f"**Found {response.get('count', 0)} results**")
                        
                        for result in response.get('results', []):
                            with st.expander(f"Document: {result.get('document_id', 'Unknown')}"):
                                if 'text_snippet' in result:
                                    st.markdown(f"**Text:** {result['text_snippet']}")
                                if 'confidence' in result:
                                    st.markdown(f"**Confidence:** {result['confidence']:.3f}")
                                if 'visual_match' in result:
                                    st.markdown(f"**Visual Match:** {result['visual_match']}")
                    else:
                        st.warning("No results found")

def main():
    """Main Streamlit application"""
    # Initialize session state
    if 'processing_started' not in st.session_state:
        st.session_state['processing_started'] = False
    if 'show_results' not in st.session_state:
        st.session_state['show_results'] = False
    if 'recent_docs' not in st.session_state:
        st.session_state['recent_docs'] = []
    
    # Initialize UI handler
    ui = DocumentIntelligenceUI()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 42px; margin-bottom: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Multi-Modal Document Intelligence
        </h1>
        <p style="font-size: 16px; color: #888; max-width: 800px; margin: 0 auto;">
            AI-powered document analysis system combining Computer Vision, OCR, Multi-Agent Reasoning, 
            and Multi-Modal RAG for comprehensive document understanding
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar(ui)
    
    # Main content area
    if st.session_state.get('processing_started') and not st.session_state.get('show_results'):
        if 'document_id' in st.session_state:
            render_processing_view(ui, st.session_state['document_id'])
        else:
            st.error("No document ID found")
    
    elif st.session_state.get('show_results') and 'results' in st.session_state:
        render_results_view(ui, st.session_state['results'])
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Process New Document", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        with col2:
            if st.button("📊 Reprocess Document", use_container_width=True):
                if 'document_id' in st.session_state:
                    with st.spinner("Reprocessing..."):
                        success = ui.reprocess_document(st.session_state['document_id'])
                        if success:
                            st.success("Reprocessing started!")
                            st.session_state['show_results'] = False
                            st.session_state.pop('results', None)
                            st.rerun()
                        else:
                            st.error("Reprocessing failed")
        with col3:
            if st.button("📥 Export Results", use_container_width=True):
                results_json = json.dumps(st.session_state['results'], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"document_analysis_{st.session_state['document_id']}.json",
                    mime="application/json"
                )
    
    else:
        # Welcome/landing page
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px;">
            <h2 style="font-size: 32px; margin-bottom: 30px;">Welcome to Document Intelligence System</h2>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 40px 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; color: white;">
                    <div style="font-size: 36px; margin-bottom: 15px;">👁️</div>
                    <h3 style="margin: 0;">Computer Vision</h3>
                    <p style="font-size: 14px; opacity: 0.9;">Detect tables, charts, figures, signatures</p>
                </div>
                
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 15px; color: white;">
                    <div style="font-size: 36px; margin-bottom: 15px;">📝</div>
                    <h3 style="margin: 0;">OCR Intelligence</h3>
                    <p style="font-size: 14px; opacity: 0.9;">Advanced text extraction with confidence scoring</p>
                </div>
                
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 15px; color: white;">
                    <div style="font-size: 36px; margin-bottom: 15px;">🤖</div>
                    <h3 style="margin: 0;">Multi-Agent System</h3>
                    <p style="font-size: 14px; opacity: 0.9;">Vision, Text, Fusion, Validation agents</p>
                </div>
            </div>
            
            <div style="margin-top: 40px;">
                <h3 style="font-size: 24px; margin-bottom: 20px;">How to Get Started</h3>
                <ol style="text-align: left; max-width: 600px; margin: 0 auto;">
                    <li style="margin-bottom: 10px;">Upload a document (PDF or image) using the sidebar</li>
                    <li style="margin-bottom: 10px;">Configure processing options as needed</li>
                    <li style="margin-bottom: 10px;">Click "Process Document" to start analysis</li>
                    <li style="margin-bottom: 10px;">View real-time processing status</li>
                    <li style="margin-bottom: 10px;">Explore comprehensive results and query your document</li>
                </ol>
            </div>
            
            <div style="margin-top: 50px; color: #888; font-size: 14px;">
                <p>Competition: AI Agents Builder System – Multi-Modal Document Intelligence</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats if API is connected
        if ui.check_api_health():
            st.markdown("---")
            st.markdown("### System Status")
            
            try:
                # Get system info from API
                response = requests.get(f"{API_BASE_URL}/")
                if response.status_code == 200:
                    api_info = response.json()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Service", api_info.get('service', 'Unknown'))
                    with col2:
                        st.metric("Version", api_info.get('version', '1.0.0'))
                    with col3:
                        st.metric("Status", api_info.get('status', 'unknown').title())
            except:
                pass

if __name__ == "__main__":
    main()