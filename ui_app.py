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
    
    /* Sidebar width fix */
    section[data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
    }
    
    .css-1d391kg {
        min-width: 320px !important;
        max-width: 320px !important;
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
    
    /* Error messages */
    .stAlert {
        background-color: #1e2229;
        border-left: 4px solid #f56565;
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
            if response.status_code == 200:
                return True
            return False
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Make sure it's running on localhost:8000")
            return False
        except Exception as e:
            st.error(f"❌ API health check failed: {str(e)}")
            return False
    
    def upload_document(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """Upload document to backend"""
        try:
            files = {'file': (filename, file_bytes)}
            response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                document_id = data.get('document_id')
                if document_id:
                    st.session_state['document_id'] = document_id
                    return document_id
                else:
                    st.error("Upload response missing document_id")
                    return None
            else:
                error_text = response.text if response.text else f"Status code: {response.status_code}"
                st.error(f"Upload failed: {error_text}")
                return None
        except requests.exceptions.Timeout:
            st.error("Upload timeout - server taking too long to respond")
            return None
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
            return None
    
    def get_processing_status(self, document_id: str) -> Dict:
        """Get processing status from backend"""
        try:
            response = requests.get(f"{STATUS_ENDPOINT}/{document_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"Status check failed: {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"status": "api_error", "error": "Cannot connect to API"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_results(self, document_id: str) -> Optional[Dict]:
        """Get processing results from backend"""
        try:
            response = requests.get(f"{RESULTS_ENDPOINT}/{document_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                st.error(f"Failed to get results: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching results: {str(e)}")
            return None
    
    def query_document(self, document_id: str, question: str) -> Optional[Dict]:
        """Query document using RAG system"""
        try:
            payload = {
                "document_id": document_id,
                "question": question
            }
            response = requests.post(QUERY_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Query failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return None
    
    def rag_search(self, query: str, query_type: str = "text", limit: int = 5) -> Optional[Dict]:
        """Search across indexed documents"""
        try:
            payload = {
                "query": query,
                "query_type": query_type,
                "limit": limit
            }
            response = requests.post(RAG_SEARCH_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"RAG search failed: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"RAG search error: {str(e)}")
            return None
    
    def reprocess_document(self, document_id: str) -> bool:
        """Trigger reprocessing of document"""
        try:
            payload = {
                "document_id": document_id,
                "reprocess": True
            }
            response = requests.post(PROCESS_ENDPOINT, json=payload, timeout=30)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Reprocess error: {str(e)}")
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
            st.error("Backend server not detected. Please run: `uvicorn main:app --reload --port 8000`")
        
        # Upload Section
        st.markdown("---")
        st.markdown("### 📤 Upload Document")
        
        uploaded_files = st.file_uploader(
            "Choose documents (PDF or images)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload multiple files for analysis",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Display file info for all uploaded files
            for i, uploaded_file in enumerate(uploaded_files):
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Type": uploaded_file.type
                }
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**File {i+1}:** {uploaded_file.name}")
                with col2:
                    if st.button("❌", key=f"remove_{i}", help="Remove this file"):
                        uploaded_files.pop(i)
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
            if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
                if api_status:
                    processing_results = []
                    
                    with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                        for uploaded_file in uploaded_files:
                            document_id = ui.upload_document(
                                uploaded_file.getvalue(),
                                uploaded_file.name
                            )
                            if document_id:
                                processing_results.append({
                                    'id': document_id,
                                    'name': uploaded_file.name,
                                    'success': True
                                })
                    
                    if processing_results:
                        st.session_state['processing_started'] = True
                        st.session_state['document_ids'] = [r['id'] for r in processing_results]
                        st.session_state['current_doc_index'] = 0
                        st.rerun()
                else:
                    st.error("Cannot process: Backend API is not connected")
        
        # Recent Documents
        st.markdown("---")
        st.markdown("### 📁 Recent Documents")
        
        if 'recent_docs' in st.session_state and st.session_state['recent_docs']:
            for doc in st.session_state['recent_docs'][:5]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📄 {doc.get('name', 'Document')}", key=f"recent_{doc.get('id')}"):
                        st.session_state['document_id'] = doc['id']
                        st.session_state['show_results'] = True
                        st.session_state['processing_started'] = False
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{doc.get('id')}", help="Remove from recent"):
                        st.session_state['recent_docs'] = [
                            d for d in st.session_state['recent_docs'] 
                            if d['id'] != doc['id']
                        ]
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
            
            **Competition:** AI Agents Builder System
            """)
            
            # Display backend info if available
            if api_status:
                try:
                    response = requests.get(f"{API_BASE_URL}/", timeout=5)
                    if response.status_code == 200:
                        info = response.json()
                        st.markdown(f"""
                        **Backend Info:**
                        - Service: {info.get('service', 'Unknown')}
                        - Version: {info.get('version', '1.0.0')}
                        - Status: {info.get('status', 'unknown').title()}
                        """)
                except:
                    pass

def render_processing_view(ui: DocumentIntelligenceUI, document_ids: List[str]):
    """Render real-time processing view"""
    current_idx = st.session_state.get('current_doc_index', 0)
    if current_idx >= len(document_ids):
        st.session_state['processing_started'] = False
        st.session_state['show_results'] = True
        st.rerun()
    
    document_id = document_ids[current_idx]
    
    st.markdown(f"## 🔄 Document Processing ({current_idx + 1}/{len(document_ids)})")
    
    # Status polling
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_elapsed = st.empty()
    error_display = st.empty()
    
    start_time = time.time()
    status_messages = {
        "uploaded": "Document uploaded successfully",
        "processing": "Processing document with AI agents",
        "completed": "Processing completed",
        "error": "Processing failed",
        "not_found": "Document not found"
    }
    
    status_icons = {
        "uploaded": "📤",
        "processing": "⚙️",
        "completed": "✅",
        "error": "❌",
        "not_found": "🔍"
    }
    
    # Poll for status updates
    max_polls = 120  # 120 seconds timeout
    poll_count = 0
    
    while poll_count < max_polls:
        status_data = ui.get_processing_status(document_id)
        current_status = status_data.get('status', 'unknown')
        
        # Update progress based on status
        if current_status == 'uploaded':
            progress = 0.2
        elif current_status == 'processing':
            progress = 0.3 + (poll_count % 20) * 0.03  # Animated progress
        elif current_status == 'completed':
            progress = 1.0
            break
        elif current_status == 'error':
            progress = 1.0
            error_msg = status_data.get('error', 'Unknown error')
            with error_display.container():
                st.error(f"Processing failed: {error_msg}")
            break
        else:
            progress = min(0.1 + (poll_count * 0.01), 0.5)
        
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
                st.markdown(f"**File:** {document_id[:12]}...")
        
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
        st.error(f"Processing failed for document {document_id}")
    elif current_status == 'completed':
        st.success("✅ Processing completed successfully!")
        
        # Fetch results
        with st.spinner("Loading results..."):
            results = ui.get_results(document_id)
            if results:
                # Store results
                if 'all_results' not in st.session_state:
                    st.session_state['all_results'] = {}
                st.session_state['all_results'][document_id] = results
                
                # Add to recent documents
                if 'recent_docs' not in st.session_state:
                    st.session_state['recent_docs'] = []
                
                st.session_state['recent_docs'].append({
                    'id': document_id,
                    'name': f"Document_{document_id[:8]}",
                    'timestamp': datetime.now().isoformat(),
                    'filename': f"doc_{document_id[:8]}"
                })
                # Keep only last 10
                st.session_state['recent_docs'] = st.session_state['recent_docs'][-10:]
        
        # Move to next document or show continue button
        if current_idx + 1 < len(document_ids):
            st.session_state['current_doc_index'] = current_idx + 1
            st.info(f"Moving to next document ({current_idx + 2}/{len(document_ids)})...")
            time.sleep(2)
            st.rerun()
        else:
            st.session_state['processing_started'] = False
            st.session_state['show_results'] = True
            st.session_state['current_document_id'] = document_ids[-1]  # Show last doc results
            st.rerun()

def render_results_view(ui: DocumentIntelligenceUI):
    """Render comprehensive results view"""
    if 'all_results' not in st.session_state or not st.session_state['all_results']:
        st.error("No results available. Please process documents first.")
        if st.button("Return Home"):
            st.session_state.clear()
            st.rerun()
        return
    
    document_ids = list(st.session_state['all_results'].keys())
    
    # Document selector
    st.markdown("## 📊 Analysis Results")
    
    selected_doc = st.selectbox(
        "Select Document to View:",
        options=document_ids,
        format_func=lambda x: f"Document {x[:12]}... ({st.session_state['all_results'][x].get('document_id', 'Unknown')})"
    )
    
    if not selected_doc:
        return
    
    results = st.session_state['all_results'][selected_doc]
    document_id = results.get('document_id', selected_doc)
    
    # Header with metrics
    st.markdown(f"### Document Analysis: `{document_id[:12]}...`")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if results.get('success'):
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: white; margin: 0;'>✅</h3>
                <p style='color: white; margin: 0; font-size: 12px;'>Processing Status</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card' style='background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);'>
                <h3 style='color: white; margin: 0;'>❌</h3>
                <p style='color: white; margin: 0; font-size: 12px;'>Processing Failed</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        agent_count = len(results.get('agent_outputs', {}))
        st.metric("Agents Executed", agent_count)
    
    with col3:
        field_count = len(results.get('extracted_fields', {}))
        st.metric("Fields Extracted", field_count)
    
    with col4:
        error_count = len(results.get('errors', []))
        st.metric("Errors", error_count)
    
    if not results.get('success'):
        st.error("Document processing failed")
        if results.get('errors'):
            for error in results['errors']:
                st.error(f"Error: {error}")
        return
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Extracted Text",
        "👁️ Visual Analysis",
        "📊 Structured Data",
        "🎯 Agent Outputs",
        "🔍 Multi-Modal Query"
    ])
    
    # Tab 1: Extracted Text
    with tab1:
        st.markdown("### Text Extraction Results")
        
        # Display extracted text from various sources
        if 'agent_outputs' in results:
            agents = results['agent_outputs']
            
            # Show text from text agents
            if 'text' in agents:
                text_data = agents['text']
                if 'extracted_text' in text_data:
                    st.text_area("Extracted Text", text_data['extracted_text'], height=300)
                elif 'summary' in text_data:
                    st.text_area("Text Summary", text_data['summary'], height=300)
            
            # Show OCR results if available
            if 'ocr_results' in results:
                st.markdown("#### OCR Results")
                for page_num, ocr_result in results['ocr_results'].items():
                    with st.expander(f"Page {page_num}"):
                        if isinstance(ocr_result, dict):
                            st.text(f"Text: {ocr_result.get('text', '')[:500]}...")
                            st.text(f"Confidence: {ocr_result.get('confidence', 0):.2%}")
        else:
            st.info("No extracted text available in results")
    
    # Tab 2: Visual Analysis
    with tab2:
        st.markdown("### Visual Element Detection")
        
        if 'agent_outputs' in results and 'vision' in results['agent_outputs']:
            vision_data = results['agent_outputs']['vision']
            
            if 'detected_elements' in vision_data:
                elements = vision_data['detected_elements']
                
                if elements:
                    # Group elements by type
                    element_types = {}
                    for element in elements:
                        elem_type = element.get('type', 'unknown')
                        if elem_type not in element_types:
                            element_types[elem_type] = []
                        element_types[elem_type].append(element)
                    
                    # Display summary
                    st.markdown("#### Detection Summary")
                    summary_data = []
                    for elem_type, elem_list in element_types.items():
                        summary_data.append({
                            'Type': elem_type,
                            'Count': len(elem_list),
                            'Avg Confidence': f"{sum(e.get('confidence', 0) for e in elem_list)/len(elem_list):.1%}"
                        })
                    
                    if summary_data:
                        df = pd.DataFrame(summary_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Visualize distribution
                        fig = px.pie(df, values='Count', names='Type', 
                                    title="Detected Element Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed elements
                    st.markdown("#### Detailed Elements")
                    for elem_type, elem_list in element_types.items():
                        with st.expander(f"{elem_type.title()}s ({len(elem_list)})"):
                            for i, element in enumerate(elem_list[:10]):  # Limit to 10
                                st.json(element)
                else:
                    st.info("No visual elements detected")
            else:
                st.info("Visual detection data not available")
        else:
            st.info("Visual analysis not performed or data not available")
    
    # Tab 3: Structured Data
    with tab3:
        st.markdown("### Structured Data Extraction")
        
        if 'extracted_fields' in results and results['extracted_fields']:
            fields = results['extracted_fields']
            
            # Display as table
            field_data = []
            for field_name, field_info in fields.items():
                if isinstance(field_info, dict):
                    field_data.append({
                        'Field': field_name,
                        'Value': str(field_info.get('value', ''))[:100],
                        'Confidence': field_info.get('confidence', 0),
                        'Sources': ', '.join(field_info.get('sources', [])),
                        'Modalities': ', '.join(field_info.get('modalities', []))
                    })
            
            if field_data:
                df = pd.DataFrame(field_data)
                st.dataframe(df, use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"extracted_fields_{document_id}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No structured fields extracted")
        else:
            st.info("No structured data available")
    
    # Tab 4: Agent Outputs
    with tab4:
        st.markdown("### Agent Execution Results")
        
        if 'agent_outputs' in results:
            agents = results['agent_outputs']
            
            for agent_name, agent_output in agents.items():
                with st.expander(f"🤖 {agent_name.title()} Agent"):
                    if isinstance(agent_output, dict):
                        st.json(agent_output)
                    else:
                        st.text(str(agent_output))
            
            # Show execution metadata
            if 'processing_metadata' in results:
                st.markdown("#### Processing Metadata")
                metadata = results['processing_metadata']
                for key, value in metadata.items():
                    st.text(f"{key}: {value}")
        else:
            st.info("No agent outputs available")
    
    # Tab 5: Multi-Modal Query
    with tab5:
        st.markdown("### 🔍 Multi-Modal Document Query")
        
        # Query interface
        query = st.text_input(
            "Ask a question about the document:",
            placeholder="e.g., 'What tables were found?' or 'Extract all dates'",
            key=f"query_{document_id}"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            query_type = st.selectbox(
                "Query Type",
                ["text", "mixed"],
                help="Text: semantic search, Mixed: combined search"
            )
        with col2:
            if st.button("🔍 Search", use_container_width=True):
                if query:
                    with st.spinner("Searching..."):
                        response = ui.query_document(document_id, query)
                        
                        if response and response.get('success'):
                            answer = response.get('answer', '')
                            confidence = response.get('confidence', 0)
                            
                            st.markdown("#### 🤖 AI Answer")
                            st.info(answer)
                            
                            st.metric("Answer Confidence", f"{confidence:.1%}")
                            
                            if response.get('sources'):
                                with st.expander("View Sources"):
                                    for i, source in enumerate(response['sources']):
                                        st.text(f"Source {i+1}: {source}")
                        else:
                            st.warning("No answer found. Try a different query.")
                else:
                    st.warning("Please enter a query")
        
        # Example queries
        st.markdown("---")
        st.markdown("#### 💡 Example Queries")
        
        examples = [
            "What tables were detected?",
            "Extract all dates mentioned",
            "What is the overall document type?",
            "Were any contradictions found?",
            "What visual elements were detected?"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example[:10]}"):
                st.session_state[f"query_{document_id}"] = example
                st.rerun()

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
    
    # Main content area based on state
    if st.session_state.get('processing_started'):
        if 'document_ids' in st.session_state:
            render_processing_view(ui, st.session_state['document_ids'])
        else:
            st.error("No document IDs found for processing")
            if st.button("Return Home"):
                st.session_state.clear()
                st.rerun()
    
    elif st.session_state.get('show_results'):
        render_results_view(ui)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Process New Documents", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        with col2:
            if st.button("📊 Reprocess All", use_container_width=True):
                if 'all_results' in st.session_state:
                    document_ids = list(st.session_state['all_results'].keys())
                    st.session_state['document_ids'] = document_ids
                    st.session_state['current_doc_index'] = 0
                    st.session_state['processing_started'] = True
                    st.session_state['show_results'] = False
                    st.session_state.pop('all_results', None)
                    st.rerun()
        with col3:
            if 'all_results' in st.session_state:
                results_json = json.dumps(st.session_state['all_results'], indent=2)
                st.download_button(
                    label="📥 Export All Results",
                    data=results_json,
                    file_name=f"document_analysis_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    else:
        # Welcome/landing page
        render_landing_page(ui)

def render_landing_page(ui: DocumentIntelligenceUI):
    """Render landing/welcome page"""
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
                <li style="margin-bottom: 10px;">Upload one or more documents (PDF or image) using the sidebar</li>
                <li style="margin-bottom: 10px;">Configure processing options as needed</li>
                <li style="margin-bottom: 10px;">Click "Process Documents" to start analysis</li>
                <li style="margin-bottom: 10px;">View real-time processing status</li>
                <li style="margin-bottom: 10px;">Explore comprehensive results and query your documents</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats if API is connected
    if ui.check_api_health():
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            if response.status_code == 200:
                api_info = response.json()
                
                st.markdown("---")
                st.markdown("### System Status")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Service", api_info.get('service', 'Unknown'))
                with col2:
                    st.metric("Version", api_info.get('version', '1.0.0'))
                with col3:
                    status = api_info.get('status', 'unknown').title()
                    st.metric("Status", status)
        except:
            pass

if __name__ == "__main__":
    main()