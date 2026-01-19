import streamlit as st
import requests
import json
import time
from typing import Dict, Any
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Vision-Fusion Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 1rem;
        background: #F3F4F6;
        border-radius: 10px;
    }
    .step {
        text-align: center;
        flex: 1;
        padding: 1rem;
    }
    .step.active {
        background: #DBEAFE;
        border-radius: 8px;
    }
    .step-circle {
        width: 50px;
        height: 50px;
        background: #3B82F6;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px;
        font-size: 1.5rem;
    }
    .step-label {
        font-weight: bold;
        color: #1F2937;
    }
    .upload-box {
        border: 2px dashed #3B82F6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #EFF6FF;
    }
    .result-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .answer-box {
        background: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .progress-container {
        background: #E5E7EB;
        border-radius: 10px;
        height: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3B82F6, #60A5FA);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .debug-box {
        background: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
API_BASE_URL = "http://localhost:8000"

class DocumentUI:
    """Main UI class for Vision-Fusion Document Intelligence."""
    
    def __init__(self):
        self.current_view = "upload"
        self.current_document_id = None
        self.processing_status = {}
        
    def render_sidebar(self):
        """Render the sidebar navigation."""
        with st.sidebar:
            st.markdown("## 📁 Navigation")
            
            view_options = {
                "📤 Upload Document": "upload",
                "🔍 View Results": "results",
                "❓ Query Document": "query"
            }
            
            selected = st.radio(
                "Go to",
                list(view_options.keys()),
                label_visibility="collapsed"
            )
            
            self.current_view = view_options[selected]
            
            st.markdown("---")
            st.markdown("### 📊 Status")
            
            if self.current_document_id:
                st.info(f"**Document ID:**\n`{self.current_document_id}`")
                
                # Check status
                try:
                    status_response = requests.get(
                        f"{API_BASE_URL}/api/v1/status/{self.current_document_id}"
                    )
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        # ✅ FIXED: Use 'status' field from backend
                        status_value = status_data.get('status', 'Unknown')
                        st.success(f"**Status:** {status_value}")
                        
                        # Progress bar
                        st.markdown("**Progress:**")
                        # Get progress from ui_state or directly
                        progress = status_data.get('ui_state', {}).get('progress', 0)
                        if not progress:
                            progress = status_data.get('progress', 0)
                        
                        progress_html = f"""
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {progress}%"></div>
                        </div>
                        """
                        st.markdown(progress_html, unsafe_allow_html=True)
                        
                        # Debug info
                        if st.session_state.get("debug_mode", False):
                            with st.expander("Debug Info"):
                                st.json(status_data)
                except Exception as e:
                    st.warning(f"Could not get status: {str(e)}")
            
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            debug_mode = st.checkbox("Debug Mode", value=st.session_state.get("debug_mode", False))
            st.session_state.debug_mode = debug_mode
            
            st.markdown("**Backend:** FastAPI")
            st.markdown("**Vector DB:** Qdrant")
            st.markdown("**Embeddings:** MiniLM-L6-v2")
            
    def render_upload_view(self):
        """Render the document upload view."""
        st.markdown('<div class="main-header">Vision-Fusion Document Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Upload PDF documents for intelligent analysis and querying</div>', unsafe_allow_html=True)
        
        # Stepper
        stepper_html = """
        <div class="step-container">
            <div class="step active">
                <div class="step-circle">📤</div>
                <div class="step-label">STEP 1<br>Upload</div>
            </div>
            <div class="step">
                <div class="step-circle">⚙️</div>
                <div class="step-label">STEP 2<br>Process</div>
            </div>
            <div class="step">
                <div class="step-circle">📊</div>
                <div class="step-label">STEP 3<br>Results</div>
            </div>
            <div class="step">
                <div class="step-circle">❓</div>
                <div class="step-label">STEP 4<br>Query</div>
            </div>
        </div>
        """
        st.markdown(stepper_html, unsafe_allow_html=True)
        
        # Upload section
        st.markdown("### 📤 Upload Document")
        
        with st.container():
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                accept_multiple_files=False,
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                st.success(f"Selected: **{uploaded_file.name}**")
                st.info("File size: {:.1f} MB".format(uploaded_file.size / (1024 * 1024)))
                
                # Upload button
                if st.button("Upload & Process", type="primary", width="stretch"):
                    with st.spinner("Uploading document..."):
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                            response = requests.post(
                                f"{API_BASE_URL}/api/v1/upload",
                                files=files
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                # ✅ FIXED: Backend returns 'document_id' not 'upload_id'
                                self.current_document_id = result.get("document_id", "")
                                
                                if not self.current_document_id:
                                    st.error("No document ID returned from server")
                                    return
                                
                                st.success("✅ Document uploaded successfully!")
                                st.info(f"**Document ID:** `{self.current_document_id}`")
                                st.info("Processing has started. You can check status in the sidebar.")
                                
                                # Store in session state
                                st.session_state.document_id = self.current_document_id
                                st.session_state.upload_time = datetime.now().isoformat()
                                
                                # Auto-redirect to results after a moment
                                time.sleep(2)  # Give a moment to see the success message
                                st.rerun()
                            else:
                                st.error(f"Upload failed with status {response.status_code}: {response.text}")
                                
                        except Exception as e:
                            st.error(f"Error uploading file: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Instructions
        with st.expander("📋 How to use"):
            st.markdown("""
            1. **Upload** a PDF document using the uploader above
            2. **Wait** for processing to complete (OCR + AI analysis)
            3. **View** extracted text and analysis results
            4. **Ask questions** about your document using natural language
            
            **Features:**
            - Automatic OCR for scanned PDFs
            - Multi-agent AI analysis
            - Semantic search and Q&A
            - Vector embedding storage
            """)
    
    def render_results_view(self):
        """Render the results view."""
        st.markdown('<div class="main-header">Document Analysis Results</div>', unsafe_allow_html=True)
        
        # Check if we have a document ID
        if not self.current_document_id:
            if "document_id" in st.session_state:
                self.current_document_id = st.session_state.document_id
            else:
                st.warning("No document uploaded. Please upload a document first.")
                self.current_view = "upload"
                return
        
        # Stepper
        stepper_html = """
        <div class="step-container">
            <div class="step">
                <div class="step-circle">📤</div>
                <div class="step-label">STEP 1<br>Upload</div>
            </div>
            <div class="step active">
                <div class="step-circle">⚙️</div>
                <div class="step-label">STEP 2<br>Process</div>
            </div>
            <div class="step">
                <div class="step-circle">📊</div>
                <div class="step-label">STEP 3<br>Results</div>
            </div>
            <div class="step">
                <div class="step-circle">❓</div>
                <div class="step-label">STEP 4<br>Query</div>
            </div>
        </div>
        """
        st.markdown(stepper_html, unsafe_allow_html=True)
        
        # Check processing status
        try:
            status_response = requests.get(
                f"{API_BASE_URL}/api/v1/status/{self.current_document_id}"
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                # Debug display
                if st.session_state.get("debug_mode", False):
                    st.markdown('<div class="debug-box">', unsafe_allow_html=True)
                    st.write("**DEBUG - Status Response:**")
                    st.json(status_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show status
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", status_data.get("status", "unknown"))
                with col2:
                    progress = status_data.get('ui_state', {}).get('progress', 0) or status_data.get('progress', 0)
                    st.metric("Progress", f"{progress}%")
                with col3:
                    timestamp = status_data.get("timestamp", "")
                    if timestamp:
                        st.metric("Last Updated", timestamp[11:19])
                    else:
                        st.metric("Last Updated", "N/A")
                
                # Progress bar
                progress = status_data.get('ui_state', {}).get('progress', 0) or status_data.get('progress', 0)
                progress_html = f"""
                <div class="progress-container">
                    <div class="progress-bar" style="width: {progress}%"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                
                # ✅ FIXED: Check for completion using multiple criteria
                status_value = status_data.get("status", "").lower()
                progress_value = progress
                message = status_data.get("message", "").lower()
                
                # Check if processing is complete
                is_complete = (
                    status_value in ["results", "completed", "done"] or
                    progress_value >= 100 or
                    "completed" in message or
                    "done" in message
                )
                
                # Debug info for completion check
                if st.session_state.get("debug_mode", False):
                    st.write(f"**Completion Check:** Status='{status_value}', Progress={progress_value}, Message='{message}', IsComplete={is_complete}")
                
                if is_complete:
                    with st.spinner("Loading results..."):
                        results_response = requests.get(
                            f"{API_BASE_URL}/api/v1/results/{self.current_document_id}"
                        )
                        
                        if results_response.status_code == 200:
                            results_data = results_response.json()
                            
                            # Debug display
                            if st.session_state.get("debug_mode", False):
                                st.markdown('<div class="debug-box">', unsafe_allow_html=True)
                                st.write("**DEBUG - Results Response:**")
                                st.json({k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in results_data.items()})
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display results in tabs
                            tab1, tab2, tab3 = st.tabs(["📝 Extracted Text", "🤖 AI Analysis", "📊 Statistics"])
                            
                            with tab1:
                                st.markdown("### 📝 Extracted Text")
                                text_content = results_data.get("text_content", "")
                                
                                if text_content:
                                    # Show text preview
                                    st.text_area(
                                        "Extracted text (first 5000 characters)",
                                        text_content[:5000],
                                        height=300,
                                        disabled=True,
                                        label_visibility="collapsed"
                                    )
                                    
                                    # Show stats
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Total Characters", len(text_content))
                                    with col2:
                                        text_segments = results_data.get("text_segments", [])
                                        if isinstance(text_segments, list):
                                            st.metric("Text Segments", len(text_segments))
                                        else:
                                            st.metric("Text Segments", 1)
                                else:
                                    st.warning("No text content found.")
                            
                            with tab2:
                                st.markdown("### 🤖 AI Analysis Results")
                                agent_results = results_data.get("agent_results", {})
                                
                                if agent_results:
                                    if isinstance(agent_results, dict):
                                        for agent, result in agent_results.items():
                                            with st.expander(f"**{agent.replace('_', ' ').title()}**"):
                                                if isinstance(result, dict):
                                                    for key, value in result.items():
                                                        st.markdown(f"**{key}:** {value}")
                                                else:
                                                    st.write(result)
                                    else:
                                        st.write(agent_results)
                                else:
                                    st.info("AI analysis results will appear here once processing is complete.")
                            
                            with tab3:
                                st.markdown("### 📊 Document Statistics")
                                
                                # Calculate statistics
                                text_content = results_data.get("text_content", "")
                                chunks = results_data.get("chunks", [])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Words", len(text_content.split()))
                                with col2:
                                    if isinstance(chunks, list):
                                        st.metric("Chunks Created", len(chunks))
                                    else:
                                        st.metric("Chunks Created", 0)
                                with col3:
                                    st.metric("Processing Time", "Completed")
                                
                                # Chunks preview
                                st.markdown("#### Document Chunks")
                                if chunks and isinstance(chunks, list):
                                    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                                        with st.container():
                                            st.markdown(f"**Chunk {i+1}** (Page {chunk.get('page', 'N/A')})")
                                            chunk_text = chunk.get("text", "")
                                            display_text = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                                            st.text(display_text)
                                            st.markdown("---")
                                else:
                                    st.info("No chunks available.")
                            
                            # Show success message
                            st.success("✅ Document processing completed successfully!")
                            st.info("You can now query the document using natural language.")
                            
                            # Add query button
                            if st.button("Go to Query", type="primary", width="stretch"):
                                self.current_view = "query"
                                st.rerun()
                            
                        else:
                            st.error(f"Failed to load results: {results_response.status_code}")
                            if results_response.status_code == 404:
                                st.info("Results might not be ready yet. Try refreshing in a moment.")
                else:
                    # Still processing
                    st.info(f"⏳ Processing in progress: {status_data.get('message', 'Processing...')}")
                    st.warning("Please wait for processing to complete. This page will auto-refresh.")
                    
                    # Add manual refresh button
                    if st.button("Refresh Status", width="stretch"):
                        st.rerun()
                    
                    # Auto-refresh
                    time.sleep(3)
                    st.rerun()
                    
            else:
                st.error(f"Failed to get status: {status_response.status_code}")
                
        except Exception as e:
            st.error(f"Error checking status: {str(e)}")
            # Debug: show full error
            if st.session_state.get("debug_mode", False):
                import traceback
                st.code(traceback.format_exc())
    
    def render_query_view(self):
        """Render the query view."""
        st.markdown('<div class="main-header">Document Q&A</div>', unsafe_allow_html=True)
        
        # Stepper
        stepper_html = """
        <div class="step-container">
            <div class="step">
                <div class="step-circle">📤</div>
                <div class="step-label">STEP 1<br>Upload</div>
            </div>
            <div class="step">
                <div class="step-circle">⚙️</div>
                <div class="step-label">STEP 2<br>Process</div>
            </div>
            <div class="step">
                <div class="step-circle">📊</div>
                <div class="step-label">STEP 3<br>Results</div>
            </div>
            <div class="step active">
                <div class="step-circle">❓</div>
                <div class="step-label">STEP 4<br>Query</div>
            </div>
        </div>
        """
        st.markdown(stepper_html, unsafe_allow_html=True)
        
        # Check if we have a document ID
        if not self.current_document_id:
            if "document_id" in st.session_state:
                self.current_document_id = st.session_state.document_id
            else:
                st.warning("No document uploaded. Please upload a document first.")
                self.current_view = "upload"
                st.rerun()
                return
        
        # Check if processing is complete
        try:
            status_response = requests.get(
                f"{API_BASE_URL}/api/v1/status/{self.current_document_id}"
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status_value = status_data.get("status", "").lower()
                progress = status_data.get('ui_state', {}).get('progress', 0) or status_data.get('progress', 0)
                
                # ✅ FIXED: Check using multiple criteria
                is_complete = (
                    status_value in ["results", "completed", "done"] or
                    progress >= 100 or
                    "completed" in status_data.get("message", "").lower()
                )
                
                if not is_complete:
                    st.warning("⚠️ Document processing is not complete yet. Please wait.")
                    st.info(f"Current status: {status_data.get('message', 'Processing...')}")
                    st.info(f"Progress: {progress}%")
                    return
                    
        except:
            st.error("Unable to check document status.")
            return
        
        # Query interface
        st.markdown("### ❓ Ask Questions About Your Document")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "Enter your question",
                placeholder="e.g., What is the first program in my document?",
                label_visibility="collapsed"
            )
        
        with col2:
            if st.button("Ask Question", type="primary", width="stretch"):
                if question.strip():
                    st.session_state.last_question = question
        
        # Handle query submission
        if "last_question" in st.session_state and st.session_state.last_question:
            question = st.session_state.last_question
            
            with st.spinner("Searching document and generating answer..."):
                try:
                    # Prepare payload
                    payload = {
                        "document_id": self.current_document_id,
                        "question": question
                    }
                    
                    # Send query request
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/query",
                        json=payload
                    )
                    
                    # Debug display
                    if st.session_state.get("debug_mode", False):
                        st.markdown('<div class="debug-box">', unsafe_allow_html=True)
                        st.write("**DEBUG - Query Response Status:**", response.status_code)
                        st.write("**DEBUG - Query Response:**")
                        try:
                            st.json(response.json())
                        except:
                            st.write(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])
                        confidence = data.get("confidence", 0.0)
                        
                        # Display answer
                        st.markdown("---")
                        st.markdown("### 📌 Answer")
                        
                        if answer and answer.strip():
                            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                            
                            # Show confidence
                            if confidence > 0:
                                st.metric("Confidence", f"{confidence*100:.1f}%")
                        else:
                            st.warning("No answer returned by backend. The document may not contain relevant information.")
                        
                        # Display sources
                        if sources:
                            st.markdown("#### 📚 Sources")
                            st.info("These are the document sections used to generate the answer:")
                            
                            for i, source in enumerate(sources, 1):
                                if isinstance(source, dict):
                                    source_text = source.get("text", str(source))
                                    page = source.get("page", "N/A")
                                    score = source.get("score", 0.0)
                                    with st.expander(f"Source {i} (Page {page}, Score: {score:.2f})"):
                                        st.write(source_text)
                                else:
                                    with st.expander(f"Source {i}"):
                                        st.write(str(source))
                    
                    elif response.status_code == 400:
                        st.error("Document processing not completed yet. Please wait.")
                    elif response.status_code == 404:
                        st.error("Document not found. Please upload a document first.")
                    else:
                        st.error(f"Query failed with status {response.status_code}: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error querying document: {str(e)}")
                    # Debug: show full error
                    if st.session_state.get("debug_mode", False):
                        import traceback
                        st.code(traceback.format_exc())
            
            # Clear the question after processing
            if "last_question" in st.session_state:
                del st.session_state.last_question
        
        # Example questions
        st.markdown("---")
        st.markdown("#### 💡 Example Questions")
        
        examples = [
            "What is the main topic of this document?",
            "What is the first program or experiment mentioned?",
            "Summarize the key points",
            "What are the learning objectives?",
            "List all programs or experiments mentioned"
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(example, width="stretch", key=f"example_{i}"):
                    st.session_state.last_question = example
                    st.rerun()
    
    def render(self):
        """Main render method."""
        # Initialize session state
        if "document_id" not in st.session_state:
            st.session_state.document_id = None
        
        if "upload_time" not in st.session_state:
            st.session_state.upload_time = None
        
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False
        
        # Set current document ID from session state
        if st.session_state.document_id:
            self.current_document_id = st.session_state.document_id
        
        # Render sidebar
        self.render_sidebar()
        
        # Render current view
        if self.current_view == "upload":
            self.render_upload_view()
        elif self.current_view == "results":
            self.render_results_view()
        elif self.current_view == "query":
            self.render_query_view()

def main():
    """Main entry point."""
    ui = DocumentUI()
    ui.render()

if __name__ == "__main__":
    main()