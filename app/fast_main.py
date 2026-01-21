# fast_main.py - Complete with real processing simulation
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import logging
import uuid
import shutil
import os
import json
import hashlib
from pathlib import Path
import asyncio
from typing import Dict, List, Optional
import pickle
import mimetypes
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Fast startup - with real processing simulation
app = FastAPI(
    title="Vision-Fusion Document Intelligence",
    version="1.0.0",
    description="Multi-modal document processing with intelligent caching",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = "uploads"
CACHE_DIR = "cache"
PROCESSING_DIR = "processing"
for directory in [UPLOAD_DIR, CACHE_DIR, PROCESSING_DIR]:
    os.makedirs(directory, exist_ok=True)
logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
logger.info(f"💾 Cache directory: {CACHE_DIR}")

# Global storage
documents_db = {}
processing_status = {}
processing_results = {}
processing_cache = {}
agent_templates = {}

class DocumentProcessor:
    """Simulate real document processing with agents"""
    
    def __init__(self):
        self.agent_configs = {
            "vision_agent": {
                "name": "Vision Agent",
                "description": "Analyzes visual elements, layout, and structure",
                "processing_steps": [
                    "Loading image/document",
                    "Detecting layout regions",
                    "Identifying visual elements",
                    "Extracting tables and figures",
                    "Detecting signatures and logos",
                    "Analyzing visual patterns",
                    "Generating semantic labels"
                ],
                "fields": ["tables", "signatures", "logos", "layout_regions", "visual_elements", "semantic_labels"],
                "processing_time": 1.5  # seconds
            },
            "text_agent": {
                "name": "Text Agent", 
                "description": "Extracts and analyzes text content",
                "processing_steps": [
                    "Performing OCR/Text extraction",
                    "Identifying paragraphs and sections",
                    "Extracting entities and keywords",
                    "Analyzing document structure",
                    "Identifying document type",
                    "Extracting metadata",
                    "Generating text summaries"
                ],
                "fields": ["raw_text", "entities", "paragraphs", "document_type", "metadata", "keywords"],
                "processing_time": 2.0
            },
            "fusion_agent": {
                "name": "Fusion Agent",
                "description": "Aligns text and visual information",
                "processing_steps": [
                    "Mapping text to visual regions",
                    "Cross-modal alignment",
                    "Resolving inconsistencies",
                    "Creating unified document model",
                    "Generating fused embeddings",
                    "Building semantic connections"
                ],
                "fields": ["aligned_regions", "cross_modal_links", "consistency_scores", "fused_embeddings"],
                "processing_time": 1.8
            },
            "reasoning_agent": {
                "name": "Reasoning Agent",
                "description": "Validates content and assesses risks",
                "processing_steps": [
                    "Validating extracted information",
                    "Detecting contradictions",
                    "Assessing document quality",
                    "Calculating risk scores",
                    "Generating recommendations",
                    "Creating validation report",
                    "Building explainability trail"
                ],
                "fields": ["risk_score", "contradictions", "validation_report", "recommendations", "explainability"],
                "processing_time": 1.7
            }
        }
    
    async def process_agent(self, agent_name: str, document_info: Dict) -> Dict:
        """Simulate real agent processing"""
        config = self.agent_configs[agent_name]
        logger.info(f"🤖 {agent_name} starting processing...")
        
        # Real processing simulation
        steps = config["processing_steps"]
        fields = []
        progress_updates = []
        
        for i, step in enumerate(steps):
            # Simulate step processing time
            step_time = config["processing_time"] / len(steps)
            await asyncio.sleep(step_time)
            
            # Generate realistic field data based on step
            if agent_name == "vision_agent":
                if "Detecting" in step:
                    fields.append(f"detected_{document_info.get('file_type', 'document').lower()}")
                elif "Identifying" in step:
                    fields.append("visual_elements_classified")
                elif "Extracting" in step:
                    fields.append("structural_elements_extracted")
                    
            elif agent_name == "text_agent":
                if "OCR" in step or "extraction" in step:
                    fields.append("text_content_extracted")
                elif "entities" in step.lower():
                    fields.append("named_entities_identified")
                elif "metadata" in step:
                    fields.append("document_metadata_extracted")
                    
            elif agent_name == "fusion_agent":
                if "alignment" in step.lower():
                    fields.append("cross_modal_alignment_completed")
                elif "connections" in step.lower():
                    fields.append("semantic_connections_built")
                    
            elif agent_name == "reasoning_agent":
                if "risk" in step.lower():
                    fields.append("risk_assessment_completed")
                elif "validation" in step.lower():
                    fields.append("validation_report_generated")
            
            progress = int((i + 1) * 100 / len(steps))
            progress_updates.append({
                "step": step,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"✅ {agent_name} completed processing")
        
        return {
            "status": "completed",
            "progress": 100,
            "fields_extracted": fields[:5],  # Limit to 5 fields
            "processing_steps": progress_updates,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": config["processing_time"]
        }

# Initialize processor
processor = DocumentProcessor()

def init_templates():
    """Initialize document templates"""
    global agent_templates
    
    agent_templates = {
        "invoice": {
            "agents": ["vision_agent", "text_agent", "fusion_agent", "reasoning_agent"],
            "priority": "high",
            "expected_fields": ["invoice_number", "date", "vendor", "total_amount", "items", "payment_terms"],
            "risk_factors": ["amount_mismatch", "missing_signature", "duplicate_invoice", "date_anomalies"]
        },
        "resume": {
            "agents": ["text_agent", "vision_agent", "reasoning_agent"],
            "priority": "medium",
            "expected_fields": ["name", "contact", "education", "experience", "skills", "achievements"],
            "risk_factors": ["date_gaps", "skill_mismatch", "format_issues", "consistency_checks"]
        },
        "contract": {
            "agents": ["text_agent", "vision_agent", "fusion_agent", "reasoning_agent"],
            "priority": "high",
            "expected_fields": ["parties", "effective_date", "terms", "signatures", "clauses", "dates"],
            "risk_factors": ["unsigned_sections", "ambiguous_terms", "missing_pages", "legal_compliance"]
        },
        "report": {
            "agents": ["text_agent", "vision_agent", "fusion_agent", "reasoning_agent"],
            "priority": "medium",
            "expected_fields": ["title", "author", "date", "sections", "conclusions", "recommendations"],
            "risk_factors": ["inconsistent_data", "missing_charts", "format_errors", "logical_flow"]
        }
    }
    
    logger.info(f"📋 Loaded {len(agent_templates)} document templates")

def get_document_fingerprint(file_path: str) -> str:
    """Generate unique fingerprint for caching"""
    try:
        stat = os.stat(file_path)
        file_size = stat.st_size
        modified_time = stat.st_mtime
        
        # Read first 2KB for content hash
        with open(file_path, 'rb') as f:
            sample = f.read(2048)
        
        # Combine for unique hash
        hash_input = f"{file_size}_{modified_time}_{hashlib.sha256(sample).hexdigest()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    except Exception as e:
        logger.error(f"Fingerprint generation failed: {e}")
        return str(uuid.uuid4())

def detect_document_type(filename: str, file_path: str) -> str:
    """Detect document type from file"""
    filename_lower = filename.lower()
    
    # Check filename patterns
    if any(keyword in filename_lower for keyword in ['invoice', 'bill', 'receipt']):
        return "invoice"
    elif any(keyword in filename_lower for keyword in ['resume', 'cv', 'curriculum']):
        return "resume"
    elif any(keyword in filename_lower for keyword in ['contract', 'agreement', 'mou']):
        return "contract"
    elif any(keyword in filename_lower for keyword in ['report', 'analysis', 'study']):
        return "report"
    
    # Check file content (simplified)
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if 'pdf' in mime_type:
                # Could add PDF content analysis here
                return "report"
            elif 'image' in mime_type:
                return "report"
            elif 'word' in mime_type or 'document' in mime_type:
                return "report"
    except:
        pass
    
    return "report"  # Default

def load_cache():
    """Load processing cache from disk"""
    global processing_cache
    
    cache_file = Path(CACHE_DIR) / "processing_cache.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                processing_cache = pickle.load(f)
            logger.info(f"📂 Loaded {len(processing_cache)} cached results")
        except Exception as e:
            logger.error(f"Cache load failed: {e}")
            processing_cache = {}

def save_cache():
    """Save cache to disk"""
    try:
        cache_file = Path(CACHE_DIR) / "processing_cache.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(processing_cache, f)
        logger.info(f"💾 Saved {len(processing_cache)} results to cache")
    except Exception as e:
        logger.error(f"Cache save failed: {e}")

# Initialize
init_templates()
load_cache()

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "service": "Vision-Fusion Document Intelligence",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "environment": "production_simulation",
        "statistics": {
            "documents_processed": len(documents_db),
            "cached_results": len(processing_cache),
            "active_processes": len([s for s in processing_status.values() if s.get("status") == "processing"]),
            "available_agents": len(processor.agent_configs)
        },
        "endpoints": {
            "upload": "/api/v1/upload",
            "status": "/api/v1/status/{document_id}",
            "results": "/api/v1/results/{document_id}",
            "query": "/api/v1/query",
            "process": "/api/v1/process/{document_id}",
            "debug": "/api/v1/debug/{document_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "service": "Vision-Fusion Document Intelligence",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "upload_dir_accessible": os.access(UPLOAD_DIR, os.W_OK),
            "cache_dir_accessible": os.access(CACHE_DIR, os.W_OK),
            "processing_dir_accessible": os.access(PROCESSING_DIR, os.W_OK),
            "memory_usage_mb": os.sys.getsizeof({}) // 1024 // 1024,
            "documents_in_memory": len(documents_db),
            "active_processing": len([s for s in processing_status.values() if s.get("status") == "processing"])
        },
        "agents": list(processor.agent_configs.keys()),
        "templates": list(agent_templates.keys())
    }

@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document for processing"""
    try:
        logger.info(f"📤 Uploading: {file.filename}")
        
        # Validate file
        allowed_extensions = ['.pdf', '.docx', '.png', '.jpg', '.jpeg', '.txt', '.doc', '.ppt', '.pptx']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if not file_ext:
            raise HTTPException(status_code=400, detail="File has no extension")
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        logger.info(f"📄 Generated ID: {document_id}")
        
        # Save file
        upload_dir = os.path.join(UPLOAD_DIR, document_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"document{file_ext}")
        
        # Read and save file
        file_content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Analyze document
        file_size = os.path.getsize(file_path)
        fingerprint = get_document_fingerprint(file_path)
        doc_type = detect_document_type(file.filename, file_path)
        
        # Check cache
        cached_result = processing_cache.get(fingerprint)
        
        # Store document info
        documents_db[document_id] = {
            "document_id": document_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": file_size,
            "file_type": file_ext[1:].upper(),
            "uploaded_at": datetime.now().isoformat(),
            "fingerprint": fingerprint,
            "document_type": doc_type,
            "mime_type": mimetypes.guess_type(file_path)[0] or "application/octet-stream",
            "has_cache": cached_result is not None,
            "template": agent_templates.get(doc_type, agent_templates["report"])
        }
        
        # Initialize processing status
        processing_status[document_id] = {
            "document_id": document_id,
            "status": "uploaded",
            "progress": 0,
            "message": "Document uploaded successfully",
            "timestamp": datetime.now().isoformat(),
            "file_info": {
                "name": file.filename,
                "size": file_size,
                "type": file_ext[1:].upper()
            }
        }
        
        # If cached, prepare immediate results
        if cached_result:
            logger.info(f"🎯 Cache hit for {document_id}")
            processing_results[document_id] = cached_result
            processing_status[document_id] = {
                **processing_status[document_id],
                "status": "completed",
                "progress": 100,
                "message": "Results loaded from cache",
                "cached": True,
                "processing_time": 0.1
            }
        
        logger.info(f"✅ Upload complete: {document_id} ({doc_type})")
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "file_size": file_size,
            "document_type": doc_type,
            "estimated_agents": len(agent_templates.get(doc_type, agent_templates["report"])["agents"]),
            "has_cache": cached_result is not None,
            "cache_ready": cached_result is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/v1/process/{document_id}")
async def start_processing(document_id: str, force_fresh: bool = False):
    """Start real document processing"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = documents_db[document_id]
    fingerprint = doc_info["fingerprint"]
    doc_type = doc_info["document_type"]
    
    # Check cache unless forced fresh
    if not force_fresh and fingerprint in processing_cache:
        logger.info(f"⚡ Using cached results for {document_id}")
        
        cached_result = processing_cache[fingerprint]
        processing_results[document_id] = cached_result
        
        processing_status[document_id] = {
            "document_id": document_id,
            "status": "completed",
            "progress": 100,
            "message": "Processing completed instantly from cache",
            "timestamp": datetime.now().isoformat(),
            "cached": True,
            "processing_time": 0.1
        }
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Processing completed instantly from cache",
            "cached": True,
            "processing_time": 0.1
        }
    
    # Start real processing
    logger.info(f"🚀 Starting processing for {document_id} ({doc_type})")
    
    # Update status
    template = agent_templates.get(doc_type, agent_templates["report"])
    processing_status[document_id] = {
        "document_id": document_id,
        "status": "processing",
        "progress": 5,
        "message": "Initializing processing pipeline...",
        "timestamp": datetime.now().isoformat(),
        "document_type": doc_type,
        "template": template["agents"],
        "priority": template.get("priority", "medium"),
        "agents": {}
    }
    
    # Start processing in background
    asyncio.create_task(process_document_real(document_id, doc_type, fingerprint))
    
    return {
        "success": True,
        "document_id": document_id,
        "message": "Processing started",
        "estimated_time": sum(
            processor.agent_configs[agent]["processing_time"] 
            for agent in template["agents"]
        ),
        "agents": template["agents"],
        "parallel_capable": True
    }

async def process_document_real(document_id: str, doc_type: str, fingerprint: str):
    """Real document processing with parallel agents"""
    try:
        start_time = time.time()
        doc_info = documents_db[document_id]
        template = agent_templates.get(doc_type, agent_templates["report"])
        
        logger.info(f"🔧 Processing {document_id} with agents: {template['agents']}")
        
        # Initialize agents status
        agents_data = {}
        for agent in template["agents"]:
            agents_data[agent] = {
                "name": processor.agent_configs[agent]["name"],
                "description": processor.agent_configs[agent]["description"],
                "status": "pending",
                "progress": 0,
                "fields_extracted": [],
                "processing_steps": [],
                "start_time": None,
                "end_time": None,
                "duration": None
            }
        
        processing_status[document_id]["agents"] = agents_data
        processing_status[document_id]["message"] = "Processing started..."
        
        # Process agents in parallel (simulated)
        agent_tasks = []
        for agent in template["agents"]:
            task = asyncio.create_task(
                process_single_agent(agent, document_id, doc_info)
            )
            agent_tasks.append(task)
        
        # Wait for all agents to complete
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Check for errors
        errors = []
        for agent, result in zip(template["agents"], agent_results):
            if isinstance(result, Exception):
                errors.append(f"{agent} failed: {str(result)}")
                agents_data[agent]["status"] = "error"
                agents_data[agent]["errors"] = [str(result)]
            else:
                agents_data[agent].update(result)
        
        # Generate final results
        if errors:
            processing_status[document_id]["status"] = "error"
            processing_status[document_id]["message"] = f"Processing completed with errors: {', '.join(errors)}"
            processing_status[document_id]["errors"] = errors
        else:
            # Generate comprehensive results
            results = await generate_real_results(document_id, doc_type, agents_data)
            
            # Store in cache and results
            processing_cache[fingerprint] = results
            processing_results[document_id] = results
            
            # Update status
            processing_status[document_id]["status"] = "completed"
            processing_status[document_id]["progress"] = 100
            processing_status[document_id]["message"] = "Processing completed successfully"
            processing_status[document_id]["processing_time"] = time.time() - start_time
            
            # Save cache periodically
            if len(processing_cache) % 5 == 0:
                save_cache()
            
            logger.info(f"✅ Processing completed for {document_id} in {time.time() - start_time:.2f}s")
    
    except Exception as e:
        logger.error(f"❌ Processing failed for {document_id}: {e}", exc_info=True)
        processing_status[document_id] = {
            "document_id": document_id,
            "status": "error",
            "progress": 0,
            "message": f"Processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

async def process_single_agent(agent_name: str, document_id: str, doc_info: Dict):
    """Process a single agent with real simulation"""
    try:
        # Update agent status
        processing_status[document_id]["agents"][agent_name]["status"] = "processing"
        processing_status[document_id]["agents"][agent_name]["start_time"] = datetime.now().isoformat()
        
        # Get agent config
        config = processor.agent_configs[agent_name]
        
        # Simulate processing with real progress updates
        steps = config["processing_steps"]
        fields_extracted = []
        processing_steps = []
        
        for i, step in enumerate(steps):
            # Calculate progress
            progress = int((i + 1) * 100 / len(steps))
            
            # Update agent progress
            processing_status[document_id]["agents"][agent_name]["progress"] = progress
            
            # Simulate step processing
            step_time = config["processing_time"] / len(steps)
            await asyncio.sleep(step_time)
            
            # Generate realistic field based on step
            if "detect" in step.lower() or "identify" in step.lower():
                field = f"{step.split()[0]}_completed"
            elif "extract" in step.lower():
                field = f"{doc_info.get('document_type', 'document').lower()}_{step.split()[0].lower()}"
            elif "analyze" in step.lower() or "validate" in step.lower():
                field = f"analysis_{step.split()[0].lower()}"
            else:
                field = f"step_{i+1}_completed"
            
            fields_extracted.append(field)
            processing_steps.append({
                "step": step,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "progress": progress
            })
            
            # Update overall progress
            overall_progress = calculate_overall_progress(document_id)
            processing_status[document_id]["progress"] = overall_progress
            processing_status[document_id]["message"] = f"{agent_name}: {step}"
            processing_status[document_id]["timestamp"] = datetime.now().isoformat()
        
        # Mark agent as completed
        end_time = datetime.now().isoformat()
        start_time = processing_status[document_id]["agents"][agent_name]["start_time"]
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
        
        return {
            "status": "completed",
            "progress": 100,
            "fields_extracted": fields_extracted[:5],  # Limit fields
            "processing_steps": processing_steps,
            "end_time": end_time,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Agent {agent_name} failed: {e}")
        raise

def calculate_overall_progress(document_id: str) -> int:
    """Calculate overall progress based on all agents"""
    if document_id not in processing_status or "agents" not in processing_status[document_id]:
        return 0
    
    agents = processing_status[document_id]["agents"]
    if not agents:
        return 0
    
    total_progress = sum(agent.get("progress", 0) for agent in agents.values())
    return int(total_progress / len(agents))

async def generate_real_results(document_id: str, doc_type: str, agents_data: Dict):
    """Generate realistic processing results"""
    
    # Calculate risk score based on document type and processing
    base_risk = {
        "invoice": 0.3,
        "contract": 0.4,
        "resume": 0.2,
        "report": 0.25
    }.get(doc_type, 0.3)
    
    # Adjust based on agents completed
    completed_agents = sum(1 for agent in agents_data.values() if agent.get("status") == "completed")
    total_agents = len(agents_data)
    completion_ratio = completed_agents / total_agents
    
    risk_score = base_risk * (1.2 - completion_ratio * 0.2)  # Better completion = lower risk
    
    # Generate extracted information based on document type
    extracted_info = {}
    if doc_type == "invoice":
        extracted_info = {
            "invoice_number": {"value": f"INV-{hash(document_id) % 10000:04d}", "confidence": 0.95},
            "date": {"value": datetime.now().strftime("%Y-%m-%d"), "confidence": 0.92},
            "vendor": {"value": "Vendor Corporation", "confidence": 0.88},
            "total_amount": {"value": f"${(hash(document_id) % 10000) / 100:.2f}", "confidence": 0.96},
            "items": {"value": [(hash(f"item{i}{document_id}") % 10) + 1 for i in range(3)], "confidence": 0.91}
        }
    elif doc_type == "resume":
        extracted_info = {
            "name": {"value": "Applicant Name", "confidence": 0.97},
            "contact": {"value": "email@example.com", "confidence": 0.89},
            "education": {"value": ["Bachelor's Degree", "Master's Degree"], "confidence": 0.93},
            "experience": {"value": f"{(hash(document_id) % 10) + 3} years", "confidence": 0.91},
            "skills": {"value": ["Python", "AI/ML", "FastAPI", "React"], "confidence": 0.94}
        }
    
    # Count extracted fields from all agents
    total_fields = sum(len(agent.get("fields_extracted", [])) for agent in agents_data.values())
    
    return {
        "success": True,
        "document_id": document_id,
        "processing_time": sum(agent.get("duration", 0) for agent in agents_data.values()),
        "document_type": doc_type,
        "risk_score": min(risk_score, 1.0),
        "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW",
        "contradictions_count": hash(document_id) % 3,
        "extracted_fields_count": total_fields,
        "agents_completed": f"{completed_agents}/{total_agents}",
        "recommendations": [
            "Document appears valid and complete",
            "All required fields extracted successfully",
            "No critical issues detected",
            "Ready for further analysis"
        ],
        "detailed_results": {
            "document_understanding": {
                "type": doc_type,
                "confidence": 0.85 + (hash(document_id) % 150) / 1000,
                "validation_score": 0.9 - (risk_score * 0.3)
            },
            "content_summary": {
                "text": {
                    "estimated_length": (hash(document_id) % 5000) + 1000,
                    "estimated_pages": (hash(document_id) % 5) + 1,
                    "entities_detected": ["dates", "names", "amounts", "addresses"][:(hash(document_id) % 4)]
                },
                "visual": {
                    "elements_detected": (hash(document_id) % 15) + 5,
                    "tables_detected": 1 if doc_type in ["invoice", "report"] else 0,
                    "signatures_detected": 1 if doc_type in ["invoice", "contract"] else 0
                }
            },
            "extracted_information": extracted_info,
            "quality_assessment": {
                "risk_score": risk_score,
                "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW",
                "completeness_score": completion_ratio,
                "consistency_score": 0.85 + (hash(document_id) % 150) / 1000,
                "recommendations": ["Review for accuracy", "Verify extracted amounts", "Check signature validity"][:2]
            },
            "agent_performance": {
                agent_name: {
                    "status": agent_data.get("status"),
                    "fields_extracted": len(agent_data.get("fields_extracted", [])),
                    "processing_time": agent_data.get("duration", 0)
                }
                for agent_name, agent_data in agents_data.items()
            }
        }
    }

@app.get("/api/v1/status/{document_id}")
async def get_status(document_id: str):
    """Get real-time processing status"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document_id in processing_status:
        status = processing_status[document_id].copy()
        
        # Add live updates if processing
        if status.get("status") == "processing":
            # Calculate real progress
            if "agents" in status:
                overall_progress = calculate_overall_progress(document_id)
                status["progress"] = overall_progress
                
                # Update message based on current state
                active_agents = [
                    name for name, data in status["agents"].items() 
                    if data.get("status") in ["processing", "running"]
                ]
                
                if active_agents:
                    status["message"] = f"Processing with {len(active_agents)} active agents"
                else:
                    # Check if all are completed
                    completed_agents = [
                        name for name, data in status["agents"].items() 
                        if data.get("status") == "completed"
                    ]
                    if len(completed_agents) == len(status["agents"]):
                        status["status"] = "completed"
                        status["message"] = "All agents completed processing"
                    else:
                        status["message"] = "Processing pipeline active"
            
            status["timestamp"] = datetime.now().isoformat()
        
        return status
    
    # Default status
    return {
        "document_id": document_id,
        "status": "uploaded",
        "progress": 0,
        "message": "Document uploaded, ready for processing",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/results/{document_id}")
async def get_results(document_id: str):
    """Get processing results"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document_id in processing_results:
        return processing_results[document_id]
    
    # Check if processing is complete
    if document_id in processing_status and processing_status[document_id].get("status") == "completed":
        # Generate results if not already generated
        if document_id not in processing_results:
            doc_info = documents_db[document_id]
            # Use cached agents data or generate
            agents_data = processing_status[document_id].get("agents", {})
            results = await generate_real_results(document_id, doc_info["document_type"], agents_data)
            processing_results[document_id] = results
        
        return processing_results[document_id]
    
    raise HTTPException(status_code=404, detail="Results not available yet. Processing may still be in progress.")

@app.post("/api/v1/query")
async def query_document(query_request: dict):
    """Query document with realistic RAG simulation"""
    document_id = query_request.get("document_id")
    query = query_request.get("query", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    if document_id and document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Generate realistic query results
    query_lower = query.lower()
    doc_type = documents_db.get(document_id, {}).get("document_type", "document") if document_id else "general"
    
    # Determine result type based on query
    if any(term in query_lower for term in ["table", "figure", "image", "visual"]):
        modality = "visual"
        content_type = "table" if "table" in query_lower else "figure"
        agent = "vision_agent"
    elif any(term in query_lower for term in ["signature", "logo", "stamp"]):
        modality = "visual"
        content_type = "signature"
        agent = "vision_agent"
    elif any(term in query_lower for term in ["amount", "total", "price", "cost"]):
        modality = "text"
        content_type = "financial_data"
        agent = "text_agent"
    elif any(term in query_lower for term in ["date", "time", "schedule"]):
        modality = "text"
        content_type = "temporal_data"
        agent = "text_agent"
    elif any(term in query_lower for term in ["risk", "issue", "problem", "error"]):
        modality = "fused"
        content_type = "risk_assessment"
        agent = "reasoning_agent"
    else:
        modality = "fused"
        content_type = "general_content"
        agent = "fusion_agent"
    
    # Generate realistic results
    results = []
    for i in range(min(3, 1 + (hash(query) % 3))):  # 1-3 results
        score = 0.9 - (i * 0.1) - ((hash(query + str(i)) % 100) / 1000)
        
        result = {
            "id": f"result_{i+1}_{hash(query) % 1000:04d}",
            "score": score,
            "content": f"Relevant information found for your query '{query}'. This content was identified by the {agent} during {doc_type} analysis.",
            "modality": modality,
            "agent": agent,
            "confidence": 0.85 + ((hash(query + str(i)) % 150) / 1000),
            "content_type": content_type,
            "page": (hash(query + str(i)) % 5) + 1 if document_id else 1,
            "reasoning": {
                "factors": ["semantic_relevance", "context_matching", "agent_confidence"],
                "explanation": f"High relevance due to {content_type.replace('_', ' ')} analysis by {agent}"
            }
        }
        
        if modality in ["visual", "fused"]:
            result["bounding_box"] = {
                "x": 10 + (hash(query + str(i)) % 70),
                "y": 15 + (hash(query + str(i)) % 65),
                "width": 20 + (hash(query + str(i)) % 40),
                "height": 15 + (hash(query + str(i)) % 35)
            }
        
        results.append(result)
    
    return {
        "success": True,
        "document_id": document_id,
        "query": query,
        "results": sorted(results, key=lambda x: x["score"], reverse=True),
        "count": len(results),
        "modality_breakdown": {
            "text": len([r for r in results if r["modality"] == "text"]),
            "visual": len([r for r in results if r["modality"] == "visual"]),
            "fused": len([r for r in results if r["modality"] == "fused"])
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/documents")
async def list_documents():
    """List all documents with processing status"""
    docs = []
    for doc_id, doc_info in documents_db.items():
        status = processing_status.get(doc_id, {})
        results = processing_results.get(doc_id, {})
        
        doc_data = {
            **doc_info,
            "processing_status": status.get("status", "uploaded"),
            "processing_progress": status.get("progress", 0),
            "has_results": doc_id in processing_results,
            "risk_score": results.get("risk_score") if results else None,
            "processing_time": status.get("processing_time") if status else None,
            "last_updated": status.get("timestamp", doc_info.get("uploaded_at"))
        }
        docs.append(doc_data)
    
    return {
        "count": len(docs),
        "documents": sorted(docs, key=lambda x: x.get("last_updated", ""), reverse=True),
        "statistics": {
            "uploaded": len([d for d in docs if d["processing_status"] == "uploaded"]),
            "processing": len([d for d in docs if d["processing_status"] == "processing"]),
            "completed": len([d for d in docs if d["processing_status"] == "completed"]),
            "cached": len([d for d in docs if d.get("has_cache", False)]),
            "average_processing_time": sum(d.get("processing_time", 0) for d in docs) / max(len([d for d in docs if d.get("processing_time", 0) > 0]), 1)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/debug/{document_id}")
async def debug_document(document_id: str):
    """Debug document processing"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = documents_db[document_id]
    status = processing_status.get(document_id, {})
    
    # Check for stuck agents
    stuck_agents = []
    if "agents" in status:
        for agent_name, agent_data in status["agents"].items():
            if agent_data.get("status") in ["processing", "running"]:
                start_time = agent_data.get("start_time")
                if start_time:
                    try:
                        elapsed = (datetime.now() - datetime.fromisoformat(start_time.replace('Z', '+00:00'))).total_seconds()
                        if elapsed > 10:  # 10 seconds threshold for "stuck"
                            stuck_agents.append({
                                "agent": agent_name,
                                "elapsed_seconds": elapsed,
                                "progress": agent_data.get("progress", 0),
                                "last_step": agent_data.get("processing_steps", [{}])[-1].get("step", "unknown") if agent_data.get("processing_steps") else "unknown"
                            })
                    except:
                        pass
    
    return {
        "document_id": document_id,
        "document_info": {
            "filename": doc_info.get("filename"),
            "type": doc_info.get("document_type"),
            "size": doc_info.get("file_size"),
            "fingerprint": doc_info.get("fingerprint", "")[:16] + "...",
            "uploaded": doc_info.get("uploaded_at")
        },
        "processing_status": {
            "status": status.get("status", "unknown"),
            "progress": status.get("progress", 0),
            "message": status.get("message", "No message"),
            "timestamp": status.get("timestamp", "Unknown"),
            "agents_count": len(status.get("agents", {})),
            "active_agents": [name for name, data in status.get("agents", {}).items() if data.get("status") in ["processing", "running"]]
        },
        "cache_info": {
            "in_cache": doc_info.get("fingerprint") in processing_cache,
            "cache_key": doc_info.get("fingerprint", "")[:16] + "..." if doc_info.get("fingerprint") else None,
            "cache_size": len(processing_cache)
        },
        "performance": {
            "stuck_agents": stuck_agents,
            "stuck_count": len(stuck_agents),
            "all_agents_completed": all(
                agent.get("status") == "completed" 
                for agent in status.get("agents", {}).values()
            ) if status.get("agents") else False,
            "overall_progress_calculated": calculate_overall_progress(document_id)
        },
        "recommendations": [
            "Check agent processing steps" if stuck_agents else "All agents running normally",
            "Consider force refresh if stuck > 30s" if len(stuck_agents) > 0 else "Processing progressing normally",
            "Check cache for instant results" if doc_info.get("fingerprint") in processing_cache else "No cache available"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/process/force-complete/{document_id}")
async def force_complete(document_id: str):
    """Force complete processing for stuck documents"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = documents_db[document_id]
    doc_type = doc_info["document_type"]
    
    # Generate instant results
    template = agent_templates.get(doc_type, agent_templates["report"])
    
    # Mark all agents as completed
    agents_data = {}
    for agent in template["agents"]:
        config = processor.agent_configs[agent]
        agents_data[agent] = {
            "name": config["name"],
            "description": config["description"],
            "status": "completed",
            "progress": 100,
            "fields_extracted": config["fields"][:3],
            "processing_steps": [{"step": s, "status": "completed"} for s in config["processing_steps"][:2]],
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": 0.1,
            "forced": True
        }
    
    # Generate results
    results = await generate_real_results(document_id, doc_type, agents_data)
    
    # Update all storage
    processing_status[document_id] = {
        "document_id": document_id,
        "status": "completed",
        "progress": 100,
        "message": "Processing force-completed",
        "timestamp": datetime.now().isoformat(),
        "agents": agents_data,
        "forced": True,
        "processing_time": 0.1
    }
    
    processing_results[document_id] = {
        **results,
        "document_id": document_id,
        "forced": True,
        "processing_time": 0.1
    }
    
    # Also cache it
    processing_cache[doc_info["fingerprint"]] = results
    
    logger.info(f"⚡ Force completed processing for {document_id}")
    
    return {
        "success": True,
        "document_id": document_id,
        "message": "Processing force-completed successfully",
        "forced": True,
        "agents_completed": len(template["agents"]),
        "processing_time": 0.1
    }

@app.get("/api/v1/system/stats")
async def system_stats():
    """Get system statistics"""
    return {
        "system": {
            "documents_processed": len(documents_db),
            "cached_results": len(processing_cache),
            "active_processes": len([s for s in processing_status.values() if s.get("status") == "processing"]),
            "completed_processes": len([s for s in processing_status.values() if s.get("status") == "completed"]),
            "upload_directory_size_mb": sum(f.stat().st_size for f in Path(UPLOAD_DIR).rglob('*')) // 1024 // 1024,
            "cache_directory_size_mb": sum(f.stat().st_size for f in Path(CACHE_DIR).rglob('*')) // 1024 // 1024
        },
        "performance": {
            "average_processing_time": sum(
                s.get("processing_time", 0) for s in processing_status.values() 
                if s.get("status") == "completed"
            ) / max(len([s for s in processing_status.values() if s.get("status") == "completed"]), 1),
            "cache_hit_rate": f"{(len([d for d in documents_db.values() if d.get('fingerprint') in processing_cache]) / len(documents_db) * 100):.1f}%" if documents_db else "0%",
            "success_rate": f"{(len([s for s in processing_status.values() if s.get('status') == 'completed']) / len(processing_status) * 100):.1f}%" if processing_status else "0%"
        },
        "agents": {
            agent_name: {
                "processing_time": config["processing_time"],
                "steps": len(config["processing_steps"]),
                "fields": len(config["fields"])
            }
            for agent_name, config in processor.agent_configs.items()
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Vision-Fusion Document Intelligence...")
    logger.info("📌 Mode: Production Simulation")
    logger.info(f"📌 Agents: {list(processor.agent_configs.keys())}")
    logger.info(f"📌 Templates: {list(agent_templates.keys())}")
    logger.info(f"📌 Cache loaded: {len(processing_cache)} entries")
    logger.info("📌 Real processing simulation enabled")
    
    uvicorn.run(
        "fast_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )