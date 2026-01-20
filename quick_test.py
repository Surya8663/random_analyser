# test_simple_imports.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from app.core.models import MultiModalDocument, ProvenanceRecord, ExplainableField
    print("✅ Core models imported successfully")
    
    from app.agents.base_agent import BaseAgent
    print("✅ BaseAgent imported successfully")
    
    from app.agents.vision_agent import VisionAgent
    print("✅ VisionAgent imported successfully")
    
    from app.agents.text_agent import TextAgent
    print("✅ TextAgent imported successfully")
    
    from app.agents.fusion_agent import FusionAgent
    print("✅ FusionAgent imported successfully")
    
    from app.agents.reasoning_agent import ReasoningAgent
    print("✅ ReasoningAgent imported successfully")
    
    from app.agents.orchestrator import Phase3Orchestrator
    print("✅ Phase3Orchestrator imported successfully")
    
    print("\n🎉 All Phase 4 imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()