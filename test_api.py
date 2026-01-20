# phase4_diagnostic.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🔍 PHASE 4 DIAGNOSTIC")
print("=" * 60)

# Check 1: Can we import Phase 4 modules?
try:
    from app.explain.provenance import ProvenanceTracker
    from app.explain.explainability import ExplainabilityGenerator
    from app.eval.evaluator import DocumentEvaluator
    print("✅ Phase 4 modules import OK")
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Check 2: Are API endpoints properly defined?
try:
    from app.main import app
    phase4_count = 0
    for route in app.routes:
        if hasattr(route, "path") and "phase4" in route.path:
            phase4_count += 1
    print(f"✅ {phase4_count} Phase 4 endpoints registered")
except Exception as e:
    print(f"❌ API check failed: {e}")

# Check 3: Can we create Phase 4 objects?
try:
    from app.core.models import MultiModalDocument, ProvenanceRecord
    doc = MultiModalDocument(document_id="diagnostic")
    print("✅ Phase 4 models work")
except Exception as e:
    print(f"❌ Model creation failed: {e}")

print("=" * 60)