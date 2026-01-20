# test_detect_mocks.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🔍 DETECTING POTENTIAL MOCKS IN PHASE 4")
print("=" * 60)

# Check 1: Evaluation metrics - are they hardcoded?
print("\n1. Checking evaluation metrics for hardcoded values...")
try:
    from app.eval.metrics import MetricsCalculator
    import inspect
    
    source = inspect.getsource(MetricsCalculator.generate_complete_report)
    if "0.67" in source or "0.86" in source or "0.82" in source:
        print("   ⚠️ WARNING: Hardcoded accuracy values found in metrics!")
    else:
        print("   ✅ Metrics appear to be calculated, not hardcoded")
except Exception as e:
    print(f"   ❌ Could not check: {e}")

# Check 2: Explainability reports - are they template-based?
print("\n2. Checking explainability report generation...")
try:
    from app.explain.explainability import ExplainabilityGenerator
    import inspect
    
    source = inspect.getsource(ExplainabilityGenerator.generate_explainability_report)
    if "template" in source.lower() or "hardcoded" in source.lower():
        print("   ⚠️ WARNING: Template-based or hardcoded reports!")
    else:
        print("   ✅ Reports appear to be dynamically generated")
except Exception as e:
    print(f"   ❌ Could not check: {e}")

# Check 3: API endpoints - do they actually process documents?
print("\n3. Checking API endpoint implementations...")
try:
    from app.api.routes import phase4_process_document, phase4_background_processing
    import inspect
    
    bg_source = inspect.getsource(phase4_background_processing)
    
    red_flags = [
        "sleep",  # Artificial delays
        "mock",   # Mock data
        "fake",   # Fake data  
        "example", # Example data
        "test_value", # Test values
        "pass",   # Empty implementation
        "...",    # Ellipsis (incomplete)
    ]
    
    flags_found = []
    for flag in red_flags:
        if flag in bg_source.lower():
            flags_found.append(flag)
    
    if flags_found:
        print(f"   ⚠️ WARNING: Potential mock indicators: {flags_found}")
    else:
        print("   ✅ Background processing appears real")
except Exception as e:
    print(f"   ❌ Could not check: {e}")

# Check 4: File output - are reports being saved?
print("\n4. Checking report file generation...")
report_dirs = ["explainability_reports", "evaluation_reports"]
for dir_name in report_dirs:
    if os.path.exists(dir_name):
        files = os.listdir(dir_name)
        print(f"   ✅ {dir_name}: {len(files)} files present")
    else:
        print(f"   ⚠️ {dir_name}: Directory exists but no files?")

print("\n" + "=" * 60)
print("⚠️ RUN THIS TEST AND SHARE THE RESULTS!")