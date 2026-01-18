#!/usr/bin/env python3
"""Test the document processing system"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_orchestrator():
    """Test the orchestrator"""
    print("🧪 Testing orchestrator...")
    
    try:
        from app.agents.orchestrator import AgentOrchestrator
        print("✅ Orchestrator imported successfully")
        
        # Create orchestrator
        print("🔄 Initializing orchestrator...")
        orchestrator = AgentOrchestrator()
        print(f"✅ Orchestrator initialized")
        
        # Test with empty images
        print("🔄 Testing document processing...")
        result = await orchestrator.process_document([], "test.pdf")
        
        print(f"\n✅ Processing completed!")
        print(f"   Document ID: {result.get('document_id')}")
        print(f"   Success: {result.get('success')}")
        print(f"   Fields extracted: {len(result.get('extracted_fields', {}))}")
        print(f"   Errors: {len(result.get('errors', []))}")
        
        # Show some extracted fields
        extracted_fields = result.get('extracted_fields', {})
        if extracted_fields:
            print(f"\n📋 Extracted fields:")
            for field_name, field in list(extracted_fields.items())[:3]:  # Show first 3
                value = field.get('value', '')
                if isinstance(value, dict):
                    value_str = str(value)[:50]
                else:
                    value_str = str(value)[:50]
                print(f"   - {field_name}: {value_str}...")
        
        # Show any errors
        errors = result.get('errors', [])
        if errors:
            print(f"\n⚠️ Errors:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"   - {error[:100]}...")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api():
    """Test the API endpoints"""
    print("\n🌐 Testing API...")
    
    try:
        import requests
        import time
        
        # Start by checking if server is running
        try:
            print("🔄 Checking if API is running...")
            response = requests.get("http://localhost:8000/", timeout=5)
            print(f"✅ API root: {response.status_code}")
            api_info = response.json()
            print(f"   Service: {api_info.get('service')}")
            print(f"   Version: {api_info.get('version')}")
            print(f"   Status: {api_info.get('status')}")
        except requests.exceptions.ConnectionError:
            print("⚠️ API not running. Start with: uvicorn app.main:app --reload")
            return False
        
        # Test health endpoint
        print("🔄 Testing health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        health = response.json()
        print(f"   Status: {health.get('status')}")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Document Intelligence System Test")
    print("=" * 60)
    
    # Test orchestrator
    orchestrator_ok = await test_orchestrator()
    
    # Test API (if orchestrator worked)
    if orchestrator_ok:
        print("\n" + "-" * 60)
        api_ok = await test_api()
    else:
        api_ok = False
        print("⚠️ Skipping API test due to orchestrator failure")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"   Orchestrator: {'✅ PASS' if orchestrator_ok else '❌ FAIL'}")
    print(f"   API: {'✅ PASS' if api_ok else '⚠️ SKIP/FAIL'}")
    print("=" * 60)
    
    if orchestrator_ok:
        print("\n🎉 System is ready! Next steps:")
        print("1. Start backend: uvicorn app.main:app --reload")
        print("2. Start UI: streamlit run ui_app.py")
        print("3. Open browser to http://localhost:8501")
    else:
        print("\n🔧 System needs fixes. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())