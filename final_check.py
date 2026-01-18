import requests
import sys

def check_system():
    print("🔍 Final System Verification")
    print("=" * 50)
    
    # Check 1: API is responding
    print("\n1. Checking API availability...")
    try:
        response = requests.get("http://localhost:8000/", timeout=3)
        if response.status_code == 200:
            print("   ✅ API is running")
        else:
            print(f"   ❌ API returned {response.status_code}")
            return False
    except:
        print("   ❌ API not reachable")
        return False
    
    # Check 2: Can upload and process
    print("\n2. Testing document processing...")
    try:
        test_content = b"Test document for final verification"
        files = {'file': ('test.txt', test_content, 'text/plain')}
        response = requests.post("http://localhost:8000/api/v1/upload", files=files, timeout=10)
        
        if response.status_code == 200:
            doc_id = response.json().get('document_id')
            print(f"   ✅ Document uploaded (ID: {doc_id})")
            
            # Quick status check
            import time
            time.sleep(2)
            status = requests.get(f"http://localhost:8000/api/v1/status/{doc_id}", timeout=5)
            print(f"   ✅ Processing status: {status.json().get('status')}")
            return True
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Processing test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = check_system()
    print("\n" + "=" * 50)
    if success:
        print("🎉 SYSTEM VERIFICATION PASSED!")
        print("\nNext steps:")
        print("1. UI: streamlit run ui_app.py")
        print("2. Open browser to http://localhost:8501")
        print("3. Upload documents and view results!")
    else:
        print("🔧 System needs attention")
        sys.exit(1)