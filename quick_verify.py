import requests
import time

print("🚀 Quick System Verification")
print("=" * 40)

# Test 1: Quick test endpoint (should be fast)
print("\n1. Testing quick endpoint...")
try:
    start = time.time()
    response = requests.get("http://localhost:8000/api/v1/quick-test", timeout=2)
    elapsed = time.time() - start
    print(f"   Response time: {elapsed:.2f}s")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Quick endpoint working!")
    else:
        print(f"   ❌ Unexpected status: {response.text}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Upload with small file
print("\n2. Testing with tiny file...")
try:
    # Create a very small test file
    test_content = b"Tiny test"
    files = {'file': ('tiny.txt', test_content, 'text/plain')}
    
    start = time.time()
    response = requests.post("http://localhost:8000/api/v1/upload", files=files, timeout=5)
    elapsed = time.time() - start
    
    print(f"   Upload time: {elapsed:.2f}s")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        doc_id = data.get('document_id')
        print(f"   ✅ Upload accepted!")
        print(f"   Document ID: {doc_id}")
        
        # Check status after a short delay
        time.sleep(1)
        print(f"\n3. Checking status...")
        status_response = requests.get(f"http://localhost:8000/api/v1/status/{doc_id}", timeout=2)
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   Status: {status_data.get('status')}")
            print(f"   ✅ Status endpoint working!")
        else:
            print(f"   ❌ Status check failed: {status_response.status_code}")
    else:
        print(f"   ❌ Upload failed: {response.text}")
        
except Exception as e:
    print(f"   ❌ Test failed: {e}")

print("\n" + "=" * 40)
print("Verification complete!")