import requests
import json

print("🚀 Testing Document Intelligence API...")
print("=" * 50)

# Test 1: Root endpoint
print("\n1. Testing root endpoint...")
try:
    response = requests.get("http://localhost:8000/", timeout=5)
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Service: {data.get('service')}")
    print(f"   Version: {data.get('version')}")
    print(f"   Status: {data.get('status')}")
    print("   ✅ Root endpoint working!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Health check
print("\n2. Testing health endpoint...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Health: {data.get('status')}")
    print("   ✅ Health check passed!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Upload a test document
print("\n3. Testing document upload...")
try:
    # Create a simple test document
    test_content = b"""TEST DOCUMENT
================
Document Type: Invoice
Date: January 19, 2024
Invoice Number: INV-2024-001
Amount: $1,250.75
Vendor: Test Corporation
Description: Professional Services

This is a test document for the Document Intelligence System.
It contains sample data that should be extracted by the agents."""
    
    files = {'file': ('test_invoice.txt', test_content, 'text/plain')}
    response = requests.post("http://localhost:8000/api/v1/upload", files=files, timeout=30)
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        document_id = data.get('document_id')
        print(f"   ✅ Upload successful!")
        print(f"   Document ID: {document_id}")
        print(f"   Message: {data.get('message')}")
        
        # Wait a bit for processing
        print("\n4. Checking processing status...")
        import time
        time.sleep(3)
        
        # Check status
        status_response = requests.get(f"http://localhost:8000/api/v1/status/{document_id}", timeout=5)
        print(f"   Status endpoint: {status_response.status_code}")
        status_data = status_response.json()
        print(f"   Processing status: {status_data.get('status')}")
        
        # Get results
        print("\n5. Getting results...")
        results_response = requests.get(f"http://localhost:8000/api/v1/results/{document_id}", timeout=10)
        
        if results_response.status_code == 200:
            results = results_response.json()
            print(f"   ✅ Results retrieved!")
            print(f"   Success: {results.get('success')}")
            
            # Show extracted fields count
            extracted_fields = results.get('results', {}).get('extracted_fields', {})
            print(f"   Fields extracted: {len(extracted_fields)}")
            
            # Show first few fields
            if extracted_fields:
                print("\n   📋 Sample extracted data:")
                for i, (field_name, field_data) in enumerate(list(extracted_fields.items())[:3]):
                    value = field_data.get('value', '')
                    if isinstance(value, list):
                        display = str(value[:2]) + ("..." if len(value) > 2 else "")
                    elif isinstance(value, dict):
                        display = str({k: v for k, v in list(value.items())[:2]}) + "..."
                    else:
                        display = str(value)[:50] + ("..." if len(str(value)) > 50 else "")
                    print(f"      • {field_name}: {display}")
            
            # Show metadata
            metadata = results.get('results', {}).get('processing_metadata', {})
            if metadata:
                print(f"\n   📊 Processing metadata:")
                print(f"      • Integrity score: {metadata.get('integrity_score', 0):.2f}")
                print(f"      • Processing time: {metadata.get('processing_time', 0):.2f}s")
                print(f"      • Agents executed: {len(metadata.get('agents_executed', []))}")
                
        else:
            print(f"   ❌ Results not ready yet (status: {results_response.status_code})")
            
    else:
        print(f"   ❌ Upload failed: {response.text}")
        
except Exception as e:
    print(f"   ❌ Upload test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Test complete! 🎉")