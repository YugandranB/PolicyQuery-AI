#!/usr/bin/env python3
"""
Test script for PolicyQuery AI API
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/api/v1/hackrx/run"
AUTH_TOKEN = "3ac5ba63774c0a827bb0436425359a4bc2519da9f4da61af84b5685a1237d4bc"

# Sample test data
TEST_REQUEST = {
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": [
        "What is this document about?",
        "What is the main topic?",
        "Is this a test document?"
    ]
}

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")

def test_api_without_auth():
    """Test API without authentication"""
    print("\n🔍 Testing API without authentication...")
    try:
        response = requests.post(API_ENDPOINT, json=TEST_REQUEST)
        if response.status_code == 401:
            print("✅ Authentication required (expected)")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Authentication test error: {e}")

def test_api_with_auth():
    """Test API with authentication"""
    print("\n🔍 Testing API with authentication...")
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Sending request (this may take a few moments)...")
        start_time = time.time()
        
        response = requests.post(API_ENDPOINT, headers=headers, json=TEST_REQUEST)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Request completed in {duration:.2f} seconds")
        
        if response.status_code == 200:
            print("✅ API request successful")
            result = response.json()
            print(f"📄 Number of questions: {len(TEST_REQUEST['questions'])}")
            print(f"📝 Number of answers: {len(result.get('answers', []))}")
            
            print("\n📋 Answers:")
            for i, answer in enumerate(result.get('answers', []), 1):
                print(f"{i}. {answer[:100]}{'...' if len(answer) > 100 else ''}")
                
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ API test error: {e}")

def test_invalid_request():
    """Test API with invalid request"""
    print("\n🔍 Testing API with invalid request...")
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    invalid_request = {
        "documents": "invalid_url",
        "questions": []
    }
    
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=invalid_request)
        print(f"Response status: {response.status_code}")
        if response.status_code in [400, 422]:
            print("✅ Invalid request properly rejected")
        else:
            print(f"❌ Unexpected response for invalid request: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid request test error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting PolicyQuery AI API Tests")
    print("=" * 50)
    
    # Test basic endpoints
    test_health_endpoint()
    test_root_endpoint()
    
    # Test API functionality
    test_api_without_auth()
    test_api_with_auth()
    test_invalid_request()
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed!")

if __name__ == "__main__":
    main() 