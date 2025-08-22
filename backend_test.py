#!/usr/bin/env python3

import requests
import sys
from datetime import datetime
import json

class YOLOBackendTester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, expected_keys=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)
            else:
                print(f"❌ Failed - Unsupported method: {method}")
                return False, {}

            print(f"   Status Code: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                
                # Try to parse JSON response
                try:
                    json_response = response.json()
                    print(f"   Response: {json.dumps(json_response, indent=2)}")
                    
                    # Check for expected keys if provided
                    if expected_keys:
                        for key in expected_keys:
                            if key not in json_response:
                                print(f"⚠️  Warning: Expected key '{key}' not found in response")
                            else:
                                print(f"   ✓ Found expected key: {key}")
                    
                    return True, json_response
                except json.JSONDecodeError:
                    print(f"   Response (non-JSON): {response.text}")
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_response = response.json()
                    print(f"   Error Response: {json.dumps(error_response, indent=2)}")
                except:
                    print(f"   Error Response (non-JSON): {response.text}")
                return False, {}

        except requests.exceptions.ConnectionError as e:
            print(f"❌ Failed - Connection Error: {str(e)}")
            return False, {}
        except requests.exceptions.Timeout as e:
            print(f"❌ Failed - Timeout Error: {str(e)}")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health(self):
        """Test health endpoint"""
        success, response = self.run_test(
            "Backend Health Check",
            "GET",
            "api/health",
            200,
            expected_keys=["status", "time"]
        )
        
        if success and response:
            if response.get("status") == "ok":
                print("   ✓ Health status is 'ok'")
            else:
                print(f"   ⚠️  Health status is '{response.get('status')}', expected 'ok'")
                
            if "time" in response:
                try:
                    # Validate ISO format
                    datetime.fromisoformat(response["time"].replace('Z', '+00:00'))
                    print("   ✓ Time is in valid ISO format")
                except:
                    print(f"   ⚠️  Time format may be invalid: {response.get('time')}")
        
        return success

    def test_models_refresh(self):
        """Test models refresh endpoint"""
        success, response = self.run_test(
            "Models Refresh (no models expected)",
            "POST",
            "api/models/refresh",
            200,
            expected_keys=["count", "models"]
        )
        
        if success and response:
            count = response.get("count", -1)
            models = response.get("models", [])
            
            if count == 0:
                print("   ✓ Count is 0 as expected (no models)")
            else:
                print(f"   ⚠️  Count is {count}, expected 0")
                
            if isinstance(models, list) and len(models) == 0:
                print("   ✓ Models array is empty as expected")
            else:
                print(f"   ⚠️  Models is not empty array: {models}")
        
        return success

    def test_list_models(self):
        """Test list models endpoint"""
        success, response = self.run_test(
            "List Models (empty expected)",
            "GET",
            "api/models",
            200,
            expected_keys=["models"]
        )
        
        if success and response:
            models = response.get("models", [])
            
            if isinstance(models, list) and len(models) == 0:
                print("   ✓ Models array is empty as expected")
            else:
                print(f"   ⚠️  Models is not empty array: {models}")
        
        return success

def main():
    print("🚀 Starting YOLO Backend API Tests")
    print("=" * 50)
    
    # Setup
    tester = YOLOBackendTester("http://localhost:8001")
    
    # Run tests in order as specified
    print("\n1️⃣  Testing Backend Health...")
    health_passed = tester.test_health()
    
    print("\n2️⃣  Testing Models Refresh...")
    refresh_passed = tester.test_models_refresh()
    
    print("\n3️⃣  Testing List Models...")
    list_passed = tester.test_list_models()
    
    # Print final results
    print("\n" + "=" * 50)
    print("📊 FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Success Rate: {(tester.tests_passed/tester.tests_run*100):.1f}%")
    
    print("\nIndividual Test Results:")
    print(f"  Health Check: {'✅ PASS' if health_passed else '❌ FAIL'}")
    print(f"  Models Refresh: {'✅ PASS' if refresh_passed else '❌ FAIL'}")
    print(f"  List Models: {'✅ PASS' if list_passed else '❌ FAIL'}")
    
    if tester.tests_passed == tester.tests_run:
        print("\n🎉 All backend tests passed!")
        return 0
    else:
        print(f"\n⚠️  {tester.tests_run - tester.tests_passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())