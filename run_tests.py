#!/usr/bin/env python3
"""
Test runner for the Medical AI Backend System
"""
import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run all tests and generate coverage report"""
    print("🧪 Running Medical AI Backend System Tests")
    print("=" * 50)
    
    # Change to app directory
    os.chdir(Path(__file__).parent)
    
    try:
        # Install test dependencies
        print("📦 Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "pytest", "pytest-asyncio", "pytest-cov", "httpx"
        ], check=True, capture_output=True)
        
        # Run tests with coverage
        print("🔬 Running unit tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--cov=.", 
            "--cov-report=term-missing",
            "--cov-report=html:tests/coverage_html",
            "--tb=short"
        ], capture_output=True, text=True)
        
        print("📊 Test Results:")
        print("-" * 30)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        # Print summary
        if result.returncode == 0:
            print("✅ All tests passed!")
            print("📈 Coverage report generated in tests/coverage_html/")
        else:
            print("❌ Some tests failed!")
            print(f"Exit code: {result.returncode}")
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tests: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def run_specific_test_category(category):
    """Run specific category of tests"""
    test_files = {
        "auth": "tests/test_auth.py",
        "models": "tests/test_models.py", 
        "services": "tests/test_services.py",
        "inference": "tests/test_inference.py",
        "utils": "tests/test_utils.py",
        "integration": "tests/test_integration.py"
    }
    
    if category not in test_files:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {', '.join(test_files.keys())}")
        return 1
    
    print(f"🧪 Running {category} tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_files[category], 
            "-v",
            "--tb=short"
        ], text=True)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running {category} tests: {e}")
        return 1

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        category = sys.argv[1]
        return run_specific_test_category(category)
    else:
        return run_tests()

if __name__ == "__main__":
    sys.exit(main())