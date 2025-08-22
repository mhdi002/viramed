"""
Test configuration and fixtures
"""
import pytest
import asyncio
import os
import tempfile
from typing import Generator
from fastapi.testclient import TestClient
from pymongo import MongoClient
from PIL import Image
import io

# Import main app
from main import app
from config.settings import settings
from utils.database import get_mongo_client, initialize_database
from utils.auth import get_password_hash

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_database():
    """Setup test database"""
    # Use test database
    test_db_name = "medical_ai_system_test"
    
    # Override settings for testing
    original_db_name = settings.DATABASE_NAME
    settings.DATABASE_NAME = test_db_name
    
    # Initialize test database
    try:
        initialize_database()
        yield test_db_name
    finally:
        # Cleanup: Drop test database
        client = get_mongo_client()
        client.drop_database(test_db_name)
        settings.DATABASE_NAME = original_db_name

@pytest.fixture
def client(test_database) -> Generator[TestClient, None, None]:
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_directory:
        yield temp_directory

@pytest.fixture
def sample_image():
    """Create sample image for testing"""
    # Create a simple RGB image
    image = Image.new('RGB', (256, 256), color='red')
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

@pytest.fixture
def admin_user(test_database):
    """Create admin user for testing"""
    from utils.database import get_collection
    from models.user_model import User, UserRole
    
    users_collection = get_collection("users")
    
    # Create admin user
    admin_user = User(
        username="test_admin",
        email="admin@test.com",
        full_name="Test Administrator",
        role=UserRole.ADMIN,
        permissions=["*"]
    )
    
    user_data = admin_user.dict()
    user_data["password_hash"] = get_password_hash("test_password")
    
    users_collection.insert_one(user_data)
    
    return admin_user

@pytest.fixture
def doctor_user(test_database):
    """Create doctor user for testing"""
    from utils.database import get_collection
    from models.user_model import User, UserRole
    
    users_collection = get_collection("users")
    
    # Create doctor user
    doctor_user = User(
        username="test_doctor",
        email="doctor@test.com",
        full_name="Test Doctor",
        role=UserRole.DOCTOR,
        permissions=["inference", "view_models", "view_reports"]
    )
    
    user_data = doctor_user.dict()
    user_data["password_hash"] = get_password_hash("test_password")
    
    users_collection.insert_one(user_data)
    
    return doctor_user

@pytest.fixture
def auth_headers_admin(client, admin_user):
    """Get auth headers for admin user"""
    response = client.post("/api/auth/login", json={
        "username": "test_admin",
        "password": "test_password"
    })
    
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def auth_headers_doctor(client, doctor_user):
    """Get auth headers for doctor user"""
    response = client.post("/api/auth/login", json={
        "username": "test_doctor", 
        "password": "test_password"
    })
    
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def sample_model_data():
    """Sample model data for testing"""
    return {
        "name": "test_model",
        "filename": "test_model.pt",
        "model_type": "yolo",
        "task_type": "detection",
        "medical_domain": "liver",
        "format": "pt",
        "description": "Test YOLO model for liver detection",
        "classes": ["tumor", "lesion"],
        "input_size": 640
    }

@pytest.fixture
def mock_yolo_model(temp_dir):
    """Create mock YOLO model file"""
    model_path = os.path.join(temp_dir, "test_model.pt")
    
    # Create empty file to simulate model
    with open(model_path, "wb") as f:
        f.write(b"mock model data")
    
    return model_path