"""
Test model management functionality
"""
import pytest
import os
from fastapi.testclient import TestClient

class TestModelManagement:
    """Test model management endpoints"""
    
    def test_list_models_empty(self, client: TestClient, auth_headers_admin):
        """Test listing models when none exist"""
        response = client.get("/api/models/", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_create_model(self, client: TestClient, auth_headers_admin, sample_model_data):
        """Test creating a new model"""
        response = client.post("/api/models/", json=sample_model_data, headers=auth_headers_admin)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["name"] == sample_model_data["name"]
        assert data["model_type"] == sample_model_data["model_type"]
        assert data["medical_domain"] == sample_model_data["medical_domain"]
    
    def test_get_model_by_name(self, client: TestClient, auth_headers_admin, sample_model_data):
        """Test getting a specific model"""
        # First create the model
        client.post("/api/models/", json=sample_model_data, headers=auth_headers_admin)
        
        # Then retrieve it
        response = client.get(f"/api/models/{sample_model_data['name']}", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == sample_model_data["name"]
    
    def test_get_nonexistent_model(self, client: TestClient, auth_headers_admin):
        """Test getting a model that doesn't exist"""
        response = client.get("/api/models/nonexistent_model", headers=auth_headers_admin)
        
        assert response.status_code == 404
    
    def test_update_model(self, client: TestClient, auth_headers_admin, sample_model_data):
        """Test updating model information"""
        # Create model first
        client.post("/api/models/", json=sample_model_data, headers=auth_headers_admin)
        
        # Update model
        update_data = {
            "description": "Updated description",
            "accuracy": 0.95
        }
        
        response = client.put(
            f"/api/models/{sample_model_data['name']}", 
            params=update_data,
            headers=auth_headers_admin
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
        assert data["accuracy"] == 0.95
    
    def test_delete_model(self, client: TestClient, auth_headers_admin, sample_model_data):
        """Test deleting a model"""
        # Create model first
        client.post("/api/models/", json=sample_model_data, headers=auth_headers_admin)
        
        # Delete model
        response = client.delete(f"/api/models/{sample_model_data['name']}", headers=auth_headers_admin)
        
        assert response.status_code == 200
        
        # Verify it's deleted
        response = client.get(f"/api/models/{sample_model_data['name']}", headers=auth_headers_admin)
        assert response.status_code == 404
    
    def test_model_permissions(self, client: TestClient, auth_headers_doctor, sample_model_data):
        """Test model management permissions"""
        # Doctor can view models
        response = client.get("/api/models/", headers=auth_headers_doctor)
        assert response.status_code == 200
        
        # Doctor cannot create models
        response = client.post("/api/models/", json=sample_model_data, headers=auth_headers_doctor)
        assert response.status_code == 403
    
    def test_list_model_templates(self, client: TestClient, auth_headers_admin):
        """Test listing available model templates"""
        response = client.get("/api/models/templates/list", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert isinstance(data["templates"], dict)
    
    def test_list_medical_domains(self, client: TestClient, auth_headers_admin):
        """Test listing medical domains"""
        response = client.get("/api/models/domains/list", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        assert "domains" in data
        assert isinstance(data["domains"], list)
    
    def test_list_model_types(self, client: TestClient, auth_headers_admin):
        """Test listing model types"""
        response = client.get("/api/models/types/list", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        assert "model_types" in data
        assert "task_types" in data
    
    def test_get_service_status(self, client: TestClient, auth_headers_admin):
        """Test getting service status"""
        response = client.get("/api/models/status/services", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        assert "services" in data

class TestModelUpload:
    """Test model file upload functionality"""
    
    def test_upload_model_file(self, client: TestClient, auth_headers_admin, mock_yolo_model):
        """Test uploading a model file"""
        with open(mock_yolo_model, "rb") as f:
            response = client.post(
                "/api/models/upload",
                headers=auth_headers_admin,
                files={"file": ("test_model.pt", f, "application/octet-stream")},
                data={
                    "name": "uploaded_model",
                    "model_type": "yolo",
                    "task_type": "detection",
                    "medical_domain": "liver",
                    "description": "Uploaded test model"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "uploaded_model"
        assert data["filename"] == "test_model.pt"
    
    def test_upload_invalid_format(self, client: TestClient, auth_headers_admin, temp_dir):
        """Test uploading invalid file format"""
        # Create text file with unsupported extension
        invalid_file = os.path.join(temp_dir, "invalid.txt")
        with open(invalid_file, "w") as f:
            f.write("invalid content")
        
        with open(invalid_file, "rb") as f:
            response = client.post(
                "/api/models/upload",
                headers=auth_headers_admin,
                files={"file": ("invalid.txt", f, "text/plain")},
                data={
                    "name": "invalid_model",
                    "model_type": "yolo",
                    "task_type": "detection",
                    "medical_domain": "liver"
                }
            )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]