"""
Integration tests for the complete system
"""
import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

class TestCompleteWorkflow:
    """Test complete workflow from user creation to inference"""
    
    def test_complete_admin_workflow(self, client: TestClient, admin_user, auth_headers_admin, mock_yolo_model):
        """Test complete admin workflow"""
        # 1. Admin creates a new user
        user_data = {
            "username": "test_researcher",
            "email": "researcher@test.com",
            "password": "test_password",
            "full_name": "Test Researcher",
            "role": "researcher"
        }
        
        response = client.post("/api/auth/register", json=user_data, headers=auth_headers_admin)
        assert response.status_code == 201
        
        # 2. Admin uploads a model
        with open(mock_yolo_model, "rb") as f:
            response = client.post(
                "/api/models/upload",
                headers=auth_headers_admin,
                files={"file": ("liver_model.pt", f, "application/octet-stream")},
                data={
                    "name": "liver_detector",
                    "model_type": "yolo",
                    "task_type": "detection",
                    "medical_domain": "liver",
                    "description": "Liver tumor detection model"
                }
            )
        assert response.status_code == 200
        
        # 3. Admin lists models to verify upload
        response = client.get("/api/models/", headers=auth_headers_admin)
        assert response.status_code == 200
        models = response.json()
        assert any(model["name"] == "liver_detector" for model in models)
        
        # 4. New user logs in
        response = client.post("/api/auth/login", json={
            "username": "test_researcher",
            "password": "test_password"
        })
        assert response.status_code == 200
        researcher_token = response.json()["access_token"]
        researcher_headers = {"Authorization": f"Bearer {researcher_token}"}
        
        # 5. Researcher views available models
        response = client.get("/api/inference/models/available", headers=researcher_headers)
        assert response.status_code == 200
        
        # 6. Check inference statistics (should be empty initially)
        response = client.get("/api/inference/statistics", headers=researcher_headers)
        assert response.status_code == 200
        stats = response.json()
        assert stats["user_statistics"]["total_inferences"] == 0
    
    def test_model_management_workflow(self, client: TestClient, auth_headers_admin, sample_model_data):
        """Test complete model management workflow"""
        # 1. Create model
        response = client.post("/api/models/", json=sample_model_data, headers=auth_headers_admin)
        assert response.status_code == 201
        model = response.json()
        model_name = model["name"]
        
        # 2. Get model details
        response = client.get(f"/api/models/{model_name}", headers=auth_headers_admin)
        assert response.status_code == 200
        
        # 3. Update model
        response = client.put(
            f"/api/models/{model_name}",
            headers=auth_headers_admin,
            params={
                "description": "Updated description",
                "accuracy": 0.92
            }
        )
        assert response.status_code == 200
        updated_model = response.json()
        assert updated_model["description"] == "Updated description"
        assert updated_model["accuracy"] == 0.92
        
        # 4. List models (should include our model)
        response = client.get("/api/models/", headers=auth_headers_admin)
        assert response.status_code == 200
        models = response.json()
        assert any(m["name"] == model_name for m in models)
        
        # 5. Delete model
        response = client.delete(f"/api/models/{model_name}", headers=auth_headers_admin)
        assert response.status_code == 200
        
        # 6. Verify deletion
        response = client.get(f"/api/models/{model_name}", headers=auth_headers_admin)
        assert response.status_code == 404
    
    def test_user_permission_workflow(self, client: TestClient, auth_headers_admin):
        """Test user permission management workflow"""
        # 1. Create doctor user
        doctor_data = {
            "username": "test_doctor_2",
            "email": "doctor2@test.com",
            "password": "test_password",
            "role": "doctor"
        }
        
        response = client.post("/api/auth/register", json=doctor_data, headers=auth_headers_admin)
        assert response.status_code == 201
        
        # 2. Doctor logs in
        response = client.post("/api/auth/login", json={
            "username": "test_doctor_2",
            "password": "test_password"
        })
        assert response.status_code == 200
        doctor_token = response.json()["access_token"]
        doctor_headers = {"Authorization": f"Bearer {doctor_token}"}
        
        # 3. Doctor can view models
        response = client.get("/api/models/", headers=doctor_headers)
        assert response.status_code == 200
        
        # 4. Doctor cannot create models
        model_data = {
            "name": "unauthorized_model",
            "filename": "test.pt",
            "model_type": "yolo",
            "task_type": "detection",
            "medical_domain": "liver",
            "format": "pt"
        }
        response = client.post("/api/models/", json=model_data, headers=doctor_headers)
        assert response.status_code == 403
        
        # 5. Admin upgrades doctor to researcher
        response = client.put(
            "/api/auth/users/test_doctor_2/role",
            headers=auth_headers_admin,
            params={"new_role": "researcher"}
        )
        assert response.status_code == 200
        
        # 6. Doctor (now researcher) logs in again to get new permissions
        response = client.post("/api/auth/login", json={
            "username": "test_doctor_2",
            "password": "test_password"
        })
        assert response.status_code == 200
        updated_token = response.json()["access_token"]
        updated_headers = {"Authorization": f"Bearer {updated_token}"}
        
        # 7. Now can access batch inference (researcher permission)
        # This would normally work with proper images and models
        response = client.get("/api/inference/history", headers=updated_headers)
        assert response.status_code == 200

class TestErrorHandling:
    """Test error handling across the system"""
    
    def test_authentication_errors(self, client: TestClient):
        """Test various authentication error scenarios"""
        # 1. No token provided
        response = client.get("/api/models/")
        assert response.status_code == 401
        
        # 2. Invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/models/", headers=invalid_headers)
        assert response.status_code == 401
        
        # 3. Login with wrong credentials
        response = client.post("/api/auth/login", json={
            "username": "nonexistent",
            "password": "wrong"
        })
        assert response.status_code == 401
    
    def test_resource_not_found_errors(self, client: TestClient, auth_headers_admin):
        """Test resource not found scenarios"""
        # 1. Get nonexistent model
        response = client.get("/api/models/nonexistent", headers=auth_headers_admin)
        assert response.status_code == 404
        
        # 2. Update nonexistent model
        response = client.put(
            "/api/models/nonexistent",
            headers=auth_headers_admin,
            params={"description": "test"}
        )
        assert response.status_code == 404
        
        # 3. Delete nonexistent model
        response = client.delete("/api/models/nonexistent", headers=auth_headers_admin)
        assert response.status_code == 404
    
    def test_validation_errors(self, client: TestClient, auth_headers_admin):
        """Test input validation errors"""
        # 1. Create model with missing required fields
        invalid_model = {
            "name": "incomplete_model"
            # Missing required fields
        }
        response = client.post("/api/models/", json=invalid_model, headers=auth_headers_admin)
        assert response.status_code == 422  # Validation error
        
        # 2. Upload file with wrong format
        invalid_file_content = b"not a model file"
        response = client.post(
            "/api/models/upload",
            headers=auth_headers_admin,
            files={"file": ("invalid.txt", io.BytesIO(invalid_file_content), "text/plain")},
            data={
                "name": "invalid_model",
                "model_type": "yolo",
                "task_type": "detection",
                "medical_domain": "liver"
            }
        )
        assert response.status_code == 400
    
    def test_permission_errors(self, client: TestClient, auth_headers_doctor):
        """Test permission-related errors"""
        # 1. Non-admin trying to create user
        user_data = {
            "username": "unauthorized_user",
            "email": "unauthorized@test.com",
            "password": "password",
            "role": "viewer"
        }
        response = client.post("/api/auth/register", json=user_data, headers=auth_headers_doctor)
        assert response.status_code == 403
        
        # 2. Non-admin trying to list users
        response = client.get("/api/auth/users", headers=auth_headers_doctor)
        assert response.status_code == 403
        
        # 3. Doctor trying to manage models
        response = client.post("/api/models/refresh", headers=auth_headers_doctor)
        assert response.status_code == 403

class TestDataConsistency:
    """Test data consistency across the system"""
    
    def test_model_count_consistency(self, client: TestClient, auth_headers_admin):
        """Test that model counts are consistent"""
        # Get initial count
        response = client.get("/api/models/", headers=auth_headers_admin)
        initial_count = len(response.json())
        
        # Create a model
        model_data = {
            "name": "consistency_test_model",
            "filename": "test.pt",
            "model_type": "yolo",
            "task_type": "detection",
            "medical_domain": "liver",
            "format": "pt"
        }
        
        response = client.post("/api/models/", json=model_data, headers=auth_headers_admin)
        assert response.status_code == 201
        
        # Check count increased
        response = client.get("/api/models/", headers=auth_headers_admin)
        new_count = len(response.json())
        assert new_count == initial_count + 1
        
        # Delete the model
        response = client.delete("/api/models/consistency_test_model", headers=auth_headers_admin)
        assert response.status_code == 200
        
        # Check count decreased
        response = client.get("/api/models/", headers=auth_headers_admin)
        final_count = len(response.json())
        assert final_count == initial_count
    
    def test_user_statistics_consistency(self, client: TestClient, auth_headers_doctor):
        """Test that user statistics are consistent"""
        # Get initial statistics
        response = client.get("/api/inference/statistics", headers=auth_headers_doctor)
        assert response.status_code == 200
        initial_stats = response.json()["user_statistics"]
        
        # The statistics should be mathematically consistent
        total = initial_stats["total_inferences"]
        successful = initial_stats["successful_inferences"]
        failed = initial_stats["failed_inferences"]
        
        assert successful + failed == total
        
        if total > 0:
            expected_rate = (successful / total) * 100
            assert abs(initial_stats["success_rate"] - expected_rate) < 0.01
        else:
            assert initial_stats["success_rate"] == 0