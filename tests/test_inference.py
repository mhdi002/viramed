"""
Test inference functionality
"""
import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

class TestInferenceEndpoints:
    """Test inference API endpoints"""
    
    @pytest.fixture
    def test_image_file(self):
        """Create test image file"""
        image = Image.new('RGB', (512, 512), color='blue')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return ("test_image.png", img_bytes, "image/png")
    
    def test_get_available_models(self, client: TestClient, auth_headers_doctor):
        """Test getting available models for inference"""
        response = client.get("/api/inference/models/available", headers=auth_headers_doctor)
        
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert isinstance(data["available_models"], list)
    
    def test_inference_no_models(self, client: TestClient, auth_headers_doctor, test_image_file):
        """Test inference when no models are available"""
        filename, file_content, content_type = test_image_file
        
        response = client.post(
            "/api/inference/single",
            headers=auth_headers_doctor,
            files={"image": (filename, file_content, content_type)},
            data={"model_name": "nonexistent_model"}
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_inference_invalid_image(self, client: TestClient, auth_headers_doctor):
        """Test inference with invalid image"""
        # Create invalid image data
        invalid_data = b"not an image"
        
        response = client.post(
            "/api/inference/single",
            headers=auth_headers_doctor,
            files={"image": ("invalid.txt", io.BytesIO(invalid_data), "text/plain")},
            data={"model_name": "test_model"}
        )
        
        assert response.status_code == 400
        assert "invalid image" in response.json()["detail"].lower()
    
    def test_inference_permissions(self, client: TestClient, test_image_file):
        """Test inference permissions (requires auth)"""
        filename, file_content, content_type = test_image_file
        
        response = client.post(
            "/api/inference/single",
            files={"image": (filename, file_content, content_type)},
            data={"model_name": "test_model"}
        )
        
        assert response.status_code == 401
    
    def test_batch_inference_size_limit(self, client: TestClient, auth_headers_doctor):
        """Test batch inference size limit"""
        # Create multiple test images (more than limit)
        test_images = []
        for i in range(25):  # More than the 20 limit
            image = Image.new('RGB', (64, 64), color='red')
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            test_images.append((f"test_{i}.png", img_bytes, "image/png"))
        
        response = client.post(
            "/api/inference/batch",
            headers=auth_headers_doctor,
            files=[("images", img) for img in test_images],
            data={"model_name": "test_model"}
        )
        
        assert response.status_code == 400
        assert "batch size too large" in response.json()["detail"].lower()
    
    def test_get_inference_history(self, client: TestClient, auth_headers_doctor):
        """Test getting inference history"""
        response = client.get("/api/inference/history", headers=auth_headers_doctor)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_count" in data
        assert "records" in data
        assert "has_more" in data
        assert isinstance(data["records"], list)
    
    def test_get_inference_statistics(self, client: TestClient, auth_headers_doctor):
        """Test getting inference statistics"""
        response = client.get("/api/inference/statistics", headers=auth_headers_doctor)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user_statistics" in data
        assert "model_usage" in data
        assert "global_statistics" in data
        
        user_stats = data["user_statistics"]
        assert "total_inferences" in user_stats
        assert "successful_inferences" in user_stats
        assert "failed_inferences" in user_stats
        assert "success_rate" in user_stats
    
    def test_admin_global_statistics(self, client: TestClient, auth_headers_admin):
        """Test admin getting global statistics"""
        response = client.get("/api/inference/statistics", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        
        # Admin should see global stats
        global_stats = data["global_statistics"]
        assert "total_system_inferences" in global_stats
        assert "active_models" in global_stats
        assert "total_users" in global_stats

class TestInferenceParameters:
    """Test inference parameter validation"""
    
    @pytest.fixture
    def test_image_file(self):
        """Create test image file"""
        image = Image.new('RGB', (256, 256), color='green')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return ("test.png", img_bytes, "image/png")
    
    def test_confidence_threshold_validation(self, client: TestClient, auth_headers_doctor, test_image_file):
        """Test confidence threshold parameter validation"""
        filename, file_content, content_type = test_image_file
        
        # Test valid confidence threshold
        response = client.post(
            "/api/inference/single",
            headers=auth_headers_doctor,
            files={"image": (filename, file_content, content_type)},
            data={
                "model_name": "test_model",
                "confidence_threshold": 0.5
            }
        )
        
        # Should fail because model doesn't exist, but parameter validation should pass
        assert response.status_code == 404  # Model not found, not parameter error
    
    def test_iou_threshold_validation(self, client: TestClient, auth_headers_doctor, test_image_file):
        """Test IoU threshold parameter validation"""
        filename, file_content, content_type = test_image_file
        
        response = client.post(
            "/api/inference/single",
            headers=auth_headers_doctor,
            files={"image": (filename, file_content, content_type)},
            data={
                "model_name": "test_model",
                "iou_threshold": 0.3
            }
        )
        
        assert response.status_code == 404  # Model not found, not parameter error
    
    def test_return_image_parameter(self, client: TestClient, auth_headers_doctor, test_image_file):
        """Test return_image parameter"""
        filename, file_content, content_type = test_image_file
        
        response = client.post(
            "/api/inference/single",
            headers=auth_headers_doctor,
            files={"image": (filename, file_content, content_type)},
            data={
                "model_name": "test_model",
                "return_image": False
            }
        )
        
        assert response.status_code == 404  # Model not found, not parameter error

class TestInferenceHistory:
    """Test inference history and logging"""
    
    def test_history_pagination(self, client: TestClient, auth_headers_doctor):
        """Test inference history pagination"""
        # Test with limit and offset
        response = client.get(
            "/api/inference/history?limit=10&offset=0",
            headers=auth_headers_doctor
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["records"]) <= 10
        assert "has_more" in data
    
    def test_history_model_filter(self, client: TestClient, auth_headers_doctor):
        """Test filtering history by model name"""
        response = client.get(
            "/api/inference/history?model_name=test_model",
            headers=auth_headers_doctor
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should only return records for the specified model
        for record in data["records"]:
            if "model_name" in record:
                assert record["model_name"] == "test_model"
    
    def test_statistics_calculation(self, client: TestClient, auth_headers_doctor):
        """Test statistics calculation"""
        response = client.get("/api/inference/statistics", headers=auth_headers_doctor)
        
        assert response.status_code == 200
        data = response.json()
        
        user_stats = data["user_statistics"]
        total = user_stats["total_inferences"]
        successful = user_stats["successful_inferences"]
        failed = user_stats["failed_inferences"]
        
        # Verify calculation consistency
        assert successful + failed == total
        
        if total > 0:
            expected_rate = (successful / total) * 100
            assert abs(user_stats["success_rate"] - expected_rate) < 0.01
        else:
            assert user_stats["success_rate"] == 0