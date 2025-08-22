"""
Test utility functions
"""
import pytest
import io
import base64
from PIL import Image
import numpy as np

from utils.auth import get_password_hash, verify_password, create_access_token, verify_token
from utils.image_utils import (
    validate_image, process_image, resize_image, image_to_base64,
    base64_to_image, annotate_detection_image, preprocess_for_tensorflow,
    preprocess_for_pytorch, get_image_info
)
from utils.model_utils import validate_model_file, get_model_info, get_medical_model_templates

class TestAuthUtils:
    """Test authentication utilities"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "test_password123"
        
        # Hash password
        hashed = get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 0
        
        # Verify correct password
        assert verify_password(password, hashed)
        
        # Verify incorrect password
        assert not verify_password("wrong_password", hashed)
    
    def test_jwt_token_creation(self):
        """Test JWT token creation and verification"""
        data = {"sub": "test_user", "role": "admin"}
        
        # Create token
        token = create_access_token(data)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"
        assert payload["role"] == "admin"
    
    def test_invalid_token_verification(self):
        """Test verification of invalid token"""
        invalid_token = "invalid.jwt.token"
        payload = verify_token(invalid_token)
        assert payload is None

class TestImageUtils:
    """Test image utility functions"""
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return Image.new('RGB', (256, 256), color='red')
    
    @pytest.fixture
    def test_image_bytes(self, test_image):
        """Create test image as bytes"""
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def test_validate_image_valid(self, test_image_bytes):
        """Test validating valid image"""
        assert validate_image(test_image_bytes)
    
    def test_validate_image_invalid(self):
        """Test validating invalid image"""
        invalid_data = b"not an image"
        assert not validate_image(invalid_data)
    
    def test_validate_image_too_large(self):
        """Test validating image that's too large"""
        # Create large dummy data (>10MB)
        large_data = b"x" * (11 * 1024 * 1024)
        assert not validate_image(large_data)
    
    def test_process_image(self, test_image_bytes):
        """Test image processing"""
        processed = process_image(test_image_bytes)
        
        assert isinstance(processed, Image.Image)
        assert processed.mode == 'RGB'
    
    def test_resize_image(self, test_image):
        """Test image resizing"""
        target_size = (128, 128)
        resized = resize_image(test_image, target_size, maintain_aspect=False)
        
        assert resized.size == target_size
    
    def test_resize_image_maintain_aspect(self, test_image):
        """Test image resizing with aspect ratio maintained"""
        target_size = (128, 128)
        resized = resize_image(test_image, target_size, maintain_aspect=True)
        
        # Should be at most the target size
        assert resized.width <= target_size[0]
        assert resized.height <= target_size[1]
    
    def test_image_to_base64(self, test_image):
        """Test converting image to base64"""
        base64_str = image_to_base64(test_image)
        
        assert isinstance(base64_str, str)
        assert base64_str.startswith("data:image/png;base64,")
    
    def test_base64_to_image(self, test_image):
        """Test converting base64 to image"""
        # Convert to base64 first
        base64_str = image_to_base64(test_image)
        
        # Convert back to image
        recovered_image = base64_to_image(base64_str)
        
        assert isinstance(recovered_image, Image.Image)
        assert recovered_image.size == test_image.size
    
    def test_annotate_detection_image(self, test_image):
        """Test annotating image with detections"""
        detections = [
            {
                "x1": 50, "y1": 50, "x2": 150, "y2": 150,
                "label": "test_object", "confidence": 0.95
            }
        ]
        
        annotated = annotate_detection_image(test_image, detections)
        
        assert isinstance(annotated, Image.Image)
        assert annotated.size == test_image.size
    
    def test_preprocess_for_tensorflow(self, test_image):
        """Test preprocessing for TensorFlow"""
        processed = preprocess_for_tensorflow(test_image, (224, 224))
        
        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_preprocess_for_pytorch(self, test_image):
        """Test preprocessing for PyTorch"""
        processed = preprocess_for_pytorch(test_image, (224, 224))
        
        assert processed.shape == (1, 3, 224, 224)
        assert processed.dtype == np.float32
    
    def test_get_image_info(self, test_image):
        """Test getting image information"""
        info = get_image_info(test_image)
        
        assert info["width"] == 256
        assert info["height"] == 256
        assert info["mode"] == "RGB"
        assert "size_bytes" in info

class TestModelUtils:
    """Test model utility functions"""
    
    def test_validate_model_file_valid(self, mock_yolo_model):
        """Test validating valid model file"""
        assert validate_model_file(mock_yolo_model)
    
    def test_validate_model_file_nonexistent(self):
        """Test validating nonexistent model file"""
        assert not validate_model_file("nonexistent_model.pt")
    
    def test_validate_model_file_invalid_format(self, temp_dir):
        """Test validating file with invalid format"""
        import os
        invalid_file = os.path.join(temp_dir, "invalid.txt")
        with open(invalid_file, "w") as f:
            f.write("invalid content")
        
        assert not validate_model_file(invalid_file)
    
    def test_get_model_info(self, mock_yolo_model):
        """Test getting model information"""
        info = get_model_info(mock_yolo_model)
        
        assert "filename" in info
        assert "size_bytes" in info
        assert "format" in info
        assert "exists" in info
        assert info["exists"] is True
        assert info["format"] == "pt"
    
    def test_get_medical_model_templates(self):
        """Test getting medical model templates"""
        templates = get_medical_model_templates()
        
        assert isinstance(templates, dict)
        assert "mammography_classifier" in templates
        assert "bone_age_estimator" in templates
        assert "retina_segmentation" in templates
        assert "liver_tumor_detector" in templates
        assert "colon_polyp_detector" in templates
        assert "brain_lesion_detector" in templates
        
        # Check template structure
        mammo_template = templates["mammography_classifier"]
        assert "name" in mammo_template
        assert "description" in mammo_template
        assert "model_type" in mammo_template
        assert "task_type" in mammo_template
        assert "medical_domain" in mammo_template

class TestDatabaseUtils:
    """Test database utility functions"""
    
    def test_get_collection(self, test_database):
        """Test getting database collection"""
        from utils.database import get_collection
        
        collection = get_collection("test_collection")
        assert collection is not None
        assert collection.name == "test_collection"
    
    def test_get_database(self, test_database):
        """Test getting database instance"""
        from utils.database import get_database
        
        db = get_database()
        assert db is not None
        assert db.name == test_database