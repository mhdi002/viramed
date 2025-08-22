"""
Test AI service functionality
"""
import pytest
import os
from PIL import Image
import numpy as np

from services.yolo_service import YOLOService
from services.mammography_service import MammographyService
from services.bone_age_service import BoneAgeService
from services.retina_service import RetinaService
from services.ms_segmentation_service import MSSegmentationService
from services.model_service import ModelService
from models.inference_model import InferenceType

class TestBaseService:
    """Test base AI service functionality"""
    
    @pytest.fixture
    def sample_image(self):
        """Create sample PIL image"""
        return Image.new('RGB', (256, 256), color='red')
    
    def test_service_initialization(self, mock_yolo_model):
        """Test service initialization"""
        service = YOLOService(mock_yolo_model, "test_model", "liver")
        
        assert service.model_path == mock_yolo_model
        assert service.model_name == "test_model"
        assert service.medical_domain == "liver"
        assert not service.is_loaded
    
    def test_image_validation(self, sample_image, mock_yolo_model):
        """Test image validation"""
        service = YOLOService(mock_yolo_model, "test_model", "liver")
        
        # Valid image
        assert service.validate_input_image(sample_image)
        
        # Invalid image (too small)
        small_image = Image.new('RGB', (10, 10), color='red')
        assert not service.validate_input_image(small_image)
    
    def test_model_info(self, mock_yolo_model):
        """Test getting model information"""
        service = YOLOService(mock_yolo_model, "test_model", "liver")
        info = service.get_model_info()
        
        assert info["model_name"] == "test_model"
        assert info["medical_domain"] == "liver"
        assert info["inference_type"] == "detection"

class TestYOLOService:
    """Test YOLO service functionality"""
    
    def test_inference_type(self, mock_yolo_model):
        """Test YOLO inference type"""
        service = YOLOService(mock_yolo_model, "test_model", "liver")
        assert service.get_inference_type() == InferenceType.DETECTION
    
    def test_supported_domains(self, mock_yolo_model):
        """Test supported medical domains"""
        service = YOLOService(mock_yolo_model, "test_model", "liver")
        domains = service.get_supported_domains()
        
        assert "liver" in domains
        assert "colon" in domains
        assert "brain" in domains

class TestMammographyService:
    """Test mammography service functionality"""
    
    def test_inference_type(self, temp_dir):
        """Test mammography inference type"""
        # Create dummy model file
        model_path = os.path.join(temp_dir, "mammo.h5")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
        
        service = MammographyService(model_path)
        assert service.get_inference_type() == InferenceType.CLASSIFICATION
    
    def test_classes(self, temp_dir):
        """Test mammography classes"""
        model_path = os.path.join(temp_dir, "mammo.h5")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
            
        service = MammographyService(model_path)
        assert service.classes == ["Benign", "Malignant"]
    
    def test_image_preprocessing(self, temp_dir):
        """Test image preprocessing"""
        model_path = os.path.join(temp_dir, "mammo.h5")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
            
        service = MammographyService(model_path)
        image = Image.new('RGB', (256, 256), color='red')
        
        processed = service.preprocess_image(image)
        
        assert processed.shape == (1, 224, 224, 3)  # Batch, H, W, C
        assert processed.dtype == np.float32

class TestBoneAgeService:
    """Test bone age service functionality"""
    
    def test_inference_type(self, temp_dir):
        """Test bone age inference type"""
        model_path = os.path.join(temp_dir, "bone_age.pth")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
        
        service = BoneAgeService(model_path)
        assert service.get_inference_type() == InferenceType.REGRESSION
    
    def test_image_preprocessing(self, temp_dir):
        """Test bone age image preprocessing"""
        model_path = os.path.join(temp_dir, "bone_age.pth")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
            
        service = BoneAgeService(model_path)
        image = Image.new('RGB', (256, 256), color='red')
        
        processed = service.preprocess_image(image)
        
        assert processed.shape == (1, 3, 300, 300)  # Batch, C, H, W
        assert processed.dtype == np.float32

class TestRetinaService:
    """Test retina segmentation service functionality"""
    
    def test_inference_type(self, temp_dir):
        """Test retina inference type"""
        model_path = os.path.join(temp_dir, "retina.pth")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
        
        service = RetinaService(model_path)
        assert service.get_inference_type() == InferenceType.SEGMENTATION
    
    def test_classes(self, temp_dir):
        """Test retina segmentation classes"""
        model_path = os.path.join(temp_dir, "retina.pth")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
            
        service = RetinaService(model_path)
        expected_classes = ["background", "vessels", "optic_disc", "lesions"]
        assert service.classes == expected_classes

class TestMSSegmentationService:
    """Test MS segmentation service functionality"""
    
    def test_inference_type(self, temp_dir):
        """Test MS segmentation inference type"""
        model_path = os.path.join(temp_dir, "ms_seg.pth")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
        
        service = MSSegmentationService(model_path)
        assert service.get_inference_type() == InferenceType.SEGMENTATION
    
    def test_classes(self, temp_dir):
        """Test MS segmentation classes"""
        model_path = os.path.join(temp_dir, "ms_seg.pth")
        with open(model_path, "wb") as f:
            f.write(b"dummy model")
            
        service = MSSegmentationService(model_path)
        expected_classes = ["background", "ms_lesion", "brain_tissue", "csf"]
        assert service.classes == expected_classes

class TestModelService:
    """Test model service functionality"""
    
    def test_service_initialization(self, test_database):
        """Test model service initialization"""
        service = ModelService()
        assert service.collection is not None
        assert isinstance(service.loaded_services, dict)
    
    def test_get_all_models(self, test_database):
        """Test getting all models"""
        service = ModelService()
        models = service.get_all_models()
        assert isinstance(models, list)
    
    def test_model_templates(self, test_database):
        """Test getting model templates"""
        service = ModelService()
        templates = service.get_model_templates()
        
        assert isinstance(templates, dict)
        assert "mammography_classifier" in templates
        assert "bone_age_estimator" in templates
        assert "retina_segmentation" in templates
    
    def test_service_status(self, test_database):
        """Test getting service status"""
        service = ModelService()
        status = service.get_service_status()
        assert isinstance(status, dict)