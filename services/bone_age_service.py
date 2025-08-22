"""
Bone age estimation service using PyTorch models
"""
import os
import logging
from typing import Dict, Any
from PIL import Image
import numpy as np
import torch

from services.base_service import BaseAIService
from models.inference_model import InferenceType, RegressionResult
from utils.image_utils import preprocess_for_pytorch

logger = logging.getLogger(__name__)

class BoneAgeService(BaseAIService):
    """Bone age estimation service"""
    
    def __init__(self, model_path: str, model_name: str = "bone_age_estimator"):
        super().__init__(model_path, model_name, "bone_age")
        self.input_size = (300, 300)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> bool:
        """Load PyTorch bone age model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model architecture (using timm for backbone)
            try:
                import timm
                self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1)
                
                # Load state dict
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                
            except ImportError:
                logger.warning("timm not available, using basic PyTorch model")
                # Fallback to basic model structure
                self.model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(64, 1)
                )
                
                # Try to load compatible state dict
                try:
                    if isinstance(checkpoint, dict):
                        compatible_dict = {}
                        for k, v in checkpoint.items():
                            if v.shape == self.model.state_dict()[k].shape:
                                compatible_dict[k] = v
                        self.model.load_state_dict(compatible_dict, strict=False)
                except:
                    logger.warning("Could not load model weights, using random initialization")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded bone age model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load bone age model {self.model_name}: {e}")
            return False
    
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run bone age estimation"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(processed_image).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Convert to age in months
        predicted_age = float(output.cpu().numpy()[0, 0])
        
        # Clamp to reasonable range (0-216 months = 0-18 years)
        predicted_age = max(0, min(216, predicted_age))
        
        # Convert to years for better interpretability
        age_years = predicted_age / 12.0
        
        regression = RegressionResult(
            predicted_value=age_years,
            confidence_interval={
                "lower": max(0, age_years - 1.0),
                "upper": min(18, age_years + 1.0)
            },
            unit="years"
        )
        
        return {"regression": regression.dict()}
    
    def get_inference_type(self) -> InferenceType:
        """Return regression type"""
        return InferenceType.REGRESSION
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for bone age model"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.input_size)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Convert HWC to CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def validate_input_image(self, image: Image.Image) -> bool:
        """Validate input for bone age model"""
        if not super().validate_input_image(image):
            return False
        
        # Should be X-ray image (typically grayscale but can be converted)
        if image.mode not in ['RGB', 'L', 'RGBA']:
            return False
        
        # Minimum size for medical images
        min_size = 128
        if image.width < min_size or image.height < min_size:
            return False
            
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get bone age model information"""
        info = super().get_model_info()
        info.update({
            "input_size": self.input_size,
            "output": "Age in years (0-18)",
            "task": "Bone age estimation from hand X-ray images",
            "device": str(self.device)
        })
        return info