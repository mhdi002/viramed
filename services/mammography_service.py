"""
Mammography classification service for breast cancer detection
"""
import os
import logging
from typing import Dict, Any
from PIL import Image
import numpy as np

from services.base_service import BaseAIService
from models.inference_model import InferenceType, ClassificationResult
from utils.image_utils import preprocess_for_tensorflow

logger = logging.getLogger(__name__)

class MammographyService(BaseAIService):
    """Mammography classification service"""
    
    def __init__(self, model_path: str, model_name: str = "mammography_classifier"):
        super().__init__(model_path, model_name, "mammography")
        self.classes = ["Benign", "Malignant"]
        self.input_size = (224, 224)
        
    def load_model(self) -> bool:
        """Load TensorFlow mammography model"""
        try:
            import tensorflow as tf
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load model with custom objects if needed
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            self.is_loaded = True
            logger.info(f"Successfully loaded mammography model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load mammography model {self.model_name}: {e}")
            return False
    
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run mammography classification"""
        # Preprocess image
        processed_image = preprocess_for_tensorflow(image, self.input_size)
        
        # Run inference
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get probabilities for each class
        probabilities = predictions[0]
        predicted_class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_id])
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(self.classes):
            prob_dict[class_name] = float(probabilities[i])
        
        classification = ClassificationResult(
            class_id=predicted_class_id,
            label=self.classes[predicted_class_id],
            confidence=confidence,
            probabilities=prob_dict
        )
        
        return {"classification": classification.dict()}
    
    def get_inference_type(self) -> InferenceType:
        """Return classification type"""
        return InferenceType.CLASSIFICATION
    
    def validate_input_image(self, image: Image.Image) -> bool:
        """Validate input for mammography model"""
        if not super().validate_input_image(image):
            return False
            
        # Convert to RGB if needed
        if image.mode not in ['RGB', 'L']:
            return False
        
        # Should be grayscale or RGB medical image
        min_size = 64
        if image.width < min_size or image.height < min_size:
            return False
            
        return True
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for mammography model"""
        # Resize to model input size
        image = image.resize(self.input_size)
        
        # Convert to RGB if grayscale
        if image.mode == 'L':
            image = image.convert('RGB')
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2 preprocessing if needed
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_array = preprocess_input(img_array)
        except ImportError:
            logger.warning("MobileNetV2 preprocessing not available, using standard normalization")
        
        return img_array
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mammography model information"""
        info = super().get_model_info()
        info.update({
            "classes": self.classes,
            "input_size": self.input_size,
            "task": "Breast cancer classification from mammography images"
        })
        return info