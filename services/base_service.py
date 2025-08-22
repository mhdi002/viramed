"""
Base service class for all AI model services
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

from models.inference_model import InferenceResponse, InferenceType
from utils.image_utils import get_image_info

logger = logging.getLogger(__name__)

class BaseAIService(ABC):
    """Base class for all AI model services"""
    
    def __init__(self, model_path: str, model_name: str, medical_domain: str):
        self.model_path = model_path
        self.model_name = model_name  
        self.medical_domain = medical_domain
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load the AI model"""
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Make prediction on image"""
        pass
    
    @abstractmethod
    def get_inference_type(self) -> InferenceType:
        """Get the type of inference this service performs"""
        pass
    
    def ensure_model_loaded(self) -> bool:
        """Ensure model is loaded before inference"""
        if not self.is_loaded:
            success = self.load_model()
            if not success:
                logger.error(f"Failed to load model {self.model_name}")
                return False
        return True
    
    def create_inference_response(self, image: Image.Image, prediction_result: Dict[str, Any], 
                                processing_time: float, **kwargs) -> InferenceResponse:
        """Create standardized inference response"""
        
        # Get image information
        image_info = get_image_info(image)
        
        # Base response
        response_data = {
            "success": True,
            "inference_type": self.get_inference_type(),
            "model_name": self.model_name,
            "model_domain": self.medical_domain,
            "processing_time": processing_time,
            "image_info": image_info,
            "metadata": kwargs.get("metadata", {})
        }
        
        # Add specific results based on inference type
        inference_type = self.get_inference_type()
        
        if inference_type == InferenceType.DETECTION:
            response_data["detections"] = prediction_result.get("detections", [])
            
        elif inference_type == InferenceType.CLASSIFICATION:
            response_data["classification"] = prediction_result.get("classification")
            
        elif inference_type == InferenceType.SEGMENTATION:
            response_data["segmentation"] = prediction_result.get("segmentation")
            
        elif inference_type == InferenceType.REGRESSION:
            response_data["regression"] = prediction_result.get("regression")
        
        # Add annotated image if provided
        if "annotated_image" in prediction_result:
            response_data["annotated_image"] = prediction_result["annotated_image"]
        
        return InferenceResponse(**response_data)
    
    def run_inference(self, image: Image.Image, **kwargs) -> InferenceResponse:
        """Run complete inference pipeline"""
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not self.ensure_model_loaded():
                raise Exception(f"Failed to load model {self.model_name}")
            
            # Run prediction
            prediction_result = self.predict(image, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create response
            response = self.create_inference_response(
                image, prediction_result, processing_time, **kwargs
            )
            
            logger.info(f"Inference completed for {self.model_name} in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Inference failed for {self.model_name}: {e}")
            
            # Return error response
            return InferenceResponse(
                success=False,
                inference_type=self.get_inference_type(),
                model_name=self.model_name,
                model_domain=self.medical_domain,
                processing_time=processing_time,
                image_info=get_image_info(image),
                metadata={"error": str(e)}
            )
    
    def validate_input_image(self, image: Image.Image) -> bool:
        """Validate input image for this model"""
        if not isinstance(image, Image.Image):
            return False
            
        # Basic validation - can be overridden by subclasses
        min_size = 32
        max_size = 4096
        
        if image.width < min_size or image.height < min_size:
            return False
            
        if image.width > max_size or image.height > max_size:
            return False
            
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "medical_domain": self.medical_domain,
            "inference_type": self.get_inference_type().value,
            "is_loaded": self.is_loaded
        }