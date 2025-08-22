"""
YOLO model inference service for medical object detection
"""
import os
import logging
from typing import Dict, Any, List
from PIL import Image
import numpy as np

from services.base_service import BaseAIService
from models.inference_model import InferenceType, DetectionBox
from utils.image_utils import annotate_detection_image, image_to_base64

logger = logging.getLogger(__name__)

class YOLOService(BaseAIService):
    """YOLO model service for medical object detection"""
    
    def __init__(self, model_path: str, model_name: str, medical_domain: str):
        super().__init__(model_path, model_name, medical_domain)
        self.classes = []
        
    def load_model(self) -> bool:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = YOLO(self.model_path)
            
            # Extract class names
            if hasattr(self.model, 'names') and self.model.names:
                self.classes = list(self.model.names.values())
                logger.info(f"Loaded YOLO model with classes: {self.classes}")
            
            self.is_loaded = True
            logger.info(f"Successfully loaded YOLO model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model {self.model_name}: {e}")
            return False
    
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run YOLO inference"""
        confidence_threshold = kwargs.get('confidence_threshold', 0.25)
        iou_threshold = kwargs.get('iou_threshold', 0.45)
        return_image = kwargs.get('return_image', True)
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = self.model.predict(
            source=img_array,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Extract box coordinates and info
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else 0.0
                    cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                    
                    # Get class label
                    label = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
                    
                    detection = DetectionBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]), 
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                        class_id=cls_id,
                        label=label
                    )
                    
                    detections.append(detection.dict())
        
        result_data = {"detections": detections}
        
        # Add annotated image if requested
        if return_image and detections:
            annotated_img = annotate_detection_image(image, detections)
            result_data["annotated_image"] = image_to_base64(annotated_img)
        
        return result_data
    
    def get_inference_type(self) -> InferenceType:
        """Return detection type"""
        return InferenceType.DETECTION
    
    def get_supported_domains(self) -> List[str]:
        """Get supported medical domains"""
        return ["liver", "colon", "brain", "general"]
    
    def validate_input_image(self, image: Image.Image) -> bool:
        """Validate input for YOLO model"""
        if not super().validate_input_image(image):
            return False
            
        # YOLO specific validation
        if image.mode not in ['RGB', 'RGBA']:
            return False
            
        return True