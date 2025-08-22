"""
MS lesion segmentation service using R2UNet model
"""
import os
import logging
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import torch

from services.base_service import BaseAIService
from services.retina_service import R2UNet  # Reuse R2UNet architecture
from models.inference_model import InferenceType, SegmentationResult
from utils.image_utils import create_segmentation_overlay, image_to_base64

logger = logging.getLogger(__name__)

class MSSegmentationService(BaseAIService):
    """MS lesion segmentation service"""
    
    def __init__(self, model_path: str, model_name: str = "ms_segmentation"):
        super().__init__(model_path, model_name, "ms_segmentation")
        self.input_size = (512, 512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ["background", "ms_lesion", "brain_tissue", "csf"]
        
    def load_model(self) -> bool:
        """Load PyTorch MS segmentation model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}, creating dummy model")
                # Create a dummy model for demonstration
                self.model = R2UNet(in_channels=3, out_channels=len(self.classes))
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                return True
            
            # Create model
            num_classes = len(self.classes)
            self.model = R2UNet(in_channels=3, out_channels=num_classes)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except:
                logger.warning("Strict loading failed, trying with strict=False")
                self.model.load_state_dict(state_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded MS segmentation model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MS segmentation model {self.model_name}: {e}")
            return False
    
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run MS lesion segmentation"""
        return_image = kwargs.get('return_image', True)
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        original_size = image.size
        
        # Convert to tensor
        input_tensor = torch.from_numpy(processed_image).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Convert output to segmentation mask
        segmentation_mask = output.cpu().numpy()[0]  # Remove batch dimension
        
        # Convert probabilities to class predictions
        predicted_mask = np.argmax(segmentation_mask, axis=0)
        
        # Count pixels for each class
        pixel_counts = {}
        classes_found = []
        
        for i, class_name in enumerate(self.classes):
            count = int(np.sum(predicted_mask == i))
            pixel_counts[class_name] = count
            if count > 0 and class_name != "background":
                classes_found.append(class_name)
        
        # Calculate lesion volume/area metrics
        ms_lesion_pixels = pixel_counts.get("ms_lesion", 0)
        total_brain_pixels = sum([pixel_counts.get(cls, 0) for cls in ["ms_lesion", "brain_tissue"]])
        lesion_load = (ms_lesion_pixels / total_brain_pixels * 100) if total_brain_pixels > 0 else 0
        
        # Create base64 mask
        mask_normalized = (predicted_mask * 255 / (len(self.classes) - 1)).astype(np.uint8)
        mask_image = Image.fromarray(mask_normalized, mode='L')
        mask_image = mask_image.resize(original_size, Image.NEAREST)
        mask_base64 = image_to_base64(mask_image, "PNG")
        
        segmentation = SegmentationResult(
            mask=mask_base64,
            classes_found=classes_found,
            pixel_counts=pixel_counts
        )
        
        result_data = {"segmentation": segmentation.dict()}
        
        # Add lesion-specific metadata
        result_data["metadata"] = {
            "lesion_load_percentage": round(lesion_load, 2),
            "total_lesion_pixels": ms_lesion_pixels,
            "analysis": "MS lesion segmentation analysis"
        }
        
        # Add overlay image if requested
        if return_image:
            # Create colored overlay for MS lesions
            colors = {
                0: (0, 0, 0),       # Background - black
                1: (255, 0, 0),     # MS lesion - red
                2: (128, 128, 128), # Brain tissue - gray
                3: (0, 0, 255),     # CSF - blue
            }
            
            mask_resized = np.array(mask_image.resize(original_size, Image.NEAREST))
            overlay_image = create_segmentation_overlay(image, mask_resized, colors=colors)
            result_data["annotated_image"] = image_to_base64(overlay_image)
        
        return result_data
    
    def get_inference_type(self) -> InferenceType:
        """Return segmentation type"""
        return InferenceType.SEGMENTATION
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for MS segmentation model"""
        # Convert to RGB if needed (MRI can be grayscale)
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
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
        """Validate input for MS segmentation model"""
        if not super().validate_input_image(image):
            return False
        
        # Should be MRI image (can be grayscale or RGB)
        if image.mode not in ['RGB', 'L', 'RGBA']:
            return False
        
        # Minimum size for MRI images
        min_size = 128
        if image.width < min_size or image.height < min_size:
            return False
            
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get MS segmentation model information"""
        info = super().get_model_info()
        info.update({
            "classes": self.classes,
            "input_size": self.input_size,
            "task": "Multiple Sclerosis lesion segmentation from MRI",
            "device": str(self.device),
            "output_metrics": ["lesion_load_percentage", "lesion_pixel_count"]
        })
        return info