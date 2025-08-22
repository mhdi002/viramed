"""
Retina segmentation service using R2UNet model
"""
import os
import logging
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

from services.base_service import BaseAIService
from models.inference_model import InferenceType, SegmentationResult
from utils.image_utils import create_segmentation_overlay, image_to_base64

logger = logging.getLogger(__name__)

class R2UNet(nn.Module):
    """R2UNet model implementation for retina segmentation"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(R2UNet, self).__init__()
        self.features = features
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i, feature in enumerate(features):
            input_ch = in_channels if i == 0 else features[i-1]
            self.encoder.append(self._make_layer(input_ch, feature))
        
        # Bottleneck
        self.bottleneck = self._make_layer(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(features)-1, -1, -1):
            input_ch = features[-1] * 2 if i == len(features)-1 else features[i+1]
            skip_ch = features[i]
            self.decoder.append(self._make_layer(input_ch + skip_ch, features[i]))
        
        # Final layer
        self.final = nn.Conv2d(features[0], out_channels, 1)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
            x = nn.functional.max_pool2d(x, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for i, decoder_layer in enumerate(self.decoder):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = decoder_layer(x)
        
        # Final
        x = self.final(x)
        return torch.sigmoid(x)

class RetinaService(BaseAIService):
    """Retina segmentation service"""
    
    def __init__(self, model_path: str, model_name: str = "retina_segmentation"):
        super().__init__(model_path, model_name, "retina")
        self.input_size = (512, 512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ["background", "vessels", "optic_disc", "lesions"]
        
    def load_model(self) -> bool:
        """Load PyTorch retina segmentation model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
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
            logger.info(f"Successfully loaded retina model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load retina model {self.model_name}: {e}")
            return False
    
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run retina segmentation"""
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
        
        # Add overlay image if requested
        if return_image:
            # Resize mask back to original image size
            mask_resized = np.array(mask_image.resize(original_size, Image.NEAREST))
            overlay_image = create_segmentation_overlay(image, mask_resized)
            result_data["annotated_image"] = image_to_base64(overlay_image)
        
        return result_data
    
    def get_inference_type(self) -> InferenceType:
        """Return segmentation type"""
        return InferenceType.SEGMENTATION
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for retina model"""
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
        """Validate input for retina model"""
        if not super().validate_input_image(image):
            return False
        
        # Should be fundus image (typically RGB)
        if image.mode not in ['RGB', 'RGBA']:
            return False
        
        # Minimum size for retinal images
        min_size = 224
        if image.width < min_size or image.height < min_size:
            return False
            
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get retina model information"""
        info = super().get_model_info()
        info.update({
            "classes": self.classes,
            "input_size": self.input_size,
            "task": "Retinal vessel and lesion segmentation",
            "device": str(self.device)
        })
        return info