"""
Image processing utilities
"""
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict, Any
import cv2

from config.settings import settings

def validate_image(file_content: bytes, max_size_mb: int = 10) -> bool:
    """Validate image file"""
    try:
        # Check file size
        if len(file_content) > max_size_mb * 1024 * 1024:
            return False
            
        # Try to open image
        image = Image.open(io.BytesIO(file_content))
        image.verify()
        return True
        
    except Exception:
        return False

def process_image(file_content: bytes, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Process uploaded image"""
    image = Image.open(io.BytesIO(file_content))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if target size specified
    if target_size:
        image = resize_image(image, target_size)
    
    return image

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """Resize image to target size"""
    if maintain_aspect:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
    else:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def annotate_detection_image(image: Image.Image, detections: List[Dict[str, Any]], 
                           colors: Optional[List[Tuple[int, int, int]]] = None) -> Image.Image:
    """Annotate image with detection results"""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Default colors
    if colors is None:
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]
        
        x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        label = detection.get('label', 'Unknown')
        confidence = detection.get('confidence', 0.0)
        
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        
        # Draw label
        text = f"{label} {confidence:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Background for text
        draw.rectangle(
            [(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)],
            fill=color
        )
        draw.text((x1 + 2, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)
    
    return annotated

def create_segmentation_overlay(image: Image.Image, mask: np.ndarray, 
                              alpha: float = 0.5, colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> Image.Image:
    """Create segmentation overlay on image"""
    if colors is None:
        colors = {
            0: (0, 0, 0),       # Background - transparent
            1: (255, 0, 0),     # Class 1 - Red
            2: (0, 255, 0),     # Class 2 - Green
            3: (0, 0, 255),     # Class 3 - Blue
            4: (255, 255, 0),   # Class 4 - Yellow
        }
    
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored_mask[mask == class_id] = color
    
    # Convert to PIL and blend
    mask_image = Image.fromarray(colored_mask)
    return Image.blend(image, mask_image, alpha)

def preprocess_for_tensorflow(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for TensorFlow models"""
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def preprocess_for_pytorch(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for PyTorch models"""
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    
    # Convert HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(3):
        img_array[i] = (img_array[i] / 255.0 - mean[i]) / std[i]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_image_info(image: Image.Image) -> Dict[str, Any]:
    """Get image information"""
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "size_bytes": len(image.tobytes()) if hasattr(image, 'tobytes') else None
    }