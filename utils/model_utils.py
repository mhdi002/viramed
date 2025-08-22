"""
Model utilities and validation
"""
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

def validate_model_file(file_path: str) -> bool:
    """Validate if model file exists and has correct format"""
    try:
        if not os.path.exists(file_path):
            return False
            
        ext = Path(file_path).suffix.lower()
        return ext in settings.SUPPORTED_MODEL_FORMATS
        
    except Exception as e:
        logger.error(f"Error validating model file {file_path}: {e}")
        return False

def get_model_info(file_path: str) -> Dict[str, Any]:
    """Extract model information from file"""
    info = {
        "filename": os.path.basename(file_path),
        "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        "format": Path(file_path).suffix.lower().lstrip('.'),
        "exists": os.path.exists(file_path)
    }
    
    return info

def get_yolo_model_classes(model_path: str) -> Optional[List[str]]:
    """Extract class names from YOLO model"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        if hasattr(model, 'names') and model.names:
            return list(model.names.values())
        return None
    except Exception as e:
        logger.error(f"Error extracting YOLO classes from {model_path}: {e}")
        return None

def get_tensorflow_model_info(model_path: str) -> Dict[str, Any]:
    """Extract information from TensorFlow model"""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        
        info = {
            "input_shape": model.input_shape if hasattr(model, 'input_shape') else None,
            "output_shape": model.output_shape if hasattr(model, 'output_shape') else None,
            "layers": len(model.layers) if hasattr(model, 'layers') else 0,
            "parameters": model.count_params() if hasattr(model, 'count_params') else None
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error extracting TensorFlow model info from {model_path}: {e}")
        return {}

def get_pytorch_model_info(model_path: str) -> Dict[str, Any]:
    """Extract information from PyTorch model"""
    try:
        import torch
        
        # Load model state dict
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            "has_state_dict": 'state_dict' in checkpoint if isinstance(checkpoint, dict) else False,
            "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
            "total_params": sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor)) if isinstance(checkpoint, dict) else 0
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error extracting PyTorch model info from {model_path}: {e}")
        return {}

def scan_models_directory() -> List[Dict[str, Any]]:
    """Scan models directory for available models"""
    models = []
    
    if not os.path.exists(settings.MODELS_DIR):
        return models
    
    for filename in os.listdir(settings.MODELS_DIR):
        file_path = os.path.join(settings.MODELS_DIR, filename)
        
        if os.path.isfile(file_path):
            ext = Path(filename).suffix.lower()
            
            if ext in settings.SUPPORTED_MODEL_FORMATS:
                model_info = get_model_info(file_path)
                model_info['full_path'] = file_path
                models.append(model_info)
    
    return models

def get_medical_model_templates() -> Dict[str, Dict[str, Any]]:
    """Get predefined medical model templates"""
    return {
        "mammography_classifier": {
            "name": "Mammography Classifier",
            "description": "Breast cancer classification from mammography images",
            "model_type": "tensorflow",
            "task_type": "classification",
            "medical_domain": "mammography",
            "classes": ["Benign", "Malignant"],
            "input_size": 224,
            "expected_format": "h5"
        },
        "bone_age_estimator": {
            "name": "Bone Age Estimator", 
            "description": "Age estimation from hand X-ray images",
            "model_type": "pytorch",
            "task_type": "regression",
            "medical_domain": "bone_age",
            "input_size": 300,
            "expected_format": "pth"
        },
        "retina_segmentation": {
            "name": "Retina Segmentation",
            "description": "Retinal vessel and lesion segmentation",
            "model_type": "pytorch", 
            "task_type": "segmentation",
            "medical_domain": "retina",
            "input_size": 512,
            "expected_format": "pth"
        },
        "liver_tumor_detector": {
            "name": "Liver Tumor Detector",
            "description": "Liver tumor detection in CT scans",
            "model_type": "yolo",
            "task_type": "detection", 
            "medical_domain": "liver",
            "classes": ["tumor", "lesion"],
            "input_size": 640,
            "expected_format": "pt"
        },
        "colon_polyp_detector": {
            "name": "Colon Polyp Detector",
            "description": "Polyp detection in colonoscopy images",
            "model_type": "yolo",
            "task_type": "detection",
            "medical_domain": "colon", 
            "classes": ["polyp"],
            "input_size": 640,
            "expected_format": "pt"
        },
        "brain_lesion_detector": {
            "name": "Brain Lesion Detector",
            "description": "Brain lesion detection in MRI scans",
            "model_type": "yolo",
            "task_type": "detection",
            "medical_domain": "brain",
            "classes": ["lesion", "tumor"],
            "input_size": 640,
            "expected_format": "pt"
        }
    }

def create_model_metadata(model_path: str, template_name: Optional[str] = None) -> Dict[str, Any]:
    """Create comprehensive model metadata"""
    base_info = get_model_info(model_path)
    
    # Get template info if provided
    templates = get_medical_model_templates()
    template_info = templates.get(template_name, {}) if template_name else {}
    
    # Extract format-specific info
    format_info = {}
    ext = base_info.get('format', '')
    
    if ext == 'pt':
        classes = get_yolo_model_classes(model_path)
        if classes:
            format_info['classes'] = classes
            
    elif ext == 'h5':
        format_info.update(get_tensorflow_model_info(model_path))
        
    elif ext == 'pth':
        format_info.update(get_pytorch_model_info(model_path))
    
    # Combine all information
    metadata = {
        **base_info,
        **template_info,
        **format_info,
        "template_used": template_name
    }
    
    return metadata