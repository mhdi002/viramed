"""
Inference request and response models
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class InferenceType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"

class DetectionBox(BaseModel):
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate") 
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Class ID")
    label: str = Field(..., description="Class label")

class ClassificationResult(BaseModel):
    class_id: int = Field(..., description="Predicted class ID")
    label: str = Field(..., description="Class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    probabilities: Dict[str, float] = Field(..., description="All class probabilities")

class SegmentationResult(BaseModel):
    mask: str = Field(..., description="Base64 encoded segmentation mask")
    classes_found: List[str] = Field(..., description="Classes found in segmentation")
    pixel_counts: Dict[str, int] = Field(..., description="Pixel count per class")

class RegressionResult(BaseModel):
    predicted_value: float = Field(..., description="Predicted numerical value")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval")
    unit: Optional[str] = Field(None, description="Unit of measurement")

class InferenceRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    return_image: bool = Field(default=True, description="Return annotated image")
    image_format: str = Field(default="PNG", description="Output image format")
    
class InferenceResponse(BaseModel):
    success: bool = Field(..., description="Inference success status")
    inference_type: InferenceType = Field(..., description="Type of inference performed")
    model_name: str = Field(..., description="Model used for inference")
    model_domain: str = Field(..., description="Medical domain")
    
    # Results (only one will be populated based on inference_type)
    detections: Optional[List[DetectionBox]] = Field(None, description="Detection results")
    classification: Optional[ClassificationResult] = Field(None, description="Classification result")
    segmentation: Optional[SegmentationResult] = Field(None, description="Segmentation result") 
    regression: Optional[RegressionResult] = Field(None, description="Regression result")
    
    # Optional annotated image
    annotated_image: Optional[str] = Field(None, description="Base64 encoded annotated image")
    
    # Metadata
    processing_time: float = Field(..., description="Processing time in seconds")
    image_info: Dict[str, Any] = Field(..., description="Original image information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class BatchInferenceRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    return_images: bool = Field(default=False, description="Return annotated images for batch")
    
class BatchInferenceResponse(BaseModel):
    success: bool = Field(..., description="Batch inference success status")
    total_images: int = Field(..., description="Total number of images processed")
    successful_inferences: int = Field(..., description="Number of successful inferences")
    failed_inferences: int = Field(..., description="Number of failed inferences")
    results: List[InferenceResponse] = Field(..., description="Individual inference results")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")