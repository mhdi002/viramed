"""
Medical AI model data schemas
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class ModelType(str, Enum):
    YOLO = "yolo"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"

class TaskType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"

class MedicalDomain(str, Enum):
    MAMMOGRAPHY = "mammography"
    BONE_AGE = "bone_age"
    RETINA = "retina"
    LIVER = "liver"
    COLON = "colon"
    BRAIN = "brain"
    MS_SEGMENTATION = "ms_segmentation"
    GENERAL = "general"

class MedicalModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Model name")
    filename: str = Field(..., description="Model file name")
    model_type: ModelType = Field(..., description="Type of model framework")
    task_type: TaskType = Field(..., description="AI task type")
    medical_domain: MedicalDomain = Field(..., description="Medical domain")
    format: str = Field(..., description="File format (pt, pth, h5, onnx)")
    version: str = Field(default="1.0.0", description="Model version")
    description: Optional[str] = Field(None, description="Model description")
    input_size: Optional[int] = Field(default=640, description="Input image size")
    classes: Optional[List[str]] = Field(default=None, description="Model classes")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="User ID who created")
    is_active: bool = Field(default=True, description="Model status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MedicalModelCreate(BaseModel):
    name: str
    filename: str
    model_type: ModelType
    task_type: TaskType
    medical_domain: MedicalDomain
    format: str
    version: str = "1.0.0"
    description: Optional[str] = None
    input_size: Optional[int] = 640
    classes: Optional[List[str]] = None
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MedicalModelResponse(BaseModel):
    id: str
    name: str
    filename: str
    model_type: ModelType
    task_type: TaskType
    medical_domain: MedicalDomain
    format: str
    version: str
    description: Optional[str]
    input_size: Optional[int]
    classes: Optional[List[str]]
    accuracy: Optional[float]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    is_active: bool
    metadata: Dict[str, Any]