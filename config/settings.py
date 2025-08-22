"""
Configuration settings for the Medical AI Backend System
"""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Application
    APP_NAME: str = "Medical AI Backend System"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = False
    
    # Database
    MONGO_URL: str = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = "medical_ai_system"
    
    # Security
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Storage
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    STORAGE_DIR: str = os.path.join(BASE_DIR, "storage")
    MODELS_DIR: str = os.path.join(STORAGE_DIR, "models")
    UPLOADS_DIR: str = os.path.join(STORAGE_DIR, "uploads")
    RESULTS_DIR: str = os.path.join(STORAGE_DIR, "results")
    
    # Model Settings
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
    DEFAULT_IOU_THRESHOLD: float = 0.45
    DEFAULT_IMAGE_SIZE: int = 640
    
    # Supported file formats
    SUPPORTED_IMAGE_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    SUPPORTED_MODEL_FORMATS: List[str] = [".pt", ".pth", ".onnx", ".h5"]
    
    # AI Model Paths
    MAMMOGRAPHY_MODEL_PATH: str = os.path.join(BASE_DIR, "mamography", "breast.h5")
    BONE_AGE_MODEL_PATH: str = os.path.join(BASE_DIR, "bone age", "bone_age_model.pth")
    RETINA_MODEL_PATH: str = os.path.join(BASE_DIR, "eye retina", "retina_model.pth")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    def __post_init__(self):
        # Create directories if they don't exist
        for directory in [self.STORAGE_DIR, self.MODELS_DIR, self.UPLOADS_DIR, self.RESULTS_DIR]:
            os.makedirs(directory, exist_ok=True)

settings = Settings()