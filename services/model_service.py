"""
Model management service
"""
import os
import logging
from typing import Dict, Any, List, Optional, Type
from pymongo.collection import Collection

from config.settings import settings
from models.medical_model import MedicalModel, MedicalModelCreate, MedicalDomain, ModelType, TaskType
from utils.database import get_collection
from utils.model_utils import scan_models_directory, get_medical_model_templates, create_model_metadata
from services.base_service import BaseAIService
from services.yolo_service import YOLOService
from services.mammography_service import MammographyService
from services.bone_age_service import BoneAgeService
from services.retina_service import RetinaService
from services.ms_segmentation_service import MSSegmentationService

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing medical AI models"""
    
    def __init__(self):
        self.collection: Collection = get_collection("medical_models")
        self.loaded_services: Dict[str, BaseAIService] = {}
        
        # Service mapping
        self.service_mapping = {
            ModelType.YOLO: YOLOService,
            ModelType.TENSORFLOW: MammographyService,
            ModelType.PYTORCH: self._get_pytorch_service_class
        }
    
    def _get_pytorch_service_class(self, medical_domain: MedicalDomain) -> Type[BaseAIService]:
        """Get appropriate PyTorch service class based on medical domain"""
        domain_mapping = {
            MedicalDomain.BONE_AGE: BoneAgeService,
            MedicalDomain.RETINA: RetinaService,
            MedicalDomain.MS_SEGMENTATION: MSSegmentationService
        }
        return domain_mapping.get(medical_domain, BoneAgeService)  # Default fallback
    
    def get_all_models(self) -> List[MedicalModel]:
        """Get all registered models"""
        try:
            models_data = list(self.collection.find({}, {"_id": 0}))
            return [MedicalModel(**data) for data in models_data]
        except Exception as e:
            logger.error(f"Error getting all models: {e}")
            return []
    
    def get_model_by_name(self, name: str) -> Optional[MedicalModel]:
        """Get model by name"""
        try:
            model_data = self.collection.find_one({"name": name}, {"_id": 0})
            if model_data:
                return MedicalModel(**model_data)
            return None
        except Exception as e:
            logger.error(f"Error getting model {name}: {e}")
            return None
    
    def get_models_by_domain(self, domain: MedicalDomain) -> List[MedicalModel]:
        """Get models by medical domain"""
        try:
            models_data = list(self.collection.find({"medical_domain": domain.value}, {"_id": 0}))
            return [MedicalModel(**data) for data in models_data]
        except Exception as e:
            logger.error(f"Error getting models for domain {domain}: {e}")
            return []
    
    def create_model(self, model_create: MedicalModelCreate, created_by: str) -> MedicalModel:
        """Create new model entry"""
        try:
            model = MedicalModel(**model_create.dict(), created_by=created_by)
            
            # Check if model with same name exists
            existing = self.collection.find_one({"name": model.name})
            if existing:
                raise ValueError(f"Model with name '{model.name}' already exists")
            
            # Insert into database
            self.collection.insert_one(model.dict())
            
            logger.info(f"Created model: {model.name}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def update_model(self, name: str, updates: Dict[str, Any]) -> Optional[MedicalModel]:
        """Update model"""
        try:
            # Update in database
            result = self.collection.update_one(
                {"name": name},
                {"$set": {**updates, "updated_at": MedicalModel().updated_at}}
            )
            
            if result.modified_count > 0:
                return self.get_model_by_name(name)
            return None
            
        except Exception as e:
            logger.error(f"Error updating model {name}: {e}")
            return None
    
    def delete_model(self, name: str) -> bool:
        """Delete model"""
        try:
            # Remove from loaded services
            if name in self.loaded_services:
                del self.loaded_services[name]
            
            # Delete from database
            result = self.collection.delete_one({"name": name})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted model: {name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting model {name}: {e}")
            return False
    
    def refresh_models_from_storage(self) -> Dict[str, Any]:
        """Scan storage directory and register found models"""
        try:
            found_models = scan_models_directory()
            templates = get_medical_model_templates()
            
            registered_count = 0
            updated_count = 0
            
            for model_info in found_models:
                filename = model_info['filename']
                name = os.path.splitext(filename)[0]
                
                # Try to match with template
                template_name = None
                for template_key, template_info in templates.items():
                    if (template_info.get('expected_format') == model_info.get('format') and
                        name.lower().find(template_info.get('medical_domain', '').lower()) != -1):
                        template_name = template_key
                        break
                
                # Create model metadata
                if template_name:
                    template = templates[template_name]
                    model_create = MedicalModelCreate(
                        name=name,
                        filename=filename,
                        model_type=ModelType(template['model_type']),
                        task_type=TaskType(template['task_type']),
                        medical_domain=MedicalDomain(template['medical_domain']),
                        format=model_info['format'],
                        description=template.get('description'),
                        input_size=template.get('input_size'),
                        classes=template.get('classes'),
                        metadata={
                            "template_used": template_name,
                            "auto_registered": True,
                            **model_info
                        }
                    )
                else:
                    # Generic model registration
                    model_create = MedicalModelCreate(
                        name=name,
                        filename=filename,
                        model_type=ModelType.PYTORCH if model_info['format'] == 'pth' else ModelType.YOLO,
                        task_type=TaskType.DETECTION,
                        medical_domain=MedicalDomain.GENERAL,
                        format=model_info['format'],
                        description=f"Auto-registered model from {filename}",
                        metadata={
                            "auto_registered": True,
                            **model_info
                        }
                    )
                
                # Check if model exists
                existing = self.get_model_by_name(name)
                if existing:
                    # Update existing
                    self.update_model(name, model_create.dict())
                    updated_count += 1
                else:
                    # Create new
                    self.create_model(model_create, "system")
                    registered_count += 1
            
            result = {
                "total_found": len(found_models),
                "registered": registered_count,
                "updated": updated_count,
                "models": [self.get_model_by_name(os.path.splitext(m['filename'])[0]).dict() 
                          for m in found_models]
            }
            
            logger.info(f"Refreshed models: {registered_count} new, {updated_count} updated")
            return result
            
        except Exception as e:
            logger.error(f"Error refreshing models: {e}")
            raise
    
    def get_model_service(self, model_name: str) -> Optional[BaseAIService]:
        """Get or create AI service for model"""
        try:
            # Check if already loaded
            if model_name in self.loaded_services:
                return self.loaded_services[model_name]
            
            # Get model info from database
            model = self.get_model_by_name(model_name)
            if not model:
                logger.error(f"Model not found: {model_name}")
                return None
            
            # Get model file path
            model_path = os.path.join(settings.MODELS_DIR, model.filename)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                # Check in domain-specific directories
                domain_paths = [
                    os.path.join(settings.BASE_DIR, "mamography", model.filename),
                    os.path.join(settings.BASE_DIR, "bone age", model.filename),
                    os.path.join(settings.BASE_DIR, "eye retina", model.filename),
                    os.path.join(settings.BASE_DIR, "ms segmentations", model.filename),
                    os.path.join(settings.BASE_DIR, "liver tumor", model.filename),
                    os.path.join(settings.BASE_DIR, "colon polyp", model.filename),
                    os.path.join(settings.BASE_DIR, "brain model", model.filename)
                ]
                
                for path in domain_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                else:
                    logger.error(f"Model file not found: {model.filename}")
                    return None
            
            # Create appropriate service
            service_class = None
            
            if model.model_type == ModelType.YOLO:
                service_class = YOLOService
            elif model.model_type == ModelType.TENSORFLOW:
                service_class = MammographyService
            elif model.model_type == ModelType.PYTORCH:
                if model.medical_domain == MedicalDomain.BONE_AGE:
                    service_class = BoneAgeService
                elif model.medical_domain == MedicalDomain.RETINA:
                    service_class = RetinaService
                elif model.medical_domain == MedicalDomain.MS_SEGMENTATION:
                    service_class = MSSegmentationService
                else:
                    service_class = BoneAgeService  # Default
            
            if not service_class:
                logger.error(f"No service class found for model type: {model.model_type}")
                return None
            
            # Create and cache service
            service = service_class(model_path, model_name, model.medical_domain.value)
            self.loaded_services[model_name] = service
            
            logger.info(f"Created service for model: {model_name}")
            return service
            
        except Exception as e:
            logger.error(f"Error getting model service {model_name}: {e}")
            return None
    
    def get_model_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available model templates"""
        return get_medical_model_templates()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all loaded services"""
        status = {}
        for name, service in self.loaded_services.items():
            status[name] = {
                "is_loaded": service.is_loaded,
                "model_info": service.get_model_info()
            }
        return status