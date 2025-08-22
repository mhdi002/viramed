"""
Medical model management endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pymongo.collection import Collection
import os

from config.settings import settings
from models.user_model import User
from models.medical_model import (
    MedicalModel, MedicalModelCreate, MedicalModelResponse, 
    ModelType, TaskType, MedicalDomain
)
from utils.auth import get_current_user, require_permission
from services.model_service import ModelService

router = APIRouter(prefix=f"{settings.API_PREFIX}/models", tags=["Medical Models"])

# Initialize model service
model_service = ModelService()

@router.get("/", response_model=List[MedicalModelResponse])
async def list_models(
    domain: Optional[MedicalDomain] = None,
    model_type: Optional[ModelType] = None,
    current_user: User = Depends(require_permission("view_models"))
):
    """List all medical models with optional filtering"""
    try:
        if domain:
            models = model_service.get_models_by_domain(domain)
        else:
            models = model_service.get_all_models()
        
        # Filter by model type if specified
        if model_type:
            models = [model for model in models if model.model_type == model_type]
        
        return [MedicalModelResponse(**model.dict()) for model in models]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving models: {str(e)}"
        )

@router.get("/{model_name}", response_model=MedicalModelResponse)
async def get_model(
    model_name: str,
    current_user: User = Depends(require_permission("view_models"))
):
    """Get specific model by name"""
    model = model_service.get_model_by_name(model_name)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    return MedicalModelResponse(**model.dict())

@router.post("/", response_model=MedicalModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_create: MedicalModelCreate,
    current_user: User = Depends(require_permission("manage_models"))
):
    """Create new medical model entry"""
    try:
        model = model_service.create_model(model_create, current_user.id)
        return MedicalModelResponse(**model.dict())
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating model: {str(e)}"
        )

@router.post("/upload", response_model=MedicalModelResponse)
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    model_type: ModelType = Form(...),
    task_type: TaskType = Form(...),
    medical_domain: MedicalDomain = Form(...),
    description: Optional[str] = Form(None),
    input_size: Optional[int] = Form(640),
    classes: Optional[str] = Form(None),  # JSON string of class list
    current_user: User = Depends(require_permission("manage_models"))
):
    """Upload and register a new model file"""
    try:
        # Validate file format
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.SUPPORTED_MODEL_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported formats: {settings.SUPPORTED_MODEL_FORMATS}"
            )
        
        # Save file to models directory
        file_path = os.path.join(settings.MODELS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse classes if provided
        class_list = None
        if classes:
            try:
                import json
                class_list = json.loads(classes)
            except json.JSONDecodeError:
                class_list = [c.strip() for c in classes.split(",")]
        
        # Create model entry
        model_create = MedicalModelCreate(
            name=name,
            filename=file.filename,
            model_type=model_type,
            task_type=task_type,
            medical_domain=medical_domain,
            format=file_ext.lstrip('.'),
            description=description,
            input_size=input_size,
            classes=class_list
        )
        
        model = model_service.create_model(model_create, current_user.id)
        return MedicalModelResponse(**model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if model creation failed
        file_path = os.path.join(settings.MODELS_DIR, file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading model: {str(e)}"
        )

@router.post("/refresh")
async def refresh_models(
    current_user: User = Depends(require_permission("manage_models"))
):
    """Refresh models from storage directory"""
    try:
        result = model_service.refresh_models_from_storage()
        return {
            "message": "Models refreshed successfully",
            "total_found": result["total_found"],
            "registered": result["registered"],
            "updated": result["updated"],
            "models": result["models"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error refreshing models: {str(e)}"
        )

@router.put("/{model_name}", response_model=MedicalModelResponse)
async def update_model(
    model_name: str,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
    classes: Optional[List[str]] = None,
    accuracy: Optional[float] = None,
    current_user: User = Depends(require_permission("manage_models"))
):
    """Update model information"""
    try:
        updates = {}
        if description is not None:
            updates["description"] = description
        if is_active is not None:
            updates["is_active"] = is_active
        if classes is not None:
            updates["classes"] = classes
        if accuracy is not None:
            updates["accuracy"] = accuracy
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )
        
        updated_model = model_service.update_model(model_name, updates)
        if not updated_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        return MedicalModelResponse(**updated_model.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating model: {str(e)}"
        )

@router.delete("/{model_name}")
async def delete_model(
    model_name: str,
    delete_file: bool = False,
    current_user: User = Depends(require_permission("manage_models"))
):
    """Delete model"""
    try:
        # Get model info before deletion
        model = model_service.get_model_by_name(model_name)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        # Delete from database
        success = model_service.delete_model(model_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete model from database"
            )
        
        # Optionally delete file
        if delete_file:
            file_path = os.path.join(settings.MODELS_DIR, model.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return {"message": f"Model '{model_name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}"
        )

@router.get("/templates/list")
async def list_model_templates(
    current_user: User = Depends(require_permission("view_models"))
):
    """Get available model templates"""
    templates = model_service.get_model_templates()
    return {"templates": templates}

@router.get("/domains/list")
async def list_medical_domains(
    current_user: User = Depends(require_permission("view_models"))
):
    """Get available medical domains"""
    domains = [{"name": domain.name, "value": domain.value} for domain in MedicalDomain]
    return {"domains": domains}

@router.get("/types/list")
async def list_model_types(
    current_user: User = Depends(require_permission("view_models"))
):
    """Get available model types"""
    model_types = [{"name": mtype.name, "value": mtype.value} for mtype in ModelType]
    task_types = [{"name": ttype.name, "value": ttype.value} for ttype in TaskType]
    
    return {
        "model_types": model_types,
        "task_types": task_types
    }

@router.get("/status/services")
async def get_service_status(
    current_user: User = Depends(require_permission("view_models"))
):
    """Get status of loaded AI services"""
    status = model_service.get_service_status()
    return {"services": status}