"""
Medical AI inference endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pymongo.collection import Collection
import time
from datetime import datetime

from config.settings import settings
from models.user_model import User
from models.inference_model import (
    InferenceRequest, InferenceResponse, BatchInferenceRequest, BatchInferenceResponse
)
from utils.auth import get_current_user, require_permission
from utils.image_utils import validate_image, process_image
from utils.database import get_collection
from services.model_service import ModelService

router = APIRouter(prefix=f"{settings.API_PREFIX}/inference", tags=["AI Inference"])

# Initialize model service
model_service = ModelService()

@router.post("/single", response_model=InferenceResponse)
async def run_single_inference(
    image: UploadFile = File(..., description="Medical image for inference"),
    model_name: str = Form(..., description="Name of the model to use"),
    confidence_threshold: float = Form(default=0.25, ge=0.0, le=1.0),
    iou_threshold: float = Form(default=0.45, ge=0.0, le=1.0),
    return_image: bool = Form(default=True, description="Return annotated image"),
    current_user: User = Depends(require_permission("inference"))
):
    """Run inference on a single medical image"""
    start_time = time.time()
    
    try:
        # Validate image
        image_content = await image.read()
        if not validate_image(image_content):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file or file too large"
            )
        
        # Process image
        pil_image = process_image(image_content)
        
        # Get AI service
        ai_service = model_service.get_model_service(model_name)
        if not ai_service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found or failed to load"
            )
        
        # Validate image for this model
        if not ai_service.validate_input_image(pil_image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image is not suitable for this model"
            )
        
        # Run inference
        result = ai_service.run_inference(
            pil_image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            return_image=return_image
        )
        
        # Log inference to database
        await log_inference(
            user_id=current_user.id,
            model_name=model_name,
            processing_time=time.time() - start_time,
            success=result.success,
            image_info=result.image_info
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        # Log failed inference
        await log_inference(
            user_id=current_user.id,
            model_name=model_name,
            processing_time=time.time() - start_time,
            success=False,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@router.post("/batch", response_model=BatchInferenceResponse)
async def run_batch_inference(
    images: List[UploadFile] = File(..., description="Multiple medical images"),
    model_name: str = Form(..., description="Name of the model to use"),
    confidence_threshold: float = Form(default=0.25, ge=0.0, le=1.0),
    iou_threshold: float = Form(default=0.45, ge=0.0, le=1.0),
    return_images: bool = Form(default=False, description="Return annotated images"),
    current_user: User = Depends(require_permission("batch_inference"))
):
    """Run inference on multiple medical images"""
    start_time = time.time()
    
    if len(images) > 20:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size too large. Maximum 20 images allowed."
        )
    
    try:
        # Get AI service
        ai_service = model_service.get_model_service(model_name)
        if not ai_service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found or failed to load"
            )
        
        results = []
        successful_count = 0
        failed_count = 0
        errors = []
        
        for i, image in enumerate(images):
            try:
                # Validate and process image
                image_content = await image.read()
                if not validate_image(image_content):
                    errors.append(f"Image {i+1} ({image.filename}): Invalid image file")
                    failed_count += 1
                    continue
                
                pil_image = process_image(image_content)
                
                # Validate for model
                if not ai_service.validate_input_image(pil_image):
                    errors.append(f"Image {i+1} ({image.filename}): Not suitable for model")
                    failed_count += 1
                    continue
                
                # Run inference
                result = ai_service.run_inference(
                    pil_image,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                    return_image=return_images
                )
                
                results.append(result)
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Image {i+1} ({image.filename}): {result.metadata.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_count += 1
                errors.append(f"Image {i+1} ({image.filename}): {str(e)}")
        
        total_processing_time = time.time() - start_time
        
        # Log batch inference
        await log_inference(
            user_id=current_user.id,
            model_name=model_name,
            processing_time=total_processing_time,
            success=successful_count > 0,
            metadata={
                "batch_size": len(images),
                "successful": successful_count,
                "failed": failed_count
            }
        )
        
        return BatchInferenceResponse(
            success=successful_count > 0,
            total_images=len(images),
            successful_inferences=successful_count,
            failed_inferences=failed_count,
            results=results,
            total_processing_time=total_processing_time,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )

@router.get("/history")
async def get_inference_history(
    limit: int = 50,
    offset: int = 0,
    model_name: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get user's inference history"""
    try:
        inference_collection: Collection = get_collection("inference_history")
        
        # Build query
        query = {"user_id": current_user.id}
        if model_name:
            query["model_name"] = model_name
        
        # Get total count
        total_count = inference_collection.count_documents(query)
        
        # Get records
        records = list(
            inference_collection
            .find(query, {"_id": 0})
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        
        return {
            "total_count": total_count,
            "records": records,
            "has_more": offset + len(records) < total_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving inference history: {str(e)}"
        )

@router.get("/statistics")
async def get_inference_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get user's inference statistics"""
    try:
        inference_collection: Collection = get_collection("inference_history")
        
        # User-specific stats
        user_query = {"user_id": current_user.id}
        
        total_inferences = inference_collection.count_documents(user_query)
        successful_inferences = inference_collection.count_documents({**user_query, "success": True})
        failed_inferences = total_inferences - successful_inferences
        
        # Model usage stats
        model_usage = list(
            inference_collection.aggregate([
                {"$match": user_query},
                {"$group": {"_id": "$model_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ])
        )
        
        # Recent activity (last 30 days)
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        recent_activity = inference_collection.count_documents({
            **user_query,
            "created_at": {"$gte": thirty_days_ago}
        })
        
        # Global stats (if admin)
        global_stats = {}
        if current_user.role == "admin":
            global_stats = {
                "total_system_inferences": inference_collection.count_documents({}),
                "active_models": len(model_service.loaded_services),
                "total_users": get_collection("users").count_documents({"is_active": True})
            }
        
        return {
            "user_statistics": {
                "total_inferences": total_inferences,
                "successful_inferences": successful_inferences,
                "failed_inferences": failed_inferences,
                "success_rate": (successful_inferences / total_inferences * 100) if total_inferences > 0 else 0,
                "recent_activity_30d": recent_activity
            },
            "model_usage": model_usage,
            "global_statistics": global_stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving statistics: {str(e)}"
        )

@router.get("/models/available")
async def get_available_models(
    current_user: User = Depends(require_permission("view_models"))
):
    """Get list of available models for inference"""
    try:
        models = model_service.get_all_models()
        active_models = [model for model in models if model.is_active]
        
        model_info = []
        for model in active_models:
            service = model_service.get_model_service(model.name)
            model_data = {
                "name": model.name,
                "medical_domain": model.medical_domain,
                "task_type": model.task_type,
                "model_type": model.model_type,
                "description": model.description,
                "classes": model.classes,
                "is_loaded": service.is_loaded if service else False
            }
            model_info.append(model_data)
        
        return {"available_models": model_info}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving available models: {str(e)}"
        )

async def log_inference(
    user_id: str,
    model_name: str,
    processing_time: float,
    success: bool,
    image_info: dict = None,
    metadata: dict = None,
    error: str = None
):
    """Log inference to database"""
    try:
        inference_collection: Collection = get_collection("inference_history")
        
        log_entry = {
            "user_id": user_id,
            "model_name": model_name,
            "processing_time": processing_time,
            "success": success,
            "created_at": datetime.utcnow(),
            "image_info": image_info or {},
            "metadata": metadata or {}
        }
        
        if error:
            log_entry["error"] = error
        
        inference_collection.insert_one(log_entry)
        
    except Exception as e:
        # Log error but don't fail the main operation
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to log inference: {e}")