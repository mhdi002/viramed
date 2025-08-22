"""
Authentication and user management endpoints
"""
from datetime import datetime, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pymongo.collection import Collection

from config.settings import settings
from models.user_model import User, UserCreate, UserLogin, UserResponse, Token, UserRole
from utils.auth import (
    get_password_hash, verify_password, create_access_token, 
    get_current_user, require_role
)
from utils.database import get_collection

router = APIRouter(prefix=f"{settings.API_PREFIX}/auth", tags=["Authentication"])
security = HTTPBearer()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_create: UserCreate, current_user: User = Depends(require_role("admin"))):
    """Register a new user (Admin only)"""
    users_collection: Collection = get_collection("users")
    
    # Check if user already exists
    existing_user = users_collection.find_one({"username": user_create.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    existing_email = users_collection.find_one({"email": user_create.email})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = User(**user_create.dict(exclude={"password"}))
    user_data = user.dict()
    user_data["password_hash"] = get_password_hash(user_create.password)
    
    # Set default permissions based on role
    if user.role == UserRole.ADMIN:
        user_data["permissions"] = ["*"]
    elif user.role == UserRole.DOCTOR:
        user_data["permissions"] = ["inference", "view_models", "view_reports"]
    elif user.role == UserRole.RESEARCHER:
        user_data["permissions"] = ["inference", "view_models", "batch_inference", "export_data"]
    else:  # VIEWER
        user_data["permissions"] = ["view_models"]
    
    users_collection.insert_one(user_data)
    
    return UserResponse(**user.dict())

@router.post("/login", response_model=Token)
async def login_user(user_login: UserLogin):
    """User login"""
    users_collection: Collection = get_collection("users")
    
    # Find user
    user_data = users_collection.find_one({"username": user_login.username}, {"_id": 0})
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    user = User(**user_data)
    
    # Verify password
    if not verify_password(user_login.password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is disabled"
        )
    
    # Update last login
    users_collection.update_one(
        {"username": user.username},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(**user.dict())
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(**current_user.dict())

@router.get("/users", response_model=List[UserResponse])
async def list_users(current_user: User = Depends(require_role("admin"))):
    """List all users (Admin only)"""
    users_collection: Collection = get_collection("users")
    
    users_data = list(users_collection.find({}, {"_id": 0, "password_hash": 0}))
    return [UserResponse(**user_data) for user_data in users_data]

@router.put("/users/{username}/role", response_model=UserResponse)
async def update_user_role(
    username: str, 
    new_role: UserRole,
    current_user: User = Depends(require_role("admin"))
):
    """Update user role (Admin only)"""
    users_collection: Collection = get_collection("users")
    
    # Update permissions based on new role
    permissions = []
    if new_role == UserRole.ADMIN:
        permissions = ["*"]
    elif new_role == UserRole.DOCTOR:
        permissions = ["inference", "view_models", "view_reports"]
    elif new_role == UserRole.RESEARCHER:
        permissions = ["inference", "view_models", "batch_inference", "export_data"]
    else:  # VIEWER
        permissions = ["view_models"]
    
    result = users_collection.update_one(
        {"username": username},
        {"$set": {"role": new_role.value, "permissions": permissions}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    updated_user_data = users_collection.find_one({"username": username}, {"_id": 0, "password_hash": 0})
    return UserResponse(**updated_user_data)

@router.put("/users/{username}/status", response_model=UserResponse)
async def update_user_status(
    username: str, 
    is_active: bool,
    current_user: User = Depends(require_role("admin"))
):
    """Update user active status (Admin only)"""
    users_collection: Collection = get_collection("users")
    
    result = users_collection.update_one(
        {"username": username},
        {"$set": {"is_active": is_active}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    updated_user_data = users_collection.find_one({"username": username}, {"_id": 0, "password_hash": 0})
    return UserResponse(**updated_user_data)

@router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: User = Depends(require_role("admin"))
):
    """Delete user (Admin only)"""
    users_collection: Collection = get_collection("users")
    
    # Prevent admin from deleting themselves
    if username == current_user.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    result = users_collection.delete_one({"username": username})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": f"User {username} deleted successfully"}

@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    users_collection: Collection = get_collection("users")
    
    # Get current password hash
    user_data = users_collection.find_one({"username": current_user.username})
    if not verify_password(current_password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    new_password_hash = get_password_hash(new_password)
    users_collection.update_one(
        {"username": current_user.username},
        {"$set": {"password_hash": new_password_hash}}
    )
    
    return {"message": "Password changed successfully"}