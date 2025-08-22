"""
User data models and schemas
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
from enum import Enum
import uuid

class UserRole(str, Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER

class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    permissions: List[str]

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class UserLogin(BaseModel):
    username: str
    password: str