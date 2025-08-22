"""
Database utilities and connection management
"""
from pymongo import MongoClient, ASCENDING
from typing import Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

# Global MongoDB client
_mongo_client: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    """Get MongoDB client singleton"""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(settings.MONGO_URL)
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    return _mongo_client

def get_database():
    """Get the main database"""
    client = get_mongo_client()
    return client[settings.DATABASE_NAME]

def get_collection(collection_name: str):
    """Get a specific collection"""
    db = get_database()
    return db[collection_name]

def initialize_database():
    """Initialize database with required collections and indexes"""
    try:
        db = get_database()
        
        # Users collection
        users_collection = db["users"]
        users_collection.create_index([("username", ASCENDING)], unique=True)
        users_collection.create_index([("email", ASCENDING)], unique=True)
        users_collection.create_index([("id", ASCENDING)], unique=True)
        
        # Medical models collection
        models_collection = db["medical_models"]
        models_collection.create_index([("name", ASCENDING)], unique=True)
        models_collection.create_index([("id", ASCENDING)], unique=True)
        models_collection.create_index([("medical_domain", ASCENDING)])
        models_collection.create_index([("model_type", ASCENDING)])
        
        # Inference history collection
        inference_collection = db["inference_history"]
        inference_collection.create_index([("user_id", ASCENDING)])
        inference_collection.create_index([("model_id", ASCENDING)])
        inference_collection.create_index([("created_at", ASCENDING)])
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def create_default_admin():
    """Create default admin user if none exists"""
    try:
        from utils.auth import get_password_hash
        from models.user_model import User, UserRole
        
        users_collection = get_collection("users")
        
        # Check if admin exists
        admin_exists = users_collection.find_one({"role": "admin"})
        if not admin_exists:
            admin_user = User(
                username="admin",
                email="admin@medical-ai.com",
                full_name="System Administrator",
                role=UserRole.ADMIN,
                permissions=["*"]  # All permissions
            )
            
            # Create admin with default password
            admin_data = admin_user.dict()
            admin_data["password_hash"] = get_password_hash("admin123")
            
            users_collection.insert_one(admin_data)
            logger.info("Default admin user created (username: admin, password: admin123)")
            
    except Exception as e:
        logger.error(f"Failed to create default admin: {e}")

def close_database_connection():
    """Close database connection"""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        logger.info("Database connection closed")