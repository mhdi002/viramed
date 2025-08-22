"""
Medical AI Backend System - Main Application
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import after logging setup
from config.settings import settings
from utils.database import initialize_database, create_default_admin, close_database_connection
from controllers import auth_router, model_router, inference_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Medical AI Backend System...")
    
    try:
        # Initialize database
        initialize_database()
        logger.info("Database initialized successfully")
        
        # Create default admin user
        create_default_admin()
        logger.info("Default admin user setup completed")
        
        # Create storage directories
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        logger.info("Storage directories created")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Medical AI Backend System...")
    close_database_connection()
    logger.info("Database connection closed")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Comprehensive Medical AI Backend System with MVC Architecture",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(model_router)
app.include_router(inference_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "api_prefix": settings.API_PREFIX,
        "documentation": f"{settings.API_PREFIX}/docs",
        "features": [
            "Medical AI Model Management",
            "Multi-format Model Support (YOLO, TensorFlow, PyTorch)",
            "Authentication & Authorization",
            "Batch Inference Processing",
            "Medical Domain Specialization",
            "Comprehensive API Documentation"
        ]
    }

# Health check endpoint
@app.get(f"{settings.API_PREFIX}/health")
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "database": "connected"  # Could add actual DB health check
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please contact support.",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        access_log=True
    )