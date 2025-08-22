from .auth_controller import router as auth_router
from .model_controller import router as model_router
from .inference_controller import router as inference_router

__all__ = ["auth_router", "model_router", "inference_router"]