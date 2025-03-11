# Routes for health checks

from fastapi import APIRouter
import os
from app.models.model_utils import MODEL_PATH

router = APIRouter()


@router.get("/health/")
def health_check():
    """
    Health check endpoint to verify API status and model availability.
    """
    # Check if the model exists
    model_status: bool = os.path.exists(MODEL_PATH)

    return {
        "status": "API is running",
        "model_loaded": model_status,
    }
