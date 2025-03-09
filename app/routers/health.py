# Routes for health checks

from fastapi import APIRouter
import joblib
import os

router = APIRouter()

# Path to the trained model
model_path = "app/models/best_model_RF.pkl"

@router.get("/health/")
def health_check():
    """
    Health check endpoint to verify API status and model availability.
    """
    # Check if the model exists
    model_status = os.path.exists(model_path)
    
    return {
        "status": "API is running",
        "model_loaded": model_status
    }
