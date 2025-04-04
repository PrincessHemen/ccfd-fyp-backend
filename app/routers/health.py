# Routes for health checks

from fastapi import APIRouter
import os

router = APIRouter()

# Define the expected model file path
MODEL_FILENAME = "collab_rf_model.pkl"
MODEL_DIRECTORY = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIRECTORY, MODEL_FILENAME)


@router.get("/health/")
def health_check():
    """
    Health check endpoint to verify API status and model availability.
    """
    # Check if the model file exists in the models directory
    model_status: bool = os.path.exists(MODEL_PATH)

    return {
        "status": "API is running",
        "model_loaded": model_status,
    }
