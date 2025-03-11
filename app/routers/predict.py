import pandas as pd
import joblib
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.model_utils import preprocess_input_data

router = APIRouter()

# Load the trained model
model_path = "app/models/best_model_RF.pkl"
model = joblib.load(model_path)


@router.post("/predict/")
async def predict_fraud(file: UploadFile = File(...)):
    """
    Upload a CSV file with transactions, and get fraud predictions.
    """
    try:
        # Read uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Preprocess the data
        df_processed = preprocess_input_data(df)

        # Predict fraud cases
        predictions = model.predict(df_processed)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
