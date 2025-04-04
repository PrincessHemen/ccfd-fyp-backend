import pandas as pd
from datetime import datetime
import joblib
import os

#To start server || uvicorn app.main:app --reload

# Paths to saved preprocessing files
base_path = os.path.dirname(__file__)
scaler_path = os.path.join(base_path, "scaler.pkl")
selected_features_path = os.path.join(base_path, "selected_features.pkl")

# Load pre-trained preprocessing objects
scaler = joblib.load(scaler_path)
selected_features = joblib.load(selected_features_path)


def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a user-uploaded CSV file to match the format required for model prediction.

    Args:
        df (pd.DataFrame): Raw DataFrame from uploaded CSV.

    Returns:
        pd.DataFrame: Processed DataFrame ready for model prediction.
    """
    # Convert datetime columns
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"], dayfirst=True)

    # Extract age and transaction hour
    df["age"] = df["dob"].apply(lambda x: datetime.now().year - x.year)
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour

    # One-hot encode categorical variables
    categorical_columns = ["category", "gender", "state", "job"]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Drop unnecessary columns
    columns_to_drop = ["cc_num", "first", "last", "street", "city", "merchant",
                       "trans_date_trans_time", "dob", "trans_num"]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Ensure "is_fraud" column is not included
    if "is_fraud" in df.columns:
        df.drop(columns=["is_fraud"], inplace=True)

    # Ensure all required features are present
    missing_features = [
        feat for feat in selected_features if feat not in df.columns]

    # Add missing features with default value 0
    for feat in missing_features:
        df[feat] = 0

    # Select only the required features (in case extra columns are present)
    df_selected = df[selected_features]

    # Apply pre-trained scaler
    df_scaled = pd.DataFrame(scaler.transform(
        df_selected), columns=selected_features)

    return df_scaled
