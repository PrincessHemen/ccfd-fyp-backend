import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the uploaded CSV for prediction by ensuring the correct columns.
    Modify this to match your model’s training setup.
    """
    # Drop columns that were not used in training
    required_columns = [...]  # List of features your model expects
    df = df[required_columns] if all(col in df.columns for col in required_columns) else None
    
    if df is None:
        raise ValueError("Uploaded file does not contain the expected columns.")
    
    # Apply necessary transformations (e.g., scaling, encoding) if needed
    # df = transform_function(df)  # Uncomment and define if required
    
    return df
