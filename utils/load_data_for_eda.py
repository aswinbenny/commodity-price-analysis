from utils.wrangle import wrangle
import pandas as pd

def load_data(file_path, ):
    """
    Load and preprocess data for EDA.
    
    Parameters:
    file_path (str): Path to the CSV file containing commodity prices.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame ready for EDA.
    """
    df = pd.read_csv(file_path)
    df = wrangle(df)
    
    return df