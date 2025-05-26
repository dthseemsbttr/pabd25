import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
from joblib import dump, load
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

TRAIN_SIZE = 0.8

def test_model():
    """Test model with new data"""
    df = pd.read_csv("data/processed/df.csv", index_col=0)
    y = df["price"]
    X = df.drop(columns="price")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_SIZE, random_state=42
    )
    model = load('models/tmp.joblib')
    predict = model.predict(X_test)

    mse = mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predict)
    r2 = r2_score(y_test, predict)

    logging.info(f"RMSE = {rmse:.4f}")
    logging.info(f"MAE = {mae:.4f}")
    logging.info(f"R² = {r2:.4f}")
    
test_model()