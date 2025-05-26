import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
from joblib import dump
import logging

TRAIN_SIZE = 0.8

def train_model():
    """Train model and save with MODEL_NAME"""

    df = pd.read_csv("data/processed/df.csv", index_col=0)
    y = df["price"]
    X = df.drop(columns="price")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_SIZE, random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    dump(model, f"models/tmp.joblib")
    logging.info(f"Train {model} and save to models/tmp.joblib")

train_model()