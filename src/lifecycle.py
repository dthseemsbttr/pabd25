"""This is full life cycle for ml model"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from collections import defaultdict
from joblib import dump, load
from sklearn.svm import SVR
import datetime
import argparse
import pandas as pd
import numpy as np
import re
import os
from joblib import dump
from lightgbm import LGBMRegressor
from parse_cian import parse
import logging
import warnings

warnings.filterwarnings("ignore")

TRAIN_SIZE = 0.2
MODEL_NAME = "best_model_2025-05-19_20-08.joblib"

logging.basicConfig(
    filename="model_training.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def parse_cian():
    """Parse data to data/raw"""
    parse(n_rooms=1, end_page=10)
    parse(n_rooms=2, end_page=10)
    parse(n_rooms=3, end_page=10)
    parse(n_rooms=4, end_page=10)


def extract_id(url):
    match = re.search(r"/(\d+)/?$", url)
    return int(match.group(1)) if match else 0


def preprocess_data():
    """Filter and remove"""

    folder_path = "data/raw"
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    file_groups = defaultdict(list)
    for file in files:
        prefix = file.split("_")[0]
        file_groups[prefix].append("data/raw/" + file)

    for prefix, file_list in file_groups.items():
        dfs = []
        for file in file_list:
            df = pd.read_csv(file)
            df["url_id"] = df["url"].apply(extract_id)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values("url_id", ascending=False)
        combined_df = combined_df.drop_duplicates(subset="url_id")
        combined_df = combined_df.drop(columns=["url_id"])
        combined_df = combined_df.head(1000)
        output_filename = f"data/raw/{prefix}_combined_sorted.csv"
        combined_df.to_csv(output_filename, index=False)

    df_1room = pd.read_csv("data/raw/1_combined_sorted.csv")
    df_2room = pd.read_csv("data/raw/2_combined_sorted.csv")
    df_3room = pd.read_csv("data/raw/3_combined_sorted.csv")
    df_4room = pd.read_csv("data/raw/4_combined_sorted.csv")

    df = pd.concat([df_1room, df_2room, df_3room, df_4room], ignore_index=True)
    df = df[df["total_meters"] <= 100]
    df = df[df["price"] <= 50000000]
    df = df[df["rooms_count"] != -1]
    columns_to_keep = ["rooms_count", "floor", "floors_count", "total_meters", "price"]
    df = df[columns_to_keep]
    df.to_csv("data/processed/df.csv")


def train_model():
    """Train model and save with MODEL_NAME"""

    df = pd.read_csv("data/processed/df.csv", index_col=0)
    y = df["price"]
    X = df.drop(columns="price")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_SIZE, random_state=42
    )

    model = LGBMRegressor(random_state=42)

    model.fit(X_train, y_train)
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    dump(model, f"models/best_model_{t}.joblib")
    logging.info(f"Train {model} and save to models/best_model_{t}.joblib")
    return f"models/best_model_{t}.joblib"


def test_model(model_path):
    """Test model with new data"""
    df = pd.read_csv("data/processed/df.csv", index_col=0)
    y = df["price"]
    X = df.drop(columns="price")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_SIZE, random_state=42
    )
    model = load(model_path)
    predict = model.predict(X_test)

    mse = mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predict)
    r2 = r2_score(y_test, predict)

    logging.info(f"MSE = {mse:.4f}")
    logging.info(f"RMSE = {rmse:.4f}")
    logging.info(f"MAE = {mae:.4f}")
    logging.info(f"R² = {r2:.4f}")


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test relative size, from 0 to 1",
        default=TRAIN_SIZE,
    )
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    parser.add_argument(
        "-d",
        "--download_new",
        help="True or False",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    if args.download_new:
        parse_cian()
        preprocess_data()
    path = train_model()
    test_model(path)
