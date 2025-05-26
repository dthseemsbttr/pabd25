import os
from collections import defaultdict
import pandas as pd
import re

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

    df = pd.DataFrame()
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
        combined_df = combined_df.head(1500)

        df = pd.concat([df, combined_df], ignore_index=True)
    
    df = df[df["total_meters"] <= 90]
    df = df[df["price"] <= 55000000]
    df = df[df["rooms_count"] != -1]
    columns_to_keep = ["rooms_count", "floor", "floors_count", "total_meters", "price"]
    df = df[columns_to_keep]
    df.to_csv("data/processed/df.csv")
    
preprocess_data()