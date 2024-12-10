import os
from os.path import dirname, join

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data_path = join(dirname(dirname(__file__)), "datasets", "osv5m")
    train_fp = join(data_path, f"train.csv")
    val_fp = join(data_path, f"val.csv")
    os.makedirs(dirname(val_fp), exist_ok=True)
    df = pd.read_csv(train_fp, dtype={"category": str, "country": str, "city": str})
    df_train, df_val = train_test_split(df, stratify=df["category"], test_size=0.1)
    df_train.to_csv(train_fp, index=False)
    df_val.to_csv(val_fp, index=False)
