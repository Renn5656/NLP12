import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_ORDER = ["Normal", "Suspicious", "Fraud"]


def clean_text(text):
    """Normalize message text for traditional ML models."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(data_path="data/messages.csv"):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    stratify = df["label"] if df["label"].nunique() > 1 else None
    return train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset.head())
    print("\nLabel distribution:")
    print(dataset["label"].value_counts())
