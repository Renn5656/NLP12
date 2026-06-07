import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


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
    if "source" not in df:
        df["source"] = "unknown"
    if "template_group" not in df:
        df["template_group"] = [f"row_{index}" for index in range(len(df))]
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    if "split" in df.columns and {"train", "test"}.issubset(set(df["split"])):
        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]
        return train["clean_text"], test["clean_text"], train["label"], test["label"]

    if "template_group" in df.columns and df["template_group"].nunique() < len(df):
        n_splits = max(2, round(1 / test_size))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        target_distribution = df["label"].value_counts(normalize=True)
        candidates = []
        for train_indices, test_indices in splitter.split(
            df["clean_text"],
            df["label"],
            groups=df["template_group"],
        ):
            test_distribution = df.iloc[test_indices]["label"].value_counts(normalize=True)
            missing_labels = len(set(target_distribution.index) - set(test_distribution.index))
            distribution_error = sum(
                abs(test_distribution.get(label, 0) - target_distribution[label])
                for label in target_distribution.index
            )
            size_error = abs((len(test_indices) / len(df)) - test_size)
            candidates.append(
                (missing_labels, distribution_error + size_error, train_indices, test_indices)
            )
        _, _, train_indices, test_indices = min(candidates, key=lambda item: (item[0], item[1]))
        return (
            df.iloc[train_indices]["clean_text"],
            df.iloc[test_indices]["clean_text"],
            df.iloc[train_indices]["label"],
            df.iloc[test_indices]["label"],
        )

    stratify = df["label"] if df["label"].nunique() > 1 else None
    return train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def split_train_validation_test(df):
    required = {"train", "validation", "test"}
    if "split" not in df.columns or not required.issubset(set(df["split"])):
        raise ValueError("Dataset must contain fixed train, validation, and test split assignments.")
    return {
        split_name: df[df["split"] == split_name].copy()
        for split_name in ["train", "validation", "test"]
    }


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset.head())
    print("\nLabel distribution:")
    print(dataset["label"].value_counts())
