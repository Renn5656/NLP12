from pathlib import Path

from src.model_training import train_models
from src.preprocessing import load_dataset


DATA_PATH = Path("data/messages.csv")
MODELS_DIR = Path("models")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    train_models(df["clean_text"], df["label"], MODELS_DIR)
    print(f"Used full dataset for training: {len(df)} messages")
    print(f"Training completed. Models saved to {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
