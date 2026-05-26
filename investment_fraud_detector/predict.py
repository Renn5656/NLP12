import argparse
from pathlib import Path

from src.model_training import train_models
from src.predictor import DEFAULT_MODEL_NAME, predict_message
from src.preprocessing import load_dataset


MODELS_DIR = Path("models")
DATA_PATH = Path("data/messages.csv")


def ensure_models_exist():
    if (MODELS_DIR / "vectorizer.pkl").exists() and (MODELS_DIR / f"{DEFAULT_MODEL_NAME}.pkl").exists():
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    train_models(df["clean_text"], df["label"], MODELS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Predict investment scam risk for one message.")
    parser.add_argument(
        "text",
        nargs="?",
        default="Guaranteed profit with zero risk. Join our VIP crypto group now!",
        help="Message text to classify.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=["naive_bayes", "logistic_regression", "svm", "random_forest"],
        help="Model file to use for prediction.",
    )
    args = parser.parse_args()

    ensure_models_exist()
    result = predict_message(args.text, MODELS_DIR, args.model)

    print(f"Prediction: {result['prediction']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']}")
    keywords = ", ".join(result["detected_keywords"]) if result["detected_keywords"] else "None"
    print(f"Detected Suspicious Keywords: {keywords}")


if __name__ == "__main__":
    main()
