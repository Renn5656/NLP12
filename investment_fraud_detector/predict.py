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
    print(f"Model Confidence: {result['confidence']:.2%}")
    print(f"Needs Review: {'Yes' if result['needs_review'] else 'No'}")
    print("Class Scores:")
    for label, score in sorted(result["class_scores"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {label}: {score:.2%}")


if __name__ == "__main__":
    main()
