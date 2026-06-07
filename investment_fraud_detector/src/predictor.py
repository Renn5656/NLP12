from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import clean_text


DEFAULT_MODEL_NAME = "svm"


def load_predictor(models_dir="models", model_name=DEFAULT_MODEL_NAME):
    models_dir = Path(models_dir)
    vectorizer = joblib.load(models_dir / "vectorizer.pkl")
    model = joblib.load(models_dir / f"{model_name}.pkl")
    return vectorizer, model


def predict_message(text, models_dir="models", model_name=DEFAULT_MODEL_NAME):
    vectorizer, model = load_predictor(models_dir=models_dir, model_name=model_name)
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    predicted_label = model.predict(features)[0]
    probabilities = get_class_scores(model, features)
    confidence = probabilities[predicted_label]
    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "class_scores": probabilities,
        "needs_review": confidence < 0.60,
    }


def get_class_scores(model, features):
    """Return model-derived class scores without keyword or rule-based bonuses."""
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(features)[0]
    else:
        decision = np.asarray(model.decision_function(features))[0]
        decision = decision - decision.max()
        values = np.exp(decision) / np.exp(decision).sum()
    return {
        label: float(score)
        for label, score in zip(model.classes_, values)
    }
