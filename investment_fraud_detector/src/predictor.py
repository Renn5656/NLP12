from pathlib import Path

import joblib

from src.preprocessing import clean_text
from src.risk_scoring import calculate_risk_breakdown, calculate_risk_score, get_risk_level


DEFAULT_MODEL_NAME = "logistic_regression"


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
    risk_score, detected_keywords = calculate_risk_score(predicted_label, text)
    risk_breakdown = calculate_risk_breakdown(predicted_label, text)
    return {
        "prediction": predicted_label,
        "risk_score": risk_score,
        "risk_level": get_risk_level(risk_score),
        "detected_keywords": detected_keywords,
        "risk_breakdown": risk_breakdown,
    }
