from pathlib import Path
import pickle
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from sklearn.svm import LinearSVC


MODEL_CONFIGS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "svm": LinearSVC(class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
}


def create_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
    )


def train_models(x_train, y_train, models_dir="models", save=True, return_stats=False):
    models_dir = Path(models_dir)
    if save:
        models_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = create_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    if save:
        joblib.dump(vectorizer, models_dir / "vectorizer.pkl")

    trained_models = {}
    training_stats = {}
    for model_name, model_config in MODEL_CONFIGS.items():
        model = clone(model_config)
        start_time = time.perf_counter()
        model.fit(x_train_tfidf, y_train)
        training_time = time.perf_counter() - start_time
        if save:
            joblib.dump(model, models_dir / f"{model_name}.pkl")
        trained_models[model_name] = model
        training_stats[model_name] = {
            "Training Time (sec)": training_time,
            "Model Size (KB)": len(pickle.dumps(model)) / 1024,
        }

    if return_stats:
        return vectorizer, trained_models, training_stats
    return vectorizer, trained_models


def load_models(models_dir="models"):
    models_dir = Path(models_dir)
    vectorizer = joblib.load(models_dir / "vectorizer.pkl")
    models = {}
    for model_name in MODEL_CONFIGS:
        models[model_name] = joblib.load(models_dir / f"{model_name}.pkl")
    return vectorizer, models
