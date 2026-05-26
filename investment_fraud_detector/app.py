from pathlib import Path

import pandas as pd
import streamlit as st

from src.model_training import train_models
from src.predictor import DEFAULT_MODEL_NAME, predict_message
from src.preprocessing import load_dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "messages.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def ensure_models_exist():
    if (MODELS_DIR / "vectorizer.pkl").exists() and (MODELS_DIR / f"{DEFAULT_MODEL_NAME}.pkl").exists():
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    train_models(df["clean_text"], df["label"], MODELS_DIR)


def show_chart(filename, caption):
    path = RESULTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} not found. Run `python evaluate.py` to generate charts.")


def show_model_comparison_table():
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        st.info("metrics.csv not found. Run `python evaluate.py` first.")
        return

    metrics_df = pd.read_csv(metrics_path)
    columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Fraud Precision",
        "Fraud Recall",
        "Fraud F1-score",
        "CV F1-score",
        "Training Time (sec)",
        "Prediction Time (ms/msg)",
        "Model Size (KB)",
    ]
    display_df = metrics_df[[column for column in columns if column in metrics_df.columns]].copy()
    numeric_columns = [column for column in display_df.columns if column != "Model"]
    for column in numeric_columns:
        if column == "Prediction Time (ms/msg)":
            display_df[column] = display_df[column].map(lambda value: f"{value:.6f}")
        elif column == "Model Size (KB)":
            display_df[column] = display_df[column].map(lambda value: f"{value:.2f}")
        else:
            display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "CV F1-score uses 5-fold cross validation. Fraud Recall shows how well the model avoids missing fraud messages."
    )


def main():
    st.set_page_config(page_title="Investment Scam Message Classification", layout="wide")
    st.title("Investment Scam Message Classification")

    ensure_models_exist()

    message = st.text_area(
        "Message",
        value="Guaranteed profit with zero risk. Join our VIP crypto group now!",
        height=140,
    )

    if st.button("Analyze", type="primary"):
        result = predict_message(message, MODELS_DIR, DEFAULT_MODEL_NAME)
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Label", result["prediction"])
        col2.metric("Risk Score", result["risk_score"])
        col3.metric("Risk Level", result["risk_level"])

        keywords = result["detected_keywords"]
        st.write("Detected Suspicious Keywords:", ", ".join(keywords) if keywords else "None")

        st.subheader("Risk Explanation")
        st.write(result["risk_explanation"])

        breakdown = result["risk_breakdown"]
        st.subheader("Risk Score Breakdown")
        st.write(f"Base score from predicted label `{result['prediction']}`: {breakdown['base_score']}")
        if breakdown["keyword_bonus"]:
            bonus_df = pd.DataFrame(breakdown["keyword_bonus"])
            st.dataframe(bonus_df, use_container_width=True, hide_index=True)
        else:
            st.write("No additional investment scam risk signal detected.")
        if breakdown["combination_bonus"]:
            st.write(f"Combination risk bonus: {breakdown['combination_bonus']}")
        cap_note = " Score capped at 100." if breakdown["score_cap_applied"] else ""
        st.write(f"Raw score: {breakdown['raw_score']} -> Final score: {breakdown['final_score']}.{cap_note}")

        st.subheader("Compare This Message Across All Models")
        comparison_rows = []
        for model_name in ["naive_bayes", "logistic_regression", "svm", "random_forest"]:
            model_result = predict_message(message, MODELS_DIR, model_name)
            comparison_rows.append(
                {
                    "Model": model_name.replace("_", " ").title(),
                    "Prediction": model_result["prediction"],
                    "Risk Score": model_result["risk_score"],
                    "Risk Level": model_result["risk_level"],
                }
            )
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Model Comparison and Application")
    show_model_comparison_table()

    chart_cols = st.columns(2)
    with chart_cols[0]:
        show_chart("model_comparison.png", "Model Comparison")
    with chart_cols[1]:
        show_chart("confusion_matrix_best_model.png", "Confusion Matrix - Best Model")

    st.divider()
    st.subheader("Data Analysis Charts")
    data_cols = st.columns(2)
    with data_cols[0]:
        show_chart("label_distribution.png", "Label Distribution")
    with data_cols[1]:
        show_chart("fraud_word_frequency.png", "Fraud Word Frequency")


if __name__ == "__main__":
    main()
