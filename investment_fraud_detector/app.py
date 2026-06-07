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
        "Table values are final test results. The default model is selected using validation Macro F1."
    )


def compare_models(message):
    rows = []
    for model_name in ["naive_bayes", "logistic_regression", "svm", "random_forest"]:
        result = predict_message(message, MODELS_DIR, model_name)
        rows.append(
            {
                "Model": model_name.replace("_", " ").title(),
                "Prediction": result["prediction"],
                "Model Score": result["confidence"],
                "Needs Review": result["needs_review"],
            }
        )
    return pd.DataFrame(rows)


def analyze_batch(uploaded_file):
    batch = pd.read_csv(uploaded_file)
    if "text" not in batch.columns:
        st.error("Uploaded CSV must contain a `text` column.")
        return

    results = [predict_message(text, MODELS_DIR, DEFAULT_MODEL_NAME) for text in batch["text"].fillna("")]
    output = batch.copy()
    output["prediction"] = [result["prediction"] for result in results]
    output["model_confidence"] = [result["confidence"] for result in results]
    output["needs_review"] = [result["needs_review"] for result in results]
    st.dataframe(output, use_container_width=True, hide_index=True)
    st.download_button(
        "Download screening results",
        output.to_csv(index=False),
        file_name="investment_message_screening.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Investment Scam Message Classification", layout="wide")
    st.title("Investment Scam Message Classification")

    ensure_models_exist()

    single_tab, batch_tab = st.tabs(["Single Message", "Batch Screening"])

    with single_tab:
        message = st.text_area(
            "Message",
            value="Guaranteed monthly returns with no possibility of loss. Transfer USDT to activate the account.",
            height=140,
        )

        if st.button("Analyze", type="primary"):
            result = predict_message(message, MODELS_DIR, DEFAULT_MODEL_NAME)
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Label", result["prediction"])
            col2.metric("Model Confidence", f"{result['confidence']:.1%}")
            col3.metric("Review Status", "Review" if result["needs_review"] else "Confident")
            st.caption("SVM class scores support comparison and review routing; they are not calibrated probabilities.")

            st.subheader("Compare This Message Across All Models")
            comparison = compare_models(message)
            comparison["Model Score"] = comparison["Model Score"].map(lambda value: f"{value:.2%}")
            st.dataframe(comparison, use_container_width=True, hide_index=True)

    with batch_tab:
        uploaded_file = st.file_uploader("Upload messages CSV", type="csv")
        if uploaded_file is not None:
            analyze_batch(uploaded_file)

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
    show_chart("split_distribution.png", "Fixed 70/20/10 Dataset Split")
    show_chart("text_length_distribution.png", "Text Length Distribution by Label")

    st.divider()
    st.subheader("Bias and Explainability Analysis")
    analysis_cols = st.columns(2)
    with analysis_cols[0]:
        show_chart("baseline_comparison.png", "Metadata Baseline vs Text Models")
    with analysis_cols[1]:
        show_chart("model_top_features.png", "Logistic Regression Top Features")
    show_chart("data_cleaning_comparison.png", "Dataset Before and After Cleaning")

    errors_path = RESULTS_DIR / "error_analysis.csv"
    if errors_path.exists():
        error_df = pd.read_csv(errors_path)
        st.subheader("High-Confidence Misclassification Examples")
        st.dataframe(error_df.head(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
