from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_utils import create_pass_fail_target, load_csv
from utils import model_utils
from utils.ui_components import load_premium_css, render_page_hero, render_sidebar


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "StudentsPerformance.csv"
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(page_title="Student Prediction", page_icon="✅", layout="wide")


@st.cache_data(show_spinner=False)
def get_dataset():
    return create_pass_fail_target(load_csv(DATASET_PATH))


def build_input_form(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    student_data = {}
    left, right = st.columns(2)

    for index, column in enumerate(feature_columns):
        container = left if index % 2 == 0 else right
        with container:
            if pd.api.types.is_numeric_dtype(df[column]):
                series = df[column].dropna()
                min_value = float(series.min()) if not series.empty else 0.0
                max_value = float(series.max()) if not series.empty else 100.0
                mean_value = float(series.mean()) if not series.empty else 50.0
                student_data[column] = st.number_input(
                    column,
                    min_value=min_value,
                    max_value=max_value,
                    value=mean_value,
                )
            else:
                options = sorted(df[column].dropna().astype(str).unique().tolist())
                student_data[column] = st.selectbox(column, options or ["Unknown"])

    return student_data


def main() -> None:
    load_premium_css()
    render_sidebar("Predict pass/fail outcomes for a student.")
    render_page_hero(
        "Student Prediction",
        "Enter student information and use the saved best model to predict PASS or FAIL with confidence.",
    )

    if not DATASET_PATH.exists():
        st.warning("Dataset not found. Add `StudentsPerformance.csv` to `project/data/` first.")
        st.stop()

    default_model, default_metadata = model_utils.load_saved_model(MODEL_DIR)
    if default_model is None or default_metadata is None:
        st.warning("No trained model found. Please train models from the Model Training page first.")
        st.stop()

    include_marks = st.checkbox(
        "Include marks/scores for this prediction",
        value=False,
        help="Turn this on if you want the prediction to use score columns. Leave it off to predict from profile parameters only.",
    )
    if hasattr(model_utils, "load_model_variant"):
        model, metadata = model_utils.load_model_variant(MODEL_DIR, include_marks)
    else:
        model, metadata = model_utils.load_saved_model(MODEL_DIR)
    if model is None or metadata is None:
        st.warning("Selected model variant was not found. Please retrain from the Model Training page.")
        st.stop()

    df = get_dataset()
    feature_columns = [column for column in metadata["feature_columns"] if column in df.columns]

    st.markdown("### Student Information")
    excluded = metadata.get("excluded_score_columns", [])
    if include_marks:
        st.success("Marks/scores are enabled for this prediction.")
    elif excluded:
        st.info("This prediction form does not ask for marks. Excluded score columns: " + ", ".join(excluded))

    with st.form("prediction_form"):
        student_data = build_input_form(df, feature_columns)
        submitted = st.form_submit_button("Predict Result", type="primary", use_container_width=True)

    if submitted:
        result = model_utils.predict_student(model, student_data)
        label = result["prediction"].upper()
        confidence = result["confidence"]
        confidence_text = "Not available" if confidence is None else f"{confidence:.2%}"
        card_class = "pass" if result["prediction"] == "pass" else "fail"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div style="font-size:0.9rem;color:#cbd5e1;font-weight:800;letter-spacing:0.08em;">PREDICTED OUTCOME</div>
                <div style="font-size:4rem;color:#ffffff;font-weight:900;margin-top:0.4rem;">{label}</div>
                <div style="font-size:1.2rem;color:#e5e7eb;margin-top:0.4rem;">Confidence: {confidence_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
