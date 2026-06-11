from pathlib import Path
from time import sleep

import streamlit as st

from utils.data_utils import load_csv
from utils.model_utils import train_models
from utils.ui_components import load_premium_css, render_metric_card, render_page_hero, render_sidebar
from utils.visualization import model_comparison_chart


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "StudentsPerformance.csv"
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_csv(DATASET_PATH)


def main() -> None:
    load_premium_css()
    render_sidebar("Train and compare pass/fail classifiers.")
    render_page_hero(
        "Model Training",
        "Train Logistic Regression, Decision Tree, and Random Forest models, then save the best performer automatically.",
    )

    if not DATASET_PATH.exists():
        st.warning("Dataset not found. Add `StudentsPerformance.csv` to `project/data/` first.")
        st.stop()

    df = get_dataset()
    st.markdown("### Training Controls")
    st.info(
        "The app uses score columns only to create the pass/fail label. "
        "Score/mark/grade columns are removed from model inputs, so prediction depends on individual parameters."
    )

    if st.button("Train Models", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        for step, message in enumerate(
            [
                "Inspecting dataset...",
                "Preparing pass/fail target...",
                "Building preprocessing pipeline...",
                "Training candidate models...",
                "Saving best model...",
            ],
            start=1,
        ):
            status.write(message)
            progress.progress(step * 18)
            sleep(0.25)

        with st.spinner("Training models..."):
            output = train_models(df, MODEL_DIR)

        progress.progress(100)
        status.success(f"Best model saved: {output['best_model_name']}")
        st.session_state["training_metadata"] = output["metadata"]

    metadata = st.session_state.get("training_metadata")
    if metadata:
        results_df = metadata["results"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Best Model", metadata["best_model_name"])
        with c2:
            render_metric_card("Accuracy", f"{metadata['best_accuracy']:.2%}")
        with c3:
            render_metric_card("Train Rows", f"{metadata['train_size']:,}")
        with c4:
            render_metric_card("Test Rows", f"{metadata['test_size']:,}")

        excluded = metadata.get("excluded_score_columns", [])
        if excluded:
            st.success("Excluded score columns from training inputs: " + ", ".join(excluded))

        st.markdown("### Model Comparison")
        st.dataframe(results_df, use_container_width=True)
        st.plotly_chart(model_comparison_chart(results_df), use_container_width=True)
    else:
        st.markdown("### Current Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
