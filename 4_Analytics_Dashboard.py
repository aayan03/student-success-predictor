from pathlib import Path

import streamlit as st

from utils.data_utils import create_pass_fail_target, load_csv
from utils.model_utils import load_saved_model
from utils.ui_components import load_premium_css, render_metric_card, render_page_hero, render_sidebar
from utils.visualization import (
    confusion_matrix_chart,
    distribution_chart,
    feature_importance_chart,
    target_pie_chart,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "StudentsPerformance.csv"
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(page_title="Analytics Dashboard", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def get_dataset():
    return create_pass_fail_target(load_csv(DATASET_PATH))


def main() -> None:
    load_premium_css()
    render_sidebar("Review outcomes, model diagnostics, and important drivers.")
    render_page_hero(
        "Analytics Dashboard",
        "Monitor pass/fail distribution, model performance diagnostics, feature importance, and key student attributes.",
    )

    if not DATASET_PATH.exists():
        st.warning("Dataset not found. Add `StudentsPerformance.csv` to `project/data/` first.")
        st.stop()

    df = get_dataset()
    model, metadata = load_saved_model(MODEL_DIR)

    pass_rate = (df["pass_fail"].eq("pass").mean() * 100) if "pass_fail" in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Students", f"{len(df):,}")
    with c2:
        render_metric_card("Pass Rate", f"{pass_rate:.1f}%")
    with c3:
        render_metric_card("Best Model", metadata["best_model_name"] if metadata else "Not trained")
    with c4:
        render_metric_card("Accuracy", f"{metadata['best_accuracy']:.2%}" if metadata else "Not trained")

    st.markdown("### Outcome Analytics")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(target_pie_chart(df, "pass_fail"), use_container_width=True)
    with right:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("Feature distribution", numeric_columns)
            st.plotly_chart(distribution_chart(df, selected_column), use_container_width=True)
        else:
            st.info("No numeric columns are available for distribution charts.")

    st.markdown("### Model Diagnostics")
    if not metadata:
        st.warning("Train a model first to unlock feature importance and confusion matrix analytics.")
        st.stop()

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            confusion_matrix_chart(metadata["confusion_matrix"], metadata["labels"]),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            feature_importance_chart(metadata["feature_importance"]),
            use_container_width=True,
        )

    st.markdown("### Model Comparison Table")
    st.dataframe(metadata["results"], use_container_width=True)


if __name__ == "__main__":
    main()
