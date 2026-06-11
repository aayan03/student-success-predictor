from pathlib import Path

import streamlit as st

from utils.data_utils import (
    get_correlation_data,
    get_dataset_summary,
    get_feature_options,
    get_missing_values,
    load_csv,
)
from utils.visualization import correlation_heatmap, distribution_chart, missing_values_chart


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "StudentsPerformance.csv"


st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📊",
    layout="wide",
)


def load_page_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.24), transparent 32rem),
                radial-gradient(circle at top right, rgba(6, 182, 212, 0.16), transparent 30rem),
                linear-gradient(135deg, #050816 0%, #08111f 50%, #0c1022 100%);
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background: rgba(5, 8, 22, 0.82);
            border-right: 1px solid rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(18px);
        }

        .page-hero {
            padding: 2rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.075);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.26);
            backdrop-filter: blur(18px);
            animation: fadeUp 650ms ease both;
        }

        .page-hero h1 {
            margin: 0;
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: 0;
        }

        .page-hero p {
            margin: 0.75rem 0 0;
            color: #cbd5e1;
            line-height: 1.7;
        }

        .mini-card {
            padding: 1.1rem;
            min-height: 122px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.075);
            backdrop-filter: blur(18px);
            transition: transform 180ms ease, border-color 180ms ease;
            animation: fadeUp 750ms ease both;
        }

        .mini-card:hover {
            transform: translateY(-3px);
            border-color: rgba(6, 182, 212, 0.45);
        }

        .mini-label {
            color: #a7b0c0;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .mini-value {
            margin-top: 0.5rem;
            color: #ffffff;
            font-size: 1.9rem;
            font-weight: 800;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_csv(DATASET_PATH)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Student ML Studio")
        st.caption("Explore the dataset before model training.")
        st.divider()
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Dataset_Explorer.py", label="Dataset Explorer", icon="📊")
        st.page_link("pages/2_Model_Training.py", label="Model Training", icon="🧠")
        st.page_link("pages/3_Student_Prediction.py", label="Student Prediction", icon="✅")
        st.page_link("pages/4_Analytics_Dashboard.py", label="Analytics Dashboard", icon="📈")


def main() -> None:
    load_page_css()
    render_sidebar()

    st.markdown(
        """
        <div class="page-hero">
            <h1>Dataset Explorer</h1>
            <p>
                Inspect the student records, review quality signals, compare
                distributions, and understand numeric relationships before training.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATASET_PATH.exists():
        st.warning(
            "Dataset not found. Place StudentsPerformance.csv in "
            "`project/data/StudentsPerformance.csv` to enable this page."
        )
        st.stop()

    df = get_dataset()
    summary = get_dataset_summary(df)
    missing_df = get_missing_values(df)
    feature_options = get_feature_options(df)

    st.markdown("### Dataset Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Rows", f"{summary['rows']:,}")
    with c2:
        metric_card("Columns", f"{summary['columns']:,}")
    with c3:
        metric_card("Missing Cells", f"{summary['missing_cells']:,}")
    with c4:
        metric_card("Duplicates", f"{summary['duplicate_rows']:,}")
    with c5:
        metric_card("Memory MB", str(summary["memory_mb"]))

    st.markdown("### Dataset Preview")
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=min(len(df), 100), value=min(10, len(df)))
    st.dataframe(df.head(preview_rows), use_container_width=True)

    tab_overview, tab_distributions, tab_quality, tab_correlation = st.tabs(
        ["Statistics", "Distributions", "Missing Values", "Correlation"]
    )

    with tab_overview:
        st.markdown("#### Numeric Statistics")
        if feature_options["numeric"]:
            st.dataframe(df[feature_options["numeric"]].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns were detected.")

        st.markdown("#### Categorical Statistics")
        if feature_options["categorical"]:
            st.dataframe(df[feature_options["categorical"]].describe().T, use_container_width=True)
        else:
            st.info("No categorical columns were detected.")

    with tab_distributions:
        selected_column = st.selectbox("Choose a feature", feature_options["all"])
        st.plotly_chart(distribution_chart(df, selected_column), use_container_width=True)

    with tab_quality:
        st.dataframe(missing_df, use_container_width=True)
        st.plotly_chart(missing_values_chart(missing_df), use_container_width=True)

    with tab_correlation:
        st.plotly_chart(correlation_heatmap(get_correlation_data(df)), use_container_width=True)


if __name__ == "__main__":
    main()
