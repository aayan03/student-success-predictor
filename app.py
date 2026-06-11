from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET = DATA_DIR / "StudentsPerformance.csv"


st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.30), transparent 34rem),
            radial-gradient(circle at top right, rgba(6, 182, 212, 0.18), transparent 30rem),
            linear-gradient(135deg, #050816 0%, #08111f 48%, #0c1022 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: rgba(5, 8, 22, 0.84);
        border-right: 1px solid rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(18px);
    }

    .hero {
        padding: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(124, 58, 237, 0.26), rgba(6, 182, 212, 0.12)),
            rgba(255, 255, 255, 0.07);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(22px);
        animation: fadeUp 700ms ease both;
    }

    .hero h1 {
        max-width: 900px;
        margin: 0;
        color: #ffffff;
        font-size: clamp(2.2rem, 5vw, 4.5rem);
        line-height: 1.05;
        letter-spacing: 0;
        font-weight: 800;
    }

    .hero p {
        max-width: 780px;
        margin-top: 1.2rem;
        color: #cbd5e1;
        font-size: 1.08rem;
        line-height: 1.75;
    }

    .pill {
        display: inline-flex;
        margin-bottom: 1.1rem;
        padding: 0.48rem 0.8rem;
        border-radius: 999px;
        color: #dbeafe;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.14);
        font-size: 0.84rem;
        font-weight: 700;
    }

    .metric-card, .glass-panel {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.075);
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(18px);
        animation: fadeUp 800ms ease both;
    }

    .metric-card {
        min-height: 150px;
        padding: 1.35rem;
        transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
    }

    .metric-card:hover {
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(125, 211, 252, 0.42);
        background: rgba(255, 255, 255, 0.105);
    }

    .metric-label {
        color: #a7b0c0;
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        margin-top: 0.65rem;
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .metric-note {
        margin-top: 0.4rem;
        color: #cbd5e1;
        font-size: 0.92rem;
    }

    .glass-panel {
        padding: 1.35rem;
    }

    .section-title {
        margin: 2rem 0 0.8rem;
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def dataset_snapshot(df: pd.DataFrame | None) -> dict[str, str]:
    if df is None:
        return {
            "Rows": "Awaiting CSV",
            "Columns": "Awaiting CSV",
            "Missing Cells": "Awaiting CSV",
            "Numeric Features": "Awaiting CSV",
        }

    return {
        "Rows": f"{df.shape[0]:,}",
        "Columns": f"{df.shape[1]:,}",
        "Missing Cells": f"{int(df.isna().sum().sum()):,}",
        "Numeric Features": f"{len(df.select_dtypes(include='number').columns):,}",
    }


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Student ML Studio")
        st.caption("Pass/fail prediction powered by Streamlit and scikit-learn.")
        st.divider()
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Dataset_Explorer.py", label="Dataset Explorer", icon="📊")
        st.page_link("pages/2_Model_Training.py", label="Model Training", icon="🧠")
        st.page_link("pages/3_Student_Prediction.py", label="Student Prediction", icon="✅")
        st.page_link("pages/4_Analytics_Dashboard.py", label="Analytics Dashboard", icon="📈")
        st.divider()

        if DEFAULT_DATASET.exists():
            st.success("Dataset found in data folder.")
        else:
            st.warning("Place StudentsPerformance.csv inside project/data.")


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    df = load_dataset(DEFAULT_DATASET)
    metrics = dataset_snapshot(df)

    st.markdown(
        """
        <div class="hero">
            <div class="pill">Premium ML Dashboard - Student Success Prediction</div>
            <h1>Predict student outcomes with clean analytics and explainable models.</h1>
            <p>
                This Streamlit application inspects the student performance dataset,
                creates a pass/fail target when needed, trains multiple classifiers,
                compares their performance, and gives an interactive prediction
                experience for individual students.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Dataset Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Rows", metrics["Rows"], "Student records available")
    with c2:
        render_metric_card("Columns", metrics["Columns"], "Features discovered")
    with c3:
        render_metric_card("Missing Cells", metrics["Missing Cells"], "Quality check")
    with c4:
        render_metric_card("Numeric Features", metrics["Numeric Features"], "Ready for scoring")

    st.markdown('<div class="section-title">Build Roadmap</div>', unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            """
            <div class="glass-panel">
                <h3 style="margin-top:0;color:white;">What this app includes</h3>
                <p style="color:#cbd5e1;line-height:1.7;">
                    Dataset exploration, automated target detection, preprocessing
                    pipelines, model training, joblib export, prediction forms,
                    model comparison charts, feature importance, and confusion
                    matrix analytics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if df is not None:
            st.dataframe(df.head(6), use_container_width=True)
        else:
            st.info("Place StudentsPerformance.csv into the data folder to preview records.")


def main() -> None:
    load_css()
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()
