from __future__ import annotations

import streamlit as st


def load_premium_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.24), transparent 32rem),
                radial-gradient(circle at top right, rgba(6, 182, 212, 0.16), transparent 30rem),
                linear-gradient(135deg, #050816 0%, #08111f 50%, #0c1022 100%);
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background: rgba(5, 8, 22, 0.84);
            border-right: 1px solid rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(18px);
        }

        .page-hero, .glass-card, .result-card {
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.075);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.26);
            backdrop-filter: blur(18px);
            animation: fadeUp 650ms ease both;
        }

        .page-hero {
            padding: 2rem;
            border-radius: 24px;
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

        .glass-card {
            padding: 1.2rem;
            border-radius: 18px;
            transition: transform 180ms ease, border-color 180ms ease;
        }

        .glass-card:hover {
            transform: translateY(-3px);
            border-color: rgba(6, 182, 212, 0.44);
        }

        .metric-label {
            color: #a7b0c0;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 0.45rem;
            color: #ffffff;
            font-size: 1.9rem;
            font-weight: 800;
        }

        .result-card {
            padding: 2rem;
            border-radius: 24px;
            text-align: center;
        }

        .pass {
            border-color: rgba(34, 197, 94, 0.45);
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.20), rgba(6, 182, 212, 0.08));
        }

        .fail {
            border-color: rgba(239, 68, 68, 0.45);
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.20), rgba(245, 158, 11, 0.08));
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active_note: str) -> None:
    with st.sidebar:
        st.markdown("### Student ML Studio")
        st.caption(active_note)
        st.divider()
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Dataset_Explorer.py", label="Dataset Explorer", icon="📊")
        st.page_link("pages/2_Model_Training.py", label="Model Training", icon="🧠")
        st.page_link("pages/3_Student_Prediction.py", label="Student Prediction", icon="✅")
        st.page_link("pages/4_Analytics_Dashboard.py", label="Analytics Dashboard", icon="📈")


def render_page_hero(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
