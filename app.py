from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET = DATA_DIR / "StudentsPerformance.csv"


st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #070b18;
            --panel: rgba(255, 255, 255, 0.08);
            --panel-strong: rgba(255, 255, 255, 0.14);
            --text: #f8fafc;
            --muted: #a7b0c0;
            --accent: #7c3aed;
            --accent-2: #06b6d4;
            --good: #22c55e;
            --warn: #f59e0b;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.30), transparent 34rem),
                radial-gradient(circle at top right, rgba(6, 182, 212, 0.18), transparent 30rem),
                linear-gradient(135deg, #050816 0%, #08111f 48%, #0c1022 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: rgba(5, 8, 22, 0.82);
            border-right: 1px solid rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(18px);
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 3.25rem 3rem;
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
            max-width: 850px;
            margin: 0;
            color: #ffffff;
            font-size: clamp(2.4rem, 5vw, 4.8rem);
            line-height: 1.02;
            letter-spacing: 0;
            font-weight: 800;
        }

        .hero p {
            max-width: 780px;
            margin-top: 1.2rem;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.75;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1.1rem;
            padding: 0.48rem 0.8rem;
            border-radius: 999px;
            color: #dbeafe;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.14);
            font-size: 0.84rem;
            font-weight: 700;
        }