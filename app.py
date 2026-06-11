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
