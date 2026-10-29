"""
Maternal Satisfaction Predictor
===============================
Maternal satisfaction among vaginal and caesarean section delivery care services in
Bahir Dar City Health Facilities, Northwest Ethiopia: application of machine learning.

Model file : Maternal_Satisfaction_LR_Final_Model.pkl  (saved in Step 12 of the notebook)
Run        : streamlit run app8.py

Design note
-----------
The interface uses one dark, high-contrast palette. There is no white anywhere: every surface is a
deep navy or blue, and every piece of text has an explicit light colour, so words stay readable
whatever theme the browser or Streamlit is set to. For the best result also copy `config.toml`
into a `.streamlit` folder next to this file (it gives Streamlit's own widgets the same palette).

Everything that comes from the notebook (metrics, threshold table, study counts) is kept in
the "Reported results" block below. After re-running the notebook, update only that block.
"""

from __future__ import annotations

import base64
import html
import itertools
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = Path(__file__).parent / "Maternal_Satisfaction_LR_Final_Model.pkl"
THRESHOLD = 0.65  # decision threshold for "satisfied" (Youden J on the test set)

# Palette (dark theme, no white). Text colours are soft ice-blue tints, never pure white.
BG, CARD, CARD2, BORDER = "#0A1628", "#112440", "#173257", "#2A4D78"
TEXT, STRONG, MUTED = "#E3EEF9", "#EEF6FE", "#9FB6D2"
TEAL, BLUE, SKY = "#2DD4BF", "#4F9DF7", "#7DD3FC"
GREEN, AMBER, RED, VIOLET = "#34D399", "#F5B841", "#F87171", "#A78BFA"
GRID = "#24425F"

st.set_page_config(
    page_title="Maternal Satisfaction Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Reported results (copied from the notebook output)
# =============================================================================
STUDY = {"n_total": 894, "n_test": 179, "n_not_satisfied": 345, "n_satisfied": 549, "n_candidates": 27}

# Confusion matrix on the test set at THRESHOLD
TN, FP, FN, TP = 56, 13, 13, 97

# Not computed from the model, so update by hand after re-running the notebook.
AUC_ROC = 0.92  # approximate
BRIER = 0.1204

THRESHOLD_TABLE = pd.DataFrame({
    "Threshold":   [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "Sensitivity": [0.991, 0.991, 0.991, 0.973, 0.973, 0.964, 0.891, 0.882, 0.791, 0.664, 0.627],
    "Specificity": [0.435, 0.464, 0.464, 0.522, 0.609, 0.638, 0.783, 0.812, 0.884, 0.913, 0.913],
    "F1 score":    [0.845, 0.852, 0.852, 0.856, 0.877, 0.880, 0.879, 0.882, 0.849, 0.772, 0.746],
    "Accuracy":    [0.777, 0.788, 0.788, 0.799, 0.832, 0.838, 0.849, 0.855, 0.827, 0.760, 0.737],
    "Youden J":    [0.426, 0.455, 0.455, 0.494, 0.581, 0.601, 0.674, 0.693, 0.675, 0.577, 0.540],
})

# Metrics derived from the confusion matrix, so they always agree with it.
N = TN + FP + FN + TP
METRICS = {
    "Accuracy": (TP + TN) / N,
    "Sensitivity": TP / (TP + FN),
    "Specificity": TN / (TN + FP),
    "Precision": TP / (TP + FP),
}
METRICS["F1 score"] = 2 * METRICS["Precision"] * METRICS["Sensitivity"] / (
    METRICS["Precision"] + METRICS["Sensitivity"]
)

# =============================================================================
# Model inputs (order and SPSS codes must match the training data)
# =============================================================================
@dataclass(frozen=True)
class Feature:
    code: str
    label: str
    help: str
    options: dict  # display text -> SPSS code


FEATURES = [
    Feature("q411owners", "Facility ownership", "Type of facility where the delivery took place.",
            {"Private": 1.0, "Government": 2.0}),
    Feature("q301greeti", "Greeted by the health professional", "Was the mother greeted on arrival?",
            {"Yes": 1.0, "No": 2.0}),
    Feature("q112waitpl", "Waiting area available", "Was there a designated waiting area?",
            {"Yes": 1.0, "No": 2.0}),
    Feature("q113privac", "Privacy during examination", "Was privacy maintained during the physical examination?",
            {"Yes": 1.0, "No": 2.0}),
    Feature("distancerecode1", "Distance to the facility", "Distance from the mother's home to the facility.",
            {"10 km or less": 1.0, "More than 10 km": 2.0}),
    Feature("q302respec", "Respectful care during delivery", "Was the mother treated respectfully by staff?",
            {"Yes": 1.0, "No": 2.0}),
    Feature("q110reason", "Reason for the visit", "Why the mother came to this facility.",
            {"Planned": 1.0, "Referred": 2.0}),
    Feature("q111timeta", "Time to reach the facility", "Travel time from home to the facility.",
            {"Less than 1 hour": 1.0, "More than 1 hour": 2.0}),
    Feature("q108reside", "Residence", "Where the mother currently lives.",
            {"Urban": 1.0, "Rural": 2.0}),
    Feature("q102mstatu", "Marital status", "Current marital status of the mother.",
            {"Single": 1.0, "Married": 2.0, "Cohabited": 3.0, "Widowed": 4.0, "Divorced": 5.0}),
]
FEATURE_ORDER = [f.code for f in FEATURES]
FEATURE_BY_CODE = {f.code: f for f in FEATURES}

# How the questions are grouped on the Predict page (display only; model order is FEATURE_ORDER).
FORM_GROUPS = [
    ("🏥", "Facility and visit", ["q411owners", "q110reason", "q112waitpl", "q113privac"]),
    ("💬", "Care experience", ["q301greeti", "q302respec"]),
    ("🏡", "Access and background", ["distancerecode1", "q111timeta", "q108reside", "q102mstatu"]),
]

# =============================================================================
# Icons (SVG sent as image data, so they render in every browser and Streamlit version)
# =============================================================================
def svg_image(svg: str, size: int, alt: str) -> str:
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt="{alt}">'


PREDICT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
    '<defs><linearGradient id="a" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#2DD4BF"/><stop offset="1" stop-color="#3B82F6"/></linearGradient></defs>'
    '<circle cx="48" cy="48" r="44" fill="url(#a)"/>'
    '<circle cx="48" cy="48" r="44" fill="none" stroke="#7DD3FC" stroke-opacity=".55" stroke-width="2"/>'
    '<path d="M14 52h16l8-22 12 40 9-26 6 8h17" fill="none" stroke="#06182C" stroke-width="6" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M70 18l2.6 6.4 6.4 2.6-6.4 2.6L70 36l-2.6-6.4L61 27l6.4-2.6z" fill="#FDE68A"/></svg>'
)
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#173257"/><stop offset="1" stop-color="#1B998B"/></linearGradient>'
    '<linearGradient id="h" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#FB7185"/><stop offset="1" stop-color="#FDA4AF"/></linearGradient></defs>'
    '<rect width="64" height="64" rx="16" fill="url(#g)"/>'
    '<path d="M32 49C20 41 14 34 14 26a9 9 0 0 1 18-3 9 9 0 0 1 18 3c0 8-6 15-18 23z" fill="url(#h)"/></svg>'
)
PREDICT_ICON = f'<div class="picon">{svg_image(PREDICT_SVG, 76, "Predict")}</div>'
PREDICT_ICON_SMALL = f'<div class="picon sm">{svg_image(PREDICT_SVG, 46, "Predict")}</div>'
LOGO_ICON = svg_image(LOGO_SVG, 40, "Logo")


def emoji_icon(symbol: str) -> str:
    return f'<div class="ebadge">{symbol}</div>'


# =============================================================================
# Styling
# =============================================================================
ROOT_VARS = f"""
:root {{
  --bg: {BG}; --card: {CARD}; --card2: {CARD2}; --border: {BORDER};
  --text: {TEXT}; --strong: {STRONG}; --muted: {MUTED};
  --teal: {TEAL}; --blue: {BLUE}; --sky: {SKY};
  --green: {GREEN}; --amber: {AMBER}; --red: {RED};
}}
"""

CSS = """
/* ---------- Page ---------- */
.stApp {
  background: radial-gradient(900px 420px at 92% -8%, rgba(45,212,191,.14), transparent 70%),
              radial-gradient(800px 420px at -6% 8%, rgba(79,157,247,.12), transparent 70%),
              linear-gradient(180deg, #0A1628 0%, #0B1B31 100%) !important;
  color: var(--text) !important;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ---------- Text: explicit light colours everywhere ---------- */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] em,
.stApp [data-testid="stMarkdownContainer"] td,
.stApp [data-testid="stMarkdownContainer"] div { color: var(--text) !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stHeading"] * { color: var(--strong) !important; }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] * { color: var(--text) !important; font-weight: 600; }
.stApp [data-testid="stTooltipIcon"] svg { color: var(--muted) !important; fill: var(--muted) !important; }
.stApp [data-testid="stTooltipContent"],
.stApp [data-testid="stTooltipContent"] * { background: var(--card2) !important; color: var(--text) !important; }
.stApp code { background: rgba(125,211,252,.12) !important; color: var(--sky) !important; border-radius: 6px; }
.stApp hr { border-color: var(--border) !important; }

/* ---------- Hero ---------- */
.stApp .hero { position: relative; overflow: hidden; display: flex; align-items: center; gap: 24px;
  background: linear-gradient(120deg, #0E2B52 0%, #134A7C 55%, #0F766E 130%);
  border: 1px solid rgba(125,211,252,.25); border-radius: 22px; padding: 30px 36px; margin-bottom: 26px;
  box-shadow: 0 16px 36px rgba(0,0,0,.45); }
.stApp .hero::after { content: ""; position: absolute; right: -60px; top: -70px; width: 260px; height: 260px;
  border-radius: 50%; background: rgba(125,211,252,.08); }
.stApp .hero::before { content: ""; position: absolute; right: 90px; bottom: -110px; width: 220px; height: 220px;
  border-radius: 50%; background: rgba(45,212,191,.08); }
.stApp .hero .icon { flex: 0 0 auto; z-index: 1; }
.stApp .hero .text { z-index: 1; }
.stApp .hero h1 { font-size: 2.05rem; line-height: 1.2; margin: 0 0 8px 0; padding: 0;
  color: var(--strong) !important; letter-spacing: -.01em; }
.stApp .hero p { font-size: 1.03rem; margin: 0; color: #BBD3EC !important; max-width: 70ch; }
.stApp .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.stApp .chip { background: rgba(45,212,191,.14); border: 1px solid rgba(45,212,191,.45);
  color: #BFF5EC !important; padding: 4px 13px; border-radius: 999px; font-size: .82rem; font-weight: 600; }
.stApp .ebadge { width: 76px; height: 76px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 2.4rem; background: rgba(125,211,252,.14);
  border: 2px solid rgba(125,211,252,.4); }

/* ---------- Predict icon (pulsing ring) ---------- */
@keyframes pulse { 0% { transform: scale(.92); opacity: .75; } 100% { transform: scale(1.28); opacity: 0; } }
.stApp .picon { position: relative; width: 76px; height: 76px; flex: 0 0 auto; }
.stApp .picon img { position: relative; z-index: 1; display: block; filter: drop-shadow(0 6px 14px rgba(0,0,0,.45)); }
.stApp .picon::after { content: ""; position: absolute; inset: 0; border-radius: 50%;
  border: 2px solid rgba(45,212,191,.7); animation: pulse 2.4s ease-out infinite; }
.stApp .picon.sm { width: 46px; height: 46px; }

/* ---------- Section titles ---------- */
.stApp .section-title { display: flex; align-items: center; gap: 10px; font-size: 1.25rem; font-weight: 700;
  color: var(--strong) !important; margin: 30px 0 14px 0; }
.stApp .section-title span { color: var(--strong) !important; }
.stApp .section-title::after { content: ""; flex: 1; height: 2px; border-radius: 2px;
  background: linear-gradient(90deg, rgba(79,157,247,.6), transparent); }
.stApp .group-title { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem;
  color: var(--strong) !important; margin-bottom: 8px; }
.stApp .group-title .badge { background: rgba(45,212,191,.16); border: 1px solid rgba(45,212,191,.35);
  border-radius: 10px; padding: 3px 8px; }

/* ---------- Cards ---------- */
[class*="st-key-box-"] { background: var(--card); border: 1px solid var(--border); border-radius: 18px;
  padding: 18px 22px; box-shadow: 0 10px 24px rgba(0,0,0,.32); }
.stApp [data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 16px;
  padding: 14px 18px; box-shadow: inset 0 3px 0 var(--teal), 0 10px 22px rgba(0,0,0,.32); }
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricLabel"] * {
  color: var(--muted) !important; font-weight: 600; }
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricValue"] * {
  color: var(--strong) !important; font-weight: 800; }
.stApp [data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.stApp [data-testid="stExpander"] details { background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; }
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * {
  color: var(--text) !important; font-weight: 600; }
.stApp [data-testid="stPlotlyChart"] { background: transparent !important; }

/* ---------- Inputs ---------- */
.stApp [data-baseweb="select"] > div { background: var(--card2) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; }
.stApp [data-baseweb="select"] *, .stApp [data-baseweb="select"] input { color: var(--text) !important; }
.stApp [data-baseweb="select"] svg { fill: var(--muted) !important; }
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, ul[role="listbox"] {
  background: var(--card2) !important; border: 1px solid var(--border); }
li[role="option"], li[role="option"] * { background: transparent !important; color: var(--text) !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background: rgba(45,212,191,.18) !important; }

/* Yes/No style options become pills */
.stApp [data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
.stApp [data-testid="stRadio"] [role="radiogroup"] label { background: var(--card2); border: 1px solid var(--border);
  border-radius: 999px; padding: 6px 18px; margin: 0; transition: all .15s ease; }
.stApp [data-testid="stRadio"] [role="radiogroup"] label:hover { border-color: var(--teal); }
.stApp [data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display: none; }
.stApp [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(45,212,191,.22); border-color: var(--teal); }
.stApp [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {
  color: #BFF5EC !important; font-weight: 700; }

/* ---------- Primary button ---------- */
.stApp [data-testid="stBaseButton-primaryFormSubmit"], .stApp [data-testid="stBaseButton-primary"] {
  width: 100%; height: 3.4rem; border: none; border-radius: 14px; font-size: 1.12rem; font-weight: 800;
  background: linear-gradient(90deg, #2DD4BF, #4F9DF7); box-shadow: 0 10px 22px rgba(45,212,191,.28);
  transition: transform .15s ease, box-shadow .15s ease; }
.stApp [data-testid="stBaseButton-primaryFormSubmit"] *, .stApp [data-testid="stBaseButton-primary"] * {
  color: #04101E !important; font-weight: 800; }
.stApp [data-testid="stBaseButton-primaryFormSubmit"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px); box-shadow: 0 14px 28px rgba(45,212,191,.42); }

/* ---------- Call to action above the button ---------- */
.stApp .cta { display: flex; align-items: center; gap: 16px; margin: 6px 0 14px 0; padding: 14px 18px;
  border-radius: 16px; background: linear-gradient(120deg, rgba(45,212,191,.14), rgba(79,157,247,.12));
  border: 1px solid rgba(45,212,191,.4); }
.stApp .cta b { display: block; font-size: 1.05rem; color: var(--strong) !important; }
.stApp .cta span { font-size: .92rem; color: var(--muted) !important; }

/* ---------- Result ---------- */
@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
.stApp .result { display: flex; align-items: center; gap: 20px; border-radius: 18px; padding: 24px 28px;
  margin: 10px 0 8px 0; animation: rise .5s ease both; box-shadow: 0 12px 26px rgba(0,0,0,.4); }
.stApp .result .emoji { font-size: 2.8rem; line-height: 1; }
.stApp .result h2 { margin: 0 0 6px 0; padding: 0; font-size: 1.7rem; }
.stApp .result p { margin: 0; font-size: 1.02rem; color: var(--text) !important; }
.stApp .result p b { color: var(--strong) !important; }
.stApp .result.ok { background: linear-gradient(120deg, rgba(52,211,153,.22), rgba(52,211,153,.07));
  border: 1px solid rgba(52,211,153,.5); border-left: 7px solid var(--green); }
.stApp .result.ok h2 { color: #A7F3D0 !important; }
.stApp .result.low { background: linear-gradient(120deg, rgba(245,184,65,.22), rgba(245,184,65,.07));
  border: 1px solid rgba(245,184,65,.5); border-left: 7px solid var(--amber); }
.stApp .result.low h2 { color: #FDE68A !important; }

.stApp .probbar { position: relative; height: 16px; border-radius: 999px; margin: 18px 4px 6px 4px; }
.stApp .probfill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }
.stApp .probmark { position: absolute; top: -6px; width: 3px; height: 28px; background: var(--sky); border-radius: 2px; }
.stApp .problabels { display: flex; justify-content: space-between; font-size: .8rem; margin: 0 4px; }
.stApp .problabels span { color: var(--muted) !important; }

.stApp .hint { display: flex; align-items: center; gap: 14px; background: rgba(79,157,247,.1);
  border: 1px dashed var(--blue); border-radius: 16px; padding: 18px 22px; margin-top: 14px; }
.stApp .hint .emoji { font-size: 1.8rem; }
.stApp .hint, .stApp .hint b { color: var(--text) !important; }

/* ---------- Tables (own HTML tables, so no white grid can appear) ---------- */
.stApp .dtwrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 14px; background: var(--card); }
.stApp table.dt, .stApp [data-testid="stMarkdownContainer"] table { width: 100%; border-collapse: collapse; font-size: .92rem; }
.stApp table.dt th, .stApp [data-testid="stMarkdownContainer"] th { background: var(--card2) !important;
  color: #CFE3F8 !important; text-align: left; padding: 10px 14px; font-weight: 700;
  border: none !important; border-bottom: 1px solid var(--border) !important; }
.stApp table.dt td, .stApp [data-testid="stMarkdownContainer"] td { padding: 9px 14px; color: var(--text) !important;
  background: transparent !important; border: none !important; border-top: 1px solid #1F3D61 !important; }
.stApp table.dt tr.hl td { background: rgba(45,212,191,.16) !important; color: #BFF5EC !important; font-weight: 700; }

/* ---------- Sidebar ---------- */
.stApp [data-testid="stSidebar"] { background: linear-gradient(180deg, #0C1D36 0%, #091526 100%) !important;
  border-right: 1px solid var(--border); }
.stApp .brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 6px 0; }
.stApp .brand b { display: block; color: var(--strong) !important; font-size: 1.05rem; line-height: 1.2; }
.stApp .brand span { color: var(--muted) !important; font-size: .8rem; }
.stApp [data-testid="stSidebar"] [role="radiogroup"] label { background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 9px 14px; margin-bottom: 8px; width: 100%; transition: all .15s ease; }
.stApp [data-testid="stSidebar"] [role="radiogroup"] label * { color: var(--text) !important; }
.stApp [data-testid="stSidebar"] [role="radiogroup"] label:hover { border-color: var(--teal); transform: translateX(3px); }
.stApp [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, #2DD4BF, #4F9DF7); border-color: transparent; }
.stApp [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
  color: #04101E !important; font-weight: 800; }

.stApp .footer { text-align: center; font-size: .82rem; margin-top: 34px; }
.stApp .footer, .stApp .footer * { color: var(--muted) !important; }
"""

st.markdown(f"<style>{ROOT_VARS}{CSS}</style>", unsafe_allow_html=True)


def hero(title: str, text: str, icon: str = "", chips: list[str] | None = None) -> None:
    chips_html = ""
    if chips:
        chips_html = '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>"
    icon_html = f'<div class="icon">{icon}</div>' if icon else ""
    st.markdown(
        f'<div class="hero">{icon_html}<div class="text"><h1>{title}</h1><p>{text}</p>{chips_html}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, icon: str = "") -> None:
    st.markdown(f'<div class="section-title"><span>{icon}</span>{title}</div>', unsafe_allow_html=True)


_card_ids = itertools.count(1)


def card():
    """A rounded dark card. Falls back to a bordered container on older Streamlit versions."""
    try:
        return st.container(key=f"box-{next(_card_ids)}")
    except TypeError:
        return st.container(border=True)


def html_table(df: pd.DataFrame, decimals: dict | None = None, highlight=None) -> None:
    """Dark, readable table drawn with HTML (st.dataframe draws a light grid that can hide text)."""
    decimals = decimals or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            value = row[c]
            if c in decimals and isinstance(value, (int, float)):
                value = f"{value:.{decimals[c]}f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        css_class = ' class="hl"' if highlight is not None and highlight(row) else ""
        rows.append(f"<tr{css_class}>{''.join(cells)}</tr>")
    st.markdown(
        f'<div class="dtwrap"><table class="dt"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Model helpers
# =============================================================================
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def check_model_inputs(model) -> None:
    """Stop with a clear message if the saved model expects different inputs than this app sends."""
    expected = list(model.named_steps["preprocessor"].feature_names_in_)
    missing, extra = set(expected) - set(FEATURE_ORDER), set(FEATURE_ORDER) - set(expected)
    if missing or extra:
        st.error(
            "The model file and this app use different input variables.\n\n"
            f"- Inputs the model expects but the app does not provide: `{sorted(missing)}`\n"
            f"- Inputs the app provides but the model does not use: `{sorted(extra)}`\n\n"
            "Update `FEATURES` in `app.py` to match the model, or save the model again from the notebook."
        )
        st.stop()


def odds_ratio_table(model) -> pd.DataFrame:
    """Odds ratios of every encoded category, with readable names."""
    prep, clf = model.named_steps["preprocessor"], model.named_steps["classifier"]
    rows = []
    for encoded, coef in zip(prep.get_feature_names_out(), clf.coef_[0]):
        name = encoded.split("__", 1)[-1]
        code, _, value = name.rpartition("_")
        feature = FEATURE_BY_CODE.get(code)
        if feature is not None:
            option = next((k for k, v in feature.options.items() if v == float(value)), value)
            name = f"{feature.label}: {option}"
        rows.append({"Category": name, "Odds ratio": float(np.exp(coef))})
    return pd.DataFrame(rows)


def ask(feature: Feature):
    """Two-option questions use compact horizontal buttons; longer lists use a dropdown."""
    if len(feature.options) <= 2:
        return st.radio(feature.label, list(feature.options), help=feature.help, key=feature.code, horizontal=True)
    return st.selectbox(feature.label, list(feature.options), help=feature.help, key=feature.code)


def finish(fig: go.Figure, **layout) -> go.Figure:
    """Give every chart the dark palette: transparent background and light text, grid and tooltips."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT), **layout)
    fig.update_layout(legend_font_color=TEXT,
                      hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     tickfont=dict(color=MUTED), title_font=dict(color=TEXT))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     tickfont=dict(color=MUTED), title_font=dict(color=TEXT))
    return fig


# =============================================================================
# Pages
# =============================================================================
def page_predict(model) -> None:
    hero("Maternal Satisfaction Predictor",
         "Estimate whether a mother is likely to be satisfied with her delivery care, using a logistic "
         f"regression model trained on {STUDY['n_total']} mothers from Bahir Dar City health facilities.",
         icon=PREDICT_ICON,
         chips=["Logistic regression", f"{len(FEATURES)} inputs", f"Threshold {THRESHOLD:.2f}"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision threshold", f"{THRESHOLD:.2f}", help="Predicted satisfied when probability ≥ threshold.")
    c2.metric("Sensitivity", f"{METRICS['Sensitivity']:.1%}")
    c3.metric("Specificity", f"{METRICS['Specificity']:.1%}")
    c4.metric("F1 score", f"{METRICS['F1 score']:.1%}")

    section("Mother's details", "📝")
    st.caption(f"Answer all {len(FEATURES)} questions. Service-rating items were excluded from the model "
               "because they are part of the satisfaction outcome itself.")

    answers: dict[str, float] = {}
    with st.form("predict_form"):
        for icon, title, codes in FORM_GROUPS:
            with card():
                st.markdown(f'<div class="group-title"><span class="badge">{icon}</span>{title}</div>',
                            unsafe_allow_html=True)
                columns = st.columns(2, gap="large")
                for i, code in enumerate(codes):
                    feature = FEATURE_BY_CODE[code]
                    with columns[i % 2]:
                        choice = ask(feature)
                        answers[code] = feature.options[choice]
        st.markdown(
            f'<div class="cta">{PREDICT_ICON_SMALL}<div><b>Ready to predict?</b>'
            '<span>Press the button to see the probability of satisfaction.</span></div></div>',
            unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict satisfaction", type="primary",
                                          icon=":material/online_prediction:")

    if not submitted:
        st.markdown('<div class="hint"><span class="emoji">🔮</span><div><b>Your result will appear here.</b><br>'
                    'You will see a verdict, a probability bar and a gauge once you press <b>Predict satisfaction</b>.'
                    '</div></div>', unsafe_allow_html=True)
        return

    row = pd.DataFrame([answers])[FEATURE_ORDER]
    p_satisfied = float(model.predict_proba(row)[0][1])
    satisfied = p_satisfied >= THRESHOLD

    section("Result", "🎯")
    if satisfied:
        st.markdown(
            f'<div class="result ok"><div class="emoji">😊</div><div><h2>Likely satisfied</h2>'
            f'<p>Predicted probability of satisfaction is <b>{p_satisfied:.1%}</b>, at or above the '
            f'threshold of {THRESHOLD:.2f}.</p></div></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result low"><div class="emoji">⚠️</div><div><h2>Likely not satisfied</h2>'
            f'<p>Predicted probability of satisfaction is <b>{p_satisfied:.1%}</b>, below the threshold of '
            f'{THRESHOLD:.2f}. Facility conditions and staff conduct are worth reviewing.</p></div></div>',
            unsafe_allow_html=True)

    fill = GREEN if satisfied else AMBER
    t = THRESHOLD * 100
    track = (f"linear-gradient(90deg, rgba(245,184,65,.30) 0%, rgba(245,184,65,.30) {t}%, "
             f"rgba(52,211,153,.30) {t}%, rgba(52,211,153,.30) 100%)")
    st.markdown(
        f'<div class="probbar" style="background:{track};">'
        f'<div class="probfill" style="width:{p_satisfied*100:.1f}%;background:{fill};"></div>'
        f'<div class="probmark" style="left:{t:.0f}%;"></div></div>'
        f'<div class="problabels"><span>0%</span><span>Threshold {THRESHOLD:.0%}</span><span>100%</span></div>',
        unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_satisfied * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": STRONG}},
        title={"text": "Probability of satisfaction", "font": {"color": TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": fill, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, THRESHOLD * 100], "color": "rgba(245,184,65,0.20)"},
                      {"range": [THRESHOLD * 100, 100], "color": "rgba(52,211,153,0.20)"}],
            "threshold": {"line": {"color": SKY, "width": 4}, "thickness": 0.85, "value": THRESHOLD * 100},
        },
    ))
    finish(gauge, height=300, margin=dict(l=30, r=30, t=60, b=10))
    with card():
        st.plotly_chart(gauge, width="stretch")
        st.caption(f"The blue line marks the decision threshold ({THRESHOLD:.0%}).")

    with st.expander("📋 Input summary"):
        summary = pd.DataFrame({
            "Question": [f.label for f in FEATURES],
            "Answer": [next(k for k, v in f.options.items() if v == answers[f.code]) for f in FEATURES],
        })
        html_table(summary)
    st.caption("Decision-support estimate for research and quality improvement. It is not a clinical diagnosis.")


def page_performance(model) -> None:
    hero("Model performance",
         f"Evaluation on the held-out test set ({STUDY['n_test']} mothers, 20% of {STUDY['n_total']}).",
         icon=emoji_icon("📊"))

    section(f"Performance at threshold {THRESHOLD:.2f}", "✅")
    top = st.columns(4)
    for col, name in zip(top, ["Accuracy", "Sensitivity", "Specificity", "Precision"]):
        col.metric(name, f"{METRICS[name]:.1%}")
    bottom = st.columns(4)
    bottom[0].metric("F1 score", f"{METRICS['F1 score']:.1%}")
    bottom[1].metric("AUC-ROC", f"{AUC_ROC:.3f}", help="Approximate value entered by hand; update from the notebook.")
    bottom[2].metric("Brier score", f"{BRIER:.4f}", help="Lower is better (0 = perfect).")
    bottom[3].metric("Test mothers", f"{N}")

    section("Confusion matrix", "🧮")
    labels = ["Not satisfied", "Satisfied"]
    matrix = go.Figure(go.Heatmap(
        z=[[TN, FP], [FN, TP]],
        x=[f"Predicted {l.lower()}" for l in labels],
        y=[f"Actual {l.lower()}" for l in labels],
        text=[[f"TN = {TN}", f"FP = {FP}"], [f"FN = {FN}", f"TP = {TP}"]],
        texttemplate="<b>%{z}</b><br>%{text}", textfont={"size": 15, "color": STRONG},
        colorscale=[[0.0, "#12304F"], [1.0, "#0F766E"]], showscale=False,
    ))
    finish(matrix, height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis={"autorange": "reversed"})
    left, right = st.columns([1.2, 1])
    with left:
        with card():
            st.plotly_chart(matrix, width="stretch")
    with right:
        with card():
            st.markdown(
                "| Term | Count | Meaning |\n|---|---|---|\n"
                f"| TN | {TN} | Not satisfied, correctly identified |\n"
                f"| FP | {FP} | Not satisfied, wrongly called satisfied |\n"
                f"| FN | {FN} | Satisfied, wrongly called not satisfied |\n"
                f"| TP | {TP} | Satisfied, correctly identified |")

    section("Metrics across thresholds", "📈")
    styles = {"Sensitivity": (GREEN, "solid"), "Specificity": (RED, "solid"),
              "F1 score": (BLUE, "dash"), "Accuracy": (VIOLET, "dot"), "Youden J": (AMBER, "dashdot")}
    lines = go.Figure()
    for metric, (color, dash) in styles.items():
        lines.add_trace(go.Scatter(x=THRESHOLD_TABLE["Threshold"], y=THRESHOLD_TABLE[metric], name=metric,
                                   mode="lines+markers", line=dict(color=color, width=2.5, dash=dash)))
    lines.add_vline(x=THRESHOLD, line_dash="dash", line_color=SKY, line_width=2,
                    annotation_text=f"Threshold {THRESHOLD:.2f}", annotation_position="top right",
                    annotation_font_color=SKY)
    finish(lines, xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0.3, 1.05],
           legend=dict(orientation="h", y=-0.25), height=420, hovermode="x unified")
    with card():
        st.plotly_chart(lines, width="stretch")
    with st.expander("📋 Threshold table"):
        html_table(THRESHOLD_TABLE,
                   decimals={c: (2 if c == "Threshold" else 3) for c in THRESHOLD_TABLE.columns},
                   highlight=lambda r: abs(r["Threshold"] - THRESHOLD) < 1e-9)

    section("Odds ratios", "⚖️")
    try:
        table = odds_ratio_table(model)
        table["Log distance"] = np.abs(np.log(table["Odds ratio"]))
        table = table.nlargest(15, "Log distance").sort_values("Odds ratio")
        table["Effect"] = np.where(table["Odds ratio"] >= 1, "Raises odds of satisfaction", "Lowers odds of satisfaction")
        chart = px.bar(table, x="Odds ratio", y="Category", orientation="h", color="Effect",
                       color_discrete_map={"Raises odds of satisfaction": GREEN, "Lowers odds of satisfaction": RED})
        chart.add_vline(x=1.0, line_dash="dash", line_color=MUTED)
        finish(chart, height=520, legend_title_text="", legend=dict(orientation="h", y=-0.15))
        with card():
            st.plotly_chart(chart, width="stretch")
            st.caption("The 15 strongest categories. Each odds ratio is relative to the first option of its question "
                       "(for example Private for facility ownership). Above 1 raises the odds of satisfaction.")
    except Exception as exc:
        st.info(f"Odds ratios are not available for this model file: {exc}")


def page_about(model) -> None:
    hero("About this study",
         "Maternal satisfaction among vaginal and caesarean section delivery care services in Bahir Dar City "
         "Health Facilities, Northwest Ethiopia: application of machine learning.",
         icon=emoji_icon("📚"))

    left, right = st.columns(2, gap="large")
    with left:
        with card():
            st.subheader("🧾 Study overview")
            ns, s = STUDY["n_not_satisfied"], STUDY["n_satisfied"]
            st.markdown(
                f"- **Participants:** {STUDY['n_total']} mothers\n"
                f"- **Outcome:** overall satisfaction with delivery care\n"
                f"  - Not satisfied: {ns} ({ns / STUDY['n_total']:.1%})\n"
                f"  - Satisfied: {s} ({s / STUDY['n_total']:.1%})\n"
                "- **Train / test split:** 80% / 20%, stratified\n"
                "- **Cross-validation:** 5-fold stratified")
        with card():
            st.subheader("🔍 Variable selection")
            st.markdown(
                "Excluded before modelling:\n\n"
                "| Reason | Variables |\n|---|---|\n"
                "| Satisfaction sub-items (data leakage) | 10 |\n"
                "| More than 50% missing values | 2 |\n"
                "| Recoded duplicates of kept variables | 6 |\n\n"
                f"{STUDY['n_candidates']} candidate features remained. The final {len(FEATURES)} were chosen by "
                "5-fold cross-validated permutation importance (ROC-AUC).")
    with right:
        with card():
            st.subheader("⚙️ Model configuration")
            clf = model.named_steps["classifier"]
            st.markdown(
                "| Parameter | Value |\n|---|---|\n"
                "| Algorithm | Logistic regression |\n"
                f"| Regularisation C | {clf.C} |\n"
                f"| Solver | {clf.solver} |\n"
                f"| Class weight | {clf.class_weight} |\n"
                f"| Max iterations | {clf.max_iter} |\n"
                "| Tuning | GridSearchCV, ROC-AUC, 5 folds |")
        with card():
            st.subheader("🎚️ Decision threshold")
            best_j = THRESHOLD_TABLE.loc[THRESHOLD_TABLE["Youden J"].idxmax()]
            best_f1 = THRESHOLD_TABLE.loc[THRESHOLD_TABLE["F1 score"].idxmax()]
            default = THRESHOLD_TABLE.loc[THRESHOLD_TABLE["Threshold"] == 0.50].iloc[0]
            st.markdown(
                f"The threshold of **{THRESHOLD:.2f}** maximises Youden J (sensitivity + specificity − 1), "
                "balancing the detection of satisfied and not-satisfied mothers.\n\n"
                "| Rule | Threshold | Result |\n|---|---|---|\n"
                f"| Best Youden J | {best_j['Threshold']:.2f} | J = {best_j['Youden J']:.3f} |\n"
                f"| Best F1 (on this grid) | {best_f1['Threshold']:.2f} | F1 = {best_f1['F1 score']:.3f} |\n"
                f"| Default | 0.50 | J = {default['Youden J']:.3f} |")

    section(f"The {len(FEATURES)} model inputs", "🧩")
    cols = st.columns(2)
    for i, feature in enumerate(FEATURES):
        with cols[i % 2]:
            with card():
                st.markdown(f"**{feature.label}**")
                st.caption(feature.help)
                st.caption("Options: " + ", ".join(feature.options))
    st.caption("Notebook: Maternal_Satisfaction_ML_Project.ipynb · Dataset: Delivery care satisfaction.sav · "
               "Credit: Getasew Mulat")


# =============================================================================
# App
# =============================================================================
NAV = {"Predict": "🔮  Predict", "Model performance": "📊  Model performance", "About": "📚  About"}

with st.sidebar:
    st.markdown(
        f'<div class="brand">{LOGO_ICON}<div><b>Maternal Satisfaction</b>'
        '<span>Bahir Dar City, Northwest Ethiopia</span></div></div>',
        unsafe_allow_html=True)
    st.write("")
    page = st.radio("Section", list(NAV), format_func=NAV.get, label_visibility="collapsed")
    st.divider()
    st.caption(f"Logistic regression · {len(FEATURES)} inputs · threshold {THRESHOLD:.2f}")
    st.caption(f"Sensitivity {METRICS['Sensitivity']:.1%} · Specificity {METRICS['Specificity']:.1%}")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Run the notebook through Step 12 and place the "
             "file in the same folder as `app8.py`.")
    st.stop()

model = load_model(MODEL_PATH)
check_model_inputs(model)

{"Predict": page_predict, "Model performance": page_performance, "About": page_about}[page](model)

st.markdown('<div class="footer">Decision-support tool for research and quality improvement · '
            'Not a clinical diagnosis</div>', unsafe_allow_html=True)
