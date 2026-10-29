"""
app.py  –  Maternal Satisfaction Predictor
==========================================
Study     : Maternal satisfaction among vaginal and caesarean section
            delivery care services in Bahir Dar City Health Facilities,
            Northwest Ethiopia: Application of Machine Learning

Model     : Logistic Regression  (C=0.1 | class_weight='balanced' | lbfgs)
            Tuned via 5-fold Stratified GridSearchCV (ROC-AUC scoring)
Model file: Maternal_Satisfaction_LR_Final_Model.pkl

Optimal threshold : 0.65  (Youden J statistic, held-out test set, n=179)
  Sensitivity = 88.2%  |  Specificity = 81.2%
  Precision   = 88.2%  |  Recall      = 88.2%
  F1 Score    = 88.2%  |  Accuracy    = 85.5%

Run:  streamlit run app.py
"""

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternal Satisfaction Predictor",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH        = Path(__file__).parent / "Maternal_Satisfaction_LR_Final_Model.pkl"
OPTIMAL_THRESHOLD = 0.65   # Youden J – do NOT change without re-running analysis

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* gradient header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2.4em; margin: 0 0 8px 0; }
.hero p  { font-size: 1.05em; opacity: 0.85; margin: 0; }

/* feature cards */
.feat-card {
    background: #f8f9ff;
    border: 1px solid #e0e4f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    transition: box-shadow 0.2s;
}
.feat-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.10); }

/* result box */
.result-satisfied {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border-left: 6px solid #28a745;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
}
.result-not-satisfied {
    background: linear-gradient(135deg, #fff3cd, #ffeeba);
    border-left: 6px solid #ffc107;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
}
.result-satisfied h2, .result-not-satisfied h2 {
    margin: 0 0 8px 0; font-size: 1.9em;
}
.result-satisfied p, .result-not-satisfied p {
    margin: 0; font-size: 1.05em;
}

/* metric badge */
.badge {
    display: inline-block;
    background: #0f3460;
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82em;
    font-weight: 600;
    margin: 2px;
}

/* section title */
.section-title {
    font-size: 1.25em;
    font-weight: 700;
    color: #0f3460;
    border-bottom: 2px solid #0f3460;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions  (exact order the model was trained on)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "q411owners",
    "q301greeti",
    "q112waitpl",
    "q113privac",
    "distancerecode1",
    "q302respec",
    "q110reason",
    "q111timeta",
    "q108reside",
    "q102mstatu",
]

FEATURE_META = {
    "q411owners": {
        "label": "Facility Ownership",
        "icon":  "🏥",
        "help":  "Type of health facility where the delivery took place.",
        "options": {"Private": 1.0, "Government": 2.0},
    },
    "q301greeti": {
        "label": "Professional Greeting",
        "icon":  "👋",
        "help":  "Was the mother greeted by the health professional on arrival?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "q112waitpl": {
        "label": "Waiting Area Available",
        "icon":  "🪑",
        "help":  "Was a designated waiting area available at the facility?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "q113privac": {
        "label": "Privacy During Examination",
        "icon":  "🔒",
        "help":  "Was gender privacy maintained during physical examination?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "distancerecode1": {
        "label": "Distance to Facility",
        "icon":  "📍",
        "help":  "Distance from the mother's home to the health facility.",
        "options": {"≤ 10 km": 1.0, "> 10 km": 2.0},
    },
    "q302respec": {
        "label": "Respectful Care During Delivery",
        "icon":  "🤝",
        "help":  "Was the mother treated respectfully by staff during delivery?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "q110reason": {
        "label": "Reason for Hospital Visit",
        "icon":  "📋",
        "help":  "Why did the mother come to this facility for delivery?",
        "options": {"Planned visit": 1.0, "Referred": 2.0},
    },
    "q111timeta": {
        "label": "Time to Reach Facility",
        "icon":  "⏱️",
        "help":  "How long did it take the mother to travel to the facility?",
        "options": {"Less than 1 hour": 1.0, "More than 1 hour": 2.0},
    },
    "q108reside": {
        "label": "Residence",
        "icon":  "🏘️",
        "help":  "Where does the mother currently live?",
        "options": {"Urban": 1.0, "Rural": 2.0},
    },
    "q102mstatu": {
        "label": "Marital Status",
        "icon":  "💍",
        "help":  "Current marital status of the mother.",
        "options": {
            "Single":    1.0,
            "Married":   2.0,
            "Cohabited": 3.0,
            "Widowed":   4.0,
            "Divorced":  5.0,
        },
    },
}

# Model performance at threshold 0.65
MODEL_PERF = {
    "Accuracy":    0.8547,
    "Precision":   0.8818,
    "Recall":      0.8818,
    "F1 Score":    0.8818,
    "Sensitivity": 0.8818,
    "Specificity": 0.8116,
    "AUC-ROC":     0.9200,   # approximate from notebook
    "Brier Score": 0.1204,   # lower is better (0=perfect, 0.25=random)
}

# Threshold comparison table
THRESHOLD_TABLE = pd.DataFrame({
    "Threshold":   [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80],
    "Sensitivity": [0.991,0.991,0.991,0.973,0.973,0.964,0.891,0.882,0.791,0.664,0.627],
    "Specificity": [0.435,0.464,0.464,0.522,0.609,0.638,0.783,0.812,0.884,0.913,0.913],
    "F1 Score":    [0.845,0.852,0.852,0.856,0.877,0.880,0.879,0.882,0.849,0.772,0.746],
    "Accuracy":    [0.777,0.788,0.788,0.799,0.832,0.838,0.849,0.855,0.827,0.760,0.737],
    "Youden J":    [0.426,0.455,0.455,0.494,0.581,0.601,0.674,0.693,0.675,0.577,0.540],
})


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤰 Maternal Satisfaction")
    st.markdown("**Bahir Dar City Health Facilities**  \nNorthwest Ethiopia")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model summary card
    st.markdown("**Model Summary**")
    st.markdown(
        """
        <div style='font-size:0.85em; line-height:1.8;'>
        🧠 &nbsp;<b>Algorithm:</b> Logistic Regression<br>
        📦 &nbsp;<b>Features:</b> 10<br>
        📐 &nbsp;<b>Threshold:</b> 0.65 (Youden J)<br>
        🎯 &nbsp;<b>F1 Score:</b> 88.2%<br>
        ✅ &nbsp;<b>Accuracy:</b> 85.5%<br>
        👁️ &nbsp;<b>Sensitivity:</b> 88.2%<br>
        🛡️ &nbsp;<b>Specificity:</b> 81.2%<br>
        📉 &nbsp;<b>Brier Score:</b> 0.1204
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "Built from `Maternal_Satisfaction_ML_Project.ipynb`  \n"
        "Model: `Maternal_Satisfaction_LR_Final_Model.pkl`"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard model file
# ─────────────────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error(
        f"**Model file not found:** `{MODEL_PATH.name}`\n\n"
        "Run the notebook through **Step 12** (`joblib.dump(...)`) "
        "to generate it, then place it in the same folder as `app.py`."
    )
    st.stop()

model = load_model(MODEL_PATH)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":

    # Hero banner
    st.markdown("""
    <div class="hero">
        <h1>🤰 Maternal Satisfaction Predictor</h1>
        <p>Predict whether a mother will be satisfied with delivery care services
        using a Logistic Regression model trained on 894 mothers from
        Bahir Dar City Health Facilities, Northwest Ethiopia.</p>
    </div>
    """, unsafe_allow_html=True)

    # Threshold info strip
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Optimal Threshold", "0.65", help="Youden J statistic")
    col_t2.metric("Sensitivity",  "88.2%")
    col_t3.metric("Specificity",  "81.2%")
    col_t4.metric("F1 Score",     "88.2%")

    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Enter Mother\'s Details</div>',
                unsafe_allow_html=True)
    st.caption(
        "Fill in all 10 fields. These are independent, non-satisfaction variables — "
        "the 10 service-quality sub-items were excluded to prevent data leakage."
    )

    user_inputs: dict = {}
    left_col, right_col = st.columns(2, gap="large")

    for idx, feature in enumerate(FEATURE_ORDER):
        meta        = FEATURE_META[feature]
        target_col  = left_col if idx % 2 == 0 else right_col
        with target_col:
            chosen_label = st.selectbox(
                label=f"{meta['icon']}  {meta['label']}",
                options=list(meta["options"].keys()),
                help=meta["help"],
                key=feature,
            )
            user_inputs[feature] = meta["options"][chosen_label]

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        predict_clicked = st.button(
            "▶  Predict Satisfaction",
            type="primary",
            use_container_width=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    if predict_clicked:
        input_df     = pd.DataFrame([user_inputs])[FEATURE_ORDER]
        proba        = model.predict_proba(input_df)[0]
        prob_sat     = float(proba[1])
        prob_not_sat = float(proba[0])
        is_satisfied = prob_sat >= OPTIMAL_THRESHOLD

        st.markdown("---")
        st.markdown('<div class="section-title">Prediction Result</div>',
                    unsafe_allow_html=True)

        # Result banner
        if is_satisfied:
            st.markdown(f"""
            <div class="result-satisfied">
                <h2>😊 Satisfied</h2>
                <p>The predicted probability of satisfaction is
                <strong>{prob_sat*100:.1f}%</strong>, which is
                above the optimal threshold of <strong>{OPTIMAL_THRESHOLD}</strong>.
                This mother is likely to be <strong>satisfied</strong>
                with the delivery care she received.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-not-satisfied">
                <h2>😟 Not Satisfied</h2>
                <p>The predicted probability of satisfaction is
                <strong>{prob_sat*100:.1f}%</strong>, which is
                below the optimal threshold of <strong>{OPTIMAL_THRESHOLD}</strong>.
                This mother is likely to be <strong>not satisfied</strong>
                with the delivery care. Consider reviewing facility
                conditions and staff behaviour.</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability gauge (Plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob_sat * 100,
            delta = {"reference": OPTIMAL_THRESHOLD * 100,
                     "valueformat": ".1f",
                     "suffix": "%",
                     "increasing": {"color": "#28a745"},
                     "decreasing": {"color": "#dc3545"}},
            number = {"suffix": "%", "font": {"size": 42}},
            title  = {"text": "Probability of Satisfaction",
                      "font": {"size": 18}},
            gauge  = {
                "axis":  {"range": [0, 100], "tickwidth": 1,
                          "tickcolor": "#444"},
                "bar":   {"color": "#28a745" if is_satisfied else "#ffc107",
                          "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#ddd",
                "steps": [
                    {"range": [0,  OPTIMAL_THRESHOLD*100], "color": "#fff0f0"},
                    {"range": [OPTIMAL_THRESHOLD*100, 100], "color": "#f0fff4"},
                ],
                "threshold": {
                    "line":  {"color": "#0f3460", "width": 5},
                    "thickness": 0.85,
                    "value": OPTIMAL_THRESHOLD * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=320, margin=dict(l=30, r=30, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            f"▲ The dark blue line marks the optimal threshold "
            f"({OPTIMAL_THRESHOLD} = {OPTIMAL_THRESHOLD*100:.0f}%). "
            f"The needle sits at your predicted probability ({prob_sat*100:.1f}%)."
        )

        # Probability bars comparison
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[prob_sat * 100, prob_not_sat * 100],
            y=["Satisfied", "Not Satisfied"],
            orientation="h",
            marker_color=["#28a745", "#dc3545"],
            text=[f"{prob_sat*100:.1f}%", f"{prob_not_sat*100:.1f}%"],
            textposition="inside",
            textfont=dict(color="white", size=15, family="Arial Black"),
        ))
        fig_bar.add_vline(
            x=OPTIMAL_THRESHOLD * 100,
            line_dash="dash", line_color="#0f3460", line_width=3,
            annotation_text=f"Threshold {OPTIMAL_THRESHOLD}",
            annotation_position="top",
            annotation_font_color="#0f3460",
        )
        fig_bar.update_layout(
            title=dict(text="Predicted Probabilities", font=dict(size=15)),
            xaxis=dict(title="Probability (%)", range=[0, 105]),
            yaxis=dict(tickfont=dict(size=13)),
            height=200,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Input summary
        with st.expander("📋  View input summary"):
            summary_rows = []
            for f in FEATURE_ORDER:
                chosen = next(
                    lbl for lbl, code in FEATURE_META[f]["options"].items()
                    if code == user_inputs[f]
                )
                summary_rows.append({
                    "Icon":      FEATURE_META[f]["icon"],
                    "Feature":   FEATURE_META[f]["label"],
                    "Answer":    chosen,
                    "SPSS Code": user_inputs[f],
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>📊 Model Performance</h1>
        <p>Full evaluation of the Logistic Regression model on the
        held-out test set (179 mothers = 20% of 894 total).</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Overall metrics ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Performance at Optimal Threshold (0.65)</div>',
                unsafe_allow_html=True)

    m_cols = st.columns(7)
    metrics_display = [
        ("Accuracy",     MODEL_PERF["Accuracy"],    "#0f3460",  None),
        ("Precision",    MODEL_PERF["Precision"],   "#1a6b3a",  None),
        ("Recall",       MODEL_PERF["Recall"],      "#1a6b3a",  None),
        ("F1 Score",     MODEL_PERF["F1 Score"],    "#7b3f00",  None),
        ("Sensitivity",  MODEL_PERF["Sensitivity"], "#6a0dad",  None),
        ("Specificity",  MODEL_PERF["Specificity"], "#8b0000",  None),
        ("Brier Score",  MODEL_PERF["Brier Score"], "#1a5276",  "lower is better"),
    ]
    for col, (name, val, color, note) in zip(m_cols, metrics_display):
        # Brier score shown as raw value (not %), others as %
        is_brier  = name == "Brier Score"
        value_str = f"{val:.4f}" if is_brier else f"{val*100:.1f}%"
        note_html = f"<div style='font-size:0.70em; color:#888; margin-top:2px;'>{note}</div>" if note else ""
        col.markdown(
            f"""<div style="text-align:center; background:#f8f9ff;
                border:1px solid #dde; border-radius:12px; padding:18px 8px;">
                <div style="font-size:0.75em; color:#666; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.5px;">
                    {name}
                </div>
                <div style="font-size:1.8em; font-weight:800; color:{color};">
                    {value_str}
                </div>
                {note_html}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Confusion Matrix  (threshold = 0.65)</div>',
                unsafe_allow_html=True)

    cm_vals = np.array([[56, 13], [13, 97]])   # TN FP / FN TP
    cm_labels = ["Not Satisfied", "Satisfied"]

    fig_cm = go.Figure(go.Heatmap(
        z       = cm_vals,
        x       = [f"Predicted<br>{l}" for l in cm_labels],
        y       = [f"Actual<br>{l}"    for l in cm_labels],
        colorscale = "Blues",
        showscale  = False,
        text    = [["TN = 56", "FP = 13"], ["FN = 13", "TP = 97"]],
        texttemplate = "<b>%{z}</b><br><span style='font-size:11px'>%{text}</span>",
        textfont     = {"size": 16},
    ))
    fig_cm.update_layout(
        title  = dict(text="Confusion Matrix", font=dict(size=16)),
        height = 340,
        margin = dict(l=10, r=10, t=50, b=10),
        xaxis  = dict(side="bottom"),
    )

    cm_col1, cm_col2 = st.columns([1.2, 1])
    with cm_col1:
        st.plotly_chart(fig_cm, use_container_width=True)
    with cm_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        | Term | Count | Meaning |
        |---|---|---|
        | **TN** | 56 | Not-satisfied correctly identified |
        | **FP** | 13 | Not-satisfied wrongly called satisfied |
        | **FN** | 13 | Satisfied wrongly called not-satisfied |
        | **TP** | 97 | Satisfied correctly identified |
        """)

    # ── Threshold comparison chart ────────────────────────────────────────────
    st.markdown('<div class="section-title">Threshold Analysis</div>',
                unsafe_allow_html=True)

    fig_thresh = go.Figure()
    line_styles = {
        "Sensitivity": ("#2ecc71", "solid"),
        "Specificity": ("#e74c3c", "solid"),
        "F1 Score":    ("#3498db", "dash"),
        "Accuracy":    ("#9b59b6", "dot"),
        "Youden J":    ("#f39c12", "dashdot"),
    }
    for metric, (color, dash) in line_styles.items():
        fig_thresh.add_trace(go.Scatter(
            x    = THRESHOLD_TABLE["Threshold"],
            y    = THRESHOLD_TABLE[metric],
            mode = "lines+markers",
            name = metric,
            line = dict(color=color, width=2.5, dash=dash),
            marker = dict(size=7),
        ))

    fig_thresh.add_vline(
        x=OPTIMAL_THRESHOLD,
        line_dash="dash", line_color="#0f3460", line_width=3,
        annotation_text=f"Optimal threshold = {OPTIMAL_THRESHOLD}",
        annotation_position="top right",
        annotation_font_color="#0f3460",
        annotation_font_size=13,
    )
    fig_thresh.update_layout(
        title   = dict(text="Metric vs Threshold", font=dict(size=16)),
        xaxis   = dict(title="Threshold", dtick=0.05, gridcolor="#eee"),
        yaxis   = dict(title="Score", range=[0.3, 1.05], gridcolor="#eee"),
        legend  = dict(orientation="h", y=-0.25),
        height  = 420,
        hovermode = "x unified",
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_thresh, use_container_width=True)

    # Threshold data table
    with st.expander("📋  View full threshold table"):
        styled = THRESHOLD_TABLE.style \
            .highlight_max(subset=["Youden J"], color="#c3f5d5") \
            .format({c: "{:.3f}" for c in THRESHOLD_TABLE.columns if c != "Threshold"}) \
            .format({"Threshold": "{:.2f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Feature importance bar ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Feature Importance</div>',
                unsafe_allow_html=True)

    # Approximate importance from model coefficients
    try:
        ohe  = model.named_steps["preprocessor"].named_transformers_["cat"]
        coef = model.named_steps["classifier"].coef_[0]
        feat_names_ohe = ohe.get_feature_names_out(FEATURE_ORDER)

        imp_df = pd.DataFrame({
            "Feature":    feat_names_ohe,
            "Coefficient": coef,
            "Odds Ratio":  np.exp(coef),
        }).sort_values("Odds Ratio", ascending=True).reset_index(drop=True)

        fig_imp = px.bar(
            imp_df.tail(20),
            x="Odds Ratio", y="Feature",
            orientation="h",
            color="Odds Ratio",
            color_continuous_scale=["#e74c3c", "#f8f9ff", "#2ecc71"],
            color_continuous_midpoint=1.0,
            title="Odds Ratios (top 20 encoded features)",
            labels={"Odds Ratio": "Odds Ratio  (>1 = increases satisfaction)"},
        )
        fig_imp.add_vline(x=1.0, line_dash="dash", line_color="#888",
                          annotation_text="OR = 1  (no effect)",
                          annotation_position="top")
        fig_imp.update_layout(
            height=520,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Odds Ratio > 1 → factor increases odds of satisfaction.  "
            "Odds Ratio < 1 → factor decreases odds of satisfaction."
        )
    except Exception as e:
        st.info(f"Feature importance chart unavailable: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About This Study</h1>
        <p>Maternal satisfaction among vaginal and caesarean section delivery
        care services in Bahir Dar City Health Facilities, Northwest Ethiopia:
        Application of Machine Learning.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### 📌 Study Overview")
        st.markdown("""
- **Dataset:** 894 mothers, Bahir Dar City, Ethiopia
- **Outcome:** Overall delivery care satisfaction
  - 0 = Not Satisfied (n = 345, 38.6%)
  - 1 = Satisfied (n = 549, 61.4%)
- **Algorithm:** Logistic Regression
- **Train/Test split:** 80% / 20% (stratified)
- **Cross-validation:** 5-fold Stratified K-Fold
        """)

        st.markdown("### 🔬 Feature Engineering")
        st.markdown("""
**Variables excluded before modeling:**

| Reason | Count |
|---|---|
| 10 satisfaction sub-item recodes (data leakage) | 10 |
| >50% missing values | 2 |
| Recoded variables (originals kept) | 6 |
| Target variable | 1 |

**Clean candidate features remaining:** 27

**Feature selection method:** 5-fold CV permutation importance (ROC-AUC)

**Final features selected:** Top 10
        """)

    with col_b:
        st.markdown("### 🎯 Model Configuration")
        st.markdown("""
| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Solver | lbfgs |
| Best C (regularisation) | 0.1 |
| class_weight | balanced |
| max_iter | 1000 |
| Tuning method | GridSearchCV |
| Tuning metric | ROC-AUC |
| CV folds | 5 (Stratified) |
        """)

        st.markdown("### 📐 Optimal Threshold")
        st.markdown("""
The decision threshold of **0.65** was selected using the
**Youden J statistic**:

> Youden J = Sensitivity + Specificity − 1

This is maximised at 0.65, giving the best simultaneous balance
between correctly identifying satisfied mothers (Sensitivity)
and correctly identifying not-satisfied mothers (Specificity).

| Method | Threshold | Key Metric |
|---|---|---|
| Youden J | **0.65** | J = 0.693 (best) |
| Max F1 | 0.58 | F1 = 0.882 |
| Default | 0.50 | J = 0.581 |

**0.65 was chosen** because in maternal health, missing a
dissatisfied mother (false negative) is as costly as
misclassifying a satisfied one (false positive).
        """)

    st.markdown("---")
    st.markdown("### 🏆 10 Selected Features")

    feat_cols = st.columns(2)
    for idx, feature in enumerate(FEATURE_ORDER):
        meta = FEATURE_META[feature]
        target_col = feat_cols[idx % 2]
        target_col.markdown(
            f"""<div class="feat-card">
                <strong>{meta['icon']} {meta['label']}</strong><br>
                <span style='font-size:0.82em; color:#666;'>
                    {meta['help']}
                </span><br>
                <span style='font-size:0.78em; color:#999;'>
                    Options: {", ".join(meta["options"].keys())}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption(
        "**Notebook:** Maternal_Satisfaction_ML_Project.ipynb  \n"
        "**Model file:** Maternal_Satisfaction_LR_Final_Model.pkl  \n"
        "**Dataset:** Delivery care satisfaction.sav (SPSS format) — "
        "credit: Getasew Mulat"
    )
