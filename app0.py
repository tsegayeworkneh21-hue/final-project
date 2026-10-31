"""
Maternal Satisfaction Predictor
===============================
Maternal satisfaction among vaginal and caesarean section delivery care services in
Bahir Dar City Health Facilities, Northwest Ethiopia: application of machine learning.

Model file : Maternal_Satisfaction_LR_Final_Model.pkl  (saved in Step 12 of the notebook)
Run        : streamlit run app.py

Everything that comes from the notebook (metrics, threshold table, study counts) is kept in
the "Reported results" block below. After re-running the notebook, update only that block.
"""

from __future__ import annotations

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

NAVY, BLUE, GREEN, AMBER, RED = "#0F3460", "#3A6EA5", "#2E8B57", "#C98A00", "#C0392B"

st.set_page_config(
    page_title="Maternal Satisfaction Predictor",
    page_icon="🤱",
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

# =============================================================================
# Styling
# =============================================================================
st.markdown(f"""
<style>
.hero {{ background: linear-gradient(135deg, #0B2545 0%, {NAVY} 100%); border-radius: 14px;
         padding: 30px 36px; margin-bottom: 24px; color: #fff; }}
.hero h1 {{ font-size: 2rem; margin: 0 0 8px 0; color: #fff; }}
.hero p  {{ font-size: 1.02rem; margin: 0; color: #D6E2F0; max-width: 70ch; }}
.section-title {{ font-size: 1.2rem; font-weight: 700; margin: 26px 0 12px 0;
                  padding-bottom: 6px; border-bottom: 2px solid {BLUE}; }}
.result {{ border-radius: 12px; padding: 22px 26px; margin: 14px 0; color: #1B2A41; }}
.result h2 {{ margin: 0 0 6px 0; font-size: 1.6rem; color: inherit; }}
.result p  {{ margin: 0; font-size: 1.02rem; }}
.result.ok  {{ background: #E3F4EA; border-left: 6px solid {GREEN}; }}
.result.low {{ background: #FBEFD5; border-left: 6px solid {AMBER}; }}
</style>
""", unsafe_allow_html=True)


def hero(title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{text}</p></div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


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


# =============================================================================
# Pages
# =============================================================================
def page_predict(model) -> None:
    hero("Maternal Satisfaction Predictor",
         "Estimate whether a mother is likely to be satisfied with her delivery care, using a logistic "
         f"regression model trained on {STUDY['n_total']} mothers from Bahir Dar City health facilities.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision threshold", f"{THRESHOLD:.2f}", help="Predicted satisfied when probability ≥ threshold.")
    c2.metric("Sensitivity", f"{METRICS['Sensitivity']:.1%}")
    c3.metric("Specificity", f"{METRICS['Specificity']:.1%}")
    c4.metric("F1 score", f"{METRICS['F1 score']:.1%}")

    section("Enter the mother's details")
    st.caption(f"Answer all {len(FEATURES)} questions. Service-rating items were excluded from the model "
               "because they are part of the satisfaction outcome itself.")

    answers: dict[str, float] = {}
    with st.form("predict_form"):
        columns = st.columns(2, gap="large")
        for i, feature in enumerate(FEATURES):
            with columns[i % 2]:
                choice = st.selectbox(feature.label, list(feature.options), help=feature.help, key=feature.code)
                answers[feature.code] = feature.options[choice]
        submitted = st.form_submit_button("▶  Predict Satisfaction", type="primary",use_container_width=True)

    if not submitted:
        return

    row = pd.DataFrame([answers])[FEATURE_ORDER]
    p_satisfied = float(model.predict_proba(row)[0][1])
    satisfied = p_satisfied >= THRESHOLD

    section("Result")
    if satisfied:
        st.markdown(
            f'<div class="result ok"><h2>Likely satisfied</h2><p>Predicted probability of satisfaction is '
            f'<b>{p_satisfied:.1%}</b>, at or above the threshold of {THRESHOLD:.2f}.</p></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result low"><h2>Likely not satisfied</h2><p>Predicted probability of satisfaction is '
            f'<b>{p_satisfied:.1%}</b>, below the threshold of {THRESHOLD:.2f}. '
            f'Facility conditions and staff conduct are worth reviewing.</p></div>',
            unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_satisfied * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Probability of satisfaction"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": GREEN if satisfied else AMBER, "thickness": 0.3},
            "steps": [{"range": [0, THRESHOLD * 100], "color": "rgba(201,138,0,0.15)"},
                      {"range": [THRESHOLD * 100, 100], "color": "rgba(46,139,87,0.15)"}],
            "threshold": {"line": {"color": NAVY, "width": 4}, "thickness": 0.85, "value": THRESHOLD * 100},
        },
    ))
    gauge.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gauge, width="stretch")
    st.caption(f"The dark line marks the decision threshold ({THRESHOLD:.0%}).")

    with st.expander("Input summary"):
        summary = pd.DataFrame({
            "Question": [f.label for f in FEATURES],
            "Answer": [next(k for k, v in f.options.items() if v == answers[f.code]) for f in FEATURES],
        })
        st.dataframe(summary, width="stretch", hide_index=True)
    st.caption("Decision-support estimate for research and quality improvement. It is not a clinical diagnosis.")


def page_performance(model) -> None:
    hero("Model performance",
         f"Evaluation on the held-out test set ({STUDY['n_test']} mothers, 20% of {STUDY['n_total']}).")

    section(f"Performance at threshold {THRESHOLD:.2f}")
    top = st.columns(4)
    for col, name in zip(top, ["Accuracy", "Sensitivity", "Specificity", "Precision"]):
        col.metric(name, f"{METRICS[name]:.1%}")
    bottom = st.columns(4)
    bottom[0].metric("F1 score", f"{METRICS['F1 score']:.1%}")
    bottom[1].metric("AUC-ROC", f"{AUC_ROC:.3f}", help="Approximate value entered by hand; update from the notebook.")
    bottom[2].metric("Brier score", f"{BRIER:.4f}", help="Lower is better (0 = perfect).")

    section("Confusion matrix")
    labels = ["Not satisfied", "Satisfied"]
    matrix = go.Figure(go.Heatmap(
        z=[[TN, FP], [FN, TP]],
        x=[f"Predicted {l.lower()}" for l in labels],
        y=[f"Actual {l.lower()}" for l in labels],
        text=[[f"TN = {TN}", f"FP = {FP}"], [f"FN = {FN}", f"TP = {TP}"]],
        texttemplate="<b>%{z}</b><br>%{text}", textfont={"size": 15},
        colorscale="Blues", showscale=False,
    ))
    matrix.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis={"autorange": "reversed"})
    left, right = st.columns([1.2, 1])
    left.plotly_chart(matrix, width="stretch")
    right.markdown(
        "| Term | Count | Meaning |\n|---|---|---|\n"
        f"| TN | {TN} | Not satisfied, correctly identified |\n"
        f"| FP | {FP} | Not satisfied, wrongly called satisfied |\n"
        f"| FN | {FN} | Satisfied, wrongly called not satisfied |\n"
        f"| TP | {TP} | Satisfied, correctly identified |")

    section("Metrics across thresholds")
    styles = {"Sensitivity": ("#2E8B57", "solid"), "Specificity": ("#C0392B", "solid"),
              "F1 score": ("#3A6EA5", "dash"), "Accuracy": ("#7D4E9E", "dot"), "Youden J": ("#C98A00", "dashdot")}
    lines = go.Figure()
    for metric, (color, dash) in styles.items():
        lines.add_trace(go.Scatter(x=THRESHOLD_TABLE["Threshold"], y=THRESHOLD_TABLE[metric], name=metric,
                                   mode="lines+markers", line=dict(color=color, width=2.5, dash=dash)))
    lines.add_vline(x=THRESHOLD, line_dash="dash", line_color=NAVY, line_width=2,
                    annotation_text=f"Threshold {THRESHOLD:.2f}", annotation_position="top right")
    lines.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0.3, 1.05],
                        legend=dict(orientation="h", y=-0.25), height=420, hovermode="x unified",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(lines, width="stretch")
    with st.expander("Threshold table"):
        st.dataframe(THRESHOLD_TABLE, width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in THRESHOLD_TABLE.columns if c != "Threshold"})

    section("Odds ratios")
    try:
        table = odds_ratio_table(model)
        table["Log distance"] = np.abs(np.log(table["Odds ratio"]))
        table = table.nlargest(15, "Log distance").sort_values("Odds ratio")
        table["Effect"] = np.where(table["Odds ratio"] >= 1, "Raises odds of satisfaction", "Lowers odds of satisfaction")
        chart = px.bar(table, x="Odds ratio", y="Category", orientation="h", color="Effect",
                       color_discrete_map={"Raises odds of satisfaction": GREEN, "Lowers odds of satisfaction": RED})
        chart.add_vline(x=1.0, line_dash="dash", line_color="#888")
        chart.update_layout(height=520, legend_title_text="", legend=dict(orientation="h", y=-0.15),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart, width="stretch")
        st.caption("The 15 strongest categories. Each odds ratio is relative to the first option of its question "
                   "(for example Private for facility ownership). Above 1 raises the odds of satisfaction.")
    except Exception as exc:
        st.info(f"Odds ratios are not available for this model file: {exc}")


def page_about(model) -> None:
    hero("About this study",
         "Maternal satisfaction among vaginal and caesarean section delivery care services in Bahir Dar City "
         "Health Facilities, Northwest Ethiopia: application of machine learning.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Study overview")
        ns, s = STUDY["n_not_satisfied"], STUDY["n_satisfied"]
        st.markdown(
            f"- **Participants:** {STUDY['n_total']} mothers\n"
            f"- **Outcome:** overall satisfaction with delivery care\n"
            f"  - Not satisfied: {ns} ({ns / STUDY['n_total']:.1%})\n"
            f"  - Satisfied: {s} ({s / STUDY['n_total']:.1%})\n"
            "- **Train / test split:** 80% / 20%, stratified\n"
            "- **Cross-validation:** 5-fold stratified")
        st.subheader("Variable selection")
        st.markdown(
            "Excluded before modelling:\n\n"
            "| Reason | Variables |\n|---|---|\n"
            "| Satisfaction sub-items (data leakage) | 10 |\n"
            "| More than 50% missing values | 2 |\n"
            "| Recoded duplicates of kept variables | 6 |\n\n"
            f"{STUDY['n_candidates']} candidate features remained. The final {len(FEATURES)} were chosen by "
            "5-fold cross-validated permutation importance (ROC-AUC).")
    with right:
        st.subheader("Model configuration")
        clf = model.named_steps["classifier"]
        st.markdown(
            "| Parameter | Value |\n|---|---|\n"
            "| Algorithm | Logistic regression |\n"
            f"| Regularisation C | {clf.C} |\n"
            f"| Solver | {clf.solver} |\n"
            f"| Class weight | {clf.class_weight} |\n"
            f"| Max iterations | {clf.max_iter} |\n"
            "| Tuning | GridSearchCV, ROC-AUC, 5 folds |")
        st.subheader("Decision threshold")
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

    section(f"The {len(FEATURES)} model inputs")
    cols = st.columns(2)
    for i, feature in enumerate(FEATURES):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{feature.label}**")
            st.caption(feature.help)
            st.caption("Options: " + ", ".join(feature.options))
    st.caption("Notebook: Maternal_Satisfaction_ML_Project.ipynb · Dataset: Delivery care satisfaction.sav · "
               "Credit: Getasew Mulat")


# =============================================================================
# App
# =============================================================================
with st.sidebar:
    st.markdown("### Maternal Satisfaction")
    st.caption("Bahir Dar City Health Facilities, Northwest Ethiopia")
    page = st.radio("Section", ["Predict", "Model performance", "About"], label_visibility="collapsed")
    st.divider()
    st.caption(f"Logistic regression · {len(FEATURES)} inputs · threshold {THRESHOLD:.2f}")
    st.caption(f"Sensitivity {METRICS['Sensitivity']:.1%} · Specificity {METRICS['Specificity']:.1%}")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Run the notebook through Step 12 and place the "
             "file in the same folder as `app.py`.")
    st.stop()

model = load_model(MODEL_PATH)
check_model_inputs(model)

{"Predict": page_predict, "Model performance": page_performance, "About": page_about}[page](model)
