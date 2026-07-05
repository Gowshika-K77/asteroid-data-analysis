import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Asteroid Analysis",
    page_icon="☄️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom styling
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 15% 10%, #10182c 0%, #05070d 55%);
    }

    [data-testid="stSidebar"] {
        background: #0a0e1a;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .app-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f0abfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 1.8rem;
    }

    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(99,102,241,0.25);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(168,85,247,0.35);
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        color: #e2e8f0;
    }

    .result-hazard {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        color: #fecaca;
        font-weight: 600;
        font-size: 1.05rem;
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        border: 1px solid rgba(34,197,94,0.4);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        color: #bbf7d0;
        font-weight: 600;
        font-size: 1.05rem;
    }

    .result-diameter {
        background: linear-gradient(135deg, rgba(129,140,248,0.18), rgba(217,70,239,0.08));
        border: 1px solid rgba(167,139,250,0.4);
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
    }

    .result-diameter .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e9d5ff;
    }

    .result-diameter .label {
        color: #c4b5fd;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7dd3fc;
        border-bottom: 1px solid rgba(125,211,252,0.25);
        padding-bottom: 0.35rem;
        margin-top: 1.4rem;
        margin-bottom: 0.9rem;
    }
    .section-label span.idx {
        color: #a78bfa;
        margin-right: 0.5rem;
    }

    hr {
        border-color: rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load models, scaler, and column references from Hugging Face
# ============================================================
HF_REPO_ID = "GKvin/asteroid-model"


@st.cache_resource
def load_artifacts():
    """Downloads each file from the HF repo once, caches it on disk,
    then loads it into memory. st.cache_resource keeps the loaded
    objects in memory across reruns so this only happens on cold start."""
    pha_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="best_pha_model.pkl")
    diameter_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="best_diameter_model.pkl")
    pha_columns_path = hf_hub_download(repo_id=HF_REPO_ID, filename="pha_columns.pkl")
    diameter_columns_path = hf_hub_download(repo_id=HF_REPO_ID, filename="diameter_columns.pkl")
    class_categories_path = hf_hub_download(repo_id=HF_REPO_ID, filename="class_categories.pkl")

    pha_model = joblib.load(pha_model_path)
    diameter_model = joblib.load(diameter_model_path)
    pha_columns = joblib.load(pha_columns_path)
    diameter_columns = joblib.load(diameter_columns_path)
    class_categories = joblib.load(class_categories_path)
    return pha_model, diameter_model, pha_columns, diameter_columns, class_categories


with st.spinner("Loading models from Hugging Face (first run only, the diameter model is large)..."):
    pha_model, diameter_model, pha_columns, diameter_columns, class_categories = load_artifacts()

st.markdown('<div class="app-title">☄️ Asteroid Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Predict hazard status or estimate diameter from an asteroid\'s orbital elements.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown('<div class="section-label"><span class="idx">—</span>Navigation</div>', unsafe_allow_html=True)
    task = st.radio("Choose a task", ["PHA Classification", "Diameter Prediction"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="section-label"><span class="idx">—</span>About</div>', unsafe_allow_html=True)
    st.caption(
        "PHA model: Decision Tree Classifier\n\n"
        "Diameter model: Random Forest Regressor\n\n"
        "Trained on NASA JPL Small-Body Database orbital elements.\n\n"
        f"Models hosted on [Hugging Face](https://huggingface.co/{HF_REPO_ID})."
    )

# ============================================================
# Shared input fields (orbital elements)
# ============================================================
def section_label(index, title):
    st.markdown(
        f'<div class="section-label"><span class="idx">{index:02d} /</span>{title}</div>',
        unsafe_allow_html=True
    )


def get_orbital_inputs(include_moid=False):
    section_label(1, "Identity & Size")
    c1, c2, c3 = st.columns(3)
    with c1:
        neo = st.selectbox("Near-Earth Object", ["N", "Y"], help="Is this a Near-Earth Object?")
    with c2:
        H = st.number_input("Absolute magnitude (H)", min_value=-5.0, value=15.0, format="%.4f")
    with c3:
        asteroid_class = st.selectbox("Orbit Class", class_categories)

    section_label(2, "Orbital Shape")
    c1, c2, c3 = st.columns(3)
    with c1:
        e = st.number_input("Eccentricity (e)", min_value=0.0, max_value=1.0, value=0.1, format="%.6f")
    with c2:
        a = st.number_input("Semi-major axis, a (AU)", min_value=0.0, value=2.0, format="%.6f")
    with c3:
        q = st.number_input("Perihelion distance, q (AU)", min_value=0.0, value=1.5, format="%.6f")

    c1, c2 = st.columns(2)
    with c1:
        ad = st.number_input("Aphelion distance, ad (AU)", min_value=0.0, value=3.0, format="%.6f")
    with c2:
        if include_moid:
            moid = st.number_input("Earth MOID (AU)", min_value=0.0, value=0.5, format="%.6f")

    section_label(3, "Orientation")
    c1, c2, c3 = st.columns(3)
    with c1:
        i = st.number_input("Inclination, i (deg)", min_value=0.0, max_value=180.0, value=5.0, format="%.6f")
    with c2:
        om = st.number_input("Ascending node, om (deg)", min_value=0.0, max_value=360.0, value=100.0, format="%.6f")
    with c3:
        w = st.number_input("Argument of perihelion, w (deg)", min_value=0.0, max_value=360.0, value=100.0, format="%.6f")

    section_label(4, "Motion & Period")
    c1, c2, c3 = st.columns(3)
    with c1:
        ma = st.number_input("Mean anomaly, ma (deg)", min_value=0.0, max_value=360.0, value=100.0, format="%.6f")
    with c2:
        n = st.number_input("Mean motion, n (deg/day)", min_value=0.0, value=0.2, format="%.6f")
    with c3:
        per_y = st.number_input("Period, per_y (years)", min_value=0.0, value=3.0, format="%.4f")

    per = st.number_input("Orbital period, per (days)", min_value=0.0, value=1000.0, format="%.4f")

    values = {
        'neo': neo, 'e': e, 'a': a, 'q': q, 'i': i, 'om': om, 'w': w,
        'H': H, 'ma': ma, 'ad': ad, 'n': n, 'per': per, 'per_y': per_y,
        'class': asteroid_class
    }
    if include_moid:
        values['moid'] = moid
    return values


def build_feature_row(values, target_columns):
    """Builds a single-row DataFrame matching the training feature columns exactly."""
    row = {}
    row['neo'] = 1 if values['neo'] == 'Y' else 0
    for col in ['e', 'a', 'q', 'i', 'om', 'w', 'H', 'ma', 'ad', 'n', 'per', 'per_y']:
        row[col] = values[col]
    if 'moid' in values:
        row['moid'] = values['moid']

    df_row = pd.DataFrame([row])

    # One-hot encode class the same way as training (drop_first=True on sorted categories)
    dummy_col = f"class_{values['class']}"
    for col in target_columns:
        if col.startswith('class_'):
            df_row[col] = 1 if col == dummy_col else 0

    # Align to exact training column order; fill any missing with 0
    df_row = df_row.reindex(columns=target_columns, fill_value=0)
    return df_row


# ============================================================
# TASK 1: PHA CLASSIFICATION
# ============================================================
if task == "PHA Classification":
    st.subheader("Potentially Hazardous Asteroid Prediction")
    st.caption("Enter the orbital elements below to check hazard classification.")
    values = get_orbital_inputs(include_moid=False)

    st.write("")
    if st.button("Predict PHA Status", use_container_width=True):
        X_input = build_feature_row(values, pha_columns)
        prediction = pha_model.predict(X_input)[0]
        proba = pha_model.predict_proba(X_input)[0] if hasattr(pha_model, "predict_proba") else None

        st.write("")
        if prediction == 1:
            st.markdown(
                '<div class="result-hazard">This asteroid is classified as a <b>Potentially Hazardous Asteroid (PHA)</b>.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-safe">This asteroid is <b>NOT</b> classified as a Potentially Hazardous Asteroid.</div>',
                unsafe_allow_html=True
            )

        if proba is not None:
            st.write("")
            c1, c2 = st.columns(2)
            c1.metric("Not PHA confidence", f"{proba[0]:.1%}")
            c2.metric("PHA confidence", f"{proba[1]:.1%}")

# ============================================================
# TASK 2: DIAMETER REGRESSION
# ============================================================
else:
    st.subheader("Asteroid Diameter Prediction")
    st.caption("Enter the orbital elements below to estimate the asteroid's diameter.")
    values = get_orbital_inputs(include_moid=True)

    st.write("")
    if st.button("Predict Diameter", use_container_width=True):
        X_input = build_feature_row(values, diameter_columns)
        prediction = diameter_model.predict(X_input)[0]

        st.write("")
        st.markdown(
            f'''<div class="result-diameter">
                    <div class="value">{prediction:.3f} km</div>
                    <div class="label">Estimated Diameter</div>
                </div>''',
            unsafe_allow_html=True
        )