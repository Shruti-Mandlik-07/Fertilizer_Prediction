import os
import html

import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Terra — Fertilizer Recommendation System
# Powered by Random Forest Classifier trained on 3,100 agricultural records
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Terra | Fertilizer Guide",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fertilizer_model.pkl")

SOIL_OPTIONS = [
    "Acidic Soil",
    "Alkaline Soil",
    "Loamy Soil",
    "Neutral Soil",
    "Peaty Soil"
]

CROP_OPTIONS = [
    "Adzuki Beans", "Black gram", "Chickpea", "Coconut", "Coffee",
    "Cotton", "Ground Nut", "Jute", "Kidney Beans", "Lentil",
    "Moth Beans", "Mung Bean", "Peas", "Pigeon Peas", "Rubber",
    "Sugarcane", "Tea", "Tobacco", "apple", "banana", "grapes",
    "maize", "mango", "millet", "muskmelon", "orange", "papaya",
    "pomegranate", "rice", "watermelon", "wheat"
]

FERTILIZER_REMARKS = {
    "Balanced NPK Fertilizer": "Provides a balanced mix of nitrogen, phosphorus, and potassium for loamy soils. Recommended for general-purpose fertilization in well-structured soils.",
    "Compost": "Enhances organic matter and improves soil structure. Recommended for low-carbon soils to naturally boost microbial and soil health.",
    "DAP": "Rich in phosphorus, essential for root development and early plant vigor. Recommended for phosphorus-deficient soils to improve crop establishment.",
    "General Purpose Fertilizer": "Suitable for general use across diverse soil and crop types when no specific acute nutrient deficiency is present.",
    "Gypsum": "Corrects alkaline soil and adds calcium and sulfur. Recommended to neutralize high soil pH and improve soil permeability.",
    "Lime": "Neutralizes acidic soil and improves pH balance. Corrects low soil pH, unlocking critical nutrient availability for root uptake.",
    "Muriate of Potash": "High potassium content, improving disease resistance, water regulation, and fruit/flower quality.",
    "Organic Fertilizer": "Enhances soil fertility naturally, ideal for peaty and organic-rich soils to support sustainable organic farming.",
    "Urea": "Provides high nitrogen (46% N), ideal for rapid vegetative leafy growth and quickly correcting nitrogen deficiency.",
    "Water Retaining Fertilizer": "Improves water retention in dry or coarse soils. Recommended to prevent moisture stress and preserve root hydration."
}

# Change this URL if you want to use a locally saved banner image instead.
HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef"
    "?auto=format&fit=crop&w=1800&q=85"
)


@st.cache_resource
def load_model(path: str):
    """Load the trained sklearn pipeline once per app session."""
    return joblib.load(path)


if not os.path.isfile(MODEL_PATH):
    st.error(
        "Model file not found. Put **fertilizer_model.pkl** in the same folder as **app.py**."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info(
        "Use a scikit-learn version compatible with the one used when the model was saved. "
        "For this project, try `scikit-learn==1.6.1`."
    )
    st.stop()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --forest: #174734;
      --forest-dark: #103728;
      --leaf: #8cad55;
      --cream: #f5f5ed;
      --paper: #fffef9;
      --ink: #20352a;
      --muted: #607267;
      --line: #dfe7d9;
    }

    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      color: var(--ink);
    }

    .stApp {
      background:
        radial-gradient(circle at 95% 3%, rgba(140,173,85,.15), transparent 24rem),
        linear-gradient(180deg, #f8f8f1 0%, #f0f4eb 100%);
    }

    .block-container {
      max-width: 1240px;
      padding-top: 1.2rem;
      padding-bottom: 2.8rem;
    }

    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* Strong, readable default text colors */
    .stApp, .stApp p, .stApp label, .stApp legend,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li {
      color: var(--ink);
    }

    .brandline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: .25rem 0 1.1rem;
      margin-bottom: 1.35rem;
      border-bottom: 1px solid var(--line);
    }

    .brand {
      font-family: 'Manrope', sans-serif;
      color: var(--forest);
      font-size: 1.7rem;
      font-weight: 800;
      letter-spacing: -1.2px;
    }

    .brand-dot { color: var(--leaf); }
    .brand-caption {
      color: #65796a;
      font-size: .72rem;
      letter-spacing: .12em;
      text-transform: uppercase;
      font-weight: 700;
    }

    .hero {
      background: linear-gradient(120deg, rgba(13,48,34,.96), rgba(31,83,55,.88));
      border-radius: 26px;
      overflow: hidden;
      padding: 2.4rem;
      color: #fffdf5;
      margin-bottom: 1.7rem;
      box-shadow: 0 16px 38px rgba(23,71,52,.12);
    }

    .hero-kicker {
      display: inline-block;
      color: #e6f0d7;
      background: rgba(255,255,255,.13);
      border: 1px solid rgba(255,255,255,.18);
      padding: .4rem .75rem;
      border-radius: 999px;
      font-size: .75rem;
      font-weight: 700;
      letter-spacing: .09em;
      text-transform: uppercase;
    }

    .hero h1 {
      color: #fffdf5 !important;
      font-family: 'Manrope', sans-serif;
      font-size: clamp(2.2rem, 4vw, 3.6rem);
      font-weight: 800;
      letter-spacing: -2px;
      line-height: 1.08;
      margin: 1rem 0 .8rem;
    }

    .hero p {
      color: #e0eadf !important;
      font-size: 1.03rem;
      line-height: 1.75;
      max-width: 650px;
      margin-bottom: 0;
    }

    .hero-foot {
      color: #d9e8cf;
      margin-top: 1.2rem;
      font-size: .88rem;
    }

    .section-title {
      color: var(--forest) !important;
      font-family: 'Manrope', sans-serif;
      font-size: 1.5rem;
      font-weight: 800;
      letter-spacing: -.6px;
      margin: .25rem 0 .25rem;
    }

    .section-sub {
      color: var(--muted) !important;
      line-height: 1.65;
      margin-bottom: 1rem;
    }

    .stApp [data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(255,254,249,.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 1.15rem 1.25rem;
      box-shadow: 0 8px 24px rgba(29,59,40,.035);
    }

    div[data-testid="stForm"] {
      border: 0;
      padding: 0;
      background: transparent;
    }

    .stApp [data-testid="stWidgetLabel"] p,
    .stApp label,
    .stApp legend {
      color: #355746 !important;
      font-weight: 600 !important;
    }

    .stApp input, .stApp textarea,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
      color: #20352a !important;
      background-color: #fffefb !important;
      border-color: #d6e0d1 !important;
      border-radius: 10px !important;
    }

    .stApp [data-baseweb="select"] span {
      color: #20352a !important;
    }

    .stButton > button, .stFormSubmitButton > button {
      border-radius: 12px;
      border: 1px solid var(--forest);
      background: var(--forest);
      color: #ffffff !important;
      font-weight: 700;
      min-height: 3.15rem;
      transition: all .2s ease;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
      background: #286246;
      border-color: #286246;
      color: #ffffff !important;
      transform: translateY(-1px);
    }

    .mini-heading {
      color: var(--forest) !important;
      font-size: 1rem;
      font-weight: 800;
      margin: .25rem 0 .7rem;
    }

    .snapshot {
      background: #fffef9;
      border: 1px solid var(--line);
      border-radius: 15px;
      padding: 1rem 1.05rem;
      margin-bottom: .8rem;
    }

    .snapshot-label {
      color: #66786b !important;
      font-size: .72rem;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin-bottom: .35rem;
    }

    .snapshot-value {
      color: var(--forest) !important;
      font-family: 'Manrope', sans-serif;
      font-size: 1.25rem;
      font-weight: 800;
      overflow-wrap: anywhere;
    }

    .result-card {
      background: linear-gradient(135deg, #eaf2df, #f8faef);
      border: 1px solid #d2e1c1;
      border-radius: 20px;
      padding: 1.4rem 1.5rem;
      margin: .6rem 0 1rem;
    }

    .result-label {
      color: #526d40 !important;
      font-size: .75rem;
      font-weight: 800;
      letter-spacing: .1em;
      text-transform: uppercase;
    }

    .result-name {
      color: #174734 !important;
      font-family: 'Manrope', sans-serif;
      font-size: clamp(1.7rem, 3vw, 2.35rem);
      font-weight: 800;
      margin: .25rem 0 .45rem;
      overflow-wrap: anywhere;
    }

    .result-copy {
      color: #294c34 !important;
      font-size: 1.02rem;
      line-height: 1.65;
      background: rgba(255,255,255,0.7);
      padding: 0.8rem 1rem;
      border-radius: 10px;
      border-left: 4px solid var(--leaf);
      margin-top: 0.8rem;
    }

    .footer {
      margin-top: 2rem;
      border-top: 1px solid var(--line);
      padding-top: 1.2rem;
      color: #68786d !important;
      font-size: .82rem;
      line-height: 1.7;
    }

    [data-testid="stAlert"] {
      border-radius: 13px;
    }

    @media (max-width: 700px) {
      .block-container { padding-left: 1rem; padding-right: 1rem; }
      .hero { padding: 1.5rem; border-radius: 20px; }
      .hero h1 { letter-spacing: -1px; }
      .brand-caption { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Brand header
st.markdown(
    """
    <div class="brandline">
      <div class="brand">terra<span class="brand-dot">.</span></div>
      <div class="brand-caption">Field notes for better growing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero section with landscape photo
hero_left, hero_right = st.columns([1.35, 0.85], gap="large")
with hero_left:
    st.markdown(
        """
        <div class="hero">
          <span class="hero-kicker">A smarter starting point for your field</span>
          <h1>Good soil.<br>Thoughtful choices.</h1>
          <p>
            Explore a fertilizer recommendation tailored to your crop, soil properties,
            NPK nutrients, pH, moisture, and rainfall conditions.
          </p>
          <div class="hero-foot">✳ &nbsp; Powered by a machine-learning pipeline trained on 3,100 field observations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    st.image(
        HERO_IMAGE_URL,
        caption="Healthy fields begin with informed decisions.",
        use_container_width=True,
    )

st.markdown(
    '<div class="section-title">Let’s understand your field</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-sub">Enter your readings below. Use values from a recent soil test or local agro-climatic readings.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    with st.form("field_form"):
        st.markdown('<div class="mini-heading">01 &nbsp; Soil Nutrients (NPK)</div>', unsafe_allow_html=True)
        n1, n2, n3 = st.columns(3)
        with n1:
            nitrogen = st.number_input(
                "Nitrogen (N)", min_value=0.0, max_value=200.0, value=60.0, step=1.0,
                help="Nitrogen content in soil."
            )
        with n2:
            phosphorous = st.number_input(
                "Phosphorous (P)", min_value=0.0, max_value=250.0, value=70.0, step=1.0,
                help="Phosphorous content in soil."
            )
        with n3:
            potassium = st.number_input(
                "Potassium (K)", min_value=0.0, max_value=250.0, value=70.0, step=1.0,
                help="Potassium content in soil."
            )

        st.markdown("---")
        st.markdown('<div class="mini-heading">02 &nbsp; Climate & Soil Chemistry</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temperature = st.number_input(
                "Temperature (°C)", min_value=0.0, max_value=60.0,
                value=26.0, step=0.5, format="%.1f"
            )
        with c2:
            moisture = st.number_input(
                "Soil Moisture", min_value=0.0, max_value=1.0,
                value=0.60, step=0.05, format="%.2f",
                help="Soil moisture saturation ratio (e.g., 0.60 for 60% saturation)."
            )
        with c3:
            rainfall = st.number_input(
                "Rainfall (mm)", min_value=0.0, max_value=500.0,
                value=110.0, step=5.0, format="%.1f",
                help="Seasonal or monthly average precipitation in mm."
            )
        with c4:
            ph = st.number_input(
                "Soil pH", min_value=2.0, max_value=12.0,
                value=6.5, step=0.1, format="%.1f",
                help="Soil acidity/alkalinity level (6.0 - 7.5 is neutral)."
            )

        st.markdown("---")
        st.markdown('<div class="mini-heading">03 &nbsp; Soil Classification & Target Crop</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1:
            soil = st.selectbox("Soil type", SOIL_OPTIONS, index=SOIL_OPTIONS.index("Loamy Soil"))
        with f2:
            crop = st.selectbox("Crop type", CROP_OPTIONS, index=CROP_OPTIONS.index("wheat"))

        submitted = st.form_submit_button(
            "Find my fertilizer suggestion  →",
            use_container_width=True,
            type="primary",
        )

# Snapshot area
st.markdown('<div class="section-title">Your field snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">A quick overview of the selected crop and measurements.</div>',
    unsafe_allow_html=True,
)

snap1, snap2, snap3, snap4 = st.columns(4)
with snap1:
    st.markdown(
        f'<div class="snapshot"><div class="snapshot-label">Crop</div>'
        f'<div class="snapshot-value">{html.escape(crop)}</div></div>',
        unsafe_allow_html=True,
    )
with snap2:
    st.markdown(
        f'<div class="snapshot"><div class="snapshot-label">Soil & pH</div>'
        f'<div class="snapshot-value">{html.escape(soil)} (pH {ph:g})</div></div>',
        unsafe_allow_html=True,
    )
with snap3:
    st.markdown(
        f'<div class="snapshot"><div class="snapshot-label">N · P · K</div>'
        f'<div class="snapshot-value">{nitrogen:g} · {phosphorous:g} · {potassium:g}</div></div>',
        unsafe_allow_html=True,
    )
with snap4:
    st.markdown(
        f'<div class="snapshot"><div class="snapshot-label">Temp · Rain · Moist</div>'
        f'<div class="snapshot-value">{temperature:g}°C · {rainfall:g}mm · {moisture*100:.0f}%</div></div>',
        unsafe_allow_html=True,
    )

if submitted:
    input_df = pd.DataFrame(
        [{
            "Nitrogen": float(nitrogen),
            "Phosphorous": float(phosphorous),
            "Potassium": float(potassium),
            "Temperature": float(temperature),
            "Moisture": float(moisture),
            "Rainfall": float(rainfall),
            "PH": float(ph),
            "Soil": soil,
            "Crop": crop,
        }]
    )

    try:
        prediction = str(model.predict(input_df)[0])
        remark_text = FERTILIZER_REMARKS.get(
            prediction,
            "Formulated for optimal nutrient replenishment given current soil and environmental parameters."
        )

        st.markdown('<div class="section-title">Your fertilizer suggestion</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Dataset-based model recommendation</div>
              <div class="result-name">{html.escape(prediction)}</div>
              <div class="result-copy">
                <b>Agronomic Insight:</b> {html.escape(remark_text)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show top candidate matches if model supports probability outputs
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(input_df)[0]
                classes = getattr(model, "classes_", [])
                ranked = sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
                if ranked:
                    st.markdown("#### Candidate Match Probabilities")
                    for label, score in ranked:
                        st.write(f"**{label}** — model score: {score:.1%}")
                        st.progress(float(score))
                    st.caption(
                        "Model scores reflect the probability distribution computed across 100 decision trees."
                    )
            except Exception:
                pass

        st.warning(
            "Safety note: This application predicts fertilizer suggestions based on 3,100 empirical observations. "
            "Always corroborate with an official laboratory soil test and local agricultural extension guidance before bulk field application."
        )
    except Exception as exc:
        st.error(f"Could not generate a prediction: {exc}")
        st.caption(
            "Confirm that fertilizer_model.pkl is trained with features: "
            "['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Moisture', 'Rainfall', 'PH', 'Soil', 'Crop']."
        )

st.markdown(
    """
    <div class="footer">
      <b>terra.</b> &nbsp; Fertilizer Recommendation System powered by Machine Learning.<br>
      Predictions reflect patterns from agricultural field datasets and should be paired with local agronomic consultation.
    </div>
    """,
    unsafe_allow_html=True,
)
