import base64
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = BASE_DIR / "assets" / "background.png"
MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def load_artifacts():
    """Load the trained XGBoost model and saved encoders once."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


@st.cache_data
def get_background_image():
    """Encode the background image for CSS."""
    if not BACKGROUND_PATH.exists():
        return None

    with open(BACKGROUND_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_input_dataframe(
    price,
    discount,
    inventory_level,
    promotion,
    competitor_pricing,
    category,
):
    """Create the exact feature structure expected by the model."""
    return pd.DataFrame(
        {
            "Price": [price],
            "Discount": [discount],
            "Inventory Level": [inventory_level],
            "Promotion": [promotion],
            "Competitor Pricing": [competitor_pricing],
            "Category": [category],
        }
    )


def encode_features(input_data, encoders):
    """Apply the same saved categorical encoders used during training."""
    encoded = input_data.copy()

    for col, encoder in encoders.items():
        if col in encoded.columns:
            encoded[col] = encoder.transform(encoded[col])

    return encoded


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model, label_encoders = load_artifacts()
except Exception as e:
    st.error("Unable to load the trained model or encoder.")
    st.exception(e)
    st.stop()


# ============================================================
# BACKGROUND + GLOBAL CSS
# ============================================================

background_image = get_background_image()

background_css = ""
if background_image:
    background_css = f"""
    .stApp {{
        background-image:
            linear-gradient(
                rgba(8, 15, 30, 0.92),
                rgba(8, 15, 30, 0.92)
            ),
            url("data:image/png;base64,{background_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """

st.markdown(
    f"""
    <style>
    {background_css}

    /* ---------- Main layout ---------- */

    .block-container {{
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* ---------- Typography ---------- */

    h1, h2, h3 {{
        letter-spacing: -0.02em;
    }}

    .hero-title {{
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}

    .hero-subtitle {{
        color: #b9c3d4;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }}

    /* ---------- Cards ---------- */

    .info-card {{
        background: rgba(20, 30, 48, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        min-height: 120px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }}

    .card-label {{
        color: #9eabc0;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }}

    .card-value {{
        color: white;
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }}

    .card-text {{
        color: #c3ccda;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }}

    /* ---------- Input section ---------- */

    .section-title {{
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }}

    .section-caption {{
        color: #9eabc0;
        margin-bottom: 1rem;
    }}

    /* ---------- Result ---------- */

    .result-card {{
        background: linear-gradient(
            135deg,
            rgba(0, 100, 150, 0.90),
            rgba(20, 50, 90, 0.90)
        );
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 22px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.30);
        margin-top: 1rem;
    }}

    .result-label {{
        color: #d7e5f5;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.10em;
    }}

    .result-number {{
        color: white;
        font-size: 3.4rem;
        font-weight: 850;
        line-height: 1.1;
        margin: 0.35rem 0;
    }}

    .result-note {{
        color: #c9d8e8;
        font-size: 0.9rem;
    }}

    /* ---------- Buttons ---------- */

    .stButton > button {{
        width: 100%;
        min-height: 3.2rem;
        border-radius: 12px;
        font-weight: 750;
        font-size: 1rem;
        border: 1px solid rgba(255,255,255,0.16);
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }}

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {{
        background: rgba(10, 18, 32, 0.96);
    }}

    .sidebar-title {{
        font-size: 1.2rem;
        font-weight: 800;
    }}

    .sidebar-text {{
        color: #aeb9ca;
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    /* ---------- Footer ---------- */

    .footer {{
        text-align: center;
        color: #7f8da3;
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">📈 Demand Forecasting</div>', unsafe_allow_html=True)
    st.caption("Machine Learning Prediction System")

    st.divider()

    st.markdown("**Model**")
    st.markdown('<div class="sidebar-text">XGBoost Regressor</div>', unsafe_allow_html=True)

    st.markdown("**Prediction Type**")
    st.markdown('<div class="sidebar-text">Product demand in units</div>', unsafe_allow_html=True)

    st.markdown("**Input Features**")
    st.markdown(
        '<div class="sidebar-text">Price · Discount · Inventory · Promotion · Competitor Price · Category</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.info(
        "Enter the product conditions and click Predict Demand to generate a forecast."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-title">📈 Demand Forecasting System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">'
    "Predict expected product demand using a trained machine learning model."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# QUICK INFO CARDS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-label">Algorithm</div>
            <div class="card-value">XGBoost</div>
            <div class="card-text">Gradient boosting regression model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-label">Output</div>
            <div class="card-value">Demand Units</div>
            <div class="card-text">Estimated product demand</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-label">Inference</div>
            <div class="card-value">Real-Time</div>
            <div class="card-text">Prediction through Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown('<div class="section-title">📋 Product Information</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Provide the current product and market conditions.</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    price = st.number_input(
        "Price",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="Current selling price of the product.",
    )

    discount = st.number_input(
        "Discount (%)",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        help="Current discount percentage.",
    )

    inventory_level = st.number_input(
        "Inventory Level",
        min_value=0,
        value=100,
        step=1,
        help="Current number of units available in inventory.",
    )

with col2:
    promotion = st.selectbox(
        "Promotion",
        options=[0, 1],
        format_func=lambda x: "Active" if x == 1 else "No Promotion",
        help="Whether the product is currently under promotion.",
    )

    competitor_pricing = st.number_input(
        "Competitor Price",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="Current competitor selling price.",
    )

    category = st.selectbox(
        "Category",
        label_encoders["Category"].classes_.tolist(),
        help="Product category used by the trained model.",
    )


# ============================================================
# PREDICT
# ============================================================

st.write("")

if st.button("🔮  Predict Demand", type="primary", use_container_width=True):
    try:
        input_data = build_input_dataframe(
            price=price,
            discount=discount,
            inventory_level=inventory_level,
            promotion=promotion,
            competitor_pricing=competitor_pricing,
            category=category,
        )

        encoded_input = encode_features(input_data, label_encoders)

        prediction = model.predict(encoded_input)[0]
        prediction = max(0, int(round(prediction)))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">FORECASTED DEMAND</div>
                <div class="result-number">{prediction:,} Units</div>
                <div class="result-note">
                    Estimated demand based on the selected product and market conditions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success("Demand forecast generated successfully.")

        with st.expander("🔍 View prediction inputs"):
            display_data = input_data.copy()
            display_data["Promotion"] = display_data["Promotion"].map(
                {0: "No Promotion", 1: "Active"}
            )
            st.dataframe(display_data, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Prediction failed. Please verify the input values and model artifacts.")
        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Demand Forecasting System · Machine Learning · XGBoost · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
