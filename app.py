import streamlit as st
import pandas as pd
import joblib

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom Styling
# =============================================================================
st.markdown("""
    <style>
    /* Overall app font size */
    html, body, [class*="css"] {
        font-size: 18px;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
    .result-box h3 {
        font-size: 1.8rem !important;
    }
    .result-box p {
        font-size: 1.3rem !important;
    }

    /* Sidebar labels and headers */
    section[data-testid="stSidebar"] * {
        font-size: 1.1rem !important;
    }

    /* Slider and selectbox labels */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        font-size: 1.15rem !important;
        font-weight: 500;
    }

    /* Tab labels */
    button[data-baseweb="tab"] {
        font-size: 1.2rem !important;
    }

    /* Subheaders */
    h2, h3 {
        font-size: 1.6rem !important;
    }

    /* Caption/footer text */
    .stCaption {
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Model Loading (cached so it only loads once, not on every interaction)
# =============================================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('satisfaction_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError as e:
        st.error(f"Model artifact not found: {e}. Ensure .pkl files are in the app directory.")
        st.stop()

model, scaler, model_columns = load_artifacts()

# =============================================================================
# Header
# =============================================================================
st.markdown('<p class="main-header">✈️ Airline Passenger Satisfaction Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether a passenger will be satisfied based on flight experience and service ratings.</p>', unsafe_allow_html=True)

# =============================================================================
# Sidebar — Passenger & Flight Details
# =============================================================================
with st.sidebar:
    st.header("Passenger & Flight Details")

    age = st.slider("Age", 7, 85, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    flight_distance = st.number_input("Flight Distance (miles)", min_value=0, value=1000, step=50)

    st.divider()
    st.subheader("Delays")
    dep_delay = st.number_input("Departure Delay (min)", min_value=0, value=0)
    arr_delay = st.number_input("Arrival Delay (min)", min_value=0, value=0)

# =============================================================================
# Main Area — Service Ratings (grouped logically, in tabs)
# =============================================================================
st.subheader("Service Ratings")
st.caption("Rate each service from 0 (not applicable) to 5 (excellent)")

tab1, tab2, tab3 = st.tabs(["Booking & Boarding", "In-Flight Experience", "Ground Services"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
        online_booking = st.slider("Ease of Online Booking", 0, 5, 3)
    with c2:
        boarding = st.slider("Online Boarding", 0, 5, 3)
        time_convenient = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
        entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
        food = st.slider("Food and Drink", 0, 5, 3)
    with c2:
        legroom = st.slider("Leg Room Service", 0, 5, 3)
        onboard_service = st.slider("On-board Service", 0, 5, 3)
        cleanliness = st.slider("Cleanliness", 0, 5, 3)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        baggage = st.slider("Baggage Handling", 0, 5, 3)
        checkin = st.slider("Checkin Service", 0, 5, 3)
    with c2:
        inflight_service = st.slider("Inflight Service", 0, 5, 3)
        gate_location = st.slider("Gate Location", 0, 5, 3)

# =============================================================================
# Prediction Logic
# =============================================================================
def build_input_row():
    """Assemble user inputs into a single-row DataFrame matching model_columns."""
    raw = {
        'Age': age,
        'Flight Distance': flight_distance,
        'Inflight wifi service': wifi,
        'Departure/Arrival time convenient': time_convenient,
        'Ease of Online booking': online_booking,
        'Gate location': gate_location,
        'Food and drink': food,
        'Online boarding': boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': entertainment,
        'On-board service': onboard_service,
        'Leg room service': legroom,
        'Baggage handling': baggage,
        'Checkin service': checkin,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Departure Delay in Minutes': dep_delay,
        'Arrival Delay in Minutes': arr_delay,
        'Gender_Male': int(gender == "Male"),
        'Customer Type_disloyal Customer': int(customer_type == "disloyal Customer"),
        'Type of Travel_Personal Travel': int(travel_type == "Personal Travel"),
        'Class_Eco': int(travel_class == "Eco"),
        'Class_Eco Plus': int(travel_class == "Eco Plus"),
    }
    df = pd.DataFrame([raw])
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    return df[model_columns]


st.divider()

if st.button("Predict Satisfaction", type="primary", use_container_width=True):
    input_df = build_input_row()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(f"""
            <div class="result-box" style="background-color:#E8F5E9; border: 1px solid #4CAF50;">
                <h3 style="color:#2E7D32; margin:0;">✅ Satisfied</h3>
                <p style="color:#2E7D32; margin:0;">Confidence: {probability*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box" style="background-color:#FDECEA; border: 1px solid #F44336;">
                <h3 style="color:#C62828; margin:0;">❌ Neutral or Dissatisfied</h3>
                <p style="color:#C62828; margin:0;">Confidence: {(1-probability)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("View raw model input"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption("Model: Tuned LightGBM · Test Accuracy: 96.4% · ROC-AUC: 0.995")