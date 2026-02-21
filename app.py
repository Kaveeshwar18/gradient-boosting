import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------
# COLORFUL BACKGROUND
# -----------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    color: white;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #f1f1f1;
    text-align: center;
}

.stButton>button {
    background-color: #ff6a00;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}

.stButton>button:hover {
    background-color: #ff3c00;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown('<div class="big-title">🩺 Diabetes Progression Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Gradient Boosting Regression Model</div>', unsafe_allow_html=True)
st.write("")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_gradient_boosting_model.pkl")
    return model

model = load_model()

# -----------------------------------
# INPUT SECTION (CORRECT FEATURES)
# -----------------------------------
st.markdown("## 📋 Enter Clinical Values")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Standardized)", -1.0, 1.0, 0.0)
    sex = st.number_input("Sex (Standardized)", -1.0, 1.0, 0.0)
    bmi = st.number_input("BMI (Standardized)", -1.0, 1.0, 0.0)
    bp = st.number_input("Blood Pressure (Standardized)", -1.0, 1.0, 0.0)
    s1 = st.number_input("S1 (Standardized)", -1.0, 1.0, 0.0)

with col2:
    s2 = st.number_input("S2 (Standardized)", -1.0, 1.0, 0.0)
    s3 = st.number_input("S3 (Standardized)", -1.0, 1.0, 0.0)
    s4 = st.number_input("S4 (Standardized)", -1.0, 1.0, 0.0)
    s5 = st.number_input("S5 (Standardized)", -1.0, 1.0, 0.0)
    s6 = st.number_input("S6 (Standardized)", -1.0, 1.0, 0.0)

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🚀 Predict Disease Progression"):

    try:
        input_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "bp": bp,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6
        }])

        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div style="background-color:#2ecc71;
                    padding:20px;
                    border-radius:15px;
                    text-align:center;
                    font-size:22px;
                    color:white;">
            📊 Predicted Disease Progression Score: {prediction:.2f}
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(prediction), 100))

    except Exception as e:
        st.error(f"Prediction Error: {e}")