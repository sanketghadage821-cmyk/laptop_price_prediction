import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Decision Tree Predictor",
    page_icon="🌳",
    layout="centered"
)

# --- Custom Styling (Modern Dark / Violet Theme) ---
st.markdown("""
    <style>
    /* Global Container Adjustments */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Title Styling */
    .title-text {
        font-size: 2.25rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        font-size: 1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Form Card Container */
    .css-1r6slb0, .stForm {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* Prediction Card Result */
    .result-card-yes {
        background-color: #064e3b;
        border: 1px solid #10b981;
        color: #a7f3d0;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 15px;
    }
    
    .result-card-no {
        background-color: #7f1d1d;
        border: 1px solid #ef4444;
        color: #fecaca;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 15px;
    }

    /* Primary Submit Button Customization */
    div.stButton > button:first-child {
        background-color: #6366f1;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #4f46e5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def load_model():
    try:
        with open("DecisionTree.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ File `DecisionTree.pkl` not found in the current directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model = load_model()

# --- App Header ---
st.markdown('<div class="title-text">🌳 Model Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Enter feature details below to generate predictions from your Decision Tree model</div>', unsafe_allow_html=True)

# --- Input Form ---
if model is not None:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
            
            # Note: Update options below if your model expects specific string levels or label-encoded numbers
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            
            region = st.selectbox("Region", options=["Urban", "Suburban", "Rural"])

        with col2:
            occupation = st.selectbox(
                "Occupation", 
                options=["Salaried", "Self-Employed", "Business", "Student", "Retired", "Other"]
            )
            
            income = st.number_input("Income", min_value=0, value=50000, step=1000)

        # Categorical Encoding Options (if your tree expects encoded numeric values, adjust dictionary mapping below)
        # Standard encoding mapping example:
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        region_map = {"Urban": 0, "Suburban": 1, "Rural": 2}
        occ_map = {"Salaried": 0, "Self-Employed": 1, "Business": 2, "Student": 3, "Retired": 4, "Other": 5}

        submit_btn = st.form_submit_button("Run Prediction 🚀")

    if submit_btn:
        # Prepare input data matching feature names expected by the pickle file:
        # ['Age', 'Gender', 'Region', 'Occupation', 'Income']
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender_map.get(gender, gender),
            'Region': region_map.get(region, region),
            'Occupation': occ_map.get(occupation, occupation),
            'Income': income
        }])

        try:
            prediction = model.predict(input_data)[0]
            
            # Show Probability if model supports predict_proba
            proba = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                proba = np.max(probabilities) * 100

            st.markdown("---")
            if str(prediction).lower() == "yes":
                res_html = f'<div class="result-card-yes">✅ Prediction Output: <b>YES</b>'
                if proba is not None:
                    res_html += f'<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {proba:.1f}%</span>'
                res_html += '</div>'
                st.markdown(res_html, unsafe_allow_html=True)
            else:
                res_html = f'<div class="result-card-no">❌ Prediction Output: <b>NO</b>'
                if proba is not None:
                    res_html += f'<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {proba:.1f}%</span>'
                res_html += '</div>'
                st.markdown(res_html, unsafe_allow_html=True)

        except Exception as err:
            st.error(f"Prediction failed. Make sure categorical features match your model's expected format. Error: {err}")
