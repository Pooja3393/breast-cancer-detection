import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺")
st.title("🩺 Breast Cancer Detection App")
st.markdown("This app predicts whether a tumor is *Benign* (non-cancerous) or *Malignant* (cancerous) based on medical test inputs.")

st.info("👩‍⚕ Just move the sliders below to simulate test results. The model will use those values to predict the outcome.")

# Load the scikit-learn model
model = joblib.load("breast_cancer_model_simple.pkl")

# Sliders for selected features
mean_radius = st.slider(
    "1️⃣ Lump Size (Mean Radius in mm)",
    min_value=0.0, max_value=30.0, value=14.0, step=0.1,
    help="Indicates the average size of the lump. Larger size may be risky."
)

mean_texture = st.slider(
    "2️⃣ Lump Texture Variation (Mean Texture)",
    min_value=0.0, max_value=40.0, value=19.0, step=0.1,
    help="Higher texture means more variation in cells. Cancerous tumors usually have more irregular texture."
)

mean_perimeter = st.slider(
    "3️⃣ Lump Border Length (Mean Perimeter in mm)",
    min_value=0.0, max_value=200.0, value=90.0, step=0.1,
    help="Longer perimeters may indicate irregular borders often found in malignant tumors."
)

# Predict
if st.button("🔍 Predict Tumor Type"):
    # Create input array (assuming model trained on just 3 features)
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter]])
    
    # Model prediction
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("✅ The model predicts that the tumor is *Benign* (Non-cancerous).")
    else:
        st.error("🚨 The model predicts that the tumor is *Malignant* (Cancerous). Please consult a doctor.")

    # Extra risk analysis
    risk_score = 0
    if mean_radius > 17:
        risk_score += 1
    if mean_texture > 22:
        risk_score += 1
    if mean_perimeter > 100:
        risk_score += 1

    st.markdown("---")
    st.subheader("📊 Risk Analysis Based on Input Values")

    if risk_score == 0:
        st.info("🟢 Based on your input values, the features indicate *low risk* of cancer.")
    elif risk_score == 1:
        st.warning("🟠 Your inputs show some signs that need attention. Please monitor and consult a doctor if needed.")
    else:
        st.error("🔴 The values suggest a *high risk* of malignancy. Please get medical consultation")